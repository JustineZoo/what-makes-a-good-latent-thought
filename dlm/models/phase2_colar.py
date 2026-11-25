"""True Phase 2a: Drop-in C1 (Step-Aligned Chunking) & C2 (Factorized Sub-Slots) inside Native CoLaR."""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dlm.data.chunking import fixed_chunk_tokens, step_aligned_chunk_tokens


def get_position_ids_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    return torch.clamp_min(torch.cumsum(attention_mask, dim=1) - 1, 0)


class LatentPolicy(nn.Module):
    """Exact native CoLaR Gaussian LatentPolicy head."""

    def __init__(self, feature_size: int, intermediate_size: int = 2048, deterministic: bool = False):
        super().__init__()
        self.deterministic = deterministic
        self.fc = nn.Sequential(
            nn.Linear(feature_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, intermediate_size),
            nn.LayerNorm(intermediate_size),
        )
        self.mean = nn.Linear(intermediate_size, feature_size)
        if not deterministic:
            self.log_std = nn.Linear(intermediate_size, feature_size)

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.distributions.Normal:
        h = self.fc(x)
        mean = self.mean(h)
        if self.deterministic:
            return torch.distributions.Normal(mean, torch.ones_like(mean) * 1e-9)
        std = self.log_std(h).clamp(-10.0, 4.0).exp() * temperature
        return torch.distributions.Normal(mean, std)


class FactorizedLatentPolicy(nn.Module):
    """Component C2: K parallel sub-slot Gaussian policy heads emitted in one forward pass."""

    def __init__(
        self,
        feature_size: int,
        n_slots: int = 4,
        intermediate_size: int = 2048,
        deterministic: bool = False,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.n_slots = n_slots
        self.deterministic = deterministic
        slot_inter = max(64, intermediate_size // n_slots)

        self.slot_fcs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_size, slot_inter),
                    nn.GELU(),
                    nn.Linear(slot_inter, slot_inter),
                    nn.LayerNorm(slot_inter),
                )
                for _ in range(n_slots)
            ]
        )
        self.slot_means = nn.ModuleList([nn.Linear(slot_inter, feature_size) for _ in range(n_slots)])
        if not deterministic:
            self.slot_log_stds = nn.ModuleList([nn.Linear(slot_inter, feature_size) for _ in range(n_slots)])

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.distributions.Normal:
        # x: (B, L, H) -> Normal over (B, L, K, H)
        means = []
        stds = []
        for k in range(self.n_slots):
            hk = self.slot_fcs[k](x)
            mk = self.slot_means[k](hk)
            means.append(mk)
            if self.deterministic:
                stds.append(torch.ones_like(mk) * 1e-9)
            else:
                stds.append(self.slot_log_stds[k](hk).clamp(-10.0, 4.0).exp() * temperature)
        mean = torch.stack(means, dim=-2)
        std = torch.stack(stds, dim=-2)
        return torch.distributions.Normal(mean, std)

    def combine_slots(self, sub_embeds: torch.Tensor) -> torch.Tensor:
        """Combine (..., K, H) sub-slot embeddings into one (..., H) latent preserving norm."""
        return sub_embeds.sum(dim=-2) / math.sqrt(self.n_slots)


def build_step_chunks(
    steps_str: str,
    tokenizer,
    thinking_separator_id: int,
    use_c1: bool = False,
    c: int = 5,
    c_prime: int = 7,
) -> List[List[int]]:
    """Build chunk token-ID lists for one example.

    Chunk 0 is always [thinking_separator_id] ('###').
    Remaining chunks never cross step boundaries when use_c1=True, or use fixed stride c when use_c1=False.
    """
    chunks: List[List[int]] = [[thinking_separator_id]]
    if not steps_str or not steps_str.strip():
        return chunks

    if use_c1:
        raw_steps = [s.strip() for s in steps_str.split("\n") if s.strip()]
        step_encs = [tokenizer.encode(s, add_special_tokens=False) for s in raw_steps]
        step_encs = [enc for enc in step_encs if enc]
        chunks.extend(step_aligned_chunk_tokens(step_encs, c_prime=c_prime))
    else:
        ids = tokenizer.encode(steps_str, add_special_tokens=False)
        if ids:
            chunks.extend(fixed_chunk_tokens(ids, c=c))
    return chunks


def sqrt_mean_pool_chunk(token_embeds: torch.Tensor) -> torch.Tensor:
    """token_embeds: (N, H) -> (H,) compressed via CoLaR's exact sqrt-mean formula."""
    n = token_embeds.shape[0]
    if n == 0:
        raise ValueError("Cannot pool empty chunk")
    return token_embeds.sum(dim=0) / (math.sqrt(n) + 1e-5)


def chunk_to_sub_slots(token_embeds: torch.Tensor, K: int) -> torch.Tensor:
    """Partition chunk embeddings (N, H) into (K, H) sub-slot embeddings.

    When N >= K, contiguous slices S_j are pooled with sqrt-mean so each slot
    carries ~N/K tokens. When N < K (e.g., separator '###'), token embeddings
    are tiled/scaled by sqrt(1/K) so combine_slots reconstructs the exact chunk.
    """
    n, H = token_embeds.shape
    if n == 1:
        return token_embeds.expand(K, H) / math.sqrt(K)
    if n < K:
        # Repeat tokens evenly and scale so sum / sqrt(K) matches sqrt_mean(token_embeds)
        idx = [(j * n) // K for j in range(K)]
        picked = token_embeds[idx]  # (K, H)
        scale = math.sqrt(K) / math.sqrt(n)
        # Re-normalize per distinct token frequency so sum_j picked / sqrt(K) == sum(e)/sqrt(n)
        counts = torch.bincount(torch.tensor(idx, device=token_embeds.device), minlength=n).float()
        w = (1.0 / counts[idx]).unsqueeze(-1).to(picked.dtype) * scale
        return picked * w

    slots = []
    for j in range(K):
        lo = (j * n) // K
        hi = ((j + 1) * n) // K
        slots.append(sqrt_mean_pool_chunk(token_embeds[lo:hi]))
    return torch.stack(slots, dim=0)


class Phase2CoLaR(nn.Module):
    """Clean, modular native CoLaR implementation supporting Vanilla, C1, C2, and C1+C2."""

    def __init__(
        self,
        llm: nn.Module,
        tokenizer,
        system: str = "colar",
        c: int = 5,
        c_prime: int = 7,
        K: int = 4,
        embeds_std: float = 0.018,
        ce_weight: float = 1.0,
        embed_modeling_weight: float = 1.0,
        entropy_weight: float = -1e-6,
        lp_intermediate_size: int = 2048,
        deterministic: bool = False,
    ):
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.system = system
        self.use_c1 = "c1" in system
        self.use_c2 = "c2" in system
        self.c = c
        self.c_prime = c_prime
        self.K = K if self.use_c2 else 1
        self.embeds_std = embeds_std
        self.ce_weight = ce_weight
        self.embed_modeling_weight = embed_modeling_weight
        self.entropy_weight = entropy_weight

        self.question_template = "Question: {} Let's think step by step:"
        self.speed_template = "(Thinking speed: {})"
        self.thinking_separator = "###"
        self.thinking_separator_id = tokenizer.convert_tokens_to_ids(self.thinking_separator)
        assert self.thinking_separator_id is not None and self.thinking_separator_id != tokenizer.unk_token_id, (
            "Tokenizer must map '###' to a single non-UNK token ID"
        )
        self.answer_template = "Answer:{}"

        hidden_size = llm.config.hidden_size
        if self.use_c2:
            self.latent_policy = FactorizedLatentPolicy(
                feature_size=hidden_size,
                n_slots=self.K,
                intermediate_size=lp_intermediate_size,
                deterministic=deterministic,
            )
        else:
            self.latent_policy = LatentPolicy(
                feature_size=hidden_size,
                intermediate_size=lp_intermediate_size,
                deterministic=deterministic,
            )

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def _prepare_texts(self, texts: List[str], padding_side: str) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            padding="longest",
            padding_side=padding_side,
        )
        device = next(self.llm.parameters()).device
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    def compress_steps_batch(
        self,
        steps_list: List[str],
        answer_sep_embed: torch.Tensor,  # (B, 1, H) embed of '###' ending thinking
        rng: Optional[random.Random] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Encode variable-length step chunks (Vanilla fixed-c or C1 step-aligned) with left padding.

        Returns:
          steps_embeds: (B, L_s, H) compressed chunk embeddings (Chunk 0 active is '###')
          steps_mask  : (B, L_s) 1 for active chunks, 0 for left pad
          steps_labels: (B, L_s) randomly sampled target token ID per chunk (-100 for pad)
          sub_targets : (B, L_s, K, H) unnormalized target sub-slot embeddings if use_c2 else None
                        Note: sub_targets[:, t] aligns with target for steps_outputs[:, t]
                        (i.e. next chunk's sub-slots, ending with answer_sep_embed!).
        """
        B = len(steps_list)
        device = answer_sep_embed.device
        dtype = answer_sep_embed.dtype
        H = answer_sep_embed.shape[-1]
        emb_layer = self.get_input_embeddings()
        py_rng = rng or random

        per_ex_chunks = [
            build_step_chunks(
                s,
                self.tokenizer,
                self.thinking_separator_id,
                use_c1=self.use_c1,
                c=self.c,
                c_prime=self.c_prime,
            )
            for s in steps_list
        ]
        max_chunks = max(len(ch) for ch in per_ex_chunks)

        steps_embeds = torch.zeros((B, max_chunks, H), device=device, dtype=dtype)
        steps_mask = torch.zeros((B, max_chunks), device=device, dtype=torch.long)
        steps_labels = torch.full((B, max_chunks), -100, device=device, dtype=torch.long)
        sub_slots_all = (
            torch.zeros((B, max_chunks, self.K, H), device=device, dtype=dtype) if self.use_c2 else None
        )

        # Embed unique token IDs cleanly per batch item
        for b, chunks in enumerate(per_ex_chunks):
            pad_left = max_chunks - len(chunks)
            for j, ch_ids in enumerate(chunks):
                pos = pad_left + j
                t_tensor = torch.tensor(ch_ids, device=device, dtype=torch.long)
                tok_embs = emb_layer(t_tensor)  # (N_chunk, H)
                steps_embeds[b, pos] = sqrt_mean_pool_chunk(tok_embs)
                steps_mask[b, pos] = 1
                steps_labels[b, pos] = py_rng.choice(ch_ids)
                if sub_slots_all is not None:
                    sub_slots_all[b, pos] = chunk_to_sub_slots(tok_embs, self.K)

        if sub_slots_all is not None:
            # Target for position t in steps_outputs is position t+1, ending at answer_sep_embed ('###')
            sep_sub = answer_sep_embed / math.sqrt(self.K)  # (B, 1, H)
            sep_sub = sep_sub.unsqueeze(2).expand(B, 1, self.K, H)
            shifted_sub_targets = torch.cat([sub_slots_all[:, 1:, :, :], sep_sub], dim=1)
        else:
            shifted_sub_targets = None

        return steps_embeds, steps_mask, steps_labels, shifted_sub_targets

    def forward(
        self,
        questions: List[str],
        steps_list: List[str],
        answers: List[str],
        rng: Optional[random.Random] = None,
    ) -> Dict[str, torch.Tensor]:
        B = len(questions)
        emb_layer = self.get_input_embeddings()

        # 1. Question: [pad, Question: ... Let's think step by step:(Thinking speed: c)]
        speed_suffix = self.speed_template.format(self.c)
        q_texts = [self.question_template.format(q) + speed_suffix for q in questions]
        q_ids, q_mask = self._prepare_texts(q_texts, padding_side="left")
        q_embeds = emb_layer(q_ids)

        # 2. Answer: [###Answer:{ans}{eos}, pad]
        eos_str = self.tokenizer.eos_token or ""
        a_texts = [self.thinking_separator + self.answer_template.format(a) + eos_str for a in answers]
        a_ids, a_mask = self._prepare_texts(a_texts, padding_side="right")
        a_embeds = emb_layer(a_ids)

        # 3. Compressed steps: [pad, ###, chunk_1, ..., chunk_M]
        steps_embeds, steps_mask, steps_labels, shifted_sub_targets = self.compress_steps_batch(
            steps_list, answer_sep_embed=a_embeds[:, 0:1, :], rng=rng
        )

        L_q = q_embeds.shape[1]
        L_s = steps_embeds.shape[1]

        inputs_embeds = torch.cat([q_embeds, steps_embeds, a_embeds], dim=1)
        if self.training and not inputs_embeds.requires_grad:
            inputs_embeds = inputs_embeds.requires_grad_(True)
        attention_mask = torch.cat([q_mask, steps_mask, a_mask], dim=1)
        position_ids = get_position_ids_from_attention_mask(attention_mask)

        # CE labels: ignore question and pads
        labels = torch.cat([q_ids, steps_labels, a_ids], dim=1)
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[:, :L_q] = -100

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            output_hidden_states=True,
        )
        ce_loss = outputs.loss

        # Latent policy loss: hidden state at steps[t] predicts inputs_embeds[L_q + t + 1]
        steps_outputs = outputs.hidden_states[-1][:, L_q : L_q + L_s, :].float()
        dists = self.latent_policy(steps_outputs)

        mask_f = steps_mask.float().clamp(min=0.0)
        denom = mask_f.sum().clamp(min=1.0)

        if self.use_c2:
            # Predict K sub-slot targets (B, L_s, K, H)
            pred_sub = dists.rsample()
            target_sub = shifted_sub_targets.detach().float() / self.embeds_std
            lat_mse = F.mse_loss(pred_sub, target_sub, reduction="none").mean(dim=(-1, -2))  # (B, L_s)
            entropy = dists.entropy().mean(dim=(-1, -2))
        else:
            pred_emb = dists.rsample()
            gold_embeds = inputs_embeds[:, L_q + 1 : L_q + L_s + 1, :].detach().float() / self.embeds_std
            lat_mse = F.mse_loss(pred_emb, gold_embeds, reduction="none").mean(dim=-1)  # (B, L_s)
            entropy = dists.entropy().mean(dim=-1)

        embed_modeling_loss = (lat_mse * mask_f).sum() / denom
        entropy_mean = (entropy * mask_f).sum() / denom

        total_loss = (
            self.ce_weight * ce_loss
            + self.embed_modeling_weight * embed_modeling_loss
            + self.entropy_weight * entropy_mean
        )
        return {
            "total_loss": total_loss,
            "ce_loss": ce_loss.detach(),
            "embed_modeling_loss": embed_modeling_loss.detach(),
            "entropy": entropy_mean.detach(),
        }

    @torch.no_grad()
    def latent_generate(
        self,
        questions: List[str],
        max_n_latent_forward: int = 64,
        max_new_tokens: int = 24,
        latent_temperature: float = 0.0,
        do_sample_answer: bool = False,
        greedy_eol: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive latent rollout matching CoLaR's inference loop with KV cache."""
        B = len(questions)
        device = next(self.llm.parameters()).device
        emb_layer = self.get_input_embeddings()
        dtype = emb_layer.weight.dtype

        suffix = self.speed_template.format(self.c) + self.thinking_separator
        q_texts = [self.question_template.format(q) + suffix for q in questions]
        q_ids, q_mask = self._prepare_texts(q_texts, padding_side="left")
        q_pos = get_position_ids_from_attention_mask(q_mask)
        q_embeds = emb_layer(q_ids)

        was_use_cache = getattr(self.llm.config, "use_cache", False)
        self.llm.config.use_cache = True
        try:
            outputs = self.llm(
                inputs_embeds=q_embeds,
                attention_mask=q_mask,
                position_ids=q_pos,
                use_cache=True,
                output_hidden_states=True,
            )

            all_inputs_embeds = [q_embeds]
            all_attention_mask = q_mask
            current_pos = q_pos[:, -1:]
            past_key_values = outputs.past_key_values
            is_done = torch.zeros((B, 1), device=device, dtype=torch.bool)
            n_latent_forward = torch.zeros((B, 1), device=device, dtype=torch.long)

            for _ in range(max_n_latent_forward):
                h_last = outputs.hidden_states[-1][:, -1:, :].float()
                dist = self.latent_policy(h_last, temperature=max(latent_temperature, 1e-6))
                if latent_temperature == 0.0:
                    sample_val = dist.mean
                else:
                    sample_val = dist.rsample()

                if self.use_c2:
                    lat_embed = self.latent_policy.combine_slots(sample_val) * self.embeds_std
                else:
                    lat_embed = sample_val * self.embeds_std
                lat_embed = lat_embed.to(dtype)

                all_inputs_embeds.append(lat_embed)
                not_done_long = (~is_done).long()
                all_attention_mask = torch.cat([all_attention_mask, not_done_long], dim=1)
                current_pos = current_pos + not_done_long
                n_latent_forward += not_done_long

                outputs = self.llm(
                    inputs_embeds=lat_embed,
                    attention_mask=all_attention_mask,
                    position_ids=current_pos,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                )
                past_key_values = outputs.past_key_values

                last_logits = outputs.logits[:, -1]
                if greedy_eol:
                    next_tok = last_logits.argmax(dim=-1, keepdim=True)
                else:
                    probs = torch.softmax(last_logits, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1)

                is_done = is_done | (next_tok == self.thinking_separator_id)
                if is_done.all():
                    break

            # Append ending '###' separator before generating final answer
            sep_ids = torch.full((B, 1), self.thinking_separator_id, device=device, dtype=torch.long)
            all_inputs_embeds.append(emb_layer(sep_ids))
            all_attention_mask = torch.cat([all_attention_mask, torch.ones((B, 1), device=device, dtype=torch.long)], dim=1)

            full_embeds = torch.cat(all_inputs_embeds, dim=1)
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample_answer,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if do_sample_answer:
                gen_kwargs["top_p"] = 0.9
                gen_kwargs["temperature"] = 1.0

            pred_ids = self.llm.generate(
                inputs_embeds=full_embeds,
                attention_mask=all_attention_mask,
                **gen_kwargs,
            )
            return pred_ids, n_latent_forward
        finally:
            self.llm.config.use_cache = was_use_cache
