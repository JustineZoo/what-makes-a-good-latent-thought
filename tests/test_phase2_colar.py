"""Offline unit tests verifying True Phase 2a CoLaR C1/C2/OOD invariants."""
from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from dlm.models.phase2_colar import (
    FactorizedLatentPolicy,
    Phase2CoLaR,
    build_step_chunks,
    chunk_to_sub_slots,
    sqrt_mean_pool_chunk,
)


class _WordLevelMockTokenizer:
    """Minimal HuggingFace-compatible tokenizer for unit testing without network."""

    def __init__(self):
        self.vocab = {"[PAD]": 0, "<eos>": 1, "###": 2, "<unk>": 3}
        self.pad_token = "[PAD]"
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self.eos_token_id = 1
        self.unk_token_id = 3

    def _id(self, tok: str) -> int:
        if tok not in self.vocab:
            self.vocab[tok] = len(self.vocab)
        return self.vocab[tok]

    def encode(self, text: str, add_special_tokens: bool = False):
        # Split cleanly into discrete pseudo-subwords
        tokens = []
        for part in text.replace("\n", " \n ").split(" "):
            if not part:
                continue
            if part in ("###", "\n"):
                tokens.append(self._id(part))
            else:
                # 2-char chunks to simulate multi-token steps like <<5+3=8>>
                for i in range(0, len(part), 2):
                    tokens.append(self._id(part[i : i + 2]))
        return tokens

    def convert_tokens_to_ids(self, tok: str) -> int:
        return self.vocab.get(tok, self.unk_token_id)

    def batch_encode_plus(self, texts, return_tensors="pt", add_special_tokens=False, padding="longest", padding_side="left"):
        encs = [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]
        max_len = max(1, max(len(e) for e in encs))
        ids, masks = [], []
        for e in encs:
            pad_n = max_len - len(e)
            if padding_side == "left":
                ids.append([self.pad_token_id] * pad_n + e)
                masks.append([0] * pad_n + [1] * len(e))
            else:
                ids.append(e + [self.pad_token_id] * pad_n)
                masks.append([1] * len(e) + [0] * pad_n)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def batch_decode(self, token_ids, skip_special_tokens=True):
        inv = {v: k for k, v in self.vocab.items()}
        out = []
        for row in token_ids.tolist():
            pieces = []
            for tid in row:
                if skip_special_tokens and tid in (0, 1):
                    continue
                pieces.append(inv.get(tid, "?"))
            out.append("".join(pieces))
        return out


class _MockCausalLM(nn.Module):
    """Minimal causal LM returning HF-like forward outputs and greedy generate."""

    def __init__(self, vocab_size: int = 256, hidden_size: int = 32):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, use_cache=False)
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.emb

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        labels: torch.Tensor = None,
        past_key_values=None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ):
        h = torch.tanh(self.proj(inputs_embeds))
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        new_cache = (torch.zeros(1),) if use_cache else None
        return SimpleNamespace(
            loss=loss,
            logits=logits,
            hidden_states=(inputs_embeds, h) if output_hidden_states else None,
            past_key_values=new_cache,
        )

    @torch.no_grad()
    def generate(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor = None, max_new_tokens: int = 8, **kwargs):
        B = inputs_embeds.shape[0]
        cur_emb = inputs_embeds
        gen = []
        for _ in range(max_new_tokens):
            out = self.forward(cur_emb)
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen.append(nxt)
            cur_emb = torch.cat([cur_emb, self.emb(nxt)], dim=1)
        return torch.cat(gen, dim=1)


def test_c1_step_alignment_and_no_boundary_crossing():
    tok = _WordLevelMockTokenizer()
    sep_id = tok.convert_tokens_to_ids("###")
    step1 = "<<12+34=46>>"
    step2 = "<<46*2=92>>"
    steps_str = f"{step1}\n{step2}"

    # Verify raw steps have distinct tokens
    s1_ids = set(tok.encode(step1))
    s2_ids = set(tok.encode(step2))
    # Build C1 chunks
    chunks = build_step_chunks(steps_str, tok, sep_id, use_c1=True, c=5, c_prime=10)
    assert chunks[0] == [sep_id], "Chunk 0 must be '###' separator"
    assert len(chunks) >= 3
    # Every chunk after chunk 0 must come strictly from one step, never mixing s1 and s2
    assert chunks[1] == tok.encode(step1)
    assert chunks[2] == tok.encode(step2)

    # Fixed chunking at c=4 should slice across the boundary
    fixed = build_step_chunks(steps_str, tok, sep_id, use_c1=False, c=4)
    assert any(len(ch) == 4 for ch in fixed[1:])


def test_c2_sub_slot_exact_algebraic_reconstruction():
    torch.manual_seed(0)
    H, K = 32, 4
    policy = FactorizedLatentPolicy(feature_size=H, n_slots=K, intermediate_size=64)

    for n_tokens in [1, 2, 3, 4, 5, 7, 8]:
        tok_embeds = torch.randn(n_tokens, H)
        full_chunk = sqrt_mean_pool_chunk(tok_embeds)
        sub_slots = chunk_to_sub_slots(tok_embeds, K=K)
        assert sub_slots.shape == (K, H)

        if n_tokens in (1, 2, 4, 8):
            # Equal-size sub-spans satisfy combine_slots == full_chunk up to eps
            combined = policy.combine_slots(sub_slots.unsqueeze(0)).squeeze(0)
            assert torch.allclose(combined, full_chunk, atol=1e-4), f"Failed exact identity at N={n_tokens}"


@pytest.mark.parametrize("system", ["colar", "colar_c1", "colar_c2", "colar_c1_c2"])
def test_phase2_colar_end_to_end_forward_backward_and_ood_rollout(system: str):
    torch.manual_seed(42)
    tok = _WordLevelMockTokenizer()
    # Pre-populate vocab so encode during batching stays < 256
    for s in ["Question:", "Let's", "think", "step", "by", "step:", "(Thinking", "speed:", "5)", "Answer:", "<<5+3=8>>", "<<8*2=16>>", "16", "42"]:
        tok.encode(s)

    llm = _MockCausalLM(vocab_size=256, hidden_size=32)
    model = Phase2CoLaR(llm=llm, tokenizer=tok, system=system, c=5, c_prime=7, K=4, lp_intermediate_size=64)

    questions = ["Tom has 5 apples and buys 3.", "Another problem with two steps."]
    steps = ["<<5+3=8>>\n<<8*2=16>>", "<<10+32=42>>"]
    answers = ["16", "42"]

    loss_dict = model(questions, steps, answers, rng=random.Random(0))
    assert torch.isfinite(loss_dict["total_loss"])
    loss_dict["total_loss"].backward()

    # Check gradients reach both LLM and LatentPolicy
    llm_grad = sum(p.grad.abs().sum().item() for p in model.llm.parameters() if p.grad is not None)
    lp_grad = sum(p.grad.abs().sum().item() for p in model.latent_policy.parameters() if p.grad is not None)
    assert llm_grad > 0.0
    assert lp_grad > 0.0

    # Test empty-steps (OOD SVAMP/MultiArith) forward + rollout does NOT truncate to 1 token
    ood_loss = model(["OOD question?"], [""], ["99"], rng=random.Random(0))
    assert torch.isfinite(ood_loss["total_loss"])

    pred_ids, n_lat = model.latent_generate(["OOD question?"], max_n_latent_forward=5, max_new_tokens=6)
    assert pred_ids.shape == (1, 6)
    assert n_lat.shape == (1, 1)
    assert 1 <= n_lat.item() <= 5
