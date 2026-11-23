"""Phase 1b: LoRA LLM Latent Codec and Autoregressive Flow-Matching Generator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dlm.models.heads import MARHead


@dataclass
class LLMCodecConfig:
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    R: int = 4
    d_z: int = 256
    max_cot_len: int = 160
    beta: float = 0.05
    free_bits: float = 0.25
    lora_r: int = 128
    lora_alpha: int = 32


class LLMLatentCodec(nn.Module):
    """Two-stage continuous latent codec built on a LoRA-adapted causal LLM."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module, cfg: LLMCodecConfig):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cfg = cfg

        hidden_size = encoder.config.hidden_size
        self.to_mu = nn.Linear(hidden_size, cfg.d_z)
        self.to_logvar = nn.Linear(hidden_size, cfg.d_z)
        nn.init.constant_(self.to_logvar.bias, -4.0)

        self.latent_in = nn.Linear(cfg.d_z, hidden_size)
        max_chunks = (cfg.max_cot_len + cfg.R - 1) // cfg.R + 4
        self.chunk_pos = nn.Embedding(max_chunks, hidden_size)
        self.offset_pos = nn.Embedding(cfg.R, hidden_size)

        self.register_buffer("z_mean", torch.zeros(cfg.d_z))
        self.register_buffer("z_std", torch.ones(cfg.d_z))

    def encode_posterior(
        self, cot_ids: torch.Tensor, cot_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode (B, L_cot) token IDs -> (mu, logvar) of shape (B, m, d_z)."""
        B, L = cot_ids.shape
        R = self.cfg.R
        pad = (-L) % R
        if pad > 0:
            pad_id = getattr(self.encoder.config, "pad_token_id", 0) or 0
            cot_ids = F.pad(cot_ids, (0, pad), value=pad_id)
            cot_mask = F.pad(cot_mask, (0, pad), value=0)

        out = self.encoder(
            input_ids=cot_ids,
            attention_mask=cot_mask,
            output_hidden_states=True,
        )
        h = out.hidden_states[-1]  # (B, m*R, H)
        m = h.shape[1] // R
        pooled = h.view(B, m, R, -1).mean(dim=2)  # (B, m, H)
        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled).clamp(-8.0, 8.0)
        return mu, logvar

    @torch.no_grad()
    def encode_standardized(
        self, cot_ids: torch.Tensor, cot_mask: torch.Tensor
    ) -> torch.Tensor:
        mu, _ = self.encode_posterior(cot_ids, cot_mask)
        return (mu - self.z_mean) / self.z_std

    def unstandardize(self, z_norm: torch.Tensor) -> torch.Tensor:
        return z_norm * self.z_std + self.z_mean

    def decode_logits(
        self,
        z: torch.Tensor,
        q_ids: torch.Tensor,
        q_mask: torch.Tensor,
        out_len: int,
    ) -> torch.Tensor:
        """Reconstruct out_len CoT tokens from latents z (B, m, d_z) and question."""
        B, m, _ = z.shape
        device = z.device
        q_emb = self.decoder.get_input_embeddings()(q_ids)
        z_emb = self.latent_in(z)

        idx = torch.arange(out_len, device=device)
        q_queries = self.chunk_pos(idx // self.cfg.R) + self.offset_pos(idx % self.cfg.R)
        q_queries = q_queries.unsqueeze(0).expand(B, -1, -1).to(q_emb.dtype)

        seq_emb = torch.cat([q_emb, z_emb, q_queries], dim=1)
        extra_len = seq_emb.shape[1] - q_emb.shape[1]
        extra_mask = torch.ones((B, extra_len), device=device, dtype=q_mask.dtype)
        seq_mask = torch.cat([q_mask, extra_mask], dim=1)
        pos_ids = (seq_mask.long().cumsum(-1) - 1).clamp(min=0)

        out = self.decoder(
            inputs_embeds=seq_emb,
            attention_mask=seq_mask,
            position_ids=pos_ids,
            output_hidden_states=True,
        )
        h_target = out.hidden_states[-1][:, -out_len:, :]
        return self.decoder.get_output_embeddings()(h_target)

    def compute_loss(
        self,
        q_ids: torch.Tensor,
        q_mask: torch.Tensor,
        cot_ids: torch.Tensor,
        cot_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode_posterior(cot_ids, cot_mask)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std) if self.training else mu

        logits = self.decode_logits(z, q_ids, q_mask, out_len=labels.shape[1])
        recon_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )
        kl_elem = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
        kl_loss = torch.clamp(kl_elem, min=self.cfg.free_bits).mean()
        total_loss = recon_loss + self.cfg.beta * kl_loss
        return total_loss, recon_loss.detach(), kl_loss.detach()


class LLMLatentGenerator(nn.Module):
    """Causal LLM + LoRA with MAR rectified-flow head over frozen codec latents."""

    def __init__(
        self,
        backbone: nn.Module,
        d_z: int = 256,
        head_inner_steps: int = 4,
    ):
        super().__init__()
        self.backbone = backbone
        self.d_z = d_z
        self.head_inner_steps = head_inner_steps

        hidden_size = backbone.config.hidden_size
        self.bos_latent = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.in_proj = nn.Linear(d_z, hidden_size)
        self.out_proj = nn.Linear(hidden_size, d_z)
        self.head = MARHead(d=d_z)

    def _predict_cond(
        self, q_ids: torch.Tensor, q_mask: torch.Tensor, prefix_z: Optional[torch.Tensor], n_steps: int
    ) -> torch.Tensor:
        B = q_ids.shape[0]
        q_emb = self.backbone.get_input_embeddings()(q_ids)
        bos = self.bos_latent.expand(B, 1, -1).to(q_emb.dtype)
        parts = [q_emb, bos]
        if prefix_z is not None and prefix_z.shape[1] > 0:
            parts.append(self.in_proj(prefix_z).to(q_emb.dtype))
        seq_emb = torch.cat(parts, dim=1)
        extra_len = seq_emb.shape[1] - q_emb.shape[1]
        extra_mask = torch.ones((B, extra_len), device=q_ids.device, dtype=q_mask.dtype)
        seq_mask = torch.cat([q_mask, extra_mask], dim=1)
        pos_ids = (seq_mask.long().cumsum(-1) - 1).clamp(min=0)

        out = self.backbone(
            inputs_embeds=seq_emb,
            attention_mask=seq_mask,
            position_ids=pos_ids,
            output_hidden_states=True,
        )
        h = out.hidden_states[-1][:, -n_steps:, :]
        return self.out_proj(h.float())

    def fit_step(
        self, q_ids: torch.Tensor, q_mask: torch.Tensor, target_z: torch.Tensor
    ) -> torch.Tensor:
        B, m, _ = target_z.shape
        prefix = target_z[:, :-1, :] if m > 1 else None
        cond = self._predict_cond(q_ids, q_mask, prefix, n_steps=m)
        return self.head.loss(cond.reshape(B * m, self.d_z), target_z.reshape(B * m, self.d_z))

    @torch.no_grad()
    def teacher_forced_sample(
        self, q_ids: torch.Tensor, q_mask: torch.Tensor, target_z: torch.Tensor, steps: Optional[int] = None
    ) -> torch.Tensor:
        """Position 0 is first-emission probe (no prefix); 1..m-1 are teacher-forced."""
        B, m, _ = target_z.shape
        prefix = target_z[:, :-1, :] if m > 1 else None
        cond = self._predict_cond(q_ids, q_mask, prefix, n_steps=m)
        n_flow = steps or self.head_inner_steps
        return self.head.sample(cond.reshape(B * m, self.d_z), steps=n_flow).view(B, m, self.d_z)

    @torch.no_grad()
    def sample(
        self, q_ids: torch.Tensor, q_mask: torch.Tensor, n_latents: int, steps: Optional[int] = None
    ) -> torch.Tensor:
        """Free-running autoregressive latent rollout from question alone."""
        B = q_ids.shape[0]
        n_flow = steps or self.head_inner_steps
        q_emb = self.backbone.get_input_embeddings()(q_ids)
        seq_emb = torch.cat([q_emb, self.bos_latent.expand(B, 1, -1).to(q_emb.dtype)], dim=1)
        seq_mask = torch.cat(
            [q_mask, torch.ones((B, 1), device=q_ids.device, dtype=q_mask.dtype)], dim=1
        )

        emitted = []
        for _ in range(n_latents):
            pos_ids = (seq_mask.long().cumsum(-1) - 1).clamp(min=0)
            out = self.backbone(
                inputs_embeds=seq_emb,
                attention_mask=seq_mask,
                position_ids=pos_ids,
                output_hidden_states=True,
            )
            h_last = out.hidden_states[-1][:, -1, :].float()
            z_next = self.head.sample(self.out_proj(h_last), steps=n_flow)
            emitted.append(z_next)
            z_step_emb = self.in_proj(z_next).unsqueeze(1).to(q_emb.dtype)
            seq_emb = torch.cat([seq_emb, z_step_emb], dim=1)
            seq_mask = torch.cat(
                [seq_mask, torch.ones((B, 1), device=q_ids.device, dtype=q_mask.dtype)], dim=1
            )
        return torch.stack(emitted, dim=1)
