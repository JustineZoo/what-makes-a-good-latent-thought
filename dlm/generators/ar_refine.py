"""Autoregressive latent generator with T-step within-latent denoising refinement (C3a)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dlm.models.backbone import DiTBlock, LearnedPositionalEmbedding, modulate, timestep_embedding
from .base import Generator


class ConceptARRefinement(Generator):
    """Causal across latents (clean prefix), iterative x0-denoising within each latent."""

    paradigm = "concept_ar_refine"

    def __init__(
        self,
        d_concept: int,
        n_layers: int = 12,
        n_heads: int = 8,
        max_concepts: int = 64,
        model_dim: int = None,
    ):
        super().__init__()
        self.d = d_concept
        w = model_dim or d_concept
        self.w = w
        self.in_proj = nn.Linear(2 * d_concept, w)  # [z_or_clean ; self_cond]
        self.pos = LearnedPositionalEmbedding(max_concepts, w)
        self.t_mlp = nn.Sequential(nn.Linear(w, 4 * w), nn.SiLU(), nn.Linear(4 * w, w))
        self.blocks = nn.ModuleList([DiTBlock(w, n_heads, cross=False) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(w, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(w, 2 * w))
        nn.init.zeros_(self.ada_out[-1].weight)
        nn.init.zeros_(self.ada_out[-1].bias)
        self.out_proj = nn.Linear(w, d_concept)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _forward_tokens(
        self,
        z_seq: torch.Tensor,
        self_cond: torch.Tensor,
        t_seq: torch.Tensor,
        pos_ids: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        z_seq: (B, L, d), self_cond: (B, L, d), t_seq: (B, L), pos_ids: (L,).
        Applies per-token timestep modulation so clean prefix tokens run at t=0
        while active target queries run at their sampled noise level t_k.
        """
        B, L, _ = z_seq.shape
        x = self.in_proj(torch.cat([z_seq, self_cond], dim=-1)) + self.pos.pos[:, pos_ids, :]
        c = self.t_mlp(timestep_embedding(t_seq.reshape(B * L), self.w)).view(B, L, self.w)

        for blk in self.blocks:
            shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = blk.ada(c).chunk(6, dim=-1)
            h_norm = blk.n1(x) * (1.0 + scale_a) + shift_a
            x = x + gate_a * blk.attn(h_norm, causal=(attn_mask is None), attn_mask=attn_mask)
            h_ff = blk.n2(x) * (1.0 + scale_f) + shift_f
            x = x + gate_f * blk.ff(h_ff)

        shift_o, scale_o = self.ada_out(c).chunk(2, dim=-1)
        x = self.norm_out(x) * (1.0 + scale_o) + shift_o
        return self.out_proj(x)

    def _build_train_mask(self, mc: int, mt: int, device: torch.device) -> torch.Tensor:
        """
        Sequence layout of length L = mc + 2*mt:
          [0 .. mc-1]           : clean prompt cond (attends to 0 .. mc-1)
          [mc .. mc+mt-1]       : clean target prefix (causal over 0 .. mc+k)
          [mc+mt .. mc+2*mt-1]  : noised queries k=0..mt-1 (attends to 0 .. mc+k-1 and self)
        """
        L = mc + 2 * mt
        idx = torch.arange(L, device=device)
        q_i = idx[:, None]
        k_j = idx[None, :]

        cond_ok = (q_i < mc) & (k_j < mc)
        clean_ok = (q_i >= mc) & (q_i < mc + mt) & (k_j <= q_i)
        query_k = q_i - (mc + mt)
        noisy_ok = (q_i >= mc + mt) & ((k_j < mc + query_k) | (k_j == q_i))
        return cond_ok | clean_ok | noisy_ok

    def _build_eval_mask(self, mc: int, total_len: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(total_len, device=device)
        q_i, k_j = idx[:, None], idx[None, :]
        return ((q_i < mc) & (k_j < mc)) | ((q_i >= mc) & (k_j <= q_i))

    def fit_step(self, target: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, mt, d = target.shape
        mc = cond.shape[1]
        device = target.device

        t_q = torch.rand(B, mt, device=device)
        noise = torch.randn_like(target)
        z_noisy = (1.0 - t_q).unsqueeze(-1) * target + t_q.unsqueeze(-1) * noise

        clean_prefix = torch.cat([cond, target], dim=1)
        z_seq = torch.cat([clean_prefix, z_noisy], dim=1)
        t_seq = torch.cat([torch.zeros(B, mc + mt, device=device), t_q], dim=1)
        pos_ids = torch.cat([
            torch.arange(mc + mt, device=device),
            torch.arange(mc, mc + mt, device=device),
        ])
        mask = self._build_train_mask(mc, mt, device)

        zeros_all = torch.zeros_like(z_seq)
        sc_noisy = torch.zeros_like(z_noisy)
        if float(torch.rand(())) < 0.5:
            with torch.no_grad():
                pred0 = self._forward_tokens(z_seq, zeros_all, t_seq, pos_ids, attn_mask=mask)
                sc_noisy = pred0[:, -mt:, :].detach()

        sc_seq = torch.cat([torch.zeros_like(clean_prefix), sc_noisy], dim=1)
        x0_hat = self._forward_tokens(z_seq, sc_seq, t_seq, pos_ids, attn_mask=mask)[:, -mt:, :]
        return F.mse_loss(x0_hat, target)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, n_target: int, T: int = 16, **_) -> torch.Tensor:
        B, mc, d = cond.shape
        device = cond.device
        prefix = cond
        zeros_1 = torch.zeros(B, 1, d, device=device)

        for k in range(n_target):
            cur_len = prefix.shape[1]
            pos_ids = torch.arange(cur_len + 1, device=device)
            eval_mask = self._build_eval_mask(mc, cur_len + 1, device)
            sc_prefix = torch.zeros_like(prefix)
            t_prefix = torch.zeros(B, cur_len, device=device)

            z = torch.randn(B, 1, d, device=device)
            x0_hat = zeros_1
            for i in range(T):
                t_val = 1.0 - i / T
                z_seq = torch.cat([prefix, z], dim=1)
                sc_seq = torch.cat([sc_prefix, x0_hat], dim=1)
                t_seq = torch.cat([t_prefix, torch.full((B, 1), t_val, device=device)], dim=1)
                x0_hat = self._forward_tokens(z_seq, sc_seq, t_seq, pos_ids, attn_mask=eval_mask)[:, -1:, :]
                z = z - (1.0 / T) * (z - x0_hat) / t_val

            prefix = torch.cat([prefix, z], dim=1)

        return prefix[:, mc:, :]

    def serial_cost(self, n_target: int, T: int = 16, **_) -> int:
        return n_target * T
