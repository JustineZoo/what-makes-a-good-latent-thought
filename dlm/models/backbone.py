"""Shared transformer primitives (torch).

Low-level, paradigm-agnostic blocks reused by the DiT denoiser (diffusion), the
concept-AR generator, and the token baselines. Two block flavours:

  * ``TransformerBlock`` -- plain pre-norm self-attention (+ optional causal mask,
    + optional cross-attention to a context sequence). Used by AR / token models.
  * ``DiTBlock``         -- adaLN-Zero block conditioned on a timestep embedding,
    with optional cross-attention to conditioning context. Used by the denoiser.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of a continuous timestep ``t`` in [0, 1] (shape (B,)).

    ``t`` is scaled by 1000 before embedding (the DiT convention): raw t in [0,1]
    would leave the high-frequency channels nearly constant, giving the network
    almost no time resolution. Training and sampling share this function, so the
    mapping stays consistent.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = (1000.0 * t)[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MultiheadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, cross: bool = False):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.cross = cross
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, ctx=None, causal: bool = False, attn_mask: Optional[torch.Tensor] = None):
        """``attn_mask``: optional bool tensor broadcastable to (B, h, N, M); True =
        may attend. Cannot be combined with ``causal``."""
        B, N, C = x.shape
        h = self.n_heads
        ctx = x if ctx is None else ctx
        q = self.q(x).view(B, N, h, C // h).transpose(1, 2)
        k, v = self.kv(ctx).chunk(2, dim=-1)
        M = ctx.shape[1]
        k = k.view(B, M, h, C // h).transpose(1, 2)
        v = v.view(B, M, h, C // h).transpose(1, 2)
        if attn_mask is not None:
            assert not causal, "attn_mask and causal are mutually exclusive"
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, mult * dim), nn.GELU(), nn.Linear(mult * dim, dim))

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Plain pre-norm block; optional causal self-attn and optional cross-attn."""

    def __init__(self, dim: int, n_heads: int, causal: bool = False, cross: bool = False):
        super().__init__()
        self.causal = causal
        self.use_cross = cross
        self.n1 = nn.LayerNorm(dim)
        self.attn = MultiheadAttention(dim, n_heads)
        if cross:
            self.nc = nn.LayerNorm(dim)
            self.cross_attn = MultiheadAttention(dim, n_heads, cross=True)
        self.n2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

    def forward(self, x, ctx: Optional[torch.Tensor] = None,
                self_mask: Optional[torch.Tensor] = None,
                cross_mask: Optional[torch.Tensor] = None):
        x = x + self.attn(self.n1(x), causal=self.causal, attn_mask=self_mask)
        if self.use_cross and ctx is not None:
            x = x + self.cross_attn(self.nc(x), ctx=ctx, attn_mask=cross_mask)
        x = x + self.ff(self.n2(x))
        return x


def block_diagonal_mask(n: int, R: int, device=None) -> torch.Tensor:
    """(n, n) bool mask: position i may attend position j iff i // R == j // R."""
    blk = torch.arange(n, device=device) // R
    return blk[:, None] == blk[None, :]


class DiTBlock(nn.Module):
    """adaLN-Zero block conditioned on a timestep embedding ``c`` (B, dim)."""

    def __init__(self, dim: int, n_heads: int, cross: bool = False):
        super().__init__()
        self.use_cross = cross
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = MultiheadAttention(dim, n_heads)
        if cross:
            self.nc = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.cross_attn = MultiheadAttention(dim, n_heads, cross=True)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c, ctx: Optional[torch.Tensor] = None):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.ada(c).chunk(6, dim=-1)
        x = x + gate_a.unsqueeze(1) * self.attn(modulate(self.n1(x), shift_a, scale_a))
        if self.use_cross and ctx is not None:
            x = x + self.cross_attn(self.nc(x), ctx=ctx)
        x = x + gate_f.unsqueeze(1) * self.ff(modulate(self.n2(x), shift_f, scale_f))
        return x


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, dim: int):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x):
        return x + self.pos[:, : x.shape[1]]
