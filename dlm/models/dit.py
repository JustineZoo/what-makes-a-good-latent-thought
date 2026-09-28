"""DiT-style denoiser over the concept space (torch).

Predicts the CLEAN concept sequence (x0-parameterization) from a noised one,
conditioned on (a) the diffusion timestep, (b) the context concepts (the encoded
problem statement) via cross-attention, and (c) optionally the previous clean
estimate (**self-conditioning**, a la Bit-Diffusion / CDCD / Diffusion-of-Thought).

Why x0 + self-conditioning: given the prompt, a reasoning trace is deterministic,
so every timestep's target is the same clean trace -- x0-prediction puts the full
reasoning signal at every t (a velocity/eps target is dominated by irreducible
noise at small t). Self-conditioning feeds the model its own previous clean
estimate, turning the T sampling steps into literal iterative refinement of a
working answer -- the serial-compute mechanism this study measures. Without it,
continuous text diffusion is a known-weak baseline.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .backbone import (DiTBlock, LearnedPositionalEmbedding, TransformerBlock,
                       timestep_embedding)


class ConceptDenoiser(nn.Module):
    def __init__(
        self,
        d_concept: int,
        n_layers: int = 12,
        n_heads: int = 8,
        max_concepts: int = 64,
        cross_condition: bool = True,
        model_dim: int = None,
        block: str = "adaln",
    ):
        """``model_dim`` decouples backbone width from the concept dimension
        (generator-scale studies over a fixed frozen codec). When None (default)
        the backbone runs at d_concept and the module tree is unchanged, so
        pre-existing checkpoints load strictly.

        ``block`` selects the conditioning architecture (the architecture
        control): "adaln" is the DiT adaLN-Zero block; "plain" reuses the AR
        generator's pre-norm ``TransformerBlock`` (bidirectional, cross-attention
        to the context kept) and supplies the noise level as one extra in-context
        time token, the DiT paper's in-context conditioning variant. Everything
        else -- self-conditioning input, cross-attention, objective, sampler --
        is shared, so the two differ only in how the timestep reaches the blocks
        (and in the adaLN-Zero gating/init that comes with it)."""
        super().__init__()
        assert block in ("adaln", "plain"), block
        self.block = block
        self.d = d_concept
        w = model_dim or d_concept
        self.w = w
        self.in_proj = nn.Linear(2 * d_concept, w)  # [z_t ; self_cond]
        self.pos = LearnedPositionalEmbedding(max_concepts, w)
        # the conditioning concepts get their own positional signal: their order
        # (program order) matters and should not rely on what the AE baked in
        self.cond_pos = LearnedPositionalEmbedding(max_concepts, w)
        if w != d_concept:
            self.cond_proj = nn.Linear(d_concept, w)
        self.t_mlp = nn.Sequential(
            nn.Linear(w, 4 * w), nn.SiLU(), nn.Linear(4 * w, w)
        )
        if block == "adaln":
            self.blocks = nn.ModuleList(
                [DiTBlock(w, n_heads, cross=cross_condition) for _ in range(n_layers)]
            )
            self.norm_out = nn.LayerNorm(w, elementwise_affine=False, eps=1e-6)
            self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(w, 2 * w))
            nn.init.zeros_(self.ada_out[-1].weight)
            nn.init.zeros_(self.ada_out[-1].bias)
        else:
            self.blocks = nn.ModuleList(
                [TransformerBlock(w, n_heads, causal=False, cross=cross_condition)
                 for _ in range(n_layers)]
            )
            self.t_pos = nn.Parameter(torch.zeros(1, 1, w))  # marks the time token
            nn.init.normal_(self.t_pos, std=0.02)
            self.norm_out = nn.LayerNorm(w)
        self.out_proj = nn.Linear(w, d_concept)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None,
                self_cond: Optional[torch.Tensor] = None):
        """z_t: (B, m, d) noised concepts; t: (B,) in [0,1]; cond: (B, mc, d) or None;
        self_cond: (B, m, d) previous clean estimate or None -> predicted x0 (B, m, d)."""
        if self_cond is None:
            self_cond = torch.zeros_like(z_t)
        c = self.t_mlp(timestep_embedding(t, self.w))
        x = self.pos(self.in_proj(torch.cat([z_t, self_cond], dim=-1)))
        if cond is not None:
            if self.w != self.d:
                cond = self.cond_proj(cond)
            cond = self.cond_pos(cond)
        if self.block == "adaln":
            for blk in self.blocks:
                x = blk(x, c, ctx=cond)
            shift, scale = self.ada_out(c).chunk(2, dim=-1)
            x = self.norm_out(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        else:
            x = torch.cat([c.unsqueeze(1) + self.t_pos, x], dim=1)
            for blk in self.blocks:
                x = blk(x, ctx=cond)
            x = self.norm_out(x[:, 1:])
        return self.out_proj(x)  # predicted clean concepts x0, (B, m, d)
