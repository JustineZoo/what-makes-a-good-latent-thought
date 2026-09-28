"""Concept-space autoregressive generator (torch) -- the fair AR baseline.

A causal transformer over the concept sequence predicts each answer concept from
the prefix (context concepts + previously generated answer concepts), and a
probabilistic head (MAR flow head or GMM) models the continuous next-concept
distribution. Serial compute = (#answer concepts) x (head inner steps), which
*shrinks* as the compression ratio R grows -- the exact quantity the diffusion
model keeps fixed at T.
"""
from __future__ import annotations

import torch

from dlm.models.backbone import LearnedPositionalEmbedding, TransformerBlock
from dlm.models.heads import HEADS
from .base import Generator


class ConceptAR(Generator):
    paradigm = "concept_ar"

    def __init__(
        self,
        d_concept: int,
        n_layers: int = 12,
        n_heads: int = 8,
        max_concepts: int = 64,
        head: str = "mar",
        head_inner_steps: int = 4,
        model_dim: int = None,
    ):
        """``model_dim`` decouples backbone width from the concept dimension (None
        = run at d_concept; module tree unchanged so old checkpoints load)."""
        super().__init__()
        self.d = d_concept
        w = model_dim or d_concept
        self.w = w
        self.head_inner_steps = head_inner_steps
        if w != d_concept:
            self.in_proj = torch.nn.Linear(d_concept, w)
            self.head_in = torch.nn.Linear(w, d_concept)
        self.pos = LearnedPositionalEmbedding(max_concepts, w)
        self.bos = torch.nn.Parameter(torch.zeros(1, 1, d_concept))
        self.blocks = torch.nn.ModuleList(
            [TransformerBlock(w, n_heads, causal=True) for _ in range(n_layers)]
        )
        self.norm = torch.nn.LayerNorm(w)
        self.head = HEADS[head](d_concept)

    def _hidden(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, n, d_concept) -> per-position features projected back to the
        concept dimension for the head."""
        x = self.in_proj(seq) if self.w != self.d else seq
        x = self.pos(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head_in(x) if self.w != self.d else x

    def fit_step(self, target: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, mt, d = target.shape
        bos = self.bos.expand(B, 1, d)
        # sequence: [cond ..., BOS, target[:-1]] ; predict each target concept from its prefix
        seq = torch.cat([cond, bos, target[:, :-1]], dim=1)
        h = self._hidden(seq)
        pred_h = h[:, -mt:, :].reshape(B * mt, d)  # hidden states aligned to each target position
        return self.head.loss(pred_h, target.reshape(B * mt, d))

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, n_target: int, **_) -> torch.Tensor:
        B, _, d = cond.shape
        seq = torch.cat([cond, self.bos.expand(B, 1, d)], dim=1)
        out = []
        for _ in range(n_target):
            h = self._hidden(seq)[:, -1, :]
            nxt = self.head.sample(h, steps=self.head_inner_steps)
            out.append(nxt)
            seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
        return torch.stack(out, dim=1)

    def serial_cost(self, n_target: int, **_) -> int:
        inner = self.head_inner_steps if hasattr(self.head, "sample") else 1
        return n_target * inner
