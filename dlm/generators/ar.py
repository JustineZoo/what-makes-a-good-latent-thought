"""Concept-space autoregressive generator (torch) with prompt-grounded EntitySlotBuffer."""
from __future__ import annotations

import torch
import torch.nn as nn

from dlm.models.backbone import LearnedPositionalEmbedding, TransformerBlock
from dlm.models.heads import HEADS
from .base import Generator


class EntitySlotBuffer(nn.Module):
    """
    Maintains explicit coordinate slots for permutation entities.
    Initializes slots dynamically from prompt context and sharpens routing via temperature scaling.
    """
    def __init__(self, d_model: int = 256, n_slots: int = 5, n_heads: int = 4, tau: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_slots = n_slots
        self.tau = tau

        # Learnable queries to extract initial object positions from prompt context
        self.slot_queries = nn.Parameter(torch.randn(1, n_slots, d_model) * 0.02)
        self.init_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.init_norm = nn.LayerNorm(d_model)

        # Cross-attention to condition hidden states on slot memory
        self.read_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

        # State router with temperature scaling
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_slots),
        )
        self.update_proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def init_slots(self, cond: torch.Tensor) -> torch.Tensor:
        """Dynamically extract object coordinates from the prompt prefix."""
        B = cond.shape[0]
        queries = self.slot_queries.expand(B, -1, -1)
        slots_0, _ = self.init_attn(query=queries, key=cond, value=cond)
        return self.init_norm(slots_0)

    def forward(self, h_t: torch.Tensor, slots: torch.Tensor):
        if h_t.dim() == 2:
            h_t = h_t.unsqueeze(1)

        # Read: Condition transition representation on slot memory
        slot_ctx, _ = self.read_attn(query=h_t, key=slots, value=slots)
        h_cond = h_t + slot_ctx

        # Write: Temperature-sharpened routing (tau = 0.1 forces coordinate specificity)
        logits = self.router(h_cond.squeeze(1))
        alpha = torch.softmax(logits / self.tau, dim=-1).unsqueeze(-1)  # (B, n_slots, 1)

        tiled_h = h_cond.expand(-1, self.n_slots, -1)
        update_val = self.update_proj(torch.cat([slots, tiled_h], dim=-1))

        new_slots = self.norm((1.0 - alpha) * slots + alpha * update_val)
        return h_cond.squeeze(1), new_slots


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
        use_slot_buffer: bool = False,
        n_slots: int = 5,
        slot_tau: float = 0.1,
    ):
        super().__init__()
        self.d = d_concept
        w = model_dim or d_concept
        self.w = w
        self.head_inner_steps = head_inner_steps
        self.use_slot_buffer = use_slot_buffer

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

        if self.use_slot_buffer:
            self.slot_buffer = EntitySlotBuffer(d_model=self.d, n_slots=n_slots, tau=slot_tau)

    def _hidden(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(seq) if self.w != self.d else seq
        x = self.pos(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head_in(x) if self.w != self.d else x

    def fit_step(self, target: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, mt, d = target.shape
        bos = self.bos.expand(B, 1, d)
        seq = torch.cat([cond, bos, target[:, :-1]], dim=1)
        h = self._hidden(seq)
        pred_h = h[:, -mt:, :]

        if self.use_slot_buffer:
            slots = self.slot_buffer.init_slots(cond)
            h_conditioned = []
            for t in range(mt):
                h_t, slots = self.slot_buffer(pred_h[:, t, :], slots)
                h_conditioned.append(h_t)
            pred_h = torch.stack(h_conditioned, dim=1)

        pred_h_flat = pred_h.reshape(B * mt, d)
        return self.head.loss(pred_h_flat, target.reshape(B * mt, d))

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, n_target: int, **_) -> torch.Tensor:
        B, _, d = cond.shape
        seq = torch.cat([cond, self.bos.expand(B, 1, d)], dim=1)
        out = []

        if self.use_slot_buffer:
            slots = self.slot_buffer.init_slots(cond)

        for _ in range(n_target):
            h = self._hidden(seq)[:, -1, :]
            if self.use_slot_buffer:
                h, slots = self.slot_buffer(h, slots)
            nxt = self.head.sample(h, steps=self.head_inner_steps)
            out.append(nxt)
            seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
        return torch.stack(out, dim=1)

    def serial_cost(self, n_target: int, **_) -> int:
        inner = self.head_inner_steps if hasattr(self.head, "sample") else 1
        return n_target * inner