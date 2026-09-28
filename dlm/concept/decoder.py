"""Concept -> token decoder (torch).

Non-autoregressive by design: L learnable positional queries cross-attend to the
concept sequence and emit token logits in parallel. Decoding tokens from the
concepts *alone* (no token-level self-conditioning) makes the concepts a genuine
information bottleneck -- the decoder cannot smuggle information around ``z``,
which is what keeps the rate term meaningful and blocks posterior collapse.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from dlm.models.backbone import LearnedPositionalEmbedding, TransformerBlock, block_diagonal_mask


class BottleneckDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_concept: int,
        max_tokens: int,
        n_layers: int = 4,
        n_heads: int = 8,
        blockwise: bool = False,
        compression_ratio: int = 0,
    ):
        super().__init__()
        self.max_tokens = max_tokens
        # blockwise: query t cross-attends only to concept t // R and self-attends
        # only within its own R-token block, so block k is decoded from concept k
        # alone (a strict information boundary per concept span)
        self.blockwise = blockwise
        self.R = compression_ratio
        if blockwise:
            assert compression_ratio > 0, "blockwise decoding needs compression_ratio"
        self.query = LearnedPositionalEmbedding(max_tokens, d_concept)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_concept, n_heads, causal=False, cross=True) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_concept)
        self.lm_head = nn.Linear(d_concept, vocab_size)

    def forward(self, z: torch.Tensor, out_len: int) -> torch.Tensor:
        """z: (B, m, d) concepts -> logits: (B, out_len, vocab)."""
        B = z.shape[0]
        x = self.query(torch.zeros(B, out_len, z.shape[-1], device=z.device, dtype=z.dtype))
        self_mask = cross_mask = None
        if self.blockwise:
            self_mask = block_diagonal_mask(out_len, self.R, z.device)          # (L, L)
            blk_id = torch.arange(out_len, device=z.device) // self.R
            cross_mask = blk_id[:, None] == torch.arange(z.shape[1], device=z.device)[None, :]  # (L, m)
        for layer in self.blocks:
            x = layer(x, ctx=z, self_mask=self_mask, cross_mask=cross_mask)
        return self.lm_head(self.norm(x))
