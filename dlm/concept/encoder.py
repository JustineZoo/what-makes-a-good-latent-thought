"""Token -> concept encoders (torch).

``FixedRatioEncoder`` is the primary, controlled dial: it deterministically groups
every ``R`` tokens into one concept (m = ceil(L/R) concepts), giving a clean
compression-ratio knob directly comparable to DLCM. It outputs a Gaussian
posterior (mu, logvar) per concept for the VAE rate term.

``LearnedBoundaryEncoder`` (Milestone 3) will instead *learn* variable-length
concept boundaries (DLCM cosine-dissimilarity detector / H-Net-style); stubbed
here with the same interface so it drops into the AE unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dlm.models.backbone import LearnedPositionalEmbedding, TransformerBlock, block_diagonal_mask


class FixedRatioEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_concept: int,
        compression_ratio: int,
        max_tokens: int,
        n_layers: int = 4,
        n_heads: int = 8,
        blockwise: bool = False,
    ):
        super().__init__()
        self.R = compression_ratio
        self.d = d_concept
        self.max_tokens = max_tokens
        # blockwise: self-attention is restricted to each R-token block, so every
        # concept is a function of its own R tokens (and their positions) only
        self.blockwise = blockwise
        self.embed = nn.Embedding(vocab_size, d_concept)
        self.pos = LearnedPositionalEmbedding(max_tokens, d_concept)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_concept, n_heads, causal=False) for _ in range(n_layers)]
        )
        self.to_mu = nn.Linear(d_concept, d_concept)
        self.to_logvar = nn.Linear(d_concept, d_concept)
        # start with a tight posterior (sigma ~ e^-2 ~ 0.14): a unit-variance init
        # would drown mu in reparameterization noise early in training and stall
        # reconstruction; the KL/free-bits term re-widens it where affordable
        nn.init.constant_(self.to_logvar.bias, -4.0)

    def forward(self, tokens: torch.Tensor):
        """tokens: (B, L) long -> mu, logvar: (B, m, d) with m = ceil(L / R)."""
        B, L = tokens.shape
        x = self.pos(self.embed(tokens))
        mask = block_diagonal_mask(L, self.R, x.device) if self.blockwise else None
        for blk in self.blocks:
            x = blk(x, self_mask=mask)
        pad = (-L) % self.R
        if pad:
            x = F.pad(x, (0, 0, 0, pad))  # pad token dim up to multiple of R
        m = x.shape[1] // self.R
        x = x.view(B, m, self.R, self.d).mean(dim=2)  # mean-pool each R-block
        return self.to_mu(x), self.to_logvar(x).clamp(-8, 8)


class FactorizedEncoder(nn.Module):
    """Product-structured concept code: each concept is the CONCAT of K sub-slot
    codes, slot j = shared Linear(d -> d/K) applied to the mean-pool of the j-th
    (R/K)-token sub-block. Total dims, concept count, and the generator-facing
    interface are identical to ``FixedRatioEncoder`` -- the only change is the
    internal factorization of the code.

    Why: the R=16 density wall (RESULTS.md finding 2) is a code-learnability
    failure. Counting must respect the chain: a priori a 3.2-state block has
    ~120^3.2 ~ 4.4M contents, but consecutive states differ by one of only 4
    generators, so REALIZABLE contents are ~120*4^3 per block phase (order 1e4
    total; each seen hundreds of times at our budgets -- exposure is not what
    binds). What binds is that an entangled code places those contents at
    arbitrary unrelated points of R^d: the generator's conditional map becomes a
    ~1e4-entry lookup table with no transfer between entries. A shared
    per-sub-block code map factors that map into K shared sub-maps over ~1e2
    sub-contents each -- compositional by construction; the wall cracking under
    identical generators shows code structure (not bits per unit) is the
    binding constraint.
    """

    def __init__(
        self,
        vocab_size: int,
        d_concept: int,
        compression_ratio: int,
        max_tokens: int,
        n_layers: int = 4,
        n_heads: int = 8,
        n_slots: int = 4,
        blockwise: bool = False,
    ):
        super().__init__()
        assert compression_ratio % n_slots == 0, "R must be divisible by n_slots"
        assert d_concept % n_slots == 0, "d_concept must be divisible by n_slots"
        self.R = compression_ratio
        self.d = d_concept
        self.K = n_slots
        self.max_tokens = max_tokens
        self.blockwise = blockwise  # see FixedRatioEncoder
        self.embed = nn.Embedding(vocab_size, d_concept)
        self.pos = LearnedPositionalEmbedding(max_tokens, d_concept)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_concept, n_heads, causal=False) for _ in range(n_layers)]
        )
        ds = d_concept // n_slots
        self.to_mu = nn.Linear(d_concept, ds)      # shared across slots
        self.to_logvar = nn.Linear(d_concept, ds)  # shared across slots
        nn.init.constant_(self.to_logvar.bias, -4.0)

    def forward(self, tokens: torch.Tensor):
        """tokens: (B, L) long -> mu, logvar: (B, m, d) with m = ceil(L / R)."""
        B, L = tokens.shape
        x = self.pos(self.embed(tokens))
        mask = block_diagonal_mask(L, self.R, x.device) if self.blockwise else None
        for blk in self.blocks:
            x = blk(x, self_mask=mask)
        pad = (-L) % self.R
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        m = x.shape[1] // self.R
        sub = x.view(B, m, self.K, self.R // self.K, self.d).mean(dim=3)  # (B, m, K, d)
        mu = self.to_mu(sub).reshape(B, m, self.d)
        logvar = self.to_logvar(sub).reshape(B, m, self.d).clamp(-8, 8)
        return mu, logvar


class LearnedBoundaryEncoder(nn.Module):
    """STUB (Milestone 3): learned variable-length concept boundaries.

    Planned: contextualize tokens, score adjacent-token dissimilarity, threshold to
    boundaries, pool within segments, and add an auxiliary loss regularizing the
    expected boundary rate to a target compression R (DLCM-style).
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("LearnedBoundaryEncoder is scheduled for Milestone 3.")
