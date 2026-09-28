"""Generator interface (torch) -- the paradigm abstraction and the SOTA swap point.

A ``Generator`` models p(target_concepts | context_concepts) over the *frozen*
concept space produced by a ``ConceptAutoencoder``. AR and diffusion implement the
same three methods, so the ablation harness treats them identically and the only
variable is the paradigm (and, for diffusion, the number of steps T).

============================  PLUGGING IN SOTA BASELINES  =======================
To evaluate a state-of-the-art diffusion LM (LLaDA / MDLM / SEDD / Dream /
block-diffusion) in this study, subclass ``Generator`` and implement:

    fit_step(target, cond) -> scalar loss        # one optimizer step of their model
    sample(cond, n_target, **kw) -> concepts     # their sampler, returning (B, n_target, d)
    serial_cost(**kw) -> int                     # critical-path serial calls (see eval.serial_compute)

Keep the concept dimensionality and the frozen AE identical to the in-repo
generators so the comparison stays controlled. Nothing else in the harness changes.
================================================================================
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Generator(nn.Module, ABC):
    paradigm: str = "abstract"

    @abstractmethod
    def fit_step(self, target: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Return a scalar training loss for one batch.
        target: (B, mt, d) answer concepts; cond: (B, mc, d) context concepts."""

    @abstractmethod
    @torch.no_grad()
    def sample(self, cond: torch.Tensor, n_target: int, **kwargs) -> torch.Tensor:
        """Return generated answer concepts (B, n_target, d)."""

    @abstractmethod
    def serial_cost(self, **kwargs) -> int:
        """Critical-path serial network calls for one generation (see eval.serial_compute)."""

    def nll(self, target: torch.Tensor, cond: torch.Tensor):  # optional
        raise NotImplementedError(f"{type(self).__name__} does not expose an exact NLL.")
