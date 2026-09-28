"""Concept-space diffusion generator (rectified-flow schedule, x0-prediction,
self-conditioning; torch).

Denoises the whole answer-concept sequence in parallel, conditioned on the context
concepts, over ``T`` steps. ``T`` -- the serial-compute knob -- is decoupled from
the number of concepts, which is the mechanism the study tests (Q2/Q3).

Design (standard-strength continuous text diffusion, see dit.py docstring): the
network predicts the CLEAN concepts x0 at every t and receives its previous clean
estimate as self-conditioning, so each of the T sampling steps refines a working
answer (Euler on the induced rectified-flow field v = (z_t - x0_hat)/t). Training
uses the Bit-Diffusion recipe: 50% of steps compute a no-grad first estimate and
condition on it, 50% condition on zeros.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from dlm.models.dit import ConceptDenoiser
from .base import Generator


class ConceptFlowMatching(Generator):
    paradigm = "concept_diffusion"

    def __init__(self, d_concept: int, n_layers: int = 12, n_heads: int = 8, max_concepts: int = 64,
                 model_dim: int = None, block: str = "adaln"):
        super().__init__()
        self.net = ConceptDenoiser(d_concept, n_layers, n_heads, max_concepts,
                                   cross_condition=True, model_dim=model_dim, block=block)
        self.d = d_concept
        self.block = block

    def fit_step(self, target: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B = target.shape[0]
        t = torch.rand(B, device=target.device)
        noise = torch.randn_like(target)
        z_t = (1 - t)[:, None, None] * target + t[:, None, None] * noise
        self_cond = None
        if float(torch.rand(())) < 0.5:
            with torch.no_grad():
                self_cond = self.net(z_t, t, cond=cond).detach()
        x0_hat = self.net(z_t, t, cond=cond, self_cond=self_cond)
        return F.mse_loss(x0_hat, target)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, n_target: int, T: int = 32, **_) -> torch.Tensor:
        B = cond.shape[0]
        z = torch.randn(B, n_target, self.d, device=cond.device)
        x0_hat = None
        for i in range(T):  # integrate t: 1 -> 0; the last step lands exactly on x0_hat
            t = 1.0 - i / T
            tv = torch.full((B,), t, device=cond.device)
            x0_hat = self.net(z, tv, cond=cond, self_cond=x0_hat)
            z = z - (1.0 / T) * (z - x0_hat) / t
        return z

    def serial_cost(self, T: int = 32, **_) -> int:
        return T
