"""Per-concept output heads for the autoregressive generator (torch).

The AR baseline must be a *fair* baseline: predicting a continuous next-concept
with a plain MSE regression collapses to the conditional mean and cannot model a
multimodal next-concept distribution, which would unfairly handicap AR. So we
give it a proper probabilistic head:

  * ``MARHead``  -- a tiny per-concept flow-matching head (à la MAR,
    arXiv:2406.11838): denoise one concept vector conditioned on the AR hidden
    state. ``head_inner_steps`` sampling steps count toward serial compute.
  * ``GMMHead``  -- a diagonal Gaussian mixture head (cheap, single-shot sample).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import timestep_embedding


class MARHead(nn.Module):
    def __init__(self, d: int, hidden: int = 4, depth: int = 3):
        super().__init__()
        h = hidden * d
        self.d = d
        layers = [nn.Linear(3 * d, h), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(h, h), nn.SiLU()]
        layers += [nn.Linear(h, d)]
        self.net = nn.Sequential(*layers)

    def _v(self, z_t, cond, t):
        te = timestep_embedding(t, self.d)
        return self.net(torch.cat([z_t, cond, te], dim=-1))

    def loss(self, cond: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Rectified-flow loss for one concept per row. cond,target: (N, d)."""
        N = target.shape[0]
        t = torch.rand(N, device=target.device)
        noise = torch.randn_like(target)
        z_t = (1 - t)[:, None] * target + t[:, None] * noise
        v_target = noise - target
        v = self._v(z_t, cond, t)
        return F.mse_loss(v, v_target)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, steps: int = 4) -> torch.Tensor:
        N = cond.shape[0]
        z = torch.randn(N, self.d, device=cond.device)
        for i in range(steps):  # integrate t: 1 -> 0
            t = torch.full((N,), 1.0 - i / steps, device=cond.device)
            z = z - (1.0 / steps) * self._v(z, cond, t)
        return z


class GMMHead(nn.Module):
    def __init__(self, d: int, k: int = 4):
        super().__init__()
        self.d, self.k = d, k
        self.to_params = nn.Linear(d, k * (2 * d + 1))

    def _params(self, cond):
        p = self.to_params(cond)
        means, logvars, logits = torch.split(p, [self.k * self.d, self.k * self.d, self.k], dim=-1)
        means = means.view(*cond.shape[:-1], self.k, self.d)
        logvars = logvars.view(*cond.shape[:-1], self.k, self.d).clamp(-8, 8)
        return means, logvars, logits

    def loss(self, cond: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        means, logvars, logits = self._params(cond)
        x = target.unsqueeze(-2)
        log_normal = -0.5 * (((x - means) ** 2) / logvars.exp() + logvars + math.log(2 * math.pi)).sum(-1)
        log_mix = torch.log_softmax(logits, dim=-1) + log_normal
        return -(torch.logsumexp(log_mix, dim=-1)).mean()

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, steps: int = 1) -> torch.Tensor:
        means, logvars, logits = self._params(cond)
        comp = torch.distributions.Categorical(logits=logits).sample()
        idx = comp[..., None, None].expand(*comp.shape, 1, self.d)
        mu = means.gather(-2, idx).squeeze(-2)
        sd = (0.5 * logvars).exp().gather(-2, idx).squeeze(-2)
        return mu + sd * torch.randn_like(mu)


HEADS = {"mar": MARHead, "gmm": GMMHead}
