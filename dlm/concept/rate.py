"""Rate term for the concept autoencoder (torch).

The KL from the Gaussian posterior q(z|x)=N(mu, exp(logvar)) to the standard
normal prior is a tractable upper bound on the mutual information I(X;Z) -- the
"bits per concept". ``beta`` scales it (the rate-distortion knob); ``free_bits``
imposes a per-dimension floor so the term cannot collapse a dimension to zero,
which is the main defence against posterior collapse.
"""
from __future__ import annotations

import torch


def kl_to_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Per-element KL, same shape as ``mu``. KL = 0.5 (mu^2 + var - 1 - logvar)."""
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)


def rate_bits(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Mean rate in *bits per token-span*, summed over concept & feature dims."""
    nats = kl_to_standard_normal(mu, logvar).sum(dim=(-2, -1)).mean()
    return nats / torch.log(torch.tensor(2.0, device=mu.device))


def kl_free_bits(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0) -> torch.Tensor:
    """Free-bits KL (nats): floor each latent dimension's KL at ``free_bits`` before
    summing. Returns a scalar (batch mean of the summed, floored KL)."""
    kl = kl_to_standard_normal(mu, logvar)  # (B, m, d)
    if free_bits > 0.0:
        kl = torch.clamp(kl, min=free_bits)
    return kl.sum(dim=(-2, -1)).mean()
