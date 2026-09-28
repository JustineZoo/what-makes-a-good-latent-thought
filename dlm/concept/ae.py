"""Concept autoencoder = the density dial (torch).

Encodes a span of L tokens into m = ceil(L/R) dense concept vectors and decodes
them back to tokens from the concepts alone. The training objective is the exact
ELBO with a free-bits rate term (plan, Component A):

    L_AE = CE(x, decode(z))  +  beta * KL_freebits(q(z|x) || N(0, I))

``R`` (concepts-per-token) is the primary density knob; ``beta`` (bits-per-concept)
is the secondary one. After training, the AE is **frozen** and shared by the AR
and diffusion generators so the only variable downstream is the paradigm.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import FactorizedEncoder, FixedRatioEncoder
from .decoder import BottleneckDecoder
from .rate import kl_free_bits, rate_bits


@dataclass
class AEConfig:
    vocab_size: int
    d_concept: int = 256
    compression_ratio: int = 8
    max_tokens: int = 64
    beta: float = 1.0
    free_bits: float = 0.5
    enc_layers: int = 4
    dec_layers: int = 4
    n_heads: int = 8
    n_slots: int = 0  # >0: FactorizedEncoder with K sub-slot codes per concept
    blockwise: bool = False  # strict per-concept information boundary in both halves


class ConceptAutoencoder(nn.Module):
    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.n_slots > 0:
            self.encoder = FactorizedEncoder(
                cfg.vocab_size, cfg.d_concept, cfg.compression_ratio, cfg.max_tokens,
                cfg.enc_layers, cfg.n_heads, n_slots=cfg.n_slots, blockwise=cfg.blockwise,
            )
        else:
            self.encoder = FixedRatioEncoder(
                cfg.vocab_size, cfg.d_concept, cfg.compression_ratio, cfg.max_tokens,
                cfg.enc_layers, cfg.n_heads, blockwise=cfg.blockwise,
            )
        self.decoder = BottleneckDecoder(
            cfg.vocab_size, cfg.d_concept, cfg.max_tokens, cfg.dec_layers, cfg.n_heads,
            blockwise=cfg.blockwise, compression_ratio=cfg.compression_ratio,
        )
        # Latent calibration (the LDM "scale factor", per-dim): generators diffuse
        # from N(0, I), so the concept space they see must be standardized or the
        # comparison silently penalizes diffusion for a scaling artifact. Identity
        # until ``calibrate`` is called (after AE training).
        self.register_buffer("latent_mean", torch.zeros(1, 1, cfg.d_concept))
        self.register_buffer("latent_std", torch.ones(1, 1, cfg.d_concept))

    @property
    def R(self) -> int:
        return self.cfg.compression_ratio

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    @torch.no_grad()
    def calibrate(self, token_batches) -> None:
        """Set latent_mean/std from the posterior means of sample spans (a list of
        (B, L) long tensors, e.g. prompt and answer batches)."""
        mus = [self.encoder(t)[0].reshape(-1, self.cfg.d_concept) for t in token_batches]
        flat = torch.cat(mus, dim=0)
        self.latent_mean.copy_(flat.mean(dim=0).view(1, 1, -1))
        self.latent_std.copy_(flat.std(dim=0).clamp(min=1e-4).view(1, 1, -1))

    def encode(self, tokens: torch.Tensor, sample: bool = False, normalize: bool = True) -> torch.Tensor:
        mu, logvar = self.encoder(tokens)
        z = self.reparameterize(mu, logvar) if sample else mu
        return (z - self.latent_mean) / self.latent_std if normalize else z

    def decode(self, z: torch.Tensor, out_len: int, denormalize: bool = True) -> torch.Tensor:
        if denormalize:
            z = z * self.latent_std + self.latent_mean
        return self.decoder(z, out_len)

    def forward(self, tokens: torch.Tensor):
        mu, logvar = self.encoder(tokens)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z, out_len=tokens.shape[1])
        return {"logits": logits, "mu": mu, "logvar": logvar, "z": z}

    def loss(self, tokens: torch.Tensor):
        out = self(tokens)
        recon = F.cross_entropy(out["logits"].reshape(-1, self.cfg.vocab_size), tokens.reshape(-1))
        kl = kl_free_bits(out["mu"], out["logvar"], self.cfg.free_bits)
        loss = recon + self.cfg.beta * kl
        with torch.no_grad():
            token_acc = (out["logits"].argmax(-1) == tokens).float().mean()
            bits = rate_bits(out["mu"], out["logvar"])
        return {"loss": loss, "recon": recon, "kl": kl, "rate_bits": bits, "token_acc": token_acc}

    # -- persistence -------------------------------------------------------
    def save(self, path: str):
        torch.save({"state_dict": self.state_dict(), "cfg": asdict(self.cfg)}, path)

    @classmethod
    def load(cls, path: str, map_location=None) -> "ConceptAutoencoder":
        ckpt = torch.load(path, map_location=map_location)
        model = cls(AEConfig(**ckpt["cfg"]))
        model.load_state_dict(ckpt["state_dict"])
        return model
