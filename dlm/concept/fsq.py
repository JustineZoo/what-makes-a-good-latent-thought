"""Discrete factorized concept codec via finite scalar quantization (torch).

Each concept is K sub-slot codes (as in ``FactorizedEncoder``), but each sub-slot
is projected to ``code_dims`` scalars, squashed by tanh, and quantized to
``levels`` uniform levels per dim (straight-through estimator; Mentzer et al.,
FSQ). A sub-slot is therefore a DISCRETE token from a codebook of size
``levels**code_dims`` -- so a trace becomes a short sequence of code tokens and
the token-level generators (softmax AR, LLaDA-style masked diffusion) run over
concepts natively. This completes the paradigm triangle: continuous flow
matching, continuous AR, and masked DISCRETE diffusion, all over the same
frozen compressed representation.

Interfaces:
  * continuous (drop-in for the recon gate / eval_reconstruction):
      encode(tokens) -> quantized values (B, m, K*code_dims)
      decode(values, out_len) -> token logits
  * discrete (for code-token generators):
      encode_ids(tokens) -> (B, m*K) long in [0, codebook_size)
      decode_ids(ids, out_len) -> token logits
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dlm.models.backbone import LearnedPositionalEmbedding, TransformerBlock
from .decoder import BottleneckDecoder


@dataclass
class FSQConfig:
    vocab_size: int
    d_concept: int = 384
    compression_ratio: int = 16
    n_slots: int = 4
    code_dims: int = 3
    levels: int = 8
    max_tokens: int = 200
    enc_layers: int = 6
    dec_layers: int = 6
    n_heads: int = 8


class FSQAutoencoder(nn.Module):
    def __init__(self, cfg: FSQConfig):
        super().__init__()
        assert cfg.compression_ratio % cfg.n_slots == 0
        self.cfg = cfg
        d, K = cfg.d_concept, cfg.n_slots
        self.embed = nn.Embedding(cfg.vocab_size, d)
        self.pos = LearnedPositionalEmbedding(cfg.max_tokens, d)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d, cfg.n_heads, causal=False) for _ in range(cfg.enc_layers)]
        )
        # Normalize the pooled FEATURES (not the 3 code dims -- a 3-dim LayerNorm
        # over-constrains the code space) before the shared code projection.
        # Without normalization the codec collapses to ~2 of 512 codes: tanh
        # saturates early on an easy low-bit solution and cannot recover.
        self.pre_norm = nn.LayerNorm(d)
        self.to_code = nn.Linear(d, cfg.code_dims)      # shared across slots
        self.from_code = nn.Linear(cfg.code_dims, d // K)  # shared across slots
        self.decoder = BottleneckDecoder(cfg.vocab_size, d, cfg.max_tokens,
                                         cfg.dec_layers, cfg.n_heads)

    # -- properties ---------------------------------------------------------
    @property
    def R(self) -> int:
        return self.cfg.compression_ratio

    @property
    def codebook_size(self) -> int:
        return self.cfg.levels ** self.cfg.code_dims

    # -- quantization -------------------------------------------------------
    def _pre_quant(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens (B, L) -> tanh code values (B, m, K, code_dims), un-quantized."""
        B, L = tokens.shape
        cfg = self.cfg
        x = self.pos(self.embed(tokens))
        for blk in self.blocks:
            x = blk(x)
        pad = (-L) % cfg.compression_ratio
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        m = x.shape[1] // cfg.compression_ratio
        sub = x.view(B, m, cfg.n_slots, cfg.compression_ratio // cfg.n_slots, cfg.d_concept).mean(3)
        return torch.tanh(self.to_code(self.pre_norm(sub)))  # (B, m, K, cd) in (-1, 1)

    def _quantize(self, y: torch.Tensor) -> torch.Tensor:
        """Uniform grid on [-1, 1]. Training uses the additive-noise proxy
        (uniform dither of one grid step; Balle et al.) -- fully differentiable,
        immune to the STE-through-saturated-tanh collapse; eval uses hard
        rounding with a straight-through formulation for API symmetry."""
        Lv = self.cfg.levels
        if self.training:
            return y + (torch.rand_like(y) - 0.5) * (2.0 / (Lv - 1))
        q = torch.round((y + 1) / 2 * (Lv - 1)) / (Lv - 1) * 2 - 1
        return y + (q - y).detach()

    def _values_to_concepts(self, q: torch.Tensor) -> torch.Tensor:
        """(B, m, K, cd) quantized values -> (B, m, d_concept) decoder input."""
        B, m = q.shape[0], q.shape[1]
        return self.from_code(q).reshape(B, m, self.cfg.d_concept)

    # -- continuous interface (recon gate compatible) ------------------------
    def encode(self, tokens: torch.Tensor, **_) -> torch.Tensor:
        q = self._quantize(self._pre_quant(tokens))
        return q.reshape(q.shape[0], q.shape[1], -1)  # (B, m, K*cd)

    def decode(self, values: torch.Tensor, out_len: int, **_) -> torch.Tensor:
        B, m = values.shape[0], values.shape[1]
        q = values.view(B, m, self.cfg.n_slots, self.cfg.code_dims)
        return self.decoder(self._values_to_concepts(q), out_len)

    # -- discrete interface ---------------------------------------------------
    @torch.no_grad()
    def encode_ids(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, L) -> (B, m*K) code-token ids (mixed radix over code dims)."""
        y = self._pre_quant(tokens)
        Lv = self.cfg.levels
        digits = torch.round((y + 1) / 2 * (Lv - 1)).long().clamp(0, Lv - 1)
        radix = Lv ** torch.arange(self.cfg.code_dims, device=tokens.device)
        ids = (digits * radix).sum(-1)  # (B, m, K)
        return ids.reshape(ids.shape[0], -1)

    def ids_to_values(self, ids: torch.Tensor) -> torch.Tensor:
        """(B, m*K) -> (B, m, K, code_dims) grid values."""
        cfg = self.cfg
        B = ids.shape[0]
        m = ids.shape[1] // cfg.n_slots
        ids = ids.view(B, m, cfg.n_slots, 1)
        radix = cfg.levels ** torch.arange(cfg.code_dims, device=ids.device)
        digits = (ids // radix) % cfg.levels
        return digits.float() / (cfg.levels - 1) * 2 - 1

    @torch.no_grad()
    def decode_ids(self, ids: torch.Tensor, out_len: int) -> torch.Tensor:
        return self.decoder(self._values_to_concepts(self.ids_to_values(ids)), out_len)

    # -- training ------------------------------------------------------------
    def loss(self, tokens: torch.Tensor):
        y = self._pre_quant(tokens)
        q = self._quantize(y)
        logits = self.decoder(self._values_to_concepts(q), tokens.shape[1])
        recon = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), tokens.reshape(-1))
        # anti-collapse variance floor: each code dim's batch std should span the
        # grid (~uniform on [-1,1] has std 0.58); hinged so it vanishes once codes
        # actually spread
        std = y.reshape(-1, self.cfg.code_dims).std(dim=0)
        spread = F.relu(0.5 - std).sum()
        loss = recon + spread
        with torch.no_grad():
            token_acc = (logits.argmax(-1) == tokens).float().mean()
        return {"loss": loss, "recon": recon, "spread": spread, "token_acc": token_acc}

    def save(self, path: str):
        torch.save({"state_dict": self.state_dict(), "cfg": asdict(self.cfg),
                    "kind": "fsq"}, path)

    @classmethod
    def load(cls, path: str, map_location=None) -> "FSQAutoencoder":
        ckpt = torch.load(path, map_location=map_location)
        model = cls(FSQConfig(**ckpt["cfg"]))
        model.load_state_dict(ckpt["state_dict"])
        return model
