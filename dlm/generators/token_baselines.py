"""Token-level baselines = the R=1 anchors (torch).

These operate directly on tokens (no concept compression) and reproduce known
behaviour before any R>1 result is trusted (plan, verification/integration):
  * ``TokenAR``               -- standard causal LM; should show accuracy
    degrading with reasoning depth (the AR-CoT scaling we anchor against).
  * ``TokenMaskedDiffusion``  -- MDLM-style absorbing-state discrete diffusion;
    accuracy should improve with the number of denoising steps T (LLaDA-style).

They condition generation on the prompt and produce the (short) answer span.
Interface is token-native (prompt_ids / answer_ids) rather than the concept-space
``Generator`` API, which is why they live here.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dlm.models.backbone import (LearnedPositionalEmbedding, TransformerBlock,
                                 timestep_embedding)


class _TokenBackbone(nn.Module):
    def __init__(self, vocab_size, d, n_layers, n_heads, max_len, causal,
                 time_cond: bool = False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.pos = LearnedPositionalEmbedding(max_len, d)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d, n_heads, causal=causal) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab_size)
        self.d = d
        # additive noise-level conditioning (uniform-state diffusion needs it:
        # the optimal denoiser depends on alpha_t, unlike absorbing-state MDLM)
        self.tmlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                  nn.Linear(4 * d, d)) if time_cond else None

    def forward(self, ids, t: torch.Tensor | None = None):
        x = self.pos(self.embed(ids))
        if self.tmlp is not None:
            assert t is not None
            x = x + self.tmlp(timestep_embedding(t, self.d))[:, None, :]
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.norm(x))


class TokenAR(nn.Module):
    paradigm = "token_ar"

    def __init__(self, vocab_size, d=256, n_layers=12, n_heads=8, max_len=128):
        super().__init__()
        self.net = _TokenBackbone(vocab_size, d, n_layers, n_heads, max_len, causal=True)
        self.vocab_size = vocab_size

    def fit_step(self, prompt: torch.Tensor, answer: torch.Tensor) -> torch.Tensor:
        seq = torch.cat([prompt, answer], dim=1)
        logits = self.net(seq[:, :-1])
        la = answer.shape[1]
        pred = logits[:, -la:, :]
        return F.cross_entropy(pred.reshape(-1, self.vocab_size), answer.reshape(-1))

    @torch.no_grad()
    def sample(self, prompt: torch.Tensor, n_answer: int, temperature: float = 0.0, **_):
        seq = prompt
        for _ in range(n_answer):
            logits = self.net(seq)[:, -1, :]
            nxt = logits.argmax(-1) if temperature == 0 else torch.distributions.Categorical(
                logits=logits / temperature).sample()
            seq = torch.cat([seq, nxt[:, None]], dim=1)
        return seq[:, -n_answer:]

    def serial_cost(self, n_answer: int, **_) -> int:
        return n_answer


class TokenMaskedDiffusion(nn.Module):
    paradigm = "token_diffusion"

    def __init__(self, vocab_size, mask_id: int, d=256, n_layers=12, n_heads=8, max_len=128,
                 objective: str = "uniform"):
        """``objective``:
          * "uniform" -- masked CE averaged over masked positions (MaskGIT-style;
            the original baselines in this study).
          * "elbo"    -- the MDLM/LLaDA discrete-diffusion NELBO estimator for
            the linear schedule: t sampled low-discrepancy across the batch
            (MDLM's variance reducer), iid masking at rate t, per-row loss
            (1/t) * sum(masked CE) / L. For the linear schedule the two
            objectives coincide in expectation over the mask draw; they differ
            in per-sample weighting/variance. No timestep conditioning is fed
            to the network in either case (per MDLM's finding that it is
            unnecessary for absorbing-state diffusion).
        """
        super().__init__()
        assert objective in ("uniform", "elbo")
        self.net = _TokenBackbone(vocab_size, d, n_layers, n_heads, max_len, causal=False)
        self.vocab_size = vocab_size
        self.mask_id = mask_id
        self.objective = objective

    def fit_step(self, prompt: torch.Tensor, answer: torch.Tensor) -> torch.Tensor:
        B, la = answer.shape
        if self.objective == "elbo":
            # low-discrepancy t across the batch, clamped away from 0 for the 1/t weight
            t = ((torch.rand(1, device=answer.device) + torch.arange(B, device=answer.device) / B)
                 % 1.0).clamp(min=1e-3)
            ratio = t[:, None]
        else:
            ratio = torch.rand(B, 1, device=answer.device)
        mask = torch.rand(B, la, device=answer.device) < ratio
        if self.objective != "elbo":
            # ensure >=1 masked position; under the 1/t-weighted estimator this
            # forcing would BIAS the loss (a forced token at tiny t inherits the
            # 1/t weight) -- there, zero-mask rows correctly contribute zero
            mask[:, 0] |= ~mask.any(dim=1)
        corrupted = torch.where(mask, torch.full_like(answer, self.mask_id), answer)
        logits = self.net(torch.cat([prompt, corrupted], dim=1))[:, -la:, :]
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), answer.reshape(-1), reduction="none")
        if self.objective == "elbo":
            per_row = (loss.view(B, la) * mask).sum(dim=1) / (ratio.squeeze(1) * la)
            return per_row.mean()
        return (loss.view(B, la) * mask).sum() / mask.sum().clamp(min=1)

    @torch.no_grad()
    def sample(self, prompt: torch.Tensor, n_answer: int, T: int = 16,
               remask: bool = False, **_):
        """Two samplers over the same trained model.

        Absorbing (default): confidence-ordered iterative unmasking with a
        cumulative reveal schedule -- after step s exactly round(n*(s+1)/T)
        positions are revealed; revealed positions are never rewritten. With
        T > n_answer some steps are no-ops: this sampler cannot productively use
        more than n_answer serial steps (measured as a hard accuracy plateau).

        Remasking (``remask=True``, LLaDA-style low-confidence remasking): every
        step re-predicts ALL positions and commits only the round(n*s/T) most
        confident tokens for the next step -- committed tokens can be demoted, so
        all T steps do useful work and the serial budget is not capped by n.
        Inference-only change; training is identical.
        """
        B = prompt.shape[0]
        ans = torch.full((B, n_answer), self.mask_id, device=prompt.device, dtype=torch.long)
        if remask:
            for step in range(1, T + 1):
                logits = self.net(torch.cat([prompt, ans], dim=1))[:, -n_answer:, :]
                probs = logits.softmax(-1)
                pred = probs.argmax(-1)
                cur = torch.where(ans == self.mask_id, pred, ans)
                conf = probs.gather(-1, cur.unsqueeze(-1)).squeeze(-1)
                k = round(n_answer * step / T)
                nxt = torch.full_like(ans, self.mask_id)
                if k > 0:
                    idx = conf.topk(k, dim=1).indices
                    nxt.scatter_(1, idx, cur.gather(1, idx))
                ans = nxt
            # k reaches n_answer at step T, so nothing is left masked
            assert not (ans == self.mask_id).any()
            return ans
        revealed = 0
        for step in range(T):
            quota = round(n_answer * (step + 1) / T) - revealed
            if quota <= 0:
                continue  # no-op step (state unchanged -> same logits next time)
            logits = self.net(torch.cat([prompt, ans], dim=1))[:, -n_answer:, :]
            conf, pred = logits.softmax(-1).max(-1)
            still = ans == self.mask_id
            # reveal the ``quota`` highest-confidence still-masked positions; the
            # schedule guarantees quota <= #still-masked in every row
            idx = conf.masked_fill(~still, -torch.inf).topk(quota, dim=1).indices
            ans.scatter_(1, idx, pred.gather(1, idx))
            revealed += quota
        return ans

    def serial_cost(self, T: int = 16, **_) -> int:
        return T


class TokenUniformDiffusion(nn.Module):
    """UDLM-style uniform-state discrete diffusion (Schiff et al., 2024).

    Forward process interpolates toward the uniform distribution over the
    vocabulary: with the linear schedule ``alpha_t = 1 - t``, each answer token
    is independently replaced by a uniform random token with probability ``t``
    (no mask symbol). Every position is re-sampled at every reverse step, so
    committed tokens stay editable throughout sampling: unlike absorbing-state
    unmasking there is NO ``min(T, n_answer)`` serial cap and no frozen
    commitments -- the self-correction property of UDLM/Duo. Trained with the
    exact continuous-time NELBO (UDLM's closed form, transcribed from the
    official implementation) plus the t=0 reconstruction term; the denoiser is
    time-conditioned, which uniform diffusion requires (the optimal
    x0-prediction depends on the noise level, unlike absorbing-state MDLM).
    """
    paradigm = "token_uniform_diffusion"

    def __init__(self, vocab_size, d=256, n_layers=12, n_heads=8, max_len=128,
                 sampling_eps: float = 1e-3, objective: str = "ce"):
        """``objective``:
          * "ce"    -- x0-prediction cross-entropy over ALL positions of the
            corrupted input at sampled t (UDLM's ``use_simple_ce_loss`` training
            option; one forward per step). Trains fast.
          * "nelbo" -- the exact continuous-time UDLM NELBO (diffusion integrand
            + t=0 reconstruction term; two forwards per step). Principled but
            slow-converging: the integrand's weight vanishes at high noise, the
            uniform-state analog of Duo's reported compute multiplier -- run it
            at extended budget as the exact-objective control.
        """
        super().__init__()
        assert objective in ("ce", "nelbo")
        self.net = _TokenBackbone(vocab_size, d, n_layers, n_heads, max_len,
                                  causal=False, time_cond=True)
        self.vocab_size = vocab_size
        self.sampling_eps = sampling_eps
        self.objective = objective

    def _diffusion_loss(self, x_theta: torch.Tensor, x0: torch.Tensor,
                        zt: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Per-token continuous-time NELBO integrand for the linear schedule
        (alpha' = -1). ``x_theta`` (B,L,V) probs, ``x0``/``zt`` (B,L) indices,
        ``alpha`` (B,) noise level. Zero iff x_theta == one-hot(x0)."""
        N = self.vocab_size
        a = alpha[:, None, None]
        x_bar = N * a * F.one_hot(x0, N).float() + (1 - a)
        x_bar_theta = N * a * x_theta + (1 - a)
        x_bar_zt = x_bar.gather(-1, zt[..., None])
        x_bar_theta_zt = x_bar_theta.gather(-1, zt[..., None])
        coeff = -1.0 / (N * a)
        term1 = N / x_bar_zt - N / x_bar_theta_zt
        term2 = ((x_bar / x_bar_zt)
                 * (x_bar_theta_zt.log() - x_bar_theta.log()
                    + x_bar.log() - x_bar_zt.log())).sum(-1, keepdim=True)
        return (coeff * (term1 - term2)).squeeze(-1)

    def _posterior(self, x: torch.Tensor, zt: torch.Tensor,
                   alpha_s: float, alpha_t: float) -> torch.Tensor:
        """q(z_s | z_t, x) for the uniform kernel; ``x`` may be soft model
        probs (UDLM's denoising parameterization substitutes x_theta)."""
        N = self.vocab_size
        zt1 = F.one_hot(zt, N).float()
        alpha_ts = alpha_t / alpha_s
        x_at_zt = x.gather(-1, zt[..., None])
        num = (alpha_t * N * x_at_zt * zt1
               + (alpha_ts - alpha_t) * zt1
               + (alpha_s - alpha_t) * x
               + (1 - alpha_ts) * (1 - alpha_s) / N)
        den = alpha_t * N * x_at_zt + (1 - alpha_t)
        return num / den

    def fit_step(self, prompt: torch.Tensor, answer: torch.Tensor) -> torch.Tensor:
        B, la = answer.shape
        dev = answer.device
        # antithetic (low-discrepancy) t across the batch, in [eps, 1)
        u = (torch.rand(1, device=dev) + torch.arange(B, device=dev) / B) % 1.0
        t = (1 - self.sampling_eps) * u + self.sampling_eps
        alpha = 1 - t
        move = torch.rand(B, la, device=dev) < t[:, None]
        rnd = torch.randint(0, self.vocab_size, (B, la), device=dev)
        zt = torch.where(move, rnd, answer)
        logits = self.net(torch.cat([prompt, zt], dim=1), t)[:, -la:, :]
        if self.objective == "ce":
            return F.cross_entropy(logits.reshape(-1, self.vocab_size),
                                   answer.reshape(-1))
        x_theta = logits.float().log_softmax(-1).exp()
        diff = self._diffusion_loss(x_theta, answer, zt, alpha)
        # t=0 reconstruction term (UDLM's default; anchors the zero-noise end)
        logits0 = self.net(torch.cat([prompt, answer], dim=1),
                           torch.zeros(B, device=dev))[:, -la:, :]
        recon = F.cross_entropy(logits0.reshape(-1, self.vocab_size),
                                answer.reshape(-1), reduction="none").view(B, la)
        return (diff + recon).mean()

    @torch.no_grad()
    def sample(self, prompt: torch.Tensor, n_answer: int, T: int = 16, **_):
        """Ancestral sampling: start from uniform noise, T forwards on a linear
        time grid; each step re-samples ALL positions from the posterior with
        x_theta substituted (every token remains editable at every step). The
        final forward commits argmax(x_theta) -- the greedy analog of the other
        samplers' argmax decodes."""
        B = prompt.shape[0]
        dev = prompt.device
        zt = torch.randint(0, self.vocab_size, (B, n_answer), device=dev)
        ts = torch.linspace(1.0, self.sampling_eps, T + 1, device=dev)
        for i in range(T):
            t = ts[i].expand(B)
            logits = self.net(torch.cat([prompt, zt], dim=1), t)[:, -n_answer:, :]
            x_theta = logits.float().log_softmax(-1).exp()
            if i == T - 1:
                return x_theta.argmax(-1)
            q = self._posterior(x_theta, zt, alpha_s=1 - ts[i + 1].item(),
                                alpha_t=1 - ts[i].item())
            # Gumbel-max sampling: Categorical() rejects fp32 rows whose sum
            # drifts off 1 at large V; argmax(log q + G) needs no normalization
            u = torch.rand_like(q)
            g = -(-u.clamp_min(1e-20).log()).clamp_min(1e-20).log()
            zt = (q.clamp_min(1e-20).log() + g).argmax(-1)
        return zt

    def serial_cost(self, T: int = 16, **_) -> int:
        return T
