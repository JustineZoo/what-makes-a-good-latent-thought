"""Generic training utilities (torch): device, numpy->torch batch, sampler, loop."""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np
import torch

from dlm.data.base import Task


def get_device(prefer: str = "cuda") -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pad_to(a: np.ndarray, length: int, pad_id: int) -> np.ndarray:
    if a.shape[1] >= length:
        return a[:, :length]
    out = np.full((a.shape[0], length), pad_id, dtype=np.int64)
    out[:, : a.shape[1]] = a
    return out


class TaskSampler:
    """Yields batches at depths from ``depths``: round-robin (default, matched to
    the eval grid) or uniform over [min(depths), max(depths)] (``uniform=True``;
    used for AE training, where round-robin leaves late trace positions covered
    only by the deepest depth -- a coverage cliff that starves their recon)."""

    def __init__(self, task: Task, depths: Sequence[int], batch_size: int, device,
                 seed: int = 0, uniform: bool = False, depth_power: float = 0.0):
        self.task = task
        self.depths = list(depths)
        self.batch_size = batch_size
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.uniform = uniform
        # depth_power > 0 (with uniform): P(d) ~ d^power over [min, max]. power=1 is
        # length-biased sampling -- each trace POSITION then receives roughly equal
        # informative-token mass. Late positions are only ever exercised by the
        # deepest programs, so uniform-depth still starves them (P(depth>=31)~6.5%
        # at max_depth 32); this is the knob that fixes that tail.
        self.depth_power = depth_power
        self._i = 0

    def next(self) -> Dict[str, torch.Tensor]:
        if self.uniform:
            lo, hi = min(self.depths), max(self.depths)
            if self.depth_power > 0:
                ds = np.arange(lo, hi + 1)
                w = ds.astype(float) ** self.depth_power
                d = int(self.rng.choice(ds, p=w / w.sum()))
            else:
                d = int(self.rng.integers(lo, hi + 1))
        else:
            d = self.depths[self._i % len(self.depths)]
        self._i += 1
        b = self.task.sample(self.batch_size, depth=int(d), rng=self.rng)
        return {
            "prompt": torch.as_tensor(b.prompt, device=self.device),
            "answer": torch.as_tensor(b.answer, device=self.device),
            "depth": int(d),
        }


def train(
    model: torch.nn.Module,
    step_fn: Callable[[torch.nn.Module, Dict[str, torch.Tensor]], torch.Tensor],
    sampler: TaskSampler,
    n_steps: int,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    log_every: int = 100,
    warmup_steps: int = 0,
    device=None,
) -> List[float]:
    device = device or get_device()
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    hist: List[float] = []
    for step in range(1, n_steps + 1):
        if warmup_steps and step <= warmup_steps:  # linear warmup, then flat
            for g in opt.param_groups:
                g["lr"] = lr * step / warmup_steps
        batch = sampler.next()
        loss = step_fn(model, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        hist.append(float(loss.detach()))
        if step % log_every == 0 or step == 1:
            recent = np.mean(hist[-log_every:])
            print(f"  step {step:6d}/{n_steps}  loss {recent:.4f}")
    return hist
