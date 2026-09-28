"""Reasoning accuracy evaluation (torch).

Generate an answer, decode it to tokens, and check exact-match against the task
oracle. Works for concept-space generators (encode prompt -> sample concepts ->
decode) and for token baselines (sample tokens directly).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from dlm.data import make_task
from dlm.concept.ae import ConceptAutoencoder
from dlm.generators.ar import ConceptAR
from dlm.generators.diffusion import ConceptFlowMatching
from dlm.generators.token_baselines import (TokenAR, TokenMaskedDiffusion,
                                            TokenUniformDiffusion)
from dlm.train.loop import get_device
from dlm.checkpoints import resolve_checkpoint_path


def load_generator(path: str, device=None):
    """Rebuild a concept generator + its frozen AE(s) from a checkpoint saved by
    ``train_generator``. Returns (generator, ae, cond_ae, meta); cond_ae is the
    prompt-side AE (same object as ae unless --cond-ae was used)."""
    device = device or get_device()
    path = resolve_checkpoint_path(path)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    ckpt['ae_path'] = resolve_checkpoint_path(ckpt['ae_path'], anchor=path)
    if ckpt.get('cond_ae_path'):
        ckpt['cond_ae_path'] = resolve_checkpoint_path(ckpt['cond_ae_path'], anchor=path)
    kind, kwargs = ckpt["kind"], ckpt["kwargs"]
    gen = (ConceptFlowMatching if kind == "diffusion" else ConceptAR)(**kwargs)
    gen.load_state_dict(ckpt["state_dict"])
    gen.to(device).eval()
    ae = ConceptAutoencoder.load(ckpt["ae_path"], map_location=device).to(device).eval()
    cond_path = ckpt.get("cond_ae_path")
    cond_ae = (ConceptAutoencoder.load(cond_path, map_location=device).to(device).eval()
               if cond_path else ae)
    return gen, ae, cond_ae, ckpt


def load_token_generator(path: str, device=None):
    """Rebuild a token-level baseline (R=1 anchor) from a checkpoint. Returns
    (generator, meta)."""
    device = device or get_device()
    path = resolve_checkpoint_path(path)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cls = {"token_ar": TokenAR, "token_diffusion": TokenMaskedDiffusion,
           "token_uniform": TokenUniformDiffusion}[ckpt["kind"]]
    gen = cls(**ckpt["kwargs"])
    gen.load_state_dict(ckpt["state_dict"])
    gen.to(device).eval()
    return gen, ckpt


def _task_from_meta(meta) -> "object":
    return make_task(meta["task"], **meta.get("task_kwargs", {}))


@torch.no_grad()
def eval_concept_generator(
    gen, ae, task, depths: List[int], n_per_depth: int = 512, T: Optional[int] = None,
    seed: int = 12345, device=None, cond_ae=None,
) -> Dict[int, float]:
    device = device or get_device()
    cond_ae = cond_ae if cond_ae is not None else ae
    rng = np.random.default_rng(seed)
    m_a = max(1, -(-task.answer_len // ae.R))
    out: Dict[int, float] = {}
    for d in depths:
        b = task.sample(n_per_depth, depth=int(d), rng=rng)
        prompt = torch.as_tensor(b.prompt, device=device)
        cond = cond_ae.encode(prompt, sample=False)
        kw = {"T": T} if (T is not None and gen.paradigm == "concept_diffusion") else {}
        z = gen.sample(cond, n_target=m_a, **kw)
        pred = ae.decode(z, out_len=task.answer_len).argmax(-1).cpu().numpy()
        out[int(d)] = float(task.check(b.prompt, pred).mean())
    return out


@torch.no_grad()
def eval_token_generator(
    gen, task, depths: List[int], n_per_depth: int = 512, T: Optional[int] = None,
    seed: int = 12345, device=None,
) -> Dict[int, float]:
    device = device or get_device()
    rng = np.random.default_rng(seed)
    out: Dict[int, float] = {}
    for d in depths:
        b = task.sample(n_per_depth, depth=int(d), rng=rng)
        prompt = torch.as_tensor(b.prompt, device=device)
        kw = {"T": T} if (T is not None and gen.paradigm.endswith("diffusion")) else {}
        pred = gen.sample(prompt, n_answer=task.answer_len, **kw).cpu().numpy()
        out[int(d)] = float(task.check(b.prompt, pred).mean())
    return out
