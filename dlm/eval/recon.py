"""Autoencoder reconstruction quality per compression ratio (torch).

This is the confound-control curve: coarser R reconstructs worse, which would hurt
*both* paradigms. Every accuracy plot is overlaid with this so representation loss
is separable from reasoning loss (plan, confound controls).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from dlm.train.loop import get_device


@torch.no_grad()
def eval_reconstruction(
    ae, task, depths: List[int], n_per_depth: int = 512, seed: int = 999, device=None,
) -> Dict[str, Dict[int, float]]:
    """Return token-level and exact-span reconstruction accuracy for the prompt and
    answer spans, per depth."""
    device = device or get_device()
    rng = np.random.default_rng(seed)
    res = {"prompt_token": {}, "answer_token": {}, "answer_exact": {}, "answer_check": {}}
    for d in depths:
        b = task.sample(n_per_depth, depth=int(d), rng=rng)
        prompt = torch.as_tensor(b.prompt, device=device)
        answer = torch.as_tensor(b.answer, device=device)
        p_logits = ae.decode(ae.encode(prompt), out_len=task.prompt_len)
        a_logits = ae.decode(ae.encode(answer), out_len=task.answer_len)
        p_pred, a_pred = p_logits.argmax(-1), a_logits.argmax(-1)
        res["prompt_token"][int(d)] = float((p_pred == prompt).float().mean())
        res["answer_token"][int(d)] = float((a_pred == answer).float().mean())
        res["answer_exact"][int(d)] = float((a_pred == answer).all(dim=1).float().mean())
        # the ceiling that actually binds: reconstruction graded by the task's own
        # correctness criterion (e.g. final-answer slot only, in trace mode)
        res["answer_check"][int(d)] = float(task.check(b.prompt, a_pred.cpu().numpy()).mean())
    return res
