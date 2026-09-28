"""Evaluate code-token generators (discrete concepts) against the task oracle.

Generate code tokens, dequantize + decode through the frozen FSQ codec, grade
with ``task.check``. Emits rows in the same schema as ``ablation`` (accuracy,
codec ceiling, both serial accountings) so ``summary``/``seed_stats`` work.

    python -m dlm.eval.code_eval --runs runs/fsq --depths 2 16 32 --T 1 4 16 64 \
        --out runs/fsq/results.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

from dlm.data import make_task
from dlm.concept.discrete import load_codec
from dlm.generators.token_baselines import (TokenAR, TokenMaskedDiffusion,
                                            TokenUniformDiffusion)
from dlm.train.loop import get_device
from dlm.checkpoints import resolve_checkpoint_path


@torch.no_grad()
def _eval(gen, fsq, task, depths, n, T, device, seed=12345, remask=False):
    rng = np.random.default_rng(seed)
    K = fsq.cfg.n_slots
    m_a = -(-task.answer_len // fsq.R) * K
    acc, ceil = {}, {}
    for d in depths:
        b = task.sample(n, depth=int(d), rng=rng)
        prompt = torch.as_tensor(b.prompt, device=device)
        answer = torch.as_tensor(b.answer, device=device)
        p_ids = fsq.encode_ids(prompt)
        kw = {"T": T, "remask": remask} if T is not None else {}
        ids = gen.sample(p_ids, n_answer=m_a, **kw)
        pred = fsq.decode_ids(ids, out_len=task.answer_len).argmax(-1).cpu().numpy()
        acc[int(d)] = float(task.check(b.prompt, pred).mean())
        rt = fsq.decode_ids(fsq.encode_ids(answer), out_len=task.answer_len).argmax(-1).cpu().numpy()
        ceil[int(d)] = float(task.check(b.prompt, rt).mean())
    return acc, ceil, m_a


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--depths", type=int, nargs="+", default=[2, 16, 32])
    p.add_argument("--T", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--sampler", choices=["absorbing", "remask"], default="absorbing",
                   help="remask = LLaDA-style low-confidence remasking (lifts the "
                        "n_codes serial-step cap of absorbing unmasking)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    device = get_device()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.runs, "*.pt"))):
        ckpt = torch.load(path, map_location=device)
        paradigm = ckpt.get("paradigm", "")
        if not paradigm.startswith("code_token"):
            print(f"[code_eval] skipping {os.path.basename(path)} ({paradigm!r})")
            continue
        fsq = load_codec(resolve_checkpoint_path(ckpt["fsq_path"], anchor=path), map_location=device).to(device).eval()
        cls = {"ar": TokenAR, "diffusion": TokenMaskedDiffusion,
               "uniform": TokenUniformDiffusion}[ckpt["kind"]]
        gen = cls(**ckpt["kwargs"])
        gen.load_state_dict(ckpt["state_dict"])
        gen.to(device).eval()
        task = make_task(ckpt["task"], **ckpt["task_kwargs"])
        Ts = args.T if paradigm.endswith("diffusion") else [None]
        remask = args.sampler == "remask"
        for T in Ts:
            acc, ceil, m_a = _eval(gen, fsq, task, args.depths, args.n, T, device,
                                   remask=remask)
            for d in args.depths:
                sc = m_a if T is None else T
                rows.append(dict(ckpt=os.path.basename(path), paradigm=paradigm,
                                 R=ckpt["R"], T=(T or 0), depth=d, accuracy=acc[d],
                                 answer_recon=ceil[d], answer_exact=ceil[d],
                                 sampler=("uniform" if "uniform" in paradigm else
                                          args.sampler if paradigm.endswith("diffusion") else "ar"),
                                 serial_compute=sc, serial_backbone=sc,
                                 n_params=ckpt.get("n_params", 0)))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[code_eval] wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
