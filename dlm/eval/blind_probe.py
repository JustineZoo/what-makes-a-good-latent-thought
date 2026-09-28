"""Context-blinding intervention for the walker inversion (eval-only).

The scope section attributes concept diffusion's walker advantage to iterative
refinement "re-reading the context at every step". This probe tests that claim
directly by controlling WHICH denoising steps may read the true context:

  * late-blind  (direction "late"):  steps 0..k-1 see the true relational
    context; steps k..T-1 see the context of a DIFFERENT problem from the same
    batch (matched marginal distribution, wrong information).
  * early-blind (direction "early"): the mirror image -- the first k steps are
    mis-conditioned, the remaining T-k steps are honest. Measures whether later
    honest reads can revise a poisoned start.

Swapping (a batch roll) rather than zeroing keeps the conditioning sequence
in-distribution, so a drop measures information use, not input-shape shock.
``k = T`` with direction "late" reproduces the honest sampler exactly, and is
how the honest T-sweep baselines are produced here; ``k = 0`` with "late" is
the wrong-context floor that validates the manipulation. Every condition at one
(checkpoint, depth) shares the same problem batch and the same initial noise,
so all comparisons are paired.

    python -m dlm.eval.blind_probe --gen runs/wseeds/gen_R3_diffusion_s0.pt \
        --depths 8 12 --T 256 --k 0 1 2 4 8 16 64 --n 512 --out runs/wseeds/blind.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from dlm.eval.accuracy import load_generator, _task_from_meta
from dlm.train.loop import get_device


@torch.no_grad()
def sample_blinded(gen, cond_true, cond_alt, n_target: int, T: int, k: int,
                   direction: str) -> torch.Tensor:
    """The ConceptFlowMatching sampler with a per-step conditioning switch."""
    B = cond_true.shape[0]
    z = torch.randn(B, n_target, gen.d, device=cond_true.device)
    x0_hat = None
    for i in range(T):
        t = 1.0 - i / T
        tv = torch.full((B,), t, device=cond_true.device)
        honest = (i >= k) if direction == "early" else (i < k)  # "late"/"honest": first k steps
        c = cond_true if honest else cond_alt
        x0_hat = gen.net(z, tv, cond=c, self_cond=x0_hat)
        z = z - (1.0 / T) * (z - x0_hat) / t
    return z


@torch.no_grad()
def probe_gen(path: str, depths, T_max: int, k_list, n: int, device, seed: int = 12345):
    gen, ae, cond_ae, ckpt = load_generator(path, device)
    assert gen.paradigm == "concept_diffusion", f"{path}: not a diffusion generator"
    task = _task_from_meta(ckpt)
    m_a = max(1, -(-task.answer_len // ae.R))
    rows = []
    for depth in depths:
        rng = np.random.default_rng(seed + depth)
        b = task.sample(n, depth=int(depth), rng=rng)
        prompt = torch.as_tensor(b.prompt, device=device)
        cond = cond_ae.encode(prompt, sample=False)
        cond_alt = torch.roll(cond, shifts=1, dims=0)  # another problem, same depth
        gold = task.oracle(b.prompt)
        nn = task.n

        def run(T, k, direction):
            torch.manual_seed(seed)  # identical initial noise across conditions
            z = sample_blinded(gen, cond, cond_alt, m_a, T, k, direction)
            pred = ae.decode(z, out_len=task.answer_len).argmax(-1).cpu().numpy()
            state_ok = (pred[:, : depth * nn].reshape(n, depth, nn)
                        == gold[:, : depth * nn].reshape(n, depth, nn)).all(axis=2)
            rows.append({
                "gen": path, "R": ckpt["R"], "depth": int(depth),
                "direction": direction, "T": T, "k": k,
                "accuracy": round(float(task.check(b.prompt, pred).mean()), 5),
                "per_state": [round(float(a), 4) for a in state_ok.mean(axis=0)],
            })

        for T in (1, 4, 16, 64, T_max):        # honest baselines (k=T == honest)
            run(T, T, "honest")
        for k in k_list:                        # context readable only first k steps
            run(T_max, k, "late")
        for k in k_list:                        # context withheld for first k steps
            if 0 < k < T_max:
                run(T_max, k, "early")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--gen", nargs="+", required=True)
    p.add_argument("--depths", type=int, nargs="+", default=[8, 12])
    p.add_argument("--T", type=int, default=256)
    p.add_argument("--k", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 64])
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--out", default="blind.json")
    args = p.parse_args(argv)
    device = get_device()
    rows = []
    for g in args.gen:
        rows += probe_gen(g, args.depths, args.T, args.k, args.n, device)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    for r in rows:
        print(f"{r['gen']}  D={r['depth']:<3d} {r['direction']:<7s} "
              f"T={r['T']:<4d} k={r['k']:<4d} acc={r['accuracy']:.3f}")
    print(f"[blind_probe] wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
