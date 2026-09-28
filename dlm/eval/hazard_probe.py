"""Chain-survival probe for concept-AR (torch, eval-only) -- the compounding law.

Finding: at R=1/D=32 concept-AR scores 0.86 ~ 0.999^160 while R in {4,8} score
1.00, and a param-matched deeper AR recovers R=1 to 0.99. Claim: AR's depth cost
is per-emission error compounding, so compression helps by SHORTENING the chain.
This probe measures the law directly: free-run each generator, decode, and record
P(states 1..k all correct) vs k. Under i.i.d. per-state hazard the curve is
exponential and its log-slope is the per-state error rate; the R-dependence of
that slope (vs chain length) is the mechanism.

    python -m dlm.eval.hazard_probe --gen runs/m2/gen_R1_ar.pt runs/m2/gen_R4_ar.pt \
        runs/m2.1/gen_R8_ar.pt runs/m2.1/gen_R1_ar_parammatch.pt --depth 32 --out hazard.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from dlm.eval.accuracy import load_generator, _task_from_meta
from dlm.train.loop import get_device


@torch.no_grad()
def probe_gen(path: str, depth: int, n: int, device, seed: int = 4242):
    gen, ae, cond_ae, ckpt = load_generator(path, device)
    task = _task_from_meta(ckpt)
    rng = np.random.default_rng(seed)
    b = task.sample(n, depth=depth, rng=rng)
    prompt = torch.as_tensor(b.prompt, device=device)
    cond = cond_ae.encode(prompt, sample=False)
    m_a = max(1, -(-task.answer_len // ae.R))
    z = gen.sample(cond, n_target=m_a)
    pred = ae.decode(z, out_len=task.answer_len).argmax(-1).cpu().numpy()
    gold = task.oracle(b.prompt)
    nn = task.n
    state_ok = (pred[:, : depth * nn].reshape(n, depth, nn)
                == gold[:, : depth * nn].reshape(n, depth, nn)).all(axis=2)
    survival = np.cumprod(state_ok, axis=1).mean(axis=0)  # P(prefix 1..k correct)
    # per-state log-hazard from the exponential fit over the surviving region
    k = np.arange(1, depth + 1)
    mask = survival > 0.02
    slope = (np.polyfit(k[mask], np.log(survival[mask]), 1)[0] if mask.sum() >= 2 else float("nan"))
    return {
        "gen": path, "R": ckpt["R"], "n_params": ckpt.get("n_params", 0),
        "m_answer": m_a, "depth": depth,
        "survival": [round(float(s), 4) for s in survival],
        "per_state_acc": round(float(np.exp(slope)), 5) if np.isfinite(slope) else None,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--gen", nargs="+", required=True)
    p.add_argument("--depth", type=int, default=32)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--out", default="hazard.json")
    args = p.parse_args(argv)
    device = get_device()
    results = [probe_gen(g, args.depth, args.n, device) for g in args.gen]
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    for r in results:
        surv = r["survival"]
        print(f"R={r['R']:<3d} m_a={r['m_answer']:<4d} params={r['n_params']/1e6:.0f}M  "
              f"P(all {r['depth']}) = {surv[-1]:.3f}   per-state acc ~= {r['per_state_acc']}")
    print(f"[hazard_probe] wrote {args.out}")


if __name__ == "__main__":
    main()
