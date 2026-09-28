"""Teacher-forced per-emission probe for concept-AR generators (torch, eval-only).

Separates single-shot per-emission fidelity from free-running compounding: with
GOLD previous concepts in the causal prefix, sample each next concept from the
head, swap it into the gold concept sequence, decode, and grade the tokens (and
any fully-contained states) inside that concept's span. High TF accuracy with
low free-running per-state accuracy = commitment-compounding dominates; low TF
accuracy = the single emission itself (retrieval and/or composition) binds.

    python -m dlm.eval.tf_probe --gen runs/walker2/gen_R3_ar_K1.pt --depth 12 \
        --n 256 --out tf.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from dlm.eval.accuracy import load_generator, _task_from_meta
from dlm.train.loop import get_device


@torch.no_grad()
def probe(path: str, depth: int, n: int, device, seed: int = 31415):
    gen, ae, cond_ae, ckpt = load_generator(path, device)
    assert gen.paradigm == "concept_ar", "TF probe applies to concept-AR generators"
    task = _task_from_meta(ckpt)
    rng = np.random.default_rng(seed)
    b = task.sample(n, depth=depth, rng=rng)
    prompt = torch.as_tensor(b.prompt, device=device)
    answer = torch.as_tensor(b.answer, device=device)
    cond = cond_ae.encode(prompt, sample=False)
    target = ae.encode(answer, sample=False)  # gold concepts, normalized space
    B, mt, d = target.shape
    R = ae.R
    state_w = getattr(task, "n", 1)

    # teacher-forced hidden states aligned to each target position (fit_step's alignment)
    bos = gen.bos.expand(B, 1, d)
    seq = torch.cat([cond, bos, target[:, :-1]], dim=1)
    h = gen._hidden(seq)[:, -mt:, :]

    per_span = []
    for k in range(mt):
        z_k = gen.head.sample(h[:, k, :], steps=gen.head_inner_steps)
        z_swap = target.clone()
        z_swap[:, k, :] = z_k
        pred = ae.decode(z_swap, out_len=task.answer_len).argmax(-1)
        lo, hi = k * R, min((k + 1) * R, task.answer_len)
        tok_acc = (pred[:, lo:hi] == answer[:, lo:hi]).float().mean().item()
        # fully-contained states within the span
        s_lo = -(-lo // state_w)                      # first state starting at/after lo
        s_hi = hi // state_w                          # last state ending at/before hi
        st_accs = []
        for s in range(s_lo, s_hi):
            a, bnd = s * state_w, (s + 1) * state_w
            st_accs.append((pred[:, a:bnd] == answer[:, a:bnd]).all(dim=1).float().mean().item())
        per_span.append({"concept": k, "token_acc": round(tok_acc, 4),
                         "state_accs": [round(x, 4) for x in st_accs]})
    all_states = [x for p in per_span for x in p["state_accs"]]
    return {"gen": path, "R": R, "depth": depth, "m_answer": mt,
            "tf_token_acc_mean": round(float(np.mean([p["token_acc"] for p in per_span])), 4),
            "tf_state_acc_mean": round(float(np.mean(all_states)), 4) if all_states else None,
            "per_span": per_span}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--gen", nargs="+", required=True)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--out", default="tf_probe.json")
    args = p.parse_args(argv)
    device = get_device()
    results = [probe(g, args.depth, args.n, device) for g in args.gen]
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    for r in results:
        line = " ".join(f"{p['state_accs']}" for p in r["per_span"][:4])
        print(f"{r['gen'].split('/')[-1]:28s} TF state-acc {r['tf_state_acc_mean']}  "
              f"TF token-acc {r['tf_token_acc_mean']}  spans[:4] {line}")
    print(f"[tf_probe] wrote {args.out}")


if __name__ == "__main__":
    main()
