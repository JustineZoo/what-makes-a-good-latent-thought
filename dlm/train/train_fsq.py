"""Train a discrete (FSQ) factorized concept codec, then freeze it.

Mirrors ``train_ae`` (both spans, length-biased depths, chain-end recon gate);
the rate is fixed at ``m*K*log2(levels^code_dims)`` bits by construction, so
there is no beta/KL knob.

    python -m dlm.train.train_fsq --task function_compose --task-max-depth 32 \
        --answer-mode trace --R 16 --slots 4 --gate-recon 0.95 --out runs/fsq/fsq.pt
"""
from __future__ import annotations

import argparse

import torch

from dlm.data import make_task
from dlm.concept.fsq import FSQAutoencoder, FSQConfig
from dlm.train.loop import TaskSampler, get_device, train


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="function_compose")
    p.add_argument("--task-n", type=int, default=5)
    p.add_argument("--task-max-depth", type=int, default=32)
    p.add_argument("--answer-mode", default="trace")
    p.add_argument("--R", type=int, default=16)
    p.add_argument("--slots", type=int, default=4)
    p.add_argument("--code-dims", type=int, default=3)
    p.add_argument("--levels", type=int, default=8)
    p.add_argument("--d-concept", type=int, default=384)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--steps", type=int, default=45000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    p.add_argument("--depth-power", type=float, default=1.0)
    p.add_argument("--gate-recon", type=float, default=0.95)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    torch.manual_seed(args.seed)
    device = get_device()
    task = make_task(args.task, n=args.task_n, max_depth=args.task_max_depth,
                     answer_mode=args.answer_mode)
    cfg = FSQConfig(vocab_size=len(task.vocab), d_concept=args.d_concept,
                    compression_ratio=args.R, n_slots=args.slots,
                    code_dims=args.code_dims, levels=args.levels,
                    max_tokens=task.max_len, enc_layers=args.layers, dec_layers=args.layers)
    model = FSQAutoencoder(cfg)
    sampler = TaskSampler(task, args.depths, args.batch_size, device, seed=args.seed,
                          uniform=True, depth_power=args.depth_power)

    def step_fn(m, batch):
        return m.loss(batch["prompt"])["loss"] + m.loss(batch["answer"])["loss"]

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_fsq] R={args.R} K={args.slots} codebook={model.codebook_size} "
          f"params={n_params/1e6:.1f}M")
    train(model, step_fn, sampler, args.steps, lr=args.lr, device=device)

    model.eval()
    if args.gate_recon > 0:
        from dlm.eval.recon import eval_reconstruction
        rec = eval_reconstruction(model, task, args.depths, n_per_depth=256, device=device)
        line = "  ".join(f"D={d}:{rec['answer_check'][d]:.3f}" for d in args.depths)
        print(f"[train_fsq] chain-end recon ceiling  {line}")
        worst = min(rec["answer_check"].values())
        if worst < args.gate_recon:
            print(f"[train_fsq] GATE FAILED: {worst:.3f} < {args.gate_recon} -- not saving")
            raise SystemExit(3)
    model.save(args.out)
    print(f"[train_fsq] saved -> {args.out}")


if __name__ == "__main__":
    main()
