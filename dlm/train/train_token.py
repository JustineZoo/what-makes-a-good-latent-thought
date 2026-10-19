"""Train an R=1 token-level baseline (the anchors of Milestone 0).

    python -m dlm.train.train_token --gen token_ar        --task function_compose --out runs/tok_ar.pt
    python -m dlm.train.train_token --gen token_diffusion  --task function_compose --out runs/tok_diff.pt
"""
from __future__ import annotations

import argparse

import torch

from dlm.data import make_task
from dlm.generators.token_baselines import (TokenAR, TokenMaskedDiffusion,
                                            TokenUniformDiffusion)
from dlm.train.loop import TaskSampler, get_device, train


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--gen", choices=["token_ar", "token_diffusion", "token_uniform"],
                   required=True)
    p.add_argument("--task", default="function_compose")
    p.add_argument("--task-n", type=int, default=5)
    p.add_argument("--task-max-depth", type=int, default=32)
    p.add_argument("--answer-mode", default="query")
    p.add_argument("--group", default="s5", choices=["s5", "affine", "walker", "walkerids"],
                   help="s5/affine select the function_compose group; walker selects the\n                        deterministic graph_reach variant")
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    p.add_argument("--out", required=True)
    p.add_argument("--objective", default="uniform", choices=["uniform", "elbo", "ce", "nelbo"],
                   help="masked-diffusion objective (uniform masked CE | MDLM/LLaDA elbo) "
                        "or uniform-state objective (ce = UDLM simple CE | nelbo = exact "
                        "UDLM continuous-time NELBO); ignored for AR")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    torch.manual_seed(args.seed)
    device = get_device()
    task_kwargs = {}
    if args.task == "function_compose":
        task_kwargs = dict(n=args.task_n, max_depth=args.task_max_depth, answer_mode=args.answer_mode, group=args.group)
    elif args.task == "graph_reach":
        task_kwargs = dict(width=args.task_n, max_depth=args.task_max_depth,
                           answer_mode=args.answer_mode, p_edge=0.6, out_degree=2,
                           walker=(args.group in ("walker", "walkerids")),
                           trace_ids=(args.group == "walkerids"))
    elif args.task == "pronto_qa":
        task_kwargs = dict(max_depth=args.task_max_depth, answer_mode=args.answer_mode)
    task = make_task(args.task, **task_kwargs)

    V = len(task.vocab)
    if args.gen == "token_ar":
        model = TokenAR(V, args.d, args.layers, args.heads, max_len=task.max_len)
        kwargs = dict(vocab_size=V, d=args.d, n_layers=args.layers, n_heads=args.heads, max_len=task.max_len)
    elif args.gen == "token_uniform":
        assert args.objective in ("ce", "nelbo"), "token_uniform takes --objective ce|nelbo"
        model = TokenUniformDiffusion(V, args.d, args.layers, args.heads,
                                      max_len=task.max_len, objective=args.objective)
        kwargs = dict(vocab_size=V, d=args.d, n_layers=args.layers, n_heads=args.heads,
                      max_len=task.max_len, objective=args.objective)
    else:
        model = TokenMaskedDiffusion(V + 1, mask_id=V, d=args.d, n_layers=args.layers,
                                     n_heads=args.heads, max_len=task.max_len,
                                     objective=args.objective)
        kwargs = dict(vocab_size=V + 1, mask_id=V, d=args.d, n_layers=args.layers,
                      n_heads=args.heads, max_len=task.max_len, objective=args.objective)

    sampler = TaskSampler(task, args.depths, args.batch_size, device, seed=args.seed)

    def step_fn(m, batch):
        return m.fit_step(batch["prompt"], batch["answer"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_token] {args.gen} task={task.name} vocab={V} params={n_params/1e6:.1f}M")
    train(model, step_fn, sampler, args.steps, lr=args.lr, device=device)
    torch.save({"state_dict": model.state_dict(), "kind": args.gen, "kwargs": kwargs,
                "paradigm": model.paradigm, "n_params": n_params,
                "task": args.task, "task_kwargs": task_kwargs}, args.out)
    print(f"[train_token] saved -> {args.out}")


if __name__ == "__main__":
    main()
