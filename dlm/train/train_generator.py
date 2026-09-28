"""Train an AR or diffusion generator over a frozen concept autoencoder.

The AE is loaded, frozen, and used (no grad) to encode prompt -> context concepts
and answer -> target concepts. The generator learns p(target | context). AR and
diffusion share this exact setup, so the paradigm is the only variable.

Example:
    python -m dlm.train.train_generator --ae runs/ae_R8.pt --gen diffusion \
        --task function_compose --steps 5000 --out runs/gen_R8_diff.pt
"""
from __future__ import annotations

import argparse
import math

import torch

from dlm.data import make_task
from dlm.concept.ae import ConceptAutoencoder
from dlm.generators.ar import ConceptAR
from dlm.generators.diffusion import ConceptFlowMatching
from dlm.train.loop import TaskSampler, get_device, train


def _concepts(n_tokens, R):
    return max(1, math.ceil(n_tokens / R))


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--ae", required=True)
    p.add_argument("--cond-ae", default=None,
                   help="optional separate frozen AE for the prompt/cond side (e.g. an R=1 AE), "
                        "isolating trace-compression from prompt-compression; default: same as --ae")
    p.add_argument("--gen", choices=["ar", "diffusion"], required=True)
    p.add_argument("--task", default="function_compose")
    p.add_argument("--task-n", type=int, default=5)
    p.add_argument("--task-max-depth", type=int, default=32)
    p.add_argument("--answer-mode", default="query")
    p.add_argument("--group", default="s5", choices=["s5", "affine", "walker", "walkerids"],
                   help="s5/affine select the function_compose group; walker selects the\n                        deterministic graph_reach variant")
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--model-dim", type=int, default=0,
                   help=">0: backbone width decoupled from d_concept (generator-scale "
                        "study over a fixed frozen codec)")
    p.add_argument("--block", default="adaln", choices=["adaln", "plain"],
                   help="diffusion only: adaLN-Zero DiT blocks (default) or the AR "
                        "generator's plain pre-norm blocks with an in-context time token "
                        "(the architecture control)")
    p.add_argument("--head", default="mar", choices=["mar", "gmm"])
    p.add_argument("--head-inner-steps", type=int, default=4)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=0,
                   help="linear LR warmup steps (needed for wide/deep backbones)")
    p.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    torch.manual_seed(args.seed)
    device = get_device()

    ae = ConceptAutoencoder.load(args.ae, map_location=device).to(device).eval()
    for pm in ae.parameters():
        pm.requires_grad_(False)
    if args.cond_ae:
        cond_ae = ConceptAutoencoder.load(args.cond_ae, map_location=device).to(device).eval()
        for pm in cond_ae.parameters():
            pm.requires_grad_(False)
        assert cond_ae.cfg.d_concept == ae.cfg.d_concept, "cond AE must share d_concept"
    else:
        cond_ae = ae
    R, d = ae.R, ae.cfg.d_concept

    task_kwargs = {}
    if args.task == "function_compose":
        task_kwargs = dict(n=args.task_n, max_depth=args.task_max_depth, answer_mode=args.answer_mode, group=args.group)
    elif args.task == "graph_reach":
        task_kwargs = dict(width=args.task_n, max_depth=args.task_max_depth,
                           answer_mode=args.answer_mode, p_edge=0.6, out_degree=2,
                           walker=(args.group in ("walker", "walkerids")),
                           trace_ids=(args.group == "walkerids"))
    task = make_task(args.task, **task_kwargs)

    m_p = _concepts(task.prompt_len, cond_ae.R)
    m_a = _concepts(task.answer_len, R)
    max_concepts = m_p + m_a + 2

    mdim = args.model_dim or None
    if args.gen == "diffusion":
        gen = ConceptFlowMatching(d, args.layers, args.heads, max_concepts, model_dim=mdim,
                                  block=args.block)
        gen_kwargs = dict(d_concept=d, n_layers=args.layers, n_heads=args.heads,
                          max_concepts=max_concepts, model_dim=mdim, block=args.block)
    else:
        gen = ConceptAR(d, args.layers, args.heads, max_concepts, args.head,
                        args.head_inner_steps, model_dim=mdim)
        gen_kwargs = dict(d_concept=d, n_layers=args.layers, n_heads=args.heads,
                          max_concepts=max_concepts, head=args.head,
                          head_inner_steps=args.head_inner_steps, model_dim=mdim)

    sampler = TaskSampler(task, args.depths, args.batch_size, device, seed=args.seed)

    def step_fn(g, batch):
        with torch.no_grad():
            cond = cond_ae.encode(batch["prompt"], sample=False)
            target = ae.encode(batch["answer"], sample=False)
        return g.fit_step(target, cond)

    n_params = sum(p.numel() for p in gen.parameters())
    print(f"[train_gen] {args.gen} over AE(R={R}) task={task.name}  m_p={m_p} m_a={m_a}  "
          f"params={n_params/1e6:.1f}M")
    train(gen, step_fn, sampler, args.steps, lr=args.lr, warmup_steps=args.warmup, device=device)

    torch.save(
        {"state_dict": gen.state_dict(), "kind": args.gen, "kwargs": gen_kwargs,
         "paradigm": gen.paradigm, "ae_path": args.ae, "cond_ae_path": args.cond_ae,
         "R": R, "m_answer": m_a, "n_params": n_params,
         "task": args.task, "task_kwargs": task_kwargs},
        args.out,
    )
    print(f"[train_gen] saved -> {args.out}")


if __name__ == "__main__":
    main()
