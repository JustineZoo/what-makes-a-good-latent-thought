"""Train a token-level generator OVER DISCRETE CONCEPT CODES (torch).

The frozen FSQ codec maps prompt/answer token spans to short sequences of
code-tokens (one per sub-slot); the standard token baselines then run over that
code alphabet unchanged: ``code_token_ar`` (softmax AR) and
``code_token_diffusion`` (LLaDA-style masked diffusion over code tokens -- the
discrete-diffusion paradigm operating natively in compressed concept space).

    python -m dlm.train.train_code_token --fsq runs/fsq/fsq_R16_K4.pt \
        --gen diffusion --steps 25000 --out runs/fsq/code_diff.pt
"""
from __future__ import annotations

import argparse

import torch

from dlm.data import make_task
from dlm.concept.discrete import load_codec
from dlm.generators.token_baselines import (TokenAR, TokenMaskedDiffusion,
                                            TokenUniformDiffusion)
from dlm.train.loop import TaskSampler, get_device, train


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--fsq", required=True)
    p.add_argument("--gen", choices=["ar", "diffusion", "uniform"], required=True)
    p.add_argument("--task", default="function_compose")
    p.add_argument("--task-n", type=int, default=5)
    p.add_argument("--task-max-depth", type=int, default=32)
    p.add_argument("--answer-mode", default="trace")
    p.add_argument("--d", type=int, default=384)
    p.add_argument("--layers", type=int, default=16)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--steps", type=int, default=25000)
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

    fsq = load_codec(args.fsq, map_location=device).to(device).eval()
    for pm in fsq.parameters():
        pm.requires_grad_(False)
    task_kwargs = (
        {"max_depth": args.task_max_depth, "answer_mode": args.answer_mode}
        if args.task == "pronto_qa"
        else dict(n=args.task_n, max_depth=args.task_max_depth,
                  answer_mode=args.answer_mode)
    )
    task = make_task(args.task, **task_kwargs)
    K = fsq.cfg.n_slots
    m_p = -(-task.prompt_len // fsq.R) * K
    m_a = -(-task.answer_len // fsq.R) * K
    V = fsq.codebook_size
    max_len = m_p + m_a

    if args.gen == "ar":
        model = TokenAR(V, args.d, args.layers, args.heads, max_len=max_len)
        kwargs = dict(vocab_size=V, d=args.d, n_layers=args.layers, n_heads=args.heads,
                      max_len=max_len)
        paradigm = "code_token_ar"
    elif args.gen == "uniform":
        assert args.objective in ("ce", "nelbo"), "uniform takes --objective ce|nelbo"
        model = TokenUniformDiffusion(V, args.d, args.layers, args.heads,
                                      max_len=max_len, objective=args.objective)
        kwargs = dict(vocab_size=V, d=args.d, n_layers=args.layers, n_heads=args.heads,
                      max_len=max_len, objective=args.objective)
        paradigm = "code_token_uniform_diffusion"
    else:
        model = TokenMaskedDiffusion(V + 1, mask_id=V, d=args.d, n_layers=args.layers,
                                     n_heads=args.heads, max_len=max_len,
                                     objective=args.objective)
        kwargs = dict(vocab_size=V + 1, mask_id=V, d=args.d, n_layers=args.layers,
                      n_heads=args.heads, max_len=max_len, objective=args.objective)
        paradigm = "code_token_diffusion"

    sampler = TaskSampler(task, args.depths, args.batch_size, device, seed=args.seed)

    def step_fn(m, batch):
        with torch.no_grad():
            p_ids = fsq.encode_ids(batch["prompt"])
            a_ids = fsq.encode_ids(batch["answer"])
        return m.fit_step(p_ids, a_ids)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_code] {paradigm} codebook={V} seq={m_p}+{m_a} params={n_params/1e6:.1f}M")
    train(model, step_fn, sampler, args.steps, lr=args.lr, device=device)
    torch.save({"state_dict": model.state_dict(), "kind": args.gen, "kwargs": kwargs,
                "paradigm": paradigm, "fsq_path": args.fsq, "R": fsq.R,
                "m_answer_codes": m_a, "n_params": n_params,
                "task": args.task, "task_kwargs": task_kwargs}, args.out)
    print(f"[train_code] saved -> {args.out}")


if __name__ == "__main__":
    main()
