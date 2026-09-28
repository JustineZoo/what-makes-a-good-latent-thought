"""Train a concept autoencoder at a chosen compression ratio, then freeze it.

Reconstructs the exact spans used later (prompt span + answer span) so encode/decode
is in-distribution at generation time. Produces one frozen AE per (R, beta), which
the AR and diffusion generators then share.

Example:
    python -m dlm.train.train_ae --task function_compose --R 8 --beta 1.0 \
        --steps 3000 --out runs/ae_R8.pt
"""
from __future__ import annotations

import argparse

import torch

from dlm.data import make_task
from dlm.concept.ae import AEConfig, ConceptAutoencoder
from dlm.train.loop import TaskSampler, get_device, train


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="function_compose")
    p.add_argument("--task-n", type=int, default=5)
    p.add_argument("--task-max-depth", type=int, default=32)
    p.add_argument("--answer-mode", default="query")
    p.add_argument("--group", default="s5", choices=["s5", "affine", "walker", "walkerids"],
                   help="s5/affine select the function_compose group; walker selects the\n                        deterministic graph_reach variant")
    p.add_argument("--R", type=int, default=8)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--free-bits", type=float, default=0.5)
    p.add_argument("--d-concept", type=int, default=256)
    p.add_argument("--slots", type=int, default=0,
                   help=">0: factorized concept code with K sub-slot codes (R and "
                        "d_concept must be divisible by K); tests the density-wall "
                        "compositionality hypothesis")
    p.add_argument("--blockwise", action="store_true",
                   help="strict per-concept information boundary: block-diagonal "
                        "self-attention in the encoder, per-block cross/self-attention "
                        "in the decoder (concept k <-> its own R tokens only)")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dec-layers", type=int, default=0,
                   help=">0: decoder depth decoupled from --layers (the decoder is "
                        "the recon bottleneck at low concept counts)")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    p.add_argument("--uniform-depths", action="store_true",
                   help="sample depth ~ U[min(depths), max(depths)] instead of round-robin "
                        "(avoids the late-position coverage cliff on long trace spans)")
    p.add_argument("--depth-power", type=float, default=0.0,
                   help="with --uniform-depths: P(d) ~ d^power; power=1 (length-biased) "
                        "gives each trace position ~equal informative-token coverage")
    p.add_argument("--gate-recon", type=float, default=0.0,
                   help="fail (exit 3) unless chain-end recon >= this at every eval depth; "
                        "an AE below the gate would silently cap every generator above it")
    p.add_argument("--out", required=True)
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
    task = make_task(args.task, **task_kwargs)

    cfg = AEConfig(
        vocab_size=len(task.vocab), d_concept=args.d_concept, compression_ratio=args.R,
        max_tokens=task.max_len, beta=args.beta, free_bits=args.free_bits,
        enc_layers=args.layers, dec_layers=(args.dec_layers or args.layers),
        n_slots=args.slots, blockwise=args.blockwise,
    )
    model = ConceptAutoencoder(cfg)
    sampler = TaskSampler(task, args.depths, args.batch_size, device, seed=args.seed,
                          uniform=args.uniform_depths, depth_power=args.depth_power)

    def step_fn(m, batch):
        out_p = m.loss(batch["prompt"])
        out_a = m.loss(batch["answer"])
        return out_p["loss"] + out_a["loss"]

    print(f"[train_ae] task={task.name} R={args.R} beta={args.beta} vocab={len(task.vocab)}")
    train(model, step_fn, sampler, args.steps, lr=args.lr, device=device)

    # calibrate the latent space (mean/std over posterior means of both spans) so
    # generators diffuse over a standardized concept space, then readout + save
    model.eval()
    with torch.no_grad():
        spans = []
        for _ in range(8):
            b = sampler.next()
            spans += [b["prompt"], b["answer"]]
        model.calibrate(spans)
        b = sampler.next()

        def _det_acc(x):  # deterministic (mu-path) recon -- the ceiling that binds
            pred = model.decode(model.encode(x), out_len=x.shape[1]).argmax(-1)
            return (pred == x).float().mean().item()

        acc_p, acc_a = _det_acc(b["prompt"]), _det_acc(b["answer"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_ae] recon token-acc  prompt={acc_p:.3f}  answer={acc_a:.3f}  params={n_params/1e6:.1f}M")
    print(f"[train_ae] latent std: mean={model.latent_std.mean():.3f} "
          f"min={model.latent_std.min():.3f} max={model.latent_std.max():.3f}")

    if args.gate_recon > 0:
        from dlm.eval.recon import eval_reconstruction
        rec = eval_reconstruction(model, task, args.depths, n_per_depth=256, device=device)
        line = "  ".join(f"D={d}:{rec['answer_check'][d]:.3f}" for d in args.depths)
        print(f"[train_ae] chain-end recon ceiling  {line}")
        worst = min(rec["answer_check"].values())
        if worst < args.gate_recon:
            print(f"[train_ae] GATE FAILED: ceiling {worst:.3f} < {args.gate_recon} "
                  f"-- not saving; raise steps / lower beta / check depth coverage")
            raise SystemExit(3)

    model.save(args.out)
    print(f"[train_ae] saved -> {args.out}")


if __name__ == "__main__":
    main()
