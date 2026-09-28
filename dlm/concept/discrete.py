"""Discrete concept codes by post-hoc k-means over a trained factorized codec.

Training a discrete codec from scratch (FSQ) collapsed to ~2 codes under three
standard fixes (see fsq.py history) -- but a validated CONTINUOUS factorized
codec already exists, and its sub-slot code vectors cluster into a small number
of true contents (~46 per slot at R=16/K=4). So: run k-means over sub-slot
vectors from the frozen ``ConceptAutoencoder`` (FactorizedEncoder), index
sub-slots by nearest centroid, and decode by feeding centroids back through the
codec's own decoder. Zero new codec training; the reconstruction ceiling is
measured (and gated) like everywhere else.

    python -m dlm.concept.discrete --ae runs/slots/ae_R16_K4.pt --k 512 \
        --gate-recon 0.95 --out runs/fsq/kmeans_R16_K4.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from dlm.concept.ae import ConceptAutoencoder


def _lloyd(x: torch.Tensor, k: int, iters: int = 60, seed: int = 0) -> torch.Tensor:
    """Plain Lloyd k-means; x (N, d) -> centroids (k, d). Empty clusters are
    re-seeded from random points."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    cent = x[torch.randperm(x.shape[0], generator=g)[:k]].clone().to(x.device)
    for _ in range(iters):
        d2 = torch.cdist(x, cent)
        assign = d2.argmin(dim=1)
        for j in range(k):
            m = assign == j
            if m.any():
                cent[j] = x[m].mean(dim=0)
            else:
                cent[j] = x[torch.randint(0, x.shape[0], (1,), generator=g).item()]
    return cent


class KMeansCodec(nn.Module):
    """Duck-typed discrete codec (same interface the code-token stack uses:
    ``R``, ``cfg.n_slots``, ``codebook_size``, ``encode_ids``, ``decode_ids``)."""

    def __init__(self, ae: ConceptAutoencoder, centroids: torch.Tensor):
        super().__init__()
        self.ae = ae
        self.register_buffer("centroids", centroids)  # (k, d_sub)
        self.cfg = ae.cfg  # exposes n_slots

    @property
    def R(self) -> int:
        return self.ae.R

    @property
    def codebook_size(self) -> int:
        return self.centroids.shape[0]

    @torch.no_grad()
    def encode_ids(self, tokens: torch.Tensor) -> torch.Tensor:
        z = self.ae.encode(tokens)  # (B, m, d), normalized
        B, m, d = z.shape
        K = self.cfg.n_slots
        sub = z.view(B, m, K, d // K).reshape(-1, d // K)
        ids = torch.cdist(sub, self.centroids).argmin(dim=1)
        return ids.view(B, m * K)

    @torch.no_grad()
    def decode_ids(self, ids: torch.Tensor, out_len: int) -> torch.Tensor:
        B = ids.shape[0]
        K = self.cfg.n_slots
        m = ids.shape[1] // K
        z = self.centroids[ids.reshape(-1)].view(B, m, K, -1).reshape(B, m, -1)
        return self.ae.decode(z, out_len)

    def save(self, path: str):
        torch.save({"kind": "kmeans_codec", "ae_state": self.ae.state_dict(),
                    "ae_cfg": self.ae.cfg.__dict__, "centroids": self.centroids.cpu()}, path)

    @classmethod
    def load(cls, path: str, map_location=None) -> "KMeansCodec":
        from dlm.concept.ae import AEConfig
        ckpt = torch.load(path, map_location=map_location)
        ae = ConceptAutoencoder(AEConfig(**ckpt["ae_cfg"]))
        ae.load_state_dict(ckpt["ae_state"])
        return cls(ae.eval(), ckpt["centroids"])


def load_codec(path: str, map_location=None):
    """Dispatch loader: kmeans_codec or (experimental) FSQ checkpoints."""
    kind = torch.load(path, map_location="cpu").get("kind", "")
    if kind == "kmeans_codec":
        return KMeansCodec.load(path, map_location)
    from dlm.concept.fsq import FSQAutoencoder
    return FSQAutoencoder.load(path, map_location)


def main(argv=None):
    from dlm.data import make_task
    from dlm.eval.recon import eval_reconstruction  # noqa: F401  (gate uses check below)
    from dlm.train.loop import TaskSampler, get_device

    p = argparse.ArgumentParser()
    p.add_argument("--ae", required=True, help="trained FACTORIZED ConceptAutoencoder")
    p.add_argument("--k", type=int, default=512)
    p.add_argument("--fit-batches", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--task", default="function_compose")
    p.add_argument("--task-n", type=int, default=5)
    p.add_argument("--task-max-depth", type=int, default=32)
    p.add_argument("--answer-mode", default="trace")
    p.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    p.add_argument("--gate-recon", type=float, default=0.95)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    device = get_device()

    ae = ConceptAutoencoder.load(args.ae, map_location=device).to(device).eval()
    assert ae.cfg.n_slots > 0, "k-means codec requires a factorized (n_slots>0) codec"
    task = make_task(args.task, n=args.task_n, max_depth=args.task_max_depth,
                     answer_mode=args.answer_mode)
    sampler = TaskSampler(task, args.depths, args.batch_size, device, seed=args.seed,
                          uniform=True, depth_power=1.0)
    subs = []
    with torch.no_grad():
        for _ in range(args.fit_batches):
            b = sampler.next()
            for span in (b["prompt"], b["answer"]):
                z = ae.encode(span)
                B, m, d = z.shape
                subs.append(z.view(B, m, ae.cfg.n_slots, -1).reshape(-1, d // ae.cfg.n_slots))
    x = torch.cat(subs, dim=0)
    print(f"[kmeans] fitting k={args.k} on {x.shape[0]} sub-slot vectors (dim {x.shape[1]})")
    cent = _lloyd(x, args.k, seed=args.seed)
    codec = KMeansCodec(ae, cent).to(device)

    # ceiling gate in the task metric
    rng = np.random.default_rng(999)
    worst, per = 1.0, []
    for dpt in args.depths:
        bb = task.sample(256, depth=int(dpt), rng=rng)
        ans = torch.as_tensor(bb.answer, device=device)
        pred = codec.decode_ids(codec.encode_ids(ans), task.answer_len).argmax(-1).cpu().numpy()
        c = float(task.check(bb.prompt, pred).mean())
        per.append(f"D={dpt}:{c:.3f}")
        worst = min(worst, c)
    used = codec.encode_ids(torch.as_tensor(bb.answer, device=device)).unique().numel()
    print(f"[kmeans] chain-end recon ceiling  {'  '.join(per)}   codes-in-use(sample) {used}/{args.k}")
    if args.gate_recon > 0 and worst < args.gate_recon:
        print(f"[kmeans] GATE FAILED: {worst:.3f} < {args.gate_recon} -- not saving")
        raise SystemExit(3)
    codec.save(args.out)
    print(f"[kmeans] saved -> {args.out}")


if __name__ == "__main__":
    main()
