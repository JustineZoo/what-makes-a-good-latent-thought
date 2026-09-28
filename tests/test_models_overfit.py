"""Overfit-one-batch smoke tests for the torch models.

Skips cleanly if torch is not installed (e.g. on a login node). On a GPU/CPU node
with the ``train`` extra it runs in a few seconds and asserts that (a) the concept
autoencoder can reconstruct a fixed batch and (b) both generators reduce their loss
on a fixed (context, target) pair -- the minimum sanity bar before any real run.

Run: ``python tests/test_models_overfit.py``  or  ``pytest tests/test_models_overfit.py``.
"""
from __future__ import annotations

import numpy as np

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False


def _skip(msg="torch not installed"):
    try:
        import pytest
        pytest.skip(msg, allow_module_level=False)
    except Exception:
        print(f"SKIP: {msg}")
    return None


def test_ae_overfits_one_batch():
    if not HAVE_TORCH:
        return _skip()
    import torch
    from dlm.data import make_task
    from dlm.concept.ae import AEConfig, ConceptAutoencoder

    torch.manual_seed(0)
    task = make_task("function_compose", n=5, max_depth=8, answer_mode="query")
    cfg = AEConfig(vocab_size=len(task.vocab), d_concept=64, compression_ratio=4,
                   max_tokens=task.max_len, beta=0.05, free_bits=0.1, enc_layers=2, dec_layers=2, n_heads=4)
    ae = ConceptAutoencoder(cfg)
    b = task.sample(64, depth=6, rng=np.random.default_rng(0))
    prompt = torch.as_tensor(b.prompt)
    opt = torch.optim.AdamW(ae.parameters(), lr=1e-3)

    def det_acc():  # deterministic recon (mu path) -- what generators/eval consume
        with torch.no_grad():
            pred = ae.decode(ae.encode(prompt), out_len=prompt.shape[1]).argmax(-1)
        return (pred == prompt).float().mean().item()

    first = det_acc()
    for _ in range(800):  # ~0.95 by step 900 at this size (see probe in review notes)
        out = ae.loss(prompt)
        opt.zero_grad(); out["loss"].backward(); opt.step()
    final = det_acc()
    assert final > 0.9, f"AE failed to overfit prompt span: {first:.2f} -> {final:.2f}"
    print(f"ok  test_ae_overfits_one_batch  det recon acc {first:.2f} -> {final:.2f}")


def _overfit_generator(kind):
    import torch
    from dlm.data import make_task
    from dlm.concept.ae import AEConfig, ConceptAutoencoder
    from dlm.generators.ar import ConceptAR
    from dlm.generators.diffusion import ConceptFlowMatching

    torch.manual_seed(0)
    task = make_task("function_compose", n=5, max_depth=8, answer_mode="query")
    ae = ConceptAutoencoder(AEConfig(vocab_size=len(task.vocab), d_concept=64, compression_ratio=4,
                                     max_tokens=task.max_len, enc_layers=2, dec_layers=2, n_heads=4))
    for p in ae.parameters():
        p.requires_grad_(False)
    b = task.sample(32, depth=6, rng=np.random.default_rng(1))
    with torch.no_grad():
        cond = ae.encode(torch.as_tensor(b.prompt))
        target = ae.encode(torch.as_tensor(b.answer))
    gen = (ConceptFlowMatching(64, 2, 4, 32) if kind == "diffusion"
           else ConceptAR(64, 2, 4, 32, head="mar", head_inner_steps=2))
    opt = torch.optim.AdamW(gen.parameters(), lr=1e-3)
    losses = []
    for _ in range(300):
        loss = gen.fit_step(target, cond)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss))
    assert losses[-1] < 0.7 * losses[0], f"{kind} gen did not reduce loss: {losses[0]:.3f}->{losses[-1]:.3f}"
    # sampling returns the right shape
    z = gen.sample(cond, n_target=target.shape[1], **({"T": 8} if kind == "diffusion" else {}))
    assert z.shape == target.shape
    print(f"ok  {kind} gen  loss {losses[0]:.3f} -> {losses[-1]:.3f}, sample shape {tuple(z.shape)}")


def test_diffusion_generator_overfits():
    if not HAVE_TORCH:
        return _skip()
    _overfit_generator("diffusion")


def test_ar_generator_overfits():
    if not HAVE_TORCH:
        return _skip()
    _overfit_generator("ar")


if __name__ == "__main__":
    if not HAVE_TORCH:
        print("SKIP: torch not installed (run on a GPU node with `pip install -e .[train]`)")
    else:
        test_ae_overfits_one_batch()
        test_diffusion_generator_overfits()
        test_ar_generator_overfits()
        print("\nall model overfit tests passed")
