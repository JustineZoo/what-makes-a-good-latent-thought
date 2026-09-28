"""Strict information boundary of the blockwise codec (torch, CPU, seconds).

With ``blockwise=True`` concept k must depend only on tokens k*R..(k+1)*R-1 and
block k of the decoded logits must depend only on concept k. Checked on random
weights (the property is architectural, not learned), for an entangled and a
factorized code, and for a span whose length is not a multiple of R.
"""
from __future__ import annotations

try:
    import torch
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False


def _ae(R, K, blockwise):
    from dlm.concept.ae import AEConfig, ConceptAutoencoder
    cfg = AEConfig(vocab_size=14, d_concept=64, compression_ratio=R, max_tokens=160,
                   enc_layers=2, dec_layers=2, n_heads=4, n_slots=K, blockwise=blockwise)
    return ConceptAutoencoder(cfg).eval()


def _check(R, K, L, blockwise):
    torch.manual_seed(0)
    ae = _ae(R, K, blockwise)
    tok = torch.randint(0, 14, (2, L))
    m = -(-L // R)
    with torch.no_grad():
        z = ae.encode(tok)
        # encoder: change one token in block k -> only concept k may change
        k = m // 2
        tok2 = tok.clone(); tok2[:, k * R] = (tok2[:, k * R] + 1) % 14
        dz = (ae.encode(tok2) - z).abs().amax(dim=(0, 2)) > 1e-6
        enc_leak = bool(dz[torch.arange(m) != k].any())
        # decoder: perturb concept k -> only tokens in block k may change
        z2 = z.clone(); z2[:, k, :] += 1.0
        dl = (ae.decode(z2, out_len=L) - ae.decode(z, out_len=L)).abs().amax(dim=(0, 2)) > 1e-6
        inside = (torch.arange(L) // R) == k
        dec_leak = bool(dl[~inside].any())
        changed_inside = bool(dz[k]) and bool(dl[inside].any())
    return enc_leak, dec_leak, changed_inside


def test_blockwise_boundary():
    if not HAVE_TORCH:
        return
    for R, K, L in ((8, 0, 160), (16, 4, 160), (8, 0, 34)):
        enc_leak, dec_leak, changed = _check(R, K, L, blockwise=True)
        assert not enc_leak, f"encoder leaks across blocks at R={R} K={K} L={L}"
        assert not dec_leak, f"decoder leaks across blocks at R={R} K={K} L={L}"
        assert changed, "perturbation had no effect inside its own block"


def test_global_codec_is_not_blockwise():
    if not HAVE_TORCH:
        return
    enc_leak, dec_leak, _ = _check(8, 0, 160, blockwise=False)
    assert enc_leak and dec_leak, "the default codec should be global on both sides"


if __name__ == "__main__":
    test_blockwise_boundary(); test_global_codec_is_not_blockwise(); print("blockwise tests passed")
