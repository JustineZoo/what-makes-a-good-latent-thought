"""Correctness tests for the UDLM-style uniform-state diffusion baseline.

The load-bearing check is the NELBO transcription test: the closed-form
continuous-time integrand must equal the dt->0 limit of the discrete-time
KL( q(z_s|z_t,x0) || q(z_s|z_t,x_theta) ) / dt at every (x0, z_t, t) -- this
validates the loss against first principles (Bayes-rule posterior of the
uniform interpolating kernel) rather than against a paraphrase of the paper.

Run: ``python tests/test_uniform_diffusion.py`` or ``pytest``.
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


def _rand_setup(B=8, L=11, V=23, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    from dlm.generators.token_baselines import TokenUniformDiffusion
    m = TokenUniformDiffusion(V, d=32, n_layers=1, n_heads=2, max_len=64)
    x0 = torch.randint(0, V, (B, L), generator=g)
    zt = torch.randint(0, V, (B, L), generator=g)
    x_theta = torch.rand(B, L, V, generator=g).softmax(-1).double().float()
    return m, x0, zt, x_theta


def test_posterior_is_normalized():
    if not HAVE_TORCH:
        return _skip()
    import torch
    m, x0, zt, x_theta = _rand_setup()
    for a_t, a_s in [(0.3, 0.7), (0.01, 0.99), (0.5, 0.500001)]:
        q = m._posterior(x_theta, zt, alpha_s=a_s, alpha_t=a_t)
        assert (q >= -1e-8).all(), "posterior has negative mass"
        err = (q.sum(-1) - 1).abs().max().item()
        assert err < 1e-5, f"posterior not normalized (max err {err:.2e})"
    print("ok  test_posterior_is_normalized")


def test_nelbo_matches_discrete_kl_limit():
    """Closed-form integrand == KL(q_true || q_pred)/dt as dt->0."""
    if not HAVE_TORCH:
        return _skip()
    import torch
    import torch.nn.functional as F
    m, x0, zt, x_theta = _rand_setup()
    V = m.vocab_size
    x0_1h = F.one_hot(x0, V).double()
    x_theta = x_theta.double()
    for t in (0.15, 0.5, 0.9):
        alpha_t, dt = 1 - t, 1e-6
        alpha_s = 1 - (t - dt)
        closed = m._diffusion_loss(x_theta, x0, zt,
                                   torch.full((x0.shape[0],), alpha_t, dtype=torch.float64))
        q_true = m._posterior(x0_1h, zt, alpha_s, alpha_t)
        q_pred = m._posterior(x_theta, zt, alpha_s, alpha_t)
        kl = (q_true * (q_true.clamp_min(1e-30).log()
                        - q_pred.clamp_min(1e-30).log())).sum(-1) / dt
        rel = ((closed - kl).abs() / kl.abs().clamp_min(1e-8)).max().item()
        assert rel < 1e-2, f"t={t}: closed-form vs discrete-KL mismatch (rel {rel:.3f})"
    print("ok  test_nelbo_matches_discrete_kl_limit")


def test_nelbo_zero_iff_perfect_and_positive_otherwise():
    if not HAVE_TORCH:
        return _skip()
    import torch
    import torch.nn.functional as F
    m, x0, zt, x_theta = _rand_setup()
    V = m.vocab_size
    alpha = torch.full((x0.shape[0],), 0.6)
    perfect = (F.one_hot(x0, V).float() * (1 - 1e-6) + 1e-6 / V)
    lp = m._diffusion_loss(perfect, x0, zt, alpha)
    assert lp.abs().max().item() < 1e-3, f"perfect model loss not ~0: {lp.abs().max():.2e}"
    lr_ = m._diffusion_loss(x_theta, x0, zt, alpha)
    assert lr_.min().item() > 0, "random-model NELBO integrand not positive"
    print("ok  test_nelbo_zero_iff_perfect_and_positive_otherwise")


def test_uniform_diffusion_overfits_one_batch():
    if not HAVE_TORCH:
        return _skip()
    import torch
    from dlm.data import make_task
    from dlm.generators.token_baselines import TokenUniformDiffusion

    torch.manual_seed(0)
    task = make_task("function_compose", n=3, max_depth=4, answer_mode="trace")
    V = len(task.vocab)
    m = TokenUniformDiffusion(V, d=64, n_layers=2, n_heads=4, max_len=task.max_len,
                              objective="ce")
    b = task.sample(32, depth=3, rng=np.random.default_rng(0))
    prompt = torch.as_tensor(b.prompt)
    answer = torch.as_tensor(b.answer)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    first = m.fit_step(prompt, answer).item()
    for _ in range(2500):
        loss = m.fit_step(prompt, answer)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    for T in (1, 32):  # the editable sampler must work at both extremes
        torch.manual_seed(1)
        pred = m.sample(prompt, n_answer=task.answer_len, T=T).numpy()
        acc = float(task.check(b.prompt, pred).mean())
        assert acc > 0.9, (f"uniform diffusion failed to overfit at T={T}: acc {acc:.2f} "
                           f"(loss {first:.2f}->{loss.item():.2f})")
    print(f"ok  test_uniform_diffusion_overfits_one_batch  acc {acc:.2f}  loss {first:.2f}->{loss.item():.2f}")


def test_nelbo_objective_trains():
    """The exact-NELBO objective is slow (uniform-state analog of Duo's compute
    multiplier) -- assert convergence direction, not task accuracy."""
    if not HAVE_TORCH:
        return _skip()
    import torch
    from dlm.data import make_task
    from dlm.generators.token_baselines import TokenUniformDiffusion

    torch.manual_seed(0)
    task = make_task("function_compose", n=3, max_depth=4, answer_mode="trace")
    m = TokenUniformDiffusion(len(task.vocab), d=64, n_layers=2, n_heads=4,
                              max_len=task.max_len, objective="nelbo")
    b = task.sample(32, depth=3, rng=np.random.default_rng(0))
    prompt = torch.as_tensor(b.prompt)
    answer = torch.as_tensor(b.answer)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    first = sum(m.fit_step(prompt, answer).item() for _ in range(5)) / 5
    for _ in range(400):
        loss = m.fit_step(prompt, answer)
        opt.zero_grad(); loss.backward(); opt.step()
    last = sum(m.fit_step(prompt, answer).item() for _ in range(5)) / 5
    assert last < 0.5 * first, f"NELBO objective not decreasing: {first:.2f} -> {last:.2f}"
    print(f"ok  test_nelbo_objective_trains  loss {first:.2f} -> {last:.2f}")


if __name__ == "__main__":
    test_posterior_is_normalized()
    test_nelbo_matches_discrete_kl_limit()
    test_nelbo_zero_iff_perfect_and_positive_otherwise()
    test_uniform_diffusion_overfits_one_batch()
    test_nelbo_objective_trains()
