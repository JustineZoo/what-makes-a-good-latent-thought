"""Tests for the torch-free analysis layer: serial-compute accounting and the
gamma(T) scaling fit. Run: ``python tests/test_accounting.py`` or via pytest."""
from __future__ import annotations

import numpy as np

from dlm.eval.serial_compute import serial_compute, concept_count
from dlm.analysis.scaling_fit import fit_gamma_per_T, gamma_trend


def test_serial_compute_decoupling():
    # concept-diffusion serial cost is independent of compression R (depends on T)
    a = serial_compute("concept_diffusion", T=16, R=4)
    b = serial_compute("concept_diffusion", T=16, R=16)
    assert a == b == 16
    # concept-AR serial cost DROPS as R grows (fewer answer concepts) -> the mechanism
    fine = serial_compute("concept_ar", n_answer_tokens=64, R=4)
    coarse = serial_compute("concept_ar", n_answer_tokens=64, R=16)
    assert coarse < fine
    assert concept_count(64, 16) == 4 and concept_count(64, 4) == 16


def test_head_inner_steps_counted():
    base = serial_compute("concept_ar", n_answer_tokens=32, R=8, head_inner_steps=1)
    withhead = serial_compute("concept_ar", n_answer_tokens=32, R=8, head_inner_steps=4)
    assert withhead == 4 * base


def test_gamma_recovery_and_trend():
    # synthesize err = e0 + a * R**gamma(T) with gamma decreasing in T, recover it
    R = np.array([1, 4, 8, 16], dtype=float)
    e0, a = 0.02, 0.5
    true_gamma = {1: 0.9, 4: 0.6, 16: 0.35, 64: 0.2}
    err_by_T = {T: (e0 + a * R ** g).tolist() for T, g in true_gamma.items()}
    est = fit_gamma_per_T(R, err_by_T, e0=e0)
    for T in true_gamma:
        assert abs(est[T] - true_gamma[T]) < 1e-6, (T, est[T], true_gamma[T])
    trend = gamma_trend(est)
    assert trend["supports_hypothesis"] == 1.0
    assert trend["monotone_decreasing"] == 1.0
    assert trend["slope_gamma_vs_logT"] < 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} accounting tests passed")
