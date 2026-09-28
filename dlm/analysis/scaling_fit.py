"""Compression x denoising-steps scaling-law fits (torch-free, numpy only).

DLCM (arXiv:2512.24617) fit a compression-aware law whose concept term scales as
``A_concept * R^gamma / (...)^delta`` -- reasoning loss grows with the compression
ratio R at exponent gamma. That law is autoregressive-only. Our central,
extending hypothesis:

    the compression exponent shrinks as denoising steps increase: gamma = gamma(T),
    monotonically decreasing, i.e. **spending denoising steps buys back the
    reasoning that coarser (denser) units cost.**

Given reasoning error measured on a grid of (R, T) at fixed model size / depth,
``fit_gamma_per_T`` recovers gamma(T) by a log-log linear fit of error vs R for
each T, and ``gamma_trend`` tests whether gamma decreases with T.

Everything uses only numpy so it runs on a login node. ``fit_augmented_law`` is an
optional joint fit that uses scipy if it happens to be installed.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def fit_gamma_per_T(
    R_values: Sequence[float],
    err_by_T: Dict[int, Sequence[float]],
    e0: float = 0.0,
    eps: float = 1e-9,
) -> Dict[int, float]:
    """For each T, fit ``log(err - e0) = const + gamma * log(R)`` and return gamma(T).

    ``err_by_T[T]`` is the error at each R in ``R_values`` (aligned order).
    """
    logR = np.log(np.asarray(R_values, dtype=float))
    gammas: Dict[int, float] = {}
    for T, errs in err_by_T.items():
        y = np.log(np.clip(np.asarray(errs, dtype=float) - e0, eps, None))
        slope, _ = np.polyfit(logR, y, 1)
        gammas[T] = float(slope)
    return gammas


def gamma_trend(gammas: Dict[int, float]) -> Dict[str, float]:
    """Slope of gamma vs log(T). Negative slope supports the hypothesis."""
    Ts = np.array(sorted(gammas), dtype=float)
    g = np.array([gammas[int(t)] for t in Ts], dtype=float)
    slope, intercept = np.polyfit(np.log(Ts), g, 1)
    monotone = bool(np.all(np.diff(g) <= 1e-9))
    return {
        "slope_gamma_vs_logT": float(slope),
        "gamma_at_min_T": float(g[0]),
        "gamma_at_max_T": float(g[-1]),
        "monotone_decreasing": float(monotone),
        "supports_hypothesis": float(slope < 0),
    }


def fit_augmented_law(R, T, err, p0=None):
    """Optional joint fit of ``err = e0 + a * R**(g_inf + (g0-g_inf)*exp(-T/tau))``.

    Requires scipy; raises ImportError otherwise. R, T, err are flat aligned arrays.
    """
    try:
        from scipy.optimize import curve_fit
    except Exception as exc:  # pragma: no cover
        raise ImportError("fit_augmented_law needs scipy; use fit_gamma_per_T instead") from exc

    R = np.asarray(R, float)
    T = np.asarray(T, float)
    err = np.asarray(err, float)

    def model(X, e0, a, g0, g_inf, tau):
        r, t = X
        gamma = g_inf + (g0 - g_inf) * np.exp(-t / max(tau, 1e-6))
        return e0 + a * r ** gamma

    p0 = p0 or [err.min() * 0.5, 1.0, 1.0, 0.2, np.median(T) + 1.0]
    popt, pcov = curve_fit(model, (R, T), err, p0=p0, maxfev=20000)
    keys = ["e0", "a", "gamma0", "gamma_inf", "tau"]
    return dict(zip(keys, popt)), pcov
