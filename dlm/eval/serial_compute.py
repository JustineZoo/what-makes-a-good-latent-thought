"""Serial-compute accounting (torch-free).

The Q3 "unification" test asks whether reasoning accuracy is a function of *total
serial compute* regardless of paradigm. "Serial compute" here = the number of
network evaluations along the **critical path** (the sequential quantity that both
chain-of-thought length and denoising steps contribute to). This module is the
single source of truth for that x-axis, so AR and diffusion are counted on the
same footing.

Counting model (per generated answer):
  * Concept-AR: answers are produced one concept at a time -> ``n_answer_concepts``
    sequential model calls; if the per-concept head is itself iterative (MAR-style
    diffusion head with ``head_inner_steps`` steps), multiply by that.
  * Concept-diffusion: ``T`` denoising steps, each a single (parallel) pass over
    the whole concept sequence -> ``T`` sequential calls, INDEPENDENT of how many
    concepts there are. This decoupling is the crux of the hypothesis.
  * Token-AR baseline: ``n_answer_tokens`` sequential calls.
  * Token masked-diffusion baseline: ``T`` sequential calls.

Multiply by ``n_layers`` for an "effective serial depth" if you want to fold in
per-call depth; report both.
"""
from __future__ import annotations

import math
from typing import Optional


def concept_count(n_tokens: int, R: int) -> int:
    """Number of concept units for a span of ``n_tokens`` at compression ratio R."""
    return max(1, math.ceil(n_tokens / R))


def concept_ar_serial(n_answer_concepts: int, head_inner_steps: int = 1) -> int:
    return n_answer_concepts * head_inner_steps


def concept_diffusion_serial(T: int) -> int:
    return T


def token_ar_serial(n_answer_tokens: int) -> int:
    return n_answer_tokens


def token_diffusion_serial(T: int) -> int:
    return T


def serial_compute(
    paradigm: str,
    *,
    n_answer_tokens: Optional[int] = None,
    R: int = 1,
    T: Optional[int] = None,
    head_inner_steps: int = 1,
) -> int:
    """Dispatch a paradigm to its critical-path serial-call count.

    paradigm in {"concept_ar", "concept_diffusion", "token_ar", "token_diffusion"}.
    """
    if paradigm == "token_ar":
        assert n_answer_tokens is not None
        return token_ar_serial(n_answer_tokens)
    if paradigm in ("token_diffusion", "token_uniform_diffusion"):
        # uniform-state diffusion has no min(T, n_answer) cap (every step edits
        # all positions), but the serial-call count is still T backbone passes
        assert T is not None
        return token_diffusion_serial(T)
    if paradigm == "concept_ar":
        assert n_answer_tokens is not None
        return concept_ar_serial(concept_count(n_answer_tokens, R), head_inner_steps)
    if paradigm == "concept_diffusion":
        assert T is not None
        return concept_diffusion_serial(T)
    raise ValueError(f"unknown paradigm {paradigm!r}")


def effective_serial_depth(serial_calls: int, n_layers: int) -> int:
    return serial_calls * n_layers
