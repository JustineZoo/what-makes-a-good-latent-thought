"""Exact-oracle tests for the synthetic tasks. Pure-Python; run with pytest or
directly: ``python tests/test_data_oracles.py``.

These guard the foundation of the whole study: accuracy is measured against these
oracles, so ``sample().answer`` must equal ``oracle(prompt)`` for every example.
"""
from __future__ import annotations

import numpy as np

from dlm.data import make_task


def _roundtrip(task, depths, seed=0, bs=64):
    rng = np.random.default_rng(seed)
    for d in depths:
        b = task.sample(bs, depth=d, rng=rng)
        gold = task.oracle(b.prompt)
        assert gold.shape == b.answer.shape, (gold.shape, b.answer.shape)
        assert np.array_equal(gold, b.answer), f"{task.name}: oracle != sampled answer at depth {d}"
        # check() must be all-True on the gold answer
        assert task.check(b.prompt, b.answer).all(), f"{task.name}: check() failed on gold at depth {d}"


def test_function_compose_s5_query():
    task = make_task("function_compose", n=5, max_depth=32, answer_mode="query", group="s5")
    _roundtrip(task, depths=[1, 2, 5, 16, 32])


def test_function_compose_s5_full():
    task = make_task("function_compose", n=5, max_depth=16, answer_mode="full", group="s5")
    _roundtrip(task, depths=[1, 4, 16])


def test_function_compose_s5_trace():
    task = make_task("function_compose", n=5, max_depth=16, answer_mode="trace", group="s5")
    assert task.answer_len == 16 * 5
    _roundtrip(task, depths=[1, 4, 16])


def test_trace_layout():
    n, D_max = 5, 8
    task = make_task("function_compose", n=n, max_depth=D_max, answer_mode="trace", group="s5")
    rng = np.random.default_rng(3)
    for depth in (1, 3, D_max):
        b = task.sample(16, depth=depth, rng=rng)
        pad = task.vocab.pad
        for row_a in b.answer:
            states = row_a.reshape(D_max, n)
            # slots 1..depth are real states (no pads); slots beyond depth are all pad
            assert (states[:depth] != pad).all()
            assert (states[depth:] == pad).all()
        assert task.trace_exact(b.prompt, b.answer).all()


def test_trace_check_grades_chain_end_only():
    n, depth = 5, 6
    task = make_task("function_compose", n=n, max_depth=8, answer_mode="trace", group="s5")
    rng = np.random.default_rng(4)
    b = task.sample(64, depth=depth, rng=rng)
    val0 = task.vocab.stoi["v0"]
    # corrupt an early-chain token -> still graded correct (chain is scaffolding)
    mid = b.answer.copy()
    mid[:, 0] = val0 + (mid[:, 0] - val0 + 1) % task.n
    assert task.check(b.prompt, mid).all()
    assert not task.trace_exact(b.prompt, mid).any()
    # corrupt a pad slot beyond the chain -> still graded correct
    padslot = b.answer.copy()
    padslot[:, -1] = val0
    assert task.check(b.prompt, padslot).all()
    # corrupt the D-th state (the graded chain end) -> wrong
    end = b.answer.copy()
    j = (depth - 1) * n
    end[:, j] = val0 + (end[:, j] - val0 + 1) % task.n
    assert not task.check(b.prompt, end).any()


def test_function_compose_affine_control():
    task = make_task("function_compose", n=7, max_depth=16, answer_mode="query", group="affine")
    _roundtrip(task, depths=[1, 8, 16])


def test_graph_reach():
    task = make_task("graph_reach", width=3, max_depth=8, p_edge=0.4)
    _roundtrip(task, depths=[1, 2, 4, 8])


def test_check_detects_wrong_answers():
    task = make_task("function_compose", n=5, answer_mode="query")
    rng = np.random.default_rng(1)
    b = task.sample(128, depth=8, rng=rng)
    # corrupt every answer to a different value -> all wrong
    val0 = task.vocab.stoi["v0"]
    wrong = b.answer.copy()
    wrong[:, 0] = val0 + (b.answer[:, 0] - val0 + 1) % task.n
    assert not task.check(b.prompt, wrong).any()


def test_determinism():
    t1 = make_task("function_compose", n=5, answer_mode="query")
    t2 = make_task("function_compose", n=5, answer_mode="query")
    a = t1.sample(32, depth=10, rng=np.random.default_rng(123))
    b = t2.sample(32, depth=10, rng=np.random.default_rng(123))
    assert np.array_equal(a.prompt, b.prompt) and np.array_equal(a.answer, b.answer)


def test_graph_reach_trace():
    w, D = 5, 12
    task = make_task("graph_reach", width=w, max_depth=D, p_edge=0.6,
                     answer_mode="trace", out_degree=2)
    _roundtrip(task, depths=[1, 4, 12])
    rng = np.random.default_rng(21)
    b = task.sample(64, depth=6, rng=rng)
    yes = task.vocab.stoi["yes"]
    # frontier content is non-degenerate: both yes and no occur in graded slots
    slot = b.answer[:, 5 * w : 6 * w]
    frac = (slot == yes).mean()
    assert 0.05 < frac < 0.95, f"degenerate frontier balance: {frac}"
    # check grades only the D-th frontier
    early = b.answer.copy()
    early[:, 0] = yes if early[0, 0] != yes else task.vocab.stoi["no"]
    assert task.check(b.prompt, early).all()
    end = b.answer.copy()
    end[:, 5 * w] = np.where(end[:, 5 * w] == yes, task.vocab.stoi["no"], yes)
    assert not task.check(b.prompt, end).any()


def test_graph_reach_label_balance():
    # sanity: both YES and NO occur (task isn't degenerate)
    task = make_task("graph_reach", width=3, max_depth=6)
    b = task.sample(512, depth=4, rng=np.random.default_rng(7))
    yes = task.vocab.stoi["yes"]
    frac_yes = (b.answer[:, 0] == yes).mean()
    assert 0.15 < frac_yes < 0.85, f"degenerate label balance: {frac_yes}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} data-oracle tests passed")
