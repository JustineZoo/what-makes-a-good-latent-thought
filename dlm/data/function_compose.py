"""Function-composition / state-tracking task (torch-free).

Primary M1 workhorse. A *program* is a sequence of D group generators applied to
an initial state; the answer is (a query into) the resulting state. Reasoning
depth D = program length.

Default group = the symmetric group S_n with n=5. This is deliberate: iterated
composition in a **non-solvable** group (S_5 = the smallest non-solvable
symmetric group) admits no constant-depth (TC0) shortcut unless TC0=NC1
(Barrington + Merrill-Sabharwal) -- required circuit depth must grow with D.
Note the precise sense: associativity still allows an O(log D) parallel-prefix
tree, so what the group choice excludes is a depth-FREE solution (contrast
abelian groups, where counting gives one), not parallelism; D is a dial on
required depth. That is the regime where the serial-compute hypothesis (AR
chain length vs diffusion denoising steps T) should bite. See plan, Q1-Q3.

Answer modes and what each one tests:
  * ``trace`` -- **the primary mode for the study.** The answer span is the full
    chain of intermediate states (state after op 1, after op 2, ..., padded to
    max_depth states). This is the chain-of-thought analog: it is the span the
    concept autoencoder compresses, so compression ratio R directly shrinks the
    number of serial units an AR generator gets to emit (ceil(answer_len/R))
    while diffusion's serial budget stays T. Without a trace there is nothing
    for compression to starve. Correctness (``check``) grades ONLY the D-th
    state (the oracle recomputes D from the prompt), so the graded metric stays
    a short span and the AE-reconstruction confound on the metric is small and
    measurable. (An earlier layout duplicated the final state into a fixed tail
    slot after the pad field; M0 showed masked diffusion pathologically
    continues the pad field into such a slot once nearby pads are revealed, so
    the graded object is the chain end itself -- nothing to commit to early and
    no copy shortcut.)
  * ``query`` -- 1-token answer (which value does position q map to). No-CoT
    control: AR's serial budget is ~1 at every R.
  * ``full``  -- the final state (n tokens), no intermediate chain. Short-answer
    control.

An easy commutative control group ("affine mod P") is also provided; its
composition *does* have a closed form, so a model can shortcut depth. Use it as a
negative control -- the AR/diffusion interaction should be weak there.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .base import Batch, Task
from .vocab import Vocab

Perm = Tuple[int, ...]


# --------------------------------------------------------------------------- #
# group machinery
# --------------------------------------------------------------------------- #
def _identity(n: int) -> Perm:
    return tuple(range(n))


def _transposition(i: int, j: int, n: int) -> Perm:
    p = list(range(n))
    p[i], p[j] = p[j], p[i]
    return tuple(p)


def _ncycle(n: int) -> Perm:
    # i -> (i+1) mod n
    return tuple((i + 1) % n for i in range(n))


def _compose(a: Perm, b: Perm) -> Perm:
    # (a after b): result[i] = a[b[i]]
    return tuple(a[b[i]] for i in range(len(a)))


def symmetric_generators(n: int) -> List[Perm]:
    """A small generating set of S_n (a transposition + an n-cycle generate S_n;
    extras add branching factor)."""
    gens = [_transposition(0, 1, n), _ncycle(n)]
    if n >= 3:
        gens.append(_transposition(1, 2, n))
    if n >= 4:
        gens.append(_compose(_transposition(2, 3, n), _transposition(0, 1, n)))
    return gens


class FunctionCompose(Task):
    def __init__(
        self,
        n: int = 5,
        max_depth: int = 32,
        answer_mode: str = "query",  # "trace" (CoT chain, primary) | "query" | "full"
        group: str = "s5",
    ):
        assert answer_mode in ("query", "full", "trace")
        self.n = n
        self.max_depth = max_depth
        self.answer_mode = answer_mode
        self.group = group
        self.name = f"function_compose[{group},n{n},{answer_mode}]"

        if group in ("s5", "symmetric"):
            self.gens: List[Perm] = symmetric_generators(n)
        elif group == "affine":
            # x -> (a*x + b) mod n ; encoded as a permutation of Z_n when gcd(a,n)=1
            self.gens = _affine_generators(n)
        else:
            raise ValueError(f"unknown group {group!r}")

        self.k = len(self.gens)
        op_syms = [f"g{j}" for j in range(self.k)]
        val_syms = [f"v{i}" for i in range(n)]
        self.vocab = Vocab(op_syms + val_syms)
        self._op0 = self.vocab.stoi["g0"]
        self._val0 = self.vocab.stoi["v0"]

        if answer_mode == "query":
            self.answer_len = 1
        elif answer_mode == "full":
            self.answer_len = n
        else:  # trace: one state slot per program step, padded to max_depth
            self.answer_len = max_depth * n
        # prompt: <bos> g..(max_depth).. <qry> v_q <sep>   (query mode)
        #         <bos> g..(max_depth).. <sep>             (full / trace mode)
        if answer_mode == "query":
            self.prompt_len = 1 + max_depth + 1 + 1 + 1
        else:
            self.prompt_len = 1 + max_depth + 1

    # -- helpers -----------------------------------------------------------
    def _apply_program(self, op_ids: Sequence[int]) -> Perm:
        state = _identity(self.n)
        for op in op_ids:
            state = _compose(self.gens[op], state)
        return state

    def _trace_answer(self, op_ids: Sequence[int]) -> List[int]:
        """[state_1 .. state_D, pad-states ..] as token ids."""
        v = self.vocab
        row: List[int] = []
        state = _identity(self.n)
        for op in op_ids:
            state = _compose(self.gens[op], state)
            row += [self._val0 + state[i] for i in range(self.n)]
        row += [v.pad] * ((self.max_depth - len(op_ids)) * self.n)
        assert len(row) == self.answer_len
        return row

    def _depths_from_prompt(self, prompt: np.ndarray) -> np.ndarray:
        """(B, Lp) -> (B,) program length D per row (count non-pad op slots)."""
        ops = prompt[:, 1 : 1 + self.max_depth]
        return (ops != self.vocab.pad).sum(axis=1)

    def _encode_prompt(self, op_ids: Sequence[int], q: int, depth: int) -> np.ndarray:
        v = self.vocab
        row = [v.bos]
        row += [self._op0 + o for o in op_ids]
        row += [v.pad] * (self.max_depth - depth)  # right-pad the program slot
        if self.answer_mode == "query":
            row += [v.qry, self._val0 + q, v.sep]
        else:
            row += [v.sep]
        assert len(row) == self.prompt_len, (len(row), self.prompt_len)
        return np.asarray(row, dtype=np.int64)

    # -- Task API ----------------------------------------------------------
    def sample(self, batch_size: int, depth: int, rng: np.random.Generator) -> Batch:
        assert 0 < depth <= self.max_depth
        prompts = np.empty((batch_size, self.prompt_len), dtype=np.int64)
        answers = np.empty((batch_size, self.answer_len), dtype=np.int64)
        for b in range(batch_size):
            op_ids = rng.integers(0, self.k, size=depth).tolist()
            q = int(rng.integers(0, self.n)) if self.answer_mode == "query" else 0
            prompts[b] = self._encode_prompt(op_ids, q, depth)
            if self.answer_mode == "trace":
                answers[b] = self._trace_answer(op_ids)
                continue
            perm = self._apply_program(op_ids)
            if self.answer_mode == "query":
                answers[b, 0] = self._val0 + perm[q]
            else:
                answers[b] = [self._val0 + perm[i] for i in range(self.n)]
        return Batch(prompts, answers, depth=np.full(batch_size, depth), meta={"task": self.name})

    def oracle(self, prompt: np.ndarray) -> np.ndarray:
        prompt = np.asarray(prompt, dtype=np.int64)
        B = prompt.shape[0]
        out = np.empty((B, self.answer_len), dtype=np.int64)
        v = self.vocab
        for b in range(B):
            row = prompt[b].tolist()
            # program = op tokens between <bos> and (<qry>|<sep>), skipping pads
            op_ids: List[int] = []
            q = 0
            i = 1  # skip <bos>
            while i < len(row) and row[i] not in (v.qry, v.sep):
                if row[i] != v.pad:
                    op_ids.append(row[i] - self._op0)
                i += 1
            if i < len(row) and row[i] == v.qry:
                q = row[i + 1] - self._val0
            if self.answer_mode == "trace":
                out[b] = self._trace_answer(op_ids)
                continue
            perm = self._apply_program(op_ids)
            if self.answer_mode == "query":
                out[b, 0] = self._val0 + perm[q]
            else:
                out[b] = [self._val0 + perm[j] for j in range(self.n)]
        return out

    def check(self, prompt: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """Trace mode grades ONLY the D-th state slot (the chain's end -- did the
        model complete the computation): the rest of the chain is scaffolding
        whose granularity we manipulate, not part of the graded answer. Other
        modes fall back to exact match on the whole answer."""
        if self.answer_mode != "trace":
            return super().check(prompt, pred)
        prompt = np.asarray(prompt, dtype=np.int64)
        gold = self.oracle(prompt)
        pred = np.asarray(pred, dtype=np.int64).reshape(gold.shape)
        depths = self._depths_from_prompt(prompt)
        cols = (depths - 1)[:, None] * self.n + np.arange(self.n)[None, :]
        return (np.take_along_axis(pred, cols, 1) == np.take_along_axis(gold, cols, 1)).all(axis=1)

    def trace_exact(self, prompt: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """Secondary metric (trace mode): the whole chain incl. pads is exact."""
        gold = self.oracle(prompt)
        pred = np.asarray(pred, dtype=np.int64).reshape(gold.shape)
        return (pred == gold).all(axis=1)


def _affine_generators(n: int) -> List[Perm]:
    """Generators x -> (a*x+b) mod n that are permutations of Z_n (gcd(a,n)=1)."""
    gens: List[Perm] = []
    for a, b in [(1, 1), (1, 2), (n - 1, 0)]:  # +1, +2, negate: all bijections mod n
        if _gcd(a % n, n) == 1:
            gens.append(tuple((a * x + b) % n for x in range(n)))
    if not gens:  # tiny-n fallback
        gens = [tuple((x + 1) % n for x in range(n))]
    return gens


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a
