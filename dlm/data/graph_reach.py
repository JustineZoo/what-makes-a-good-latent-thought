"""Graph reachability task (torch-free).

Second task family, with two answer modes:

  * ``query`` (original): a layered DAG is emitted as an edge list; the query
    asks whether ``t`` is reachable from ``s``; the answer is one YES/NO token.
  * ``trace`` -- the frontier-trace mode used for the generality study. The
    answer span is the chain of REACHABLE-SETS: for each layer k = 1..D, a
    ``width``-token slot whose j-th token is yes/no for "node j of layer k is
    reachable from s" (layers beyond D are pad slots, mirroring the S5 trace
    layout). Grading (``check``) reads only the D-th frontier -- the chain end.
    The underlying computation is iterated composition of boolean relations on
    ``width`` elements; at width >= 5 the transition monoid contains S_5, so the
    family embeds the same NC1-hard word problem as the primary task while the
    SURFACE FORM (relational edge-list prompts, set-valued states) is entirely
    different -- which is exactly what a generality test needs.

    Trace mode caps per-node out-degree (``out_degree``) so prompt length stays
    manageable, lays edges out in fixed per-(layer, node, slot) prompt positions
    (pad pairs where absent), and appends an explicit depth token to the prompt
    (analogous to the program length being visible in the S5 prompt).

Ground truth is per-layer BFS, so labels are exactly correct by construction.
"""
from __future__ import annotations

from collections import deque
from typing import List, Tuple

import numpy as np

from .base import Batch, Task
from .vocab import Vocab


class GraphReach(Task):
    def __init__(self, width: int = 3, max_depth: int = 8, p_edge: float = 0.4,
                 answer_mode: str = "query", out_degree: int = 2,
                 walker: bool = False, trace_ids: bool = False):
        """``walker=True`` (trace mode only): exactly ONE out-edge per node per
        layer, i.e. each layer applies a total function [width]->[width] and the
        frontier is always a single node (1-hot rows). This removes the
        absorption artifact of the stochastic variant (dead frontiers make deep
        chain-ends trivially predictable, so accuracy RISES with depth); the
        walker never dies, depth stays a genuine computation dial, and the
        transition monoid is the full transformation monoid on ``width`` points
        (contains S_width, so the family is NC1-hard at width>=5)."""
        assert answer_mode in ("query", "trace")
        assert not walker or answer_mode == "trace"
        assert not trace_ids or walker, "trace_ids requires walker mode"
        # trace_ids: VALUE-CODED walker trace -- the answer is one node-id token
        # per layer (the walker's position), not a yes/no frontier row. Rationale:
        # 1-hot yes/no rows are position-coded (identical pooled content for every
        # state), which defeats mean-pool concept codecs (two gate failures at
        # 0.000); node-id tokens are value-coded like the S5 traces that compress
        # cleanly. Grading = the D-th token.
        self.walker = walker
        self.trace_ids = trace_ids
        self.width = width
        self.max_depth = max_depth
        self.p_edge = p_edge
        self.answer_mode = answer_mode
        self.out_degree = out_degree
        self.n = 1 if trace_ids else width  # state-slot size (chain-survival probe)
        mode = "walker_ids" if trace_ids else ("walker" if walker else answer_mode)
        self.name = f"graph_reach[w{width},D<={max_depth},{mode}]"

        self.num_nodes = (max_depth + 1) * width  # max over depths
        node_syms = [f"n{i}" for i in range(self.num_nodes)]
        if answer_mode == "query":
            self.max_edges = max_depth * width * width  # full bipartite between layers
            self.vocab = Vocab(node_syms + ["yes", "no"])
        else:
            self.max_edges = max_depth * width * out_degree  # capped, fixed layout
            depth_syms = [f"d{k}" for k in range(1, max_depth + 1)]
            self.vocab = Vocab(node_syms + ["yes", "no"] + depth_syms)
            self._d0 = self.vocab.stoi["d1"]
        self._node0 = self.vocab.stoi["n0"]
        self._yes = self.vocab.stoi["yes"]
        self._no = self.vocab.stoi["no"]

        if answer_mode == "query":
            self.answer_len = 1
            # <bos> (src dst)*max_edges <qry> s t <sep>
            self.prompt_len = 1 + 2 * self.max_edges + 1 + 2 + 1
        else:
            self.answer_len = max_depth if trace_ids else max_depth * width
            # <bos> (src dst | pad pad)*max_edges <qry> s d_D <sep>
            self.prompt_len = 1 + 2 * self.max_edges + 1 + 2 + 1

    def _node(self, layer: int, idx: int) -> int:
        return layer * self.width + idx

    # ------------------------------------------------------------------ query
    def sample(self, batch_size: int, depth: int, rng: np.random.Generator) -> Batch:
        assert 0 < depth <= self.max_depth
        if self.answer_mode == "trace":
            return self._sample_trace(batch_size, depth, rng)
        prompts = np.empty((batch_size, self.prompt_len), dtype=np.int64)
        answers = np.empty((batch_size, 1), dtype=np.int64)
        for b in range(batch_size):
            edges: List[Tuple[int, int]] = []
            for layer in range(depth):
                for i in range(self.width):
                    for j in range(self.width):
                        if rng.random() < self.p_edge:
                            edges.append((self._node(layer, i), self._node(layer + 1, j)))
            s = self._node(0, int(rng.integers(0, self.width)))
            t = self._node(depth, int(rng.integers(0, self.width)))
            want_positive = bool(rng.integers(0, 2))
            edges = self._make_label(edges, s, t, depth, want_positive, rng)
            prompts[b] = self._encode(edges, s, t)
            answers[b, 0] = self._yes if self._reachable(edges, s, t) else self._no
        return Batch(prompts, answers, depth=np.full(batch_size, depth), meta={"task": self.name})

    def _make_label(self, edges, s, t, depth, want_positive, rng):
        reach = self._reachable(edges, s, t)
        if want_positive and not reach:
            # plant a chain s -> ... -> t through one node per layer
            chain = [s] + [self._node(l, int(rng.integers(0, self.width))) for l in range(1, depth)] + [t]
            edges = edges + [(chain[i], chain[i + 1]) for i in range(depth)]
        elif not want_positive and reach:
            # sever all incoming edges to t
            edges = [(u, w) for (u, w) in edges if w != t]
        return edges

    def _reachable(self, edges, s, t) -> bool:
        adj = {}
        for u, w in edges:
            adj.setdefault(u, []).append(w)
        seen = {s}
        dq = deque([s])
        while dq:
            u = dq.popleft()
            if u == t:
                return True
            for w in adj.get(u, ()):
                if w not in seen:
                    seen.add(w)
                    dq.append(w)
        return t in seen

    def _encode(self, edges, s, t) -> np.ndarray:
        v = self.vocab
        edges = edges[: self.max_edges]
        row = [v.bos]
        for (u, w) in edges:
            row += [self._node0 + u, self._node0 + w]
        row += [v.pad, v.pad] * (self.max_edges - len(edges))
        row += [v.qry, self._node0 + s, self._node0 + t, v.sep]
        assert len(row) == self.prompt_len, (len(row), self.prompt_len)
        return np.asarray(row, dtype=np.int64)

    # ------------------------------------------------------------------ trace
    def _sample_edge_slots(self, depth: int, rng: np.random.Generator) -> np.ndarray:
        """(max_depth, width, out_degree) target index in the next layer, or -1.
        Fixed per-slot layout keeps the prompt parse unambiguous. Walker mode
        fills slot 0 always (a total function per layer) and leaves the rest
        empty."""
        slots = np.full((self.max_depth, self.width, self.out_degree), -1, dtype=np.int64)
        for layer in range(depth):
            for i in range(self.width):
                if self.walker:
                    slots[layer, i, 0] = int(rng.integers(0, self.width))
                    continue
                for k in range(self.out_degree):
                    if rng.random() < self.p_edge:
                        slots[layer, i, k] = int(rng.integers(0, self.width))
        return slots

    def _frontiers(self, slots: np.ndarray, s_idx: int, depth: int) -> np.ndarray:
        """(depth, width) boolean reachability per layer, from node s_idx of layer 0."""
        f = np.zeros((depth, self.width), dtype=bool)
        cur = np.zeros(self.width, dtype=bool)
        cur[s_idx] = True
        for layer in range(depth):
            nxt = np.zeros(self.width, dtype=bool)
            for i in range(self.width):
                if cur[i]:
                    for k in range(self.out_degree):
                        j = slots[layer, i, k]
                        if j >= 0:
                            nxt[j] = True
            f[layer] = nxt
            cur = nxt
        return f

    def _encode_trace_prompt(self, slots: np.ndarray, s_idx: int, depth: int) -> np.ndarray:
        v = self.vocab
        row = [v.bos]
        for layer in range(self.max_depth):
            for i in range(self.width):
                for k in range(self.out_degree):
                    j = slots[layer, i, k]
                    if layer < depth and j >= 0:
                        row += [self._node0 + self._node(layer, i),
                                self._node0 + self._node(layer + 1, int(j))]
                    else:
                        row += [v.pad, v.pad]
        row += [v.qry, self._node0 + self._node(0, s_idx), self._d0 + depth - 1, v.sep]
        assert len(row) == self.prompt_len, (len(row), self.prompt_len)
        return np.asarray(row, dtype=np.int64)

    def _trace_answer_row(self, f: np.ndarray, depth: int) -> np.ndarray:
        v = self.vocab
        row = np.full(self.answer_len, v.pad, dtype=np.int64)
        if self.trace_ids:
            # value-coded: node id of the walker at each layer (layer-local index
            # encoded as the ABSOLUTE node symbol of that layer, like the prompt)
            for k in range(depth):
                j = int(np.argmax(f[k]))
                row[k] = self._node0 + self._node(k + 1, j)
            return row
        flat = np.where(f, self._yes, self._no).reshape(-1)
        row[: depth * self.width] = flat
        return row

    def _sample_trace(self, batch_size: int, depth: int, rng: np.random.Generator) -> Batch:
        prompts = np.empty((batch_size, self.prompt_len), dtype=np.int64)
        answers = np.empty((batch_size, self.answer_len), dtype=np.int64)
        for b in range(batch_size):
            slots = self._sample_edge_slots(depth, rng)
            s_idx = int(rng.integers(0, self.width))
            prompts[b] = self._encode_trace_prompt(slots, s_idx, depth)
            answers[b] = self._trace_answer_row(self._frontiers(slots, s_idx, depth), depth)
        return Batch(prompts, answers, depth=np.full(batch_size, depth), meta={"task": self.name})

    def _parse_trace_prompt(self, row: np.ndarray):
        v = self.vocab
        slots = np.full((self.max_depth, self.width, self.out_degree), -1, dtype=np.int64)
        p = 1
        for layer in range(self.max_depth):
            for i in range(self.width):
                for k in range(self.out_degree):
                    u, w = int(row[p]), int(row[p + 1])
                    p += 2
                    if u != v.pad:
                        slots[layer, i, k] = (w - self._node0) % self.width
        s_idx = (int(row[p + 1]) - self._node0) % self.width
        depth = int(row[p + 2]) - self._d0 + 1
        return slots, s_idx, depth

    def oracle(self, prompt: np.ndarray) -> np.ndarray:
        prompt = np.asarray(prompt, dtype=np.int64)
        if self.answer_mode == "trace":
            out = np.empty((prompt.shape[0], self.answer_len), dtype=np.int64)
            for b in range(prompt.shape[0]):
                slots, s_idx, depth = self._parse_trace_prompt(prompt[b])
                out[b] = self._trace_answer_row(self._frontiers(slots, s_idx, depth), depth)
            return out
        B = prompt.shape[0]
        out = np.empty((B, 1), dtype=np.int64)
        v = self.vocab
        for b in range(B):
            row = prompt[b].tolist()
            i = 1
            edges = []
            while i < len(row) and row[i] != v.qry:
                a, c = row[i], row[i + 1]
                if a != v.pad and c != v.pad:
                    edges.append((a - self._node0, c - self._node0))
                i += 2
            s = row[i + 1] - self._node0
            t = row[i + 2] - self._node0
            out[b, 0] = self._yes if self._reachable(edges, s, t) else self._no
        return out

    def check(self, prompt: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """Trace mode grades only the D-th frontier (the chain end); query mode
        falls back to exact match."""
        if self.answer_mode != "trace":
            return super().check(prompt, pred)
        prompt = np.asarray(prompt, dtype=np.int64)
        gold = self.oracle(prompt)
        pred = np.asarray(pred, dtype=np.int64).reshape(gold.shape)
        dpos = self.prompt_len - 2  # fixed depth-token position: ... s d_D <sep>
        depths = prompt[:, dpos] - self._d0 + 1
        w = 1 if self.trace_ids else self.width
        cols = (depths - 1)[:, None] * w + np.arange(w)[None, :]
        return (np.take_along_axis(pred, cols, 1) == np.take_along_axis(gold, cols, 1)).all(axis=1)

    def trace_exact(self, prompt: np.ndarray, pred: np.ndarray) -> np.ndarray:
        gold = self.oracle(prompt)
        pred = np.asarray(pred, dtype=np.int64).reshape(gold.shape)
        return (pred == gold).all(axis=1)
