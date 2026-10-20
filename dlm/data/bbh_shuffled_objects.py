"""BBH Tracking Shuffled Objects task (torch-free)."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .base import Batch, Task
from .vocab import (
    BBH_ENTITIES,
    BBH_OBJECTS,
    BBH_SHUFFLED_OBJECTS_VOCAB,
)


class BBHShuffledObjectsTask(Task):
    """Track pairwise item swaps described with fixed-vocabulary statements."""

    _INITIAL_PREFIX_LEN = 2
    _INITIAL_STATEMENT_LEN = 3
    _SWAP_STATEMENT_LEN = 5

    def __init__(self, n: int = 5, max_depth: int = 8, answer_mode: str = "trace"):
        if not 1 < n <= len(BBH_ENTITIES):
            raise ValueError(f"n must be between 2 and {len(BBH_ENTITIES)}")
        if max_depth < 1:
            raise ValueError("max_depth must be positive")
        if answer_mode not in ("trace", "query", "full"):
            raise ValueError(f"unknown answer_mode {answer_mode!r}")

        self.n = n
        self.max_depth = max_depth
        self.answer_mode = answer_mode
        self.name = f"bbh_shuffled_objects[n{n},D<={max_depth},{answer_mode}]"
        self.vocab = BBH_SHUFFLED_OBJECTS_VOCAB
        self.transpositions: Tuple[Tuple[int, int], ...] = tuple(
            (i, j) for i in range(n) for j in range(i + 1, n)
        )
        self._entity_ids = self.vocab.encode(BBH_ENTITIES[:n])
        self._object_ids = self.vocab.encode(BBH_OBJECTS[:n])
        self._entity_to_index = {
            token_id: i for i, token_id in enumerate(self._entity_ids)
        }
        self._object_to_index = {
            token_id: i for i, token_id in enumerate(self._object_ids)
        }
        self._has = self.vocab.stoi["has"]
        self._and = self.vocab.stoi["and"]
        self._swap = self.vocab.stoi["swap"]
        self._balls = self.vocab.stoi["balls"]
        self._who = self.vocab.stoi["who"]
        self._at = self.vocab.stoi["at"]
        self._start = self.vocab.stoi["start"]

        self.answer_len = (
            max_depth * n if answer_mode == "trace"
            else 1 if answer_mode == "query"
            else n
        )
        self._swap_start = (
            1 + self._INITIAL_PREFIX_LEN + self._INITIAL_STATEMENT_LEN * n
        )
        query_len = 5 if answer_mode == "query" else 1
        self.prompt_len = (
            self._swap_start
            + self._SWAP_STATEMENT_LEN * max_depth
            + query_len
        )

    def _encode_prompt(
        self, swaps: List[Tuple[int, int]], query_object: int
    ) -> np.ndarray:
        v = self.vocab
        row = [v.bos, self._at, self._start]
        for entity_id, object_id in zip(self._entity_ids, self._object_ids):
            row.extend([entity_id, self._has, object_id])
        for i, j in swaps:
            row.extend([
                self._entity_ids[i], self._and, self._entity_ids[j],
                self._swap, self._balls,
            ])
        row.extend([v.pad] * (self._SWAP_STATEMENT_LEN * (self.max_depth - len(swaps))))
        if self.answer_mode == "query":
            row.extend([v.qry, self._who, self._has, self._object_ids[query_object], v.sep])
        else:
            row.append(v.sep)
        assert len(row) == self.prompt_len, (len(row), self.prompt_len)
        return np.asarray(row, dtype=np.int64)

    def _depths_from_prompt(self, prompt: np.ndarray) -> np.ndarray:
        slots = prompt[
            :, self._swap_start:self._swap_start + self._SWAP_STATEMENT_LEN * self.max_depth
        ].reshape(-1, self.max_depth, self._SWAP_STATEMENT_LEN)
        return (slots[:, :, 0] != self.vocab.pad).sum(axis=1)

    def _parse_row(
        self, row: np.ndarray
    ) -> tuple[np.ndarray, List[Tuple[int, int]], int | None]:
        v = self.vocab
        if row.shape != (self.prompt_len,) or int(row[0]) != v.bos:
            raise ValueError(f"prompt row must have shape ({self.prompt_len},) and begin with <bos>")
        if int(row[1]) != self._at or int(row[2]) != self._start:
            raise ValueError("prompt must begin its initial state with 'at start'")

        state = np.full(self.n, -1, dtype=np.int64)
        for i in range(self.n):
            start = 1 + self._INITIAL_PREFIX_LEN + self._INITIAL_STATEMENT_LEN * i
            entity, relation, obj = (int(token) for token in row[start:start + 3])
            if entity not in self._entity_to_index or relation != self._has:
                raise ValueError("prompt contains an invalid initial assignment statement")
            entity_index = self._entity_to_index[entity]
            if state[entity_index] != -1 or obj not in self._object_to_index:
                raise ValueError("prompt contains an invalid or duplicate initial assignment")
            state[entity_index] = self._object_to_index[obj]
        if set(state.tolist()) != set(range(self.n)):
            raise ValueError("initial assignments must contain each object exactly once")

        swaps: List[Tuple[int, int]] = []
        for slot in range(self.max_depth):
            start = self._swap_start + slot * self._SWAP_STATEMENT_LEN
            statement = [int(token) for token in row[start:start + self._SWAP_STATEMENT_LEN]]
            if statement[0] == v.pad:
                if any(token != v.pad for token in statement):
                    raise ValueError("partially padded swap statement")
                continue
            first, conjunction, second, verb, noun = statement
            if (
                first not in self._entity_to_index
                or second not in self._entity_to_index
                or first == second
                or conjunction != self._and
                or verb != self._swap
                or noun != self._balls
            ):
                raise ValueError("prompt contains an invalid swap statement")
            swaps.append((
                self._entity_to_index[first], self._entity_to_index[second]
            ))

        query_object = None
        tail_start = self._swap_start + self._SWAP_STATEMENT_LEN * self.max_depth
        if self.answer_mode == "query":
            tail = [int(token) for token in row[tail_start:]]
            if (
                len(tail) != 5
                or tail[0] != v.qry
                or tail[1] != self._who
                or tail[2] != self._has
                or tail[4] != v.sep
                or tail[3] not in self._object_to_index
            ):
                raise ValueError("prompt contains an invalid object-holder query")
            query_object = self._object_to_index[tail[3]]
        elif int(row[-1]) != v.sep:
            raise ValueError("prompt must end with <sep>")
        return state, swaps, query_object

    def sample(self, batch_size: int, depth: int, rng: np.random.Generator) -> Batch:
        if not 0 < depth <= self.max_depth:
            raise ValueError(f"depth must be between 1 and {self.max_depth}")
        prompts = np.empty((batch_size, self.prompt_len), dtype=np.int64)
        answers = np.full((batch_size, self.answer_len), self.vocab.pad, dtype=np.int64)
        for b in range(batch_size):
            chosen = rng.integers(0, len(self.transpositions), size=depth)
            swaps = [self.transpositions[int(index)] for index in chosen]
            query_object = int(rng.integers(0, self.n)) if self.answer_mode == "query" else 0
            prompts[b] = self._encode_prompt(swaps, query_object)

            state = np.arange(self.n)
            if self.answer_mode == "trace":
                for step, (i, j) in enumerate(swaps):
                    state[i], state[j] = state[j], state[i]
                    start = step * self.n
                    answers[b, start:start + self.n] = [
                        self._object_ids[int(object_index)] for object_index in state
                    ]
            else:
                for i, j in swaps:
                    state[i], state[j] = state[j], state[i]
                if self.answer_mode == "query":
                    holder = int(np.flatnonzero(state == query_object)[0])
                    answers[b, 0] = self._entity_ids[holder]
                else:
                    answers[b] = [self._object_ids[int(index)] for index in state]
        return Batch(
            prompts, answers, depth=np.full(batch_size, depth),
            meta={"task": self.name},
        )

    def oracle(self, prompt: np.ndarray) -> np.ndarray:
        prompt = np.asarray(prompt, dtype=np.int64)
        if prompt.ndim != 2 or prompt.shape[1] != self.prompt_len:
            raise ValueError(
                f"prompt must have shape (batch, {self.prompt_len}), got {prompt.shape}"
            )
        out = np.full((prompt.shape[0], self.answer_len), self.vocab.pad, dtype=np.int64)
        for b, row in enumerate(prompt):
            state, swaps, query_object = self._parse_row(row)
            if self.answer_mode == "trace":
                for step, (i, j) in enumerate(swaps):
                    state[i], state[j] = state[j], state[i]
                    start = step * self.n
                    out[b, start:start + self.n] = [
                        self._object_ids[int(object_index)] for object_index in state
                    ]
            else:
                for i, j in swaps:
                    state[i], state[j] = state[j], state[i]
                if self.answer_mode == "query":
                    holder = int(np.flatnonzero(state == query_object)[0])
                    out[b, 0] = self._entity_ids[holder]
                else:
                    out[b] = [self._object_ids[int(index)] for index in state]
        return out

    def check(self, prompt: np.ndarray, pred: np.ndarray) -> np.ndarray:
        gold = self.oracle(prompt)
        pred = np.asarray(pred, dtype=np.int64).reshape(gold.shape)
        if self.answer_mode != "trace":
            return (pred == gold).all(axis=1)
        depths = self._depths_from_prompt(np.asarray(prompt, dtype=np.int64))
        cols = (depths - 1)[:, None] * self.n + np.arange(self.n)[None, :]
        return (
            np.take_along_axis(pred, cols, axis=1)
            == np.take_along_axis(gold, cols, axis=1)
        ).all(axis=1)

    def trace_exact(self, prompt: np.ndarray, pred: np.ndarray) -> np.ndarray:
        gold = self.oracle(prompt)
        pred = np.asarray(pred, dtype=np.int64).reshape(gold.shape)
        return (pred == gold).all(axis=1)
