"""Procedural ProntoQA-style entailment chains (torch-free)."""
from __future__ import annotations

import numpy as np

from .base import Batch, Task
from .vocab import PRONTO_QA_CLASSES, pronto_qa_vocab


class ProntoQATask(Task):
    """Class-inclusion reasoning with a one-token boolean answer.

    Prompts contain up to ``max_depth`` rules of the form
    ``all A are B``, a fact assigning one class to an entity, and a question
    asking whether that entity belongs to another class. Unused rule slots are
    padded to keep every prompt the same length. Trace mode returns the
    intermediate class IDs along the deduction chain, padded to max_depth.
    """

    _RULE_LEN = 5
    _FACT_LEN = 6
    _QUESTION_LEN = 8

    def __init__(self, max_depth: int = 6, answer_mode: str = "query"):
        if answer_mode not in ("query", "trace"):
            raise ValueError(f"unknown answer_mode {answer_mode!r}")
        if not 0 < max_depth <= len(PRONTO_QA_CLASSES) - 2:
            raise ValueError(
                f"max_depth must be between 1 and {len(PRONTO_QA_CLASSES) - 2}"
            )
        self.max_depth = max_depth
        self.answer_mode = answer_mode
        self.name = f"pronto_qa[D<={max_depth},{answer_mode}]"
        self.vocab = pronto_qa_vocab()
        self.n = 1
        self.answer_len = max_depth * self.n if answer_mode == "trace" else 1
        self.prompt_len = (
            1 + self._RULE_LEN * max_depth + self._FACT_LEN + self._QUESTION_LEN
        )
        self._class_ids = [self.vocab.stoi[name] for name in PRONTO_QA_CLASSES]
        self._class_to_index = {
            token_id: i for i, token_id in enumerate(self._class_ids)
        }
        self._true = self.vocab.stoi["true"]
        self._false = self.vocab.stoi["false"]

    def _encode_prompt(
        self, rules: list[tuple[int, int]], source: int, target: int
    ) -> np.ndarray:
        v = self.vocab
        row = [v.bos]
        for subject, predicate in rules:
            row.extend([v.stoi["all"], subject, v.stoi["are"], predicate, v.stoi["."]])
        row.extend([v.pad] * (self._RULE_LEN * (self.max_depth - len(rules))))
        row.extend([
            v.stoi["the"], v.stoi["entity"], v.stoi["is"], v.stoi["a"],
            source, v.stoi["."],
            v.qry, v.stoi["is"], v.stoi["the"], v.stoi["entity"],
            v.stoi["a"], target, v.stoi["?"], v.sep,
        ])
        assert len(row) == self.prompt_len, (len(row), self.prompt_len)
        return np.asarray(row, dtype=np.int64)

    def sample(self, batch_size: int, depth: int, rng: np.random.Generator) -> Batch:
        if not 0 < depth <= self.max_depth:
            raise ValueError(f"depth must be between 1 and {self.max_depth}")
        prompts = np.empty((batch_size, self.prompt_len), dtype=np.int64)
        answers = np.empty((batch_size, self.answer_len), dtype=np.int64)
        for b in range(batch_size):
            chain = rng.choice(len(self._class_ids), size=depth + 1, replace=False)
            rules = [
                (self._class_ids[int(chain[i])], self._class_ids[int(chain[i + 1])])
                for i in range(depth)
            ]
            rules = [rules[int(i)] for i in rng.permutation(depth)]
            is_true = self.answer_mode == "trace" or bool(rng.integers(0, 2))
            target_index = int(chain[-1]) if is_true else int(
                rng.choice(np.setdiff1d(np.arange(len(self._class_ids)), chain))
            )
            prompts[b] = self._encode_prompt(
                rules, self._class_ids[int(chain[0])], self._class_ids[target_index]
            )
            if self.answer_mode == "trace":
                answers[b] = self._trace_answer(
                    [self._class_ids[int(chain[i])] for i in range(1, depth + 1)],
                    depth,
                )
            else:
                answers[b, 0] = self._true if is_true else self._false
        return Batch(
            prompts, answers, depth=np.full(batch_size, depth),
            meta={"task": self.name},
        )

    def _parse_prompt(
        self, row: np.ndarray
    ) -> tuple[list[tuple[int, int]], int, int]:
        fact_start = 1 + self._RULE_LEN * self.max_depth
        query_start = fact_start + self._FACT_LEN
        rules = []
        for slot in range(self.max_depth):
            start = 1 + slot * self._RULE_LEN
            subject = int(row[start + 1])
            predicate = int(row[start + 3])
            if subject == self.vocab.pad:
                continue
            if subject not in self._class_to_index or predicate not in self._class_to_index:
                raise ValueError("prompt contains an unknown ProntoQA class token")
            rules.append((subject, predicate))
        source = int(row[fact_start + 4])
        target = int(row[query_start + 5])
        if source not in self._class_to_index:
            raise ValueError("prompt contains an unknown ProntoQA source class")
        if target not in self._class_to_index:
            raise ValueError("prompt contains an unknown ProntoQA target class")
        return rules, source, target

    def _trace_path(
        self, rules: list[tuple[int, int]], source: int, target: int
    ) -> list[int]:
        """Follow the unique deduction chain encoded by the rule set."""
        path = []
        used_rules: set[int] = set()
        current = source
        while current != target:
            matches = [
                i for i, (subject, _) in enumerate(rules)
                if subject == current and i not in used_rules
            ]
            if len(matches) != 1:
                raise ValueError("prompt does not encode a unique source-to-target chain")
            rule_index = matches[0]
            used_rules.add(rule_index)
            current = rules[rule_index][1]
            path.append(current)
            if len(path) > len(rules):
                raise ValueError("prompt contains a cyclic ProntoQA deduction chain")
        if len(used_rules) != len(rules):
            raise ValueError("prompt contains rules outside the queried deduction chain")
        return path

    def _trace_answer(self, steps: list[int], depth: int) -> np.ndarray:
        answer = np.full(self.answer_len, self.vocab.pad, dtype=np.int64)
        answer[:depth] = steps
        return answer

    def _trace_depths(self, prompt: np.ndarray) -> np.ndarray:
        rule_start = 1
        rule_end = rule_start + self._RULE_LEN * self.max_depth
        subjects = prompt[:, rule_start + 1:rule_end:self._RULE_LEN]
        return (subjects != self.vocab.pad).sum(axis=1)

    def oracle(self, prompt: np.ndarray) -> np.ndarray:
        prompt = np.asarray(prompt, dtype=np.int64)
        if prompt.ndim != 2 or prompt.shape[1] != self.prompt_len:
            raise ValueError(
                f"prompt must have shape (batch, {self.prompt_len}), got {prompt.shape}"
            )
        out = np.empty((prompt.shape[0], self.answer_len), dtype=np.int64)
        for b, row in enumerate(prompt):
            rules, source, target = self._parse_prompt(row)
            if self.answer_mode == "trace":
                path = self._trace_path(rules, source, target)
                out[b] = self._trace_answer(path, len(path))
                continue
            reachable = {source}
            changed = True
            while changed:
                changed = False
                for subject, predicate in rules:
                    if subject in reachable and predicate not in reachable:
                        reachable.add(predicate)
                        changed = True
            out[b, 0] = self._true if target in reachable else self._false
        return out

    def check(self, prompt: np.ndarray, pred: np.ndarray) -> np.ndarray:
        if self.answer_mode != "trace":
            return super().check(prompt, pred)
        prompt = np.asarray(prompt, dtype=np.int64)
        gold = self.oracle(prompt)
        pred = np.asarray(pred, dtype=np.int64).reshape(gold.shape)
        depths = self._trace_depths(prompt)
        cols = (depths - 1)[:, None]
        return (
            np.take_along_axis(pred, cols, axis=1)
            == np.take_along_axis(gold, cols, axis=1)
        ).all(axis=1)

    def trace_exact(self, prompt: np.ndarray, pred: np.ndarray) -> np.ndarray:
        gold = self.oracle(prompt)
        pred = np.asarray(pred, dtype=np.int64).reshape(gold.shape)
        return (pred == gold).all(axis=1)


# Retain the shorter name for callers that imported the original task class.
ProntoQA = ProntoQATask