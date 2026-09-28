"""Task interface for the synthetic reasoning suite (torch-free).

Every task must expose an **exact oracle**: a deterministic solver that recomputes
the answer tokens purely from the prompt tokens. This is what makes the whole
study trustworthy -- accuracy is measured against ground truth, and the data
generators are unit-tested by asserting ``sample().answer == oracle(prompt)``.

Design choices tied to the experiment:
  * ``depth`` (reasoning depth D) is an explicit, dial-able knob. The ablation
    harness sweeps it to build accuracy-vs-D curves.
  * Answers are kept **short** (often a single token) so that autoencoder
    reconstruction error on the *answer* is negligible; this decouples "cannot
    reason" from "cannot reconstruct" (see plan, Component C / confound control).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .vocab import Vocab


@dataclass
class Batch:
    prompt: np.ndarray  # (B, Lp) int64 token ids (the problem statement / program)
    answer: np.ndarray  # (B, La) int64 token ids (short)
    depth: np.ndarray   # (B,)   int reasoning depth D of each example
    meta: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.prompt = np.asarray(self.prompt, dtype=np.int64)
        self.answer = np.asarray(self.answer, dtype=np.int64)
        self.depth = np.asarray(self.depth, dtype=np.int64)

    @property
    def batch_size(self) -> int:
        return self.prompt.shape[0]


class Task(ABC):
    """Abstract synthetic reasoning task.

    Subclasses set ``name``, ``vocab`` and ``answer_len`` and implement
    ``sample`` and ``oracle``.
    """

    name: str
    vocab: Vocab
    answer_len: int
    prompt_len: int  # fixed, padded length of the prompt

    @abstractmethod
    def sample(self, batch_size: int, depth: int, rng: np.random.Generator) -> Batch:
        """Draw a batch whose examples all have reasoning depth ``depth``."""

    @abstractmethod
    def oracle(self, prompt: np.ndarray) -> np.ndarray:
        """Exact solver. ``(B, Lp) -> (B, answer_len)``, recomputed from prompt only."""

    def check(self, prompt: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """Elementwise exact-match correctness against the oracle. -> (B,) bool."""
        gold = self.oracle(prompt)
        pred = np.asarray(pred, dtype=np.int64).reshape(gold.shape)
        return (pred == gold).all(axis=1)

    @property
    def max_len(self) -> int:
        return self.prompt_len + self.answer_len
