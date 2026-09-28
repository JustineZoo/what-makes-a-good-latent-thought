"""Tiny symbol vocabulary shared by the synthetic tasks (torch-free)."""
from __future__ import annotations

from typing import Iterable, List

SPECIALS: List[str] = ["<pad>", "<bos>", "<eos>", "<sep>", "<qry>"]


class Vocab:
    """Ordered symbol table. Specials always occupy the first ids."""

    def __init__(self, symbols: Iterable[str]):
        syms = list(SPECIALS)
        for s in symbols:
            if s not in syms:
                syms.append(s)
        self.itos: List[str] = syms
        self.stoi = {s: i for i, s in enumerate(syms)}

    def __len__(self) -> int:
        return len(self.itos)

    # --- special-token shortcuts ---
    @property
    def pad(self) -> int:
        return self.stoi["<pad>"]

    @property
    def bos(self) -> int:
        return self.stoi["<bos>"]

    @property
    def eos(self) -> int:
        return self.stoi["<eos>"]

    @property
    def sep(self) -> int:
        return self.stoi["<sep>"]

    @property
    def qry(self) -> int:
        return self.stoi["<qry>"]

    def encode(self, symbols: Iterable[str]) -> List[int]:
        return [self.stoi[s] for s in symbols]

    def decode(self, ids: Iterable[int]) -> List[str]:
        return [self.itos[int(i)] for i in ids]
