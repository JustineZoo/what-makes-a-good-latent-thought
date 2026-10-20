"""Tiny symbol vocabulary shared by the synthetic tasks (torch-free)."""
from __future__ import annotations

from typing import Iterable, List

SPECIALS: List[str] = ["<pad>", "<bos>", "<eos>", "<sep>", "<qry>"]

# Keep ProntoQA's ontology and template words in a fixed order so token IDs do
# not depend on which examples happen to be sampled.
PRONTO_QA_CLASSES = (
    "zumpus", "gorpus", "blicket", "dax", "wug", "fep", "kiki", "toma",
    "lor", "niff", "gazzer", "norp", "wubble", "florp", "snarp", "glippet",
    "plumbus", "tulver", "blen", "mip", "rindle", "vump", "jib", "krel",
    "yump", "dorb", "quig", "floop", "trel", "bim", "zib", "wemb",
    "frotz", "blit",
)

PRONTO_QA_SYMBOLS = (
    "all", "are", "the", "entity", "is", "a", ".", "?", "true", "false",
    *PRONTO_QA_CLASSES,
)

BBH_ENTITIES = ["Alice", "Bob", "Charlie", "Dave", "Eve"]
BBH_OBJECTS = ["ball0", "ball1", "ball2", "ball3", "ball4"]
BBH_KEYWORDS = ["has", "and", "swap", "balls", "who", "at", "start"]


class Vocab:
    """Closed ordered symbol table. Specials always occupy the first ids.

    Encoding an unknown symbol raises ``KeyError``; the vocabulary never grows
    as a side effect of processing data.
    """

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


def pronto_qa_vocab() -> Vocab:
    """Return the fixed ProntoQA vocabulary with stable token IDs."""
    return Vocab(PRONTO_QA_SYMBOLS)


BBH_SHUFFLED_OBJECTS_VOCAB = Vocab(BBH_ENTITIES + BBH_OBJECTS + BBH_KEYWORDS)
