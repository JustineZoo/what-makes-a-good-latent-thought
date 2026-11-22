"""GSM8K-Aug dataset loader and prompt formatter (torch-free at import time)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional


PromptMode = Literal["cot", "no_cot"]


@dataclass(frozen=True)
class GSM8KExample:
    question: str
    steps: List[str]
    answer: str
    prompt: str
    target: str


def format_prompt(question: str) -> str:
    return f"Question: {question.strip()}\nSolution:\n"


def format_target(steps: List[str], answer: str, mode: PromptMode = "cot") -> str:
    ans = str(answer).strip()
    if mode == "no_cot":
        return f"#### {ans}"
    if mode == "cot":
        trace = " ".join(s.strip() for s in steps if s.strip())
        return f"{trace}\n#### {ans}" if trace else f"#### {ans}"
    raise ValueError(f"Unknown mode {mode!r}; expected 'cot' or 'no_cot'")


def load_gsm8k_aug_splits(
    dataset_name: str = "whynlp/gsm8k-aug",
    mode: PromptMode = "cot",
    extra_dev_size: int = 0,
    seed: int = 42,
) -> Dict[str, List[GSM8KExample]]:
    """Load train, validation, and official test splits from whynlp/gsm8k-aug."""
    from datasets import load_dataset

    raw = load_dataset(dataset_name)
    train_ds = raw["train"]
    val_ds = raw["validation"]
    test_ds = raw["test"]

    if extra_dev_size > 0:
        split = train_ds.train_test_split(test_size=extra_dev_size, seed=seed)
        train_ds = split["train"]
        extra_val = split["test"]
    else:
        extra_val = None

    def _convert(ds) -> List[GSM8KExample]:
        out: List[GSM8KExample] = []
        for row in ds:
            q = row["question"]
            steps = list(row["steps"])
            ans = str(row["answer"])
            out.append(
                GSM8KExample(
                    question=q,
                    steps=steps,
                    answer=ans,
                    prompt=format_prompt(q),
                    target=format_target(steps, ans, mode=mode),
                )
            )
        return out

    val_examples = _convert(val_ds)
    if extra_val is not None:
        val_examples.extend(_convert(extra_val))

    return {
        "train": _convert(train_ds),
        "validation": val_examples,
        "test": _convert(test_ds),
    }
