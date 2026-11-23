"""Phase 2d: ProsQA loader and prompt-only shortcut audit."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from dlm.data.gsm8k_aug import format_prompt

_QUERY_RE = re.compile(r"Is\s+([A-Za-z0-9_-]+)\s+(?:an?|the)\s+([A-Za-z0-9_-]+)\s+or\s+(?:(?:an?|the)\s+)?([A-Za-z0-9_-]+)\?", re.I)


@dataclass
class ProsQAExample:
    prompt: str
    question: str
    steps: List[str]
    answer: str
    target: str
    candidates: List[str]


def extract_prosqa_candidates(question: str) -> List[str]:
    m = _QUERY_RE.search(question)
    if m:
        return [m.group(2).strip(), m.group(3).strip()]
    return []


def parse_prosqa_record(rec: Dict[str, object]) -> Optional[ProsQAExample]:
    question = str(rec.get("question", rec.get("input", ""))).strip()
    raw_steps = rec.get("steps", [])
    if isinstance(raw_steps, str):
        steps = [s.strip() for s in raw_steps.split("\n") if s.strip()]
    else:
        steps = [str(s).strip() for s in raw_steps if str(s).strip()]

    ans = str(rec.get("answer", rec.get("target", ""))).strip()
    if "####" in ans:
        ans = ans.split("####")[-1].strip()
    ans = ans.rstrip(".")
    if not question or not ans:
        return None

    cands = [str(c).strip() for c in rec.get("candidates", [])] if rec.get("candidates") else extract_prosqa_candidates(question)
    if ans and ans not in cands:
        cands.append(ans)

    prompt = format_prompt(question)
    step_str = " ".join(steps)
    target = f"{step_str}\n#### {ans}" if step_str else f"#### {ans}"
    return ProsQAExample(
        prompt=prompt,
        question=question,
        steps=steps,
        answer=ans,
        target=target,
        candidates=cands,
    )


def load_prosqa_split(
    split: str = "test",
    local_path: Optional[str] = None,
    hf_repo: str = "whynlp/prosqa",
) -> List[ProsQAExample]:
    if local_path is not None:
        p = Path(local_path)
        if p.suffix == ".jsonl":
            rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        else:
            payload = json.loads(p.read_text())
            rows = payload if isinstance(payload, list) else payload.get(split, payload.get("data", []))
    else:
        from datasets import load_dataset

        rows = list(load_dataset(hf_repo, split=split))

    out: List[ProsQAExample] = []
    for r in rows:
        ex = parse_prosqa_record(r)
        if ex is not None:
            out.append(ex)
    return out


def audit_prosqa_shortcuts(examples: Sequence[ProsQAExample]) -> Dict[str, object]:
    """Measure prompt-only heuristic accuracies against the random candidate floor."""
    n = len(examples)
    if n == 0:
        raise ValueError("Cannot audit an empty ProsQA example list")

    chance_sum = 0.0
    first_cand_hits = 0
    second_cand_hits = 0
    freq_hits = 0

    for ex in examples:
        cands = ex.candidates if ex.candidates else [ex.answer]
        chance_sum += 1.0 / max(1, len(cands))
        gt = ex.answer.lower()

        if cands and cands[0].lower() == gt:
            first_cand_hits += 1
        if len(cands) > 1 and cands[1].lower() == gt:
            second_cand_hits += 1

        # Count occurrences in the premise portion before the final question sentence
        premise = ex.question.rsplit(".", 1)[0].lower()
        counts = [len(re.findall(rf"\b{re.escape(c.lower())}\b", premise)) for c in cands]
        if counts and max(counts) > 0 and cands[counts.index(max(counts))].lower() == gt:
            freq_hits += 1

    chance = chance_sum / n
    first_acc = first_cand_hits / n
    second_acc = second_cand_hits / n
    freq_acc = freq_hits / n
    max_heuristic = max(first_acc, second_acc, freq_acc)

    return {
        "n": n,
        "random_chance_floor": chance,
        "first_candidate_acc": first_acc,
        "second_candidate_acc": second_acc,
        "premise_frequency_acc": freq_acc,
        "max_prompt_heuristic_acc": max_heuristic,
        "shortcut_gap": max_heuristic - chance,
        "shortcut_free": (max_heuristic - chance) <= 0.05,
    }
