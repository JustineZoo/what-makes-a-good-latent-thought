"""Phase 2d: Out-of-distribution final-answer math loaders (SVAMP, GSM-Hard, MultiArith)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dlm.data.gsm8k_aug import GSM8KExample, format_prompt

_DATASET_SPECS: Dict[str, Tuple[str, str]] = {
    "svamp": ("ChilleD/SVAMP", "test"),
    "gsm_hard": ("reasoning-machines/gsm-hard", "train"),
    "multiarith": ("ChilleD/MultiArith", "test"),
}


def _clean_numeric_answer(raw: object) -> str:
    s = str(raw).strip().replace(",", "").replace("$", "")
    try:
        val = float(s)
        if val.is_integer():
            return str(int(val))
        return f"{val:.6g}"
    except ValueError:
        return s


def _normalize_record(rec: Dict[str, object], dataset_name: str) -> Optional[GSM8KExample]:
    if dataset_name == "svamp":
        body = str(rec.get("Body", "")).strip()
        q_part = str(rec.get("Question", rec.get("question", ""))).strip()
        question = f"{body} {q_part}".strip() if body and body not in q_part else q_part
        ans = _clean_numeric_answer(rec.get("Answer", rec.get("answer", "")))
    elif dataset_name == "gsm_hard":
        question = str(rec.get("input", rec.get("question", ""))).strip()
        ans = _clean_numeric_answer(rec.get("target", rec.get("answer", "")))
    elif dataset_name == "multiarith":
        question = str(rec.get("question", rec.get("sQuestion", ""))).strip()
        raw_ans = rec.get("final_ans", rec.get("answer", rec.get("lSolutions", "")))
        if isinstance(raw_ans, list):
            raw_ans = raw_ans[0] if raw_ans else ""
        ans = _clean_numeric_answer(raw_ans)
    else:
        raise ValueError(f"Unsupported OOD math dataset: {dataset_name!r}")

    if not question or not ans:
        return None
    prompt = format_prompt(question)
    target = f" #### {ans}"
    return GSM8KExample(
        prompt=prompt,
        question=question,
        steps=[],
        answer=ans,
        target=target,
    )


def load_ood_math_split(
    name: str,
    local_path: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> List[GSM8KExample]:
    """Load SVAMP, GSM-Hard, or MultiArith into GSM8KExample format."""
    key = name.lower().replace("-", "_")
    if key not in _DATASET_SPECS:
        raise ValueError(f"Unknown OOD dataset {name!r}; expected one of {list(_DATASET_SPECS)}")

    if local_path is not None:
        p = Path(local_path)
        if p.suffix == ".jsonl":
            rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        else:
            payload = json.loads(p.read_text())
            rows = payload if isinstance(payload, list) else payload.get("data", [])
    else:
        from datasets import load_dataset

        hf_id, split = _DATASET_SPECS[key]
        rows = list(load_dataset(hf_id, split=split))

    examples: List[GSM8KExample] = []
    for rec in rows:
        ex = _normalize_record(rec, key)
        if ex is not None:
            examples.append(ex)
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples
