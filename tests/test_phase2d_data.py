"""Offline unit tests for Phase 2d OOD math loaders and ProsQA shortcut audit."""
import json
from pathlib import Path

from dlm.data.ood_math import load_ood_math_split
from dlm.data.prosqa import audit_prosqa_shortcuts, parse_prosqa_record
from dlm.eval.gsm8k_metrics import score_prediction


def test_ood_math_loaders_and_scoring(tmp_path: Path):
    svamp_f = tmp_path / "svamp.json"
    svamp_f.write_text(json.dumps([{"Body": "Tom has 5 apples.", "Question": "He buys 3 more. How many?", "Answer": 8}]))
    svamp = load_ood_math_split("svamp", local_path=str(svamp_f))
    assert len(svamp) == 1 and svamp[0].answer == "8"
    assert score_prediction("#### 8", svamp[0].steps, svamp[0].answer).answer_correct

    hard_f = tmp_path / "gsm_hard.jsonl"
    hard_f.write_text(json.dumps({"input": "Compute 1234 * 2.", "target": 2468.0}) + "\n")
    hard = load_ood_math_split("gsm_hard", local_path=str(hard_f))
    assert len(hard) == 1 and hard[0].answer == "2468"

    ma_f = tmp_path / "multiarith.json"
    ma_f.write_text(json.dumps([{"question": "2 + 2?", "final_ans": "4"}]))
    ma = load_ood_math_split("multiarith", local_path=str(ma_f))
    assert len(ma) == 1 and ma[0].answer == "4"


def test_prosqa_shortcut_audit():
    records = [
        {
            "question": "Every fumpus is a wumpus. Every wumpus is a numpus. Alex is a fumpus. Is Alex a numpus or a zumpus?",
            "steps": ["Alex is a fumpus.", "Alex is a wumpus.", "Alex is a numpus."],
            "answer": "numpus",
        },
        {
            "question": "Every lorpus is a gorpus. Every gorpus is a torpus. Sam is a lorpus. Is Sam a quxus or a torpus?",
            "steps": ["Sam is a lorpus.", "Sam is a gorpus.", "Sam is a torpus."],
            "answer": "torpus",
        },
    ]
    examples = [parse_prosqa_record(r) for r in records]
    assert all(ex is not None for ex in examples)
    res = audit_prosqa_shortcuts(examples)
    assert res["random_chance_floor"] == 0.5
    assert res["first_candidate_acc"] == 0.5
    assert res["second_candidate_acc"] == 0.5
