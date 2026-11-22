"""Run the Phase 2d ProsQA prompt-only shortcut audit and save report to outputs/."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dlm.data.prosqa import audit_prosqa_shortcuts, load_prosqa_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--local-path", default=None)
    p.add_argument("--hf-repo", default="whynlp/prosqa")
    p.add_argument("--out", default="outputs/prosqa_shortcut_audit.txt")
    args = p.parse_args()

    examples = load_prosqa_split(split=args.split, local_path=args.local_path, hf_repo=args.hf_repo)
    res = audit_prosqa_shortcuts(examples)

    lines = [
        "============================================================",
        f" ProsQA Prompt-Only Shortcut Audit (split={args.split}, n={res['n']})",
        "============================================================",
        f"Random Candidate Chance Floor : {res['random_chance_floor']*100:6.2f}%",
        f"First-Candidate Position Bias : {res['first_candidate_acc']*100:6.2f}%",
        f"Second-Candidate Position Bias: {res['second_candidate_acc']*100:6.2f}%",
        f"Premise Mention-Frequency Bias: {res['premise_frequency_acc']*100:6.2f}%",
        f"Max Prompt-Only Heuristic     : {res['max_prompt_heuristic_acc']*100:6.2f}% (gap={res['shortcut_gap']*100:+.2f} pp)",
        f"Verdict                       : {'PASS (Shortcut-Free)' if res['shortcut_free'] else 'WARNING (Shortcut Detected)'}",
        "============================================================",
    ]
    report = "\n".join(lines) + "\n"
    print(report)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    out_path.with_suffix(".json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
