import glob
import json
import os
import re
import numpy as np

def main():
    os.makedirs("outputs", exist_ok=True)
    lines = [
        "=" * 88,
        " TIER 1 CPU LOG RE-SCORING: FULL COVERAGE, MEAN PREFIX LENGTH & FIXED-T",
        "=" * 88,
    ]

    # 1. Teacher-Forced JSON Probes (Top-Level List + per_span Un-Truncated Coverage)
    lines.append("\n[1] Teacher-Forced JSON Probes (Full Coverage Across All Concepts & States)")
    lines.append("-" * 88)
    for jf in sorted(glob.glob("outputs/tf_*.json")):
        with open(jf) as f:
            raw = json.load(f)
        entries = raw if isinstance(raw, list) else [raw]
        for idx, data in enumerate(entries):
            per_span = data.get("per_span", [])
            tok_accs = [float(s["token_acc"]) for s in per_span if "token_acc" in s]
            state_accs = [float(st) for s in per_span for st in s.get("state_accs", [])]

            tf_tok = data.get("tf_token_acc_mean")
            tf_st = data.get("tf_state_acc_mean")
            tok_str = f"{tf_tok * 100:6.2f}%" if isinstance(tf_tok, (int, float)) else "  N/A  "
            st_str = f"{tf_st * 100:6.2f}%" if isinstance(tf_st, (int, float)) else "  N/A  "
            st_mpl = f"{np.sum(state_accs):5.2f}/{len(state_accs)}" if state_accs else "  N/A  "

            tag = os.path.basename(jf) if len(entries) == 1 else f"{os.path.basename(jf)}[{idx}]"
            lines.append(
                f"{tag:<24} | Concepts={len(tok_accs):<2} | States={len(state_accs):<2} | "
                f"TokMean={tok_str} | StateMean={st_str} | StateMPL={st_mpl}"
            )
            c_str = " ".join([f"c{s['concept']}:{s['token_acc']*100:5.1f}%" for s in per_span])
            if c_str:
                lines.append(f"   Concept TokAcc : {c_str}")
            if state_accs:
                s_str = " ".join([f"s{i}:{v*100:5.1f}%" for i, v in enumerate(state_accs)])
                lines.append(f"   Contained State: {s_str}")

    # 2. Truncation-Free Mean Prefix Length (MPL) vs. OLS Log-Slope on Multi-Seed Curves
    lines.append("\n[2] Truncation-Free Mean Prefix Length (MPL) vs. OLS Slope (Multi-Seed Logs)")
    lines.append("-" * 88)
    for tf in sorted(glob.glob("outputs/*multiseed*.txt")):
        with open(tf) as f:
            txt = f.read()
        # Split multi-model files (e.g., bbh_multiseed_sweep_results.txt) into separate sections
        sections = re.split(r"(?=^---\s*Model:)", txt, flags=re.MULTILINE)
        for sec in sections:
            m_hdr = re.search(r"^---\s*Model:\s*(.+?)\s*---", sec, flags=re.MULTILINE)
            label = f"{os.path.basename(tf)} [{m_hdr.group(1)}]" if m_hdr else os.path.basename(tf)
            pairs = re.findall(r"Depth\s+(\d+)\s*:\s*([0-9.]+)%", sec)
            if not pairs:
                continue
            depths = [int(d) for d, _ in pairs]
            accs = [float(a) / 100.0 for _, a in pairs]
            d_max = max(depths)

            # Piecewise-linear survival curve S(1..D_max) anchored at S(0)=1.0
            s_curve = np.interp(np.arange(1, d_max + 1), [0] + depths, [1.0] + accs)
            mpl = float(np.sum(s_curve))

            # OLS log-slope with S > 0.02 filter (matching Appendix B.2)
            valid = [(d, a) for d, a in zip(depths, accs) if a > 0.02]
            if len(valid) >= 2:
                xs = np.array([v[0] for v in valid], dtype=float)
                ys = np.log([v[1] for v in valid])
                p_ols = float(np.exp(np.polyfit(xs, ys, 1)[0]))
            else:
                p_ols = float("nan")

            curve_str = ", ".join([f"D{d}={a * 100:.2f}%" for d, a in zip(depths, accs)])
            lines.append(
                f"{label:<44} | MPL={mpl:5.2f}/{d_max:<2} | p_OLS={p_ols:5.3f} | {curve_str}"
            )

    # 3. Fixed-T vs. Best-of-T Diffusion Summary
    lines.append("\n[3] Diffusion Fixed-T (T=16, T=64) vs. Post-Hoc Best-of-T")
    lines.append("-" * 88)
    for df in sorted(glob.glob("outputs/*diff*75k*.txt")):
        lines.append(f"Source: {df}")
        with open(df) as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(f"   {line}")

    lines.append("\n" + "=" * 88)
    report = "\n".join(lines)
    print(report)
    with open("outputs/tier1_log_rescoring_summary.txt", "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()

