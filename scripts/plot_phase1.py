"""Build Phase 1 deliverables: codec-ceiling table and recon-vs-gen figure."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    out_dir = Path("outputs")
    Rs = [1, 2, 4, 8]

    # 1. Print Codec-Ceiling Table
    print("=== Phase 1b Codec-Ceiling Table (Llama-3.2-1B-Instruct, GSM8K) ===")
    print(f"{'R':>3} | {'Dev Step Exact':>14} | {'Dev Ans Acc':>11} | {'Test Step Exact':>15} | {'Test Ans Acc':>12} | {'Gate':>6}")
    print("-" * 74)
    ceilings = {}
    for r in Rs:
        c_data = load_json(out_dir / f"phase1b_codec_r{r}_s0" / "codec_metrics.json")
        if c_data:
            dev = c_data["dev_ceiling"]
            tst = c_data["test_ceiling"]
            ceilings[r] = tst["step_exact_accuracy"] * 100.0
            gate = "PASS" if c_data.get("gate_passed") else "FAIL"
            print(
                f"{r:>3} | {dev['step_exact_accuracy']*100:>13.2f}% | {dev['answer_accuracy']*100:>10.2f}% | "
                f"{tst['step_exact_accuracy']*100:>14.2f}% | {tst['answer_accuracy']*100:>11.2f}% | {gate:>6}"
            )
        else:
            print(f"{r:>3} | {'(pending)':>14} | {'-':>11} | {'-':>15} | {'-':>12} | {'-':>6}")

    # 2. Load Phase 1c CoLaR Probe
    colar = load_json(out_dir / "phase1c_colar_probe" / "colar_probe_metrics.json")
    if colar:
        print("\n=== Phase 1c CoLaR Probe (Reconstruction vs. Generation) ===")
        for k, v in colar.items():
            rc = v["reconstruction_ceiling"]["step_exact_accuracy"] * 100.0
            tf = v["teacher_forced"]["step_exact_accuracy"] * 100.0
            fr = v["free_running"]["step_exact_accuracy"] * 100.0
            ans = v["free_running"]["answer_accuracy"] * 100.0
            print(f"{k:<4} | Recon Step: {rc:5.2f}% | TF Step: {tf:5.2f}% | Free-Run Step: {fr:5.2f}% | Free-Run Ans: {ans:5.2f}%")

    # 3. Render 3-panel figure: [1x Budget | 3x Budget | CoLaR Probe Bars]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, budget in zip(axes[:2], ["1x", "3x"]):
        xs, ceil_y, tf_y, fr_y = [], [], [], []
        for r in Rs:
            m_data = load_json(out_dir / f"phase1b_gen_r{r}_s0" / f"metrics_{budget}.json")
            if m_data and r in ceilings:
                xs.append(r)
                ceil_y.append(ceilings[r])
                tf_y.append(m_data["teacher_forced"]["step_exact_accuracy"] * 100.0)
                fr_y.append(m_data["free_running"]["step_exact_accuracy"] * 100.0)
        if xs:
            ax.plot(xs, ceil_y, "o--", label="Codec Ceiling", color="#2b5c8f")
            ax.plot(xs, tf_y, "s-", label="Teacher-Forced Step", color="#2e8b57")
            ax.plot(xs, fr_y, "^-", label="Free-Running Step", color="#c0392b")
        ax.set_title(f"LoRA Codec + Flow AR ({budget} Budget)")
        ax.set_xlabel("Compression Ratio R")
        ax.set_xticks(Rs)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Step Exact Accuracy (%)")
    axes[0].legend(loc="lower left", frameon=True)

    ax_c = axes[2]
    if colar:
        factors = [1, 2, 5]
        x = np.arange(len(factors))
        w = 0.26
        rc_vals = [colar[f"c={c}"]["reconstruction_ceiling"]["step_exact_accuracy"] * 100.0 for c in factors]
        tf_vals = [colar[f"c={c}"]["teacher_forced"]["step_exact_accuracy"] * 100.0 for c in factors]
        fr_vals = [colar[f"c={c}"]["free_running"]["step_exact_accuracy"] * 100.0 for c in factors]
        ax_c.bar(x - w, rc_vals, width=w, label="Recon Ceiling", color="#2b5c8f")
        ax_c.bar(x, tf_vals, width=w, label="Teacher-Forced", color="#2e8b57")
        ax_c.bar(x + w, fr_vals, width=w, label="Free-Running", color="#c0392b")
        ax_c.set_xticks(x)
        ax_c.set_xticklabels([f"c={c}" for c in factors])
        ax_c.legend(loc="upper right", frameon=True)
    ax_c.set_title("CoLaR Probe (c ∈ {1, 2, 5})")
    ax_c.set_xlabel("CoLaR Compression Factor c")
    ax_c.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "phase1_recon_vs_gen.png", dpi=200)
    fig.savefig(out_dir / "phase1_recon_vs_gen.pdf")
    print("\nSaved -> outputs/phase1_recon_vs_gen.{png,pdf}")


if __name__ == "__main__":
    main()
