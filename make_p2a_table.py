import json, os, numpy as np

systems = [
    ("colar", r"Vanilla \textsc{CoLaR} ($c=5$)", 3),
    ("colar_c1", r"+ \textbf{C1} (Step-Aligned Chunks, $c'\le 10$)", 1),
    ("colar_c2", r"+ \textbf{C2} (Factorized Sub-Slots, $K=4$)", 1),
    ("colar_c1_c2", r"+ \textbf{C1} + \textbf{C2}", 3),
    ("colar_c1_c2_c3a", r"+ \textbf{C1} + \textbf{C2} + \textbf{C3a} ($T=4$ Refine)", 1),
]

rows = []
for sys_id, label, n_seeds in systems:
    ans_list, step_list, svamp_list, hard_list, multi_list, prosqa_list = [], [], [], [], [], []
    for s in range(n_seeds):
        d = f"outputs/phase2a_{sys_id}_s{s}"
        with open(os.path.join(d, "metrics_gsm8k.json")) as f:
            gsm = json.load(f)
        with open(os.path.join(d, "metrics_ood.json")) as f:
            ood = json.load(f)
        ans_list.append(gsm["answer_accuracy"] * 100)
        step_list.append(gsm["step_exact_accuracy"] * 100)
        svamp_list.append(ood.get("svamp", {}).get("answer_accuracy", 0.0) * 100)
        hard_list.append(ood.get("gsm_hard", {}).get("answer_accuracy", 0.0) * 100)
        multi_list.append(ood.get("multiarith", {}).get("answer_accuracy", 0.0) * 100)
        if "prosqa" in ood and "answer_accuracy" in ood["prosqa"]:
            prosqa_list.append(ood["prosqa"]["answer_accuracy"] * 100)
        
    seed_str = r"\textbf{3 train}" if n_seeds == 3 else r"1 seed (\texttt{s0})"
    if n_seeds > 1:
        ans_fmt = f"${np.mean(ans_list):.2f} \\pm {np.std(ans_list, ddof=1):.2f}\\%$"
        step_fmt = f"${np.mean(step_list):.2f} \\pm {np.std(step_list, ddof=1):.2f}\\%$"
        svamp_fmt = f"${np.mean(svamp_list):.2f} \\pm {np.std(svamp_list, ddof=1):.2f}\\%$"
        hard_fmt = f"${np.mean(hard_list):.2f} \\pm {np.std(hard_list, ddof=1):.2f}\\%$"
        multi_fmt = f"${np.mean(multi_list):.2f} \\pm {np.std(multi_list, ddof=1):.2f}\\%$"
        if len(prosqa_list) == n_seeds:
            prosqa_fmt = f"${np.mean(prosqa_list):.2f} \\pm {np.std(prosqa_list, ddof=1):.2f}\\%$"
        elif len(prosqa_list) > 0:
            prosqa_fmt = f"${prosqa_list[0]:.2f}\\%$"
        else:
            prosqa_fmt = "---"
    else:
        ans_fmt = f"${ans_list[0]:.2f}\\%$"
        step_fmt = f"${step_list[0]:.2f}\\%$"
        svamp_fmt = f"${svamp_list[0]:.2f}\\%$"
        hard_fmt = f"${hard_list[0]:.2f}\\%$"
        multi_fmt = f"${multi_list[0]:.2f}\\%$"
        prosqa_fmt = f"${prosqa_list[0]:.2f}\\%$" if prosqa_list else "---"
        
    rows.append(f"    {label} & {seed_str} & {ans_fmt} & {step_fmt} & {svamp_fmt} & {hard_fmt} & {multi_fmt} & {prosqa_fmt} \\\\")

body = "\n".join(rows)
table_tex = rf"""\begin{{table*}}[t]
  \caption{{\textbf{{Phase 2a Core Ablations on \textsc{{Llama-3.2-1B-Instruct}} (Matched Latent Budget $c=5$)}}. All models train for $4{{,}}521$ updates ($\approx 27.5\text{{M}}$ tokens) using a frozen $R=4$ LoRA codec. All continuous free-running trajectories collapse to the $\sim 3.5\%$--$5.5\%$ band, demonstrating that architectural restructuring alone cannot halt continuous autoregressive drift without discrete grounding.}}
  \label{{tab:phase2a_ablations}}
  \centering
  \small
  \setlength{{\tabcolsep}}{{5pt}}
  \begin{{tabular}}{{lccccccc}}
    \toprule
    \textbf{{System / Configuration}} & \textbf{{Protocol}} & \textbf{{GSM8K Test Ans}} & \textbf{{Step Exact Acc}} & \textbf{{SVAMP}} & \textbf{{GSM-Hard}} & \textbf{{MultiArith}} & \textbf{{ProsQA}} \\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}
\end{{table*}}"""

with open("outputs/table_phase2a_ablations.tex", "w") as f:
    f.write(table_tex)

print(table_tex)
