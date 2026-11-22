import os
import torch
import numpy as np
from dlm.data import make_task
from dlm.eval.accuracy import load_generator, eval_concept_generator

device = "cuda" if torch.cuda.is_available() else "cpu"
seeds = [12345, 12346, 12347]
depths = [8, 10, 12, 16]

# Instantiate with max_depth=16 so task.sample allows D > 8
task = make_task("bbh_shuffled_objects", n=5, max_depth=16, answer_mode="trace")

models = [
    ("Vanilla Latent-AR (R=8)", "outputs/gen_ar_bbh_r8.pt"),
    ("EntitySlotBuffer (R=8)",  "outputs/gen_slot_ar_bbh_r8_25k.pt"),
    ("GRUMemoryBuffer (R=8)",   "outputs/gen_gru_ar_bbh_r8_25k.pt"),
]

lines = [
    "=" * 72,
    " BBH R=8 OOD Depth Extrapolation (N=2048 x 3 Seeds, D_train <= 8)",
    "=" * 72,
    f"{'Model':<26} | " + " | ".join([f"D={d:<8}" for d in depths]),
    "-" * 72,
]

for name, path in models:
    if not os.path.exists(path):
        print(f"Skipping missing checkpoint: {path}")
        continue
    gen, ae, cond_ae, meta = load_generator(path, device=device)
    scores = {d: [] for d in depths}
    for s in seeds:
        res = eval_concept_generator(gen, ae, task, depths=depths, n_per_depth=2048, seed=s, device=device)
        for d in depths:
            scores[d].append(res[d] * 100)
    row = f"{name:<26} | " + " | ".join([f"{np.mean(scores[d]):5.2f}+/-{np.std(scores[d]):4.2f}" for d in depths])
    lines.append(row)

lines.append("=" * 72)
report = "\n".join(lines)
print("\n" + report)
with open("outputs/bbh_r8_ood_depth_multiseed.txt", "w") as f:
    f.write(report + "\n")
