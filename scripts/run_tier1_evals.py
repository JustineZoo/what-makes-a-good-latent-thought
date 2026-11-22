import os
import torch
import torch.nn.functional as F
import numpy as np
from dlm.data import make_task
from dlm.eval.accuracy import load_generator, eval_concept_generator, _task_from_meta

device = "cuda" if torch.cuda.is_available() else "cpu"
seeds = [12345, 12346, 12347]

def extend_pos_emb(gen, new_len=64):
    """Extend learned positional embeddings in-memory for OOD depth extrapolation."""
    if hasattr(gen, "pos") and hasattr(gen.pos, "pos"):
        old_pos = gen.pos.pos.data  # (1, L, D)
        if old_pos.shape[1] < new_len:
            interp = F.interpolate(
                old_pos.transpose(1, 2), size=new_len, mode="linear", align_corners=True
            ).transpose(1, 2)
            gen.pos.pos = torch.nn.Parameter(interp, requires_grad=False)

# 1. Aligned-Boundary (k=5, 40 tokens = exact 8 states) Multi-Seed Snapping Check
gen_r8, ae_r8, cond_r8, meta_r8 = load_generator("outputs/gen_ar_bbh_r8.pt", device=device)
task_r8 = _task_from_meta(meta_r8)
enc_r8 = cond_r8 if cond_r8 is not None else ae_r8
snap5_scores = []
with torch.no_grad():
    for s in seeds:
        rng = np.random.default_rng(s)
        hits = []
        for _ in range(0, 2048, 256):
            batch = task_r8.sample(256, depth=8, rng=rng)
            prompt_t = torch.as_tensor(batch.prompt, device=device)
            cond = enc_r8.encode(prompt_t, sample=False)
            z_free = gen_r8.sample(cond, n_target=5)
            toks = ae_r8.decode(z_free, out_len=task_r8.answer_len).argmax(dim=-1)
            snapped = ae_r8.encode(toks, sample=False)[:, :5, :]
            pred = ae_r8.decode(snapped, out_len=task_r8.answer_len).argmax(dim=-1).cpu().numpy()
            hits.append(task_r8.check(batch.prompt, pred))
        snap5_scores.append(float(np.concatenate(hits).mean()) * 100)
snap5_line = f"Discrete-Snapped (Trace-End / Exact 8 States): {np.mean(snap5_scores):6.2f}% +/- {np.std(snap5_scores):4.2f}%"
print(snap5_line)
with open("outputs/bbh_r8_d8_snapping_multiseed.txt", "a") as f:
    f.write(snap5_line + "\n")

# 2. OOD Depth Extrapolation (D in {8, 10, 12, 16})
depths = [8, 10, 12, 16]
task_ood = make_task("bbh_shuffled_objects", n=5, max_depth=16, answer_mode="trace")
models = [
    ("Vanilla Latent-AR (R=8)", "outputs/gen_ar_bbh_r8.pt"),
    ("EntitySlotBuffer (R=8)",  "outputs/gen_slot_ar_bbh_r8_25k.pt"),
    ("GRUMemoryBuffer (R=8)",   "outputs/gen_gru_ar_bbh_r8_25k.pt"),
]

ood_lines = [
    "=" * 76,
    " BBH R=8 OOD Depth Extrapolation (N=2048 x 3 Seeds, D_train <= 8)",
    "=" * 76,
    f"{'Model':<26} | " + " | ".join([f"D={d:<9}" for d in depths]),
    "-" * 76,
]
for name, path in models:
    if not os.path.exists(path):
        continue
    gen, ae, cond_ae, meta = load_generator(path, device=device)
    extend_pos_emb(gen, new_len=64)
    scores = {d: [] for d in depths}
    for s in seeds:
        res = eval_concept_generator(gen, ae, task_ood, depths=depths, n_per_depth=2048, seed=s, device=device)
        for d in depths:
            scores[d].append(res[d] * 100)
    ood_lines.append(
        f"{name:<26} | " + " | ".join([f"{np.mean(scores[d]):5.2f}+/-{np.std(scores[d]):4.2f}" for d in depths])
    )
ood_lines.append("=" * 76)
with open("outputs/bbh_r8_ood_depth_multiseed.txt", "w") as f:
    f.write("\n".join(ood_lines) + "\n")
print("\n".join(ood_lines))

# 3. Causal Mid-Rollout Intervention (EntitySlotBuffer vs GRUMemoryBuffer at D=8)
task_d8 = make_task("bbh_shuffled_objects", n=5, max_depth=8, answer_mode="trace")
rng = np.random.default_rng(12345)
batch = task_d8.sample(512, depth=8, rng=rng)
prompt_t = torch.as_tensor(batch.prompt, device=device)

causal_lines = [
    "=" * 76,
    " Causal Mid-Rollout Latent Intervention (BBH R=8, D=8, N=512)",
    "=" * 76,
]
for name, path in [
    ("EntitySlotBuffer (R=8)", "outputs/gen_slot_ar_bbh_r8_25k.pt"),
    ("GRUMemoryBuffer (R=8)",  "outputs/gen_gru_ar_bbh_r8_25k.pt"),
]:
    if not os.path.exists(path):
        continue
    gen, ae, cond_ae, meta = load_generator(path, device=device)
    enc = cond_ae if cond_ae is not None else ae
    with torch.no_grad():
        cond = enc.encode(prompt_t, sample=False)
        z_clean = gen.sample(cond, n_target=5)
        pred_clean = ae.decode(z_clean, out_len=task_d8.answer_len).argmax(-1).cpu().numpy()
        acc_clean = task_d8.check(batch.prompt, pred_clean).mean() * 100

        # Swap first two 1/5th coordinate slices at step >= 2 and decode counterfactual trace
        z_edit = z_clean.clone()
        w = z_edit.shape[-1] // 5
        s0 = z_edit[:, 2:, :w].clone()
        s1 = z_edit[:, 2:, w:2*w].clone()
        z_edit[:, 2:, :w] = s1
        z_edit[:, 2:, w:2*w] = s0
        pred_edit = ae.decode(z_edit, out_len=task_d8.answer_len).argmax(-1).cpu().numpy()

        final_clean = pred_clean[:, -5:]
        final_edit = pred_edit[:, -5:]
        swapped_01 = ((final_edit[:, 0] == final_clean[:, 1]) & (final_edit[:, 1] == final_clean[:, 0])).mean() * 100
        unchanged = (final_edit == final_clean).all(axis=1).mean() * 100
        causal_lines.append(
            f"{name:<24} | Clean Acc: {acc_clean:6.2f}% | Entity 0<->1 Swapped: {swapped_01:6.2f}% | Unchanged: {unchanged:6.2f}%"
        )

causal_lines.append("=" * 76)
with open("outputs/bbh_r8_causal_slot_swap.txt", "w") as f:
    f.write("\n".join(causal_lines) + "\n")
print("\n".join(causal_lines))
