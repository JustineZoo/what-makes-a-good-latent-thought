import os
import torch
import numpy as np
from collections import Counter
from dlm.eval.accuracy import load_generator, eval_concept_generator, _task_from_meta

device = "cuda" if torch.cuda.is_available() else "cpu"
seeds = [12345, 12346, 12347]

# --- A1. Internal Memory-State Causal Swap (EntitySlotBuffer vs GRUMemoryBuffer) ---
def eval_causal_memory_swap(ckpt_path, mode):
    gen, ae, cond_ae, meta = load_generator(ckpt_path, device=device)
    task = _task_from_meta(meta)
    rng = np.random.default_rng(12345)
    b = task.sample(2048, depth=8, rng=rng)
    prompt = torch.as_tensor(b.prompt, device=device)
    cond = (cond_ae or ae).encode(prompt, sample=False)
    B, _, d = cond.shape
    m_a = 5

    with torch.no_grad():
        # Clean rollout
        z_clean = gen.sample(cond, n_target=m_a)
        pred_clean = ae.decode(z_clean, out_len=task.answer_len).argmax(-1).cpu().numpy()
        clean_acc = task.check(b.prompt, pred_clean).mean() * 100

        # Intervened rollout: swap entity slots 0 <-> 1 (or equivalent GRU slice) at step t=2 (after Swap 4)
        seq = torch.cat([cond, gen.bos.expand(B, 1, d)], dim=1)
        state = gen.slot_buffer.init_slots(cond) if mode == "slot" else gen.gru_buffer.init_state(cond)
        zs = []
        for t in range(m_a):
            if t == 2:
                if mode == "slot":
                    s0, s1 = state[:, 0, :].clone(), state[:, 1, :].clone()
                    state[:, 0, :], state[:, 1, :] = s1, s0
                else:
                    w = d // 5
                    c0, c1 = state[:, :w].clone(), state[:, w:2*w].clone()
                    state[:, :w], state[:, w:2*w] = c1, c0
            h = gen._hidden(seq)[:, -1, :]
            if mode == "slot":
                h, state = gen.slot_buffer(h.unsqueeze(1), state)
                h = h.squeeze(1)
            else:
                h, state = gen.gru_buffer(h, state)
            z_t = gen.head.sample(h)
            zs.append(z_t)
            seq = torch.cat([seq, z_t.unsqueeze(1)], dim=1)

        pred_swap = ae.decode(torch.stack(zs, dim=1), out_len=task.answer_len).argmax(-1).cpu().numpy()

    # Check if final decoded balls at positions 0 and 1 swapped relative to gold
    gold_final = b.answer[:, 35:40]
    pred_final = pred_swap[:, 35:40]
    swapped_gold = gold_final.copy()
    swapped_gold[:, 0], swapped_gold[:, 1] = gold_final[:, 1], gold_final[:, 0]

    unchanged_rate = (pred_final == gold_final).all(axis=1).mean() * 100
    swapped_rate = (pred_final == swapped_gold).all(axis=1).mean() * 100
    return clean_acc, swapped_rate, unchanged_rate

def main():
    os.makedirs("outputs", exist_ok=True)
    if os.path.exists("outputs/gen_slot_ar_bbh_r8_25k.pt") and os.path.exists("outputs/gen_gru_ar_bbh_r8_25k.pt"):
        slot_clean, slot_swapped, slot_unchanged = eval_causal_memory_swap("outputs/gen_slot_ar_bbh_r8_25k.pt", "slot")
        gru_clean, gru_swapped, gru_unchanged = eval_causal_memory_swap("outputs/gen_gru_ar_bbh_r8_25k.pt", "gru")

        swap_lines = [
            "=" * 80,
            " Causal Mid-Rollout Internal Memory Intervention (BBH R=8, D=8, N=2048)",
            "=" * 80,
            f"EntitySlotBuffer (R=8)   | Clean: {slot_clean:6.2f}% | Entity 0<->1 Swapped: {slot_swapped:6.2f}% | Unchanged: {slot_unchanged:6.2f}%",
            f"GRUMemoryBuffer (R=8)    | Clean: {gru_clean:6.2f}% | Entity 0<->1 Swapped: {gru_swapped:6.2f}% | Unchanged: {gru_unchanged:6.2f}%",
            "=" * 80,
        ]
        print("\n".join(swap_lines))
        with open("outputs/bbh_r8_causal_slot_swap.txt", "w") as f:
            f.write("\n".join(swap_lines) + "\n")
    else:
        print("Memory checkpoints not found; skipping A1.")

    # --- A2. Upgrade diff_75k_bbh_results.txt to N=2048 x 3 Seeds ---
    if os.path.exists("outputs/gen_diff_bbh_r16_75k.pt"):
        gen_diff, ae_diff, cond_diff, meta_diff = load_generator("outputs/gen_diff_bbh_r16_75k.pt", device=device)
        task_bbh = _task_from_meta(meta_diff)
        diff_lines = [
            "=" * 64,
            " 75k Continuous Diffusion Sweep (BBH R=16, K=4, N=2048 x 3 Seeds)",
            "=" * 64,
        ]
        for T in [1, 16, 64, 256]:
            diff_lines.append(f"\n--- Denoising Budget: T = {T} ---")
            recs = {d: [] for d in [1, 2, 4, 8]}
            for s in seeds:
                res = eval_concept_generator(gen_diff, ae_diff, task_bbh, depths=[1, 2, 4, 8], n_per_depth=2048, seed=s, T=T, device=device, cond_ae=cond_diff)
                for d in [1, 2, 4, 8]:
                    recs[d].append(res[d] * 100)
            for d in [1, 2, 4, 8]:
                diff_lines.append(f"Depth {d}: {np.mean(recs[d]):6.2f}% +/- {np.std(recs[d]):4.2f}%")
        diff_lines.append("=" * 64)
        print("\n".join(diff_lines))
        with open("outputs/diff_75k_bbh_results.txt", "w") as f:
            f.write("\n".join(diff_lines) + "\n")
    else:
        print("Diffusion checkpoint not found; skipping A2.")

    # --- A3. Upgrade algebraic_manifold_audit.txt to N=2048 x 3 Seeds ---
    manifold_lines = [
        "=" * 68,
        " BBH Depth 8: Algebraic Manifold Departure Audit (N=2048 x 3 Seeds)",
        "=" * 68,
    ]
    checkpoints_to_audit = [("Monolithic R=8 (K=1)", "outputs/gen_ar_bbh_r8.pt"), ("Factorized R=32 (K=8)", "outputs/gen_ar_bbh_r32_k8.pt")]
    active_audits = [(lbl, cp) for lbl, cp in checkpoints_to_audit if os.path.exists(cp)]
    if active_audits:
        for label, ckpt in active_audits:
            gen, ae, cond_ae, meta = load_generator(ckpt, device=device)
            t = _task_from_meta(meta)
            m_a = max(1, -(-t.answer_len // ae.R))
            accs, off_rates, on_rates = [], [], []
            for s in seeds:
                b = t.sample(2048, depth=8, rng=np.random.default_rng(s))
                with torch.no_grad():
                    cond = (cond_ae or ae).encode(torch.as_tensor(b.prompt, device=device), sample=False)
                    pred = ae.decode(gen.sample(cond, n_target=m_a), out_len=t.answer_len).argmax(-1).cpu().numpy()
                ok = t.check(b.prompt, pred)
                accs.append(ok.mean() * 100)
                fails = pred[~ok, 35:40]
                off = sum(len(set(row.tolist())) < 5 for row in fails)
                on = len(fails) - off
                off_rates.append(off / max(1, len(fails)) * 100)
                on_rates.append(on / max(1, len(fails)) * 100)
            manifold_lines += [
                f"{label} (Depth 8, N=2048 x 3):",
                f"  Accuracy: {np.mean(accs):6.2f}% +/- {np.std(accs):4.2f}%",
                f"  - Off-Manifold (Duplicate Objects / Non-Bijective): {np.mean(off_rates):5.1f}% +/- {np.std(off_rates):4.1f}%",
                f"  - On-Manifold (Valid S5 Permutation, Wrong State) : {np.mean(on_rates):5.1f}% +/- {np.std(on_rates):4.1f}%",
                "",
            ]
        manifold_lines.append("=" * 68)
        print("\n".join(manifold_lines))
        with open("outputs/algebraic_manifold_audit.txt", "w") as f:
            f.write("\n".join(manifold_lines) + "\n")
    else:
        print("Manifold audit checkpoints not found; skipping A3.")


if __name__ == "__main__":
    main()

