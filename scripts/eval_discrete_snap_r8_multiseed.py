import torch
import numpy as np
from dlm.eval.accuracy import load_generator, _task_from_meta

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_path = "outputs/gen_ar_bbh_r8.pt"
seeds = [12345, 12346, 12347]
N_TOTAL = 2048
CHUNK = 256
DEPTH = 8

gen, ae, cond_ae, meta = load_generator(ckpt_path, device=device)
task = _task_from_meta(meta)
m_a = meta.get("m_answer", task.answer_len // ae.R)  # 40 // 8 = 5 latents at D=8
encoder = cond_ae if cond_ae is not None else ae

def decode_latents(z_tensor):
    B, k, d = z_tensor.shape
    if k < m_a:
        pad = torch.zeros(B, m_a - k, d, device=device, dtype=z_tensor.dtype)
        full_z = torch.cat([z_tensor, pad], dim=1)
    else:
        full_z = z_tensor
    return ae.decode(full_z, out_len=task.answer_len).argmax(dim=-1)

@torch.no_grad()
def eval_seed(seed):
    rng = np.random.default_rng(seed)
    scores = {"free": [], "snap_1": [], "snap_2": [], "tf_last": []}

    for _ in range(0, N_TOTAL, CHUNK):
        batch = task.sample(CHUNK, depth=DEPTH, rng=rng)
        prompt_t = torch.as_tensor(batch.prompt, device=device)
        gold_t = torch.as_tensor(batch.answer, device=device)
        cond = encoder.encode(prompt_t, sample=False)
        gold_latents = ae.encode(gold_t, sample=False)
        B, _, d = cond.shape
        bos = gen.bos.expand(B, 1, d)

        # 1. Free-Running Baseline
        z_free = gen.sample(cond, n_target=m_a)
        pred_free = decode_latents(z_free).cpu().numpy()
        scores["free"].append(task.check(batch.prompt, pred_free))

        # 2 & 3. Periodic Discrete Snapping (every 1 emission & every 2 emissions)
        for mode, snap_every in [("snap_1", 1), ("snap_2", 2)]:
            seq = torch.cat([cond, bos], dim=1)
            latents = []
            for step in range(m_a):
                h = gen._hidden(seq)[:, -1, :]
                nxt = gen.head.sample(h, steps=gen.head_inner_steps)
                latents.append(nxt.unsqueeze(1))
                k = len(latents)
                if k % snap_every == 0 or k == m_a:
                    curr_z = torch.cat(latents, dim=1)
                    toks_k = decode_latents(curr_z)
                    snapped_k = ae.encode(toks_k, sample=False)[:, :k, :]
                    latents = [snapped_k[:, i:i+1, :] for i in range(k)]
                    seq = torch.cat([cond, bos, snapped_k], dim=1)
                else:
                    seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
            pred_snap = decode_latents(torch.cat(latents, dim=1)).cpu().numpy()
            scores[mode].append(task.check(batch.prompt, pred_snap))

        # 4. Teacher-Forced Final Emission Ceiling (Step 5 / Swaps 7-8)
        tf_seq = torch.cat([cond, bos, gold_latents[:, :m_a - 1]], dim=1)
        h_tf = gen._hidden(tf_seq)[:, -1, :]
        nxt_tf = gen.head.sample(h_tf, steps=gen.head_inner_steps)
        tf_latents = torch.cat([gold_latents[:, :m_a - 1], nxt_tf.unsqueeze(1)], dim=1)
        pred_tf = decode_latents(tf_latents).cpu().numpy()
        scores["tf_last"].append(task.check(batch.prompt, pred_tf))

    return {k: float(np.concatenate(v).mean()) * 100 for k, v in scores.items()}

if __name__ == "__main__":
    print(f"Running R=8, D=8 Discrete-Snapping Multi-Seed Eval on {device}...")
    agg = {"free": [], "snap_1": [], "snap_2": [], "tf_last": []}
    for s in seeds:
        res = eval_seed(s)
        print(f"Seed {s}: Free={res['free']:.2f}% | Snap-1={res['snap_1']:.2f}% | Snap-2={res['snap_2']:.2f}% | TF-Last={res['tf_last']:.2f}%")
        for k in agg:
            agg[k].append(res[k])

    labels = [
        ("Free-Running Baseline (No Snapping)", "free"),
        ("Discrete-Snapped (Every 1 Emission)", "snap_1"),
        ("Discrete-Snapped (Every 2 Emissions)", "snap_2"),
        ("Teacher-Forced Ceiling (Final Emission)", "tf_last"),
    ]
    out_lines = [
        "=" * 64,
        " BBH R=8 (K=1) Depth 8 Discrete-Snapping Probe (N=2048 x 3 Seeds)",
        "=" * 64,
    ]
    for title, key in labels:
        m, sd = np.mean(agg[key]), np.std(agg[key])
        out_lines.append(f"{title:<42}: {m:6.2f}% +/- {sd:4.2f}%")
    out_lines.append("=" * 64)

    report = "\n".join(out_lines)
    print("\n" + report)
    with open("outputs/bbh_r8_d8_snapping_multiseed.txt", "w") as f:
        f.write(report + "\n")
