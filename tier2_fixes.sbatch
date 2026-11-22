#!/bin/bash
#SBATCH --job-name=t2_fixes
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH -A general
#SBATCH -p regular
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-2%1

set -e
module load cuda/12.6
source ~/.venvs/dlm/bin/activate
cd $HOME/what-makes-a-good-latent-thought

export TORCH_DISABLE_NATIVE_JIT=1
export PYTHONUNBUFFERED=1

case $SLURM_ARRAY_TASK_ID in
  0)
    # Task 2 Affine Z5 Multi-Seed Verification (using --group affine)
    python -m dlm.train.train_ae \
      --task function_compose --group affine --task-n 5 --task-max-depth 32 --answer-mode trace \
      --R 16 --slots 4 --depths 2 4 8 16 32 --uniform-depths --depth-power 1.0 \
      --beta 0.05 --steps 15000 --batch-size 64 --gate-recon 0.95 \
      --out outputs/ae_z5_r16_k4.pt

    python -m dlm.train.train_generator \
      --ae outputs/ae_z5_r16_k4.pt --gen ar \
      --task function_compose --group affine --task-n 5 --task-max-depth 32 --answer-mode trace \
      --depths 2 4 8 16 32 --steps 25000 --batch-size 128 --lr 3e-4 \
      --out outputs/gen_ar_z5_r16_k4.pt

    python -c "
import numpy as np
from dlm.eval.accuracy import load_generator, eval_concept_generator, _task_from_meta
gen, ae, _, meta = load_generator('outputs/gen_ar_z5_r16_k4.pt', device='cuda')
task = _task_from_meta(meta)
with open('outputs/affine_z5_multiseed.txt', 'w') as f:
    for d in [2, 4, 8, 16, 32]:
        rec = [eval_concept_generator(gen, ae, task, depths=[d], n_per_depth=2048, seed=s, device='cuda')[d]*100 for s in [12345,12346,12347]]
        line = f'Depth {d}: {np.mean(rec):6.2f}% +/- {np.std(rec):4.2f}%'
        print(line); f.write(line + '\n')
"
    ;;
  1)
    # S5 D=32 AE (30k steps, beta=0.02 to pass D=32 tail gate) + 75k Continuous Diffusion
    python -m dlm.train.train_ae \
      --task function_compose --group s5 --task-n 5 --task-max-depth 32 --answer-mode trace \
      --R 16 --slots 4 --depths 2 4 8 16 32 --uniform-depths --depth-power 1.0 \
      --beta 0.02 --steps 30000 --batch-size 64 --gate-recon 0.95 \
      --out outputs/ae_s5_r16_k4_d32.pt

    python -m dlm.train.train_generator \
      --ae outputs/ae_s5_r16_k4_d32.pt --gen diffusion \
      --task function_compose --group s5 --task-n 5 --task-max-depth 32 --answer-mode trace \
      --depths 2 4 8 16 32 --steps 75000 --batch-size 128 --lr 3e-4 \
      --out outputs/gen_diff_s5_r16_k4_d32_75k.pt

    python -c "
from dlm.eval.accuracy import load_generator, eval_concept_generator, _task_from_meta
gen, ae, _, meta = load_generator('outputs/gen_diff_s5_r16_k4_d32_75k.pt', device='cuda')
task = _task_from_meta(meta)
with open('outputs/s5_diff_75k_results.txt', 'w') as f:
    for T in [1, 16, 64, 256]:
        res = eval_concept_generator(gen, ae, task, depths=[2, 4, 8, 16, 32], n_per_depth=512, T=T, device='cuda')
        line = f'T={T}: ' + ' '.join([f'D{d}={res[d]*100:.2f}%' for d in [2,4,8,16,32]])
        print(line); f.write(line + '\n')
"
    ;;
  2)
    # Clean up broken OOD log and run internal memory-state causal swap on Slot vs GRU
    rm -f outputs/bbh_r8_ood_depth_multiseed.txt
    python -c "
import torch, numpy as np
from dlm.data import make_task
from dlm.eval.accuracy import load_generator

device = 'cuda'
task = make_task('bbh_shuffled_objects', n=5, max_depth=8, answer_mode='trace')
batch = task.sample(512, depth=8, rng=np.random.default_rng(12345))
prompt_t = torch.as_tensor(batch.prompt, device=device)

lines = ['='*72, ' Internal Memory-State Causal Intervention (BBH R=8, D=8, N=512)', '='*72]
for name, path in [('EntitySlotBuffer (R=8)', 'outputs/gen_slot_ar_bbh_r8_25k.pt'), ('GRUMemoryBuffer (R=8)', 'outputs/gen_gru_ar_bbh_r8_25k.pt')]:
    gen, ae, cond_ae, _ = load_generator(path, device=device)
    enc = cond_ae if cond_ae is not None else ae
    with torch.no_grad():
        cond = enc.encode(prompt_t, sample=False)
        z_clean = gen.sample(cond, n_target=5)
        pred_clean = ae.decode(z_clean, out_len=task.answer_len).argmax(-1).cpu().numpy()
        acc_clean = task.check(batch.prompt, pred_clean).mean() * 100

        # Hook internal buffer after emission step 2 to permute slot 0 <-> slot 1 (or matching GRU slice)
        step_counter = {'n': 0}
        def hook_fn(mod, inp, out):
            step_counter['n'] += 1
            if step_counter['n'] == 3:
                if out.dim() == 3 and out.shape[1] >= 2:
                    edited = out.clone()
                    edited[:, 0], edited[:, 1] = out[:, 1].clone(), out[:, 0].clone()
                    return edited
                elif out.dim() == 2:
                    edited = out.clone()
                    w = out.shape[-1] // 5
                    edited[:, :w], edited[:, w:2*w] = out[:, w:2*w].clone(), out[:, :w].clone()
                    return edited
            return out

        target_mod = getattr(gen, 'slot_buffer', getattr(gen, 'gru_buffer', None))
        h = target_mod.register_forward_hook(hook_fn) if target_mod is not None else None
        z_edit = gen.sample(cond, n_target=5)
        if h is not None: h.remove()
        pred_edit = ae.decode(z_edit, out_len=task.answer_len).argmax(-1).cpu().numpy()

        fc, fe = pred_clean[:, -5:], pred_edit[:, -5:]
        swapped = ((fe[:, 0] == fc[:, 1]) & (fe[:, 1] == fc[:, 0])).mean() * 100
        unchanged = (fe == fc).all(axis=1).mean() * 100
        lines.append(f'{name:<24} | Clean: {acc_clean:6.2f}% | Entity 0<->1 Swapped: {swapped:6.2f}% | Unchanged: {unchanged:6.2f}%')
lines.append('='*72)
out_str = '\n'.join(lines)
print(out_str)
with open('outputs/bbh_r8_causal_slot_swap.txt', 'w') as f:
    f.write(out_str + '\n')
"
    ;;
esac
