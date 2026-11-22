#!/bin/bash
#SBATCH --job-name=t2_75k
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
    python -m dlm.train.train_generator \
      --ae outputs/ae_bbh_r8.pt --gen ar \
      --task bbh_shuffled_objects --task-n 5 --task-max-depth 8 --answer-mode trace \
      --depths 1 2 4 8 --steps 75000 --batch-size 128 --lr 3e-4 \
      --out outputs/gen_ar_bbh_r8_75k.pt

    python -c "
import numpy as np
from dlm.eval.accuracy import load_generator, eval_concept_generator, _task_from_meta
gen, ae, _, meta = load_generator('outputs/gen_ar_bbh_r8_75k.pt', device='cuda')
task = _task_from_meta(meta)
with open('outputs/bbh_ar_r8_75k_multiseed.txt', 'w') as f:
    for d in [1, 2, 4, 8]:
        rec = [eval_concept_generator(gen, ae, task, depths=[d], n_per_depth=2048, seed=s, device='cuda')[d]*100 for s in [12345,12346,12347]]
        line = f'Depth {d}: {np.mean(rec):6.2f}% +/- {np.std(rec):4.2f}%'
        print(line); f.write(line + '\n')
"
    ;;
  1)
    python -m dlm.train.train_generator \
      --ae outputs/ae_bbh_r32_k8.pt --gen ar \
      --task bbh_shuffled_objects --task-n 5 --task-max-depth 8 --answer-mode trace \
      --depths 1 2 4 8 --steps 75000 --batch-size 128 --lr 3e-4 \
      --out outputs/gen_ar_bbh_r32_75k.pt

    python -c "
import numpy as np
from dlm.eval.accuracy import load_generator, eval_concept_generator, _task_from_meta
gen, ae, _, meta = load_generator('outputs/gen_ar_bbh_r32_75k.pt', device='cuda')
task = _task_from_meta(meta)
with open('outputs/bbh_ar_r32_75k_multiseed.txt', 'w') as f:
    for d in [1, 2, 4, 8]:
        rec = [eval_concept_generator(gen, ae, task, depths=[d], n_per_depth=2048, seed=s, device='cuda')[d]*100 for s in [12345,12346,12347]]
        line = f'Depth {d}: {np.mean(rec):6.2f}% +/- {np.std(rec):4.2f}%'
        print(line); f.write(line + '\n')
"
    ;;
  2)
    python -m dlm.train.train_ae \
      --task function_compose --task-n 5 --task-max-depth 32 --answer-mode trace \
      --R 16 --slots 4 --depths 2 4 8 16 32 --uniform-depths --depth-power 1.0 \
      --beta 0.05 --steps 15000 --batch-size 64 --gate-recon 0.95 \
      --out outputs/ae_s5_r16_k4_d32.pt

    python -m dlm.train.train_generator \
      --ae outputs/ae_s5_r16_k4_d32.pt --gen diffusion \
      --task function_compose --task-n 5 --task-max-depth 32 --answer-mode trace \
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
esac
