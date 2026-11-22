#!/bin/bash
#SBATCH --job-name=r30_alig
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -A general
#SBATCH -p regular
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -e

module load cuda/12.6
source ~/.venvs/dlm/bin/activate
cd $HOME/what-makes-a-good-latent-thought

mkdir -p outputs logs

echo "=== Stage 1: Training State-Aligned Codec (R=30, K=6, d=384) ==="
python -m dlm.train.train_ae \
  --task bbh_shuffled_objects \
  --task-n 5 \
  --task-max-depth 8 \
  --answer-mode trace \
  --R 30 \
  --slots 6 \
  --d-concept 384 \
  --depths 1 2 4 8 \
  --uniform-depths \
  --depth-power 1.0 \
  --beta 0.05 \
  --steps 15000 \
  --batch-size 64 \
  --gate-recon 0.95 \
  --out outputs/ae_bbh_r30_k6.pt

echo "=== Stage 2: Training Latent-AR Generator on Certified Codec ==="
python -m dlm.train.train_generator \
  --ae outputs/ae_bbh_r30_k6.pt \
  --gen ar \
  --task bbh_shuffled_objects \
  --task-n 5 \
  --task-max-depth 8 \
  --answer-mode trace \
  --depths 1 2 4 8 \
  --steps 25000 \
  --batch-size 128 \
  --lr 3e-4 \
  --out outputs/gen_ar_bbh_r30_k6.pt

echo "=== Stage 3: Immediate Multi-Seed Evaluation ==="
python -c "
import numpy as np
from dlm.eval.accuracy import load_generator, eval_concept_generator, _task_from_meta

gen, ae, cond_ae, meta = load_generator('outputs/gen_ar_bbh_r30_k6.pt', device='cpu')
task = _task_from_meta(meta)
seeds = [12345, 12346, 12347]
depths = [1, 2, 4, 8]

print('\n' + '=' * 55)
print(' BBH R=30 (K=6) State-Aligned Sweep (N=2048 x 3 Seeds)')
print('=' * 55)
with open('outputs/bbh_r30_aligned_multiseed.txt', 'w') as f:
    for d in depths:
        records = [eval_concept_generator(gen, ae, task, depths=[d], n_per_depth=2048, seed=s, device='cpu')[d] * 100 for s in seeds]
        line = f'Depth {d}: {np.mean(records):6.2f}% +/- {np.std(records):4.2f}%'
        print(line)
        f.write(line + '\n')
print('=' * 55)
"
