#!/bin/bash
#SBATCH --job-name=gru_ctrl
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -A general
#SBATCH -p regular
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=03:30:00

set -e

module load cuda/12.6
source ~/.venvs/dlm/bin/activate
cd $HOME/what-makes-a-good-latent-thought

mkdir -p outputs logs

# Bypass torch._native Triton JIT outer-product compilation
export TORCH_DISABLE_NATIVE_JIT=1
# Unbuffered logging for real-time monitoring
export PYTHONUNBUFFERED=1

echo "=== Stage 1: Training Parameter-Matched GRU-AR (25,000 Steps, BS=128) ==="
python -m dlm.train.train_generator \
  --ae outputs/ae_bbh_r8.pt \
  --gen ar \
  --gru-buffer \
  --task bbh_shuffled_objects \
  --task-n 5 \
  --task-max-depth 8 \
  --answer-mode trace \
  --depths 1 2 4 8 \
  --steps 25000 \
  --batch-size 128 \
  --lr 3e-4 \
  --out outputs/gen_gru_ar_bbh_r8_25k.pt

echo "=== Stage 2: Multi-Seed Evaluation Across Reasoning Depths ==="
python -c "
import numpy as np
from dlm.eval.accuracy import load_generator, eval_concept_generator, _task_from_meta

gen, ae, cond_ae, meta = load_generator('outputs/gen_gru_ar_bbh_r8_25k.pt', device='cpu')
task = _task_from_meta(meta)
seeds = [12345, 12346, 12347]
depths = [1, 2, 4, 8]

print('\n' + '=' * 55)
print(' BBH GRU-AR Control (R=8) 25k Sweep (N=2048 x 3 Seeds)')
print('=' * 55)
with open('outputs/bbh_gru_ar_25k_multiseed.txt', 'w') as f:
    for d in depths:
        records = [eval_concept_generator(gen, ae, task, depths=[d], n_per_depth=2048, seed=s, device='cpu')[d] * 100 for s in seeds]
        line = f'Depth {d}: {np.mean(records):6.2f}% +/- {np.std(records):4.2f}%'
        print(line)
        f.write(line + '\n')
print('=' * 55)
"
