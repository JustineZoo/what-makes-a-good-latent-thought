#!/bin/bash
#SBATCH --job-name=t3_train_seeds
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH -A general
#SBATCH -p regular
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-9%1

set -e
module load cuda/12.6
source ~/.venvs/dlm/bin/activate
cd $HOME/what-makes-a-good-latent-thought

export TORCH_DISABLE_NATIVE_JIT=1
export PYTHONUNBUFFERED=1

# 5 headline BBH configurations x 2 new training seeds (1 and 2; seed 0 is your existing checkpoint)
AES=(
  "outputs/ae_bbh_r8.pt" "outputs/ae_bbh_r8.pt"
  "outputs/ae_bbh_r16_k4.pt" "outputs/ae_bbh_r16_k4.pt"
  "outputs/ae_bbh_r32_k8.pt" "outputs/ae_bbh_r32_k8.pt"
  "outputs/ae_bbh_r8.pt" "outputs/ae_bbh_r8.pt"
  "outputs/ae_bbh_r8.pt" "outputs/ae_bbh_r8.pt"
)
EXTRA_FLAGS=(
  "" ""
  "" ""
  "" ""
  "--slot-buffer --n-slots 5" "--slot-buffer --n-slots 5"
  "--gru-buffer --n-slots 5" "--gru-buffer --n-slots 5"
)
NAMES=(
  "bbh_r8_s1" "bbh_r8_s2"
  "bbh_r16_k4_s1" "bbh_r16_k4_s2"
  "bbh_r32_k8_s1" "bbh_r32_k8_s2"
  "bbh_slot_r8_s1" "bbh_slot_r8_s2"
  "bbh_gru_r8_s1" "bbh_gru_r8_s2"
)
SEEDS=(1 2 1 2 1 2 1 2 1 2)

AE=${AES[$SLURM_ARRAY_TASK_ID]}
FLAGS=${EXTRA_FLAGS[$SLURM_ARRAY_TASK_ID]}
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python -m dlm.train.train_generator \
  --ae $AE --gen ar $FLAGS --seed $SEED \
  --task bbh_shuffled_objects --task-n 5 --task-max-depth 8 --answer-mode trace \
  --depths 1 2 4 8 --steps 25000 --batch-size 128 --lr 3e-4 \
  --out outputs/gen_${NAME}.pt

python -c "
import numpy as np
from dlm.eval.accuracy import load_generator, eval_concept_generator, _task_from_meta
gen, ae, _, meta = load_generator('outputs/gen_${NAME}.pt', device='cuda')
task = _task_from_meta(meta)
with open('outputs/${NAME}_eval.txt', 'w') as f:
    for d in [1, 2, 4, 8]:
        acc = eval_concept_generator(gen, ae, task, depths=[d], n_per_depth=2048, seed=12345, device='cuda')[d]*100
        line = f'Depth {d}: {acc:6.2f}%'
        print(line); f.write(line + '\n')
"
