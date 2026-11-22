#!/bin/bash
#SBATCH --job-name=slot_pilot
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -A general
#SBATCH -p regular
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -e

module load cuda/12.6
source ~/.venvs/dlm/bin/activate
cd $HOME/what-makes-a-good-latent-thought

mkdir -p outputs logs

# Bypass torch._native Triton JIT outer-product compilation (avoids missing Python.h)
export TORCH_DISABLE_NATIVE_JIT=1
# Flush stdout immediately so logs stream in real time
export PYTHONUNBUFFERED=1

python -m dlm.train.train_generator \
  --ae outputs/ae_bbh_r8.pt \
  --gen ar \
  --slot-buffer \
  --task bbh_shuffled_objects \
  --task-n 5 \
  --task-max-depth 8 \
  --answer-mode trace \
  --depths 1 2 4 8 \
  --steps 5000 \
  --batch-size 64 \
  --lr 3e-4 \
  --out outputs/gen_slot_ar_pilot_r8.pt
