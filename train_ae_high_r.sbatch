#!/bin/bash
#SBATCH --job-name=ae_high_r
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -A general
#SBATCH -p regular
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:45:00

module load cuda/12.6
source ~/.venvs/dlm/bin/activate
cd $HOME/what-makes-a-good-latent-thought

mkdir -p outputs logs

echo "=== 1. Training BBH Codec R=8 (10,000 Steps) ==="
python -m dlm.train.train_ae \
  --task bbh_shuffled_objects \
  --task-n 5 \
  --task-max-depth 8 \
  --answer-mode trace \
  --R 8 \
  --depths 1 2 4 8 \
  --uniform-depths \
  --depth-power 1.0 \
  --beta 0.05 \
  --steps 10000 \
  --batch-size 64 \
  --gate-recon 0.95 \
  --out outputs/ae_bbh_r8.pt

echo "=== 2. Training BBH Factorized Codec R=16 K=4 (15,000 Steps) ==="
python -m dlm.train.train_ae \
  --task bbh_shuffled_objects \
  --task-n 5 \
  --task-max-depth 8 \
  --answer-mode trace \
  --R 16 \
  --slots 4 \
  --depths 1 2 4 8 \
  --uniform-depths \
  --depth-power 1.0 \
  --beta 0.05 \
  --steps 15000 \
  --batch-size 64 \
  --gate-recon 0.95 \
  --out outputs/ae_bbh_r16_k4.pt
