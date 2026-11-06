#!/bin/bash
#SBATCH --job-name=ae_bbh_r32
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -A general
#SBATCH -p regular
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=01:30:00

module load cuda/12.6
source ~/.venvs/dlm/bin/activate
cd $HOME/what-makes-a-good-latent-thought

python -m dlm.train.train_ae \
  --task bbh_shuffled_objects \
  --task-n 5 \
  --task-max-depth 8 \
  --answer-mode trace \
  --R 32 \
  --slots 8 \
  --depths 1 2 4 8 \
  --uniform-depths \
  --depth-power 1.0 \
  --beta 0.05 \
  --steps 15000 \
  --batch-size 64 \
  --gate-recon 0.95 \
  --out outputs/ae_bbh_r32_k8.pt
