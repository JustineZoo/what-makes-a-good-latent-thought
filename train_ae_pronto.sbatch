#!/bin/bash
#SBATCH --job-name=ae_pronto
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -A general
#SBATCH -p regular
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

module load cuda/12.6
source ~/.venvs/dlm/bin/activate
cd $HOME/what-makes-a-good-latent-thought

mkdir -p outputs logs

python -m dlm.train.train_ae \
  --task pronto_qa \
  --task-max-depth 6 \
  --answer-mode trace \
  --R 2 \
  --depths 1 2 4 6 \
  --uniform-depths \
  --depth-power 1.0 \
  --beta 0.05 \
  --steps 5000 \
  --batch-size 64 \
  --gate-recon 0.95 \
  --out outputs/ae_pronto_r2.pt
