#!/bin/bash
#SBATCH --job-name=tok_bbh
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -A general
#SBATCH -p regular
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

module load cuda/12.6
source ~/.venvs/dlm/bin/activate
cd $HOME/what-makes-a-good-latent-thought

mkdir -p outputs logs

python -m dlm.train.train_token \
  --gen token_ar \
  --task bbh_shuffled_objects \
  --task-n 5 \
  --task-max-depth 8 \
  --answer-mode trace \
  --depths 1 2 4 8 \
  --d 256 \
  --layers 12 \
  --heads 8 \
  --steps 5000 \
  --batch-size 64 \
  --out outputs/token_ar_bbh.pt
