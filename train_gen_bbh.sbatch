#!/bin/bash
#SBATCH --job-name=gen_bbh
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

python -m dlm.train.train_generator \
  --ae outputs/ae_bbh_r2.pt \
  --gen ar \
  --task bbh_shuffled_objects \
  --task-n 5 \
  --task-max-depth 8 \
  --answer-mode trace \
  --depths 1 2 4 8 \
  --steps 5000 \
  --batch-size 64 \
  --lr 3e-4 \
  --out outputs/gen_ar_bbh_r2.pt
