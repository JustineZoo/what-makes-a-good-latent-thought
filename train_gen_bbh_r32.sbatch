#!/bin/bash
#SBATCH --job-name=gen_bbh_r32
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

python -m dlm.train.train_generator \
  --ae outputs/ae_bbh_r32_k8.pt \
  --gen ar \
  --task bbh_shuffled_objects \
  --task-n 5 \
  --task-max-depth 8 \
  --answer-mode trace \
  --depths 1 2 4 8 \
  --steps 25000 \
  --batch-size 128 \
  --lr 3e-4 \
  --out outputs/gen_ar_bbh_r32_k8.pt
