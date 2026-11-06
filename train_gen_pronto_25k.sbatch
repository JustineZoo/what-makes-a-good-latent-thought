#!/bin/bash
#SBATCH --job-name=gen_pronto_25k
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
  --ae outputs/ae_pronto_r2.pt \
  --gen ar \
  --task pronto_qa \
  --task-max-depth 6 \
  --answer-mode trace \
  --depths 1 2 4 6 \
  --steps 25000 \
  --batch-size 128 \
  --lr 3e-4 \
  --out outputs/gen_ar_pronto_r2_25k.pt
