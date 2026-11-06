#!/bin/bash
#SBATCH --job-name=ae_sweep
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH -A general
#SBATCH -p regular
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-3%1

module load cuda/12.6
source ~/.venvs/dlm/bin/activate
cd $HOME/what-makes-a-good-latent-thought

RATIOS=(1 4 8 16)
R=${RATIOS[$SLURM_ARRAY_TASK_ID]}

echo "Training BBH Codec at R=$R..."
python -m dlm.train.train_ae \
  --task bbh_shuffled_objects \
  --task-n 5 \
  --task-max-depth 8 \
  --answer-mode trace \
  --R $R \
  --depths 1 2 4 8 \
  --uniform-depths \
  --depth-power 1.0 \
  --beta 0.05 \
  --steps 5000 \
  --batch-size 64 \
  --gate-recon 0.95 \
  --out outputs/ae_bbh_r${R}.pt
