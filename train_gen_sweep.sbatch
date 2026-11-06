#!/bin/bash
#SBATCH --job-name=gen_sweep
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH -A general
#SBATCH -p regular
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=03:30:00
#SBATCH --array=0-2%1

module load cuda/12.6
source ~/.venvs/dlm/bin/activate
cd $HOME/what-makes-a-good-latent-thought

AE_FILES=("outputs/ae_bbh_r4.pt" "outputs/ae_bbh_r8.pt" "outputs/ae_bbh_r16_k4.pt")
OUT_FILES=("outputs/gen_ar_bbh_r4.pt" "outputs/gen_ar_bbh_r8.pt" "outputs/gen_ar_bbh_r16_k4.pt")

AE=${AE_FILES[$SLURM_ARRAY_TASK_ID]}
OUT=${OUT_FILES[$SLURM_ARRAY_TASK_ID]}

echo "Training Latent-AR generator on $AE..."
python -m dlm.train.train_generator \
  --ae $AE \
  --gen ar \
  --task bbh_shuffled_objects \
  --task-n 5 \
  --task-max-depth 8 \
  --answer-mode trace \
  --depths 1 2 4 8 \
  --steps 25000 \
  --batch-size 128 \
  --lr 3e-4 \
  --out $OUT
