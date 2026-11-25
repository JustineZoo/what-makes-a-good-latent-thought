#!/bin/bash
#SBATCH --job-name=true_p2a
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH -A allaccess
#SBATCH -p full
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --array=0-7%8

set -e
module load cuda/12.6
source ~/.venvs/dlm/bin/activate
cd $HOME/what-makes-a-good-latent-thought
mkdir -p logs outputs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

case $SLURM_ARRAY_TASK_ID in
  0) SYSTEM="colar"; SEED=0 ;;
  1) SYSTEM="colar"; SEED=1 ;;
  2) SYSTEM="colar"; SEED=2 ;;
  3) SYSTEM="colar_c1"; SEED=0 ;;
  4) SYSTEM="colar_c2"; SEED=0 ;;
  5) SYSTEM="colar_c1_c2"; SEED=0 ;;
  6) SYSTEM="colar_c1_c2"; SEED=1 ;;
  7) SYSTEM="colar_c1_c2"; SEED=2 ;;
esac

OUT="outputs/true_p2a_${SYSTEM}_s${SEED}"

echo "================================================================="
echo " Starting True Phase 2a Native CoLaR: ${SYSTEM} (Seed=${SEED})"
echo " Node: $(hostname) | Partition: full | Target: ${OUT}"
echo "================================================================="

python -m dlm.train.train_phase2_colar \
  --system "${SYSTEM}" \
  --c 5 \
  --c-prime 7 \
  --K 4 \
  --epochs 3 \
  --batch-size 256 \
  --micro-batch-size 4 \
  --eval-batch-size 32 \
  --lr 1e-4 \
  --seed "${SEED}" \
  --out-dir "${OUT}"

echo "================================================================="
echo " Completed True Phase 2a: ${SYSTEM} (Seed=${SEED})"
echo "================================================================="
