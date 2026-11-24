#!/bin/bash
#SBATCH --job-name=p2a_ablations
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH -A allaccess
#SBATCH -p full
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --array=0-2,5-8%7

set -e
module load cuda/12.6
source ~/.venvs/dlm/bin/activate
cd $HOME/what-makes-a-good-latent-thought
mkdir -p logs outputs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# Array task mapping
case $SLURM_ARRAY_TASK_ID in
  0) SYSTEM="colar"; SEED=0 ;;
  1) SYSTEM="colar_c1"; SEED=0 ;;
  2) SYSTEM="colar_c2"; SEED=0 ;;
  3) SYSTEM="colar_c1_c2"; SEED=0 ;;
  4) SYSTEM="colar_c1_c2_c3a"; SEED=0 ;;
  5) SYSTEM="colar_c1_c2"; SEED=1 ;;
  6) SYSTEM="colar_c1_c2"; SEED=2 ;;
  7) SYSTEM="colar"; SEED=1 ;;
  8) SYSTEM="colar"; SEED=2 ;;
esac

CODEC_CKPT="outputs/phase1b_codec_r4_s0/codec_final.pt"
OUT="outputs/phase2a_${SYSTEM}_s${SEED}"

echo "================================================================="
echo " Starting Phase 2a Ablation: ${SYSTEM} (Seed=${SEED})"
echo " Codec: ${CODEC_CKPT}"
echo " Node: $(hostname) | Partition: full | Account: allaccess"
echo " Target Output: ${OUT}"
echo "================================================================="

python -m dlm.train.train_phase2_ablation \
  --system "${SYSTEM}" \
  --codec-ckpt "${CODEC_CKPT}" \
  --seed "${SEED}" \
  --c 5 \
  --c-prime 10 \
  --K 4 \
  --refine-steps 4 \
  --epochs 3 \
  --batch-size 256 \
  --micro-batch-size 4 \
  --lr 1e-4 \
  --out-dir "${OUT}"

echo "================================================================="
echo " Completed Phase 2a Ablation: ${SYSTEM} (Seed=${SEED})"
echo "================================================================="
