#!/bin/bash
#SBATCH --job-name=p1b_gens
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH -A allaccess
#SBATCH -p full
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-3%2

set -e
module load cuda/12.6
source ~/.venvs/dlm/bin/activate
cd $HOME/what-makes-a-good-latent-thought
mkdir -p logs outputs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

case $SLURM_ARRAY_TASK_ID in
  0) R=1 ;;
  1) R=2 ;;
  2) R=4 ;;
  3) R=8 ;;
esac

CODEC_CKPT="outputs/phase1b_codec_r${R}_s0/codec_final.pt"
OUT="outputs/phase1b_gen_r${R}_s0"

if [ ! -f "$CODEC_CKPT" ]; then
  echo "Error: Codec checkpoint $CODEC_CKPT not found! Run sweep_phase1b_codecs.sbatch first."
  exit 1
fi

echo "================================================================="
echo " Starting Phase 1c Stage 2: Flow-Matching Generator (R=${R})"
echo " Codec: ${CODEC_CKPT}"
echo " Node: $(hostname) | Partition: full | Account: allaccess"
echo " Target Output: ${OUT}"
echo "================================================================="

python -m dlm.train.train_llm_generator \
  --codec-ckpt "$CODEC_CKPT" \
  --epochs 3 \
  --batch-size 256 \
  --micro-batch-size 16 \
  --lr 1e-4 \
  --head-inner-steps 4 \
  --seed 0 \
  --out-dir "$OUT"

echo "================================================================="
echo " Completed Generator R=${R}"
echo "================================================================="
