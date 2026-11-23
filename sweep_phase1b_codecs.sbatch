#!/bin/bash
#SBATCH --job-name=p1b_codecs
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH -A allaccess
#SBATCH -p full
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-5%2

set -e
module load cuda/12.6
source ~/.venvs/dlm/bin/activate
cd $HOME/what-makes-a-good-latent-thought
mkdir -p logs outputs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# Array mapping: R=1, 2, 4 (seeds 0,1,2), 8
case $SLURM_ARRAY_TASK_ID in
  0) R=1; SEED=0 ;;
  1) R=2; SEED=0 ;;
  2) R=4; SEED=0 ;;
  3) R=4; SEED=1 ;;
  4) R=4; SEED=2 ;;
  5) R=8; SEED=0 ;;
esac

OUT="outputs/phase1b_codec_r${R}_s${SEED}"

echo "================================================================="
echo " Starting Phase 1c Stage 1: LoRA Codec (R=${R}, Seed=${SEED})"
echo " Node: $(hostname) | Partition: full | Account: allaccess"
echo " Target Output: ${OUT}"
echo "================================================================="

python -m dlm.train.train_llm_codec \
  --model-name meta-llama/Llama-3.2-1B-Instruct \
  --R $R \
  --d-z 256 \
  --lora-r 128 \
  --lora-alpha 32 \
  --epochs 3 \
  --batch-size 256 \
  --micro-batch-size 16 \
  --lr 1e-4 \
  --gate-threshold 0.95 \
  --seed $SEED \
  --out-dir $OUT

echo "================================================================="
echo " Completed Codec R=${R} Seed=${SEED}"
echo "================================================================="
