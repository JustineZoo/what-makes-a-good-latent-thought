#!/bin/bash
#SBATCH --job-name=p1a_nocot
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -A general
#SBATCH -p regular
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -e
module load cuda/12.6
source ~/.bashrc
source ~/.venvs/dlm/bin/activate
cd $HOME/what-makes-a-good-latent-thought

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

python -m dlm.train.train_llm_baseline \
  --model-name meta-llama/Llama-3.2-1B-Instruct \
  --mode no_cot \
  --epochs 3 \
  --batch-size 256 \
  --micro-batch-size 8 \
  --max-len 160 \
  --lr 1e-4 \
  --seed 0 \
  --out-dir outputs/phase1a_nocot_s0
