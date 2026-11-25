#!/bin/bash
#SBATCH --job-name=p2a_prosqa
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -A allaccess
#SBATCH -p full
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -e
module load cuda/12.6
source ~/.venvs/dlm/bin/activate
cd $HOME/what-makes-a-good-latent-thought
mkdir -p logs outputs

echo "================================================================="
echo " Starting ProsQA Batch Evaluation on Phase 2a Checkpoints"
echo " Host: $(hostname) | Date: $(date)"
echo "================================================================="

python scripts/eval_phase2a_prosqa.py --batch-size 16

echo "================================================================="
echo " Completed ProsQA Batch Evaluation: $(date)"
echo "================================================================="
