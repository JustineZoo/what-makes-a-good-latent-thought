#!/bin/bash
#SBATCH --job-name=p1c_colar
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -A general
#SBATCH -p regular
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -e
module load cuda/12.6
source ~/.bashrc
source ~/.venvs/dlm/bin/activate
cd $HOME/what-makes-a-good-latent-thought

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

python -m dlm.eval.colar_probe --factors 1 2 5 --probe-steps 4000 --batch-size 8
