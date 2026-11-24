#!/bin/bash
#SBATCH --job-name=p2a_eval
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH -A allaccess
#SBATCH -p full
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -e
module load cuda/12.6
source ~/.venvs/dlm/bin/activate
cd $HOME/what-makes-a-good-latent-thought
mkdir -p logs outputs

echo "================================================================="
echo " Starting Standalone Phase 2a Evaluation"
echo " Host: $(hostname) | Date: $(date)"
echo "================================================================="

if [ -f "outputs/phase2a_colar_c1_c2_s0/gen_final.pt" ]; then
    echo "--- Evaluating Task 3: colar_c1_c2 (Seed=0) ---"
    python -m dlm.eval.eval_phase2 \
        --ckpt outputs/phase2a_colar_c1_c2_s0/gen_final.pt \
        --codec-ckpt outputs/phase1b_codec_r4_s0/codec_final.pt \
        --batch-size 16
fi

if [ -f "outputs/phase2a_colar_c1_c2_c3a_s0/gen_final.pt" ]; then
    echo "--- Evaluating Task 4: colar_c1_c2_c3a (Seed=0) ---"
    python -m dlm.eval.eval_phase2 \
        --ckpt outputs/phase2a_colar_c1_c2_c3a_s0/gen_final.pt \
        --codec-ckpt outputs/phase1b_codec_r4_s0/codec_final.pt \
        --batch-size 16
fi

echo "================================================================="
echo " Evaluation Completed Successfully: $(date)"
echo "================================================================="
