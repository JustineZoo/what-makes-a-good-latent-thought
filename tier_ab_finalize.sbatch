#!/bin/bash
#SBATCH --job-name=tier_ab
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -A general
#SBATCH -p regular
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -e
module load cuda/12.6
source ~/.venvs/dlm/bin/activate
cd $HOME/what-makes-a-good-latent-thought

export TORCH_DISABLE_NATIVE_JIT=1
export PYTHONUNBUFFERED=1

# Step 1: Run Task A (~5 mins)
python scripts/run_tier_a_upgrades.py

# Step 2: Run Task B (Uniform S_5 N=5, R=5, d_concept=320 Anchor for Entity Scaling)
if [ ! -f outputs/ae_s5_r5_d320.pt ]; then
  python -m dlm.train.train_ae \
    --task function_compose --task-n 5 --task-max-depth 8 --answer-mode trace \
    --R 5 --slots 1 --d-concept 320 --dec-layers 6 \
    --depths 1 2 4 8 --uniform-depths --depth-power 1.0 \
    --beta 0.02 --steps 15000 --batch-size 64 --gate-recon 0.95 \
    --out outputs/ae_s5_r5_d320.pt
fi

for MODE_TAG in "--slot-buffer:slot_n5" "--gru-buffer:gru_n5"; do
  MODE="${MODE_TAG%%:*}"
  TAG="${MODE_TAG##*:}"
  python -m dlm.train.train_generator \
    --ae outputs/ae_s5_r5_d320.pt --gen ar $MODE --n-slots 5 \
    --task function_compose --task-n 5 --task-max-depth 8 --answer-mode trace \
    --depths 1 2 4 8 --steps 25000 --batch-size 128 --lr 3e-4 \
    --out outputs/gen_${TAG}_25k.pt

  python -c "
import numpy as np
from dlm.eval.accuracy import load_generator, eval_concept_generator, _task_from_meta
gen, ae, _, meta = load_generator('outputs/gen_${TAG}_25k.pt', device='cuda')
task = _task_from_meta(meta)
with open('outputs/s5_${TAG}_multiseed.txt', 'w') as f:
    for d in [1, 2, 4, 8]:
        rec = [eval_concept_generator(gen, ae, task, depths=[d], n_per_depth=2048, seed=s, device='cuda')[d]*100 for s in [12345,12346,12347]]
        line = f'Depth {d}: {np.mean(rec):6.2f}% +/- {np.std(rec):4.2f}%'
        print(line); f.write(line + '\n')
"
done
