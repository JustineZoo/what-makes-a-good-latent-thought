#!/bin/bash
#SBATCH --job-name=s5_r32_3s
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -A general
#SBATCH -p regular
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=03:00:00

set -e
module load cuda/12.6
source ~/.venvs/dlm/bin/activate
cd $HOME/what-makes-a-good-latent-thought

export TORCH_DISABLE_NATIVE_JIT=1
export PYTHONUNBUFFERED=1

python -m dlm.train.train_ae \
  --task function_compose --task-n 5 --task-max-depth 32 --answer-mode trace \
  --R 32 --slots 8 --depths 2 4 8 16 32 --uniform-depths --depth-power 1.0 \
  --beta 0.02 --steps 45000 --batch-size 64 --gate-recon 0.95 \
  --out outputs/ae_s5_r32_k8_d32.pt

for S in 0 1 2; do
  python -m dlm.train.train_generator \
    --ae outputs/ae_s5_r32_k8_d32.pt --gen ar --seed $S \
    --task function_compose --task-n 5 --task-max-depth 32 --answer-mode trace \
    --depths 2 4 8 16 32 --steps 25000 --batch-size 128 --lr 3e-4 \
    --out outputs/gen_s5_r32_k8_s${S}.pt
done

python -c "
import numpy as np
from dlm.eval.accuracy import load_generator, eval_concept_generator, _task_from_meta
with open('outputs/s5_r32_k8_3trainseeds.txt', 'w') as f:
    for d in [2, 4, 8, 16, 32]:
        rec = []
        for s in [0, 1, 2]:
            gen, ae, _, meta = load_generator(f'outputs/gen_s5_r32_k8_s{s}.pt', device='cuda')
            rec.append(eval_concept_generator(gen, ae, _task_from_meta(meta), depths=[d], n_per_depth=2048, seed=12345, device='cuda')[d]*100)
        line = f'Depth {d}: {np.mean(rec):6.2f}% +/- {np.std(rec):4.2f}%'
        print(line); f.write(line + '\n')
"
