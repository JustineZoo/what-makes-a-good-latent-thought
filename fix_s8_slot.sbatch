#!/bin/bash
#SBATCH --job-name=fix_s8_slot
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -A general
#SBATCH -p regular
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -e
module load cuda/12.6
source ~/.venvs/dlm/bin/activate
cd $HOME/what-makes-a-good-latent-thought

export TORCH_DISABLE_NATIVE_JIT=1
export PYTHONUNBUFFERED=1

python -m dlm.train.train_generator \
  --ae outputs/ae_s8_r8.pt --gen ar --slot-buffer --n-slots 8 \
  --task function_compose --task-n 8 --task-max-depth 8 --answer-mode trace \
  --depths 1 2 4 8 --steps 25000 --batch-size 128 --lr 3e-4 \
  --out outputs/gen_slot_n8_25k.pt

python -c "
import numpy as np
from dlm.eval.accuracy import load_generator, eval_concept_generator, _task_from_meta
gen, ae, _, meta = load_generator('outputs/gen_slot_n8_25k.pt', device='cuda')
task = _task_from_meta(meta)
with open('outputs/s8_slot_n8_multiseed.txt', 'w') as f:
    for d in [1, 2, 4, 8]:
        rec = [eval_concept_generator(gen, ae, task, depths=[d], n_per_depth=2048, seed=s, device='cuda')[d]*100 for s in [12345,12346,12347]]
        line = f'Depth {d}: {np.mean(rec):6.2f}% +/- {np.std(rec):4.2f}%'
        print(line); f.write(line + '\n')
"
