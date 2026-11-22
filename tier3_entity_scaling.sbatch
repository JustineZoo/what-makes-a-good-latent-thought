#!/bin/bash
#SBATCH --job-name=t3_ent_scale
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH -A general
#SBATCH -p regular
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%1

set -e
module load cuda/12.6
source ~/.venvs/dlm/bin/activate
cd $HOME/what-makes-a-good-latent-thought

export TORCH_DISABLE_NATIVE_JIT=1
export PYTHONUNBUFFERED=1

NS=(8 8 10 10)
MODES=("--slot-buffer" "--gru-buffer" "--slot-buffer" "--gru-buffer")
TAGS=("slot_n8" "gru_n8" "slot_n10" "gru_n10")

N=${NS[$SLURM_ARRAY_TASK_ID]}
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
TAG=${TAGS[$SLURM_ARRAY_TASK_ID]}

if [ ! -f outputs/ae_s${N}_r${N}.pt ]; then
  python -m dlm.train.train_ae \
    --task function_compose --task-n $N --task-max-depth 8 --answer-mode trace \
    --R $N --slots 1 --d-concept 320 --dec-layers 6 \
    --depths 1 2 4 8 --uniform-depths --depth-power 1.0 \
    --beta 0.02 --steps 30000 --batch-size 64 --gate-recon 0.95 \
    --out outputs/ae_s${N}_r${N}.pt
fi

python -m dlm.train.train_generator \
  --ae outputs/ae_s${N}_r${N}.pt --gen ar $MODE --n-slots $N \
  --task function_compose --task-n $N --task-max-depth 8 --answer-mode trace \
  --depths 1 2 4 8 --steps 25000 --batch-size 128 --lr 3e-4 \
  --out outputs/gen_${TAG}_25k.pt

python -c "
import numpy as np
from dlm.eval.accuracy import load_generator, eval_concept_generator, _task_from_meta
gen, ae, _, meta = load_generator('outputs/gen_${TAG}_25k.pt', device='cuda')
task = _task_from_meta(meta)
with open('outputs/s${N}_${TAG}_multiseed.txt', 'w') as f:
    for d in [1, 2, 4, 8]:
        rec = [eval_concept_generator(gen, ae, task, depths=[d], n_per_depth=2048, seed=s, device='cuda')[d]*100 for s in [12345,12346,12347]]
        line = f'Depth {d}: {np.mean(rec):6.2f}% +/- {np.std(rec):4.2f}%'
        print(line); f.write(line + '\n')
"
