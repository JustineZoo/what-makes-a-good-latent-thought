#!/bin/bash
#SBATCH --job-name=t2_codecs
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH -A general
#SBATCH -p regular
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-2%1

set -e
module load cuda/12.6
source ~/.venvs/dlm/bin/activate
cd $HOME/what-makes-a-good-latent-thought

export TORCH_DISABLE_NATIVE_JIT=1
export PYTHONUNBUFFERED=1

case $SLURM_ARRAY_TASK_ID in
  0)
    TAG="r32_k8_d768"
    AE_ARGS="--R 32 --slots 8 --d-concept 768 --steps 15000"
    ;;
  1)
    TAG="r8_blockwise"
    AE_ARGS="--R 8 --slots 1 --blockwise --steps 15000"
    ;;
  2)
    TAG="r16_k4_blockwise"
    AE_ARGS="--R 16 --slots 4 --blockwise --steps 15000"
    ;;
esac

python -m dlm.train.train_ae \
  --task bbh_shuffled_objects --task-n 5 --task-max-depth 8 --answer-mode trace \
  $AE_ARGS --depths 1 2 4 8 --uniform-depths --depth-power 1.0 \
  --beta 0.05 --batch-size 64 --gate-recon 0.95 \
  --out outputs/ae_bbh_${TAG}.pt

python -m dlm.train.train_generator \
  --ae outputs/ae_bbh_${TAG}.pt --gen ar \
  --task bbh_shuffled_objects --task-n 5 --task-max-depth 8 --answer-mode trace \
  --depths 1 2 4 8 --steps 25000 --batch-size 128 --lr 3e-4 \
  --out outputs/gen_ar_bbh_${TAG}.pt

python -c "
import numpy as np
from dlm.eval.accuracy import load_generator, eval_concept_generator, _task_from_meta
gen, ae, cond_ae, meta = load_generator('outputs/gen_ar_bbh_${TAG}.pt', device='cuda')
task = _task_from_meta(meta)
seeds, depths = [12345, 12346, 12347], [1, 2, 4, 8]
with open('outputs/bbh_${TAG}_multiseed.txt', 'w') as f:
    for d in depths:
        rec = [eval_concept_generator(gen, ae, task, depths=[d], n_per_depth=2048, seed=s, device='cuda')[d]*100 for s in seeds]
        line = f'Depth {d}: {np.mean(rec):6.2f}% +/- {np.std(rec):4.2f}%'
        print(line); f.write(line + '\n')
"
