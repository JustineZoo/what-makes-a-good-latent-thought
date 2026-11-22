#!/bin/bash
#SBATCH --job-name=t2_mem_z5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH -A general
#SBATCH -p regular
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-4%1

set -e
module load cuda/12.6
source ~/.venvs/dlm/bin/activate
cd $HOME/what-makes-a-good-latent-thought

export TORCH_DISABLE_NATIVE_JIT=1
export PYTHONUNBUFFERED=1

case $SLURM_ARRAY_TASK_ID in
  0)
    TAG="slot_r16"
    AE="outputs/ae_bbh_r16_k4.pt"
    FLAG="--slot-buffer"
    ;;
  1)
    TAG="gru_r16"
    AE="outputs/ae_bbh_r16_k4.pt"
    FLAG="--gru-buffer"
    ;;
  2)
    TAG="slot_r32"
    AE="outputs/ae_bbh_r32_k8.pt"
    FLAG="--slot-buffer"
    ;;
  3)
    TAG="gru_r32"
    AE="outputs/ae_bbh_r32_k8.pt"
    FLAG="--gru-buffer"
    ;;
  4)
    python -m dlm.train.train_ae \
      --task function_compose --task-n 5 --task-max-depth 32 --task-mode affine --answer-mode trace \
      --R 16 --slots 4 --depths 2 4 8 16 32 --uniform-depths --depth-power 1.0 \
      --beta 0.05 --steps 15000 --batch-size 64 --gate-recon 0.95 \
      --out outputs/ae_z5_r16_k4.pt

    python -m dlm.train.train_generator \
      --ae outputs/ae_z5_r16_k4.pt --gen ar \
      --task function_compose --task-n 5 --task-max-depth 32 --task-mode affine --answer-mode trace \
      --depths 2 4 8 16 32 --steps 25000 --batch-size 128 --lr 3e-4 \
      --out outputs/gen_ar_z5_r16_k4.pt

    python -c "
import numpy as np
from dlm.eval.accuracy import load_generator, eval_concept_generator, _task_from_meta
gen, ae, _, meta = load_generator('outputs/gen_ar_z5_r16_k4.pt', device='cuda')
task = _task_from_meta(meta)
with open('outputs/affine_z5_multiseed.txt', 'w') as f:
    for d in [2, 4, 8, 16, 32]:
        rec = [eval_concept_generator(gen, ae, task, depths=[d], n_per_depth=2048, seed=s, device='cuda')[d]*100 for s in [12345,12346,12347]]
        line = f'Depth {d}: {np.mean(rec):6.2f}% +/- {np.std(rec):4.2f}%'
        print(line); f.write(line + '\n')
"
    exit 0
    ;;
esac

python -m dlm.train.train_generator \
  --ae $AE --gen ar $FLAG \
  --task bbh_shuffled_objects --task-n 5 --task-max-depth 8 --answer-mode trace \
  --depths 1 2 4 8 --steps 25000 --batch-size 128 --lr 3e-4 \
  --out outputs/gen_${TAG}_bbh_25k.pt

python -c "
import numpy as np
from dlm.eval.accuracy import load_generator, eval_concept_generator, _task_from_meta
gen, ae, _, meta = load_generator('outputs/gen_${TAG}_bbh_25k.pt', device='cuda')
task = _task_from_meta(meta)
with open('outputs/bbh_${TAG}_multiseed.txt', 'w') as f:
    for d in [1, 2, 4, 8]:
        rec = [eval_concept_generator(gen, ae, task, depths=[d], n_per_depth=2048, seed=s, device='cuda')[d]*100 for s in [12345,12346,12347]]
        line = f'Depth {d}: {np.mean(rec):6.2f}% +/- {np.std(rec):4.2f}%'
        print(line); f.write(line + '\n')
"
