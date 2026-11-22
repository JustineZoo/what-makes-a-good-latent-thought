#!/bin/bash
#SBATCH --job-name=t4_ctrls
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH -A general
#SBATCH -p regular
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-3%1

set -e
module load cuda/12.6
source ~/.venvs/dlm/bin/activate
cd $HOME/what-makes-a-good-latent-thought

export TORCH_DISABLE_NATIVE_JIT=1
export PYTHONUNBUFFERED=1

case $SLURM_ARRAY_TASK_ID in
  0)
    # 3-seed Abelian Aff(Z_5) control (eliminates single-seed D=2 variance)
    if [ ! -f outputs/ae_affine_z5_r16.pt ]; then
      python -m dlm.train.train_ae \
        --task function_compose --group affine --task-n 5 --task-max-depth 32 --answer-mode trace \
        --R 16 --slots 4 --depths 2 4 8 16 32 --uniform-depths --depth-power 1.0 \
        --beta 0.05 --steps 15000 --batch-size 64 --gate-recon 0.95 \
        --out outputs/ae_affine_z5_r16.pt
    fi
    for S in 0 1 2; do
      python -m dlm.train.train_generator \
        --ae outputs/ae_affine_z5_r16.pt --gen ar --seed $S \
        --task function_compose --group affine --task-n 5 --task-max-depth 32 --answer-mode trace \
        --depths 2 4 8 16 32 --steps 25000 --batch-size 128 --lr 3e-4 \
        --out outputs/gen_affine_z5_s${S}.pt
    done
    python -c "
import numpy as np
from dlm.eval.accuracy import load_generator, eval_concept_generator, _task_from_meta
with open('outputs/affine_z5_3trainseeds.txt', 'w') as f:
    for d in [2, 4, 8, 16, 32]:
        rec = []
        for s in [0, 1, 2]:
            gen, ae, _, meta = load_generator(f'outputs/gen_affine_z5_s{s}.pt', device='cuda')
            rec.append(eval_concept_generator(gen, ae, _task_from_meta(meta), depths=[d], n_per_depth=2048, seed=12345, device='cuda')[d]*100)
        line = f'Depth {d}: {np.mean(rec):6.2f}% +/- {np.std(rec):4.2f}%'
        print(line); f.write(line + '\n')
"
    ;;
  1)
    # 75k-step state-aligned (R=30, K=6) budget-parity control
    python -m dlm.train.train_generator \
      --ae outputs/ae_bbh_r30_k6.pt --gen ar \
      --task bbh_shuffled_objects --task-n 5 --task-max-depth 8 --answer-mode trace \
      --depths 1 2 4 8 --steps 75000 --batch-size 128 --lr 3e-4 \
      --out outputs/gen_ar_bbh_r30_75k.pt
    python -c "
import numpy as np
from dlm.eval.accuracy import load_generator, eval_concept_generator, _task_from_meta
gen, ae, _, meta = load_generator('outputs/gen_ar_bbh_r30_75k.pt', device='cuda')
task = _task_from_meta(meta)
with open('outputs/bbh_r30_aligned_75k_multiseed.txt', 'w') as f:
    for d in [1, 2, 4, 8]:
        rec = [eval_concept_generator(gen, ae, task, depths=[d], n_per_depth=2048, seed=s, device='cuda')[d]*100 for s in [12345,12346,12347]]
        line = f'Depth {d}: {np.mean(rec):6.2f}% +/- {np.std(rec):4.2f}%'
        print(line); f.write(line + '\n')
"
    ;;
  2)
    # Matched-recipe codec control (identical 30k AE steps & uniform depth sampling across R=8 and R=32)
    for R_K in "8:1" "32:8"; do
      R="${R_K%%:*}"
      K="${R_K##*:}"
      python -m dlm.train.train_ae \
        --task bbh_shuffled_objects --task-n 5 --task-max-depth 8 --answer-mode trace \
        --R $R --slots $K --depths 1 2 4 8 --uniform-depths --depth-power 1.0 \
        --beta 0.05 --steps 30000 --batch-size 64 --gate-recon 0.95 \
        --out outputs/ae_bbh_matched_r${R}.pt
      python -m dlm.train.train_generator \
        --ae outputs/ae_bbh_matched_r${R}.pt --gen ar \
        --task bbh_shuffled_objects --task-n 5 --task-max-depth 8 --answer-mode trace \
        --depths 1 2 4 8 --steps 25000 --batch-size 128 --lr 3e-4 \
        --out outputs/gen_bbh_matched_r${R}.pt
      python -c "
import numpy as np
from dlm.eval.accuracy import load_generator, eval_concept_generator, _task_from_meta
gen, ae, _, meta = load_generator('outputs/gen_bbh_matched_r${R}.pt', device='cuda')
task = _task_from_meta(meta)
with open('outputs/bbh_matched_recipe_r${R}_multiseed.txt', 'w') as f:
    for d in [1, 2, 4, 8]:
        rec = [eval_concept_generator(gen, ae, task, depths=[d], n_per_depth=2048, seed=s, device='cuda')[d]*100 for s in [12345,12346,12347]]
        line = f'Depth {d}: {np.mean(rec):6.2f}% +/- {np.std(rec):4.2f}%'
        print(line); f.write(line + '\n')
"
    done
    ;;
  3)
    # Foundational S_5 (R=32, K=8, D_max=32) 3-training-seed upgrade for Table 5
    python -m dlm.train.train_ae \
      --task function_compose --task-n 5 --task-max-depth 32 --answer-mode trace \
      --R 32 --slots 8 --depths 2 4 8 16 32 --uniform-depths --depth-power 1.0 \
      --beta 0.05 --steps 25000 --batch-size 64 --gate-recon 0.95 \
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
    ;;
esac
