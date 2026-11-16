#!/bin/bash
#SBATCH --job-name=t3_bij_gr
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH -A general
#SBATCH -p regular
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --array=0-3%1

set -e
module load cuda/12.6
source ~/.venvs/dlm/bin/activate
cd $HOME/what-makes-a-good-latent-thought

export TORCH_DISABLE_NATIVE_JIT=1
export PYTHONUNBUFFERED=1

if [ ! -f outputs/ae_bij_graph_r6.pt ]; then
  python -m dlm.train.train_ae \
    --task graph_reach --group walkerids --task-n 5 --task-max-depth 12 --bijective --answer-mode trace \
    --R 6 --slots 1 --depths 2 8 12 --uniform-depths --depth-power 1.0 \
    --beta 0.05 --steps 15000 --batch-size 64 --gate-recon 0.95 \
    --out outputs/ae_bij_graph_r6.pt
fi

case $SLURM_ARRAY_TASK_ID in
  0)
    python -m dlm.train.train_generator \
      --ae outputs/ae_bij_graph_r6.pt --gen ar \
      --task graph_reach --group walkerids --task-n 5 --task-max-depth 12 --bijective --answer-mode trace \
      --depths 2 8 12 --steps 75000 --batch-size 128 --lr 3e-4 \
      --out outputs/gen_ar_bij_graph_75k.pt

    python -c "
import numpy as np
from dlm.eval.accuracy import load_generator, eval_concept_generator, _task_from_meta
gen, ae, cond_ae, meta = load_generator('outputs/gen_ar_bij_graph_75k.pt', device='cuda')
task = _task_from_meta(meta)
with open('outputs/bij_graph_ar_75k_multiseed.txt', 'w') as f:
    for d in [2, 8, 12]:
        rec = [eval_concept_generator(gen, ae, task, depths=[d], n_per_depth=2048, seed=s, device='cuda', cond_ae=cond_ae)[d]*100 for s in [12345,12346,12347]]
        line = f'Depth {d}: {np.mean(rec):6.2f}% +/- {np.std(rec):4.2f}%'
        print(line); f.write(line + '\n')
"
    ;;
  1)
    python -m dlm.train.train_ae \
      --task graph_reach --group walkerids --task-n 5 --task-max-depth 12 --bijective --answer-mode trace \
      --R 6 --slots 1 --blockwise --depths 2 8 12 --uniform-depths --depth-power 1.0 \
      --beta 0.05 --steps 15000 --batch-size 64 --gate-recon 0.95 \
      --out outputs/ae_bij_graph_r6_blockwise.pt

    python -m dlm.train.train_generator \
      --ae outputs/ae_bij_graph_r6_blockwise.pt --gen ar \
      --task graph_reach --group walkerids --task-n 5 --task-max-depth 12 --bijective --answer-mode trace \
      --depths 2 8 12 --steps 25000 --batch-size 128 --lr 3e-4 \
      --out outputs/gen_ar_bij_graph_blockwise_25k.pt

    python -c "
import numpy as np
from dlm.eval.accuracy import load_generator, eval_concept_generator, _task_from_meta
gen, ae, cond_ae, meta = load_generator('outputs/gen_ar_bij_graph_blockwise_25k.pt', device='cuda')
task = _task_from_meta(meta)
with open('outputs/bij_graph_blockwise_multiseed.txt', 'w') as f:
    for d in [2, 8, 12]:
        rec = [eval_concept_generator(gen, ae, task, depths=[d], n_per_depth=2048, seed=s, device='cuda', cond_ae=cond_ae)[d]*100 for s in [12345,12346,12347]]
        line = f'Depth {d}: {np.mean(rec):6.2f}% +/- {np.std(rec):4.2f}%'
        print(line); f.write(line + '\n')
"
    ;;
  2)
    python -m dlm.train.train_generator \
      --ae outputs/ae_bij_graph_r6.pt --gen diffusion \
      --task graph_reach --group walkerids --task-n 5 --task-max-depth 12 --bijective --answer-mode trace \
      --depths 2 8 12 --steps 75000 --batch-size 128 --lr 3e-4 \
      --out outputs/gen_diff_bij_graph_75k.pt

    python -c "
from dlm.eval.accuracy import load_generator, eval_concept_generator, _task_from_meta
gen, ae, cond_ae, meta = load_generator('outputs/gen_diff_bij_graph_75k.pt', device='cuda')
task = _task_from_meta(meta)
with open('outputs/bij_graph_diff_75k_results.txt', 'w') as f:
    for T in [1, 16, 64, 256]:
        res = eval_concept_generator(gen, ae, task, depths=[2, 8, 12], n_per_depth=512, T=T, device='cuda', cond_ae=cond_ae)
        line = f'T={T}: ' + ' '.join([f'D{d}={res[d]*100:.2f}%' for d in [2,8,12]])
        print(line); f.write(line + '\n')
"
    python -m dlm.eval.blind_probe --gen outputs/gen_diff_bij_graph_75k.pt \
      --depths 8 12 --T 256 --n 512 --out outputs/bij_graph_blind_probe.json
    ;;
  3)
    python -m dlm.train.train_ae \
      --task graph_reach --group walkerids --task-n 5 --task-max-depth 12 --bijective --answer-mode trace \
      --R 1 --slots 1 --depths 2 8 12 --uniform-depths --depth-power 1.0 \
      --beta 0.001 --free-bits 0.0 --steps 15000 --batch-size 64 --gate-recon 0.95 \
      --out outputs/ae_bij_graph_r1_cond.pt

    python -m dlm.train.train_generator \
      --ae outputs/ae_bij_graph_r6.pt --cond-ae outputs/ae_bij_graph_r1_cond.pt --gen ar \
      --task graph_reach --group walkerids --task-n 5 --task-max-depth 12 --bijective --answer-mode trace \
      --depths 2 8 12 --steps 25000 --batch-size 128 --lr 3e-4 \
      --out outputs/gen_ar_bij_graph_r1cond.pt

    python -c "
import numpy as np
from dlm.eval.accuracy import load_generator, eval_concept_generator, _task_from_meta
gen, ae, cond_ae, meta = load_generator('outputs/gen_ar_bij_graph_r1cond.pt', device='cuda')
task = _task_from_meta(meta)
with open('outputs/bij_graph_r1cond_multiseed.txt', 'w') as f:
    for d in [2, 8, 12]:
        rec = [eval_concept_generator(gen, ae, task, depths=[d], n_per_depth=2048, seed=s, device='cuda', cond_ae=cond_ae)[d]*100 for s in [12345,12346,12347]]
        line = f'Depth {d}: {np.mean(rec):6.2f}% +/- {np.std(rec):4.2f}%'
        print(line); f.write(line + '\n')
"
    ;;
esac
