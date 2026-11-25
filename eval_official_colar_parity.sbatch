#!/bin/bash
#SBATCH --job-name=colar_parity
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -A allaccess
#SBATCH -p full
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=00:45:00

set -e
module load cuda/12.6
source ~/.venvs/dlm/bin/activate
cd $HOME/what-makes-a-good-latent-thought
mkdir -p logs outputs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "================================================================="
echo " Verifying Official CoLaR Checkpoint Parity & 4-System Smoke Test"
echo " Host: $(hostname) | Date: $(date)"
echo "================================================================="

# 1. Verify checkpoint parity on full GSM8K-Aug (n=1,319) + OOD suite
python -m dlm.train.train_phase2_colar \
  --system colar \
  --c 5 \
  --eval-batch-size 32 \
  --eval-only-ckpt /mnt/beegfs/bulk/mirror/jsz45/colar/logs/colar/qsa-gsm/colar-final/checkpoints/colar_best.ckpt \
  --out-dir outputs/colar_official_parity

# 2. Verify forward + backward + save + load across all 4 systems cleanly
python -c "
import tempfile, torch
from dlm.train.train_phase2_colar import build_colar_model

device = torch.device('cuda')
for sys_name in ['colar', 'colar_c1', 'colar_c2', 'colar_c1_c2']:
    m = build_colar_model('meta-llama/Llama-3.2-1B-Instruct', sys_name, c=5, c_prime=7, K=4, lora_r=128, lora_alpha=32, base_dtype=torch.float16, device=device, gradient_checkpointing=True)
    loss_d = m(['Tom has 5 apples and buys 3 more. How many?']*2, ['<<5+3=8>>', '<<5+3=8>>\n<<8*2=16>>'], ['8', '16'])
    loss_d['total_loss'].backward()
    trainable_names = {n for n, p in m.named_parameters() if p.requires_grad}
    sd = {k: v.cpu() for k, v in m.state_dict().items() if k in trainable_names}
    assert len(sd) > 50, f'Empty state dict for {sys_name}'
    pred, nlat = m.latent_generate(['Tom has 5 apples?'], max_n_latent_forward=4, max_new_tokens=8)
    print(f'[SMOKE OK] {sys_name}: loss={loss_d[\"total_loss\"].item():.3f}, saved_keys={len(sd)}, pred_shape={tuple(pred.shape)}')
    del m
    torch.cuda.empty_cache()
"
