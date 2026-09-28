# What Makes a Good Latent Thought for Reasoning?

Code and representative checkpoints for studying compressed latent reasoning
on permutation composition, affine composition, and graph walking.

## Setup

Use Python 3.11 and install PyTorch for your hardware. The models were trained
with PyTorch 2.10.0 and NumPy 1.26.4. From this repository:

```bash
python -m pip install -e '.[train,dev,hub]'
python -m pytest -q
```

Data are generated procedurally with exact oracles; no dataset download is needed.
Full training and evaluation use a CUDA GPU. Small inference checks can use CPU.

## Checkpoints

The companion HF repository contains **16 representative models and 10 required
codecs, approximately 3.62 GB**. Each setting has one seed: seed 0, except walker
diffusion, which uses seed 3 from the paper's repeat cohort. Additional seeds
can be trained with the same code.

```bash
python scripts/checkpoints.py download --claim compression \
  --repo-id OWNER/what-makes-a-good-latent-thought-checkpoints --root checkpoints
```

Replace `OWNER` when the repository is published. Use `--claim all` for the full
main-model set, or `--model s5_ar_r8` for one model and its codec. Downloads are
verified against SHA-256 checksums. An existing HF snapshot can be used directly
as `--checkpoint-root`.

## Reproduce the main claims

| Claim | What to compare |
|---|---|
| `compression` | Token AR vs latent AR at R4/R8 |
| `structure` | R16 entangled, factorized K4, and blockwise latent AR |
| `capacity` | R16/K4 vs R32/K8; high-compression affine control |
| `refinement` | Latent AR/diffusion and the discrete-code comparison |
| `walker` | Latent AR/diffusion at R1/R3 with the prompt codec held fixed |

```bash
python -m reproduce.evaluate --claim compression \
  --checkpoint-root checkpoints --out outputs/compression.json
```

Replace `compression` with any claim above. Use `--model MODEL` to evaluate one
checkpoint, `--list` to list the models, and `--T 1 16 64 256` to vary diffusion
steps. Results include final-state accuracy, gold reconstruction accuracy,
per-state accuracy, and prefix survival. All models for a task receive the same
programs. Defaults are 512 programs per depth, batch size 64, and T=256.

These single-seed comparisons reproduce the **qualitative main claims**, rather
than the paper's complete multi-seed tables. The evaluator uses a shared fresh
test set; it does not replay each historical evaluation's RNG state. For a quick
functional check, add `--n 2 --depths 2 --T 2 --device cpu`.

The two main mechanism diagnostics use the same checkpoints:

```bash
python -m reproduce.probe --model s5_ar_r32_k8 --probe prefix \
  --checkpoint-root checkpoints --out outputs/prefix.json
python -m reproduce.probe --model walker_diffusion_r3 --probe context \
  --checkpoint-root checkpoints --out outputs/context.json
```

The prefix probe supplies correct preceding latents; it also supports
`walker_ar_r3`. The context probe switches the graph prompt during denoising.

## Train additional models

Retrain a released architecture with its frozen codec and a new seed:

```bash
python -m reproduce.train --model s5_ar_r16_k4 --seed 1 \
  --checkpoint-root checkpoints --out outputs/s5_ar_r16_k4_s1.pt
```

Use `--dry-run` to inspect the command. The recipe uses 25k generator updates,
batch 128 and AdamW at 3e-4; the converged masked-code diffusion model uses 75k
updates. Training more generator seeds should keep the codec fixed.

To train a new R8 codec, then use it with `dlm.train.train_generator`:

```bash
mkdir -p outputs
python -m dlm.train.train_ae --task function_compose --task-n 5 \
  --task-max-depth 32 --answer-mode trace --R 8 --d-concept 384 \
  --beta 0.05 --layers 6 --uniform-depths --depth-power 1 \
  --gate-recon 0.95 --steps 45000 --out outputs/ae_r8.pt
```

Use `--slots 4` for a factorized codec or `--blockwise` for a block-local codec.
Training CLIs expose architecture and task options through `--help`.

`dlm/` contains the implementation, `reproduce/` the evaluation/training entry
points, and `configs/main.json` the small model-to-claim map. Codec pretraining
is additional compute; fewer latent emissions do not imply equal FLOPs or latency.

Paper link, author citation, repository URLs and licenses will be added before
publication.
