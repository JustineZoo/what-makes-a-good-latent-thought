"""True Phase 2a: Train and evaluate C1/C2 inside Native CoLaR on Llama-3.2-1B-Instruct.

Systems:
  - colar       : Vanilla CoLaR (fixed chunk c=5, single Gaussian LatentPolicy)
  - colar_c1    : + C1 (Step-aligned, load-capped chunking <= c')
  - colar_c2    : + C2 (Factorized K sub-slot Gaussian policy)
  - colar_c1_c2 : + C1 + C2 (Gate B candidate)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from dlm.data.gsm8k_aug import GSM8KExample, load_gsm8k_aug_splits
from dlm.data.ood_math import load_ood_math_split
from dlm.data.prosqa import load_prosqa_split
from dlm.eval.gsm8k_metrics import score_prediction, summarize_results
from dlm.models.phase2_colar import Phase2CoLaR


class CoLaRQSADataset(Dataset):
    """Dataset wrapper converting GSM8KExample / ProsQAExample to CoLaR fields."""

    def __init__(self, examples: Sequence[object]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        ex = self.examples[idx]
        steps_str = "\n".join(s.strip() for s in ex.steps if s.strip()) if ex.steps else ""
        return {
            "question": ex.question.strip(),
            "steps": steps_str,
            "answer": str(ex.answer).strip(),
            "example": ex,
        }


def collate_qsa(batch: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "question": [b["question"] for b in batch],
        "steps": [b["steps"] for b in batch],
        "answer": [b["answer"] for b in batch],
        "example": [b["example"] for b in batch],
    }


def extract_colar_pred_answer(raw_text: str) -> str:
    """Parse CoLaR generated output ('Answer:18' or '###Answer:18') to canonical string."""
    s = raw_text.strip().strip("#").strip()
    if "Answer:" in s:
        s = s.split("Answer:")[-1].strip()
    elif "####" in s:
        s = s.split("####")[-1].strip()
    return s


@torch.no_grad()
def evaluate_phase2_colar(
    model: Phase2CoLaR,
    examples: Sequence[object],
    batch_size: int = 32,
    max_n_latent_forward: int = 64,
    max_new_tokens: int = 24,
    latent_temperature: float = 0.0,
    do_sample_answer: bool = False,
) -> Dict[str, object]:
    model.eval()
    ds = CoLaRQSADataset(examples)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_qsa)

    all_res = []
    latent_counts: List[int] = []
    raw_samples = []
    t0 = time.perf_counter()

    for batch in loader:
        pred_ids, n_lat = model.latent_generate(
            questions=batch["question"],
            max_n_latent_forward=max_n_latent_forward,
            max_new_tokens=max_new_tokens,
            latent_temperature=latent_temperature,
            do_sample_answer=do_sample_answer,
            greedy_eol=True,
        )
        pred_strs = model.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        n_lat_list = n_lat.squeeze(-1).cpu().tolist()
        latent_counts.extend(int(x) for x in n_lat_list)

        for ex, p_str in zip(batch["example"], pred_strs):
            ans_extracted = extract_colar_pred_answer(p_str)
            # Pass extracted answer in canonical #### format for exact grading
            res = score_prediction(f"#### {ans_extracted}", ex.steps, ex.answer)
            all_res.append(res)
            if len(raw_samples) < 5:
                raw_samples.append({"gold": ex.answer, "raw_pred": p_str, "extracted": ans_extracted, "correct": res.answer_correct})

    summary = summarize_results(all_res)
    summary["mean_n_latent_forward"] = sum(latent_counts) / max(1, len(latent_counts))
    summary["wall_clock_sec"] = time.perf_counter() - t0
    summary["n_evaluated"] = len(examples)
    summary["sample_predictions"] = raw_samples
    model.train()
    return summary


def build_colar_model(
    model_name: str,
    system: str,
    c: int,
    c_prime: int,
    K: int,
    lora_r: int,
    lora_alpha: int,
    base_dtype: torch.dtype,
    device: torch.device,
    gradient_checkpointing: bool = True,
    target_modules: Optional[List[str]] = None,
) -> Phase2CoLaR:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    base_llm = AutoModelForCausalLM.from_pretrained(model_name, dtype=base_dtype)
    base_llm.resize_token_embeddings(len(tokenizer))
    base_llm.config.use_cache = False
    base_llm.generation_config.pad_token_id = tokenizer.pad_token_id
    base_llm.generation_config.eos_token_id = tokenizer.eos_token_id

    mods = target_modules or ["q_proj", "v_proj"]
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=mods,
    )
    llm = get_peft_model(base_llm, lora_cfg)
    for p in llm.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)

    if gradient_checkpointing:
        llm.enable_input_require_grads()
        llm.gradient_checkpointing_enable()

    model = Phase2CoLaR(
        llm=llm,
        tokenizer=tokenizer,
        system=system,
        c=c,
        c_prime=c_prime,
        K=K,
    ).to(device)
    return model


def parse_args():
    p = argparse.ArgumentParser(description="True Phase 2a: Native CoLaR Ablations")
    p.add_argument("--system", choices=["colar", "colar_c1", "colar_c2", "colar_c1_c2"], required=True)
    p.add_argument("--model-name", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--c", type=int, default=5)
    p.add_argument("--c-prime", type=int, default=7, help="Matched to c=5 budget (mean ~4.2 vs ~5.4)")
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--lora-r", type=int, default=128)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-only-ckpt", default=None, help="Load checkpoint and run evaluation only")
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    base_dtype = amp_dtype if device.type == "cuda" else torch.float32

    out_dir = Path(args.out_dir or f"outputs/true_p2a_{args.system}_s{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_colar_model(
        model_name=args.model_name,
        system=args.system,
        c=args.c,
        c_prime=args.c_prime,
        K=args.K,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        base_dtype=base_dtype,
        device=device,
        gradient_checkpointing=(args.eval_only_ckpt is None),
    )

    splits = load_gsm8k_aug_splits(mode="cot", seed=args.seed)

    if args.eval_only_ckpt:
        payload = torch.load(args.eval_only_ckpt, map_location="cpu", weights_only=False)
        sd = payload["state_dict"] if "state_dict" in payload else payload
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[eval-only] Loaded {args.eval_only_ckpt} (missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        train_ds = CoLaRQSADataset(splits["train"])
        loader = DataLoader(
            train_ds,
            batch_size=args.micro_batch_size,
            shuffle=True,
            collate_fn=collate_qsa,
            drop_last=True,
        )
        grad_accum = max(1, args.batch_size // args.micro_batch_size)
        total_updates = math.ceil(len(loader) / grad_accum) * args.epochs
        trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert len(trainable) > 0, "No trainable parameters found!"
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not use_bf16))

        update_step, run_loss, run_ce, run_lat = 0, 0.0, 0.0, 0.0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        print("=================================================================")
        print(f" Starting True Phase 2a CoLaR Training: {args.system} (seed={args.seed})")
        print(f" Updates={total_updates} | Micro-Batch={args.micro_batch_size} | Eff-Batch={args.batch_size}")
        print(f" Device={device} | c={args.c} | c_prime={args.c_prime} | K={args.K} | Out={out_dir}")
        print("=================================================================")

        for epoch in range(1, args.epochs + 1):
            for micro_idx, batch in enumerate(loader, start=1):
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                    loss_dict = model(batch["question"], batch["steps"], batch["answer"])
                    loss = loss_dict["total_loss"] / grad_accum

                scaler.scale(loss).backward()
                run_loss += float(loss.detach())
                run_ce += float(loss_dict["ce_loss"]) / grad_accum
                run_lat += float(loss_dict["embed_modeling_loss"]) / grad_accum

                if micro_idx % grad_accum == 0 or micro_idx == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    update_step += 1

                    if update_step % args.log_every == 0 or update_step == 1:
                        denom = min(args.log_every, update_step)
                        print(
                            f"  [{args.system}] update {update_step:5d}/{total_updates} "
                            f"| loss={run_loss/denom:.4f} (ce={run_ce/denom:.4f}, lat={run_lat/denom:.4f}) "
                            f"| {(time.time()-t0)/60:.1f}m",
                            flush=True,
                        )
                        run_loss, run_ce, run_lat = 0.0, 0.0, 0.0

                    if update_step % 500 == 0:
                        torch.save(
                            {
                                "system": args.system,
                                "state_dict": {k: v.cpu() for k, v in model.state_dict().items() if k in trainable_names},
                                "cfg": vars(args),
                                "step": update_step,
                            },
                            out_dir / "colar_latest.pt",
                        )

            # End-of-epoch validation accuracy check
            val_res = evaluate_phase2_colar(model, splits["validation"], batch_size=args.eval_batch_size)
            print(
                f"  [epoch {epoch}] Validation (n={val_res['n']}): "
                f"Ans Acc={val_res['answer_accuracy']*100:.2f}% | #L={val_res['mean_n_latent_forward']:.2f}",
                flush=True,
            )

        torch.save(
            {
                "system": args.system,
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items() if k in trainable_names},
                "cfg": vars(args),
            },
            out_dir / "colar_final.pt",
        )

    # Full Test Evaluations (GSM8K-Aug + OOD Suite + ProsQA)
    print("\n--- Running Official GSM8K-Aug Test Evaluation (n=1,319) ---")
    gsm8k_res = evaluate_phase2_colar(model, splits["test"], batch_size=args.eval_batch_size)
    (out_dir / "metrics_gsm8k.json").write_text(json.dumps(gsm8k_res, indent=2))

    print("\n--- Running Out-of-Distribution Evaluations (SVAMP, GSM-Hard, MultiArith, ProsQA) ---")
    ood_res = {}
    for ds_name in ["svamp", "gsm_hard", "multiarith"]:
        exs = load_ood_math_split(ds_name)
        ood_res[ds_name] = evaluate_phase2_colar(model, exs, batch_size=args.eval_batch_size)
    try:
        prosqa_path = "data/prosqa/prosqa_test.json" if Path("data/prosqa/prosqa_test.json").exists() else None
        prosqa_exs = load_prosqa_split("test", local_path=prosqa_path)
        ood_res["prosqa"] = evaluate_phase2_colar(model, prosqa_exs, batch_size=args.eval_batch_size)
    except Exception as e:
        print(f"Warning: ProsQA evaluation skipped due to {e}")
    (out_dir / "metrics_ood.json").write_text(json.dumps(ood_res, indent=2))

    prosqa_acc = f"{ood_res['prosqa']['answer_accuracy']*100:.2f}%" if "prosqa" in ood_res else "N/A"
    summary = (
        f"True Phase 2a System: {args.system} (Seed={args.seed})\n"
        f"GSM8K Test Ans Acc  : {gsm8k_res['answer_accuracy']*100:.2f}% (#L={gsm8k_res['mean_n_latent_forward']:.2f})\n"
        f"SVAMP Ans Acc       : {ood_res['svamp']['answer_accuracy']*100:.2f}% (#L={ood_res['svamp']['mean_n_latent_forward']:.2f})\n"
        f"GSM-Hard Ans Acc    : {ood_res['gsm_hard']['answer_accuracy']*100:.2f}% (#L={ood_res['gsm_hard']['mean_n_latent_forward']:.2f})\n"
        f"MultiArith Ans Acc  : {ood_res['multiarith']['answer_accuracy']*100:.2f}% (#L={ood_res['multiarith']['mean_n_latent_forward']:.2f})\n"
        f"ProsQA Ans Acc      : {prosqa_acc}\n"
    )
    print("\n" + summary)
    (out_dir / "summary.txt").write_text(summary)


if __name__ == "__main__":
    main()
