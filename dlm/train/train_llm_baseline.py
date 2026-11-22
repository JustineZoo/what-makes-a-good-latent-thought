"""Phase 1a: Fine-tune and evaluate Token-CoT (R=1) and No-CoT LoRA baselines on GSM8K-Aug."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from dlm.data.gsm8k_aug import GSM8KExample, load_gsm8k_aug_splits
from dlm.eval.gsm8k_metrics import score_prediction, summarize_results


class TokenizedGSM8KDataset(Dataset):
    def __init__(self, examples: List[GSM8KExample], tokenizer, max_len: int = 256):
        self.items = []
        eos_id = tokenizer.eos_token_id
        for ex in examples:
            p_ids = tokenizer.encode(ex.prompt, add_special_tokens=True)
            t_ids = tokenizer.encode(ex.target, add_special_tokens=False) + [eos_id]
            input_ids = (p_ids + t_ids)[:max_len]
            labels = ([-100] * len(p_ids) + t_ids)[:max_len]
            if any(y != -100 for y in labels):
                self.items.append((input_ids, labels))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def make_collate_fn(pad_id: int):
    def collate(batch):
        max_b = max(len(x[0]) for x in batch)
        input_ids, attn_mask, labels = [], [], []
        for ids, labs in batch:
            pad_n = max_b - len(ids)
            input_ids.append(ids + [pad_id] * pad_n)
            attn_mask.append([1] * len(ids) + [0] * pad_n)
            labels.append(labs + [-100] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    return collate


@torch.no_grad()
def evaluate_split(
    model,
    tokenizer,
    examples: List[GSM8KExample],
    device: torch.device,
    batch_size: int = 32,
    max_new_tokens: int = 160,
    amp_dtype=torch.float16,
) -> Dict[str, object]:
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    results = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        prompts = [ex.prompt for ex in chunk]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(device)
        prompt_len = enc["input_ids"].shape[1]

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            gen_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for row_ids, ex in zip(gen_ids[:, prompt_len:], chunk):
            pred_text = tokenizer.decode(row_ids, skip_special_tokens=True)
            results.append(score_prediction(pred_text, ex.steps, ex.answer))

    tokenizer.padding_side = orig_side
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    return summarize_results(results)


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--mode", choices=["cot", "no_cot"], required=True)
    p.add_argument("--lora-r", type=int, default=128)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256, help="Effective global batch size")
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    base_dtype = amp_dtype if device.type == "cuda" else torch.float32

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phase1a] model={args.model_name} mode={args.mode} device={device} amp={amp_dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    splits = load_gsm8k_aug_splits(mode=args.mode, seed=args.seed)
    train_ds = TokenizedGSM8KDataset(splits["train"], tokenizer, max_len=args.max_len)
    loader = DataLoader(
        train_ds,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer.pad_token_id),
        drop_last=True,
    )

    grad_accum = max(1, args.batch_size // args.micro_batch_size)
    updates_per_epoch = math.ceil(len(loader) / grad_accum)
    total_updates = updates_per_epoch * args.epochs
    snap_1 = max(1, total_updates // 3)
    snap_2 = max(1, (2 * total_updates) // 3)
    print(
        f"[phase1a] train_rows={len(train_ds)} micro_bs={args.micro_batch_size} "
        f"grad_accum={grad_accum} effective_bs={args.batch_size} "
        f"total_updates={total_updates} snapshots=({snap_1}, {snap_2})"
    )

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=base_dtype)
    base_model.config.use_cache = False
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base_model, lora_cfg)
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model = model.to(device)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not use_bf16))

    update_step = 0
    tokens_seen = 0
    running_loss = 0.0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        for micro_idx, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            tokens_seen += int(batch["attention_mask"].sum().item())

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                loss = model(**batch).loss / grad_accum

            scaler.scale(loss).backward()
            running_loss += float(loss.detach())

            if micro_idx % grad_accum == 0 or micro_idx == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1

                if update_step % args.log_every == 0 or update_step == 1:
                    avg_l = running_loss / min(args.log_every, update_step)
                    running_loss = 0.0
                    elapsed = time.time() - t0
                    print(f"  update {update_step:5d}/{total_updates} | tokens={tokens_seen:,} | loss={avg_l:.4f} | {elapsed/60:.1f}m")

                if update_step in (snap_1, snap_2):
                    snap_dir = out_dir / f"ckpt_step_{update_step}"
                    model.save_pretrained(snap_dir)
                    print(f"[phase1a] saved snapshot -> {snap_dir}")

    final_dir = out_dir / "ckpt_final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[phase1a] saved final adapter -> {final_dir}")

    max_gen = 160 if args.mode == "cot" else 32
    val_metrics = evaluate_split(model, tokenizer, splits["validation"], device, args.eval_batch_size, max_gen, amp_dtype)
    test_metrics = evaluate_split(model, tokenizer, splits["test"], device, args.eval_batch_size, max_gen, amp_dtype)

    report = {
        "model_name": args.model_name,
        "mode": args.mode,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lr": args.lr,
        "effective_batch_size": args.batch_size,
        "total_updates": update_step,
        "tokens_seen": tokens_seen,
        "seed": args.seed,
        "validation": val_metrics,
        "test": test_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))

    summary_txt = (
        f"================================================================\n"
        f" Phase 1a Sanity Baseline: {args.mode} ({args.model_name})\n"
        f"================================================================\n"
        f"Updates: {update_step} | Batch Size: {args.batch_size} | Tokens Seen: {tokens_seen:,}\n"
        f"LoRA: r={args.lora_r}, alpha={args.lora_alpha} | LR: {args.lr}\n"
        f"Validation (n={val_metrics['n']}): Answer Acc = {val_metrics['answer_accuracy']*100:.2f}% | "
        f"Step Exact = {val_metrics['step_exact_accuracy']*100:.2f}% | Arith Consist = {val_metrics['arithmetic_consistency']*100:.2f}%\n"
        f"Official Test (n={test_metrics['n']}): Answer Acc = {test_metrics['answer_accuracy']*100:.2f}% | "
        f"Step Exact = {test_metrics['step_exact_accuracy']*100:.2f}% | Arith Consist = {test_metrics['arithmetic_consistency']*100:.2f}%\n"
        f"Mean First-Error Pos: {test_metrics['mean_first_error_pos']:.2f} | Errors: {test_metrics['error_type_counts']}\n"
        f"================================================================\n"
    )
    print("\n" + summary_txt)
    Path(f"outputs/phase1a_{args.mode}_s{args.seed}.txt").write_text(summary_txt)


if __name__ == "__main__":
    main()
