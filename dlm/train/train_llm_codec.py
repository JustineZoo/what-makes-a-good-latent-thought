"""Phase 1b Stage 1: Train LoRA LLM codec at compression R and check >= 0.95 dev gate."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from dlm.data.gsm8k_aug import GSM8KExample, load_gsm8k_aug_splits
from dlm.eval.gsm8k_metrics import score_prediction, summarize_results
from dlm.models.llm_codec import LLMCodecConfig, LLMLatentCodec


class CodecGSM8KDataset(Dataset):
    def __init__(self, examples: List[GSM8KExample], tokenizer, max_q_len: int = 128, max_cot_len: int = 160):
        self.items = []
        eos_id = tokenizer.eos_token_id
        for ex in examples:
            q_ids = tokenizer.encode(ex.prompt, add_special_tokens=True)[:max_q_len]
            t_ids = (tokenizer.encode(ex.target, add_special_tokens=False) + [eos_id])[:max_cot_len]
            if q_ids and t_ids:
                self.items.append((q_ids, t_ids, ex))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def make_codec_collate(pad_id: int, R: int):
    def collate(batch):
        max_q = max(len(x[0]) for x in batch)
        raw_t = max(len(x[1]) for x in batch)
        max_t = ((raw_t + R - 1) // R) * R

        q_ids, q_mask, cot_ids, cot_mask, labels, exs = [], [], [], [], [], []
        for q, t, ex in batch:
            qp = max_q - len(q)
            q_ids.append(q + [pad_id] * qp)
            q_mask.append([1] * len(q) + [0] * qp)

            tp = max_t - len(t)
            cot_ids.append(t + [pad_id] * tp)
            cot_mask.append([1] * len(t) + [0] * tp)
            labels.append(t + [-100] * tp)
            exs.append(ex)

        return {
            "q_ids": torch.tensor(q_ids, dtype=torch.long),
            "q_mask": torch.tensor(q_mask, dtype=torch.long),
            "cot_ids": torch.tensor(cot_ids, dtype=torch.long),
            "cot_mask": torch.tensor(cot_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "examples": exs,
        }
    return collate


@torch.no_grad()
def calibrate_latent_stats(codec: LLMLatentCodec, loader: DataLoader, device: torch.device, amp_dtype, max_batches: int = 16):
    codec.eval()
    mus = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        cot_ids = batch["cot_ids"].to(device)
        cot_mask = batch["cot_mask"].to(device)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            mu, _ = codec.encode_posterior(cot_ids, cot_mask)
        mus.append(mu.float().reshape(-1, codec.cfg.d_z).cpu())
    all_mu = torch.cat(mus, dim=0)
    codec.z_mean.copy_(all_mu.mean(dim=0).to(device))
    codec.z_std.copy_(all_mu.std(dim=0).clamp(min=1e-4).to(device))


@torch.no_grad()
def evaluate_codec_ceiling(
    codec: LLMLatentCodec,
    tokenizer,
    examples: List[GSM8KExample],
    device: torch.device,
    batch_size: int = 16,
    max_q_len: int = 128,
    max_cot_len: int = 160,
    amp_dtype=torch.float16,
) -> Dict[str, object]:
    codec.eval()
    ds = CodecGSM8KDataset(examples, tokenizer, max_q_len=max_q_len, max_cot_len=max_cot_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=make_codec_collate(tokenizer.pad_token_id, codec.cfg.R))
    results = []
    for batch in loader:
        q_ids = batch["q_ids"].to(device)
        q_mask = batch["q_mask"].to(device)
        cot_ids = batch["cot_ids"].to(device)
        cot_mask = batch["cot_mask"].to(device)
        out_len = batch["labels"].shape[1]

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            mu, _ = codec.encode_posterior(cot_ids, cot_mask)
            logits = codec.decode_logits(mu, q_ids, q_mask, out_len=out_len)
        pred_ids = logits.argmax(dim=-1).cpu()
        for row_ids, ex in zip(pred_ids, batch["examples"]):
            pred_text = tokenizer.decode(row_ids, skip_special_tokens=True)
            results.append(score_prediction(pred_text, ex.steps, ex.answer))
    codec.train()
    return summarize_results(results)


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--R", type=int, required=True)
    p.add_argument("--d-z", type=int, default=256)
    p.add_argument("--lora-r", type=int, default=128)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--free-bits", type=float, default=0.25)
    p.add_argument("--gate-threshold", type=float, default=0.95)
    p.add_argument("--max-q-len", type=int, default=128)
    p.add_argument("--max-cot-len", type=int, default=160)
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    splits = load_gsm8k_aug_splits(mode="cot", seed=args.seed)
    train_ds = CodecGSM8KDataset(splits["train"], tokenizer, args.max_q_len, args.max_cot_len)
    loader = DataLoader(
        train_ds,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=make_codec_collate(tokenizer.pad_token_id, args.R),
        drop_last=True,
    )

    grad_accum = max(1, args.batch_size // args.micro_batch_size)
    total_updates = math.ceil(len(loader) / grad_accum) * args.epochs
    snap_1, snap_2 = max(1, total_updates // 3), max(1, (2 * total_updates) // 3)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    def _build_lora_llm():
        m = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=base_dtype)
        m.config.use_cache = False
        m = get_peft_model(m, lora_cfg)
        for p in m.parameters():
            if p.requires_grad:
                p.data = p.data.to(torch.float32)
        m.enable_input_require_grads()
        m.gradient_checkpointing_enable()
        return m

    cfg = LLMCodecConfig(
        model_name=args.model_name,
        R=args.R,
        d_z=args.d_z,
        max_cot_len=args.max_cot_len,
        beta=args.beta,
        free_bits=args.free_bits,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    codec = LLMLatentCodec(_build_lora_llm(), _build_lora_llm(), cfg).to(device)
    optimizer = torch.optim.AdamW(codec.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not use_bf16))

    update_step, tokens_seen, run_loss, run_rec = 0, 0, 0.0, 0.0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for _ in range(1, args.epochs + 1):
        for micro_idx, batch in enumerate(loader, start=1):
            q_ids, q_mask = batch["q_ids"].to(device), batch["q_mask"].to(device)
            cot_ids, cot_mask = batch["cot_ids"].to(device), batch["cot_mask"].to(device)
            labels = batch["labels"].to(device)
            tokens_seen += int(cot_mask.sum().item())

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                loss, rec_l, _ = codec.compute_loss(q_ids, q_mask, cot_ids, cot_mask, labels)
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            run_loss += float(loss.detach())
            run_rec += float(rec_l) / grad_accum

            if micro_idx % grad_accum == 0 or micro_idx == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1

                if update_step % args.log_every == 0 or update_step == 1:
                    denom = min(args.log_every, update_step)
                    print(f"  [codec R={args.R}] update {update_step:5d}/{total_updates} | tokens={tokens_seen:,} | loss={run_loss/denom:.4f} | recon={run_rec/denom:.4f} | {(time.time()-t0)/60:.1f}m")
                    run_loss, run_rec = 0.0, 0.0

                if update_step in (snap_1, snap_2):
                    torch.save({"cfg": asdict(cfg), "state_dict": codec.state_dict()}, out_dir / f"codec_step_{update_step}.pt")

    calibrate_latent_stats(codec, loader, device, amp_dtype)
    dev_metrics = evaluate_codec_ceiling(codec, tokenizer, splits["validation"], device, 16, args.max_q_len, args.max_cot_len, amp_dtype)
    test_metrics = evaluate_codec_ceiling(codec, tokenizer, splits["test"], device, 16, args.max_q_len, args.max_cot_len, amp_dtype)

    gate_passed = dev_metrics["step_exact_accuracy"] >= args.gate_threshold
    ckpt_path = out_dir / "codec_final.pt"
    torch.save({"cfg": asdict(cfg), "state_dict": codec.state_dict(), "gate_passed": gate_passed}, ckpt_path)

    report = {
        "R": args.R,
        "d_z": args.d_z,
        "updates": update_step,
        "batch_size": args.batch_size,
        "tokens_seen": tokens_seen,
        "gate_threshold": args.gate_threshold,
        "gate_passed": gate_passed,
        "dev_ceiling": dev_metrics,
        "test_ceiling": test_metrics,
    }
    (out_dir / "codec_metrics.json").write_text(json.dumps(report, indent=2))
    summary = (
        f"Phase 1b Codec R={args.R} (d_z={args.d_z}) | Updates={update_step} | Tokens={tokens_seen:,}\n"
        f"Dev Ceiling : Step Exact = {dev_metrics['step_exact_accuracy']*100:.2f}% | Answer Acc = {dev_metrics['answer_accuracy']*100:.2f}% | Gate (>=95%): {'PASS' if gate_passed else 'FAIL'}\n"
        f"Test Ceiling: Step Exact = {test_metrics['step_exact_accuracy']*100:.2f}% | Answer Acc = {test_metrics['answer_accuracy']*100:.2f}%\n"
    )
    print("\n" + summary)
    Path(f"outputs/phase1b_codec_r{args.R}_s{args.seed}.txt").write_text(summary)


if __name__ == "__main__":
    main()
