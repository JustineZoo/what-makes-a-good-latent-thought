"""Phase 1b Stage 2: Train flow-matching AR generator over frozen LoRA LLM codec."""
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
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from dlm.data.gsm8k_aug import GSM8KExample, load_gsm8k_aug_splits
from dlm.eval.gsm8k_metrics import score_prediction, summarize_results
from dlm.models.llm_codec import LLMCodecConfig, LLMLatentCodec, LLMLatentGenerator
from dlm.train.train_llm_codec import CodecGSM8KDataset, make_codec_collate


def load_frozen_codec(ckpt_path: str, device: torch.device, base_dtype) -> LLMLatentCodec:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = LLMCodecConfig(**payload["cfg"])
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    def _base():
        m = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=base_dtype)
        m.config.use_cache = False
        return get_peft_model(m, lora_cfg)

    codec = LLMLatentCodec(_base(), _base(), cfg)
    codec.load_state_dict(payload["state_dict"])
    codec.to(device).eval()
    for p in codec.parameters():
        p.requires_grad_(False)
    return codec


@torch.no_grad()
def evaluate_generator(
    gen: LLMLatentGenerator,
    codec: LLMLatentCodec,
    tokenizer,
    examples: List[GSM8KExample],
    device: torch.device,
    batch_size: int = 16,
    max_q_len: int = 128,
    max_cot_len: int = 160,
    amp_dtype=torch.float16,
) -> Dict[str, object]:
    gen.eval()
    codec.eval()
    ds = CodecGSM8KDataset(examples, tokenizer, max_q_len=max_q_len, max_cot_len=max_cot_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=make_codec_collate(tokenizer.pad_token_id, codec.cfg.R))

    tf_results, fr_results = [], []
    first_emit_hits, first_emit_total = 0, 0
    latent_counts, ctx_lens = [], []
    t_eval_start = time.perf_counter()

    for batch in loader:
        q_ids, q_mask = batch["q_ids"].to(device), batch["q_mask"].to(device)
        cot_ids, cot_mask = batch["cot_ids"].to(device), batch["cot_mask"].to(device)
        out_len = batch["labels"].shape[1]

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            gold_z = codec.encode_standardized(cot_ids, cot_mask)
            n_lat = gold_z.shape[1]

            tf_z = gen.teacher_forced_sample(q_ids, q_mask, gold_z)
            tf_logits = codec.decode_logits(codec.unstandardize(tf_z), q_ids, q_mask, out_len=out_len)

            # First-emission probe: only z_0 is generated, remaining latents gold
            fe_z = gold_z.clone()
            fe_z[:, 0, :] = tf_z[:, 0, :]
            fe_logits = codec.decode_logits(codec.unstandardize(fe_z), q_ids, q_mask, out_len=out_len)

            fr_z = gen.sample(q_ids, q_mask, n_latents=n_lat)
            latent_counts.extend([n_lat] * q_ids.shape[0])
            ctx_lens.extend([int(q_ids.shape[1] + n_lat)] * q_ids.shape[0])
            fr_logits = codec.decode_logits(codec.unstandardize(fr_z), q_ids, q_mask, out_len=out_len)

        tf_ids = tf_logits.argmax(dim=-1).cpu()
        fe_ids = fe_logits.argmax(dim=-1).cpu()
        fr_ids = fr_logits.argmax(dim=-1).cpu()

        for t_row, f_row, r_row, ex in zip(tf_ids, fe_ids, fr_ids, batch["examples"]):
            tf_res = score_prediction(tokenizer.decode(t_row, skip_special_tokens=True), ex.steps, ex.answer)
            fe_res = score_prediction(tokenizer.decode(f_row, skip_special_tokens=True), ex.steps, ex.answer)
            fr_res = score_prediction(tokenizer.decode(r_row, skip_special_tokens=True), ex.steps, ex.answer)
            tf_results.append(tf_res)
            fr_results.append(fr_res)
            first_emit_hits += int(fe_res.first_error_pos != 0)
            first_emit_total += 1

    eval_elapsed = max(1e-9, time.perf_counter() - t_eval_start)
    n_eval = max(1, len(fr_results))
    mean_lat = sum(latent_counts) / n_eval
    gen.train()
    return {
        "teacher_forced": summarize_results(tf_results),
        "first_emission_step0_acc": first_emit_hits / max(1, first_emit_total),
        "free_running": summarize_results(fr_results),
        "mean_latent_steps": mean_lat,
        "mean_forward_passes": mean_lat * gen.head_inner_steps,
        "mean_context_len": sum(ctx_lens) / n_eval,
        "eval_wall_clock_sec": eval_elapsed,
        "eval_ms_per_example": (eval_elapsed * 1000.0) / n_eval,
    }


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--codec-ckpt", required=True)
    p.add_argument("--epochs", type=int, default=3, help="3 epochs = 3x budget; 1/3 snapshot = 1x budget")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head-inner-steps", type=int, default=4)
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

    codec = load_frozen_codec(args.codec_ckpt, device, base_dtype)
    R = codec.cfg.R
    tokenizer = AutoTokenizer.from_pretrained(codec.cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    splits = load_gsm8k_aug_splits(mode="cot", seed=args.seed)
    train_ds = CodecGSM8KDataset(splits["train"], tokenizer, args.max_q_len, args.max_cot_len)
    loader = DataLoader(
        train_ds,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=make_codec_collate(tokenizer.pad_token_id, R),
        drop_last=True,
    )

    grad_accum = max(1, args.batch_size // args.micro_batch_size)
    total_updates = math.ceil(len(loader) / grad_accum) * args.epochs
    snap_1x, snap_2x = max(1, total_updates // 3), max(1, (2 * total_updates) // 3)

    base_llm = AutoModelForCausalLM.from_pretrained(codec.cfg.model_name, dtype=base_dtype)
    base_llm.config.use_cache = False
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=codec.cfg.lora_r,
        lora_alpha=codec.cfg.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    backbone = get_peft_model(base_llm, lora_cfg)
    for p in backbone.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)
    backbone.enable_input_require_grads()
    backbone.gradient_checkpointing_enable()

    gen = LLMLatentGenerator(backbone, d_z=codec.cfg.d_z, head_inner_steps=args.head_inner_steps).to(device)
    optimizer = torch.optim.AdamW(gen.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not use_bf16))

    update_step, tokens_seen, run_loss = 0, 0, 0.0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for _ in range(1, args.epochs + 1):
        for micro_idx, batch in enumerate(loader, start=1):
            q_ids, q_mask = batch["q_ids"].to(device), batch["q_mask"].to(device)
            cot_ids, cot_mask = batch["cot_ids"].to(device), batch["cot_mask"].to(device)
            tokens_seen += int(cot_mask.sum().item())

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                target_z = codec.encode_standardized(cot_ids, cot_mask)
                loss = gen.fit_step(q_ids, q_mask, target_z) / grad_accum

            scaler.scale(loss).backward()
            run_loss += float(loss.detach())

            if micro_idx % grad_accum == 0 or micro_idx == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1

                if update_step % args.log_every == 0 or update_step == 1:
                    denom = min(args.log_every, update_step)
                    print(f"  [gen R={R}] update {update_step:5d}/{total_updates} | tokens={tokens_seen:,} | flow_loss={run_loss/denom:.4f} | {(time.time()-t0)/60:.1f}m")
                    run_loss = 0.0

                if update_step == snap_1x:
                    torch.save({"R": R, "state_dict": gen.state_dict()}, out_dir / "gen_1x.pt")
                    m_1x = evaluate_generator(gen, codec, tokenizer, splits["test"], device, 16, args.max_q_len, args.max_cot_len, amp_dtype)
                    (out_dir / "metrics_1x.json").write_text(json.dumps(m_1x, indent=2))
                elif update_step == snap_2x:
                    torch.save({"R": R, "state_dict": gen.state_dict()}, out_dir / "gen_2x.pt")

    torch.save({"R": R, "state_dict": gen.state_dict()}, out_dir / "gen_3x.pt")
    m_3x = evaluate_generator(gen, codec, tokenizer, splits["test"], device, 16, args.max_q_len, args.max_cot_len, amp_dtype)
    (out_dir / "metrics_3x.json").write_text(json.dumps(m_3x, indent=2))

    tf, fr = m_3x["teacher_forced"], m_3x["free_running"]
    summary = (
        f"Phase 1b Generator R={R} (3x Budget) | Updates={update_step} | Tokens={tokens_seen:,}\n"
        f"First-Emission Probe (z_0): {m_3x['first_emission_step0_acc']*100:.2f}%\n"
        f"Teacher-Forced Step Acc   : {tf['step_exact_accuracy']*100:.2f}% | Answer Acc = {tf['answer_accuracy']*100:.2f}%\n"
        f"Free-Running Accuracy     : {fr['answer_accuracy']*100:.2f}% | Step Exact = {fr['step_exact_accuracy']*100:.2f}% | "
        f"First-Err = {fr['mean_first_error_pos']:.2f} | Errors = {fr['error_type_counts']}\n"
    )
    print("\n" + summary)
    Path(f"outputs/phase1b_gen_r{R}_s{args.seed}.txt").write_text(summary)


if __name__ == "__main__":
    main()
