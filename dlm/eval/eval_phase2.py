"""Standalone evaluator for Phase 2a CoLaR ablation checkpoints.

Evaluates any trained generator checkpoint on:
  - GSM8K-Aug official test set (1,319)
  - SVAMP (1,000)
  - GSM-Hard (1,319)
  - MultiArith (600)
  - ProsQA (500)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

from dlm.data.gsm8k_aug import load_gsm8k_aug_splits
from dlm.data.ood_math import load_ood_math_split
from dlm.data.prosqa import load_prosqa_split
from dlm.train.train_llm_generator import load_frozen_codec
from dlm.train.train_phase2_ablation import CoLaRAblationGenerator, evaluate_ablation


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Phase 2a Checkpoints")
    p.add_argument("--ckpt", required=True, help="Path to generator checkpoint (e.g. gen_final.pt)")
    p.add_argument("--codec-ckpt", default="outputs/phase1b_codec_r4_s0/codec_final.pt", help="Path to frozen codec checkpoint")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=None, help="Output directory for metrics (default: checkpoint directory)")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    base_dtype = amp_dtype if device.type == "cuda" else torch.float32

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=================================================================")
    print(f" Evaluating Checkpoint: {ckpt_path}")
    print(f" Device={device} | Base Dtype={base_dtype} | Output={out_dir}")
    print(f"=================================================================")

    # 1. Load frozen codec
    codec = load_frozen_codec(args.codec_ckpt, device, base_dtype)
    tokenizer = AutoTokenizer.from_pretrained(codec.cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load checkpoint payload
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    system = payload.get("system", "colar")
    cfg = payload.get("cfg", {})
    K = cfg.get("K", 4)
    refine_steps = cfg.get("refine_steps", 4)
    flow_steps = refine_steps if ("c3a" in system) else 4

    print(f" System: {system} | K: {K} | Head Inner Steps: {flow_steps}")

    # 3. Build generator backbone
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
    gen = CoLaRAblationGenerator(
        backbone, system=system, d_z=codec.cfg.d_z, K=K, head_inner_steps=flow_steps
    ).to(device)

    # 4. Load weights
    gen.load_state_dict(payload["state_dict"])
    gen.eval()
    print(" Checkpoint weights loaded successfully into generator.")

    # 5. Evaluate on GSM8K-Aug test split
    print("\n--- [1/5] Evaluating on GSM8K-Aug Test Split (1,319 examples) ---")
    splits = load_gsm8k_aug_splits(mode="cot", seed=args.seed)
    gsm8k_metrics = evaluate_ablation(
        gen, codec, tokenizer, splits["test"], device, batch_size=args.batch_size, steps=flow_steps, amp_dtype=amp_dtype
    )
    (out_dir / "metrics_gsm8k.json").write_text(json.dumps(gsm8k_metrics, indent=2))
    print(f" GSM8K Test Ans Acc: {gsm8k_metrics['answer_accuracy']*100:.2f}% | Step Acc: {gsm8k_metrics['step_exact_accuracy']*100:.2f}%")

    # 6. Evaluate on OOD Math
    print("\n--- [2/5] Evaluating OOD Benchmarks ---")
    ood_results: Dict[str, object] = {}
    for ds_name in ["svamp", "gsm_hard", "multiarith"]:
        exs = load_ood_math_split(ds_name)
        print(f" Evaluating {ds_name} ({len(exs)} examples)...")
        res = evaluate_ablation(
            gen, codec, tokenizer, exs, device, batch_size=args.batch_size, steps=flow_steps, amp_dtype=amp_dtype
        )
        ood_results[ds_name] = res
        print(f"  -> {ds_name} Ans Acc: {res['answer_accuracy']*100:.2f}%")

    # 7. Evaluate on ProsQA
    try:
        prosqa_exs = load_prosqa_split("test")
        print(f" Evaluating ProsQA ({len(prosqa_exs)} examples)...")
        res_prosqa = evaluate_ablation(
            gen, codec, tokenizer, prosqa_exs, device, batch_size=args.batch_size, steps=flow_steps, amp_dtype=amp_dtype
        )
        ood_results["prosqa"] = res_prosqa
        print(f"  -> ProsQA Ans Acc: {res_prosqa['answer_accuracy']*100:.2f}%")
    except Exception as e:
        print(f" Warning: ProsQA evaluation failed with error: {e}")

    (out_dir / "metrics_ood.json").write_text(json.dumps(ood_results, indent=2))

    prosqa_acc = ood_results.get("prosqa", {}).get("answer_accuracy", 0.0)
    summary = (
        f"=================================================================\n"
        f" Evaluation Summary for {system} (Ckpt: {ckpt_path.name})\n"
        f"=================================================================\n"
        f" GSM8K-Aug Test Ans Acc: {gsm8k_metrics['answer_accuracy']*100:.2f}%\n"
        f" GSM8K-Aug Step Acc    : {gsm8k_metrics['step_exact_accuracy']*100:.2f}%\n"
        f" SVAMP Ans Acc         : {ood_results.get('svamp', {}).get('answer_accuracy', 0.0)*100:.2f}%\n"
        f" GSM-Hard Ans Acc      : {ood_results.get('gsm_hard', {}).get('answer_accuracy', 0.0)*100:.2f}%\n"
        f" MultiArith Ans Acc    : {ood_results.get('multiarith', {}).get('answer_accuracy', 0.0)*100:.2f}%\n"
        f" ProsQA Ans Acc        : {prosqa_acc*100:.2f}%\n"
        f" Wall-clock Eval Time  : {gsm8k_metrics['wall_clock_sec']:.1f}s\n"
        f"=================================================================\n"
    )
    print("\n" + summary)
    (out_dir / "summary.txt").write_text(summary)


if __name__ == "__main__":
    main()
