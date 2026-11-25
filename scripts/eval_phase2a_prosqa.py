"""Batch evaluate ProsQA across all Phase 2a checkpoints."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from dlm.data.prosqa import load_prosqa_split
from dlm.train.train_llm_generator import load_frozen_codec
from dlm.train.train_phase2_ablation import CoLaRAblationGenerator, evaluate_ablation


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ProsQA on all Phase 2a Checkpoints")
    p.add_argument("--codec-ckpt", default="outputs/phase1b_codec_r4_s0/codec_final.pt")
    p.add_argument("--prosqa-path", default="data/prosqa/prosqa_test.json")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    use_bf16 = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    base_dtype = amp_dtype if device.type == "cuda" else torch.float32

    print("=================================================================")
    print(" Batch Evaluating ProsQA Across Phase 2a Checkpoints")
    print(f" Device={device} | Base Dtype={base_dtype} | Codec={args.codec_ckpt}")
    print("=================================================================")

    # 1. Load codec & tokenizer
    codec = load_frozen_codec(args.codec_ckpt, device, base_dtype)
    tokenizer = AutoTokenizer.from_pretrained(codec.cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load ProsQA test set
    prosqa_path = args.prosqa_path if Path(args.prosqa_path).exists() else None
    prosqa_exs = load_prosqa_split("test", local_path=prosqa_path)
    print(f" Loaded {len(prosqa_exs)} ProsQA test examples.")

    dirs = [
        Path("outputs/phase2a_colar_s0"),
        Path("outputs/phase2a_colar_s1"),
        Path("outputs/phase2a_colar_s2"),
        Path("outputs/phase2a_colar_c1_s0"),
        Path("outputs/phase2a_colar_c2_s0"),
        Path("outputs/phase2a_colar_c1_c2_s0"),
        Path("outputs/phase2a_colar_c1_c2_s1"),
        Path("outputs/phase2a_colar_c1_c2_s2"),
        Path("outputs/phase2a_colar_c1_c2_c3a_s0"),
    ]

    for idx, d in enumerate(dirs, 1):
        ckpt_path = d / "gen_final.pt"
        if not ckpt_path.exists():
            print(f"[{idx}/{len(dirs)}] Skipping {d.name}: {ckpt_path} not found")
            continue

        print(f"\n[{idx}/{len(dirs)}] Evaluating {d.name}...")
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        system = payload.get("system", "colar")
        cfg = payload.get("cfg", {})
        K = cfg.get("K", 4)
        refine_steps = cfg.get("refine_steps", 4)
        flow_steps = refine_steps if ("c3a" in system) else 4

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
        gen.load_state_dict(payload["state_dict"])
        gen.eval()

        t0 = time.perf_counter()
        res_prosqa = evaluate_ablation(
            gen, codec, tokenizer, prosqa_exs, device, batch_size=args.batch_size, steps=flow_steps, amp_dtype=amp_dtype
        )
        elapsed = time.perf_counter() - t0
        ans_acc = res_prosqa["answer_accuracy"] * 100
        print(f"  -> {d.name}: ProsQA Ans Acc = {ans_acc:.2f}% (wall-clock: {elapsed:.1f}s)")

        # Save to metrics_ood.json
        ood_path = d / "metrics_ood.json"
        ood_data = json.loads(ood_path.read_text()) if ood_path.exists() else {}
        ood_data["prosqa"] = res_prosqa
        ood_path.write_text(json.dumps(ood_data, indent=2))

        # Update summary.txt
        summary_path = d / "summary.txt"
        if summary_path.exists():
            lines = summary_path.read_text().splitlines()
            new_lines = []
            for line in lines:
                if "ProsQA Answer Acc" in line:
                    new_lines.append(f"ProsQA Answer Acc    : {ans_acc:.2f}%")
                else:
                    new_lines.append(line)
            summary_path.write_text("\n".join(new_lines) + "\n")

        # Free GPU memory
        del gen, backbone, base_llm, payload
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=================================================================")
    print(" Regenerating Table 1 with ProsQA numbers...")
    print("=================================================================")
    subprocess.run(["python", "make_p2a_table.py"], check=True)
    print(" All ProsQA evaluations complete and Table 1 updated!")


if __name__ == "__main__":
    main()
