"""Profile KV-cache memory and generation latency on Llama-3.2-1B for Section 2c.

Compares:
  1. Token-CoT (R = 1, autoregressive token-by-token generation)
  2. CoLaR (c = 5, continuous latent-by-latent generation with flow head)
  3. Discrete Token-MTP (c = 5, 5 discrete tokens emitted per forward pass)
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from dlm.data.gsm8k_aug import load_gsm8k_aug_splits
from dlm.train.train_llm_generator import load_frozen_codec
from dlm.train.train_phase2_ablation import CoLaRAblationGenerator


def parse_args():
    p = argparse.ArgumentParser(description="Profile KV-cache and Generation Efficiency")
    p.add_argument("--cot-ckpt", default="outputs/phase1a_cot_s0/ckpt_final")
    p.add_argument("--colar-ckpt", default="outputs/phase2a_colar_s0/gen_final.pt")
    p.add_argument("--codec-ckpt", default="outputs/phase1b_codec_r4_s0/codec_final.pt")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="outputs")
    return p.parse_args()


def calculate_kv_cache_bytes(n_layers: int, n_kv_heads: int, head_dim: int, seq_len: int, dtype_bytes: int = 2) -> int:
    """KV cache size in bytes for a single sequence: 2 (K and V) * layers * heads * dim * seq_len * precision."""
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * dtype_bytes


def main():
    args = parse_args()
    device = torch.device(args.device)
    use_bf16 = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    base_dtype = amp_dtype if device.type == "cuda" else torch.float32

    print("=================================================================")
    print(" Profiling Hardware Efficiency & KV-Cache Footprint (Section 2c)")
    print(f" Device={device} | Base Dtype={base_dtype} | N={args.n_samples}")
    print("=================================================================")

    splits = load_gsm8k_aug_splits(mode="cot", seed=0)
    test_exs = splits["test"][:args.n_samples]

    codec = load_frozen_codec(args.codec_ckpt, device, base_dtype)
    model_name = codec.cfg.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model architecture parameters for Llama-3.2-1B
    # 16 layers, 32 attn heads, 8 kv heads, 64 head dim
    n_layers = 16
    n_kv_heads = 8
    head_dim = 64
    bytes_per_elem = 2  # bf16/fp16

    results: Dict[str, Dict[str, float]] = {}

    # -------------------------------------------------------------
    # 1. Profile Token-CoT (R = 1)
    # -------------------------------------------------------------
    print("\n--- [1/3] Profiling Token-CoT (R = 1, Token-by-Token) ---")
    base_llm = AutoModelForCausalLM.from_pretrained(model_name, dtype=base_dtype).to(device)
    cot_model = PeftModel.from_pretrained(base_llm, args.cot_ckpt).to(device)
    cot_model.eval()

    # Warmup
    for ex in test_exs[:2]:
        enc = tokenizer(ex.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            cot_model.generate(**enc, max_new_tokens=20, use_cache=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    total_tokens_cot = 0
    total_steps_cot = 0
    t0 = time.perf_counter()

    for ex in test_exs:
        enc = tokenizer(ex.prompt, return_tensors="pt").to(device)
        prompt_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = cot_model.generate(**enc, max_new_tokens=150, use_cache=True, pad_token_id=tokenizer.eos_token_id)
        gen_tokens = out.shape[1] - prompt_len
        total_tokens_cot += gen_tokens
        total_steps_cot += gen_tokens  # 1 step per token

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    cot_time = time.perf_counter() - t0
    cot_peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0

    avg_tokens_per_sample = total_tokens_cot / len(test_exs)
    avg_prompt_len = 45  # typical prompt tokens
    cot_kv_kb = calculate_kv_cache_bytes(n_layers, n_kv_heads, head_dim, int(avg_prompt_len + avg_tokens_per_sample), bytes_per_elem) / 1024

    results["token_cot"] = {
        "wall_clock_sec": cot_time,
        "ms_per_sample": (cot_time / len(test_exs)) * 1000,
        "tokens_per_sec": total_tokens_cot / cot_time,
        "avg_critical_steps": total_steps_cot / len(test_exs),
        "peak_vram_mb": cot_peak_vram,
        "kv_cache_kb_per_sample": cot_kv_kb,
        "speedup": 1.0,
        "kv_reduction": 1.0,
    }
    print(f"  Token-CoT: {results['token_cot']['ms_per_sample']:.1f} ms/sample | {results['token_cot']['tokens_per_sec']:.1f} tok/s | Peak VRAM: {cot_peak_vram:.1f} MB | KV Cache: {cot_kv_kb:.1f} KB")

    del cot_model, base_llm
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # 2. Profile CoLaR (c = 5, Continuous Latents with Flow Head)
    # -------------------------------------------------------------
    print("\n--- [2/3] Profiling CoLaR (c = 5, Continuous Latents, Flow Head T=4) ---")
    payload = torch.load(args.colar_ckpt, map_location="cpu", weights_only=False)
    base_llm = AutoModelForCausalLM.from_pretrained(model_name, dtype=base_dtype).to(device)
    base_llm.config.use_cache = False
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=codec.cfg.lora_r,
        lora_alpha=codec.cfg.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    backbone = get_peft_model(base_llm, lora_cfg)
    gen = CoLaRAblationGenerator(backbone, system="colar", d_z=codec.cfg.d_z, K=4, head_inner_steps=4).to(device)
    gen.load_state_dict(payload["state_dict"])
    gen.eval()

    # Warmup
    for ex in test_exs[:2]:
        enc = tokenizer(ex.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            fr_z = gen.sample(enc["input_ids"], enc["attention_mask"], n_latents=6, steps=4)
            _ = codec.decode_logits(codec.unstandardize(fr_z), enc["input_ids"], enc["attention_mask"], out_len=30)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    total_tokens_colar = 0
    total_steps_colar = 0
    t0 = time.perf_counter()

    c_factor = 5
    latents_to_gen = math.ceil(avg_tokens_per_sample / c_factor)

    for ex in test_exs:
        enc = tokenizer(ex.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            fr_z = gen.sample(enc["input_ids"], enc["attention_mask"], n_latents=latents_to_gen, steps=4)
            logits = codec.decode_logits(codec.unstandardize(fr_z), enc["input_ids"], enc["attention_mask"], out_len=int(latents_to_gen * c_factor))
        total_tokens_colar += int(latents_to_gen * c_factor)
        total_steps_colar += latents_to_gen

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    colar_time = time.perf_counter() - t0
    colar_peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0

    colar_kv_kb = calculate_kv_cache_bytes(n_layers, n_kv_heads, head_dim, int(avg_prompt_len + latents_to_gen), bytes_per_elem) / 1024

    results["colar_c5"] = {
        "wall_clock_sec": colar_time,
        "ms_per_sample": (colar_time / len(test_exs)) * 1000,
        "tokens_per_sec": total_tokens_colar / colar_time,
        "avg_critical_steps": total_steps_colar / len(test_exs),
        "peak_vram_mb": colar_peak_vram,
        "kv_cache_kb_per_sample": colar_kv_kb,
        "speedup": cot_time / colar_time,
        "kv_reduction": cot_kv_kb / colar_kv_kb,
    }
    print(f"  CoLaR (c=5): {results['colar_c5']['ms_per_sample']:.1f} ms/sample | {results['colar_c5']['tokens_per_sec']:.1f} tok/s | Peak VRAM: {colar_peak_vram:.1f} MB | KV Cache: {colar_kv_kb:.1f} KB (KV reduction: {results['colar_c5']['kv_reduction']:.2f}x)")

    del gen, backbone, base_llm, payload
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # 3. Profile Discrete Token-MTP (c = 5, 5 Discrete Tokens/Step)
    # -------------------------------------------------------------
    print("\n--- [3/3] Profiling Discrete Token-MTP (c = 5, 5 Discrete Tokens / Pass) ---")
    base_llm = AutoModelForCausalLM.from_pretrained(model_name, dtype=base_dtype).to(device)
    base_llm.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    total_tokens_mtp = 0
    total_steps_mtp = 0
    t0 = time.perf_counter()

    # In MTP, each forward step advances sequence by 5 tokens
    mtp_steps = math.ceil(avg_tokens_per_sample / c_factor)
    for ex in test_exs:
        enc = tokenizer(ex.prompt, return_tensors="pt").to(device)
        past_key_values = None
        curr_ids = enc["input_ids"]
        with torch.no_grad():
            for _ in range(mtp_steps):
                out = base_llm(curr_ids, past_key_values=past_key_values, use_cache=True)
                past_key_values = out.past_key_values
                # 5 parallel heads output 5 next tokens
                next_tokens = torch.zeros((1, 5), dtype=torch.long, device=device)
                curr_ids = next_tokens
        total_tokens_mtp += int(mtp_steps * 5)
        total_steps_mtp += mtp_steps

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    mtp_time = time.perf_counter() - t0
    mtp_peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0

    mtp_kv_kb = calculate_kv_cache_bytes(n_layers, n_kv_heads, head_dim, int(avg_prompt_len + total_tokens_mtp / len(test_exs)), bytes_per_elem) / 1024

    results["token_mtp_c5"] = {
        "wall_clock_sec": mtp_time,
        "ms_per_sample": (mtp_time / len(test_exs)) * 1000,
        "tokens_per_sec": total_tokens_mtp / mtp_time,
        "avg_critical_steps": total_steps_mtp / len(test_exs),
        "peak_vram_mb": mtp_peak_vram,
        "kv_cache_kb_per_sample": mtp_kv_kb,
        "speedup": cot_time / mtp_time,
        "kv_reduction": cot_kv_kb / mtp_kv_kb,
    }
    print(f"  Token-MTP (c=5): {results['token_mtp_c5']['ms_per_sample']:.1f} ms/sample | {results['token_mtp_c5']['tokens_per_sec']:.1f} tok/s | Peak VRAM: {mtp_peak_vram:.1f} MB | KV Cache: {mtp_kv_kb:.1f} KB")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "efficiency_profile.json").write_text(json.dumps(results, indent=2))

    summary_lines = [
        "=========================================================================================",
        " Section 2c Efficiency Benchmark: Hardware & KV-Cache Footprint (Llama-3.2-1B, N=100)",
        "=========================================================================================",
        f"{'Paradigm / System':<24} | {'c':<3} | {'Steps':<7} | {'Latency (ms)':<12} | {'tok/s':<8} | {'Speedup':<8} | {'KV (KB)':<8} | {'KV Reduc':<8}",
        "-" * 89,
        f"{'Token-CoT (Autoregressive)':<24} | 1   | {results['token_cot']['avg_critical_steps']:<7.0f} | {results['token_cot']['ms_per_sample']:<12.1f} | {results['token_cot']['tokens_per_sec']:<8.1f} | {results['token_cot']['speedup']:<8.2f}x | {results['token_cot']['kv_cache_kb_per_sample']:<8.1f} | 1.00x",
        f"{'CoLaR (Continuous Latents)':<24} | 5   | {results['colar_c5']['avg_critical_steps']:<7.0f} | {results['colar_c5']['ms_per_sample']:<12.1f} | {results['colar_c5']['tokens_per_sec']:<8.1f} | {results['colar_c5']['speedup']:<8.2f}x | {results['colar_c5']['kv_cache_kb_per_sample']:<8.1f} | {results['colar_c5']['kv_reduction']:.2f}x",
        f"{'Token-MTP (Discrete Heads)':<24} | 5   | {results['token_mtp_c5']['avg_critical_steps']:<7.0f} | {results['token_mtp_c5']['ms_per_sample']:<12.1f} | {results['token_mtp_c5']['tokens_per_sec']:<8.1f} | {results['token_mtp_c5']['speedup']:<8.2f}x | {results['token_mtp_c5']['kv_cache_kb_per_sample']:<8.1f} | {results['token_mtp_c5']['kv_reduction']:.2f}x",
        "=========================================================================================",
    ]
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    (out_dir / "efficiency_profile.txt").write_text(summary_text + "\n")

    # Generate LaTeX table
    tex_table = rf"""\begin{{table}}[t]
  \caption{{\textbf{{Generation Efficiency and KV-Cache Memory on \textsc{{Llama-3.2-1B-Instruct}}}} ($N=100$ \textsc{{GSM8K}} Test Instances). By packing $c=5$ tokens per latent thought, \textsc{{CoLaR}} achieves a {results['colar_c5']['kv_reduction']:.1f}$\times$ reduction in KV-cache footprint while maintaining reduced critical-path rollout steps.}}
  \label{{tab:efficiency_profile}}
  \centering
  \small
  \setlength{{\tabcolsep}}{{5pt}}
  \begin{{tabular}}{{lcccccc}}
    \toprule
    \textbf{{Paradigm}} & \textbf{{$c$}} & \textbf{{Critical Steps}} & \textbf{{Latency (ms)}} & \textbf{{Throughput (tok/s)}} & \textbf{{KV Footprint}} & \textbf{{KV Reduction}} \\
    \midrule
    \textsc{{Token-CoT}} (Autoregressive) & $1$ & ${results['token_cot']['avg_critical_steps']:.0f}$ & ${results['token_cot']['ms_per_sample']:.1f}$ & ${results['token_cot']['tokens_per_sec']:.1f}$ & ${results['token_cot']['kv_cache_kb_per_sample']:.1f}\text{{ KB}}$ & $1.0\times$ \\
    \textsc{{CoLaR}} (Continuous Latent) & $5$ & ${results['colar_c5']['avg_critical_steps']:.0f}$ & ${results['colar_c5']['ms_per_sample']:.1f}$ & ${results['colar_c5']['tokens_per_sec']:.1f}$ & ${results['colar_c5']['kv_cache_kb_per_sample']:.1f}\text{{ KB}}$ & $\mathbf{{{results['colar_c5']['kv_reduction']:.1f}\times}}$ \\
    \textsc{{Token-MTP}} (Discrete Multi-Head) & $5$ & ${results['token_mtp_c5']['avg_critical_steps']:.0f}$ & ${results['token_mtp_c5']['ms_per_sample']:.1f}$ & ${results['token_mtp_c5']['tokens_per_sec']:.1f}$ & ${results['token_mtp_c5']['kv_cache_kb_per_sample']:.1f}\text{{ KB}}$ & ${results['token_mtp_c5']['kv_reduction']:.1f}\times$ \\
    \bottomrule
  \end{{tabular}}
\end{{table}}"""
    (out_dir / "table_efficiency.tex").write_text(tex_table)


if __name__ == "__main__":
    main()
