"""Phase 2a: Core Ablation Grid on Llama-3.2-1B-Instruct.

Systems:
  0. colar            : Baseline fixed chunking (c=5), single vector emission
  1. colar_c1         : Step-aligned chunking (c'=10), single vector emission
  2. colar_c2         : Fixed chunking (c=5), K=4 factorized sub-slots
  3. colar_c1_c2      : Step-aligned chunking (c'=10), K=4 factorized sub-slots
  4. colar_c1_c2_c3a  : Step-aligned chunking (c'=10), K=4 factorized sub-slots, T=4 refinement

Evaluates on:
  - GSM8K-Aug official test set (1,319)
  - SVAMP (1,000)
  - GSM-Hard (1,319)
  - MultiArith (600)
  - ProsQA (500)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

from dlm.data.gsm8k_aug import GSM8KExample, load_gsm8k_aug_splits
from dlm.data.chunking import fixed_chunk_tokens, step_aligned_chunk_tokens
from dlm.data.ood_math import load_ood_math_split
from dlm.data.prosqa import load_prosqa_split
from dlm.eval.gsm8k_metrics import score_prediction, summarize_results
from dlm.models.heads import MARHead


class AblationDataset(Dataset):
    """GSM8K dataset formatted for Phase 2 ablations."""

    def __init__(
        self,
        examples: List[GSM8KExample],
        tokenizer,
        system: str,
        c: int = 5,
        c_prime: int = 10,
        max_q_len: int = 128,
        max_cot_len: int = 160,
    ):
        self.tokenizer = tokenizer
        self.system = system
        self.c = c
        self.c_prime = c_prime
        self.max_q_len = max_q_len
        self.max_cot_len = max_cot_len
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        ex = self.examples[idx]
        q_enc = self.tokenizer(
            ex.prompt,
            truncation=True,
            max_length=self.max_q_len,
            add_special_tokens=True,
            return_tensors=None,
        )

        # Chunking strategy:
        use_c1 = "c1" in self.system
        if use_c1 and ex.steps:
            step_encs = [self.tokenizer.encode(f" {s.strip()}", add_special_tokens=False) for s in ex.steps if s.strip()]
            chunks = step_aligned_chunk_tokens(step_encs, c_prime=self.c_prime)
        else:
            cot_enc = self.tokenizer.encode(f" {ex.target.strip()}", add_special_tokens=False)
            chunks = fixed_chunk_tokens(cot_enc, c=self.c)

        # Flatten capped CoT tokens
        flat_tokens = [t for chunk in chunks for t in chunk][: self.max_cot_len]

        return {
            "q_ids": q_enc["input_ids"],
            "cot_ids": flat_tokens,
            "num_chunks": len(chunks),
            "example": ex,
        }


def make_ablation_collate(pad_id: int):
    def collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
        max_q = max(len(b["q_ids"]) for b in batch)
        max_cot = max(len(b["cot_ids"]) for b in batch)
        max_cot = max(1, max_cot)

        q_ids, q_mask = [], []
        cot_ids, cot_mask = [], []
        examples = []
        num_chunks = []

        for b in batch:
            q = b["q_ids"]
            cot = b["cot_ids"]
            q_pad = max_q - len(q)
            cot_pad = max_cot - len(cot)

            q_ids.append(q + [pad_id] * q_pad)
            q_mask.append([1] * len(q) + [0] * q_pad)

            cot_ids.append(cot + [pad_id] * cot_pad)
            cot_mask.append([1] * len(cot) + [0] * cot_pad)

            examples.append(b["example"])
            num_chunks.append(b["num_chunks"])

        return {
            "q_ids": torch.tensor(q_ids, dtype=torch.long),
            "q_mask": torch.tensor(q_mask, dtype=torch.long),
            "cot_ids": torch.tensor(cot_ids, dtype=torch.long),
            "cot_mask": torch.tensor(cot_mask, dtype=torch.long),
            "num_chunks": torch.tensor(num_chunks, dtype=torch.long),
            "examples": examples,
        }

    return collate


class CoLaRAblationGenerator(nn.Module):
    """
    Unified Generator supporting:
      - Vanilla 1-vector emission (c=5)
      - C1: step-aligned load-capped chunking
      - C2: K-slot factorized emission
      - C3a: within-latent iterative refinement (T steps)
    """

    def __init__(
        self,
        backbone: nn.Module,
        system: str,
        d_z: int = 256,
        K: int = 4,
        head_inner_steps: int = 4,
    ):
        super().__init__()
        self.backbone = backbone
        self.system = system
        self.d_z = d_z
        self.K = K if ("c2" in system) else 1
        self.d_sub = d_z // self.K
        self.head_inner_steps = head_inner_steps

        hidden_size = backbone.config.hidden_size
        self.bos_latent = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.in_proj = nn.Linear(d_z, hidden_size)

        if self.K > 1:
            self.sub_heads = nn.ModuleList([MARHead(d=self.d_sub) for _ in range(self.K)])
            self.out_projs = nn.ModuleList([nn.Linear(hidden_size, self.d_sub) for _ in range(self.K)])
        else:
            self.head = MARHead(d=d_z)
            self.out_proj = nn.Linear(hidden_size, d_z)

    def _predict_cond(
        self, q_ids: torch.Tensor, q_mask: torch.Tensor, prefix_z: Optional[torch.Tensor], n_steps: int
    ) -> torch.Tensor:
        B = q_ids.shape[0]
        q_emb = self.backbone.get_input_embeddings()(q_ids)
        bos = self.bos_latent.expand(B, 1, -1).to(q_emb.dtype)
        parts = [q_emb, bos]
        if prefix_z is not None and prefix_z.shape[1] > 0:
            parts.append(self.in_proj(prefix_z).to(q_emb.dtype))
        seq_emb = torch.cat(parts, dim=1)
        extra_len = seq_emb.shape[1] - q_emb.shape[1]
        extra_mask = torch.ones((B, extra_len), device=q_ids.device, dtype=q_mask.dtype)
        seq_mask = torch.cat([q_mask, extra_mask], dim=1)
        pos_ids = (seq_mask.long().cumsum(-1) - 1).clamp(min=0)

        out = self.backbone(
            inputs_embeds=seq_emb,
            attention_mask=seq_mask,
            position_ids=pos_ids,
            output_hidden_states=True,
        )
        return out.hidden_states[-1][:, -n_steps:, :].float()

    def fit_step(
        self, q_ids: torch.Tensor, q_mask: torch.Tensor, target_z: torch.Tensor
    ) -> torch.Tensor:
        B, m, _ = target_z.shape
        prefix = target_z[:, :-1, :] if m > 1 else None
        h = self._predict_cond(q_ids, q_mask, prefix, n_steps=m)

        if self.K > 1:
            # Factorized C2 emission
            z_sub = target_z.view(B, m, self.K, self.d_sub)
            total_loss = 0.0
            for k in range(self.K):
                cond_k = self.out_projs[k](h).reshape(B * m, self.d_sub)
                target_k = z_sub[:, :, k, :].reshape(B * m, self.d_sub)
                total_loss = total_loss + self.sub_heads[k].loss(cond_k, target_k)
            return total_loss / self.K
        else:
            cond = self.out_proj(h).reshape(B * m, self.d_z)
            return self.head.loss(cond, target_z.reshape(B * m, self.d_z))

    @torch.no_grad()
    def sample(
        self, q_ids: torch.Tensor, q_mask: torch.Tensor, n_latents: int, steps: Optional[int] = None
    ) -> torch.Tensor:
        B = q_ids.shape[0]
        n_flow = steps or self.head_inner_steps
        q_emb = self.backbone.get_input_embeddings()(q_ids)
        seq_emb = torch.cat([q_emb, self.bos_latent.expand(B, 1, -1).to(q_emb.dtype)], dim=1)
        seq_mask = torch.cat([q_mask, torch.ones((B, 1), device=q_ids.device, dtype=q_mask.dtype)], dim=1)

        emitted = []
        for _ in range(n_latents):
            pos_ids = (seq_mask.long().cumsum(-1) - 1).clamp(min=0)
            out = self.backbone(
                inputs_embeds=seq_emb,
                attention_mask=seq_mask,
                position_ids=pos_ids,
                output_hidden_states=True,
            )
            h_last = out.hidden_states[-1][:, -1, :].float()

            if self.K > 1:
                subs = [self.sub_heads[k].sample(self.out_projs[k](h_last), steps=n_flow) for k in range(self.K)]
                z_next = torch.cat(subs, dim=-1)
            else:
                z_next = self.head.sample(self.out_proj(h_last), steps=n_flow)

            emitted.append(z_next)
            z_step_emb = self.in_proj(z_next).unsqueeze(1).to(q_emb.dtype)
            seq_emb = torch.cat([seq_emb, z_step_emb], dim=1)
            seq_mask = torch.cat([seq_mask, torch.ones((B, 1), device=q_ids.device, dtype=q_mask.dtype)], dim=1)

        return torch.stack(emitted, dim=1)


@torch.no_grad()
def evaluate_ablation(
    gen: CoLaRAblationGenerator,
    codec,
    tokenizer,
    examples: List[GSM8KExample],
    device: torch.device,
    batch_size: int = 16,
    max_q_len: int = 128,
    max_cot_len: int = 160,
    steps: Optional[int] = None,
    amp_dtype: torch.dtype = torch.float16,
) -> Dict[str, object]:
    gen.eval()
    codec.eval()
    ds = AblationDataset(examples, tokenizer, gen.system, max_q_len=max_q_len, max_cot_len=max_cot_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=make_ablation_collate(tokenizer.pad_token_id))

    all_res = []
    t_start = time.perf_counter()

    for batch in loader:
        q_ids, q_mask = batch["q_ids"].to(device), batch["q_mask"].to(device)
        cot_ids, cot_mask = batch["cot_ids"].to(device), batch["cot_mask"].to(device)
        batch_exs = batch["examples"]
        out_len = max(1, cot_ids.shape[1])

        # Compute required latents for free-running
        R = codec.cfg.R
        n_latents = max(1, math.ceil(out_len / R))

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            fr_z = gen.sample(q_ids, q_mask, n_latents=n_latents, steps=steps)
            logits = codec.decode_logits(codec.unstandardize(fr_z), q_ids, q_mask, out_len=out_len)
        pred_ids = logits.argmax(dim=-1).cpu().tolist()

        for ex, p_ids in zip(batch_exs, pred_ids):
            pred_text = tokenizer.decode(p_ids, skip_special_tokens=True)
            res = score_prediction(pred_text, ex.steps, ex.answer)
            all_res.append(res)

    summary = summarize_results(all_res)
    summary["wall_clock_sec"] = time.perf_counter() - t_start
    summary["n_evaluated"] = len(examples)
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2a: CoLaR Ablation Sweep on Llama-3.2-1B")
    p.add_argument("--system", choices=["colar", "colar_c1", "colar_c2", "colar_c1_c2", "colar_c1_c2_c3a"], required=True)
    p.add_argument("--codec-ckpt", default="outputs/phase1b_codec_r4_s0/codec_final.pt")
    p.add_argument("--c", type=int, default=5, help="Vanilla CoLaR chunk size")
    p.add_argument("--c-prime", type=int, default=10, help="Component C1 chunk cap per step")
    p.add_argument("--K", type=int, default=4, help="Component C2 factorized sub-slots")
    p.add_argument("--refine-steps", type=int, default=4, help="Component C3a denoising steps")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-q-len", type=int, default=128)
    p.add_argument("--max-cot-len", type=int, default=160)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    base_dtype = amp_dtype if device.type == "cuda" else torch.float32

    out_dir = Path(args.out_dir or f"outputs/phase2a_{args.system}_s{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import codec loader
    from dlm.train.train_llm_generator import load_frozen_codec
    codec = load_frozen_codec(args.codec_ckpt, device, base_dtype)
    tokenizer = AutoTokenizer.from_pretrained(codec.cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    splits = load_gsm8k_aug_splits(mode="cot", seed=args.seed)
    train_ds = AblationDataset(
        splits["train"], tokenizer, args.system, c=args.c, c_prime=args.c_prime,
        max_q_len=args.max_q_len, max_cot_len=args.max_cot_len
    )
    collate = make_ablation_collate(tokenizer.pad_token_id)
    loader = DataLoader(train_ds, batch_size=args.micro_batch_size, shuffle=True, collate_fn=collate, drop_last=True)

    grad_accum = max(1, args.batch_size // args.micro_batch_size)
    total_updates = math.ceil(len(loader) / grad_accum) * args.epochs

    # Load generator backbone
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

    flow_steps = args.refine_steps if ("c3a" in args.system) else 4
    gen = CoLaRAblationGenerator(
        backbone, system=args.system, d_z=codec.cfg.d_z, K=args.K, head_inner_steps=flow_steps
    ).to(device)

    optimizer = torch.optim.AdamW(gen.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not use_bf16))

    update_step, tokens_seen, run_loss = 0, 0, 0.0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    print(f"=================================================================")
    print(f" Starting Phase 2a Training: {args.system} (seed={args.seed})")
    print(f" Updates={total_updates} | Micro-Batch={args.micro_batch_size} | Effective Batch={args.batch_size}")
    print(f" Device={device} | Target Out={out_dir}")
    print(f"=================================================================")

    for epoch in range(1, args.epochs + 1):
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
                    print(f"  [{args.system}] update {update_step:5d}/{total_updates} | tokens={tokens_seen:,} | loss={run_loss/denom:.4f} | {(time.time()-t0)/60:.1f}m")
                    run_loss = 0.0

                if update_step % 500 == 0:
                    torch.save({"system": args.system, "state_dict": gen.state_dict(), "cfg": vars(args), "step": update_step}, out_dir / "gen_latest.pt")

    # Save final model
    torch.save({"system": args.system, "state_dict": gen.state_dict(), "cfg": vars(args)}, out_dir / "gen_final.pt")

    # Final Evaluations
    print("\n--- Running Official GSM8K-Aug Test Evaluation ---")
    gsm8k_metrics = evaluate_ablation(gen, codec, tokenizer, splits["test"], device, batch_size=16, steps=flow_steps, amp_dtype=amp_dtype)
    (out_dir / "metrics_gsm8k.json").write_text(json.dumps(gsm8k_metrics, indent=2))

    print("\n--- Running Out-of-Distribution Evaluations (SVAMP, GSM-Hard, MultiArith, ProsQA) ---")
    ood_results = {}
    for ds_name in ["svamp", "gsm_hard", "multiarith"]:
        exs = load_ood_math_split(ds_name)
        ood_results[ds_name] = evaluate_ablation(gen, codec, tokenizer, exs, device, batch_size=16, steps=flow_steps, amp_dtype=amp_dtype)
    try:
        prosqa_path = "data/prosqa/prosqa_test.json" if Path("data/prosqa/prosqa_test.json").exists() else None
        prosqa_exs = load_prosqa_split("test", local_path=prosqa_path)
        ood_results["prosqa"] = evaluate_ablation(gen, codec, tokenizer, prosqa_exs, device, batch_size=16, steps=flow_steps, amp_dtype=amp_dtype)
    except Exception as e:
        print(f"Warning: ProsQA evaluation skipped due to {e}")
    (out_dir / "metrics_ood.json").write_text(json.dumps(ood_results, indent=2))

    prosqa_acc_str = f"{ood_results['prosqa']['answer_accuracy']*100:.2f}%" if "prosqa" in ood_results else "N/A"
    summary = (
        f"Phase 2a System: {args.system} (Seed={args.seed})\n"
        f"Updates: {update_step} | Tokens Seen: {tokens_seen:,}\n"
        f"GSM8K Test Answer Acc: {gsm8k_metrics['answer_accuracy']*100:.2f}% | Step Acc: {gsm8k_metrics['step_exact_accuracy']*100:.2f}%\n"
        f"SVAMP Answer Acc     : {ood_results['svamp']['answer_accuracy']*100:.2f}%\n"
        f"GSM-Hard Answer Acc  : {ood_results['gsm_hard']['answer_accuracy']*100:.2f}%\n"
        f"MultiArith Answer Acc: {ood_results['multiarith']['answer_accuracy']*100:.2f}%\n"
        f"ProsQA Answer Acc    : {prosqa_acc_str}\n"
    )
    print("\n" + summary)
    (out_dir / "summary.txt").write_text(summary)


if __name__ == "__main__":
    main()
