"""Phase 1c: Reconstruction-vs-generation probe on CoLaR compressed embeddings."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dlm.concept.decoder import BottleneckDecoder
from dlm.data.gsm8k_aug import GSM8KExample, load_gsm8k_aug_splits
from dlm.eval.gsm8k_metrics import score_prediction, summarize_results


class CoLaRProbeDecoder(nn.Module):
    """Maps CoLaR compressed latent embeddings (B, m, H) -> original CoT token logits (B, L, V)."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        d_probe: int = 512,
        max_tokens: int = 256,
        n_layers: int = 3,
        n_heads: int = 8,
    ):
        super().__init__()
        self.in_proj = nn.Linear(hidden_size, d_probe) if hidden_size != d_probe else nn.Identity()
        self.decoder = BottleneckDecoder(
            vocab_size=vocab_size,
            d_concept=d_probe,
            max_tokens=max_tokens,
            n_layers=n_layers,
            n_heads=n_heads,
            blockwise=False,
        )

    def forward(self, z_embeds: torch.Tensor, out_len: int) -> torch.Tensor:
        return self.decoder(self.in_proj(z_embeds.float()), out_len=out_len)


def compress_embeddings_sqrt_mean(
    token_embeds: torch.Tensor,
    attn_mask: torch.Tensor,
    token_ids: torch.Tensor,
    c: int,
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replicate CoLaR's exact sqrt-mean embedding compression for factor c."""
    B, L, H = token_embeds.shape
    if c == 1:
        return token_embeds, attn_mask, token_ids

    pad = (-L) % c
    if pad > 0:
        token_embeds = F.pad(token_embeds, (0, 0, 0, pad), value=0.0)
        attn_mask = F.pad(attn_mask, (0, pad), value=0)
        token_ids = F.pad(token_ids, (0, pad), value=pad_id)

    m = token_embeds.shape[1] // c
    masked_emb = token_embeds * attn_mask.unsqueeze(-1).to(token_embeds.dtype)
    summed_emb = masked_emb.view(B, m, c, H).sum(dim=2)
    counts = attn_mask.view(B, m, c).sum(dim=2).float()
    denom = counts.sqrt().unsqueeze(-1) + 1e-5
    comp_embeds = summed_emb / denom.to(summed_emb.dtype)
    comp_mask = (counts > 0).long()
    return comp_embeds, comp_mask, token_ids


class ProbeTraceDataset(Dataset):
    def __init__(self, examples: List[GSM8KExample], tokenizer, max_cot_len: int = 192):
        self.items = []
        eos_id = tokenizer.eos_token_id
        for ex in examples:
            trace_str = " ".join(ex.steps) if ex.steps else ""
            t_ids = (tokenizer.encode(trace_str, add_special_tokens=False) + [eos_id])[:max_cot_len]
            if t_ids:
                self.items.append((t_ids, ex))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def make_probe_collate(pad_id: int):
    def collate(batch):
        max_l = max(len(x[0]) for x in batch)
        ids, mask, labels, exs = [], [], [], []
        for t, ex in batch:
            p = max_l - len(t)
            ids.append(t + [pad_id] * p)
            mask.append([1] * len(t) + [0] * p)
            labels.append(t + [-100] * p)
            exs.append(ex)
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "examples": exs,
        }
    return collate


def load_colar_model(colar_dir: str, ckpt_path: str, hparams_path: str, device: torch.device, dtype):
    sys.path.insert(0, str(Path(colar_dir).resolve()))
    from omegaconf import OmegaConf
    from src.models.colar import LitCoLaR

    cfg = OmegaConf.load(hparams_path).all_config
    cfg.args = OmegaConf.create({"workspace_path": str(Path(colar_dir).resolve()), "no_log": True})
    model = LitCoLaR(model_kwargs=cfg.model.model_kwargs, training_kwargs=cfg.model.training_kwargs, all_config=cfg)
    if not hasattr(model.tokenizer, "batch_encode_plus"):
        model.tokenizer.batch_encode_plus = model.tokenizer
    if not hasattr(model.tokenizer, "encode_plus"):
        model.tokenizer.encode_plus = model.tokenizer
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=False)
    model.to(device=device, dtype=dtype).eval()
    model.latent_policy.to(dtype=torch.float32)
    orig_lp_forward = model.latent_policy.forward
    def _fp32_lp_forward(x, *args, **kwargs):
        dist = orig_lp_forward(x.float(), *args, **kwargs)
        orig_rsample = dist.rsample
        dist.rsample = lambda *a, **kw: orig_rsample(*a, **kw).to(dtype)
        return dist
    model.latent_policy.forward = _fp32_lp_forward
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def teacher_forced_colar_embeds(
    colar_model,
    questions: List[str],
    gold_comp_embeds: torch.Tensor,
    gold_comp_mask: torch.Tensor,
    c: int,
) -> torch.Tensor:
    """Predict each compressed latent embedding from gold question + gold compressed prefix."""
    from src.utils.utils import get_position_ids_from_attention_mask

    suffix = colar_model.speed_template.format(c) + colar_model.thinking_separator
    q_ids, q_mask = colar_model.prepare_inputs(questions, padding_side="left", part="question", suffix=suffix)
    q_emb = colar_model.embedding(q_ids)

    q_len = q_emb.shape[1]
    m = gold_comp_embeds.shape[1]
    prefix_emb = gold_comp_embeds[:, :-1, :].to(q_emb.dtype) if m > 1 else q_emb[:, :0, :]
    prefix_mask = gold_comp_mask[:, :-1] if m > 1 else q_mask[:, :0]

    seq_emb = torch.cat([q_emb, prefix_emb], dim=1)
    seq_mask = torch.cat([q_mask, prefix_mask], dim=1)
    pos_ids = get_position_ids_from_attention_mask(seq_mask)

    out = colar_model.llm.forward(
        inputs_embeds=seq_emb,
        attention_mask=seq_mask,
        position_ids=pos_ids,
        output_hidden_states=True,
    )
    h_for_latents = out.hidden_states[-1][:, q_len - 1 : q_len + m - 1, :]
    dist = colar_model.latent_policy.forward(h_for_latents, temperature=1.0)
    return dist.mean * colar_model.embeds_std


def evaluate_factor_c(
    colar_model,
    probe: CoLaRProbeDecoder,
    examples: List[GSM8KExample],
    c: int,
    device: torch.device,
    batch_size: int = 8,
    max_cot_len: int = 192,
) -> Dict[str, object]:
    probe.eval()
    colar_model.model_kwargs.latent_generation_config.compression_factor = c
    tok = colar_model.tokenizer
    embed_layer = colar_model.embedding

    ds = ProbeTraceDataset(examples, tok, max_cot_len=max_cot_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=make_probe_collate(tok.pad_token_id))

    recon_res, tf_res, fr_res = [], [], []
    for batch in loader:
        ids = batch["ids"].to(device)
        mask = batch["mask"].to(device)
        out_len = ids.shape[1]
        questions = [ex.question for ex in batch["examples"]]

        with torch.no_grad():
            tok_emb = embed_layer(ids)
            gold_comp, gold_mask, _ = compress_embeddings_sqrt_mean(tok_emb, mask, ids, c=c, pad_id=tok.pad_token_id)
            tf_comp = teacher_forced_colar_embeds(colar_model, questions, gold_comp, gold_mask, c=c)

            gen_out = colar_model.latent_generate(questions=questions, rl_mode=True)
            fr_comp = gen_out[2]
            m = gold_comp.shape[1]
            if fr_comp.shape[1] < m:
                fr_comp = F.pad(fr_comp, (0, 0, 0, m - fr_comp.shape[1]))
            else:
                fr_comp = fr_comp[:, :m, :]

            recon_ids = probe(gold_comp, out_len=out_len).argmax(dim=-1).cpu()
            tf_ids = probe(tf_comp, out_len=out_len).argmax(dim=-1).cpu()
            fr_ids = probe(fr_comp, out_len=out_len).argmax(dim=-1).cpu()

        for r_row, t_row, f_row, ex in zip(recon_ids, tf_ids, fr_ids, batch["examples"]):
            recon_res.append(score_prediction(tok.decode(r_row, skip_special_tokens=True), ex.steps, ex.answer))
            tf_res.append(score_prediction(tok.decode(t_row, skip_special_tokens=True), ex.steps, ex.answer))
            fr_res.append(score_prediction(tok.decode(f_row, skip_special_tokens=True), ex.steps, ex.answer))

    recon_sum = summarize_results(recon_res)
    tf_sum = summarize_results(tf_res)
    fr_sum = summarize_results(fr_res)

    ceiling_step = recon_sum["step_exact_accuracy"]
    fr_step = fr_sum["step_exact_accuracy"]
    return {
        "compression_factor": c,
        "reconstruction_ceiling": recon_sum,
        "teacher_forced": tf_sum,
        "free_running": fr_sum,
        "error_decomposition": {
            "reconstruction_drop": 1.0 - ceiling_step,
            "generation_drop": max(0.0, ceiling_step - fr_step),
            "total_step_drop": 1.0 - fr_step,
        },
    }


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--colar-dir", default="/mnt/beegfs/bulk/mirror/jsz45/colar")
    p.add_argument("--ckpt", default="/mnt/beegfs/bulk/mirror/jsz45/colar/logs/colar/qsa-gsm/colar-final/checkpoints/colar_best.ckpt")
    p.add_argument("--hparams", default="/mnt/beegfs/bulk/mirror/jsz45/colar/logs/colar/qsa-gsm/colar-final/hparams.yaml")
    p.add_argument("--factors", type=int, nargs="+", default=[1, 2, 5])
    p.add_argument("--probe-steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="outputs/phase1c_colar_probe")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 and torch.cuda.is_bf16_supported()
    base_dtype = (torch.bfloat16 if use_bf16 else torch.float16) if device.type == "cuda" else torch.float32

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    colar_model = load_colar_model(args.colar_dir, args.ckpt, args.hparams, device, base_dtype)
    tok = colar_model.tokenizer
    embed_layer = colar_model.embedding
    hidden_size = colar_model.llm.config.hidden_size

    splits = load_gsm8k_aug_splits(mode="cot", seed=args.seed)
    train_ds = ProbeTraceDataset(splits["train"], tok, max_cot_len=192)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=make_probe_collate(tok.pad_token_id), drop_last=True)

    all_reports = {}
    lines = [
        "================================================================",
        " Phase 1c: CoLaR Reconstruction vs. Generation Probe (GSM8K)",
        "================================================================",
    ]

    for c in args.factors:
        probe = CoLaRProbeDecoder(vocab_size=len(tok), hidden_size=hidden_size, d_probe=512, max_tokens=224).to(device)
        ckpt_file = out_dir / f"probe_c{c}.pt"
        if ckpt_file.exists():
            probe.load_state_dict(torch.load(ckpt_file, map_location=device, weights_only=True))
            print(f"  [colar_probe c={c}] loaded existing {ckpt_file}")
            res_c = evaluate_factor_c(colar_model, probe, splits["test"], c=c, device=device, batch_size=8)
            all_reports[f"c={c}"] = res_c
            dec = res_c["error_decomposition"]
            lines.append(
                f"c={c} | Recon Ceiling Step={res_c['reconstruction_ceiling']['step_exact_accuracy']*100:5.2f}% "
                f"| TF Step={res_c['teacher_forced']['step_exact_accuracy']*100:5.2f}% "
                f"| Free-Run Step={res_c['free_running']['step_exact_accuracy']*100:5.2f}% "
                f"(Recon Drop={dec['reconstruction_drop']*100:5.2f}%, Gen Drop={dec['generation_drop']*100:5.2f}%)"
            )
            continue
        opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=0.01)

        step = 0
        probe.train()
        while step < args.probe_steps:
            for batch in train_loader:
                ids = batch["ids"].to(device)
                mask = batch["mask"].to(device)
                labels = batch["labels"].to(device)
                with torch.no_grad():
                    tok_emb = embed_layer(ids)
                    gold_comp, _, _ = compress_embeddings_sqrt_mean(tok_emb, mask, ids, c=c, pad_id=tok.pad_token_id)

                logits = probe(gold_comp, out_len=labels.shape[1])
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
                opt.step()
                step += 1
                if step % 500 == 0 or step == 1:
                    print(f"  [colar_probe c={c}] step {step:4d}/{args.probe_steps} | loss={float(loss.detach()):.4f}")
                if step >= args.probe_steps:
                    break

        torch.save(probe.state_dict(), out_dir / f"probe_c{c}.pt")
        res_c = evaluate_factor_c(colar_model, probe, splits["test"], c=c, device=device, batch_size=8)
        all_reports[f"c={c}"] = res_c
        dec = res_c["error_decomposition"]
        lines.append(
            f"c={c} | Recon Ceiling Step={res_c['reconstruction_ceiling']['step_exact_accuracy']*100:5.2f}% "
            f"| TF Step={res_c['teacher_forced']['step_exact_accuracy']*100:5.2f}% "
            f"| Free-Run Step={res_c['free_running']['step_exact_accuracy']*100:5.2f}% "
            f"(Recon Drop={dec['reconstruction_drop']*100:5.2f}%, Gen Drop={dec['generation_drop']*100:5.2f}%)"
        )

    lines.append("================================================================")
    summary = "\n".join(lines) + "\n"
    print("\n" + summary)
    (out_dir / "colar_probe_metrics.json").write_text(json.dumps(all_reports, indent=2))
    Path("outputs/phase1c_colar_probe.txt").write_text(summary)


if __name__ == "__main__":
    main()
