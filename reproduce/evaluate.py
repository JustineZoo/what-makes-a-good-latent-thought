"""Evaluate a main claim on shared, newly sampled synthetic programs."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from dlm.concept.discrete import load_codec
from dlm.checkpoints import resolve_checkpoint_path
from dlm.eval.accuracy import load_generator, load_token_generator, _task_from_meta
from dlm.generators.token_baselines import TokenAR, TokenMaskedDiffusion
from reproduce.common import CONFIG, checkpoint


def trace_metrics(task, prompt, predicted, depth):
    gold = task.oracle(prompt)
    final = task.check(prompt, predicted)
    
    # Handle single-token classification / query tasks (e.g., pronto_qa)
    if not hasattr(task, "n") or task.answer_len == 1:
        acc = float(final.mean())
        return dict(
            final_state_accuracy=acc,
            exact_trace_accuracy=acc,
            state_accuracy=[acc],
            prefix_survival=[acc],
            prefix_survival_factor=1.0,
        )

    active = task.n * depth
    token_ok = predicted[:, :active] == gold[:, :active]
    state_ok = token_ok.reshape(len(prompt), depth, task.n).all(2)
    assert np.array_equal(final, state_ok[:, -1])
    survival = np.logical_and.accumulate(state_ok, axis=1).mean(0)
    mask = survival > 0.02
    factor = (
        float(np.exp(np.polyfit(np.arange(1, depth + 1)[mask], np.log(survival[mask]), 1)[0]))
        if mask.sum() >= 2
        else None
    )
    return dict(
        final_state_accuracy=float(final.mean()),
        exact_trace_accuracy=float(survival[-1]),
        state_accuracy=state_ok.mean(0).tolist(),
        prefix_survival=survival.tolist(),
        prefix_survival_factor=factor,
    )


@torch.no_grad()
def evaluate(model_id, args):
    path = checkpoint(model_id, args.checkpoint_root)
    meta = torch.load(path, map_location='cpu', weights_only=True)
    device = torch.device(args.device)
    task = _task_from_meta(meta)
    paradigm = meta['paradigm']
    codec = cond = None
    if paradigm == 'token_ar':
        gen, _ = load_token_generator(path, device)
    elif paradigm.startswith('code_token'):
        codec = load_codec(resolve_checkpoint_path(meta['fsq_path'], anchor=path), device).to(device).eval()
        gen = (TokenAR if meta['kind'] == 'ar' else TokenMaskedDiffusion)(**meta['kwargs']).to(device).eval()
        gen.load_state_dict(meta['state_dict'])
    else:
        gen, codec, cond, _ = load_generator(path, device)
    depths = args.depths or (
        [1, 2, 4, 6] if meta['task'] == 'pronto_qa'
        else [2, 4, 8, 12] if meta['task'] == 'graph_reach'
        else [2, 16, 32]
    )
    if any(d > task.max_depth or d < 1 for d in depths):
        raise ValueError(f'Invalid depths for {model_id}: {depths}')
    steps = args.T if 'diffusion' in paradigm else [0]
    rows = []
    for depth in depths:
        data = task.sample(args.n, depth=depth, rng=np.random.default_rng(args.data_seed + depth))
        gold_metrics = None
        if codec is not None:
            reconstructed = []
            for lo in range(0, args.n, args.batch_size):
                gold = torch.as_tensor(data.answer[lo:lo+args.batch_size], device=device)
                if paradigm.startswith('code_token'):
                    logits = codec.decode_ids(codec.encode_ids(gold), out_len=task.answer_len)
                else:
                    logits = codec.decode(codec.encode(gold, sample=False), out_len=task.answer_len)
                reconstructed.append(logits.argmax(-1).cpu().numpy())
            gold_metrics = trace_metrics(task, data.prompt, np.concatenate(reconstructed), depth)
        for T in steps:
            torch.manual_seed(args.sampling_seed + depth)
            predicted = []
            for lo in range(0, args.n, args.batch_size):
                prompt = torch.as_tensor(data.prompt[lo:lo+args.batch_size], device=device)
                if paradigm == 'token_ar':
                    pred = gen.sample(prompt, n_answer=task.answer_len, temperature=0)
                elif paradigm.startswith('code_token'):
                    n_codes = math.ceil(task.answer_len / codec.R) * codec.cfg.n_slots
                    ids = gen.sample(codec.encode_ids(prompt), n_answer=n_codes, **({'T': T} if T else {}))
                    pred = codec.decode_ids(ids, out_len=task.answer_len).argmax(-1)
                else:
                    z = gen.sample(cond.encode(prompt, sample=False), n_target=math.ceil(task.answer_len / codec.R),
                                   **({'T': T} if T else {}))
                    pred = codec.decode(z, out_len=task.answer_len).argmax(-1)
                predicted.append(pred.cpu().numpy())
            row = dict(model=model_id, training_seed=CONFIG['models'][model_id]['seed'], depth=depth, T=T,
                       corpus_sha256=hashlib.sha256(data.prompt.tobytes()+data.answer.tobytes()).hexdigest(),
                       **trace_metrics(task, data.prompt, np.concatenate(predicted), depth))
            if gold_metrics is not None:
                row['gold_reconstruction_accuracy'] = gold_metrics['final_state_accuracy']
            rows.append(row)
            print(f"{model_id:26} D={depth:2} T={T:3} accuracy={row['final_state_accuracy']:.4f}", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    select = p.add_mutually_exclusive_group()
    select.add_argument('--claim', choices=list(CONFIG['claims']))
    select.add_argument('--model', choices=list(CONFIG['models']))
    p.add_argument('--list', action='store_true')
    p.add_argument('--checkpoint-root', type=Path)
    p.add_argument('--out', type=Path, default=Path('outputs/results.json'))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--n', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--depths', type=int, nargs='+')
    p.add_argument('--T', type=int, nargs='+', default=[256])
    p.add_argument('--data-seed', type=int, default=12345)
    p.add_argument('--sampling-seed', type=int, default=20260928)
    args = p.parse_args()
    if args.list:
        for claim, models in CONFIG['claims'].items():
            print(claim + ': ' + ', '.join(models))
        return
    if not (args.claim or args.model):
        p.error('Choose --claim or --model')
    if args.n < 1 or args.batch_size < 1 or any(t < 1 for t in args.T):
        p.error('n, batch-size and T must be positive')
    if args.out.exists():
        raise FileExistsError(f'{args.out} already exists; choose a new --out')
    torch.set_num_threads(4)
    models = CONFIG['claims'][args.claim] if args.claim else [args.model]
    rows = [row for model in models for row in evaluate(model, args)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(protocol={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                                       rows=rows), indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
