"""Correct-prefix or context-switching diagnostic on a selected main model."""
import argparse
import json
from pathlib import Path
import torch
from reproduce.common import CONFIG, checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=list(CONFIG['models']), required=True)
    p.add_argument('--probe', choices=['prefix', 'context'], required=True)
    p.add_argument('--checkpoint-root', type=Path)
    p.add_argument('--n', type=int, default=512)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if args.n < 1:
        p.error('--n must be positive')
    if args.out.exists():
        raise FileExistsError(args.out)
    path = checkpoint(args.model, args.checkpoint_root)
    meta = torch.load(path, map_location='cpu', weights_only=True)
    torch.set_num_threads(4)
    if args.probe == 'prefix':
        if meta['paradigm'] != 'concept_ar':
            p.error('The prefix probe requires a continuous latent AR model')
        from dlm.eval.tf_probe import probe
        torch.manual_seed(20260924)
        rows = [probe(path, meta['task_kwargs']['max_depth'], args.n, torch.device(args.device), seed=31415)]
    else:
        from dlm.eval.blind_probe import probe_gen
        if meta['task'] != 'graph_reach' or meta['paradigm'] != 'concept_diffusion' or args.n < 2:
            p.error('The context-switching probe is intended for walker diffusion')
        rows = probe_gen(path, [12], 256, [0, 64], args.n, torch.device(args.device), seed=12345)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(model=args.model, probe=args.probe, n=args.n, rows=rows), indent=2)+'\n')


if __name__ == '__main__':
    main()
