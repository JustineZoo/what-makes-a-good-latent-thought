"""Retrain a main model, reusing its frozen codec, with an optional new seed."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys
import torch
from dlm.checkpoints import resolve_checkpoint_path
from reproduce.common import CONFIG, ROOT, checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=list(CONFIG['models']), required=True)
    p.add_argument('--checkpoint-root', type=Path)
    p.add_argument('--seed', type=int)
    p.add_argument('--steps', type=int)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    path = checkpoint(args.model, args.checkpoint_root)
    meta = torch.load(path, map_location='cpu', weights_only=True)
    kw, task = meta['kwargs'], meta['task_kwargs']
    walker = meta['task'] == 'graph_reach'
    depths = [2, 4, 8, 12] if walker else [2, 4, 8, 16, 32]
    seed = CONFIG['models'][args.model]['seed'] if args.seed is None else args.seed
    steps = args.steps if args.steps is not None else (75000 if args.model == 's5_code_diffusion' else 25000)
    if steps < 1:
        p.error('--steps must be positive')
    base = ['--task', meta['task'], '--task-n', str(task.get('n', task.get('width', 5))),
            '--task-max-depth', str(task['max_depth']), '--answer-mode', 'trace',
            '--layers', str(kw['n_layers']), '--heads', str(kw['n_heads']),
            '--steps', str(steps), '--batch-size', '128', '--lr', '0.0003',
            '--depths', *map(str, depths), '--seed', str(seed), '--out', str(args.out.resolve())]
    if meta['paradigm'].startswith('code_token'):
        module = 'dlm.train.train_code_token'
        extra = ['--gen', meta['kind'], '--d', str(kw['d']), '--fsq',
                 resolve_checkpoint_path(meta['fsq_path'], anchor=path),
                 '--objective', kw.get('objective', 'uniform')]
    elif meta['kind'] == 'token_ar':
        module = 'dlm.train.train_token'
        extra = ['--gen', 'token_ar', '--d', str(kw['d']), '--group', 's5']
    else:
        module = 'dlm.train.train_generator'
        extra = ['--gen', meta['kind'], '--ae', resolve_checkpoint_path(meta['ae_path'], anchor=path),
                 '--group', 'walkerids' if walker else task.get('group', 's5'),
                 '--model-dim', str(kw.get('model_dim') or 0)]
        if meta.get('cond_ae_path'):
            extra += ['--cond-ae', resolve_checkpoint_path(meta['cond_ae_path'], anchor=path)]
        if meta['kind'] == 'ar':
            extra += ['--head', kw['head'], '--head-inner-steps', str(kw['head_inner_steps'])]
        else:
            extra += ['--block', kw.get('block', 'adaln')]
    cmd = [sys.executable, '-m', module, *base, *extra]
    print(shlex.join(cmd), flush=True)
    if not args.dry_run:
        if not torch.cuda.is_available():
            raise RuntimeError('Full training requires a CUDA GPU')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
