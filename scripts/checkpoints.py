"""Download and verify just the models needed for a main claim."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / 'configs/main.json').read_text())
    manifest = json.loads((ROOT / 'configs/checkpoints.json').read_text())['files']
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['list', 'download', 'verify'])
    select = p.add_mutually_exclusive_group()
    select.add_argument('--claim', choices=['all', *config['claims']], default='all')
    select.add_argument('--model', choices=list(config['models']))
    p.add_argument('--root', type=Path, default=Path('checkpoints'))
    p.add_argument('--repo-id')
    p.add_argument('--revision', default='main')
    args = p.parse_args()
    if args.action == 'list':
        for name, models in config['claims'].items():
            print(name + ': ' + ', '.join(models))
        return
    models = ([args.model] if args.model else list(config['models']) if args.claim == 'all'
              else config['claims'][args.claim])
    files = set()
    for name in models:
        entry = config['models'][name]
        files.update([entry['checkpoint'], *entry['dependencies']])
    if args.action == 'download':
        if not args.repo_id:
            p.error('--repo-id is required')
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=args.repo_id, revision=args.revision, local_dir=str(args.root),
                          allow_patterns=sorted(files) + ['README.md', 'manifest.json', 'SHA256SUMS'])
    for name in sorted(files):
        path = args.root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != manifest[name]['bytes']:
            raise ValueError(f'Size mismatch: {path}')
        h = hashlib.sha256()
        with path.open('rb') as f:
            for data in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(data)
        if h.hexdigest() != manifest[name]['sha256']:
            raise ValueError(f'SHA-256 mismatch: {path}')
    print(f'Verified {len(files)} model/codec files.')


if __name__ == '__main__':
    main()
