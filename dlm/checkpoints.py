"""Locate models and codecs in a downloaded checkpoint repository."""
import os
from pathlib import Path


def resolve_checkpoint_path(path, anchor=None):
    path = Path(path).expanduser()
    if not path.is_absolute() and path.parts and path.parts[0] in ('models', 'codecs'):
        root = os.environ.get('DLM_CHECKPOINT_ROOT')
        if root is None and anchor:
            root = next((p for p in Path(anchor).resolve().parents
                         if (p / 'manifest.json').is_file()), None)
        if root is not None:
            candidate = Path(root).expanduser() / path
            if not candidate.is_file():
                raise FileNotFoundError(f'Missing checkpoint dependency: {candidate}')
            return str(candidate.resolve())
    if path.is_file():
        return str(path.resolve())
    raise FileNotFoundError(f'{path}: set DLM_CHECKPOINT_ROOT to the downloaded HF directory')
