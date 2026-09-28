import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'configs/main.json').read_text())


def checkpoint(model_id, root=None):
    from dlm.checkpoints import resolve_checkpoint_path
    if root is not None:
        os.environ['DLM_CHECKPOINT_ROOT'] = str(Path(root).resolve())
    return resolve_checkpoint_path(CONFIG['models'][model_id]['checkpoint'])
