import pytest
from dlm.checkpoints import resolve_checkpoint_path


def test_root_is_authoritative(tmp_path, monkeypatch):
    (tmp_path / 'codecs').mkdir()
    (tmp_path / 'codecs/ae.pt').write_bytes(b'old')
    monkeypatch.chdir(tmp_path)
    root = tmp_path / 'download'
    monkeypatch.setenv('DLM_CHECKPOINT_ROOT', str(root))
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint_path('codecs/ae.pt')
    (root / 'codecs').mkdir(parents=True)
    (root / 'codecs/ae.pt').write_bytes(b'new')
    assert resolve_checkpoint_path('codecs/ae.pt') == str(root / 'codecs/ae.pt')


def test_codec_resolves_from_parent_snapshot(tmp_path, monkeypatch):
    monkeypatch.delenv('DLM_CHECKPOINT_ROOT', raising=False)
    (tmp_path / 'manifest.json').write_text('{}')
    (tmp_path / 'codecs').mkdir()
    (tmp_path / 'codecs/ae.pt').write_bytes(b'codec')
    assert resolve_checkpoint_path('codecs/ae.pt', anchor=tmp_path / 'models/ar.pt') == str(tmp_path / 'codecs/ae.pt')
