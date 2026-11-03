"""Locate HyCell weights.

The released weights ship with the package (``checkpoints/``; ``checkpoints/index.json`` names the models, the size
and the sha256 of their weights files). ``weights=`` loads an explicit ``.safetensors`` file whose ``.json`` sidecar
sits next to it, or a directory holding ``weights.safetensors`` and ``weights.json``.
"""
from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def index() -> dict:
    return json.loads(resources.files('hycell').joinpath('checkpoints/index.json').read_text())


def packaged(name: str | None = None) -> Path:
    """Weights file of a model that ships with the package."""
    idx = index()
    name = name or idx['default']
    if name not in idx['models']:
        raise KeyError(f'unknown model {name!r}; available: {", ".join(idx["models"])}')
    path = Path(str(resources.files('hycell').joinpath('checkpoints', idx['models'][name]['weights'])))
    if not path.is_file():
        raise FileNotFoundError(f'{path} is missing; reinstall hycell (the weights are part of the package)')
    return path


def verify(name: str | None = None) -> bool:
    """True if the packaged weights of the model have the size and sha256 of the checkpoint index."""
    idx = index()
    entry = idx['models'][name or idx['default']]
    path = packaged(name)
    return path.stat().st_size == entry['size'] and sha256(path) == entry['sha256']


def resolve(weights=None) -> tuple[Path, dict]:
    """(weights file, sidecar config) for an explicit path, or for a packaged model."""
    if weights is None or str(weights) in index()['models']:
        path = packaged(None if weights is None else str(weights))
    else:
        path = Path(weights).expanduser()
        if path.is_dir():
            path = path / 'weights.safetensors'
        if not path.is_file():
            raise FileNotFoundError(f'no weights at {path}')
    side = path.with_suffix('.json')
    if not side.is_file():
        raise FileNotFoundError(f'{path.name} needs its config sidecar {side.name} next to it')
    meta = json.loads(side.read_text())
    if meta.get('format') != 'hycell-weights':
        raise ValueError(f'{side} is not a HyCell weights sidecar')
    return path, meta
