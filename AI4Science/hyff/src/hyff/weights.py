"""Weight files: the architecture JSON and single-file or sharded safetensors weights.

A checkpoint is one of
- ``name.safetensors`` (or a tensors-only ``name.pt``) with the architecture in ``name.json``;
- a sharded checkpoint in the Hugging Face layout: ``name.safetensors.index.json`` (``weight_map``
  tensor -> shard file, ``metadata.total_size``) next to its shards, with the architecture in ``name.json``;
- a directory holding exactly one of the above.
The checkpoint digest is the sha256 of the entry file (the weight file or the shard index). An index may
list the sha256 of every shard under ``metadata.sha256``; the shards are then checked when loaded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

INDEX_SUFFIX = '.safetensors.index.json'
_DIGESTS = {}


def sha256(path):
    """sha256 of a file, cached per process while the file (inode, size, mtime) is unchanged."""
    path = Path(path).resolve()
    stat = path.stat()
    key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    digest = _DIGESTS.get(key)
    if digest is None:
        with path.open('rb') as stream:
            digest = _DIGESTS[key] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return digest


def is_index(path):
    return str(path).endswith(INDEX_SUFFIX)


def entry_path(weights):
    """Return the entry file of a checkpoint given as a file or a directory (absolute, not resolved)."""
    path = Path(weights).expanduser().absolute()
    if path.is_dir():
        indexes = sorted(path.glob('*' + INDEX_SUFFIX))
        if len(indexes) > 1:
            raise ValueError(f'{path} holds several sharded indexes; pass one of them')
        if indexes:
            return indexes[0]
        singles = [file for file in sorted(path.glob('*.safetensors')) if file.with_suffix('.json').is_file()]
        if len(singles) == 1:
            return singles[0]
        raise FileNotFoundError(f'{path} does not hold one checkpoint (a .safetensors file with its .json, '
                                f'or a sharded {INDEX_SUFFIX} with its shards)')
    if not (is_index(path) or path.suffix in ('.safetensors', '.pt')):
        raise ValueError(f'weights must be a .safetensors file, a tensors-only .pt file, a sharded '
                         f'{INDEX_SUFFIX} file or a directory holding one of them')
    return path


def config_path(weights):
    """Architecture JSON of a checkpoint entry: ``name.json`` for ``name.safetensors`` or its sharded index."""
    path = Path(weights).expanduser().absolute()
    if is_index(path):
        return path.with_name(path.name[:-len(INDEX_SUFFIX)] + '.json')
    return path.with_suffix('.json')


def read_config(weights):
    # The requested spelling selects the architecture sidecar, even when the
    # weight file is a symlink. Never fall back to a target-side architecture.
    selected = entry_path(weights)
    path = selected.resolve(strict=True)
    if not path.is_file():
        raise FileNotFoundError(path)
    config = json.loads(config_path(selected).read_text())
    allowed = {'format', 'model_id', 'model_config', 'numerical_execution', 'weights_sha256'}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError('Expected a HyFF architecture JSON with only the keys format, model_id, '
                         'model_config, numerical_execution and weights_sha256')
    from hyff.models.factory import FORMAT, LEGACY_FORMATS
    if config.get('format') not in (FORMAT,) + LEGACY_FORMATS:
        raise ValueError(f'Unsupported weights format; expected {FORMAT}')
    if not isinstance(config.get('model_id'), str) or not isinstance(config.get('model_config'), dict):
        raise ValueError('model_id and model_config are required')
    expected = config.get('weights_sha256')
    if expected and sha256(path) != expected:
        raise ValueError('Weight file SHA256 differs from architecture JSON')
    return path, config


def read_shard_index(path):
    """Validated contents of a sharded index: ``(weight_map, shard file names, metadata)``."""
    path = Path(path)
    index = json.loads(path.read_text())
    weight_map = index.get('weight_map') if isinstance(index, dict) else None
    metadata = index.get('metadata', {}) if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map or not isinstance(metadata, dict):
        raise ValueError(f'{path.name} is not a sharded safetensors index')
    files = []
    for name, file in weight_map.items():
        if not isinstance(name, str) or not isinstance(file, str):
            raise ValueError(f'{path.name}: weight_map must map tensor names to shard file names')
        if file not in files:
            if Path(file).name != file or file in ('.', '..') or not file.endswith('.safetensors'):
                raise ValueError(f'{path.name}: shard {file!r} must be a .safetensors file next to the index')
            files.append(file)
    digests = metadata.get('sha256', {})
    if not isinstance(digests, dict) or set(digests) - set(files):
        raise ValueError(f'{path.name}: metadata.sha256 must list shard files of the weight_map')
    return weight_map, files, metadata


def shard_paths(index):
    """Shard files of a sharded index, in weight_map order."""
    index = Path(index)
    return [index.parent / file for file in read_shard_index(index)[1]]


def _read_sharded(path):
    from safetensors.torch import load_file
    weight_map, files, metadata = read_shard_index(path)
    digests = metadata.get('sha256', {})
    state = {}
    for file in files:
        shard = path.parent / file
        if not shard.is_file():
            raise FileNotFoundError(f'Missing shard {shard}')
        if file in digests and sha256(shard) != digests[file]:
            raise ValueError(f'Shard {file} SHA256 differs from {path.name}')
        tensors = load_file(str(shard), device='cpu')
        if set(tensors) != {name for name, owner in weight_map.items() if owner == file}:
            raise ValueError(f'Shard {file} tensors differ from the weight_map of {path.name}')
        state.update(tensors)
    total = metadata.get('total_size')
    if total is not None and total != sum(value.numel() * value.element_size() for value in state.values()):
        raise ValueError(f'{path.name}: metadata.total_size differs from the shard tensors')
    return state


def read_weights(path):
    path = Path(path)
    if is_index(path):
        state = _read_sharded(path)
    elif path.suffix == '.safetensors':
        from safetensors.torch import load_file
        state = load_file(str(path), device='cpu')
    elif path.suffix == '.pt':
        import torch
        state = torch.load(path, map_location='cpu', weights_only=True)
    else:
        raise ValueError('Expected a .safetensors, sharded index or tensors-only .pt weight file')
    import torch
    if not isinstance(state, dict) or not state or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in state.items()):
        raise ValueError('Expected a nonempty tensors-only state dictionary')
    return state


def checkpoint_files(weights):
    """All files of a checkpoint entry: architecture JSON, entry file and shards."""
    path = Path(weights).expanduser().absolute()
    files = [config_path(path), path]
    if is_index(path):
        files += shard_paths(path)
    return files


def checkpoint_bytes(weights):
    return sum(file.stat().st_size for file in checkpoint_files(weights) if file.is_file())


def _model_spec(config):
    return {key: value for key, value in config.items() if key != 'weights_sha256'}


def load_model(weights, device='cpu'):
    from hyff.models.factory import build_model
    path, config = read_config(weights)
    model = build_model(_model_spec(config))
    state = read_weights(path)
    live = model.state_dict()
    if set(live) != set(state):
        raise ValueError('Weight tensor names differ from the architecture')
    for name, value in state.items():
        if value.shape != live[name].shape or value.dtype != live[name].dtype:
            raise ValueError(f'Tensor shape/dtype differs for {name}; weights are never cast')
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False).to(device), config


def read_adapter_export(weights):
    """Checkpoint as the dict read by the engine builders (architecture, factory, config, state).

    Built only from the weight file and its architecture JSON.
    """
    from hyff.models.factory import factory_id
    model, config = load_model(weights, 'cpu')
    factory = factory_id(config['model_id'])
    return dict(architecture=getattr(model, 'architecture', 'hyff'),
                model_factory=factory, model_config=config['model_config'],
                model_state=model.state_dict(),
                numerical_execution=getattr(model, 'numerical_execution', None),
                selection_kind='standalone_weights_only')


__all__ = ['INDEX_SUFFIX', 'checkpoint_bytes', 'checkpoint_files', 'config_path', 'entry_path', 'is_index',
           'load_model', 'read_config', 'read_shard_index', 'read_weights', 'sha256', 'shard_paths']
