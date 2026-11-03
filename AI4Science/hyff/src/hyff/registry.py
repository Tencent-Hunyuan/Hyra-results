"""Model names, default engine options and checkpoint discovery.

The package ships ``checkpoints/index.json`` (the files, sizes and sha256 of every model and the release
location of its weights) and the architecture JSON of every model. Weights are looked up in the cache
directory (``$HYFF_HOME/checkpoints`` or ``~/.cache/hyff/checkpoints``) and, in a source checkout, in
``<root>/checkpoints``; a model that is not installed is downloaded into the cache on first use. Every
model can also be loaded from an explicit ``weights=`` path.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from hyff.weights import config_path, entry_path, is_index, read_config, sha256

MODELS = ('presto', 'vivace', 'moderato', 'andante', 'largo')
# Earlier internal names, accepted silently.
MODEL_ALIASES = {'fast2': 'presto', 'medium': 'vivace', 'large': 'moderato', 'xl': 'andante', 'main24': 'largo'}

NATIVE_ENGINES = ('presto', 'vivace', 'moderato')
CAPTURE_ENGINES = ('andante', 'largo')
REFERENCE_ENGINES = ('reference', 'reference_cuda_graph')
ENGINES = NATIVE_ENGINES + CAPTURE_ENGINES + REFERENCE_ENGINES
PRECISIONS = ('fp32', 'tf32x3')
STATISTICS = ('fp64', 'fp32')
LONG_RANGE = ('native', 'dense', 'tree', 'auto', 'spme')
CAPACITY_POLICIES = ('native', 'partial', 'tiled', 'auto')

# Fastest validated settings per model, used whenever the model runs on its own engine.
DEFAULT_OPTIONS = {
    'presto': dict(engine='presto', precision='tf32x3', statistics='fp32'),
    'vivace': dict(engine='vivace', precision='tf32x3', statistics='fp32'),
    'moderato': dict(engine='moderato', precision='tf32x3', statistics='fp32'),
    'andante': dict(engine='andante', precision='fp32', statistics='fp64', edge_reserve=1.05,
                    capacity_policy='auto'),
    'largo': dict(engine='largo', precision='tf32x3', statistics='fp64', edge_reserve=1.05, long_range='auto',
                  capacity_policy='auto'),
}
MODEL_INFO = {
    'presto': dict(parameters='2.31 M', active_per_atom='0.05 M'),
    'vivace': dict(parameters='6.66 M', active_per_atom='0.16 M'),
    'moderato': dict(parameters='34.8 M', active_per_atom='0.83 M'),
    'andante': dict(parameters='175 M', active_per_atom='5.0 M'),
    'largo': dict(parameters='1.20 B', active_per_atom='34 M'),
}


def model_name(name):
    """Return the public model name for a name or a silent alias, else None."""
    if not isinstance(name, str):
        return None
    name = MODEL_ALIASES.get(name, name)
    return name if name in MODELS else None


def engine_name(name):
    if name is None:
        return None
    if name not in ENGINES:
        raise ValueError(f'Unknown engine {name!r}; choose from {", ".join(ENGINES)}')
    return name


PACKAGE_CHECKPOINTS = Path(__file__).resolve().parent / 'checkpoints'


def manifest():
    """The checkpoint index shipped with the package: release location and the files of every model."""
    data = json.loads((PACKAGE_CHECKPOINTS / 'index.json').read_text())
    if data.get('format') != 'hyff-checkpoints' or not isinstance(data.get('models'), dict):
        raise ValueError('The packaged checkpoints/index.json is not a HyFF checkpoint index')
    return data


def cache_dir():
    """Where downloaded weights go: ``$HYFF_HOME/checkpoints``, else ``~/.cache/hyff/checkpoints``."""
    home = os.environ.get('HYFF_HOME')
    return Path(home or '~/.cache/hyff').expanduser().absolute() / 'checkpoints'


def checkpoint_dirs():
    """Directories searched for installed weights, in order: the cache, then a source checkout."""
    dirs = [cache_dir()]
    package = Path(__file__).resolve().parent
    if package.parent.name == 'src':  # a source checkout: <root>/src/hyff next to <root>/checkpoints
        source = package.parents[1] / 'checkpoints'
        if source not in dirs:
            dirs.append(source)
    return dirs


def model_files(name):
    """Index row of a model: ``weights`` (entry file), ``sha256``, ``files`` and ``defaults``."""
    row = manifest()['models'].get(name)
    if not row:
        raise ValueError(f'{name} is not in the checkpoint index')
    for file in row['files']:
        path = Path(file['path'])
        if path.is_absolute() or '..' in path.parts or path.parts[0] != name:
            raise ValueError('Checkpoint index paths must stay inside the model directory')
    return row


def download_bytes(name):
    """Bytes downloaded on first use of a model (its release assets)."""
    return sum(file['size'] for file in model_files(name)['files'] if file.get('asset'))


def installed(name):
    """The first checkpoint directory that holds every file of a model, or None (no network access)."""
    files = model_files(name)['files']
    for directory in checkpoint_dirs():
        if all((directory / file['path']).is_file() for file in files):
            return directory
    return None


def verify(name, directory):
    """Check every installed file of a model against the index sha256."""
    for file in model_files(name)['files']:
        if sha256(directory / file['path']) != file['sha256']:
            raise ValueError(f'Checksum mismatch for the {name} file {directory / file["path"]}; run '
                             f'`hyff download {name}` to fetch a verified copy into {cache_dir()}')


def install(name):
    """Install a model into the cache directory (downloading its release assets); return the directory."""
    from hyff.download import install_files, locked
    data, row, directory = manifest(), model_files(name), cache_dir()
    with locked(directory / name):
        install_files(row['files'], data['release'], directory, PACKAGE_CHECKPOINTS)
    return directory


def bundled_weights(name, with_digest=False):
    """Return the installed entry file of a model (and its indexed sha256), or None when not installed."""
    directory = installed(name)
    row = model_files(name)
    path = None if directory is None else directory / row['weights']
    if with_digest:
        return path, row['sha256']
    return path


def identify(weights):
    """Infer the model name from a checkpoint's architecture JSON, or None."""
    from hyff.models.factory import CHARGE_FACTORIES, factory_id
    _, config = read_config(weights)
    factory = factory_id(config['model_id'])
    if factory in CHARGE_FACTORIES:
        return 'largo'
    for name, row in manifest()['models'].items():
        architecture = config_path(PACKAGE_CHECKPOINTS / row['weights'])
        if not architecture.is_file():
            continue
        bundled = json.loads(architecture.read_text())
        if (factory_id(bundled['model_id']) == factory and
                bundled['model_config'] == config['model_config']):
            return name
    return None


@dataclass
class Resolved:
    """A model selection: name (None for unrecognized weights), weight file, digest and engine options."""
    name: str | None
    weights: str
    sha256: str
    engine: str
    options: dict = field(default_factory=dict)


def _check_options(engine, options):
    precision, statistics = options.get('precision'), options.get('statistics')
    if precision not in PRECISIONS:
        raise ValueError('precision must be fp32 or tf32x3')
    if statistics not in STATISTICS:
        raise ValueError('statistics must be fp64 or fp32')
    if engine == 'andante' and precision != 'fp32':
        raise ValueError('The andante engine runs in fp32')
    if engine == 'largo' and precision != 'tf32x3':
        raise ValueError('The largo engine runs with tf32x3 products')
    if statistics == 'fp32' and engine not in NATIVE_ENGINES:
        raise ValueError('fp32 statistics are available for presto, vivace and moderato engines only')
    if statistics == 'fp32' and engine in ('vivace', 'moderato') and precision != 'tf32x3':
        raise ValueError('fp32 statistics require tf32x3 products')
    if engine in REFERENCE_ENGINES and precision != 'fp32':
        raise ValueError('The reference engines run in fp32')
    edge_reserve = options.get('edge_reserve')
    if edge_reserve is not None:
        if engine not in CAPTURE_ENGINES:
            raise ValueError('edge_reserve applies to the andante and largo engines only')
        if isinstance(edge_reserve, bool) or not isinstance(edge_reserve, (int, float)) or \
                not math.isfinite(edge_reserve) or edge_reserve < 1:
            raise ValueError('edge_reserve must be finite and at least 1')
    for key, allowed, engines in (('long_range', LONG_RANGE, ('largo',)),
                                  ('conservation', ('native', 'fp64'), ('largo',)),
                                  ('capacity_policy', CAPACITY_POLICIES, CAPTURE_ENGINES)):
        value = options.get(key)
        if value is None:
            continue
        if value not in allowed:
            raise ValueError(f'{key} must be one of {", ".join(allowed)}')
        if engine not in engines:
            raise ValueError(f'{key} applies to the {" and ".join(engines)} engine only')
    if options.get('capacity_policy') == 'partial' and engine != 'largo':
        raise ValueError('capacity_policy partial applies to the largo engine only')
    return options


def resolve(model=None, weights=None, *, engine=None, **overrides):
    """Resolve a model name and/or weight path to weights, engine and options.

    Without ``weights`` the model's installed checkpoint is used; when none is installed it is
    downloaded into the cache directory first. ``weights`` may be a ``.safetensors`` file, a sharded
    ``.safetensors.index.json``, a directory holding one of them, or a tensors-only ``.pt`` file.
    ``overrides`` are engine options (precision, statistics, edge_reserve, long_range, conservation,
    capacity_policy); None values keep the default.
    """
    name = model_name(model) if model is not None else None
    if model is not None and name is None:
        if weights is None and isinstance(model, (str, os.PathLike)) and (
                Path(model).suffix in ('.safetensors', '.pt') or is_index(model) or Path(model).is_dir()):
            weights, model = model, None
        else:
            raise ValueError(f'Unknown model {model!r}; choose from {", ".join(MODELS)} or pass weights=')
    if weights is not None:
        path = entry_path(weights)
        if not path.is_file():
            raise FileNotFoundError(path)
        identified = identify(path)
        if name is None:
            name = identified
        elif identified is not None and identified != name:
            raise ValueError(f'{path} holds {identified} weights, not {name}')
    else:
        if name is None:
            raise ValueError('Pass a model name or weights=')
        directory = installed(name) or install(name)
        verify(name, directory)
        path = directory / model_files(name)['weights']
    default = DEFAULT_OPTIONS.get(name, {}) if name else {}
    engine = engine_name(engine) or default.get('engine', 'reference')
    if engine in REFERENCE_ENGINES:
        options = dict(precision='fp32', statistics='fp64')
    else:
        # Accelerated engines are model specific; their defaults are the model's own.
        options = {key: value for key, value in DEFAULT_OPTIONS[engine].items() if key != 'engine'}
    options.update({key: value for key, value in overrides.items() if value is not None})
    _check_options(engine, options)
    return Resolved(name=name, weights=str(path), sha256=sha256(path), engine=engine, options=options)
