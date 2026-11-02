"""Public API checks on CPU: tiny-water inference for each bundled model, names, defaults and the CLI."""
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import hyff
from hyff import api, registry

ROOT = Path(__file__).resolve().parents[1]
BUNDLED = ('presto', 'vivace', 'moderato', 'andante')
WATER = np.array([[0., 0., 0.], [.9572, 0., 0.], [-.2399872, .927297, 0.]])
# Eager CPU energies (eV) and the force on O (eV/A) of the water above, from the previous release.
WATER_REFERENCE = {
    'presto': (-0.7895724028348923, (-0.14660155773162842, -0.14384034276008606)),
    'vivace': (-0.7874108105897903, (-0.1478596329689026, -0.1455327570438385)),
    'moderato': (-0.7849707305431366, (-0.14591895043849945, -0.14302551746368408)),
    'andante': (-0.7848068177700043, (-0.1456299126148224, -0.14264577627182007)),
}


def _installed(name):
    return registry.bundled_weights(name) is not None


@pytest.mark.parametrize('name', BUNDLED)
def test_cpu_water_inference(name):
    if not _installed(name):
        pytest.skip(f'{name} weights are not installed (set HYFF_HOME)')
    model = hyff.load(name, device='cpu')
    energy, forces = model.energy_forces(WATER, [8, 1, 1])
    assert np.isfinite(energy)
    assert forces.shape == (3, 3) and np.isfinite(forces).all()
    reference_energy, reference_force = WATER_REFERENCE[name]
    assert abs(energy - reference_energy) <= 1e-6
    assert np.abs(forces[0, :2] - reference_force).max() <= 1e-6
    # Translation invariance: forces sum to zero for an isolated molecule.
    assert np.abs(forces.sum(axis=0)).max() < 1e-4
    shifted, _ = model.energy_forces(WATER + 3.0, [8, 1, 1])
    assert abs(shifted - energy) < 1e-4


def test_names_and_silent_aliases():
    assert hyff.MODELS == ('presto', 'vivace', 'moderato', 'andante', 'largo')
    assert sorted(registry.MODEL_ALIASES.values()) == sorted(hyff.MODELS)
    for old, new in registry.MODEL_ALIASES.items():
        assert registry.model_name(old) == new
    assert registry.model_name('largo') == 'largo'
    assert registry.model_name('allegro') is None
    with pytest.raises(ValueError):
        registry.engine_name('fastest')


@pytest.mark.parametrize('name', BUNDLED)
def test_default_options(name):
    if not _installed(name):
        pytest.skip(f'{name} weights are not installed')
    selection = registry.resolve(name)
    assert selection.name == name and selection.engine == name
    assert selection.options == {k: v for k, v in registry.DEFAULT_OPTIONS[name].items() if k != 'engine'}
    # An alias resolves to the same selection.
    alias = {v: k for k, v in registry.MODEL_ALIASES.items()}[name]
    assert registry.resolve(alias).options == selection.options


def test_largo_defaults():
    defaults = registry.DEFAULT_OPTIONS['largo']
    assert defaults == dict(engine='largo', precision='tf32x3', statistics='fp64',
                            edge_reserve=1.05, long_range='auto', capacity_policy='auto')
    assert registry.DEFAULT_OPTIONS['andante']['edge_reserve'] == 1.05
    assert registry.DEFAULT_OPTIONS['andante']['capacity_policy'] == 'auto'
    # auto keeps the default edge-frame-mixing route for small systems.
    from hyff.engines.capture.capacity import resolve_capacity_policy
    assert resolve_capacity_policy('largo', 'auto', 2586)['selected'] == 'native'
    assert resolve_capacity_policy('largo', 'auto', 103440)['selected'] == 'tiled'
    assert resolve_capacity_policy('andante', 'auto', 2586)['selected'] == 'native'
    # largo is a regular model: sharded weights listed in the packaged index, resolved by name.
    row = registry.model_files('largo')
    assert row['weights'] == 'largo/weights.safetensors.index.json'
    if _installed('largo'):
        selection = registry.resolve('largo')
        assert (selection.name, selection.engine) == ('largo', 'largo')
        assert selection.weights.endswith('weights.safetensors.index.json') and selection.sha256 == row['sha256']


def test_explicit_weights_are_identified():
    path = registry.bundled_weights('presto')
    if path is None:
        pytest.skip('presto weights are not installed')
    # Without a model name the model comes from the weights (the default is presto only without weights).
    with mock.patch.object(api, 'Model', side_effect=lambda selection, **_: selection):
        assert api.load(weights=path).name == 'presto'
        assert api.load().name == 'presto'
    selection = registry.resolve(None, path)
    assert selection.name == 'presto' and selection.engine == 'presto'
    selection = registry.resolve(str(path))
    assert selection.name == 'presto'
    if _installed('vivace'):
        with pytest.raises(ValueError, match='presto'):
            registry.resolve('vivace', path)


def test_option_validation():
    if not _installed('andante'):
        pytest.skip('andante weights are not installed')
    with pytest.raises(ValueError):
        registry.resolve('andante', precision='tf32x3')
    with pytest.raises(ValueError):
        registry.resolve('andante', long_range='dense')
    assert registry.resolve('andante', engine='reference').options == dict(precision='fp32', statistics='fp64')


def test_simulation_config_defaults():
    from hyff.md.simulation import RunConfig
    cfg = RunConfig(system='unused.npz', model='presto')
    assert (cfg.steps, cfg.timestep_fs, cfg.tau_ps, cfg.gpus, cfg.route) == (10000, 1.0, 0.5, 1, 'native')
    cfg = RunConfig(system='unused.npz', model='largo', gpus=4)
    assert (cfg.route, cfg.ranks) == ('auto', 4)
    with pytest.raises(ValueError):
        RunConfig(system='unused.npz', gpus=3)


def _cli(*args, cwd=None):
    env = dict(os.environ)
    if env.get('HYFF_TEST_INSTALLED') != '1':
        env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'src'), env.get('PYTHONPATH', '')])
    return subprocess.run([sys.executable, '-m', 'hyff', *args], capture_output=True, text=True,
                          env=env, cwd=cwd)


def test_cli_help():
    for command in ([], ['infer'], ['md'], ['predict'], ['models'], ['download']):
        completed = _cli(*command, '--help')
        assert completed.returncode == 0, completed.stderr
        assert 'usage: hyff' in completed.stdout
    assert 'presto' in _cli('md', '--help').stdout


def test_cli_models(tmp_path):
    completed = _cli('models')
    assert completed.returncode == 0, completed.stderr
    for name in hyff.MODELS:
        assert name in completed.stdout
    assert 'cache directory' in completed.stdout
    # With an empty cache and no source checkout every model is downloaded on first use (no network access here).
    env_home = dict(os.environ, HYFF_HOME=str(tmp_path / 'home'), HYFF_MIRROR=str(tmp_path / 'no-mirror'))
    completed = subprocess.run([sys.executable, '-c', 'import sys; from hyff import cli, registry; '
                                'registry.checkpoint_dirs = lambda: [registry.cache_dir()]; sys.exit(cli.main(["models"]))'],
                               capture_output=True, text=True, cwd=tmp_path,
                               env=dict(env_home, PYTHONPATH=os.pathsep.join([str(ROOT / 'src'), os.environ.get('PYTHONPATH', '')]))
                               if os.environ.get('HYFF_TEST_INSTALLED') != '1' else env_home)
    assert completed.returncode == 0, completed.stderr
    lines = {line.split()[0]: line for line in completed.stdout.splitlines()[1:6]}
    assert set(lines) == set(hyff.MODELS)
    assert all('downloaded on first use' in line for line in lines.values())
    assert 'GB' in lines['largo'] and 'MB' in lines['presto']
    assert not (tmp_path / 'home').exists()


def test_cli_infer_cpu(tmp_path):
    if not _installed('presto'):
        pytest.skip('presto weights are not installed')
    np.savez(tmp_path / 'water.npz', positions_A=WATER, atomic_numbers=np.array([8, 1, 1]))
    alias = {v: k for k, v in registry.MODEL_ALIASES.items()}['presto']
    completed = _cli('infer', '--model', alias, '--device', 'cpu', '--system', str(tmp_path / 'water.npz'),
                     '--output', str(tmp_path / 'out.npz'))
    assert completed.returncode == 0, completed.stderr
    out = np.load(tmp_path / 'out.npz')
    energy, _ = hyff.load('presto', device='cpu').energy_forces(WATER, [8, 1, 1])
    assert abs(float(out['energy_eV']) - energy) < 1e-6
