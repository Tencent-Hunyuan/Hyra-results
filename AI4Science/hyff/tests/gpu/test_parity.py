"""GPU parity against reference outputs of the previous release (skipped without CUDA or reference files).

Set HYFF_GOLDEN to a directory with golden_<name>.npz files (keys e0, f0, e1, f1, md_positions)
produced on the 2,586-atom benchmark system HYFF_PARITY_SYSTEM. The largo golden file was produced with
an earlier largo checkpoint and the isolated (dense) long-range sum; pass that checkpoint as
HYFF_LARGO_WEIGHTS (the largo golden tests are skipped otherwise). Tolerances: |dE| <= 1e-4 eV,
|dF| <= 1e-4 eV/A, MD |dx| <= 1e-5 A; largo with its periodic default (SPME, a slightly different
energy than the isolated golden) is checked at |dE|, |dF| <= 1e-4 and MD |dx| <= 1e-4 A. Each model runs
in its own process (a process may construct at most two compiled engines).

HYFF_LARGO_FRAMES selects an NPZ of OMol25 validation frames with the per-frame errors of the
full-validation evaluation of the released largo weights (keys positions, numbers, offsets, natoms, charge,
multiplicity, raw_energy, reference_energy, target_forces, pipeline_signed_energy_meV_atom,
pipeline_force_mae_meV_A); the released model must reproduce them within 0.01 meV/atom and 0.05 meV/A.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')

from hyff.registry import MODEL_ALIASES, MODELS  # noqa: E402

GOLDEN = Path(os.environ.get('HYFF_GOLDEN', ''))
SYSTEM = os.environ.get('HYFF_PARITY_SYSTEM', '')
# Golden files may carry the earlier model names.
OLD_NAMES = {new: old for old, new in MODEL_ALIASES.items()}

CHECK = r'''
import json, sys
import numpy as np
import hyff
name, system_path, golden_path, weights, out, long_range = sys.argv[1:7]
weights = weights or None
lr = dict(long_range=long_range) if long_range else {}
gold = np.load(golden_path)
system = np.load(system_path)
positions = system['positions_A'].astype(np.float64)
result = {}
with hyff.create_engine(name, positions, system['atomic_numbers'], system['box_A'], weights=weights, **lr) as engine:
    for index, shift in enumerate((0.0, 0.01)):
        rng = np.random.default_rng(7)
        energy, forces = engine.energy_forces(positions + shift * rng.standard_normal(positions.shape))
        result[f'dE{index}'] = abs(energy - float(gold[f'e{index}']))
        result[f'dF{index}'] = float(np.abs(forces.astype(np.float64) - gold[f'f{index}']).max())
sim = hyff.Simulation(system_path, name, weights=weights, steps=20, chunk=10, timestep_fs=1.0, tau_ps=0.0,
                      recovery='off', save_every=10 ** 9, output=out, **lr)
with sim:
    status = sim.run()['status']
state = np.load(out + '/final_state.npz')
result.update(status=status, dx=float(np.abs(state['positions_A'] - gold['md_positions']).max()))
print('RESULT ' + json.dumps(result))
'''


def _golden(name):
    for stem in (name, OLD_NAMES[name]):
        path = GOLDEN / f'golden_{stem}.npz'
        if GOLDEN.name and path.is_file():
            return path
    pytest.skip(f'no golden file for {name}')


CASES = [(name, '', 1e-5) for name in MODELS if name != 'largo'] + [('largo', 'dense', 1e-5), ('largo', '', 1e-4)]


@pytest.mark.parametrize('name,long_range,md_tolerance', CASES,
                         ids=[name + ('-' + lr if lr else '') for name, lr, _ in CASES])
def test_engine_and_md_parity(name, long_range, md_tolerance, tmp_path):
    if not SYSTEM or not Path(SYSTEM).is_file():
        pytest.skip('HYFF_PARITY_SYSTEM is not set')
    golden = _golden(name)
    weights = os.environ.get('HYFF_LARGO_WEIGHTS', '') if name == 'largo' else ''
    if name == 'largo' and not weights:
        pytest.skip('HYFF_LARGO_WEIGHTS is not set')
    completed = subprocess.run([sys.executable, '-c', CHECK, name, SYSTEM, str(golden), weights,
                                str(tmp_path / 'md'), long_range], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr[-4000:]
    line = next(l for l in completed.stdout.splitlines() if l.startswith('RESULT '))
    result = json.loads(line[len('RESULT '):])
    print(name, long_range or 'default', json.dumps(result))
    assert result['status'] == 'complete'
    assert max(result['dE0'], result['dE1']) <= 1e-4
    assert max(result['dF0'], result['dF1']) <= 1e-4
    assert result['dx'] <= md_tolerance


FULLVAL = r'''
import json, sys
import numpy as np
import hyff
z = dict(np.load(sys.argv[1]))
count = int(sys.argv[2])
off = z['offsets']
result = hyff.predict_batch('largo', z['positions'][:off[count]], z['numbers'][:off[count]], off[:count + 1],
                            z['charge'][:count].astype(np.float32), z['multiplicity'][:count])
n = z['natoms'][:count]
signed = 1000.0 * (result['energies'] - (z['raw_energy'][:count] - z['reference_energy'][:count])) / n
diff = np.abs(result['forces'].astype(np.float64) - z['target_forces'][:off[count]]) * 1000.0
mae = np.array([diff[off[i]:off[i + 1]].mean() for i in range(count)])
print('RESULT ' + json.dumps(dict(
    energy=float(np.abs(signed - z['pipeline_signed_energy_meV_atom'][:count]).max()),
    force=float(np.abs(mae - z['pipeline_force_mae_meV_A'][:count]).max()))))
'''


def test_largo_full_validation_frames():
    frames = os.environ.get('HYFF_LARGO_FRAMES', '')
    if not frames or not Path(frames).is_file():
        pytest.skip('HYFF_LARGO_FRAMES is not set')
    completed = subprocess.run([sys.executable, '-c', FULLVAL, frames, '256'], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr[-4000:]
    line = next(l for l in completed.stdout.splitlines() if l.startswith('RESULT '))
    result = json.loads(line[len('RESULT '):])
    assert result['energy'] <= 0.01 and result['force'] <= 0.05, result
