"""GPU checks of the ASE calculator and of fine-tuned weights on the fast engine (skipped without CUDA,
ASE or installed presto weights). Each check runs in its own process (a process may construct at most two
compiled engines). Tolerances: forces 1e-4 eV/A, energies 1e-3 eV for a 918-atom box.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('ase')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')

from hyff.registry import installed  # noqa: E402

if installed('presto') is None:
    pytestmark = [pytestmark, pytest.mark.skip(reason='presto weights are not installed (hyff download presto)')]

WATER_BOX = r'''
import numpy as np
from ase import Atoms
rng = np.random.default_rng(0)
grid = np.stack(np.meshgrid(*[np.arange(6)] * 3, indexing='ij'), -1).reshape(-1, 3) * 3.1 + 0.5
water = np.array([[0., 0., 0.], [0.9572, 0., 0.], [-0.24, 0.927, 0.]])
positions = (grid[:, None] + water[None]).reshape(-1, 3) + rng.normal(scale=0.03, size=(len(grid) * 3, 3))
box = Atoms(numbers=[8, 1, 1] * len(grid), positions=positions, cell=np.eye(3) * 18.6, pbc=True)
'''

CALCULATOR = WATER_BOX + r'''
import json, sys
from hyff.calculator import HyFFCalculator
eager, fast = HyFFCalculator('presto', engine='eager'), HyFFCalculator('presto', engine='fast')
def evaluate(atoms, calc):
    a = atoms.copy(); a.calc = calc
    return a.get_potential_energy(), a.get_forces()
out = {}
for label, atoms in (('box', box),):
    (e1, f1), (e2, f2) = evaluate(atoms, eager), evaluate(atoms, fast)
    out[label] = [abs(e1 - e2), float(np.abs(f1 - f2).max())]
scaled = box.copy(); scaled.set_cell(box.cell.array * 1.02, scale_atoms=True)
(e1, f1), (e2, f2) = evaluate(scaled, eager), evaluate(scaled, fast)
out['scaled_cell'] = [abs(e1 - e2), float(np.abs(f1 - f2).max())]
print(json.dumps(out))
'''

FINETUNE = WATER_BOX + r'''
import json, sys
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
import hyff
from hyff.calculator import HyFFCalculator
root = sys.argv[1]
frames, calc = [], HyFFCalculator('presto', engine='eager')
for k in range(12):
    atoms = box.copy()
    atoms.positions += rng.normal(scale=0.05, size=atoms.positions.shape)
    atoms.calc = calc
    e, f = atoms.get_potential_energy() + 0.01 * len(atoms), atoms.get_forces()
    atoms.calc = SinglePointCalculator(atoms, energy=e, forces=f)
    frames.append(atoms)
write(root + '/train.extxyz', frames)
summary = hyff.finetune('presto', train=root + '/train.extxyz', output=root + '/ft', epochs=1, verbose=False)
weights = summary['final']
e_ref, f_ref = hyff.load(weights=weights).energy_forces(box.positions, box.numbers, box.cell.array)
with hyff.create_engine('presto', box.positions, box.numbers, box.cell.array, weights=weights) as engine:
    e_fast, f_fast = engine.energy_forces(box.positions)
print(json.dumps(dict(parity=[abs(e_ref - e_fast), float(np.abs(f_ref - f_fast).max())],
                      shifts=summary['element_shifts_eV'])))
'''


def run(script, *args):
    import hyff
    # The child imports the same hyff as this process (a source checkout or the installed package).
    package_root = str(Path(hyff.__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [package_root, os.environ.get('PYTHONPATH')])))
    result = subprocess.run([sys.executable, '-c', script, *args], capture_output=True, text=True, timeout=1800,
                            env=env)
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_fast_calculator_matches_eager_including_a_cell_change():
    out = run(CALCULATOR)
    for label, (de, df) in out.items():
        assert de < 1e-3 and df < 1e-4, (label, de, df)


def test_finetuned_weights_run_on_the_fast_engine():
    with tempfile.TemporaryDirectory() as root:
        out = run(FINETUNE, root)
    de, df = out['parity']
    assert de < 1e-3 and df < 1e-4, out
    # 0.01 eV per atom was added to every label: the refit takes it up as an even per-atom shift.
    assert abs(out['shifts']['1'] * 2 + out['shifts']['8'] - 0.03) < 1e-3, out['shifts']
