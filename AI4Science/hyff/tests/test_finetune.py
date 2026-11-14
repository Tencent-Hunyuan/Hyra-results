"""CPU checks of fine-tuning: data reading, element-energy refit, freezing, resume and exported weights.

A randomly initialized presto-architecture model stands in for the released weights (no download needed).
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from helpers.checkpoints import random_checkpoint, random_molecules
from hyff import finetuning as ft

try:
    import ase  # noqa: F401
    HAS_ASE = True
except ImportError:
    HAS_ASE = False


def springs(x, k=2.0, r0=1.3, cutoff=2.0):
    """A harmonic pair term between atoms closer than ``cutoff``: energy and forces."""
    d = x[None] - x[:, None]
    r = np.linalg.norm(d, axis=-1)
    i, j = np.triu_indices(len(x), 1)
    keep = r[i, j] < cutoff
    i, j = i[keep], j[keep]
    energy = 0.5 * k * np.sum((r[i, j] - r0) ** 2)
    g = (k * (r[i, j] - r0) / r[i, j])[:, None] * d[i, j]     # dE/dx_j for each pair
    forces = np.zeros_like(x)
    np.add.at(forces, j, -g)
    np.add.at(forces, i, g)
    return energy, forces


def labelled_npz(path, weights, molecules, shifts=None, periodic_box=False, seed=0, extra=False):
    """Labels from the model itself, plus optional per-element energy shifts, a systematic pair term and noise."""
    import hyff
    model = hyff.load(weights=weights, device='cpu')
    rng = np.random.default_rng(seed)
    positions, numbers, energies, forces, cells = [], [], [], [], []
    for x, z in molecules:
        cell = None
        if periodic_box:
            cell = np.eye(3) * 8.0
            x = x + 2.0
        e, f = model.energy_forces(x, z, cell)
        if extra:
            de, df = springs(x)
            e, f = e + de, f + df
        e += sum((shifts or {}).get(int(a), 0.0) for a in z) + rng.normal(scale=0.01)
        positions.append(x); numbers.append(z); energies.append(e)
        forces.append(f + rng.normal(scale=0.005, size=f.shape)); cells.append(cell if cell is not None else np.zeros((3, 3)))
    arrays = dict(positions=np.concatenate(positions), numbers=np.concatenate(numbers),
                  offsets=np.concatenate([[0], np.cumsum([len(z) for z in numbers])]),
                  energies=np.array(energies), forces=np.concatenate(forces))
    if periodic_box:
        arrays['cells'] = np.array(cells)
    np.savez(path, **arrays)
    return path


class FineTuneTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.weights = random_checkpoint(self.root / 'start')
        self.molecules = random_molecules(16)

    def tearDown(self):
        self.tmp.cleanup()

    def options(self, output, **extra):
        options = dict(weights=self.weights, output=self.root / output, device='cpu', batch_atoms=48, epochs=3,
                       valid_fraction=0.25, verbose=False)
        options.update(extra)
        return options

    def test_element_energy_refit_absorbs_a_new_energy_reference(self):
        shifts = {1: -13.6, 6: -1029.5, 7: -1484.2, 8: -2041.9}
        data = labelled_npz(self.root / 'shifted.npz', self.weights, self.molecules, shifts=shifts)
        run = ft.FineTune(ft.FineTuneConfig(train=str(data), weights=str(self.weights), output=str(self.root / 'a'),
                                            device='cpu', valid_fraction=0.0), verbose=False)
        run.refit_element_energies()
        for z, value in shifts.items():
            self.assertAlmostEqual(run.element_shifts[z], value, delta=0.02)
        evaluation = run.evaluate(run.data)
        self.assertLess(evaluation['energy_mae_meV_atom'], 5.0)

    def test_training_lowers_the_validation_loss_and_exports_loadable_weights(self):
        import hyff
        data = labelled_npz(self.root / 'train.npz', self.weights, random_molecules(48, seed=3), extra=True)
        summary = ft.finetune(train=data, **self.options('run', epochs=4, lr=1e-3))
        start, best = summary['history'][0]['valid'], summary['history'][summary['best_epoch']]['valid']
        self.assertLess(best['loss'], start['loss'])
        for name in ('best', 'final'):
            exported = Path(summary[name])
            config = json.loads((exported / 'weights.json').read_text())
            self.assertEqual(set(config), {'format', 'model_id', 'model_config', 'weights_sha256'})
            from hyff.registry import identify
            self.assertEqual(identify(exported / 'weights.safetensors'), 'presto')
        model = hyff.load(weights=summary['final'], device='cpu')
        x, z = self.molecules[0]
        e, f = model.energy_forces(x, z)
        self.assertTrue(np.isfinite(e) and np.isfinite(f).all())
        self.assertTrue((Path(summary['final']).parent / 'finetune.json').is_file())

    def test_freezing_keeps_parameters_unchanged(self):
        data = labelled_npz(self.root / 'train.npz', self.weights, self.molecules)
        from safetensors.torch import load_file
        before = load_file(str(self.weights))
        summary = ft.finetune(train=data, **self.options('frozen', train_only=['readouts.*'],
                                                          refit_element_energies=False, lr=1e-3))
        after = load_file(str(Path(summary['final']) / 'weights.safetensors'))
        changed = sorted(name for name in before if not torch.equal(before[name], after[name]))
        self.assertTrue(changed)
        self.assertTrue(all(name.startswith('readouts.') for name in changed), changed)
        with self.assertRaisesRegex(ValueError, 'matches no parameter'):
            ft.finetune(train=data, **self.options('bad', freeze=['no_such_module.*']))

    def test_resume_continues_exactly(self):
        data = labelled_npz(self.root / 'train.npz', self.weights, self.molecules)
        straight = ft.finetune(train=data, **self.options('straight'))

        class Interrupted(ft.FineTune):
            def save_checkpoint(self):
                super().save_checkpoint()
                if self.epoch == 1:
                    raise KeyboardInterrupt

        config = ft.FineTuneConfig(train=str(data), weights=str(self.weights), output=str(self.root / 'resumed'),
                                   device='cpu', batch_atoms=48, epochs=3, valid_fraction=0.25)
        with self.assertRaises(KeyboardInterrupt):
            Interrupted(config, verbose=False).run()
        with self.assertRaisesRegex(FileExistsError, 'resume'):
            ft.FineTune(config, verbose=False)
        config.resume = True
        resumed = ft.FineTune(config, verbose=False).run()
        from safetensors.torch import load_file
        a = load_file(str(Path(straight['final']) / 'weights.safetensors'))
        b = load_file(str(Path(resumed['final']) / 'weights.safetensors'))
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
        self.assertEqual([h['epoch'] for h in resumed['history']], [0, 1, 2, 3])

    def test_moving_average_of_the_weights(self):
        data = labelled_npz(self.root / 'train.npz', self.weights, self.molecules, extra=True)
        plain = ft.finetune(train=data, **self.options('plain', epochs=2, lr=1e-3))
        averaged = ft.finetune(train=data, **self.options('ema', epochs=2, lr=1e-3, ema_decay=0.9))
        from safetensors.torch import load_file
        a = load_file(str(Path(plain['final']) / 'weights.safetensors'))
        b = load_file(str(Path(averaged['final']) / 'weights.safetensors'))
        self.assertTrue(any(not torch.equal(a[k], b[k]) for k in a))   # the exported weights are the average
        self.assertTrue(np.isfinite(averaged['history'][-1]['valid']['loss']))

    def test_periodic_structures_train(self):
        data = labelled_npz(self.root / 'box.npz', self.weights, self.molecules[:6], periodic_box=True)
        summary = ft.finetune(train=data, **self.options('box', epochs=1))
        self.assertTrue(np.isfinite(summary['history'][-1]['train_loss']))

    @unittest.skipUnless(HAS_ASE, 'ase is not installed')
    def test_extxyz_reading(self):
        from ase import Atoms
        from ase.calculators.singlepoint import SinglePointCalculator
        from ase.io import write
        frames = []
        for k, (x, z) in enumerate(self.molecules[:4]):
            atoms = Atoms(numbers=z, positions=x)
            atoms.info.update(charge=k % 2, spin=1 + k % 2)
            atoms.calc = SinglePointCalculator(atoms, energy=-float(k), forces=np.full((len(z), 3), 0.1 * k))
            frames.append(atoms)
        write(self.root / 'frames.extxyz', frames)
        data = ft.read_structures(self.root / 'frames.extxyz')
        self.assertEqual(len(data), 4)
        np.testing.assert_allclose(data.energies, [0., -1., -2., -3.])
        np.testing.assert_array_equal(data.charges, [0, 1, 0, 1])
        np.testing.assert_array_equal(data.multiplicities, [1, 2, 1, 2])
        self.assertFalse(data.periodic.any())
        frames[0].pbc = [True, True, False]
        frames[0].cell = np.eye(3) * 10
        write(self.root / 'mixed.extxyz', frames)
        with self.assertRaisesRegex(ValueError, 'pbc must be all True'):
            ft.read_structures(self.root / 'mixed.extxyz')

    def test_batches_are_shared_evenly_between_gpus(self):
        groups = [[k] for k in range(10)]
        for world in (1, 2, 3, 4, 8, 16):
            shards = [ft.shard(groups, rank, world) for rank in range(world)]
            self.assertEqual(len({len(x) for x in shards}), 1)                       # equal step counts
            seen = [g[0] for x in shards for g in x]
            self.assertEqual(set(seen), set(range(10)))                               # every batch is used
            self.assertEqual(len(seen), -(-10 // world) * world)                      # padding only to fill
        self.assertIs(type(ft.shard(groups, 0, 1)), list)

    def test_configuration_is_checked(self):
        for options, message in ((dict(epochs=0), 'epochs'), (dict(lr=-1.), 'lr'), (dict(valid_fraction=1.), 'valid_fraction'),
                                 (dict(ema_decay=1.5), 'ema_decay'), (dict(batch_atoms=0), 'batch_atoms'),
                                 (dict(gpus=0), 'gpus'), (dict(gpus=2, device='cpu'), 'CUDA')):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                ft.FineTuneConfig(train='x.npz', **options)


if __name__ == '__main__':
    unittest.main()
