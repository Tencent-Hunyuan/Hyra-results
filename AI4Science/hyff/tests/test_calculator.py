"""CPU checks of the ASE calculator (eager engine) with a randomly initialized presto-architecture model."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from helpers.checkpoints import random_checkpoint, random_molecules

try:
    from ase import Atoms
    from ase.calculators.calculator import PropertyNotImplementedError
    from hyff.calculator import HyFFCalculator
    HAS_ASE = True
except ImportError:
    HAS_ASE = False


@unittest.skipUnless(HAS_ASE, 'ase is not installed')
class CalculatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.weights = random_checkpoint(Path(cls.tmp.name) / 'start')
        cls.calc = HyFFCalculator(weights=cls.weights, engine='eager', device='cpu')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def atoms(self, k=0, **info):
        x, z = random_molecules(4, seed=5)[k]
        atoms = Atoms(numbers=z, positions=x)
        atoms.info.update(info)
        atoms.calc = self.calc
        return atoms

    def test_matches_the_model_and_forces_are_the_energy_gradient(self):
        import hyff
        atoms = self.atoms()
        energy, forces = atoms.get_potential_energy(), atoms.get_forces()
        ref_energy, ref_forces = hyff.load(weights=self.weights, device='cpu').energy_forces(atoms.positions, atoms.numbers)
        self.assertAlmostEqual(energy, ref_energy, places=10)
        np.testing.assert_array_equal(forces, ref_forces)
        self.assertEqual(atoms.get_potential_energy(force_consistent=True), energy)
        h = 1e-3
        for atom, axis in ((0, 0), (1, 2)):
            shifted = []
            for sign in (1, -1):
                moved = atoms.copy()
                moved.calc = self.calc
                moved.positions[atom, axis] += sign * h
                shifted.append(moved.get_potential_energy())
            self.assertAlmostEqual(-(shifted[0] - shifted[1]) / (2 * h), forces[atom, axis], delta=2e-3)

    def test_charge_and_multiplicity_from_info(self):
        neutral = self.atoms(charge=0, multiplicity=1).get_potential_energy()
        atoms = self.atoms(charge=0, multiplicity=1)
        atoms.get_potential_energy()
        atoms.info['spin'] = 3          # OMol25 convention: the multiplicity
        del atoms.info['multiplicity']
        triplet = atoms.get_potential_energy()   # recomputed although only info changed
        self.assertNotEqual(triplet, neutral)
        self.assertEqual(self.atoms(spin=3).get_potential_energy(), triplet)
        default = HyFFCalculator(weights=self.weights, engine='eager', device='cpu', charge=1, multiplicity=2)
        atoms = self.atoms()
        atoms.calc = default
        self.assertEqual(atoms.get_potential_energy(), self.atoms(charge=1, multiplicity=2).get_potential_energy())
        with self.assertRaisesRegex(ValueError, 'multiplicity'):
            self.atoms(spin=2, multiplicity=3).get_potential_energy()
        with self.assertRaisesRegex(ValueError, 'multiplicity'):
            self.atoms(multiplicity=1.5).get_potential_energy()

    def test_periodicity(self):
        atoms = self.atoms()
        atoms.cell = np.eye(3) * 3.5
        atoms.pbc = True
        periodic = atoms.get_potential_energy()
        atoms.pbc = False
        self.assertNotEqual(atoms.get_potential_energy(), periodic)
        atoms.pbc = [True, True, False]
        with self.assertRaisesRegex(ValueError, 'fully periodic'):
            atoms.get_potential_energy()

    def test_stress_and_options(self):
        with self.assertRaises(PropertyNotImplementedError):
            self.atoms().get_stress()
        with self.assertRaisesRegex(ValueError, 'CUDA'):
            HyFFCalculator(weights=self.weights, engine='fast', device='cpu')
        with self.assertRaisesRegex(ValueError, "engine='fast'"):
            HyFFCalculator(weights=self.weights, engine='eager', device='cpu', precision='fp32')
        with self.assertRaisesRegex(ValueError, 'engine must be'):
            HyFFCalculator(weights=self.weights, engine='compiled', device='cpu')


if __name__ == '__main__':
    unittest.main()
