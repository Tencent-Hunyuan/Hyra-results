"""CPU checks of the Monte Carlo barostat: ensemble, group scaling, trial moves, SPME cell changes, options."""
import math
import unittest

import numpy as np
import torch

from hyff.io import validate_restart_state
from hyff.kernels.long_range import spme
from hyff.md import simulation as md
from hyff.md.barostat import BAR_A3_EV, MonteCarloBarostat, engine_cell, make_whole
from hyff.md.units import KB_EV_K

WATER = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.24, 0.927, 0.0]])


def barostat(mode='isotropic', n_groups=100, cell=None, pressure=400.0, seed=3):
    cell = np.eye(3) * 21.5 if cell is None else cell
    return MonteCarloBarostat(pressure, 300.0, mode, np.arange(n_groups), np.ones(n_groups), engine_cell(cell),
                              np.random.default_rng(seed), 'cpu')


class IdealGasEnsembleTests(unittest.TestCase):
    def test_mean_volume_of_rigid_groups_without_interactions(self):
        # p(V) ~ V^N exp(-PV/kT) for N scaled groups and dE = 0, so <V> = (N + 1) kT / P.
        n, pressure = 100, 400.0
        expected = (n + 1) * KB_EV_K * 300.0 / (pressure * BAR_A3_EV)
        for mode in ('isotropic', 'semi-isotropic', 'anisotropic'):
            with self.subTest(mode=mode):
                b = barostat(mode, n, np.diag([20., 21.5, 23.]), pressure)
                cell, volumes = engine_cell(np.diag([20., 21.5, 23.])), []
                for step in range(30000):
                    k, new, _ = b.propose(cell)
                    if b.decide(k, 0.0, abs(np.linalg.det(cell)), abs(np.linalg.det(new))):
                        cell = new
                    if step >= 3000:
                        volumes.append(abs(np.linalg.det(cell)))
                self.assertAlmostEqual(np.mean(volumes) / expected, 1.0, delta=0.02)
                self.assertTrue(0.25 < b.acceptances / b.attempts < 0.75)

    def test_acceptance_work(self):
        b = barostat(n_groups=10)
        kT = KB_EV_K * 300.0
        w = b.work(0.05, 1000.0, 1010.0)
        self.assertAlmostEqual(w, 0.05 + 400.0 * BAR_A3_EV * 10.0 - 10 * kT * math.log(1.01), places=12)


class GroupScalingTests(unittest.TestCase):
    def test_groups_stay_rigid_and_centres_follow_the_cell(self):
        rng = np.random.default_rng(0)
        x = np.concatenate([WATER + rng.uniform(0, 18, 3) for _ in range(5)])
        ids = np.repeat(np.arange(5), 3)
        masses = np.tile([15.999, 1.008, 1.008], 5)
        b = MonteCarloBarostat(1.0, 300.0, 'isotropic', ids, masses, engine_cell(np.eye(3) * 20.), rng, 'cpu')
        M = np.diag([1.01, 1.01, 1.01])
        y = b.scaled_positions(torch.tensor(x), M).numpy()
        for g in range(5):
            a, c = x[3 * g:3 * g + 3], y[3 * g:3 * g + 3]
            np.testing.assert_allclose(c - c[0], a - a[0], atol=1e-12)
            centre_a = (masses[3 * g:3 * g + 3, None] * a).sum(0) / masses[3 * g:3 * g + 3].sum()
            centre_c = (masses[3 * g:3 * g + 3, None] * c).sum(0) / masses[3 * g:3 * g + 3].sum()
            np.testing.assert_allclose(centre_c, centre_a @ M, atol=1e-12)

    def test_split_molecules_are_made_whole(self):
        cell = np.eye(3) * 10.
        x = WATER + np.array([9.7, 5., 5.])
        x[1:] -= np.array([10., 0., 0.]) * (x[1:, 0] > 10.)[:, None]     # H atoms wrapped to the other side
        whole, moved = make_whole(x, cell, 3, [np.array([[0, 1], [0, 2]])])
        self.assertEqual(moved, 1)
        np.testing.assert_allclose(whole - whole[0], WATER, atol=1e-12)
        again, moved = make_whole(whole, cell, 3, [np.array([[0, 1], [0, 2]])])
        self.assertEqual(moved, 0)


class TrialMoveTests(unittest.TestCase):
    def test_coupled_axes(self):
        cell = engine_cell(np.diag([20., 21., 22.]))
        for mode, sets in (('isotropic', [{0, 1, 2}]), ('semi-isotropic', [{0, 1}, {2}]),
                           ('anisotropic', [{0}, {1}, {2}])):
            b = barostat(mode, cell=cell)
            seen = set()
            for _ in range(60):
                k, new, M = b.propose(cell)
                np.testing.assert_allclose(cell @ M, new, rtol=1e-12)
                self.assertTrue(np.array_equal(new, new.astype(np.float32)))
                changed = {i for i in range(3) if abs(new[i, i] / cell[i, i] - 1) > 1e-7}
                self.assertIn(changed, sets)
                ratios = [new[i, i] / cell[i, i] for i in changed]
                self.assertLess(max(ratios) - min(ratios), 1e-6)
                seen.add(frozenset(changed))
            self.assertEqual(len(seen), len(sets))

    def test_triclinic_cells_need_isotropic_coupling(self):
        cell = np.array([[22., 0., 0.], [11., 19., 0.], [0., 0., 22.]])
        b = barostat('isotropic', cell=cell)
        k, new, M = b.propose(engine_cell(cell))
        np.testing.assert_allclose(new, engine_cell(cell) * (new[0, 0] / 22.), rtol=1e-6, atol=1e-6)
        for mode in ('semi-isotropic', 'anisotropic'):
            with self.assertRaisesRegex(ValueError, 'orthorhombic'):
                barostat(mode, cell=cell)


class SPMECellTests(unittest.TestCase):
    def test_set_cell_matches_a_mesh_built_for_the_new_cell(self):
        rng = np.random.default_rng(4)
        sigma = 2.5
        old, new = np.diag([12., 13., 11.]), np.diag([12., 13., 11.]) * 1.02
        x = torch.tensor(rng.uniform(0, 11, (10, 3)))
        q = torch.tensor(rng.normal(size=10))
        q -= q.mean()
        moved = spme.GaussianSPME(old, spacing=0.5 * sigma, order=6, dtype=torch.float64)
        moved.set_sigma(sigma)
        fresh = spme.GaussianSPME(new, spacing=0.5 * sigma, order=6, dtype=torch.float64, mesh=moved.mesh)
        fresh.set_sigma(sigma)
        tables = (moved.m2, moved.pref, moved.frac)
        moved.set_cell(new, sigma)
        self.assertTrue(all(a is b for a, b in zip(tables, (moved.m2, moved.pref, moved.frac))))
        s = torch.tensor(sigma, dtype=torch.float64)
        self.assertAlmostEqual(float(moved.energy(q, x * 1.02, s)[0]), float(fresh.energy(q, x * 1.02, s)[0]),
                               delta=1e-12)

    def test_an_omitted_mesh_sum_stays_omitted_only_while_negligible(self):
        mesh = spme.GaussianSPME(np.eye(3) * 29.6, spacing=0.5 * 72.95, order=6)
        self.assertTrue(mesh.set_sigma(72.95))
        mesh.set_cell(np.eye(3) * 31.0, 72.95)          # still below 1e-30
        with self.assertRaisesRegex(ValueError, 'RECIPROCAL_REGIME_CHANGED'):
            mesh.set_cell(np.eye(3) * 60.0, 72.95)


class NPTOptionTests(unittest.TestCase):
    def config(self, **options):
        return md.RunConfig(system='unused.npz', model='presto', **options)

    def test_options_are_checked(self):
        self.config(pressure_bar=1.0, barostat='semi-isotropic', barostat_interval=25)
        for options, message in ((dict(pressure_bar=1.0, tau_ps=0), 'thermostat'),
                                 (dict(pressure_bar=float('nan')), 'pressure_bar'),
                                 (dict(pressure_bar=1.0, barostat='xy'), 'barostat must be'),
                                 (dict(pressure_bar=1.0, barostat_interval=0), 'barostat_interval'),
                                 (dict(pressure_bar=1.0, gpus=2), 'one GPU')):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                self.config(**options)

    def test_restart_box_is_validated(self):
        system = dict(total_charge=np.float64(0), atomic_numbers=np.array([8, 1, 1]))
        validate_restart_state(system, dict(box_A=np.eye(3) * 10.))
        with self.assertRaisesRegex(ValueError, 'box_A'):
            validate_restart_state(system, dict(box_A=-np.eye(3)))


if __name__ == '__main__':
    unittest.main()
