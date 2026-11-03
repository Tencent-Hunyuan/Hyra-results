"""Settings checks without a model or GPU: flexible-MD inputs, SPME mesh and policy, the CPU neighbour list,
multiplicity and input validation, frame cadence and the CLI's one-line errors.

GPU-only parts (RATTLE without constraints) are skipped without CUDA.
"""
import itertools
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from hyff import api, registry
from hyff.engines.capture import long_range as policy
from hyff.graph import dense_reference_edges, neighbor_pairs, radius_edges
from hyff.io import check_md_system
from hyff.kernels.long_range import spme
from hyff.md import simulation as md
from hyff.md.masses import repartition_hydrogen_mass, water_indices
from hyff.md.periodic import molecule_ids_from_pairs
from hyff.models.latent_charges import LatentChargeEnergy

ROOT = Path(__file__).resolve().parents[1]
WATER = np.array([[0., 0., 0.], [.9572, 0., 0.], [-.2399872, .927297, 0.]])


def water_box(copies=2, spacing=3.1, constraints=True):
    """A small periodic water system in the MD file layout."""
    grid = np.array(list(itertools.product(range(copies), repeat=3)), dtype=np.float64) * spacing
    positions = np.concatenate([WATER + g for g in grid])
    n = len(positions)
    pairs = np.array([[3 * k + a, 3 * k + b] for k in range(len(grid)) for a, b in ((0, 1), (0, 2), (1, 2))])
    distances = np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1)
    system = dict(positions_A=positions, atomic_numbers=np.tile([8, 1, 1], len(grid)),
                  velocities_A_ps=np.zeros((n, 3)), masses=np.tile([15.999, 1.008, 1.008], len(grid)),
                  box_A=np.eye(3) * copies * spacing, total_charge=np.float64(0.))
    if constraints:
        system.update(constraint_pairs=pairs, constraint_distances_A=distances)
    else:
        system.update(constraint_pairs=np.zeros((0, 2), np.int64), constraint_distances_A=np.zeros(0))
    return system


class FlexibleSystemTests(unittest.TestCase):
    def test_empty_constraints_are_a_valid_md_system(self):
        for empty in (np.zeros((0, 2), np.int64), np.zeros(0, np.int64)):
            system = water_box(constraints=False)
            system['constraint_pairs'] = empty
            self.assertIs(check_md_system(system), system)

    def test_bad_constraints_are_named(self):
        system = water_box()
        system['constraint_pairs'] = system['constraint_pairs'].copy()
        system['constraint_pairs'][0, 1] = 999
        with self.assertRaisesRegex(ValueError, 'constraint_pairs'):
            check_md_system(system)
        system = water_box()
        system['constraint_distances_A'] = system['constraint_distances_A'][:-1]
        with self.assertRaisesRegex(ValueError, 'one distance per constraint'):
            check_md_system(system)

    def test_groups_without_constraints_are_single_atoms(self):
        ids = molecule_ids_from_pairs(5)
        np.testing.assert_array_equal(ids, np.arange(5))
        ids = molecule_ids_from_pairs(6, [[0, 1], [1, 2]], np.array([[4, 5]]))
        np.testing.assert_array_equal(ids, [0, 0, 0, 1, 2, 2])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
    def test_rattle_without_constraints_is_plain_velocity_verlet(self):
        from hyff.md.captured_rattle import CapturedRattleStep
        from hyff.md.constraints import RattleConstraints
        mass = torch.tensor([15.999, 1.008, 1.008], dtype=torch.float64, device='cuda')
        constraints = RattleConstraints(mass, np.zeros((0, 2), np.int64), np.zeros(0), solver='coupled')
        self.assertTrue(constraints.empty)
        x = torch.tensor(WATER, dtype=torch.float64, device='cuda')
        v = torch.full_like(x, 0.01)
        constraints.project_positions(x.clone(), x, v, 1.)
        constraints.project_velocities(x, v)
        force = torch.full((3, 3), 0.1, dtype=torch.float32, device='cuda')
        step = CapturedRattleStep(x, v, force, mass, 0.5, constraints)
        position, velocity, *_ = step.step(lambda current: (torch.zeros(1, device='cuda'), force.clone()))
        from hyff.md.units import ACCELERATION
        a = ACCELERATION * force.double() / mass[:, None]
        torch.testing.assert_close(position, x + 0.5 * (v + 0.25 * a), rtol=0, atol=1e-9)  # kick in fp32 * fp64
        torch.testing.assert_close(velocity, v + 0.5 * a, rtol=0, atol=1e-9)


class SPMETests(unittest.TestCase):
    def test_mesh_is_never_smaller_than_the_spline_order(self):
        # The released largo sigma (~73 A) and 0.5 sigma spacing used to give 1-5 points below ~180 A.
        for length in (5., 23.3, 29.6, 100., 179.):
            mesh = spme.GaussianSPME(np.eye(3) * length, spacing=0.5 * 72.95, order=6)
            self.assertGreaterEqual(min(mesh.mesh), 6)
            self.assertTrue(all(length / k <= 0.5 * 72.95 for k in mesh.mesh))
        with self.assertRaisesRegex(ValueError, 'SPME_MESH_SMALLER_THAN_ORDER'):
            spme.GaussianSPME(np.eye(3) * 20., spacing=1., order=6, mesh=(4, 6, 6))

    def test_underflowing_reciprocal_sum_is_omitted_exactly(self):
        sigma = torch.tensor(72.95, dtype=torch.float64)
        mesh = spme.GaussianSPME(np.eye(3) * 23.3, spacing=0.5 * 72.95, order=6, dtype=torch.float64)
        self.assertTrue(mesh.reciprocal_negligible(72.95))
        rng = np.random.default_rng(1)
        x = torch.tensor(rng.uniform(0, 23.3, (12, 3)))
        q = torch.tensor(rng.normal(size=12))
        computed, _ = mesh.energy(q, x, sigma)          # explicit mesh sum, in fp64: ~1e-80 of the total
        mesh.set_sigma(72.95)
        omitted, parts = mesh.energy(q, x, sigma)
        self.assertEqual(float(parts['reciprocal']), 0.)
        self.assertAlmostEqual(float(omitted), float(computed), delta=1e-12 * abs(float(computed)))
        self.assertIn('omitted', mesh.describe()['reciprocal'])
        large = spme.GaussianSPME(np.eye(3) * 400., spacing=0.5 * 72.95, order=6)
        self.assertFalse(large.reciprocal_negligible(72.95))

    def test_small_mesh_matches_the_explicit_ewald_reference(self):
        rng = np.random.default_rng(2)
        cell = np.diag([12., 13., 11.]) + rng.uniform(-1, 1, (3, 3))
        x = torch.tensor(rng.uniform(-5, 20, (10, 3)))
        q = torch.tensor(rng.normal(size=10))
        q -= q.mean()
        sigma = 2.5
        mesh = spme.GaussianSPME(cell, spacing=0.5 * sigma, order=6, dtype=torch.float64)
        mesh.set_sigma(sigma)
        self.assertFalse(mesh.skip_reciprocal)
        energy, _ = mesh.energy(q, x, torch.tensor(sigma, dtype=torch.float64))
        reference, _, _ = spme.ewald_direct(q, x, cell, sigma, sigma)
        self.assertAlmostEqual(float(energy), float(reference), delta=2e-4 * abs(float(reference)) + 1e-8)
        # Invariant to lattice translations of single atoms.
        shifted = x.clone()
        shifted[3] += torch.tensor(cell[1])
        moved, _ = mesh.energy(q, shifted, torch.tensor(sigma, dtype=torch.float64))
        self.assertAlmostEqual(float(moved), float(energy), delta=1e-10)

    def test_minimum_mesh_is_accurate_for_the_released_sigma_in_a_large_box(self):
        # sigma = 73 A in a 150 A box: the mesh has 6 points per axis (the order) and the reciprocal sum is
        # not negligible; it must agree with the explicit Ewald reference.
        rng = np.random.default_rng(5)
        sigma = 72.95
        cell = np.eye(3) * 150.
        x = torch.tensor(rng.uniform(0, 150, (40, 3)))
        q = torch.tensor(rng.normal(size=40))
        mesh = spme.GaussianSPME(cell, spacing=0.5 * sigma, order=6, dtype=torch.float64)
        self.assertEqual(mesh.mesh, (6, 6, 6))
        self.assertFalse(mesh.set_sigma(sigma))
        energy, parts = mesh.energy(q, x, torch.tensor(sigma, dtype=torch.float64))
        reference, ref_parts, _ = spme.ewald_direct(q, x, cell, sigma, sigma)
        self.assertGreater(abs(float(ref_parts['reciprocal'])), 1e-4 * abs(float(reference)))
        self.assertAlmostEqual(float(parts['reciprocal']), float(ref_parts['reciprocal']),
                               delta=1e-6 * abs(float(reference)))

    def test_backend_validates_on_the_cpu_before_compilation(self):
        from hyff.kernels.long_range.backends import validate_spme
        term = LatentChargeEnergy(8)
        with torch.no_grad():
            term.sigma_raw.fill_(math.log(math.expm1(72.95 - 1e-3)))
        described = validate_spme(term, np.eye(3) * 23.3)
        self.assertTrue(described['reciprocal_omitted'])
        self.assertGreaterEqual(min(described['mesh']), 6)
        with self.assertRaisesRegex(ValueError, 'NONSINGULAR'):
            validate_spme(term, np.diag([20., 20., 0.]))


class LongRangePolicyTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {})
        patcher.start()
        self.addCleanup(patcher.stop)
        for key in list(os.environ):
            if key.startswith('LQ_'):
                del os.environ[key]

    def test_auto_is_periodic_with_a_cell_and_isolated_without(self):
        self.assertEqual(policy.resolve_long_range('largo', 'auto', 918, periodic=True)['selected'], 'spme')
        self.assertEqual(policy.resolve_long_range('largo', 'auto', 918)['selected'], 'dense')
        self.assertEqual(policy.resolve_long_range('largo', 'auto', 5000)['selected'], 'tree')
        self.assertEqual(policy.resolve_long_range('largo', 'dense', 918, periodic=True)['selected'], 'dense')
        self.assertIsNone(policy.resolve_long_range('largo', None, 918, periodic=True))
        with self.assertRaisesRegex(ValueError, 'needs a periodic cell'):
            policy.resolve_long_range('largo', 'spme', 918)

    def test_environment_settings_are_used_and_checked(self):
        os.environ.update(LQ_DENSE_MAX='10', LQ_SPME_H_SIGMA='0.25', LQ_SPME_DTYPE='fp64')
        self.assertEqual(policy.resolve_long_range('largo', 'auto', 11)['selected'], 'tree')
        values = policy.settings()
        self.assertEqual((values['LQ_SPME_H_SIGMA'], values['LQ_SPME_MAX_H_SIGMA'], values['LQ_SPME_DTYPE']),
                         (0.25, 0.25, 'fp64'))
        for name, value in (('LQ_SPME_ORDER', '2'), ('LQ_SPME_DTYPE', 'fp16'), ('LQ_DENSE_MAX', 'many'),
                            ('LQ_SPME_MAX_H_SIGMA', '0.1'), ('LQ_THETA', '1.5')):
            with self.subTest(name=name), mock.patch.dict(os.environ, {name: value}):
                with self.assertRaisesRegex(ValueError, name):
                    policy.resolve_long_range('largo', 'auto', 100, periodic=True)


class CPUNeighborListTests(unittest.TestCase):
    def brute(self, x, cutoff, cell, pbc, reach=12):
        found = set()
        for shift in itertools.product(*[range(-reach, reach + 1) if p else [0] for p in pbc]):
            shift = np.array(shift)
            d = x[None] - x[:, None] + (shift @ cell if cell is not None else 0)
            for i, j in zip(*np.nonzero((d ** 2).sum(-1) < cutoff ** 2)):
                if i != j or shift.any():
                    found.add((int(i), int(j), *map(int, shift)))
        return found

    def test_matches_brute_force_including_cells_shorter_than_the_cutoff(self):
        rng = np.random.default_rng(3)
        for trial in range(6):
            cell = np.diag(rng.uniform(4, 9, 3)) + rng.uniform(-1.5, 1.5, (3, 3))
            pbc = [True, True, True] if trial < 3 else [bool(b) for b in rng.integers(0, 2, 3)]
            x = rng.uniform(-4, 12, (int(rng.integers(4, 25)), 3))
            cutoff = float(rng.uniform(2, 7))
            i, j, s = neighbor_pairs(x, cutoff, cell, pbc)
            with self.subTest(trial=trial):
                self.assertEqual(set(zip(i.tolist(), j.tolist(), *s.T.tolist())), self.brute(x, cutoff, cell, pbc))

    def test_nonperiodic_cell_list_equals_the_dense_reference(self):
        rng = np.random.default_rng(4)
        positions = torch.tensor(rng.uniform(0, 40, (2100, 3)), dtype=torch.float32)
        batch = dict(positions=positions, batch=torch.zeros(2100, dtype=torch.long), ptr=torch.tensor([0, 2100]))
        cell_list = radius_edges(batch, 4.0)                   # >= 2048 atoms: cell list
        dense = dense_reference_edges(batch, 4.0)
        self.assertEqual(set(map(tuple, cell_list.T.tolist())), set(map(tuple, dense.T.tolist())))

    def test_periodic_cpu_inference_without_ase(self):
        if registry.bundled_weights('presto') is None:
            self.skipTest('presto weights are not installed')
        class NoASE:   # fail any import of ase while the model runs
            def find_spec(self, name, path=None, target=None):
                if name == 'ase' or name.startswith('ase.'):
                    raise ImportError('ase is not available')
                return None
        model = api.load('presto', device='cpu')
        hidden = {name: module for name, module in sys.modules.items() if name == 'ase' or name.startswith('ase.')}
        for name in hidden:
            del sys.modules[name]
        sys.meta_path.insert(0, NoASE())
        self.addCleanup(sys.modules.update, hidden)
        self.addCleanup(sys.meta_path.pop, 0)
        box = water_box()
        energy, forces = model.energy_forces(box['positions_A'], box['atomic_numbers'], box['box_A'])
        shifted = box['positions_A'].copy()
        shifted[:3] += box['box_A'][2]          # one molecule by a lattice vector
        energy2, forces2 = model.energy_forces(shifted, box['atomic_numbers'], box['box_A'])
        self.assertTrue(np.isfinite(energy) and np.isfinite(forces).all())
        self.assertAlmostEqual(energy, energy2, delta=1e-4)
        np.testing.assert_allclose(forces, forces2, atol=1e-4)


class InferenceInputTests(unittest.TestCase):
    def test_multiplicity_must_be_an_integer(self):
        self.assertEqual(api._multiplicity(2), 2)
        self.assertEqual(api._multiplicity(np.int64(3)), 3)
        self.assertEqual(api._multiplicity(2.0), 2)
        for bad in (2.5, 0, -1, True, float('nan'), '2', [1]):
            with self.subTest(value=bad), self.assertRaisesRegex(ValueError, 'multiplicity must be an integer'):
                api._multiplicity(bad)

    def test_structure_inputs_have_clear_errors(self):
        api._structure(WATER, [8, 1, 1], np.eye(3) * 10)
        for args, pattern in (((WATER[:, :2], [8, 1, 1]), 'positions'), ((WATER, [8, 1]), 'numbers'),
                              ((WATER, [8., 1., 1.]), 'numbers'), ((WATER, [8, 1, 1], [10., 10., 10.]), 'cell'),
                              ((WATER, [8, 1, 1], np.diag([20., 20., 0.])), 'zero volume'),
                              ((WATER, [8, 1, 1], None, float('inf')), 'charge')):
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                api._structure(*args)

    def test_model_evaluates_structures_of_different_sizes(self):
        if registry.bundled_weights('presto') is None:
            self.skipTest('presto weights are not installed')
        model = api.load('presto', device='cpu')
        dimer = np.concatenate([WATER, WATER + [3.0, 0.2, 0.1]])
        e1, _ = model.energy_forces(WATER, [8, 1, 1])
        e2, f2 = model.energy_forces(dimer, [8, 1, 1] * 2)
        e3, _ = model.energy_forces(WATER, [8, 1, 1])
        fresh = api.load('presto', device='cpu').energy_forces(dimer, [8, 1, 1] * 2)
        self.assertEqual(f2.shape, (6, 3))
        self.assertEqual(e1, e3)
        self.assertAlmostEqual(e2, fresh[0], delta=1e-6)
        with self.assertRaisesRegex(ValueError, 'multiplicity'):
            model.energy_forces(WATER, [8, 1, 1], multiplicity=2.5)


class MDValidationTests(unittest.TestCase):
    def config(self, **options):
        return md.RunConfig(system='unused.npz', model='presto', **options)

    def test_run_options_are_checked(self):
        self.config(tau_ps=0, temperature_K=0)
        for options, name in ((dict(tau_ps=-0.1), 'tau_ps'), (dict(temperature_K=-5), 'temperature_K'),
                              (dict(tau_ps=0.5, temperature_K=0), 'temperature_K'), (dict(chunk=0), 'chunk'),
                              (dict(chunk=2000), 'chunk'), (dict(save_every=0), 'save_every'),
                              (dict(timestep_fs=-1.), 'timestep_fs'), (dict(timestep_fs=0.), 'timestep_fs'),
                              (dict(timestep_fs=float('nan')), 'timestep_fs'), (dict(timestep_fs=20.), 'timestep_fs'),
                              (dict(steps=-1), 'steps'), (dict(steps=1.5), 'steps'), (dict(skin=0), 'skin'),
                              (dict(max_wall_seconds=-1), 'max_wall_seconds'), (dict(hmr_mass=0), 'hmr_mass'),
                              (dict(recenter_interval_ps=-1.), 'recenter_interval_ps')):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, name):
                self.config(**options)

    def test_missing_keys_are_listed(self):
        single = dict(positions_A=WATER, atomic_numbers=np.array([8, 1, 1]))
        with self.assertRaisesRegex(ValueError, 'missing MD keys: velocities_A_ps, masses, box_A'):
            check_md_system(single, 'water.npz')
        system = water_box()
        del system['total_charge']
        with self.assertRaisesRegex(ValueError, 'missing MD keys: total_charge'):
            check_md_system(system)
        system = water_box()
        system['box_A'] = -system['box_A']
        with self.assertRaisesRegex(ValueError, 'positive volume'):
            check_md_system(system)
        system = water_box()
        system['multiplicity'] = np.float64(2.)
        with self.assertRaisesRegex(ValueError, 'multiplicity'):
            check_md_system(system)

    def test_hmr_without_water_idx(self):
        # No water and no water_idx: HMR applies to every constrained X-H.
        methanol = dict(atomic_numbers=np.array([6, 8, 1, 1, 1, 1]), masses=np.array([12.011, 15.999] + [1.008] * 4),
                        constraint_pairs=np.array([[0, 2], [0, 3], [0, 4], [1, 5]]))
        masses = repartition_hydrogen_mass(methanol, 3.0)
        np.testing.assert_allclose(masses[2:], 3.0)
        self.assertAlmostEqual(masses.sum(), methanol['masses'].sum())
        # Water is found from the O-H-H constraint clusters and keeps its masses.
        box = water_box(copies=1)
        np.testing.assert_array_equal(water_indices(box), [0, 1, 2])
        np.testing.assert_allclose(repartition_hydrogen_mass(dict(box, masses=box['masses']), 3.0), box['masses'])
        with self.assertRaisesRegex(ValueError, 'X-H bonds in constraint_pairs'):
            repartition_hydrogen_mass(water_box(constraints=False), 3.0)

    def test_predict_batch_options_are_checked_before_loading(self):
        with mock.patch.object(api, 'resolve', side_effect=AssertionError('no model may be resolved')):
            with self.assertRaisesRegex(ValueError, 'isolated \\(nonperiodic\\) molecules'):
                api.predict_batch('presto', WATER, [8, 1, 1], [0, 3], cell=np.eye(3) * 10)
            with self.assertRaisesRegex(ValueError, "Unknown predict_batch option 'batch_size'"):
                api.predict_batch('presto', WATER, [8, 1, 1], [0, 3], batch_size=4)

    def test_unknown_simulation_option(self):
        with self.assertRaisesRegex(ValueError, "Unknown Simulation option 'thermostat'"):
            api.Simulation('unused.npz', 'presto', thermostat='langevin')

    def test_cli_prints_one_line_errors(self):
        with tempfile.TemporaryDirectory() as work:
            np.savez(Path(work) / 'water.npz', positions_A=WATER, atomic_numbers=np.array([8, 1, 1]))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
            if env.get('HYFF_TEST_INSTALLED') != '1':
                env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'src'), env.get('PYTHONPATH', '')])
            command = [sys.executable, '-m', 'hyff', 'md', '--system', str(Path(work) / 'water.npz'),
                       '--output', str(Path(work) / 'run'), '--steps', '10']
            done = subprocess.run(command, capture_output=True, text=True, env=env, cwd=work)
            self.assertEqual(done.returncode, 1)
            lines = [line for line in done.stderr.splitlines() if line.startswith('hyff')]
            self.assertEqual(len(lines), 1, done.stderr)
            self.assertIn('missing MD keys', lines[0])
            self.assertNotIn('Traceback', done.stderr)
            debug = subprocess.run(command + ['--debug'], capture_output=True, text=True, env=env, cwd=work)
            self.assertEqual(debug.returncode, 1)
            self.assertIn('Traceback', debug.stderr)
            bad = subprocess.run(command[:-2] + ['--tau-ps', '-1'], capture_output=True, text=True, env=env, cwd=work)
            self.assertIn('tau_ps must be', bad.stderr)


class LargoMemoryDefaultsTests(unittest.TestCase):
    def test_capacity_auto_and_shared_expert_storage_are_defaults(self):
        import inspect
        from hyff.engines import dispatch
        from hyff.engines.capture.capacity import resolve_capacity_policy
        self.assertEqual(registry.DEFAULT_OPTIONS['largo']['capacity_policy'], 'auto')
        self.assertEqual(registry.DEFAULT_OPTIONS['andante']['capacity_policy'], 'auto')
        for small in (2586, 20388):
            self.assertEqual(resolve_capacity_policy('largo', 'auto', small)['selected'], 'native')
        self.assertEqual(resolve_capacity_policy('largo', 'auto', 51720)['selected'], 'partial')
        self.assertEqual(resolve_capacity_policy('largo', 'auto', 103440)['selected'], 'tiled')
        self.assertEqual(resolve_capacity_policy('andante', 'auto', 50000)['selected'], 'native')
        # None means: on for the largo engine, off elsewhere.
        self.assertIsNone(inspect.signature(api.create_engine).parameters['expert_storage_alias'].default)
        self.assertIsNone(md.RunConfig(system='unused.npz').expert_storage_alias)
        with self.assertRaisesRegex(ValueError, 'only by the largo engine'):
            dispatch.create_engine('andante', 'unused', np.zeros((3, 3), np.float32), np.array([8, 1, 1]),
                                   expert_storage_alias=True)
        with self.assertRaisesRegex(ValueError, 'bool or None'):
            dispatch.create_engine('largo', 'unused', np.zeros((3, 3), np.float32), np.array([8, 1, 1]),
                                   expert_storage_alias='yes')


class CardCountRuleTests(unittest.TestCase):
    def test_measured_table_keeps_the_selection(self):
        from hyff.multigpu import card_guidance as guidance
        from hyff.multigpu.launcher import _auto_timing_service_policy, _service_policy
        cfg = SimpleNamespace(model='largo', engine=None, precision=None, statistics=None, long_range=None,
                              timestep_fs=1.0, chunk=100, skin=1.5, reserve=1.5, hmr_mass=None,
                              inclusive_hmr=False, tau_ps=0.0)
        policy = _auto_timing_service_policy(_service_policy())
        device = dict(device_class='NVIDIA H20', capability=(9, 0))
        for atoms, expected in ((2586, [1, 1, 1, 1]), (103439, [1, 1, 1, 1]),
                                (103440, [8, 4, 1, 1]), (400000, [8, 4, 1, 1])):
            physical = dict(atoms=atoms, native_engine='largo', native_precision='tf32x3', native_statistics='fp64')
            chosen = [guidance.select_card_count(cfg, physical, ranks, device, service_policy=policy)['ranks']
                      for ranks in ((1, 2, 4, 8), (1, 2, 4), (1, 2), (1,))]
            with self.subTest(atoms=atoms):
                self.assertEqual(chosen, expected)
        means = {row.ranks: row.mean_ms for row in guidance.MEASURED_TABLE if row.atoms == 103440}
        self.assertEqual({k: round(v) for k, v in means.items()}, {1: 6347, 4: 2207, 8: 1173})


class FrameCadenceTests(unittest.TestCase):
    def test_frames_start_at_zero_and_are_exactly_save_every_apart(self):
        run = object.__new__(md.MDRun)
        run.cfg = md.RunConfig(system='unused.npz', model='presto', steps=1000, chunk=100, save_every=150,
                               tau_ps=0, timestep_fs=1.0)
        run.stepper = SimpleNamespace(position=torch.zeros((2, 3), dtype=torch.float64),
                                      velocity=torch.zeros((2, 3), dtype=torch.float64))
        run.periodic_molecules = None
        run.mass = torch.ones(2, dtype=torch.float64)
        run.ndof = 3
        run.dt = 1.0
        run.result = dict(status='running', trace=[])
        run.driver_base_steps = 0
        counts, saved = [], []
        run.advance = lambda count: (counts.append(count), (count, torch.zeros(1)))[1]
        run.save = run._write_final_state = mock.Mock()

        def flush(frames, times, wall, final=False):
            saved.extend(times)
            frames.clear()
            times.clear()
        run.flush = flush
        with mock.patch.object(md.torch.cuda, 'synchronize'), \
                mock.patch.object(md.torch, 'Generator', return_value=torch.Generator()):
            run.run()
        self.assertEqual(sum(counts), 1000)
        self.assertLessEqual(max(counts), 100)
        np.testing.assert_allclose(saved, np.arange(0, 1000, 150) / 1000.)


if __name__ == '__main__':
    unittest.main()
