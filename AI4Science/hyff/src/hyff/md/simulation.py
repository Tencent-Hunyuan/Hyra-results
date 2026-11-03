"""Constrained or flexible NVE/NVT/NPT molecular dynamics on a prepared system NPZ.

FP64 RATTLE integration of FP32 forces (no constraints: plain velocity Verlet), native or captured
neighbour rebuilding on the GPU, a chunk-boundary Bussi thermostat and, for NPT, a chunk-boundary
Monte Carlo barostat. ``RunConfig`` holds every option and checks it; ``MDRun`` checks the system and
executes one run. Every input check runs before any engine is built. The public entry point is
``hyff.Simulation``.
"""
from __future__ import annotations

import json
import math
import warnings
import statistics
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import torch

from hyff.engines.dispatch import create_engine
from hyff.io import (
    atomic_numbers_sha256, check_md_system, load_restart_state, load_system, restart_source,
    system_multiplicity, validate_restart_state,
)
from hyff.md.barostat import AMU_A3_G_CM3, MODES as BAROSTAT_MODES, MonteCarloBarostat, engine_cell, make_whole
from hyff.md.masses import repartition_hydrogen_mass, repartition_hydrogen_mass_inclusive
from hyff.md.thermostat import bussi_factor
from hyff.md.units import ACCELERATION, KB_EV_K
from hyff.registry import NATIVE_ENGINES, resolve

GPU_COUNTS = (1, 2, 4, 8)
MAX_CHUNK = 1024          # steps per device launch (CUDA conditional loop)
MAX_TIMESTEP_FS = 10.
CELL_HEADROOM = 1.1       # NPT: neighbour bins sized for every cell length grown by 10%


def _integer(name, value, minimum, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum or (
            maximum is not None and value > maximum):
        bound = f'from {minimum} to {maximum}' if maximum is not None else f'>= {minimum}'
        raise ValueError(f'{name} must be an integer {bound}, got {value!r}')
    return int(value)


def _real(name, value, *, minimum=None, above=None, maximum=None, optional=False):
    if value is None and optional:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = math.nan
    if (isinstance(value, bool) or not math.isfinite(number) or (minimum is not None and number < minimum) or
            (above is not None and number <= above) or (maximum is not None and number > maximum)):
        parts = [f'>= {minimum}' if minimum is not None else None, f'> {above}' if above is not None else None,
                 f'<= {maximum}' if maximum is not None else None]
        raise ValueError(f'{name} must be a finite number {" and ".join(x for x in parts if x)}, got {value!r}')
    return number


@dataclass
class RunConfig:
    system: str
    model: str = 'presto'
    weights: str | None = None       # explicit checkpoint (file, sharded index or directory)
    engine: str | None = None        # None selects the model's own engine
    steps: int = 10000
    timestep_fs: float = 1.0
    temperature_K: float = 300.0
    tau_ps: float = 0.5              # 0 runs NVE
    chunk: int = 100                 # steps per device launch; the thermostat acts between chunks
    save_every: int = 500            # steps between saved frames
    skin: float = 1.5
    reserve: float = 1.5
    precision: str | None = None     # None keeps the model default
    statistics: str | None = None    # None keeps the model default
    edge_reserve: float | None = None
    long_range: str | None = None
    conservation: str | None = None
    capacity_policy: str | None = None
    expert_storage_alias: bool | None = None   # None: on for the largo engine (shared expert storage)
    seed: int = 1
    output: str = 'hyff_md'
    initial_positions: str | None = None   # final_state.npz of an earlier run (restart)
    max_wall_seconds: float | None = None
    hmr_mass: float | None = None    # hydrogen mass (amu) for non-water X-H; the heavy partner loses the difference
    inclusive_hmr: bool = False      # every H, including water H, becomes 3 amu; heavy donors compensate
    constraint_max_iterations: int = 100
    fallback_timestep_fs: float = 1.0
    fallback_steps: int = 1000
    max_recoveries: int = 200
    recovery: str = 'on'             # NVT only; off fails on the first rejected step
    recenter_interval_ps: float | None = None  # lattice recentering of bonded groups at chunk boundaries
    molecule_system_xml: str | None = None     # optional OpenMM system.xml for the groups (default: constraints + bond_pairs)
    pressure_bar: float | None = None          # NPT: Monte Carlo barostat at this pressure; None keeps the cell fixed
    barostat: str = 'isotropic'                # isotropic, semi-isotropic (xy | z) or anisotropic (x | y | z)
    barostat_interval: int = 100               # steps between volume moves (a chunk ends at every move)
    capture_retries: int = 0
    capture_branch_policy: str = 'first'
    gpus: int = 1                    # >1: experimental multi-GPU largo MD with the automatic card policy
    route: str = 'native'            # internal: native, auto or multigpu
    ranks: int | None = None         # internal: rank cap (auto) or rank count (multigpu)
    debug: bool = False              # keep engine diagnostics under "debug" in result.json
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        _integer('steps', self.steps, 0)
        _real('timestep_fs', self.timestep_fs, above=0., maximum=MAX_TIMESTEP_FS)
        _real('tau_ps', self.tau_ps, minimum=0.)
        _real('temperature_K', self.temperature_K, minimum=0.)
        if self.tau_ps > 0:
            _real('temperature_K', self.temperature_K, above=0.)
        _integer('chunk', self.chunk, 1, MAX_CHUNK)
        _integer('save_every', self.save_every, 1)
        _real('skin', self.skin, above=0.)
        _real('reserve', self.reserve, minimum=1.)
        _integer('seed', self.seed, 0)
        _real('max_wall_seconds', self.max_wall_seconds, above=0., optional=True)
        _real('hmr_mass', self.hmr_mass, above=0., optional=True)
        _real('recenter_interval_ps', self.recenter_interval_ps, above=0., optional=True)
        _real('fallback_timestep_fs', self.fallback_timestep_fs, above=0., maximum=MAX_TIMESTEP_FS)
        _integer('fallback_steps', self.fallback_steps, 1)
        _integer('max_recoveries', self.max_recoveries, 0)
        _integer('constraint_max_iterations', self.constraint_max_iterations, 1)
        _integer('capture_retries', self.capture_retries, 0)
        if self.gpus not in GPU_COUNTS:
            raise ValueError('gpus must be 1, 2, 4 or 8')
        if self.gpus > 1 and self.route == 'native':
            self.route, self.ranks = 'auto', self.gpus
        if self.route not in ('native', 'auto', 'multigpu'):
            raise ValueError('route must be native, auto or multigpu')
        if self.ranks is not None and (type(self.ranks) is not int or self.ranks not in GPU_COUNTS):
            raise ValueError('ranks must be one of 1, 2, 4 or 8')
        if self.recovery not in ('on', 'off'):
            raise ValueError('recovery must be on or off')
        if self.capture_branch_policy not in ('off', 'first', 'every'):
            raise ValueError('capture_branch_policy must be off, first or every')
        if type(self.inclusive_hmr) is not bool:
            raise ValueError('inclusive_hmr must be a bool')
        if self.inclusive_hmr and self.hmr_mass is not None:
            raise ValueError('inclusive_hmr and hmr_mass are mutually exclusive')
        _real('pressure_bar', self.pressure_bar, optional=True)
        _integer('barostat_interval', self.barostat_interval, 1)
        if self.barostat not in BAROSTAT_MODES:
            raise ValueError(f'barostat must be one of {", ".join(BAROSTAT_MODES)}')
        if self.pressure_bar is not None:
            if self.tau_ps == 0:
                raise ValueError('pressure_bar needs the thermostat (tau_ps > 0): NPT samples at temperature_K')
            if self.route != 'native':
                raise ValueError('constant-pressure MD runs on one GPU (gpus=1)')

    def resolved(self):
        """The model selection with engine options; explicit options override the defaults."""
        return resolve(self.model, self.weights, engine=self.engine, precision=self.precision,
                       statistics=self.statistics, edge_reserve=self.edge_reserve,
                       long_range=self.long_range, conservation=self.conservation,
                       capacity_policy=self.capacity_policy)


class MDRun:
    barostat = None     # NPT only
    cell = None         # the current periodic cell (FP32 values), set up with the engine

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.out = Path(cfg.output)
        for name in ('result.json', 'final_state.npz'):
            existing = self.out / name
            if existing.exists() or existing.is_symlink():
                raise FileExistsError('MD output already contains ' + name + '; choose a fresh output directory')
        self.engine = None
        self.driver = None
        self.retired_native_neighbor_rebuilds = 0
        self.run_started = None
        self.capture_failure = None
        self.sys = self._load_system(cfg.system)
        check_md_system(self.sys, Path(cfg.system).name)
        self.multiplicity = system_multiplicity(self.sys)
        selection = cfg.resolved()
        self.model_name = selection.name
        self.checkpoint = selection.weights
        self.checkpoint_sha256 = selection.sha256
        self.periodic_molecules = None
        self.kind = selection.engine
        self.execution_options = dict(selection.options)
        # Every remaining input check happens here, before any engine is built.
        self.masses_amu = self._masses()
        self.long_range_route = self._long_range_route(selection)
        self.molecule_ids = self.topology = self.group_pairs = None
        if cfg.pressure_bar is not None:
            if self.long_range_route not in (None, 'spme'):
                raise ValueError(
                    f'pressure_bar needs a periodic long-range sum, but largo would use the isolated '
                    f'{self.long_range_route!r} sum, which has no cell; use long_range=\'spme\' (selected by the '
                    "default 'auto' for MD systems)")
            if cfg.barostat != 'isotropic' and np.any(self.sys['box_A'][~np.eye(3, dtype=bool)] != 0):
                raise ValueError(f'the {cfg.barostat} barostat needs an orthorhombic (diagonal) box_A; use isotropic')
            self.molecule_ids, self.topology = self._molecule_groups()
        if cfg.recenter_interval_ps is not None:
            if self.long_range_route not in (None, 'spme'):
                raise ValueError(
                    f'recenter_interval_ps needs a periodic long-range sum, but largo would use the isolated '
                    f'{self.long_range_route!r} sum, which changes when a molecule is translated by a lattice vector; '
                    "use long_range='spme' (selected by the default 'auto' for MD systems)")
            self.molecule_ids, self.topology = self._molecule_groups()
        self.out.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(cfg.seed)
        self.result = dict(status='setup', model=self.model_name, engine=self.kind, gpus=1,
                           checkpoint_sha256=self.checkpoint_sha256, options=dict(self.execution_options),
                           ensemble='NVE' if cfg.tau_ps == 0 else ('NVT' if cfg.pressure_bar is None else 'NPT'),
                           timestep_fs=cfg.timestep_fs,
                           temperature_K=cfg.temperature_K if cfg.tau_ps else None, tau_ps=cfg.tau_ps,
                           steps=cfg.steps, trace=[], frames_flushed_through_ps=None)
        if cfg.pressure_bar is not None:
            self.result.update(pressure_bar=cfg.pressure_bar,
                               barostat=dict(mode=cfg.barostat, interval_steps=cfg.barostat_interval))

    def _load_system(self, name):
        return load_system(name)

    def _masses(self):
        """Integration masses (amu): the system masses, or hydrogen-repartitioned ones."""
        cfg, s = self.cfg, self.sys
        if cfg.inclusive_hmr:
            return repartition_hydrogen_mass_inclusive(s)
        if cfg.hmr_mass is not None:
            return repartition_hydrogen_mass(s, cfg.hmr_mass)
        return s['masses'].astype(np.float64)

    def _long_range_route(self, selection):
        """The latent-charge sum of this run: spme/dense/tree/native for largo, None without the term."""
        if selection.name != 'largo':
            return None
        if selection.engine != 'largo':
            return 'native'
        from hyff.engines.capture.long_range import resolve_long_range
        policy = resolve_long_range('largo', selection.options.get('long_range'), len(self.sys['atomic_numbers']),
                                    periodic=True)
        return 'native' if policy is None else policy['selected']

    def _molecule_groups(self):
        """Recentering groups: an OpenMM system.xml if given, else the constraint graph plus bond_pairs."""
        from hyff.md.periodic import molecule_ids_from_pairs, molecule_ids_from_xml
        cfg, s = self.cfg, self.sys
        n = len(s['atomic_numbers'])
        if cfg.molecule_system_xml:
            from hyff.md.periodic import pairs_from_xml
            xml = Path(cfg.molecule_system_xml).expanduser()
            if not xml.is_file():
                raise FileNotFoundError(f'molecule_system_xml {xml} does not exist')
            self.group_pairs = [s['constraint_pairs'], pairs_from_xml(xml, n)]
            return molecule_ids_from_xml(xml, n), str(xml.resolve())
        self.group_pairs = [s['constraint_pairs'], s.get('bond_pairs')]
        return (molecule_ids_from_pairs(n, s['constraint_pairs'], s.get('bond_pairs')),
                'constraint_pairs + bond_pairs' if 'bond_pairs' in s else 'constraint_pairs')

    def _load_initial_positions(self, name):
        return load_restart_state(name)

    PUBLIC_KEYS = ('status', 'model', 'engine', 'gpus', 'checkpoint_sha256', 'options', 'ensemble', 'timestep_fs',
                   'temperature_K', 'tau_ps', 'pressure_bar', 'barostat', 'steps', 'steps_done', 'simulated_ps',
                   'ms_per_step', 'ns_per_day',
                   'md_wall_seconds', 'run_wall_seconds', 'setup_seconds', 'n_atoms', 'n_constraints', 'ndof',
                   'total_charge', 'multiplicity', 'neighbor_rebuilds', 'recoveries', 'stop_reason',
                   'restart', 'recenter_events', 'error_type', 'trace')

    def public_result(self):
        """The result.json content: run summary, timings and energy trace; diagnostics only with debug."""
        record = dict(self.result)
        record['steps_done'] = getattr(self, 'steps_done', 0)
        public = {key: record[key] for key in self.PUBLIC_KEYS if key in record}
        if self.cfg.debug:
            debug = {key: value for key, value in record.items() if key not in self.PUBLIC_KEYS}
            if self.engine is not None:
                debug['capture_check'] = self._capture_check()
            debug['config'] = dict(self.cfg.__dict__)
            public['debug'] = debug
        return public

    def save(self):
        self.result['committed_steps'] = getattr(self, 'steps_done', 0)
        temporary = self.out / 'result.pending.json'
        temporary.write_text(json.dumps(self.public_result(), indent=2, default=str) + '\n')
        temporary.replace(self.out / 'result.json')

    def _capture_check(self, error=None):
        """Capture-check record for result.json: the latest strict check and, for the base engine, the last branch check."""
        engine = self.engine
        own = self.kind in ('andante', 'largo')
        record = dict(force_atol_eV_A=getattr(engine, 'capture_force_atol', 1e-4),
                      force_rtol=1e-4, requested_branch_policy=self.cfg.capture_branch_policy,
                      guard_scope='own_prepared_CSR_guard' if own else 'base_ForceEngine')
        if own:
            checks = getattr(engine, 'capture_checks', ())
            record.update(branch_policy_applicable=False,
                          latest=deepcopy(checks[-1]) if checks else None)
        else:
            record.update(branch_policy_applicable=True,
                          latest=deepcopy(getattr(engine, 'capture_guard_diagnostics', None)),
                          branch_status=deepcopy(getattr(engine, 'capture_branch_status', None)),
                          last_branch_check=deepcopy(getattr(engine, 'capture_branch_last_check', None)))
        if error is not None:
            self.capture_failure = dict(error_type=type(error).__name__,
                                        failure_details=deepcopy(getattr(error, 'details', None)))
        if getattr(self, 'capture_failure', None) is not None:
            record.update(deepcopy(self.capture_failure))
        return record

    def _adopt_engine(self, made):
        if self.cfg.recenter_interval_ps is not None:
            from hyff.models.latent_charges import LatentChargeEnergy
            from hyff.kernels.long_range.backends import check_recenter
            try:
                modules = tuple(made['model'].modules())
                for term in modules:
                    if (isinstance(term, LatentChargeEnergy) and
                            bool((term.scale.detach() != 0).any().item())):
                        owners = [owner for owner in modules
                                  if getattr(owner, 'latent_charge_energy', None) is term]
                        if not owners:
                            raise ValueError('LR_RECENTER_REQUIRES_PERIODIC_LR: '
                                             'active latent-charge term has no verified SPME owner')
                        for owner in owners:
                            check_recenter(owner)
            except ValueError as error:
                self.result.update(status='failed', setup_failure='recenter_with_isolated_latent_charge')
                try:
                    made['engine'].close()
                except Exception:
                    pass
                raise ValueError('LR_RECENTER_REQUIRES_PERIODIC_LR: '
                                 'recenter_interval_ps is incompatible with active isolated latent-charge LR: '
                                 'the isolated all-pairs sum uses raw unwrapped coordinates; '
                                 'lattice recentering changes its energy. Keep recentering off, '
                                 'or explicitly select the experimental fixed-cell SPME image sum.') from error
        self.model, self.batch, self.engine = made['model'], made['batch'], made['engine']

    def _create_force_engine(self, positions32, numbers, cell, total_charge, multiplicity):
        """Force-engine construction hook; distributed MD overrides only this method."""
        if self.cfg.route != 'native':
            raise ValueError('Multi-GPU MD is started through hyff.multigpu.launcher.run_md')
        cfg, options = self.cfg, self.execution_options
        return create_engine(self.kind, self.checkpoint, positions32, numbers, cell,
                             total_charge, multiplicity, cfg.skin, cfg.reserve, options['precision'],
                             statistics=options['statistics'], edge_reserve=options.get('edge_reserve'),
                             expert_storage_alias=cfg.expert_storage_alias,
                             capacity_policy=options.get('capacity_policy'),
                             long_range=options.get('long_range'),
                             conservation=options.get('conservation'),
                             capture_branch_policy=cfg.capture_branch_policy)

    def setup(self):
        from hyff.md.captured_rattle import CapturedRattleStep
        from hyff.md.constraints import RattleConstraints
        cfg, s = self.cfg, self.sys
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        t0 = time.perf_counter()
        positions = s['positions_A'].astype(np.float64)
        velocities = s['velocities_A_ps'].astype(np.float64) / 1000.0   # A/fs
        periodic_restart = None
        cell = s['box_A'].astype(np.float64)
        if cfg.initial_positions:
            init = self._load_initial_positions(cfg.initial_positions)
            validate_restart_state(s, init)
            self.result['restart'] = restart_source(init)
            positions = init['positions_A'].astype(np.float64)
            velocities = init['velocities_A_fs'].astype(np.float64)
            if 'box_A' in init:
                # The cell the earlier run ended in (it changes under NPT).
                cell = init['box_A'].astype(np.float64)
            if cfg.recenter_interval_ps is not None and 'bounded_positions_A' in init:
                periodic_restart = {key: init[key].copy() for key in
                    ('bounded_positions_A', 'molecule_images', 'molecule_ids', 'periodic_cell_A')}
                positions = periodic_restart['bounded_positions_A'].astype(np.float64)
        numbers = s['atomic_numbers'].astype(np.int64)
        self.n = n = len(numbers)
        if cfg.pressure_bar is not None:
            # Scaling group centres needs contiguous groups; lattice translations leave the energy unchanged.
            cell = engine_cell(cell)
            positions, moved = make_whole(positions, cell, n, self.group_pairs)
            self.result['barostat']['atoms_made_whole'] = moved
        masses = self.masses_amu
        mass = torch.as_tensor(masses, device='cuda', dtype=torch.float64)
        if cfg.hmr_mass or cfg.inclusive_hmr:
            # Equipartition at the new masses; configurational sampling is mass independent.
            velocities = velocities * np.sqrt(s['masses'] / masses)[:, None]
        position = torch.as_tensor(positions, device='cuda', dtype=torch.float64).contiguous()
        velocity = torch.as_tensor(velocities, device='cuda', dtype=torch.float64).contiguous()
        velocity -= (velocity * mass[:, None]).sum(0) / mass.sum()
        if cfg.recenter_interval_ps is not None:
            from hyff.md.periodic import PeriodicMolecules
            ids = self.molecule_ids
            self.periodic_molecules = PeriodicMolecules(ids, cell, position.device)
            if periodic_restart is not None:
                if (not np.array_equal(ids, periodic_restart['molecule_ids']) or
                        not np.array_equal(self.periodic_molecules.cell.cpu().numpy(), periodic_restart['periodic_cell_A'])):
                    raise ValueError('Periodic restart topology or force cell differs from this system')
                self.periodic_molecules.images.copy_(torch.as_tensor(periodic_restart['molecule_images'], device=position.device))
            event = self.periodic_molecules.recenter(position)
            self.result['recenter_events'] = [dict(time_ps=0., phase='setup', **event)]
            self.result['coordinate_policy'] = dict(kind='bonded_group_lattice_translations',
                interval_ps=cfg.recenter_interval_ps, topology=self.topology,
                n_molecules=len(self.periodic_molecules.anchors), output='unwrapped via integer image counters',
                force_cell_A=self.periodic_molecules.cell.cpu().tolist())
            self.next_recenter_fs = cfg.recenter_interval_ps * 1000
        constraints = RattleConstraints(mass, pairs=s['constraint_pairs'], distances=s['constraint_distances_A'],
                                        source=dict(system=cfg.system, kind='OpenMM HBonds + rigid water lengths'),
                                        solver='coupled', tolerance=1e-10, max_iterations=cfg.constraint_max_iterations, check=True)
        previous = position.clone()
        constraints.project_positions(previous, position, torch.zeros_like(position), 1.)
        constraints.project_velocities(position, velocity)
        # Optional jitter retries apply only to a strict captured-versus-uncaptured
        # rejection. A gross prepared-versus-branch discrepancy is terminal and
        # is never jitter-retried; changing coordinates does not repair it.
        if cfg.capture_retries < 0:
            raise ValueError('capture_retries must be nonnegative')
        for attempt in range(cfg.capture_retries + 1):
            try:
                made = self._create_force_engine(position.float().cpu().numpy(), numbers, cell,
                                                 float(s['total_charge']), self.multiplicity)
                break
            except Exception as error:
                if type(error).__name__ != 'CaptureEquivalenceError' or attempt == cfg.capture_retries:
                    self.result.update(status='failed', capture_check=self._capture_check(error))
                    try:
                        self.save()
                    except Exception:
                        pass
                    raise
                warnings.warn(f'Capture self-check failed (attempt {attempt + 1}); '
                              'jittering the start state by 2e-4 A', RuntimeWarning, stacklevel=2)
                previous = position.clone()
                position += 2e-4 * torch.randn_like(position)
                constraints.project_positions(previous, position, torch.zeros_like(position), 1.)
                constraints.project_velocities(position, velocity)
                torch.cuda.empty_cache()
        self._adopt_engine(made)
        # Eager reference engines return E/F without CUDA-graph static outputs.
        if self.engine.backend == 'eager':
            energy0, force0 = self.engine(self.batch)
            energy0, force0 = energy0.detach().clone(), force0.detach().clone()
        else:
            energy0 = self.engine.static_energy.detach().clone()
            force0 = self.engine.static_force.detach().clone()
        self.mass = mass
        self.base_mass = torch.as_tensor(s['masses'], device='cuda', dtype=torch.float64)
        self.stepper = CapturedRattleStep(position, velocity, force0, mass, cfg.timestep_fs, constraints)
        self.cell = engine_cell(cell)
        self.barostat = None
        if cfg.pressure_bar is not None:
            self.barostat = MonteCarloBarostat(cfg.pressure_bar, cfg.temperature_K, cfg.barostat, self.molecule_ids,
                                               s['masses'], self.cell, np.random.default_rng(cfg.seed + 104729),
                                               position.device)
            self.barostat_work = 0.
            self.barostat_seconds = 0.
            self.barostat_reserve_growths = 0
            self.total_mass_amu = float(s['masses'].sum())
        self.driver = None
        if self.kind in NATIVE_ENGINES:
            from hyff.md.device.matrix_phases import NativeMatrixPhases
            self.driver = NativeMatrixPhases(self.engine, self.stepper, self.batch, cell_headroom=self._cell_headroom())
            self.driver.prepare()
        if self.barostat is not None:
            # Warm the move path (same cell, positions unchanged) so its first use is not timed in the run.
            self._apply_cell(self.cell)
            self.barostat.scaled_positions(self.stepper.position, np.eye(3))
        torch.cuda.synchronize()
        self.dt = float(cfg.timestep_fs)
        self.reserve = float(cfg.reserve)
        self.driver_base_steps = 0
        self.ndof = 3 * n - len(s['constraint_pairs']) - 3
        self.energy = energy0
        self.result.update(status='running', setup_seconds=time.perf_counter() - t0, n_atoms=n, ndof=self.ndof,
                           n_constraints=int(len(s['constraint_pairs'])), total_charge=float(s['total_charge']),
                           multiplicity=self.multiplicity,
                           engine_metadata={k: v for k, v in made['metadata'].items()
                                            if isinstance(v, (str, int, float, bool, list, dict))},
                           gpu_memory_allocated_GB=torch.cuda.memory_allocated() / 1e9)
        self.save()

    def kinetic(self):
        return float(.5 * (self.mass[:, None] * self.stepper.velocity.square()).sum() / ACCELERATION)

    def thermostat(self, steps):
        cfg = self.cfg
        k = self.kinetic()
        if not cfg.tau_ps:
            return k, 0.
        target = .5 * self.ndof * KB_EV_K * cfg.temperature_K
        alpha = bussi_factor(k, target, self.ndof, steps * self.dt / (cfg.tau_ps * 1000.), self.rng)
        with torch.no_grad():
            self.stepper.velocity.mul_(alpha)
        return k, (alpha * alpha - 1) * k

    def _cell_headroom(self):
        return CELL_HEADROOM if self.cfg.pressure_bar is not None else 1.0

    def volume(self):
        return abs(float(np.linalg.det(self.cell)))

    def _apply_cell(self, cell):
        """Adopt a new periodic cell wherever it is held (constant-pressure MD), between chunks."""
        from hyff.kernels.long_range.backends import set_cell as set_long_range_cell
        cell = engine_cell(cell)
        if self.driver is not None:
            self.driver.set_cell(cell)
        else:
            cell32 = torch.from_numpy(cell.astype(np.float32))
            static = getattr(self.engine, 'static_batch', None) or {}
            for target in (self.batch['cell'], static.get('cell')):
                if target is not None:
                    target.copy_(cell32.reshape(target.shape))
        set_long_range_cell(self.model, cell)
        if self.periodic_molecules is not None:
            self.periodic_molecules.set_cell(cell)
        self.cell = cell

    def _evaluate_here(self):
        """Energy and force at the integrator's positions between chunks; stepper.force is updated."""
        if self.driver is not None:
            return self.driver.evaluate_here()
        self.batch['positions'] = self.stepper.position.float()
        energy, force = self.engine(self.batch)
        with torch.no_grad():
            self.stepper.force.copy_(force)
        return energy.detach().clone()

    def barostat_move(self, energy):
        """One Monte Carlo volume move between chunks; returns the energy of the resulting state."""
        from hyff.md.device.matrix_phases import NeighborCapacityError
        from hyff.kernels.fused_messages import configure_neighbor_reserve
        b, started = self.barostat, time.perf_counter()
        old_cell, old_energy = self.cell, energy
        k, new_cell, M = b.propose(old_cell)
        old_volume, new_volume = abs(float(np.linalg.det(old_cell))), abs(float(np.linalg.det(new_cell)))
        old_position = self.stepper.position.detach().clone()
        old_force = self.stepper.force.detach().clone()
        for attempt in range(3):
            with torch.no_grad():
                self.stepper.position.copy_(b.scaled_positions(old_position, M))
            self._apply_cell(new_cell)
            try:
                trial = self._evaluate_here()
                break
            except NeighborCapacityError:
                # The compressed cell needs more neighbour slots: grow them at the old state and retry.
                with torch.no_grad():
                    self.stepper.position.copy_(old_position)
                self._apply_cell(old_cell)
                if attempt == 2:
                    raise
                self.reserve *= 1.25
                self.barostat_reserve_growths += 1
                configure_neighbor_reserve(self.model, self.reserve)
                self.rebuild_integrator(self.dt)
                old_force = self.stepper.force.detach().clone()
        dE = float(trial.sum()) - float(old_energy.sum())
        if b.decide(k, dE, old_volume, new_volume):
            self.barostat_work += dE
            energy = trial
        else:
            with torch.no_grad():
                self.stepper.position.copy_(old_position)
            self._apply_cell(old_cell)
            if self.driver is not None:
                self.driver.rebuild_here()
            with torch.no_grad():
                self.stepper.force.copy_(old_force)
        self.barostat_seconds += time.perf_counter() - started
        return energy

    def saved_positions(self):
        if self.periodic_molecules is None:
            return self.stepper.position
        return self.periodic_molecules.unwrapped(self.stepper.position)

    def advance(self, count):
        if self.periodic_molecules is not None and self.sim_fs >= self.next_recenter_fs - 1e-6:
            started = time.perf_counter()
            # Translate groups in place in the integrator state. The force is periodic, so the stored force
            # and energy stay valid and no graph is recaptured; the next step's skin check sees the moved
            # atoms and rebuilds the neighbour list at the new coordinates (on the device for presto,
            # vivace and moderato).
            event = self.periodic_molecules.recenter(self.stepper.position)
            if event['molecules_shifted']:
                # A later rollback must restore coordinates in the current image frame.
                self.snapshot = (self.stepper.position.detach().clone(), self.stepper.velocity.detach().clone(),
                                 self.sim_fs, self.steps_done)
                self.snapshot_cell = self.cell
            event.update(time_ps=self.sim_fs/1000., phase='committed_chunk_boundary',
                         wall_seconds=time.perf_counter()-started)
            self.result['recenter_events'].append(event)
            self.next_recenter_fs = self.sim_fs + self.cfg.recenter_interval_ps*1000
        if self.driver is not None:
            committed, energy, force = self.driver.advance(count)
            self.energy = energy
            return committed, energy
        def evaluate(current):
            self.batch['positions'] = current.float()
            return self.engine(self.batch)
        energy = None
        for _ in range(count):
            _, _, energy, force, _ = self.stepper.step(evaluate)
        self.energy = energy
        return count, energy

    def rebuild_integrator(self, dt):
        """New captured integrator/driver at the current (last valid) state; the force engine is reused."""
        from hyff.md.captured_rattle import CapturedRattleStep
        old = self.stepper
        if self.driver is not None:
            self.retired_native_neighbor_rebuilds += self.driver.rebuild_steps
            self.driver.close()
            self.driver = None
        position = old.position.detach().clone()
        velocity = old.velocity.detach().clone()
        self.engine.cache.reference = None
        self.engine.graph = None
        self.batch['positions'] = position.float()
        energy, force = self.engine(self.batch)
        self.stepper = CapturedRattleStep(position, velocity, force.detach(), self.mass, dt, old.constraints)
        if self.kind in NATIVE_ENGINES:
            from hyff.md.device.matrix_phases import NativeMatrixPhases
            self.driver = NativeMatrixPhases(self.engine, self.stepper, self.batch, cell_headroom=self._cell_headroom())
            self.driver.prepare()
        self.dt = float(dt)
        self.driver_base_steps = self.steps_done
        torch.cuda.synchronize()
        return energy

    def recover(self, error):
        """Recover an NVT rejected step after the driver restores its last valid state."""
        if self.cfg.tau_ps == 0 or self.cfg.recovery == 'off':
            self.result.update(status='failed', nve_interrupted=self.cfg.tau_ps == 0,
                               recovery_disabled=self.cfg.recovery == 'off')
            raise error
        from hyff.kernels.fused_messages import configure_neighbor_reserve
        cfg = self.cfg
        details = getattr(error, 'details', {}) or {}
        status = details.get('native_neighbor_status') or {}
        started = time.perf_counter()
        event = dict(sim_ps=self.sim_fs / 1000, dt_fs=self.dt, phase=details.get('phase'), flags=details.get('flags'))
        if details.get('phase') == 'neighbor_rebuild' and status.get('error_bits') == 1:
            self.reserve *= 1.25
            configure_neighbor_reserve(self.model, self.reserve)
            event.update(action='grow_neighbor_reserve', reserve=self.reserve)
            self.rebuild_integrator(self.dt)
        else:
            # Roll back to the last completed chunk and redraw Maxwell-Boltzmann velocities: the rejected step's
            # predecessor can already carry the runaway velocity that caused the failure.
            position, velocity, self.sim_fs, self.steps_done = self.snapshot
            s = self.stepper
            if self.barostat is not None and not np.array_equal(self.snapshot_cell, self.cell):
                self._apply_cell(self.snapshot_cell)
            with torch.no_grad():
                s.position.copy_(position)
                v = torch.randn(velocity.shape, generator=self.torch_rng, device=velocity.device, dtype=velocity.dtype)
                v *= torch.sqrt(KB_EV_K * cfg.temperature_K * ACCELERATION / self.mass)[:, None]
                v -= (v * self.mass[:, None]).sum(0) / self.mass.sum()
                s.velocity.copy_(v)
            s.constraints.project_velocities(s.position, s.velocity)
            event.update(action=f'rollback_to_{self.sim_fs / 1000:.3f}ps_redraw_velocities_dt_{cfg.fallback_timestep_fs}fs_for_{cfg.fallback_steps}_steps')
            self.rebuild_integrator(cfg.fallback_timestep_fs)
            self.fallback_remaining = cfg.fallback_steps
        event['recovery_seconds'] = time.perf_counter() - started
        self.result.setdefault('recoveries', []).append(event)
        self.save()
        stalled = sum(1 for e in self.result['recoveries'][-20:] if abs(e['sim_ps'] - event['sim_ps']) < 1e-9)
        if len(self.result['recoveries']) > cfg.max_recoveries or stalled >= 20:
            raise RuntimeError(f'Recovery limit reached ({len(self.result["recoveries"])} total)') from error

    def _failed_advance(self, error, frames, frame_times, wall_md):
        """Reporting hook for a failed advance; the multi-GPU run overrides it, the single-GPU run does nothing."""

    def run(self):
        self.run_started = time.perf_counter()
        from hyff.md.checked_phases import PhaseCheckError
        from hyff.engines.force_engine import CaptureEquivalenceError
        cfg = self.cfg
        self.sim_fs = 0.
        self.steps_done = 0
        # The starting state is the first frame (t = 0); then one frame every save_every steps.
        frames, frame_times = [self.saved_positions().float().cpu().numpy()], [0.]
        self.frame_boxes = [] if self.barostat is None else [self.cell.copy()]
        self.next_barostat_step = cfg.barostat_interval
        self.fallback_remaining = 0
        self.torch_rng = torch.Generator(device='cuda').manual_seed(cfg.seed + 7919)
        self.snapshot = (self.stepper.position.detach().clone(), self.stepper.velocity.detach().clone(), 0., 0)
        self.snapshot_cell = self.cell
        thermostat_work = 0.
        total_fs = cfg.steps * cfg.timestep_fs
        save_fs = cfg.save_every * cfg.timestep_fs
        next_save, next_flush = save_fs, save_fs * 20
        wall_md = 0.
        while self.sim_fs < total_fs - 1e-6:
            count = max(1, min(cfg.chunk, int(round((total_fs - self.sim_fs) / self.dt))))
            # End the chunk at the next frame, so frames are exactly save_every steps apart.
            count = min(count, max(1, int(round((next_save - self.sim_fs) / self.dt))))
            if self.fallback_remaining:
                count = min(count, self.fallback_remaining)
            if self.barostat is not None:
                count = min(count, max(1, self.next_barostat_step - self.steps_done))
            t = time.perf_counter()
            try:
                committed, energy = self.advance(count)
            except CaptureEquivalenceError as error:
                # Includes CaptureGrossDeploymentError: neither capture failure
                # is an NVT phase recovery or a mid-run coordinate-jitter retry.
                self.result.update(status='failed', nve_interrupted=cfg.tau_ps == 0,
                                   capture_check=self._capture_check(error))
                try:
                    self.flush(frames, frame_times, wall_md)
                except Exception as report_error:
                    self.result['failure_report_error_type'] = type(report_error).__name__
                raise
            except PhaseCheckError as error:
                torch.cuda.synchronize()
                wall_md += time.perf_counter() - t
                total = int((getattr(error, 'details', {}) or {}).get('committed_steps', 0))
                committed = max(0, total - (self.steps_done - self.driver_base_steps))
                self.steps_done += committed
                self.sim_fs += committed * self.dt
                if cfg.tau_ps == 0 or cfg.recovery == 'off':
                    self.result.update(status='failed', nve_interrupted=cfg.tau_ps == 0,
                                       recovery_disabled=cfg.recovery == 'off')
                    # Reporting must not replace the original phase failure.
                    try:
                        self.flush(frames, frame_times, wall_md)
                    except Exception as report_error:
                        self.result['failure_report_error_type'] = type(report_error).__name__
                    raise
                self.recover(error)
                continue
            except BaseException as error:
                self._failed_advance(error, frames, frame_times, wall_md)
                raise
            torch.cuda.synchronize()
            wall_md += time.perf_counter() - t
            self.steps_done += committed
            self.sim_fs += committed * self.dt
            k_before, work = self.thermostat(committed)
            thermostat_work += work
            entry = {}
            if self.barostat is not None:
                if self.steps_done >= self.next_barostat_step:
                    energy = self.energy = self.barostat_move(energy)
                    self.next_barostat_step = self.steps_done + cfg.barostat_interval
                volume = self.volume()
                entry = dict(volume_A3=volume, density_g_cm3=self.total_mass_amu * AMU_A3_G_CM3 / volume)
            epot = float(energy.sum())
            temp = 2 * k_before / (self.ndof * KB_EV_K)
            if not math.isfinite(epot) or not math.isfinite(temp):
                raise FloatingPointError(f'Nonfinite state at {self.sim_fs / 1000:.3f} ps')
            barostat_work = self.barostat_work if self.barostat is not None else 0.
            self.result['trace'].append(dict(step=self.steps_done, time_ps=self.sim_fs / 1000, dt_fs=self.dt, epot_eV=epot,
                                             T_K=temp, conserved_eV=epot + k_before - thermostat_work + work - barostat_work,
                                             **entry))
            if self.fallback_remaining:
                self.fallback_remaining = max(0, self.fallback_remaining - committed)
                if not self.fallback_remaining:
                    self.rebuild_integrator(cfg.timestep_fs)
            self.snapshot = (self.stepper.position.detach().clone(), self.stepper.velocity.detach().clone(),
                             self.sim_fs, self.steps_done)
            self.snapshot_cell = self.cell
            if self.sim_fs >= next_save - 1e-6:
                frames.append(self.saved_positions().float().cpu().numpy())
                frame_times.append(self.sim_fs / 1000)
                if self.barostat is not None:
                    self.frame_boxes.append(self.cell.copy())
                next_save += save_fs
                # The chunk, thermostat and finite checks completed above.
                # Checkpoint only at the frame-save cadence, outside
                # the integration timer; keep bulk trajectory flushing below.
                self._write_final_state()
                self.result.update(steps_done=self.steps_done, simulated_ps=self.sim_fs / 1000)
                self.save()
            if self.sim_fs >= next_flush - 1e-6 or self.sim_fs >= total_fs - 1e-6:
                self.flush(frames, frame_times, wall_md)
                next_flush += save_fs * 20
            if cfg.max_wall_seconds and time.perf_counter() - self.run_started > cfg.max_wall_seconds:
                if self.sim_fs < total_fs - 1e-6:
                    self.result['stop_reason'] = 'MAX_WALL_SECONDS'
                break
        self.flush(frames, frame_times, wall_md, final=True)
        return self.public_result()

    def _write_final_state(self):
        """Atomically replace the periodic/final committed restart, preserving Q/M.

        Call only at an accepted chunk boundary or after an explicitly verified
        rollback. The file format and mass-adjusted velocity convention are the
        same at periodic save points and at normal completion.
        """
        started = time.perf_counter()
        periodic_state = {}
        if self.periodic_molecules is not None:
            periodic_state = dict(bounded_positions_A=self.stepper.position.cpu().numpy(),
                molecule_images=self.periodic_molecules.images.cpu().numpy(),
                molecule_ids=self.periodic_molecules.ids.cpu().numpy(),
                periodic_cell_A=self.periodic_molecules.cell.cpu().numpy())
        temporary = self.out / 'final_state.pending.npz'
        np.savez(temporary, positions_A=self.saved_positions().cpu().numpy(),
            velocities_A_fs=(self.stepper.velocity * torch.sqrt(self.mass / self.base_mass)[:, None]).cpu().numpy(),
            time_ps=self.sim_fs / 1000, total_charge=float(self.sys['total_charge']),
            multiplicity=self.multiplicity, committed_steps=self.steps_done,
            n_atoms=len(self.sys['atomic_numbers']),
            atomic_numbers_sha256=atomic_numbers_sha256(self.sys['atomic_numbers']), **periodic_state,
            **({} if self.cell is None else dict(box_A=self.cell)))
        temporary.replace(self.out / 'final_state.npz')
        duration = time.perf_counter() - started
        if not hasattr(self, '_checkpoint_write_seconds'):
            self._checkpoint_write_seconds = []
        self._checkpoint_write_seconds.append(duration)
        self.result['checkpoint_write_timing'] = dict(clock='time.perf_counter',
            n=len(self._checkpoint_write_seconds), mean_seconds=statistics.mean(self._checkpoint_write_seconds),
            median_seconds=statistics.median(self._checkpoint_write_seconds),
            scope='SUCCESSFUL_CHECKPOINT_HOST_COPIES_SERIALIZATION_AND_ATOMIC_REPLACE_EXCLUDES_RESULT_JSON')
        self.result['restart_checkpoint'] = dict(file='final_state.npz',
            committed_steps=self.steps_done, time_ps=self.sim_fs / 1000,
            frames_flushed_through_ps=self.result.get('frames_flushed_through_ps'),
            save_every_steps=self.cfg.save_every,
            cadence='FRAME_SAVE_POINTS_AND_FINAL_OR_VERIFIED_RESTORE',
            state_source='LAST_COMPLETED_CHUNK_OR_VERIFIED_RESTORED_STATE')

    def flush(self, frames, frame_times, wall_md, final=False):
        """Write pending frames and update the timings.

        ``ms_per_step``/``ns_per_day`` cover the integration (device launches and
        synchronization). ``run_wall_seconds`` also includes thermostat,
        snapshots, output and recovery; both exclude setup. Neighbour rebuild
        counts include rejected trajectory attempts, not setup warmups.
        """
        cfg = self.cfg
        if frames:
            tag = f'{int(round(frame_times[0] * 1000)):012d}_{int(round(frame_times[-1] * 1000)):012d}'
            np.save(self.out / f'frames_{tag}.npy', np.stack(frames))
            np.save(self.out / f'times_ps_{tag}.npy', np.array(frame_times))
            if self.barostat is not None:
                np.save(self.out / f'boxes_A_{tag}.npy', np.stack(self.frame_boxes[-len(frames):]))
                self.frame_boxes.clear()
            self.result['frames_flushed_through_ps'] = float(frame_times[-1])
            frames.clear()
            frame_times.clear()
        ps = self.sim_fs / 1000
        if self.barostat is not None:
            self.result['barostat'].update(self.barostat.description(), seconds=self.barostat_seconds,
                                           neighbor_reserve_growths=self.barostat_reserve_growths,
                                           engine_captures=getattr(self.engine, 'captures', None),
                                           final_box_A=self.cell.tolist(), final_volume_A3=self.volume())
        if self.periodic_molecules is not None:
            self.result['max_abs_force_coordinate_A'] = float(self.stepper.position.abs().max())
            self.result['max_abs_unwrapped_coordinate_A'] = float(self.saved_positions().abs().max())
        # Failure reporting alone does not prove that an in-flight native
        # state is restartable. Distributed terminal reporting may write only
        # after its chunk restoration check succeeds.
        restored = self.result.get('distributed_failure', {}).get('chunk_state_restored') is True
        if final and (self.result.get('status') != 'failed' or restored):
            self._write_final_state()
        run_wall = time.perf_counter() - self.run_started if self.run_started is not None else None
        integration_ms = wall_md * 1000 / max(self.steps_done, 1)
        integration_rate = ps * 1e-3 / (wall_md / 86400) if wall_md else None
        native_rebuilds = self.retired_native_neighbor_rebuilds + getattr(self.driver, 'rebuild_steps', 0)
        cache_rebuilds = getattr(self.engine.cache, 'rebuilds', 0)
        status = ('complete' if final and self.sim_fs >= cfg.steps * cfg.timestep_fs - 1e-6
                  else ('stopped' if final else 'running'))
        if self.result.get('status') == 'failed':
            status = 'failed'
        self.result.update(steps_done=self.steps_done, simulated_ps=ps,
                           md_wall_seconds=wall_md, ms_per_step=integration_ms, ns_per_day=integration_rate,
                           run_wall_seconds=run_wall,
                           run_ms_per_step=run_wall * 1000 / max(self.steps_done, 1) if run_wall is not None else None,
                           run_ns_per_day=ps * 1e-3 / (run_wall / 86400) if run_wall else None,
                           native_neighbor_rebuilds=native_rebuilds, neighbor_cache_rebuilds=cache_rebuilds,
                           neighbor_rebuilds=native_rebuilds + cache_rebuilds, status=status)
        self.save()

    def close(self):
        if self.driver is not None:
            self.driver.close()
        if self.engine is not None:
            self.engine.close()
