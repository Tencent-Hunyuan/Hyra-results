"""Public Python API: models, engines, molecular dynamics and batched inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from hyff.registry import resolve


def _structure(positions, numbers, cell=None, charge=0, multiplicity=1):
    """Checked single-structure inputs: positions (N, 3), numbers (N,), cell (3, 3) or None, charge, 2S+1."""
    if hasattr(positions, 'detach'):
        positions = positions.detach().cpu().numpy()
    if hasattr(numbers, 'detach'):
        numbers = numbers.detach().cpu().numpy()
    x, z = np.asarray(positions), np.asarray(numbers)
    if x.ndim != 2 or x.shape[1:] != (3,) or not len(x) or x.dtype.kind not in 'fiu' or not np.isfinite(x).all():
        raise ValueError(f'positions must be a finite (N, 3) array in Angstrom, got shape {x.shape}')
    if z.shape != (len(x),) or z.dtype.kind not in 'iu':
        raise ValueError(f'numbers must be N = {len(x)} integer atomic numbers, got shape {z.shape} '
                         f'and dtype {z.dtype}')
    if cell is not None:
        if hasattr(cell, 'detach'):
            cell = cell.detach().cpu().numpy()
        c = np.asarray(cell)
        if c.shape != (3, 3) or c.dtype.kind not in 'fiu' or not np.isfinite(c).all():
            raise ValueError(f'cell must be a finite (3, 3) array (rows are lattice vectors) or None, '
                             f'got shape {c.shape}')
        if abs(float(np.linalg.det(c.astype(np.float64)))) <= 0:
            raise ValueError('cell has zero volume; pass a nonsingular (3, 3) cell or None for an isolated system')
        cell = c.astype(np.float64)
    return x.astype(np.float64), z.astype(np.int64), cell, _charge(charge), _multiplicity(multiplicity)


def _charge(value):
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in 'fiu' or not np.isfinite(array):
        raise ValueError(f'charge must be a finite number, got {array.tolist()!r}')
    return float(array)


def _multiplicity(value):
    """Spin multiplicity 2S+1 as an int; a non-integer value is an error, never truncated."""
    array = np.asarray(value)
    shown = array.tolist() if array.dtype.kind in 'biufU' else value
    if array.shape != () or array.dtype.kind not in 'fiu' or isinstance(value, bool):
        raise ValueError(f'multiplicity must be an integer 2S+1 >= 1, got {shown!r}')
    number = float(array)
    if not np.isfinite(number) or number != int(number) or number < 1:
        raise ValueError(f'multiplicity must be an integer 2S+1 >= 1, got {shown!r}')
    return int(number)


class Model:
    """A HyFF model with its weights, evaluated eagerly (CPU or CUDA).

    Use :func:`hyff.load` to create one. ``energy_forces`` is the simple
    single-structure call; :meth:`create_engine` builds the fast engine for
    repeated evaluations of one system (CUDA only).
    """

    def __init__(self, selection, device='cuda', skin=1.5):
        from hyff.engines.dispatch import Potential
        self.name = selection.name
        self.weights = selection.weights
        self.sha256 = selection.sha256
        self.engine = selection.engine
        self.options = dict(selection.options)
        self.device = str(device)
        self._potential = Potential(self.weights, device=self.device, skin=skin,
                                    long_range=self.options.get('long_range', 'auto'))

    @property
    def cutoff(self):
        return float(self._potential.model.cutoff)

    def energy_forces(self, positions, numbers, cell=None, charge=0, multiplicity=1):
        """Energy (eV) and forces (eV/A) of one structure.

        ``positions`` is (N, 3) in Angstrom, ``numbers`` the atomic numbers,
        ``cell`` an optional (3, 3) periodic box in Angstrom (rows are lattice
        vectors), ``charge`` the total charge and ``multiplicity`` 2S+1.
        """
        energy, forces = self._potential(*_structure(positions, numbers, cell, charge, multiplicity))
        return float(energy.sum()), forces.cpu().numpy()

    def create_engine(self, positions, numbers, cell=None, **options):
        """Fast fixed-topology engine for this model and one system; see :func:`hyff.create_engine`."""
        return create_engine(self.name or self.weights, positions, numbers, cell,
                             weights=self.weights, **options)

    def __repr__(self):
        return f'Model(name={self.name!r}, engine={self.engine!r}, device={self.device!r})'


def load(model=None, *, weights=None, device='cuda', skin=1.5, long_range=None):
    """Load a model by name (presto, vivace, moderato, andante, largo) or from a checkpoint.

    A named model's weights are downloaded into the cache directory on first
    use (see ``hyff models``). ``weights`` selects an explicit checkpoint: a
    ``.safetensors`` file with its sibling ``.json``, a sharded
    ``.safetensors.index.json`` or a directory holding one. A path may also be
    given as ``model``. Without a model name the model is identified from
    ``weights``; with neither, presto is loaded. ``long_range`` (largo only):
    'auto' (default) evaluates the latent-charge term as the periodic SPME sum
    when a cell is given and as the isolated sum otherwise; 'native' always
    uses the isolated sum.
    """
    if model is None and weights is None:
        model = 'presto'
    return Model(resolve(model, weights, long_range=long_range), device=device, skin=skin)


class Engine:
    """Energy/force engine for one system with a fixed atom list (CUDA).

    ``engine(positions)`` returns ``(energy, forces)`` as CUDA tensors and
    reuses captured graphs and neighbour lists between calls; use
    ``energy_forces`` for Python/NumPy outputs. Close it (or use it as a
    context manager) to release the captured graphs.
    """

    def __init__(self, made, selection):
        self._made = made
        self.model = selection.name
        self.engine = selection.engine
        self.options = dict(selection.options)
        self.sha256 = selection.sha256
        self.metadata = dict(model=self.model, engine=self.engine, checkpoint_sha256=self.sha256,
                             **self.options)

    @property
    def batch(self):
        return self._made['batch']

    def __call__(self, positions):
        import torch
        batch = self._made['batch']
        batch['positions'] = torch.as_tensor(positions, dtype=torch.float32, device=batch['positions'].device)
        return self._made['engine'](batch)

    def energy_forces(self, positions):
        energy, forces = self(positions)
        return float(energy.sum()), forces.detach().cpu().numpy()

    def close(self):
        engine = self._made.get('engine')
        if engine is not None and hasattr(engine, 'close'):
            engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def create_engine(model, positions, numbers, cell=None, *, charge=0, multiplicity=1, weights=None,
                  engine=None, precision=None, statistics=None, skin=1.5, reserve=1.5,
                  edge_reserve=None, long_range=None, conservation=None, capacity_policy=None,
                  expert_storage_alias=None, capture_branch_policy='first', debug=False):
    """Create the fast energy/force engine of a model for one system.

    Without options every model uses its fastest validated settings: native
    engines for presto, vivace and moderato, captured engines for andante and
    largo (edge reserve 1.05; largo with tf32x3 products, fp64 statistics and
    long range: SPME with a cell, dense/tree without). ``engine='reference'``
    selects the plain PyTorch model. Positions are in Angstrom; ``cell`` is a
    (3, 3) box or None.
    """
    from hyff.engines.dispatch import create_engine as build
    positions, numbers, cell, charge, multiplicity = _structure(positions, numbers, cell, charge, multiplicity)
    selection = resolve(model, weights, engine=engine, precision=precision, statistics=statistics,
                        edge_reserve=edge_reserve, long_range=long_range, conservation=conservation,
                        capacity_policy=capacity_policy)
    options = selection.options
    made = build(selection.engine, selection.weights, positions.astype(np.float32), numbers, cell,
                 total_charge=charge, multiplicity=multiplicity, skin=skin, reserve=reserve,
                 precision=options['precision'], statistics=options['statistics'],
                 edge_reserve=options.get('edge_reserve'), expert_storage_alias=expert_storage_alias,
                 capacity_policy=options.get('capacity_policy'), long_range=options.get('long_range'),
                 conservation=options.get('conservation'), capture_branch_policy=capture_branch_policy)
    result = Engine(made, selection)
    if debug:
        result.metadata['debug'] = made.get('metadata', {})
    return result


class Simulation:
    """Constrained NVE/NVT/NPT molecular dynamics on a prepared system NPZ.

    ``tau_ps=0`` runs NVE; ``pressure_bar`` adds a Monte Carlo barostat
    (NPT). Output (``result.json``, ``final_state.npz``, trajectory frames)
    goes to ``output``. ``gpus>1`` enables experimental multi-GPU largo MD
    with the automatic card policy. Other keyword options are those of
    :class:`hyff.md.simulation.RunConfig`.
    """

    def __init__(self, system, model=None, *, steps=10000, timestep_fs=1.0, temperature_K=300.0,
                 tau_ps=0.5, output='hyff_md', weights=None, gpus=1, **options):
        import dataclasses
        from hyff.md.simulation import RunConfig
        if model is None and weights is None:
            model = 'presto'
        known = {f.name for f in dataclasses.fields(RunConfig)} - {'system', 'extra'}
        unknown = sorted(set(options) - known)
        if unknown:
            raise ValueError(f'Unknown Simulation option {unknown[0]!r}; options are: {", ".join(sorted(known))}')
        self.config = RunConfig(system=str(system), model=model, weights=None if weights is None else str(weights),
                                steps=steps, timestep_fs=timestep_fs, temperature_K=temperature_K,
                                tau_ps=tau_ps, output=str(output), gpus=gpus, **options)
        self._run = None
        self.result = None

    @property
    def output(self):
        return Path(self.config.output)

    def run(self):
        """Run the simulation and return the result dictionary (also written to result.json)."""
        cfg = self.config
        if cfg.route != 'native':
            self.result = self._run_multigpu()
            return self.result
        from hyff.md.simulation import MDRun
        self._run = run = MDRun(cfg)
        try:
            run.setup()
            self.result = run.run()
        except BaseException as error:
            try:
                run.result.update(status='failed', error_type=type(error).__name__)
                run.save()
            except BaseException:
                pass
            raise
        finally:
            run.close()
        return self.result

    def _run_multigpu(self):
        # The automatic card policy runs first in this process; when it selects
        # several GPUs, the run continues under torchrun in child processes.
        import json
        import subprocess
        from hyff.multigpu.launcher import run_md
        result = run_md(self.config)
        if result.get('status') != 'launch_required':
            return result
        subprocess.run(result['argv'], check=True)
        return json.loads((self.output / 'result.json').read_text())

    def close(self):
        if self._run is not None:
            self._run.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def predict_batch(model, positions, numbers, offsets, charge=None, multiplicity=None, *, weights=None,
                  device='cuda', **options):
    """Energies and forces of many nonperiodic molecules in one flat collection.

    ``positions`` (N, 3) and ``numbers`` (N,) concatenate all molecules;
    ``offsets`` (M+1,) marks their boundaries; ``charge`` and ``multiplicity``
    are per molecule (default 0 and 1). Returns a dict with ``energies`` (M,)
    in eV and ``forces`` (N, 3) in eV/A. Options: ``route`` (native, reference),
    ``mode`` (energy, energy_forces), capacities and ``precision``/``statistics``.
    """
    import inspect
    from hyff.batch.collection import CollectionPotential, predict_collection
    if 'cell' in options:
        raise ValueError('predict_batch evaluates isolated (nonperiodic) molecules; for a periodic system use '
                         'hyff.load(...).energy_forces or hyff.create_engine with cell=')
    allowed = set(inspect.signature(CollectionPotential.__init__).parameters) - {'self', 'model', 'tier', 'device'}
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(f'Unknown predict_batch option {unknown[0]!r}; options are: {", ".join(sorted(allowed))}')
    selection = resolve(model, weights)
    if selection.name is None:
        raise ValueError('Batched inference needs one of the named model architectures')
    count = len(np.asarray(offsets)) - 1
    charge = np.zeros(count, np.float32) if charge is None else charge
    multiplicity = np.ones(count, np.int64) if multiplicity is None else multiplicity
    result = predict_collection(selection.weights, positions, numbers, offsets, charge, multiplicity,
                                tier=selection.name, device=device, **options)
    result['metadata'] = dict(result['metadata'], model=selection.name, checkpoint_sha256=selection.sha256)
    return result
