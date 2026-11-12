"""ASE calculator for the HyFF models.

    from ase.build import molecule
    from hyff.calculator import HyFFCalculator

    atoms = molecule("H2O")
    atoms.calc = HyFFCalculator("presto")
    energy, forces = atoms.get_potential_energy(), atoms.get_forces()

The calculator returns the energy (eV) and forces (eV/A) of isolated (``pbc=False``) or fully periodic
(``pbc=True``) systems. The total charge and the spin multiplicity 2S+1 are read from
``atoms.info["charge"]`` and ``atoms.info["multiplicity"]`` (or ``atoms.info["spin"]``, the OMol25
convention for the multiplicity); without them the calculator's ``charge`` and ``multiplicity`` are used.
Stress is not available.

``engine="fast"`` builds the model's compiled CUDA engine for one system (fixed atoms, charge,
multiplicity and periodicity) and reuses it while positions or the cell change, as in molecular dynamics
and geometry optimization. A process can build at most two compiled engines, so a calculator that visits
many different systems should use ``engine="eager"``, which evaluates the plain PyTorch model on any
sequence of structures on the CPU or a GPU. ``engine="auto"`` (default) is ``fast`` on CUDA and ``eager``
on the CPU.
"""
from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from hyff.api import _charge, _multiplicity, _structure

ENGINES = ('auto', 'fast', 'eager')
EAGER_OPTIONS = ('skin', 'long_range')


class HyFFCalculator(Calculator):
    """ASE calculator: ``HyFFCalculator(model, weights=None, charge=0, multiplicity=1, engine='auto', device='cuda')``.

    ``model`` is a model name (presto, vivace, moderato, andante, largo); ``weights`` an explicit checkpoint
    such as fine-tuned weights. Further keyword options are those of :func:`hyff.create_engine` for the fast
    engine (precision, statistics, skin, reserve, edge_reserve, long_range, conservation, capacity_policy,
    expert_storage_alias) and ``skin`` and ``long_range`` for the eager engine.
    """

    implemented_properties = ['energy', 'free_energy', 'forces']

    def __init__(self, model=None, *, weights=None, charge=0, multiplicity=1, engine='auto', device='cuda',
                 **options):
        super().__init__()
        if engine not in ENGINES:
            raise ValueError(f'engine must be one of {", ".join(ENGINES)}, got {engine!r}')
        self.device = str(device)
        cuda = self.device.startswith('cuda')
        if engine == 'auto':
            engine = 'fast' if cuda else 'eager'
        if engine == 'fast' and not cuda:
            raise ValueError("engine='fast' needs a CUDA device; use engine='eager' on the CPU")
        if model is None and weights is None:
            model = 'presto'
        self.model_name, self.weights = model, weights
        self.default_charge = _charge(charge)
        self.default_multiplicity = _multiplicity(multiplicity)
        self.engine = engine
        self.options = dict(options)
        self._fast = self._fast_key = self._fast_cell = None
        self._eager = None
        if engine == 'eager':
            unknown = sorted(set(self.options) - set(EAGER_OPTIONS))
            if unknown:
                raise ValueError(f"Option {unknown[0]!r} applies to engine='fast'; the eager engine takes "
                                 f"{', '.join(EAGER_OPTIONS)}")
            import hyff
            self._eager = hyff.load(model, weights=weights, device=self.device, **self.options)

    def system_state(self, atoms):
        """(charge, multiplicity, periodic) of ``atoms``, from ``atoms.info`` or the calculator defaults."""
        info = atoms.info
        if 'multiplicity' in info and 'spin' in info and info['multiplicity'] != info['spin']:
            raise ValueError('atoms.info has different "multiplicity" and "spin"; give the multiplicity 2S+1 once')
        charge = _charge(info.get('charge', self.default_charge))
        multiplicity = _multiplicity(info.get('multiplicity', info.get('spin', self.default_multiplicity)))
        pbc = np.asarray(atoms.pbc, dtype=bool)
        if pbc.any() and not pbc.all():
            raise ValueError('HyFF evaluates isolated (pbc=False) or fully periodic (pbc=True) systems; '
                             'for a slab use pbc=True with vacuum along the surface normal')
        return charge, multiplicity, bool(pbc.all())

    def check_state(self, atoms, tol=1e-15):
        changes = super().check_state(atoms, tol=tol)
        # ASE compares positions, numbers, cell and pbc; a new charge or multiplicity must also count.
        if self.atoms is not None and self.system_state(atoms)[:2] != self.system_state(self.atoms)[:2]:
            changes.append('charge_or_multiplicity')
        return changes

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        atoms = self.atoms
        charge, multiplicity, periodic = self.system_state(atoms)
        positions, numbers, cell, charge, multiplicity = _structure(
            atoms.positions, atoms.numbers, atoms.cell.array if periodic else None, charge, multiplicity)
        if self.engine == 'eager':
            energy, forces = self._eager.energy_forces(positions, numbers, cell, charge, multiplicity)
        else:
            energy, forces = self._evaluate_fast(positions, numbers, cell, charge, multiplicity)
        self.results = dict(energy=energy, free_energy=energy, forces=np.asarray(forces, dtype=np.float64))

    def _evaluate_fast(self, positions, numbers, cell, charge, multiplicity):
        key = (numbers.tobytes(), charge, multiplicity, cell is not None)
        if self._fast is None or key != self._fast_key:
            self._build_fast(positions, numbers, cell, charge, multiplicity)
            self._fast_key = key
        elif cell is not None and not np.array_equal(cell, self._fast_cell):
            self._set_cell(cell)
        energy, forces = self._fast(positions)
        return float(energy.double().sum()), forces.detach().double().cpu().numpy()

    def _build_fast(self, positions, numbers, cell, charge, multiplicity):
        import hyff
        from hyff.engines.compiler_budget import CompiledEngineProcessLimitError
        self.close()
        try:
            self._fast = hyff.create_engine(self.model_name, positions, numbers, cell, charge=charge,
                                            multiplicity=multiplicity, weights=self.weights, **self.options)
        except CompiledEngineProcessLimitError as error:
            raise RuntimeError("A process can build at most two compiled HyFF engines, one per system (atoms, "
                               "charge, multiplicity, periodicity). To evaluate many different systems, use "
                               "HyFFCalculator(..., engine='eager').") from error
        self._fast_cell = cell

    def _set_cell(self, cell):
        """A new periodic cell for the same system, set in place (the engine reads the cell at run time)."""
        import torch
        from hyff.kernels.long_range.backends import lr_kind, set_cell
        made = self._fast._made
        values = torch.from_numpy(np.asarray(cell, dtype=np.float32))
        static = getattr(made['engine'], 'static_batch', None) or {}
        for target in (made['batch']['cell'], static.get('cell')):
            if target is not None:
                target.copy_(values.reshape(target.shape))
        if lr_kind(made['model']) == 'spme':
            set_cell(made['model'], cell)
        self._fast_cell = cell

    def close(self):
        """Release the compiled engine (it is also released when the calculator is garbage collected)."""
        if self._fast is not None:
            self._fast.close()
            self._fast = self._fast_key = self._fast_cell = None
