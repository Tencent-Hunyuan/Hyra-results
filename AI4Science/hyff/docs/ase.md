# ASE calculator

```python
from ase.build import molecule
from ase.optimize import BFGS
from hyff.calculator import HyFFCalculator

atoms = molecule("CH3OH")
atoms.calc = HyFFCalculator("presto")
BFGS(atoms).run(fmax=0.01)
print(atoms.get_potential_energy(), atoms.get_forces())
```

The calculator needs ASE (`pip install ase`). It provides `energy` (eV), `free_energy` (equal to the
energy) and `forces` (eV/Å) for any of the five models or for explicit weights such as a fine-tuned
model: `HyFFCalculator(weights="ft/best")`.

## Charge, spin and periodicity

- The total charge and the spin multiplicity 2S+1 come from `atoms.info["charge"]` and
  `atoms.info["multiplicity"]`; `atoms.info["spin"]` is read as the multiplicity, as in OMol25.
  Without them the calculator's `charge` and `multiplicity` arguments (defaults 0 and 1) apply.
  Changing them recomputes the energy even when the atoms do not move.
- `pbc=False` evaluates an isolated system, `pbc=True` a periodic one in `atoms.cell`. Mixed
  periodicity is refused; model a slab with `pbc=True` and vacuum along the surface normal.

## Engines

| `engine` | Device | Use |
|---|---|---|
| `fast` | CUDA | repeated evaluations of one system (MD, geometry optimization); the model's compiled engine |
| `eager` | CPU or CUDA | any sequence of structures, e.g. single points of many molecules |
| `auto` (default) | | `fast` on CUDA, `eager` on the CPU |

The fast engine is built at the first evaluation; compilation takes from seconds (presto) to minutes
(largo) and is cached on disk. It is kept while positions and the cell change; scaling the cell, as for
an equation of state, updates it in place. A different atom list, charge, multiplicity or periodicity
builds a new engine, and a process can build at most two compiled engines, so use `engine="eager"` to
evaluate many different systems in one process. Options of `hyff.create_engine` (`precision`,
`statistics`, `skin`, `long_range`, ...) pass through to the fast engine; the eager engine takes `skin`
and `long_range`.

On the 918-atom periodic water box the fast and eager engines agree to 8 × 10⁻⁶ eV/Å in the forces for
all five models, also after an in-place 2% cell scaling.

## MD with ASE or with hyff.Simulation

ASE integrators (`VelocityVerlet`, `Langevin`, ...) work with the calculator, but every step copies
positions and forces between GPU and CPU and integrates in Python. On the 2,586-atom system (one H20,
ASE `VelocityVerlet`, fast engine) this adds about 0.7 ms per step:

| Model | ASE MD (ms/step) | `hyff.Simulation` (ms/step) |
|---|---:|---:|
| presto | 1.42 | 0.71 |
| moderato | 4.78 | 4.04 |
| andante | 29.9 | 29.5 |

The ASE runs were flexible (no constraints) at 0.5 fs. `hyff.Simulation` holds X–H bonds and water
rigid with RATTLE on the GPU and runs at 1 fs; ASE's `FixBondLengths` constraint can do the same on the
CPU at additional cost per step. Use the calculator for geometry optimization, single points and
ASE-based workflows, and `hyff.Simulation` for production MD.

## Limits

No stress or virial: ASE variable-cell relaxations (for example `FrechetCellFilter`) and ASE barostats
need the stress. For constant-pressure MD use `hyff.Simulation(..., pressure_bar=1.0)` (see
[md.md](md.md)), which also keeps the whole integration on the GPU.
