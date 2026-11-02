# HyFF

This directory contains Hyra's HyFF force fields: five machine-learning interatomic potentials,
from presto (2.3 M parameters) to largo (1.2 B), and the `hyff` Python package that runs them for
single-structure inference, molecular dynamics and batched prediction. All models cover 83 elements
(H to Bi), take the total charge and spin multiplicity as inputs, and are trained on OMol25.

Technical report: [paper/HyFF_Technical_Report.pdf](paper/HyFF_Technical_Report.pdf).

## Recorded result

| Model | Parameters | Active per atom | ms/step | ns/day | Force MAE (meV/Å) | Energy MAE (meV/atom) | Download |
|---|---:|---:|---:|---:|---:|---:|---:|
| presto | 2.31 M | 0.05 M | 0.71 | 121 | 47.7 | 5.78 | 9.3 MB |
| vivace | 6.66 M | 0.16 M | 1.27 | 68 | 33.5 | 4.36 | 26.7 MB |
| moderato | 34.8 M | 0.83 M | 4.04 | 21 | 16.5 | 1.36 | 139 MB |
| andante | 175 M | 5.0 M | 29.5 | 2.9 | 7.65 | 0.620 | 700 MB |
| largo | 1.20 B | 34 M | 125 | 0.69 | 5.79 | 0.364 | 4.81 GB |

Speed: molecular dynamics of a 2,586-atom solvated system (chorismate in water) on one NVIDIA H20,
1 fs NVE with bond constraints, including neighbour-list rebuilds. Accuracy: the full OMol25
validation set (2,762,021 structures). Compared with other models in the same setting, andante
reaches a force MAE of 7.65 meV/Å against 7.55 for UMA-S-1.2 at 29× its speed, and largo has the
lowest energy error, 0.364 meV/atom against 0.380 for DPA4-Pro, at 9.7× its speed.

Hardware and software: all results were recorded on NVIDIA H20 GPUs (sm_90) with Python 3.12,
torch 2.7.1 (CUDA 12.x) and triton 3.3.1 (`constraints.txt`). Other GPUs are untested.

## Method

All five models are smooth, conservative message-passing potentials: atoms carry scalar, vector and
rank-2 Cartesian tensor features (l ≤ 2), messages are switched off smoothly at per-layer cutoffs,
per-element expert networks update the atom features, and forces are exact negative gradients of the
energy. presto, vivace and moderato use this base model at increasing width and depth (2 to 4
interactions, 4.5 to 8 Å cutoff). andante and largo add edge-frame-mixing blocks, a Cartesian
analogue of SO(2) edge-frame convolutions; largo also adds an attention-based embedding of the total
charge and spin and a learned latent-charge electrostatic term ([docs/largo.md](docs/largo.md)).

The package evaluates the models with fused Triton kernels captured in CUDA graphs and integrates
molecular dynamics on the GPU in FP64 with RATTLE bond constraints. Every model runs with its
fastest validated settings by default; see [docs/performance.md](docs/performance.md).

## Run

Linux with an NVIDIA GPU and CUDA for engines and MD; CPU works for single-structure inference.
Python 3.11 or newer.

```bash
pip install "git+https://github.com/Tencent-Hunyuan/Hyra-results.git#subdirectory=AI4Science/hyff" \
    -c https://raw.githubusercontent.com/Tencent-Hunyuan/Hyra-results/main/AI4Science/hyff/constraints.txt
hyff models                           # sizes, and whether each model is installed or downloaded on first use
hyff download presto                  # optional: fetch weights ahead of time (or: hyff download all)
```

Model weights are assets of the `hyff-v0.2.0` release of this repository and are not part of the
package. On first use of a model (`hyff.load`, `hyff.Simulation`, `hyff.predict_batch`, the CLI)
HyFF downloads its weights into `$HYFF_HOME/checkpoints` (default `~/.cache/hyff/checkpoints`),
resuming interrupted downloads and checking every file against the sha256 in the package's
checkpoint index. Installed weights are used without network access. `HYFF_MIRROR=<URL or directory>`
downloads the same file names from another location. Any model can also be loaded from an explicit
`weights=` checkpoint: a `.safetensors` file with its sibling `.json`, a sharded
`.safetensors.index.json` with its shards, or a directory holding one. The first MD step on a new
system compiles kernels; this takes from seconds (presto) to minutes (largo).

## Inference

```python
import numpy as np
import hyff

model = hyff.load("presto", device="cpu")          # or device="cuda"
positions = np.array([[0., 0., 0.], [0.9572, 0., 0.], [-0.24, 0.927, 0.]])   # Å
energy, forces = model.energy_forces(positions, [8, 1, 1], cell=None, charge=0, multiplicity=1)
```

Energies are in eV and forces in eV/Å. For repeated evaluations of one system on a GPU use an engine:

```python
with hyff.create_engine("andante", positions, numbers, cell) as engine:
    energy, forces = engine.energy_forces(new_positions)
```

Command line: `hyff infer --model presto --system input.npz --output out.npz`.

## Molecular dynamics

```python
sim = hyff.Simulation("system.npz", model="presto", steps=10000, timestep_fs=1.0,
                      temperature_K=300, tau_ps=0.5, output="run/")
result = sim.run()             # also usable as a context manager
print(result["ms_per_step"], result["ns_per_day"])
```

```bash
hyff md --model presto --system system.npz --output run/ --steps 10000 --tau-ps 0.5
```

`tau_ps=0` runs NVE. The output directory receives `result.json` (timings, energy trace),
`final_state.npz` (restart) and trajectory frames.
MD runs NVE, NVT (Bussi thermostat) or NPT (Monte Carlo barostat, `pressure_bar=1.0`) in a periodic
cell, with bond constraints or flexible (no constraints, 0.5 fs). See [docs/md.md](docs/md.md) for
thermostat, barostat, constraints, hydrogen mass repartitioning, restarts, periodic recentering and
what is not supported (stress output).

## Batched inference

```python
result = hyff.predict_batch("vivace", positions, numbers, offsets, charge, multiplicity)
result["energies"], result["forces"]
```

`positions` and `numbers` concatenate all molecules and `offsets` (M+1) marks their boundaries.
CLI: `hyff predict --model vivace --system collection.npz --output predictions.npz`.
See [docs/batch.md](docs/batch.md).

## Input format

| Key | Shape | Meaning |
|---|---|---|
| `positions_A` | (N, 3) | Cartesian positions in Å |
| `atomic_numbers` | (N,) | atomic numbers |
| `box_A` | (3, 3) | periodic cell in Å (rows are lattice vectors); optional for inference |
| `total_charge`, `multiplicity` | scalars | defaults 0 and 1 |

MD additionally needs velocities, masses, a box and the bond constraints (possibly none); see
[docs/inputs.md](docs/inputs.md). Inputs are checked before any kernel is compiled.
`docs/examples/make_tiny_system.py` writes a small example of every input file.

## Documentation

[inputs](docs/inputs.md) · [molecular dynamics](docs/md.md) · [batched inference](docs/batch.md) ·
[largo](docs/largo.md) · [performance](docs/performance.md)

## Citation

See [CITATION.cff](CITATION.cff): Hyra Team, "HyFF: Atomic Foundation Models on the Speed–Accuracy
Frontier", technical report, 2026 ([PDF](paper/HyFF_Technical_Report.pdf)).

## License

Apache-2.0, see the repository LICENSE.
