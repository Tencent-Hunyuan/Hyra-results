# Batched inference

```python
import hyff
result = hyff.predict_batch("moderato", positions, numbers, offsets, charge, multiplicity)
energies, forces = result["energies"], result["forces"]     # (M,) eV and (N, 3) eV/Å
```

`hyff predict --model moderato --system collection.npz --output predictions.npz` does the same
from a collection file (see [inputs.md](inputs.md)). Molecules are isolated (nonperiodic) and the
results keep the input order.

## How it works

Molecules are packed into fixed-capacity batches (atoms, molecules, edges, neighbours per atom);
capacities are planned from the geometry of the whole collection, and every batch shape is
compiled and captured once and then replayed. Use one call for a large collection rather than
many small calls: setup and capture dominate short jobs, and the singleton `Model.energy_forces`
can be faster below about a thousand molecules.

The batched engines use the same numerics as the single-system engines: tf32x3 products with
fp32 LayerNorm statistics for presto, vivace and moderato, fp32 with fp64 statistics for andante,
tf32x3 with fp64 statistics for largo. largo evaluates the latent-charge long-range term over all
atom pairs within each molecule.

## Options

| Option | Default | Meaning |
|---|---|---|
| `route` | `native` | `reference` evaluates the plain PyTorch model (IEEE fp32), also on CPU |
| `mode` | `energy_forces` | `energy` skips forces |
| `graph_capacity` | 512 (largo 64) | molecules per batch (`--batch-size`) |
| `atom_capacity` | 8192 (largo 2048) | atoms per batch |
| `edge_capacity`, `neighbor_capacity` | planned | override the planned capacities (neighbour capacity: multiple of 8) |
| `bucket_policy` | `single` | `size_bands` captures separate batches for molecules of <50, <100, <200 and more atoms |
| `device` | `cuda` | the native route requires CUDA |

A collection whose batches exceed a capacity raises `CollectionCapacityOverflowError` naming the
molecules of the failing batch; rerun with larger capacities. A predictor keeps at most four
capacity shapes; create a new one for very different collections. andante needs a GPU with at
least 80 GB for typical batches.
