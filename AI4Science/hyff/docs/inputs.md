# Inputs

All files are NumPy `.npz` archives. Units: Å, Å/ps (initial velocities), amu, eV.
`docs/examples/make_tiny_system.py --output DIR` writes one example of each kind.

## Single structure (`hyff infer`, `Model.energy_forces`)

| Key | Shape / dtype | Meaning |
|---|---|---|
| `positions_A` | (N, 3) float | positions in Å |
| `atomic_numbers` | (N,) int | atomic numbers 1–83 |
| `box_A` | (3, 3) float, optional | periodic cell in Å, rows are lattice vectors, nonzero volume; omit for an isolated system |
| `total_charge` | scalar, optional | default 0 |
| `multiplicity` | integer scalar, optional | spin multiplicity 2S+1, default 1; a non-integer value is an error |

`hyff infer` writes `energy_eV` and `forces_eV_A` (N, 3).

## Prepared MD system (`hyff md`, `hyff.Simulation`)

The single-structure keys plus:

| Key | Shape / dtype | Meaning |
|---|---|---|
| `velocities_A_ps` | (N, 3) float64 | initial velocities in Å/ps (used as given; not drawn from the temperature) |
| `masses` | (N,) float64 | original atomic masses in amu (HMR is applied by the run, not in the file) |
| `box_A` | (3, 3) float64 | required; periodic cell with positive volume (rows are lattice vectors; the starting cell under NPT) |
| `constraint_pairs` | (C, 2) int64 | zero-based atom pairs with fixed distance (X–H bonds, rigid water); (0, 2) for flexible MD |
| `constraint_distances_A` | (C,) float64 | target distance of each pair in Å |
| `total_charge` | scalar | required for MD |
| `multiplicity` | integer scalar, optional | default 1; use an integer (e.g. `np.int64(2)`), not a float |
| `water_idx` | (W, 3) int64, optional | water atoms (O, H, H) for hydrogen mass repartitioning; default: every constraint cluster of one O and two H |
| `bond_pairs` | (B, 2) int64, optional | further bonds (e.g. heavy-atom bonds) that join atoms into groups for periodic recentering and the NPT barostat |

All arrays use the same atom order. Positions should satisfy the constraint distances and
velocities should be tangent to the constraints; the run projects both once at the start.
Missing keys, wrong shapes, atomic numbers outside 1–83 and similar problems are reported by name
before any kernel is compiled.
Building a system (topology, constraints, solvation) from a PDB file is outside this package.

A run directory contains `result.json`, `final_state.npz` and `frames_*.npy` / `times_ps_*.npy`.
`final_state.npz` holds `positions_A`, `velocities_A_fs` (Å/fs) and `box_A` plus the charge,
multiplicity, atom count and an ordered-species digest; pass it as `--restart` / `initial_positions=`
to continue. NPT runs also write `boxes_A_*.npy`, the cell of every saved frame.
An output directory that already contains `result.json` or `final_state.npz` is refused.

## Molecule collection (`hyff predict`, `hyff.predict_batch`)

| Key | Shape / dtype | Meaning |
|---|---|---|
| `positions` | (N, 3) float | all molecules concatenated, Å |
| `numbers` | (N,) int | atomic numbers |
| `offsets` | (M+1,) int64 | molecule boundaries, starting at 0 and ending at N |
| `total_charge` | (M,) float | charge per molecule |
| `multiplicity` | (M,) int64 | multiplicity per molecule |

Collections are isolated (nonperiodic) molecules; results keep the input order.
`hyff predict` writes `energies_eV` (M,), `forces_eV_A` (N, 3) and `offsets`.

## Weights

A checkpoint is a `.safetensors` file with a sibling `.json` of the same name that describes the
architecture (`format`, `model_id`, `model_config`, `weights_sha256`), or a sharded checkpoint in the
Hugging Face layout: `weights.safetensors.index.json` (`weight_map` from tensor name to shard file,
`metadata.total_size`, and the sha256 of every shard) next to its shards and `weights.json`. A directory
holding one checkpoint may be passed instead of the file.

The package ships `checkpoints/index.json`: for every model its files, sizes, sha256 and release asset
names, and the release location: the `hyff-v0.2.0` release of `Tencent-Hunyuan/Hyra-results`, with URLs
`https://github.com/{repo}/releases/download/{tag}/{file}` and asset names starting with `hyff-`
(e.g. `hyff-presto-weights.safetensors`). Weights are
downloaded into `$HYFF_HOME/checkpoints` (default `~/.cache/hyff/checkpoints`) on first use or with
`hyff download <model>|all`:

- a download goes to `<file>.part` and resumes with an HTTP Range request after an interruption;
- the size and sha256 are checked before the file is renamed into place; a file that fails the check
  is kept as `<file>.corrupt` and refused;
- `HYFF_MIRROR=<base URL or directory>` replaces the release base URL; file names stay the same;
- installed weights (in the cache, or in `checkpoints/` of a source checkout) are verified against the
  index and used without network access.
