# largo

largo is the largest HyFF model (1.20 B parameters, 34 M active per atom). The released weights
are the exponential-moving-average weights after 720,000 optimizer steps (the final model). They
are distributed as three safetensors shards plus their index (4.81 GB) and are downloaded into
the cache directory on first use, like the other models:

```python
model = hyff.load("largo")
sim = hyff.Simulation("system.npz", model="largo", output="run/")
```

```bash
hyff download largo          # optional; otherwise the first use downloads it
hyff md --model largo --system system.npz --output run/
```

A checkpoint stored elsewhere can be passed as `weights=` / `--weights`: the shard index
(`weights.safetensors.index.json`, with `weights.json` and the shards next to it) or the directory
that holds it. HyFF identifies largo weights from the architecture JSON and uses the largo engine
automatically.

## Defaults

The largo engine captures the force evaluation in CUDA graphs with tf32x3 products, fp64 LayerNorm
statistics, an edge buffer reserve of 1.05, the automatic long-range route, the automatic capacity
policy and shared expert storage. These are the fastest validated settings (125 ms/step on 2,586
atoms on one H20). Kernels are compiled at startup, which takes a few minutes; IEEE fp32 matrix
products and Triton kernels without floating-point contraction are set for the whole process.

- `capacity_policy='auto'` (default; also for andante): see [Memory of large systems](#memory-of-large-systems).
- `expert_storage_alias=True` (default for the largo engine): the frozen expert weights and the packed
  copies the grouped-expert kernels read share one storage instead of two. Same values and kernels:
  forces differ from the unshared run by no more than the engine's own run-to-run variation (2.5e-6 eV,
  9.5e-7 eV/Å on 2,586 atoms), the speed is unchanged, and peak memory falls by 4.7 GB (see the table
  below). Pass `expert_storage_alias=False` to keep separate copies.

Use `create_engine` or `Simulation` for systems beyond a few hundred atoms. The eager model
(`hyff.load("largo").energy_forces`) keeps the whole autograd graph: energy and forces of 918 atoms
peak at 40 GB of GPU memory, and 2,586 atoms do not fit on a 96 GB H20. The engine evaluates the same
2,586 atoms with a peak of 12.8 GB.

## Latent-charge term

largo adds a learned electrostatic term: per-atom charges from the network, conserved to the total
charge, interact through a Gaussian-screened Coulomb kernel erf(r/(√2 σ))/r with a learned energy
scale. In the released weights the scale is −4.66 × 10⁻⁸ (times 14.4 eV Å) and σ = 72.9 Å, so the
term contributes little:

| system | latent-charge energy | largest latent-charge force |
|---|---:|---:|
| 306 waters (918 atoms) | 1.4 × 10⁻⁸ eV | 2.0 × 10⁻⁷ eV/Å |
| chorismate, ions and water (2,586 atoms) | 3.5 × 10⁻⁴ eV | – |

The periodic and isolated sums differ by 5.7 × 10⁻⁶ eV (forces 5.5 × 10⁻⁶ eV/Å) on the 2,586-atom
system. With σ much larger than the box, the Gaussian charge clouds overlap all their images; the
periodic sum then reduces to the charges' self energy (the reciprocal mesh sum is below 10⁻⁴⁵ and is
omitted).

## Long range

| `long_range` | Evaluation |
|---|---|
| `auto` (default) | periodic system (a cell is given, including every MD run): `spme`; isolated system: `dense` up to 4,096 atoms, `tree` above |
| `spme` | periodic sum over all images in the cell (smooth-particle-mesh Ewald, tin-foil boundary); invariant to lattice translations of molecules |
| `dense` | isolated sum, exact fused all-pairs evaluation |
| `tree` | isolated sum, treecode approximation for large systems |
| `native` | isolated sum, the model's plain PyTorch evaluation |

The isolated sums change when a molecule is translated by a lattice vector and have no cell, so
periodic recentering (`recenter_interval_ps`) and NPT (`pressure_bar`) require `spme`; under NPT the
mesh keeps its size and follows the cell. The SPME mesh has a spacing of at most σ/2 and at least as
many points per lattice vector as the B-spline order (6). On the 2,586-atom system MD takes 122.5
ms/step with SPME and 123.4 ms/step with `dense` (median of three 500-step runs). `conservation='fp64'` evaluates the
charge conservation in fp64 (charges stay fp32). `hyff.load("largo")` uses the same rule: SPME when
a cell is passed, the isolated sum otherwise.

Settings can be changed through environment variables read when an engine is built: `LQ_DENSE_MAX`
(4096), `LQ_SPME_H_SIGMA` (0.5, mesh spacing in units of σ), `LQ_SPME_ORDER` (6), `LQ_SPME_DTYPE`
(`fp32` or `fp64`) and the tree settings `LQ_ORDER`, `LQ_LEAF`, `LQ_THETA`, `LQ_DELTA`,
`LQ_RCUT_SIGMA`, `LQ_HEADROOM`. Invalid values are reported before any compilation.

## Memory of large systems

`capacity_policy` bounds the memory of the edge-frame-mixing layers. The default `auto` keeps the
untiled route through 20,388 atoms (andante: 50,000), so smaller systems run exactly as without it;
largo then tiles part of the layers through 51,720 atoms (`partial`) and all of them above (`tiled`),
recomputing tiles in the backward pass instead of storing them. `native` forces the untiled route.

Measured on one H20 (96 GB), NVE (500 steps at 2,586 atoms, 20 at 103,440), peak memory from PyTorch's allocator:

| system | capacity route | expert storage | ms/step | peak allocated (GB) | peak reserved (GB) |
|---|---|---|---:|---:|---:|
| 2,586 atoms | untiled (auto) | separate | 121.9 | 17.5 | 18.9 |
| 2,586 atoms | untiled (auto) | shared (default) | 121.9 | 12.8 | 18.8 |
| 103,440 atoms | tiled (auto) | separate | 6,337 | 73.7 | 81.4 |
| 103,440 atoms | tiled (auto) | shared (default) | 6,305 | 68.9 | 78.2 |

At 103,440 atoms the untiled route (`capacity_policy='native'`) runs out of memory while the first
force graph is captured.

## Multi-GPU MD (experimental)

```python
hyff.Simulation("system.npz", model="largo", gpus=8, tau_ps=0, output="run/")
```

or `hyff md --gpus 8 ...` on one node. `gpus=N` sets the most GPUs the run may use; the automatic
policy picks the count:

- NVE runs (`tau_ps=0`, 1 fs, chunk 100, default skin and reserve, no HMR) of at least 103,440 atoms
  on H20-class GPUs: 8 GPUs if available, otherwise 4;
- everything else: one GPU (multi-GPU MD is slower for small systems; at 2,586 atoms two GPUs took
  301 ms/step against 121 ms/step on one).

Measured at 103,440 atoms (1 fs NVE, chunk 100, defaults): 6,347 ms/step on one GPU, 2,207 on four and 1,173 on eight. Above that
size the choice is a forecast. When several GPUs are selected, `hyff.Simulation` restarts the run
under `torchrun` in child processes and reads their `result.json`. The multi-GPU service runs in
fp32 with the isolated dense long-range sum (which differs from the single-GPU SPME default by about
10⁻⁵ eV); recentering, SPME and capacity policies are not supported, and failures are terminal (no
NVT recovery).
An explicit count is possible with `route='multigpu', ranks=N` under `torchrun`
(`python -m torch.distributed.run --standalone --nproc-per-node N --module hyff.multigpu.launcher --config-json ...`).

Environment switches read at startup (all default on): `HYFF_MULTIGPU_COMPILED_EFM=0` (no compiled
edge-frame-mixing tiles; also turns off both compaction options), `HYFF_MULTIGPU_COMPACT_EFM_EDGES=0` (do not skip
inactive edge-frame-mixing edges), `HYFF_MULTIGPU_COMPACT_BASE_PAIR=0` (do not skip inactive base/pair rows),
`HYFF_CAPACITY_RETENTION=0` (do not reuse edge intermediates; lower memory). The automatic GPU-count rule applies
only with the default options. Compiler caches go to `$XDG_CACHE_HOME` or `~/.cache/hyff/compile-cache`
per rank unless `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` are set.
