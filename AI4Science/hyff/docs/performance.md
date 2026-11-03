# Performance

## Default settings

Every model runs with its fastest validated settings without extra options:

| Model | Engine | Products | LayerNorm statistics | Other |
|---|---|---|---|---|
| presto | native fused kernels | tf32x3 | fp32 | |
| vivace | native fused kernels | tf32x3 | fp32 | |
| moderato | native fused kernels | tf32x3 | fp32 | |
| andante | captured CUDA graphs | fp32 | fp64 | edge reserve 1.05, capacity policy auto |
| largo | captured CUDA graphs | tf32x3 | fp64 | edge reserve 1.05, long range auto (periodic: SPME; isolated: dense up to 4,096 atoms, tree above), capacity policy auto, shared expert storage |

`tf32x3` evaluates fp32 matrix products as three TF32 tensor-core products (error comparable to
fp32). `fp32` statistics compute the LayerNorm mean and variance in fp32 at force time; `fp64`
computes them in fp64. Learned weights are never cast. Override with `precision=`, `statistics=`
(`--precision`, `--statistics`); `engine='reference'` runs the plain PyTorch model.

## Measured MD speed

2,586-atom solvated system (chorismate in water), one NVIDIA H20, 1 fs NVE with bond constraints,
mean over 500 timed steps after 100 warm-up steps, including neighbour-list rebuilds:

| Model | ms/step | ns/day |
|---|---:|---:|
| presto | 0.71 | 121 |
| vivace | 1.27 | 68 |
| moderato | 4.04 | 21 |
| andante | 29.5 | 2.9 |
| largo | 125 | 0.69 |

## Energy conservation

NVE on 306 waters (918 atoms, periodic, 300 K start), conserved-energy drift from a linear fit and the
range of the conserved energy over the run (presto):

| Water model | Timestep | Length | Drift (meV/atom/ns) | Range (meV/atom) |
|---|---:|---:|---:|---:|
| rigid (constraints) | 1 fs | 2 ps | +0.4 | 0.006 |
| flexible (no constraints) | 1 fs | 2 ps | −26.8 | 0.095 |
| flexible | 0.5 fs | 20 ps | −2.8 | 0.084 |
| flexible | 0.25 fs | 20 ps | −0.7 | 0.024 |

The flexible drift falls with the square of the timestep, as expected for velocity Verlet; moderato,
andante and largo show the same behaviour at 0.5 fs (−4 to −7 meV/atom/ns over 1–2 ps). Rigid water at
1 fs is the default; flexible water needs 0.5 fs or less.

## Constant pressure

Cost of the Monte Carlo barostat on the 2,586-atom system: NPT at 1 bar and 300 K against NVT, both
continuing from the same equilibrated NPT state, one volume move every 100 steps, one H20:

| Model | One move (ms) | In steps | Run time vs NVT |
|---|---:|---:|---:|
| presto | 2.3 | 3.6 | +3.3% |
| vivace | 3.1 | 2.6 | +2.7% |
| moderato | 7.4 | 1.9 | +1.3% |
| andante | 36 | 1.2 | +1.0% |
| largo | 141 | 1.1 | +1.1% |

The MD step itself is unchanged. A move costs one force evaluation, one or two neighbour-list
rebuilds and some host work, so its share falls for larger systems and models.

Checks: an in-place cell change gives the energy and forces of an engine built for the new cell
(|ΔE| ≤ 1 × 10⁻⁵ eV, |ΔF| ≤ 2 × 10⁻⁶ eV/Å for all five models; for largo also on a 20,688-atom box
whose SPME mesh sum is computed). The pressure from energy differences under molecular scaling, a
route independent of the barostat, averages +24 ± 80 bar over a 1 bar presto NPT run and
−1,507 ± 54 bar in NVT at the starting density (1.006 g/cm³), from which NPT compresses the box. For
groups without interactions the sampled volume follows the ideal-gas NPT distribution in all three
coupling modes (unit test).

## Tips

- Use one Python process per simulation; each process may construct at most two compiled engines.
- Kernel compilation happens at the first force call and is cached on disk (Triton and Inductor caches),
  so a second run on the same system starts faster.
- `chunk` (steps per device launch, default 100) amortizes launch overhead for the native engines;
  larger values reduce thermostat and output frequency.
- A larger `skin` reduces neighbour-list rebuilds at the cost of more neighbours per atom.
- For many small molecules use `hyff.predict_batch` rather than one engine per molecule.
- Multi-GPU MD only pays off for very large largo systems: at 103,440 atoms (1 fs NVE, chunk 100) one H20 takes
  6,347 ms/step, four 2,207 and eight 1,173; at 2,586 atoms two GPUs (301 ms/step) are slower than one (121).
  See [largo.md](largo.md).
