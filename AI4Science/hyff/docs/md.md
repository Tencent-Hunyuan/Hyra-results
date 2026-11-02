# Molecular dynamics

```python
import hyff
sim = hyff.Simulation("system.npz", model="moderato", steps=50000, timestep_fs=1.0,
                      temperature_K=300, tau_ps=0.5, output="run/")
result = sim.run()
```

or `hyff md --model moderato --system system.npz --output run/ --steps 50000`.

All options and the system file are checked before any kernel is compiled; an invalid value stops
with a one-line message (the CLI prints a traceback only with `--debug`).

## Integrator and engines

Velocity Verlet with RATTLE bond constraints in FP64; forces come from the model's own engine in FP32.
presto, vivace and moderato run whole chunks of steps on the GPU, including neighbour-list rebuilds,
with CUDA conditional graphs; andante and largo replay captured force graphs per step.
`chunk` (default 100, at most 1,024) is the number of steps per device launch.

The first chunk compiles kernels for the system. Each process can construct at most two compiled
engines; use a fresh Python process per simulation.

## Thermostat

`tau_ps > 0` applies the Bussi–Donadio–Parrinello stochastic velocity rescaling thermostat at
`temperature_K` (> 0) between chunks; `tau_ps=0` runs NVE. `seed` sets the thermostat random stream.
The trace in `result.json` reports the potential energy, temperature and the conserved
energy (including the thermostat work) at every chunk.

If an NVT step is rejected by the integrity checks, the run rolls back to the last chunk, redraws
velocities and continues for `fallback_steps` with `fallback_timestep_fs`; `recovery='off'` /
`--no-recovery` stops at the first rejected step instead. NVE runs always stop.

## Constant pressure (NPT)

```python
hyff.Simulation("system.npz", model="moderato", steps=200000, temperature_K=300, tau_ps=0.5,
                pressure_bar=1.0, output="npt/")
```

or `hyff md ... --pressure-bar 1`. A Monte Carlo barostat changes the cell between chunks. Every
`barostat_interval` steps (default 100) it proposes a random volume change, scales the cell and the
mass-weighted centres of the bonded groups, evaluates the energy once and accepts the move with the
Metropolis criterion of the isothermal–isobaric ensemble at `temperature_K` and `pressure_bar`.
Groups keep their internal geometry, so constraints stay exact, and velocities are unchanged. The
largest volume change adapts towards an acceptance rate between 25% and 75%. NPT needs the
thermostat (`tau_ps > 0`) and runs on one GPU.

| `barostat` | Moves |
|---|---|
| `isotropic` (default) | the whole cell; any box shape |
| `semi-isotropic` | x and y together, or z (membranes); orthorhombic box |
| `anisotropic` | x, y or z; orthorhombic box |

The groups are those of periodic recentering: connected components of `constraint_pairs` plus
`bond_pairs`, or of the bonds and constraints of `molecule_system_xml`. Groups split across the box
boundary in the input are made whole by lattice translations at the start (the energy is unchanged).
For flexible molecules give `bond_pairs` (or the topology) so that molecules are scaled as wholes;
without them every unconstrained atom is scaled on its own, which is equally correct but accepts
smaller volume changes (34% instead of 50% acceptance for flexible water).

A move costs one force evaluation and up to two neighbour-list rebuilds, 1–3% of the run time at the
default interval (see [performance.md](performance.md#constant-pressure)). andante and largo rebuild
their captured graphs (tens of seconds) when compression outgrows their edge buffers, typically once
or twice while the density settles from a low starting value. largo needs the periodic SPME sum (the
default for MD systems).

The trace adds `volume_A3` and `density_g_cm3` at every chunk, and the conserved energy also
subtracts the energy change of accepted moves. `boxes_A_*.npy` holds the cell of every saved frame,
`result.json` reports the barostat under `barostat` (moves, acceptances, largest volume change,
final box) and `final_state.npz` holds `box_A`; a restart continues in that cell.

## Constraints and flexible MD

`constraint_pairs` / `constraint_distances_A` of the system file are held fixed (typically all X–H
bonds and rigid water); use 1 fs with ordinary masses. With no constraints (`constraint_pairs` of
shape (0, 2)) the run is flexible: plain velocity Verlet with every bond free. O–H stretches then
need a short timestep, 0.5 fs or less. Measured NVE on 306 flexible waters (presto, 0.5 fs, 20 ps):
see [performance.md](performance.md#energy-conservation). The timestep must be positive and at most 10 fs.

## Hydrogen mass repartitioning

- `hmr_mass=3.0` (`--hmr-mass 3`): every non-water hydrogen gets the given mass; its bonded heavy
  atom loses the difference.
- `inclusive_hmr=True` (`--inclusive-hmr`): every hydrogen, including water hydrogens, becomes 3 amu;
  heavy atoms compensate and must stay at or above 3 amu.

The heavy partner of each hydrogen comes from `constraint_pairs`, so HMR needs the X–H constraints.
Water is taken from `water_idx` when present, otherwise every constraint cluster of exactly one O and
two H is water. Initial velocities are rescaled for equipartition at the new masses. HMR does not
change the timestep.

## Restart

```python
hyff.Simulation("system.npz", model="moderato", initial_positions="run/final_state.npz", output="run2/")
```

(`--restart run/final_state.npz`). Positions, velocities and the cell (`box_A`, which changes under
NPT) continue; step counters, time and the thermostat random stream start anew. `final_state.npz`
is rewritten at every frame save (`save_every`, default 500 steps) and at the end.

## Periodic recentering

`recenter_interval_ps` (`--recenter-interval-ps`) translates bonded groups back into the cell at
chunk boundaries. The translation is applied in place: the force is periodic, so nothing is
recomputed or recaptured, and the next step's neighbour-list check rebuilds the list at the new
coordinates (well under a millisecond per event). The saved trajectory and `final_state.npz` stay
unwrapped.

The groups are the connected components of `constraint_pairs` plus the optional `bond_pairs`
(B, 2) of the system file, or the bonds and constraints of an OpenMM `system.xml` given as
`molecule_system_xml` / `--molecule-topology`. Constrained atoms always move together; a molecule
whose heavy-atom bonds are not listed is translated in fragments, which leaves the energy unchanged.

largo supports recentering with the periodic SPME long-range sum, the default for MD systems
(see [largo.md](largo.md)); the isolated sums (`dense`, `tree`, `native`) are refused.

## Output

`result.json`: status, model, engine, checkpoint sha256, engine options, `ms_per_step`, `ns_per_day`,
wall times, atom/constraint counts, charge and multiplicity, neighbour rebuilds and the energy trace.
Frames (`frames_*.npy`, `times_ps_*.npy`): the starting state at t = 0, then one frame exactly every
`save_every` steps (chunks end at frame times). Pass `debug=True` (`--debug`) to also store engine
diagnostics under `debug`.

## Options

| Option | Default | Meaning |
|---|---|---|
| `engine` | model's own | `reference` runs the plain PyTorch model (slow; for checks) |
| `precision`, `statistics` | model default | see [performance.md](performance.md) |
| `long_range` | `auto` | largo only: `spme` for MD systems (periodic); see [largo.md](largo.md) |
| `skin`, `reserve` | 1.5, 1.5 | neighbour-list skin (Å) and buffer reserve |
| `edge_reserve` | 1.05 | andante/largo edge buffer reserve |
| `capacity_policy` | `auto` | andante/largo memory route for large systems; see [largo.md](largo.md#memory-of-large-systems) |
| `expert_storage_alias` | on for largo | largo: frozen expert weights share storage with the kernel buffers (less memory) |
| `chunk` | 100 | steps per device launch, 1–1,024 |
| `save_every` | 500 | steps between saved frames |
| `max_wall_seconds` | none | stop cleanly after this wall time (`--max-wall-seconds`) |
| `recenter_interval_ps`, `molecule_system_xml` | none | periodic recentering (above) |
| `pressure_bar` | none | NPT at this pressure (above); none keeps the cell fixed |
| `barostat`, `barostat_interval` | `isotropic`, 100 | coupled cell axes and steps between volume moves |

## Not supported

- stress or virial output (the Monte Carlo barostat needs neither);
- NPT on several GPUs;
- other thermostats (Langevin, Nosé–Hoover) and other integrators;
- isolated (nonperiodic) MD: the system needs a box; use a box larger than the molecule plus the model cutoff;
- constraint clusters of more than four atoms or three constraints (e.g. full-bond constraints of a ring).
