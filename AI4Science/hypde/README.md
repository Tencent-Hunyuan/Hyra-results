# HyPDE

This directory contains Hyra's HyPDE, a 323-million-parameter 2D foundation model for
partial-differential-equation simulations, and the `hypde` Python package that runs it. Given the last six frames
of a simulated field (velocity, pressure, density, concentration, stress components, ...) on a regular 2D grid,
HyPDE predicts the next four frames; applied to its own output it forecasts trajectories of any length. One set of
weights covers thirteen physical systems, from acoustics and compressible flow to active matter, reaction-diffusion 
and shallow water on the sphere.

## Recorded result

Zero-shot forecasting error on the test split of thirteen 2D datasets of The Well and FlowBench (1,874
trajectories), compared with Walrus, a 1.29-billion-parameter cross-domain foundation model for continuum dynamics
([McCabe et al., 2025](https://arxiv.org/abs/2511.15684)), run with its released weights on the same trajectories.
Neither model is fine-tuned. The
metric is VRMSE (root mean squared error normalised by the spatial variance of the reference, mean over fields),
median over trajectories, in the three readings of Walrus's Table 1: **one-step**, the error of a single forecast
from true frames averaged over every start time; **T[1:20]** and **T[21:60]**, the mean error over frames 1-20 and
21-60 of one autoregressive forecast from frame 17. Each cell is HyPDE / Walrus (ratio); lower is better, and the
better model is in bold.

| Dataset | Traj. | One-step | T[1:20] | T[21:60] |
|---|---:|---:|---:|---:|
| Acoustic scattering, discontinuous | 200 | **0.00342** / 0.00578 (0.59) | **0.0164** / 0.0240 (0.68) | **0.0558** / 0.155 (0.36) |
| Acoustic scattering, inclusions | 400 | **0.0111** / 0.0164 (0.68) | **0.0501** / 0.0636 (0.79) | **0.132** / 0.169 (0.78) |
| Acoustic scattering, maze | 200 | **0.0146** / 0.0443 (0.33) | **0.0556** / 0.163 (0.34) | **0.0894** / 0.255 (0.35) |
| Active matter | 26 | 0.0251 / **0.0132** (1.91) | 0.460 / **0.242** (1.90) | 1.41 / **1.32** (1.07) |
| Euler multi-quadrants, open | 500 | **0.0200** / 0.0238 (0.84) | **0.0881** / 0.106 (0.83) | **0.368** / 0.442 (0.83) |
| Euler multi-quadrants, periodic | 32 | **0.0293** / 0.0396 (0.74) | **0.101** / 0.147 (0.69) | **0.397** / 0.611 (0.65) |
| FlowBench FPO harmonics | 40 | **0.00121** / 0.00141 (0.86) | **0.0157** / 0.0201 (0.78) | **0.0618** / 0.0907 (0.68) |
| Gray-Scott reaction-diffusion | 120 | 0.000326 / **0.000311** (1.05) | **0.0196** / 0.0397 (0.50) | **0.0874** / 0.211 (0.41) |
| Helmholtz staircase | 48 | **0.000352** / 0.00122 (0.29) | 0.0101 / **0.00965** (1.05) | 0.0230 / **0.0208** (1.10) |
| PlanetSWE | 12 | 0.00121 / **0.000570** (2.12) | 0.0230 / **0.00715** (3.21) | 0.155 / **0.0384** (4.04) |
| Rayleigh-Bénard | 175 | **0.0147** / 0.0164 (0.90) | **0.0281** / 0.102 (0.28) | 0.709 / **0.686** (1.03) |
| Shear flow | 112 | **0.00142** / 0.00152 (0.94) | **0.0143** / 0.0176 (0.81) | 0.0708 / **0.0683** (1.04) |
| Turbulent radiative layer | 9 | 0.176 / **0.115** (1.53) | **0.485** / 0.522 (0.93) | **0.825** / 1.05 (0.79) |
| **HyPDE better on** | | **9 of 13** | **10 of 13** | **8 of 13** |

## Method

A call takes the last six frames, normalises each field of the window by its root mean square over the window
(not centred), and predicts the next four frames as increments over the last input frame, in units of the
window's root-mean-square frame-to-frame change. 

Each frame is embedded by a per-field-type convolution (kernel 8, stride 4; every field and constant input has its
own weights, so datasets with different fields share one model), merged into patches of 4, 8 or 16 cells per axis
so that a frame has 1,024 to 2,048 tokens, and the six frames of each patch are concatenated and projected to one
token. A 24-layer transformer (width 1,024, 16 heads, SwiGLU, a depthwise 3x3 convolution on the token grid before
each MLP) attends over all tokens together with conditioning tokens for the dataset, the boundary conditions, the
fields and the grid. Positions enter through axial rotary embeddings that are exactly periodic on periodic axes,
and boundaries through learned embeddings near non-periodic edges; there is no absolute position. A head maps each
token back to four frames at full resolution, and a small convolutional refiner, which also sees the last input
frame, adds sub-patch detail. Inputs are not centred, so the operations that see them at full precision (patch
embedding and refiner) run with TF32 disabled.

The model was trained from scratch for 27,280 steps (9.15 × 10^10 tokens) on the training splits of the datasets, 
predicting the next four frames from true history windows, on 32 NVIDIA H20 GPUs.

## Install

Python 3.10 or newer, torch 2.4 or newer. A CUDA GPU is recommended; CPU inference works and is much slower.
The weights (1.3 GB, FP32) are released as an asset of this repository's GitHub Releases:

```bash
git clone https://github.com/Tencent-Hunyuan/Hyra-results.git && cd Hyra-results/AI4Science/hypde
bash weights/download.sh      # downloads weights/model.safetensors and checks its sha256
pip install -e .
```

`hypde.load()` reads `weights/` next to the package; `hypde.load(weights=...)` loads a directory holding
`model.safetensors` and `config.json` from any path, and `hypde info --verify` checks the file's sha256.

## Forecast

```python
import hypde

model = hypde.load()                                   # CUDA when available, otherwise CPU
traj = hypde.load_trajectory("shear_flow_Reynolds_5e4_Schmidt_1e0.hdf5", index=0)   # a file of The Well

next4 = model.predict(traj.fields[11:17], traj.info, traj.constants)      # (4, C, X, Y): frames 17-20
future = model.rollout(traj.fields[11:17], traj.info, steps=60,
                       constants=traj.constants)                           # (60, C, X, Y): frames 17-76
```

`traj.fields` is a (T, C, X, Y) float32 tensor in physical units, channels in the order of `traj.info.fields`;
`traj.info` describes the dataset (fields, constant inputs, boundary conditions). Arrays from elsewhere can be
passed directly with `hypde.dataset("<name>")` as the description. Prediction cells inside an obstacle mask are
set to zero.

Command line:

```bash
hypde forecast shear_flow_Reynolds_5e4_Schmidt_1e0.hdf5 --index 0 --t0 17 --steps 60 -o forecast.npz
```

## Evaluate

```bash
hypde evaluate --data-root /path/to/the_well/datasets -o results.json           # all 13 datasets
hypde evaluate --data-root ... --datasets shear_flow --max-traj 10 -o shear.json  # a subset
```

`--data-root` holds The Well's `<dataset>/data/test/*.hdf5`; the trajectories scored are listed in
`src/hypde/assets/test_panel.json`. The one-step reading costs one forward pass per frame of every trajectory;
`--no-onestep` skips it. The FlowBench set was converted to The Well's format with our own split (test: ten held-out
geometries), and the converted files are not part of this release.

The package reproduces the original evaluation: on one test trajectory of each of six datasets (acoustic
scattering, Helmholtz staircase, shear flow, PlanetSWE, FlowBench, active matter), its
data loading matches Walrus's loader exactly, a forecast call is bitwise identical to the training code's, and the
VRMSE of an 8-frame forecast and of 12 one-step forecasts matches the recorded GPU evaluation to a relative 3e-5.

## Datasets

| Dataset | Grid | Fields | Boundaries (x / y) |
|---|---|---|---|
| `acoustic_scattering_discontinuous`, `_inclusions`, `_maze` | 256 × 256 | pressure, velocity | wall-open / wall-open |
| `active_matter` | 256 × 256 | concentration, velocity, D, E tensors | periodic / periodic |
| `euler_multi_quadrants_openBC` | 512 × 512 | energy, density, pressure, momentum | open / open |
| `euler_multi_quadrants_periodicBC` | 512 × 512 | energy, density, pressure, momentum | periodic / periodic |
| `flowbench_FPO_NS_2D_512x128_harmonics` | 512 × 128 | pressure, velocity (+ obstacle mask) | open / open |
| `gray_scott_reaction_diffusion` | 128 × 128 | A, B | periodic / periodic |
| `helmholtz_staircase` | 1024 × 256 | real and imaginary pressure (+ mask) | open / open |
| `planetswe` | 256 × 512 | height, velocity | open / periodic |
| `rayleigh_benard` | 512 × 128 | buoyancy, pressure, velocity | periodic / wall |
| `shear_flow` | 256 × 512 | tracer, pressure, velocity | periodic / periodic |
| `turbulent_radiative_layer_2D` | 128 × 384 | density, pressure, velocity | periodic / open |

The model has been trained and evaluated on these datasets; other systems on a regular 2D grid can be passed
with a `DatasetInfo` whose fields are in the model's vocabulary (`hypde.layers.NAMES`), but this has not been tested.

## Layout

- `src/hypde/`: model and field vocabulary (`model.py`, `layers.py`), dataset descriptions and HDF5 reader
  (`data.py`, `assets/datasets.json`), forecasting API with the normalisation (`api.py`), evaluation
  (`evaluate.py`, `assets/test_panel.json`), command line (`cli.py`)
- `weights/`: `config.json`; `download.sh` fetches `model.safetensors`
- `tools/convert_checkpoint.py`: conversion of the training checkpoint into the released weights
- `tests/`

## Citation

See [CITATION.cff](CITATION.cff): Hyra Team, "HyPDE: a 2D Foundation Model for Partial Differential Equations",
2026.

## License

Apache-2.0, see the repository [LICENSE](../../LICENSE).
