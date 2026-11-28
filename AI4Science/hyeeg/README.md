# HyEEG

HyEEG is a family of self-supervised, montage-agnostic EEG foundation models trained from scratch by our
research agent **Hyra 2.0** on over 64,000 hours of public human EEG verified to have zero recording or
participant overlap with downstream evaluation benchmarks. Released at three scales (**HyEEG-Small**,
3.25M parameters; **HyEEG-Base**, 9.53M parameters; and **HyEEG-Large**, 25.23M parameters), HyEEG
combines continuous 4D spatio-temporal positional encoding with channel-major block masking, automated
flat-channel filtering, and sub-window amplitude normalization to deliver strong out-of-the-box frozen
representations across arbitrary electrode montages and trial lengths.

Under a controlled frozen-encoder linear-probing evaluation across six public EEG benchmarks (LargeMI,
EEGMMIDB, SHU-DB, BCIC IV-2a, EEGMAT, and DTU; 17 cross-subject tasks, 79 subject-disjoint splits),
HyEEG-Large, HyEEG-Base, and HyEEG-Small rank first, second, and third overall, scale monotonically with
depth, and establish a continuous accuracy-scale Pareto frontier ahead of five public EEG foundation
models.

Technical report: [HyEEG_Technical_Report.pdf](HyEEG_Technical_Report.pdf)

## Benchmark summary

Dataset-equal composite Balanced Accuracy (BA) and Chance-Normalized Balanced Accuracy (cnBA; `0.000` =
random chance, `1.000` = perfect accuracy) under frozen-encoder linear probing across all six public
benchmarks (17 cross-subject tasks; HyEEG scores averaged across two independent pretraining seeds):

| Model | Encoder Params | Layers | GFLOPs / win | Composite BA ↑ | Composite cnBA ↑ | Split win rate vs. baselines |
|---|---:|---:|---:|---:|---:|---:|
| **HyEEG-Large** | 25.23M | 8 | 3.91 | **0.557** | **0.229** | 76% - 92% (of 79 splits) |
| **HyEEG-Base** | 9.53M | 3 | 1.48 | 0.548 | 0.213 | 71% - 86% (of 79 splits) |
| **HyEEG-Small** | **3.25M** | **1** | 0.51 | 0.540 | 0.194 | 66% - 77% (of 79 splits) |
| REVE-small | 12.67M | 4 | 1.97 | 0.537 | 0.196 | - |
| REVE-B | 69.19M | 22 | 10.77 | 0.528 | 0.178 | - |
| LUNA-B | 6.57M | 8 | 0.41 | 0.508 | 0.142 | - |
| LaBraM-B | 5.82M | 12 | 0.95 | 0.490 | 0.110 | - |
| CBraMod | 4.88M | 12 | 0.75 | 0.487 | 0.104 | - |

## Installation

Needs Python 3.10+ and PyTorch 2.2+ (works on CUDA GPUs and CPU; GPU recommended for large batches).
The FP32 weights for all three scales (152 MB total: 13.0 MB Small, 38.1 MB Base, 100.9 MB Large) are
released as assets in this repository's GitHub Releases; download and verify them with:

```bash
bash weights/download.sh           # downloads small, base, and large; verifies sha256
# or download a single scale:
# bash weights/download.sh large
pip install -e .                   # torch, safetensors, numpy
```

Set `GITHUB_TOKEN` only if you hit GitHub API rate limits.

## Usage

Inputs are multi-channel EEG segments resampled to **200 Hz** (at least 1.0 s / 200 samples), shaped
`(C, T)` for a single trial or `(B, C, T)` for a batch, paired with either standard 10-20 / 10-10 /
10-05 channel names (`channels=`) or explicit 3D scalp coordinates in meters (`positions=`, shape
`(C, 3)` or `(B, C, 3)`):

```python
import numpy as np
from hyeeg import HyEEGEncoder

encoder = HyEEGEncoder.from_pretrained("weights/large")     # or "weights/base", "weights/small"

channels = ["Fp1", "Fp2", "F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4", "O1", "O2"]
eeg = np.random.randn(8, len(channels), 800).astype(np.float32)  # 8 trials x 13 channels x 4 s at 200 Hz

emb = encoder.encode(eeg, channels=channels)                     # [8, 512] pretrained query-token readout
mean_emb = encoder.encode(eeg, channels=channels, pooling="mean")    # [8, 512] mean over patches
tokens = encoder.encode(eeg, channels=channels, pooling="tokens")    # [8, 13, 4, 512] spatio-temporal tokens
layer4 = encoder.encode(eeg, channels=channels, layer=4)             # [8, 512] intermediate block readout
```

Returns a CPU FP32 tensor in input order. By default, `encode()` applies per-window, per-channel
z-normalization with a relative standard-deviation floor (`1e-3` of the window's median channel std) and
`[-15, 15]` clipping, matching the evaluation pipeline. Pass `normalize=False` if your trials are already
normalized. For non-standard electrode layouts or custom digitized sensor positions, pass 3D coordinates
in meters directly via `positions=`.

## Model scales

| Scale | Directory | Params | Layers | Size (FP32) | Recommended for |
|---|---|---:|---:|---:|---|
| `HyEEG-Small` | `weights/small` | 3.25M | 1 | 13.0 MB | edge / real-time BCI and localized sensorimotor decoding |
| `HyEEG-Base` | `weights/base` | 9.53M | 3 | 38.1 MB | compact general-purpose frozen feature extraction |
| `HyEEG-Large` | `weights/large` | 25.23M | 8 | 100.9 MB | highest overall accuracy, cognitive workload, and multi-class decoding |

Each released checkpoint is the primary EMA encoder checkpoint (`seed 0`, 64,069 steps) trained from
scratch on the waveform-decontaminated pretraining corpus.

## Command line

```bash
python -m hyeeg trials.npy \
  --channels Fp1 Fp2 F3 Fz F4 C3 Cz C4 P3 Pz P4 O1 O2 \
  --checkpoint weights/large \
  --pooling attention \
  --output embeddings.pt
```

Supports `.npy` and `.pt` input arrays of shape `(C, T)` or `(B, C, T)`, plus `--positions coords.npy`
for custom 3D electrode coordinates and `--precision bf16` for faster GPU inference.

## Layout

- `hyeeg/`: standalone pure-PyTorch model, 3D electrode position lookup (`assets/positions.json`), frozen
  inference API, and CLI
- `weights/`: per-scale `config.json` files (`small/`, `base/`, `large/`) and `download.sh`; FP32
  `model.safetensors` weights are fetched from the GitHub Release (`hyeeg-weights`)
- `tests/`: unit tests for coordinate lookup, window normalization, model forward pass, and checkpoint
  loading
- `HyEEG_Technical_Report.pdf`: technical report

## License

MIT; see [LICENSE](LICENSE).
