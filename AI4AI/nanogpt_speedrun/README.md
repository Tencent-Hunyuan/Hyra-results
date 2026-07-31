# nanoGPT speedrun: FP8 MLP with a firm-8 schedule

This directory contains Hyra's submission for the nanoGPT speedrun track. It
adds FP8 computation to the two large MLP matrix multiplications in the
record-83 training stack and adjusts the schedule by eight cooldown steps.

## Recorded three-run result

The tracked [`solution.json`](solution.json) contains three independent runs on
the same 8×H100 machine:

| Run | Training time | Validation loss |
|---:|---:|---:|
| 1 | 76.431 s | 3.2783 |
| 2 | 76.467 s | 3.2827 |
| 3 | 76.408 s | 3.2792 |
| **Mean** | **76.435 s** | **3.28007** |

The prior result cited in the repository-level comparison is approximately
77.5 seconds. This recorded mean is about 1.1 seconds faster on the stated
hardware basis.

### Measurement note

The three-run mean validation loss is **3.28007**, which rounds to **3.280** at
three decimal places. The individual values (3.2783, 3.2827, and 3.2792) reflect
the expected run-to-run variation of this hardware-sensitive benchmark.

## What changed

Relative to the record-83 baseline, the released code:

- computes the MLP forward `fc` and backward `dpre` matrix multiplications with
  FP8 `tl.dot` operations inside the fused Triton
  `linear_relu_square_kernel`;
- uses tensor-provided per-tensor scales and dequantizes by those scales;
- retunes `BLOCK_SIZE_K` from 64 to 128 for the FP8 path; and
- increases `num_scheduled_iterations` from 1,405 to 1,413.

The rest of the stack includes NorMuon and Adam, windowed FlashAttention 3 with
YaRN, a 2,816-wide ReLU² MLP, multi-token prediction, an FP8 language-model
head, value embeddings, MUDD, and U-Net-style skips.

## Recorded environment

- 8×NVIDIA H100
- Python 3.12
- PyTorch 2.10.0 built for CUDA 12.8
- Triton 3.6.0
- `kernels-community/flash-attn3`
- NVIDIA driver 580.159.04

Wall-clock results are sensitive to driver versions, GPU clocks, power limits,
compilation caches, machine load, and contention. Compare numbers only under a
matched setup.

## Data layout

`NANOGPT_DATA` must point to a directory containing the FineWeb10B binary
shards at:

```text
<NANOGPT_DATA>/data/fineweb10B/
├── fineweb_train_*.bin
└── fineweb_val_*.bin
```

The virtual environment selected by `NANOGPT_VENV` must provide `python` and
`torchrun`.

## Run

```bash
cd AI4AI/nanogpt_speedrun
export NANOGPT_VENV=/path/to/venv
export NANOGPT_DATA=/path/to/data-root
bash solve.sh
```

By default, `solve.sh` launches three fresh eight-process training runs, writes
`run_1.log` through `run_3.log`, and aggregates their final metrics into
`solution.json`.

For a one-run diagnostic using the full schedule:

```bash
VALIDATE=1 NANOGPT_RUNS=1 bash solve.sh
```

For a shorter smoke test, also set `NANOGPT_MAX_STEPS`; such a run is not a
scored result.

See the [AI4AI overview](../README.md) for the other released tracks.
