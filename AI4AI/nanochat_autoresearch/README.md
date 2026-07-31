# nanochat autoresearch result

This directory contains Hyra's single-B200 language-model training result for
the nanochat autoresearch track, evaluated with the same setup as the Recursive
baseline.

## Recorded result

| Metric | Value |
|---|---:|
| Validation BPB (lower is better) | **0.901543** |
| Training time | 300.1 s |
| Total time, including evaluation and setup | 348.4 s |
| Training steps | 3,410 |
| Training tokens | 447.0M |
| Peak GPU memory reported by the run | 168,547.7 MB |
| Transformer parameters, excluding sparse n-gram tables | 120.4M |

The complete console output is preserved in
[`FULL_TRAINING_LOG_0.901543.log`](FULL_TRAINING_LOG_0.901543.log).

The result was recorded on 1×NVIDIA B200, matching the Recursive baseline
hardware and evaluation setup. The environment reported Python 3.12, PyTorch
2.9.1 built for CUDA 13.0, and CUDA compute capability 10.0.

## Method

The model is a 9-layer, 768-dimensional transformer trained under a fixed
300-second budget. The released configuration combines:

- sliding-window attention with short and long windows;
- ReLU² MLPs, QK normalization, RoPE, value embeddings, and softcapping;
- Muon and AdamW optimizers with component-specific learning rates;
- sparse bigram and trigram hash features with frequency-aware learning rates;
- intra-document position buckets;
- causal unigram/bigram induction features; and
- a learned multi-layer readout and per-head attention temperature.

See [`train.py`](train.py) and [`solve.sh`](solve.sh) for the exact
configuration.

## Data layout

Prepare the data with the compatible nanochat preprocessing pipeline. The
directory selected by `NANOCHAT_DATA_DIR` must contain:

```text
<data-dir>/
├── shard_00000.parquet
├── ...
├── shard_06542.parquet
└── tokenizer/
    ├── tokenizer.pkl
    └── token_bytes.pt
```

`shard_06542.parquet` is used for validation; the other Parquet shards are used
for training.

## Run

The recorded configuration requires 1×NVIDIA B200, or a compatible CUDA
environment with approximately 169 GB of available GPU memory.

```bash
cd AI4AI/nanochat_autoresearch
export NANOCHAT_DATA_DIR=/path/to/prepared/nanochat/data
bash solve.sh
```

Optional environment variables in `solve.sh` expose the n-gram table sizes and
learning rates for controlled ablations. With the defaults unchanged, the
script:

1. trains for the fixed time budget;
2. writes the full run output to `.run/train.log`; and
3. extracts `val_bpb`, `training_seconds`, and `num_steps` into
   `.run/solution.json`.

## Notes

- Validation BPB is computed over a fixed evaluation budget using byte lengths
  from the released tokenizer.
- Runtime and memory use are hardware- and software-stack-sensitive.
- The full log is a recorded result, not a guarantee that another environment
  will reproduce the last decimal place.

See the [AI4AI overview](../README.md) for the other released tracks.
