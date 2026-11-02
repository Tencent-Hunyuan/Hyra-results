# HyPLM-Flash

HyPLM-Flash is an efficient, open protein language model: a 650-million-parameter masked language model
trained by our research agent Hyra 2.0 on 0.73 times the data of ESM-2 650M and a fifth to an eighth of the
data of E1-600M and ESM C 600M. In a matched evaluation of 28 public checkpoints on 32 tasks it reaches a
ProteinGym zero-shot Spearman correlation of 0.409, statistically tied with ESM-2 650M, ESM C 600M and
E1-600M, gives the best long-range contact precision of all evaluated models, and its inference engine
embeds about 130,000 residues per second on a single H20, 1.9 to 2.7 times faster than other 600M-class
encoders.

Technical report: [HyPLM_Technical_Report.pdf](HyPLM_Technical_Report.pdf)

## Installation

Needs a CUDA GPU (Hopper GPUs such as the H20 or H100 recommended) and Python 3.10+. The 2.6 GB of FP32
weights are released as two shards in this repository's GitHub Releases; download, merge and verify them
with:

```bash
bash weights/download.sh     # needs curl; set GITHUB_TOKEN only if you hit API rate limits
pip install -e .             # torch, safetensors, triton, flash-attn
```

FlashAttention-3 (`flash_attn_interface`) is used automatically when installed, otherwise FlashAttention-2.

## Usage

```python
from hyplm import HyPLMEncoder

encoder = HyPLMEncoder.from_pretrained("weights")       # INT8 by default
sequences = ["MKTAYIAKQRQISFVKSHFSRQ", "ACDEFGHIKLMNPQRSTVWY"]

residue = encoder.encode(sequences)                     # list of [L, 1280]
pooled = encoder.encode(sequences, pooling="mean")      # list of [1280]
layer16 = encoder.encode(sequences, layer=16)           # output of block 16
```

Returns a list of CPU FP32 tensors in input order, without the leading and trailing special tokens. A single
sequence may be up to 8,190 residues; input is upper-cased, whitespace is removed and `J` is converted to
`X`, while invalid characters raise an error. Proteins are automatically packed into padding-free batches
with `batch_tokens` (default 32768) tokens per batch.

Masked-language-model logits, e.g. for zero-shot mutation scoring:

```python
sequence = "MKTAYIAKQRQISFVKSHFSRQ"
logits = encoder.logits(sequence, masks=[2])[0]         # [L, 33]; residue 3 is replaced by <mask> first
log_probs = logits[2].log_softmax(-1)
tokens = encoder.tokenizer.tokens                       # vocabulary order
score = log_probs[tokens.index("V")] - log_probs[tokens.index("T")]   # masked-marginal score of T3V
```

## Precision

| Mode | H20 throughput (residues/s) | Recommended for |
|---|---|---|
| `precision="int8"` (default) | about 130,000 | representation extraction, zero-shot scoring, structure and function prediction |
| `precision="bf16"` | about 90,000 | training supervised regressors on near-identical variants (e.g. AAV and GB1 in FLIP) |

Under INT8, zero-shot, structure, function-annotation and likelihood metrics are within 0.01 of BF16.
Representations of variants of the same protein differ by only about 1%, so BF16 is recommended when
training regressors on such data.

## Command line and multiple GPUs

```bash
python -m hyplm MKTAYIAKQRQISFVKSHFSRQ --pooling mean --output embeddings.pt
```

```python
from hyplm.engine import encode_multi_gpu

if __name__ == "__main__":  # one worker process per GPU
    vectors = encode_multi_gpu(sequences, checkpoint="weights", pooling="mean")
```

For low-latency inference on a single protein, use `hyplm.engine.graphs.GraphRunner` (CUDA graphs).

## Layout

- `hyplm/`: inference interface and engine (padding-free packing, fused Triton kernels, INT8 matrix
  multiplication)
- `weights/`: model config and `download.sh`; the FP32 weights (safetensors, 2.61 GB) are fetched from the
  GitHub Release
- `HyPLM_Technical_Report.pdf`: technical report

## License

MIT; see [LICENSE](LICENSE).
