# HyInv

HyInv is a 1.96-million-parameter graph neural network for protein inverse folding (backbone → sequence),
trained by our research agent Hyra 2.0. It follows the message-passing design of ProteinMPNN with a 4-layer
encoder and a 3-layer decoder, sees only the N, Cα, C and O backbone atoms, and runs on both CPU and GPU.
The released weights are the structure-reward reinforcement-learning version; deployment samples in 16
rounds — encode once, 8 rounds of blind drafting (temperature 0.6), then 8 rounds of rewriting by fixed
groups (temperature 0.34).

Technical report: [HyInv_Technical_Report.pdf](HyInv_Technical_Report.pdf)

## Installation

Python 3.10+ and PyTorch; the only other dependencies are `safetensors` and `numpy`. Accelerated inference
additionally needs `triton`.

```bash
pip install -e .
pip install -e '.[fast]'   # optional: accelerated inference on CUDA GPUs
```

## Usage

```bash
python -m hyinv design --pdb structure.pdb --n 128 --seed 0 --out designs.fasta
python -m hyinv design --pdb structure.pdb --chain A --fixed 12-20,45 --n 16   # keep residues 12-20 and 45
```

Options: `--draft-temperature 0.6`, `--rewrite-temperature 0.34`, `--device cpu|cuda`,
`--fast off|exact|bf16` (see *Accelerated inference*), `--weights DIR`. For multi-chain structures pass
`--chain AB`, or omit `--chain` to read all chains; chains are separated by `/` in the output, and fixed
positions must be qualified with a chain name (e.g. `A12-20,B7`).

```python
from hyinv import HyInv, read_pdb, design

model = HyInv.from_pretrained(device="cuda")          # reads the bundled weights/ by default
structure = read_pdb("structure.pdb", chains="A")
sequences = design(model, structure, n=8, seed=0, fixed="12-20,45")   # list[str]
```

## Accelerated inference

On a CUDA GPU the decoder can run as fused Triton kernels, with the whole 16-round sampler captured in a
single CUDA graph. This needs a CUDA build of PyTorch and `triton` (`pip install -e '.[fast]'`; verified
with triton 3.3.1 on an H20). `--fast` has three settings and defaults to `off`:

- `off`: reference implementation (FP32, CPU or GPU).
- `exact`: FP32-level acceleration (TF32x3 accumulation, erf GELU). With the same arguments (pdb, seed, n,
  batch size, fixed positions) the sequences are almost always identical to `off`: batch splitting, seeds
  (batch k uses `seed+k`) and position groups are exactly the same. TF32x3 does not accumulate in bitwise
  FP32 order, so when the sampling probabilities are nearly tied an occasional position can flip; across
  about 8,000 independently checked sequences of varying length (including missing atoms, two chains and
  fixed positions), only one differed at one position.
- `bf16`: BF16 operands with FP32 accumulation and tanh GELU, the fastest setting. This is a
  numerically-different path: sequences with the same seed are not identical to `off` (they are another
  sample from the same distribution; see the technical report for the S@D gap and speed measurements on the
  development set). It is reproducible with itself under the same arguments, and fixed positions are kept.

```bash
python -m hyinv design --pdb structure.pdb --n 128 --fast bf16 --out designs.fasta
```

```python
sequences = design(model, structure, n=128, seed=0, fast="bf16")   # or "exact"; model must be on CUDA
```

Speed (128 designs, max candidate batch 32, one H20, excluding compilation and graph capture, median):
`bf16` about 25 / 54 ms for 148 / 402 residues (24.7 / 50.6 ms in the technical report); `exact` about
119 / 281 ms (the report's exact path uses another set of kernels: 130 / 303 ms); the reference
implementation takes 999 / 2,633 ms under the same conditions (`bf16` and reference numbers are from the
technical report). Notes:

- The first call compiles the Triton kernels (about 25 s for `exact` and 14 s for `bf16` with a cold cache,
  then cached on disk) and captures a CUDA graph per (backbone length, batch size) (about 0.1–0.2 s for a new
  shape); later calls with the same shape only replay. A different batch size or a smaller final batch
  captures once more, with at most 4 lengths cached.
- Captured graphs occupy GPU memory, and the model weights must not change after the first call.
- Verified on a Hopper-class GPU (H20, sm_90); the kernels should compile on other architectures but have
  not been verified.
- Speed is sensitive to the number of CPU threads (we suggest `torch.set_num_threads(2)`).

## Notes

- Input: ATOM records of the first model in the PDB file (MSE is treated as MET); each residue takes one
  slot, and residues with missing backbone atoms keep their slot and are designed; residues are numbered
  0..n-1 within each chain, without filling gaps in the PDB numbering. Convert mmCIF to PDB first.
- Fixed positions are given by PDB residue number; they are visible as the native residue from the first
  round on and are kept unchanged in the output.
- Reproducibility: outputs are identical for the same `seed`, batch size (default 64, automatically
  tightened by length), device type and PyTorch version; each batch has its own random stream (batch k uses
  `seed+k`), so changing the batch size changes the result. The CPU and GPU random streams differ, so their
  sequences are not identical.
- The default path is the reference implementation (FP32); see *Accelerated inference* above.

## Layout

- `hyinv/`: model (`model.py`), sampler (`sampler.py`), PDB reader (`pdb.py`), command line
  (`__main__.py`), accelerated inference (`fast.py`, `_kernels.py`; requires triton)
- `weights/`: `model.safetensors` (FP32, 7.8 MB) and `config.json`
- `HyInv_Technical_Report.pdf`: technical report

## Citation and license

The model modules come from ProteinMPNN (MIT license):

> Dauparas et al., Robust deep learning–based protein sequence design using ProteinMPNN,
> Science 378, 49–56 (2022).

This package is released under the MIT license; see [LICENSE](LICENSE).
