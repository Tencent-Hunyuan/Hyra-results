# Hyra AI4AI Results

This directory contains the AI-for-AI artifacts released with
[Hyra](../README.md). The three tracks cover language model research,
end-to-end training-system optimization, and GPU kernel optimization.

These are research artifacts rather than a packaged library. Each result should
be interpreted with its benchmark protocol, hardware, software stack, and
measurement notes.

## Results at a glance

| Track | Recorded result | What is included |
|---|---:|---|
| [`nanochat_autoresearch`](nanochat_autoresearch/) | **0.901543 validation BPB** | Single-B200 training code, launcher, runtime utilities, and the complete recorded log |
| [`nanogpt_speedrun`](nanogpt_speedrun/) | **76.435 s** mean training time; **3.28007** mean validation loss | Eight-GPU training code, Triton kernels, three-run launcher, and the aggregated result |
| [`sol_execbench`](sol_execbench/) | **10** correctness-checked kernel submissions | Submission JSON files and a machine-readable manifest with per-kernel scores and latency |

The repository-level [results table](../README.md#results) contains the
corresponding prior-work comparisons and source references.

## Reproducing the results

The entry point for each training result is `solve.sh`:

```bash
# Single-GPU nanochat autoresearch result
cd nanochat_autoresearch
NANOCHAT_DATA_DIR=/path/to/prepared/nanochat/data bash solve.sh

# Eight-GPU nanoGPT speedrun result
cd ../nanogpt_speedrun
NANOGPT_VENV=/path/to/venv \
NANOGPT_DATA=/path/to/data-root \
bash solve.sh
```

See the README in each subdirectory before running. In particular:

- The `nanochat_autoresearch` result was measured on 1×NVIDIA B200, matching the
  Recursive baseline setup, and used about 168.5 GB of peak GPU memory.
- The `nanogpt_speedrun` result was measured on 8×H100 and is sensitive to the
  driver, clocks, compilation cache, and machine load.
- The `sol_execbench` files are benchmark submission artifacts. Running them
  requires the corresponding SOL-ExecBench harness; this repository does not
  vendor that harness.

Datasets, virtual environments, and compilation caches are intentionally not
stored in this repository.

## License

The repository is released under the
[Apache License 2.0](../LICENSE). Source files derived from upstream projects
retain their original notices and contributor attributions.
