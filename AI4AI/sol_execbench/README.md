# SOL-ExecBench showcase solutions

This directory contains ten correctness-checked GPU-kernel submissions produced
by Hyra for SOL-ExecBench. The JSON files preserve the submitted source code and
benchmark metadata; [`manifest.json`](manifest.json) is the machine-readable
index.

## Results

| Kernel | Workload | Submission | SOL score | Latency | Artifact |
|---:|---|---:|---:|---:|---|
| 145 | Hyena block | #25621 | 0.965872 | 0.021479 ms | [`JSON`](k145__sid25621__hyena_complete_forward_block__051_seqlen-finetuned-reconstructed_hyena_complete_forward_block.json) |
| 138 | Mamba scan | #25471 | 0.913815 | 0.025766 ms | [`JSON`](k138__sid25471__mamba_discretization_and_segsum__044_mamba_discretization_and_segsum.json) |
| 154 | Linear attention | #25648 | 0.909533 | 0.059290 ms | [`JSON`](k154__sid25648__chunk_gated_delta_rule_linear_attention__060_chunk_gated_delta_rule_linear_attention.json) |
| 067 | Flash/GQA attention | #24692 | 0.692890 | 0.177612 ms | [`JSON`](k067__sid24692__flash_attention_gqa_ultralong__067_flash_attention_gqa_ultralong.json) |
| 175 | MoE dispatch | #25683 | 0.924206 | 0.887775 ms | [`JSON`](k175__sid25683__moe_sparse_expert_dispatch__081_moe_sparse_expert_dispatch.json) |
| 185 | NVFP4 vision | #25691 | 0.948448 | 0.038247 ms | [`JSON`](k185__sid25691__nvfp4_vision_temporal_patch_merge_with_projection__009_nvfp4_vision_temporal_patch_merge_with_projection.json) |
| 215 | GEMM | #25844 | 0.536168 | 0.023509 ms | [`JSON`](k215__sid25844__gemm_n2048_k4096__006_gemm_n2048_k4096.json) |
| 225 | GQA serving prefill | #25891 | 0.925048 | 0.008197 ms | [`JSON`](k225__sid25891__gqa_ragged_prefill_causal_h32_kv4_d128__016_gqa_ragged_prefill_causal_h32_kv4_d128.json) |
| 227 | MLA paged decode | #25893 | 0.956477 | 0.018297 ms | [`JSON`](k227__sid25893__mla_paged_decode_h16_ckv512_kpe64_ps1__018_mla_paged_decode_h16_ckv512_kpe64_ps1.json) |
| 228 | MLA paged prefill | #25896 | 0.966660 | 0.045929 ms | [`JSON`](k228__sid25896__mla_paged_prefill_causal_h16_ckv512_kpe64_ps1__019_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1.json) |

All entries in the manifest have `status: "COMPLETED"` and
`is_correct: true`.

## Artifact format

Each filename encodes:

```text
k<kernel>__sid<submission>__<task-slug>__<original-filename>.json
```

The corresponding manifest entry records the kernel and submission IDs, task
slug, original and saved filenames, file size, SOL score, latency, submission
mode, completion status, and correctness result.

The submitted implementation is embedded in its JSON artifact. These files are
intended to be consumed by the SOL-ExecBench tooling; they are not standalone
Python programs.

## Reproduction notes

- Use the matching SOL-ExecBench harness and workload definitions.
- Scores and latency are the recorded benchmark outputs and may vary with
  hardware, drivers, compiler versions, and harness revisions.
- Preserve correctness checks when retuning a kernel; latency without the
  recorded correctness status is not an equivalent result.

See the [AI4AI overview](../README.md) for the other released tracks.
