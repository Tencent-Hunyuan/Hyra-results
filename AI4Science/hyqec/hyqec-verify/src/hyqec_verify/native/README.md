# Native engines

The Python package builds these optional sources at first use. No native
compiler runs during installation. `hyqec_verify.native.build(target)` accepts
`bnb-cpu`, `bnb-cuda`, `ris-cpu`, and `ris-cuda`. `version(path)` reads the
binary's one-line JSON build metadata. The engines are driven only through
`hyqec_verify.engines` (subprocess protocol); `docs/SOUNDNESS.md` describes the trust boundary.

B&B sources were ported from an earlier research prototype
(`pauli_cert`: `core.h`, `pauli_cert.cu`, `warp.cuh`). The CPU core's
branching and packing bound are unchanged. Changes add restricted X/Z initial
domains, bounds-checked input, reset of task state, build metadata, portable
warp reductions, and checkpoint validation. CPU and GPU implement identical
search decisions. Exact node-count identity applies to **no-find runs**;
parallel cap reductions after finding a witness can change node totals.

RIS CUDA was ported from the same prototype's `pauli_ris_g.cu`, which
supports both shared-memory and global-memory matrices. The CPU implementation
uses the same qubit-aligned elimination and rank-one/rank-two pivot swaps.
These are upper-bound searches. Python checks every witness against the
stabilizer group and recounts its Pauli weight. Neither RIS implementation
produces a lower-bound certificate.

## Builds

B&B CPU/CUDA builds require zlib development headers and library (`-lz`) for
checkpoint integrity checks; RIS has no zlib dependency.

`CXX` and `NVCC` select compilers. `HYQEC_OPENMP=0` disables OpenMP,
`HYQEC_OPENMP=1` requires it, and the default probes support. Without OpenMP,
CPU engines use one thread and report that fact in `--version`.
`HYQEC_CXXFLAGS` and `HYQEC_NVCCFLAGS` add explicit user flags; the default
does not enable host-specific instruction sets. CUDA defaults to
`cuda_arch="native"`; a list such as `"70,80,86,89,90"` builds a fat binary.
The last listed architecture also supplies PTX for forward compatibility.

The cache is `$HYQEC_CACHE/native/` or `~/.cache/hyqec-verify/native/`. Its key
covers sources, interface version, compiler identity, and flags. Metadata and
compiler logs accompany each binary. Publication is atomic and concurrent
first use is tested. `HYQEC_NO_BUILD=1` uses an existing matching binary or
reports unavailable. The implementation currently targets POSIX hosts
(Linux/macOS); CUDA requires a supported NVIDIA host/toolkit.

B&B `variant=1024` or `variant=4096` selects NMAX and MMAX together, with
generator weight 32, column degree 64, and maximum cap 46. CPU-only variants
`"1024-w64"`, `"1024-w128"`, `"4096-w64"` and `"4096-w128"` raise generator
weight to 64 or 128 and column degree to 128. They keep the same search rules;
wide undo records store each position's original state instead of packed
32-position masks. CUDA and RIS targets reject these wide variants.

Wide coverage artifacts need a checker supporting specification revision 1.1:
the companion's wide profile (`hyqec-check-wide`, or `hyqec-check-v2`), which
independently accepted real wide artifacts; the original-limits profile
(`hyqec-check`) rejects them by design. The CUDA
driver reduces block size when necessary to fit shared memory, and rejects an
instance if even one warp does not fit. RIS supports indices up to n=16000;
actual CUDA limits also depend on shared-memory bookkeeping and free global
memory. Large RIS matrices automatically use global memory; `--gmem` forces
that mode for reproducibility tests.

B&B CUDA uses blocking host waits for stream completion. Coverage export
creates one host worker per GPU per `--slots` value; `--threads` controls CPU
search separately. Waiting workers sleep instead of occupying CPU cores.
The defaults remain two slots per GPU and a 2,000,000-task chunk ceiling.
For count-only GPU searches, each round uses the smaller of that ceiling
and `ceil(round_tasks / (GPU_count * slots * 4))` tasks per chunk. This gives
small rounds several pieces per slot. Task states and no-find node totals
are unchanged; CPU-only searches retain the configured chunk ceiling.

## Checkpoints

For a checkpointed B&B run, Python computes SHA-256 over the instance and
executable and passes `--instance-sha256` and `--binary-sha256`. The standalone
binary does not compute these digests. The caller must re-hash files on every
run and resume. The checkpoint binds both values, the compiled source hash,
domain, cap, resolved split, delta, requested stop-first mode, and initial task
count. A mismatch exits 6 without modifying the checkpoint. Chunk sizes,
threads, node budgets, and selected devices can change when resuming the same
executable. Switching CPU and CUDA executables changes the binary hash and is
therefore rejected.

Completed chunks and aborted tasks are appended in the same locked order
used to construct the next round. Malformed records, overlapping chunks, bad
indices, and an incomplete predecessor round are rejected. A partially written
last line is discarded only after validating all committed records. The
checkpoint and its run records are trusted local state, not a cryptographic
proof object.

A checkpointed find writes an `F` record and stops. Its witness is an upper
bound; it need not be minimum. `done.stopped_on_find=true` communicates that
case, including `--stop-first`. Resuming an `F` record re-emits its verified
witness and reports zero nodes: schedule-dependent partial node totals are not
stored in an F record. Re-execute without stop-first/checkpoint to establish
minimality. No-find completed runs preserve exact node totals across resumes.

## RIS budgets and measurement

`--max-steps` limits attempted swaps, including failed swaps; in fresh mode
(`--no-chain`) or with zero steps per round, it limits initializations. CPU
checks this limit per attempt; CUDA checks at a whole-launch boundary, so it
can overshoot by one launch. `steps` counts successful swaps,
`attempted_steps` counts attempts, and `inits` counts fresh eliminations.
`--seconds` is a wall-clock limit checked between CPU operations or CUDA
launches. One CPU thread with a fixed seed and a fixed work budget is
reproducible when the time limit does not intervene. Wall-clock-only runs
cannot promise equal amounts of work.

Fresh eliminations and correlated chain swaps are distinct work units; a ratio
of their measured rates does not measure time to equal witness quality.

## Coverage export (experimental v1)

B&B CPU and CUDA accept `--emit-cert FILE --cert-metadata META.json` with
`--instance-sha256 HEX --binary-sha256 HEX`. Metadata contains exactly
`{"group_sha256":"<64 lowercase hex>","permutations":[...]}`; each permutation
is an n-element zero-based qubit map. Their generated orbits must match the
instance. The independent checker additionally verifies that each permutation
preserves the stabilizer group.

The native output is a raw streaming `SBCOV1` container regardless of suffix.
A caller can gzip the completed file (mtime=0, empty original filename);
`engines/bnb.py` does so, and the checker accepts both transports. The destination
is published atomically only after a complete no-find run. `--cert-detail full`
adds every task event and is intended for small cases. Node budgets and re-split
rounds preserve the original task IDs, node counts and composable traces.
Coverage export rejects `--stop-first`, old count-only checkpoints, and
`--bound none`. Source/binary hashes are provenance, never proof checking.

The native ablation switch `--bound none|greedy` defaults to greedy. `none`
disables the packing bound while retaining weight/dead/leaf pruning and the
same branch order. It is incompatible with version-2 checkpoints.

The exact format and independent replay obligations are in
`docs/BNB_CERT_SPEC.md`. A completed artifact still requires a complete check;
matching node counts or sampling tasks cannot establish its lower bound.
