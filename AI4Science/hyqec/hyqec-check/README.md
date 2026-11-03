# hyqec-check

**Version 2.0.2** is an independent companion package for checking stabilizer-code
branch-and-bound coverage certificates produced by `hyqec-verify`. It validates
instance algebra and automorphisms, reconstructs the complete top frontier,
replays every selected subtree, and checks counts, traces and the footer.

Full acceptance proves that the identified stabilizer code has **no nontrivial
logical Pauli of weight ≤ cap in the stated domain**. Thus its domain distance
is at least `cap + 1`, if that domain contains a logical operator. Domain `P`
allows all Paulis; `X` and `Z` restrict to that type. This is a lower bound, not
an exact distance. Checking can require as much search as producing a certificate;
the noncryptographic trace alone does not prove the bound.

The checker has a separate repository, package and authorship from the prover.
It was implemented from the written specification and soundness prose without
reading or importing prover source or tests. Python code and native search are
checker-owned. There is no runtime dependency on `hyqec-verify`.

## Version 2 profile

`hyqec-check-v2 verify proof.cov.gz --threads 3` checks container
`stabcert-bnb-coverage/2`, including the `greedy`, `s0`, `s0s1`, and `s0s1s2`
rule sets. It also accepts v1 containers through the frozen wide v1 profile.
The existing `hyqec-check` and `hyqec-check-wide` profiles retain their 1.3.0
source, native executables, behavior and reported profile version.

The v2 checker independently recomputes each pruning decision from the written
specification, then verifies a separate mathematical justification before
accepting that prune. Packing is checked for active, disjoint candidate sets;
S0 replays a parity refutation; an S1 filter prune replays the `J2`
refutation on the unmodified parity system and verifies aligned killer lists
whose supports leave no undecided qubit, and any other S1 prune verifies
an anchor and a reason excluding every relevant extra action; S2 checks separating linear functionals, with
exact exception proofs for fingerprint collisions. No justification supplied
by a prover is trusted. Every selected task is still executed completely.

The v2 implementation targets exported specification `2.0-rc1` (SHA256
`408806c15fbbfc8597fec20f4c0a0861846dc56cec027d214b984de3c59f43ee`).
Conformance passes for 56 exported artifacts, all 42,250 canonical node vectors,
seven VERIFY cases and 888 complete searches (node totals), totaling 2,396,942,083 nodes. S1 filter
prunes use the separately verified `J_S1F` justification. See the
[validation summary](docs/PUBLIC_VALIDATION.md),
[v2 design](docs/V2.md), and [performance analysis](docs/V2_PERFORMANCE.md).
This 2.0.2 release corrects documentation and updates version metadata;
its native payload is identical to 2.0.0. Release artifacts are built and tested before distribution.

With `--threads N` and N > 1, the v2 checker uses N replay workers plus one
native reading thread: **N + 1 native threads**. The Python transport also
uses a thread. Use `--threads 3` for a four-native-thread allocation. Top-tree
and full-transcript decision checks run in the worker pool.

The following full checks used the final native code, `--threads 3`, on two
physical cores and their SMT siblings. Every timing is **shared CCD** and is
not an isolated scaling comparison.

| Certificate | Nodes | Wall seconds | Placement |
|---|---:|---:|---|
| 452 c13, S0/S1/S2 | 5,113,500 | 28.191 | shared CCD |
| 488 c16, S0/S1/S2 | 7,005,476 | 56.975 | shared CCD |
| 488 c16, v2 greedy | 1,322,235,538 | 464.827 | shared CCD |
| Same greedy tree, frozen v1 | 1,322,235,538 | 297.159 | shared CCD |

Optimisation removes repeated allocation and syndrome work, uses exact S2
membership checks after a fingerprint filter, and schedules recorded-node
checks across the worker pool. Every prune still requires independent VERIFY.

Contributions include independent clean-room implementation and conformance,
performance optimisation, independent specification and code review, and
certificate test-data preparation. The independent soundness review through
`93deceb` found no blocker or should-fix issue. The implementation review
verified the corrected S1 order and strict vector runner.

## Names and compatibility

The distribution is `hyqec-check`; imports are `hyqec_check`,
`hyqec_check_wide` and `hyqec_check_v2`; commands are `hyqec-check`,
`hyqec-check-wide` and `hyqec-check-v2`.
These replace the former `stabcert-bnbcheck`, `bnbcheck` and `bnbcheck_wide`
package names. The old Python import paths and console entry points are not
installed by this release; an existing old checker executable can still be
used explicitly by the main package's checker integration.

**File formats keep their historical `stabcert-*` ids.** In particular,
`stabcert-group/1`, `stabcert-bnb-instance 1`, `stabcert-bnb-coverage/1`, `SBCOV1`,
and all search/trace identifiers are unchanged, preserving existing certificate
bytes and hashes. The development fixture ID `bnbcheck-reference-code/0` also
stays unchanged. See [soundness and format compatibility](docs/SOUNDNESS.md).

The test/tool settings are `HYQEC_CHECK_NATIVE_ENGINE`,
`HYQEC_CHECK_WIDE_ENGINE`, `HYQEC_CHECK_REFERENCE_ENGINE` and
`HYQEC_CHECK_MODULE`. Their `BNBCHECK_*` spellings are deprecated aliases for
release 1.3; the new spelling wins when both are set. `HYQEC_CHECK_MODULE` uses
`hyqec_check` or `hyqec_check_wide` (old module values are accepted by the
artifact smoke script). This checker has no persistent cache directory or
`STABCERT_*` settings. Standard compiler settings `CXX`/`CXXFLAGS` are unchanged.

Native executable contents are preserved from the 1.2.0 release. The private
wide executable's `--version` diagnostic therefore still reports its historical
engine label `bnbcheck-native 1.1.0`; the public CLI and verdict JSON report the
frozen profile version 1.3.0. The distribution and Python v2 profile report 2.0.2. The unchanged v2 native
executable retains its 2.0.0 version and filename.

## Install or build

Supported platform: Linux, Python 3.10+, a C++20 compiler (GCC recommended) and
OpenMP. Source builds require `g++`; wheels require compatible C++ and OpenMP
runtime libraries (`libstdc++`, `libgomp`). Wheels are platform-specific and
are not claimed to be portable to older Linux systems.

```sh
python3 -m pip install .
hyqec-check --version
hyqec-check-wide --version
hyqec-check-v2 --version
```

The PEP 517 build compiles and bundles all four native executables. There are
no third-party Python runtime dependencies. To build an sdist and wheel:

```sh
python3 -m pip install build
python3 -m build
```

Alternatively, build and test directly from a clean checkout:

```sh
make -j3
make test
python3 -m hyqec_check --version
python3 -m hyqec_check_wide --version
```

Both builds honor `CXX` and `CXXFLAGS`. Default flags are
`-O3 -std=c++20 -Wall -Wextra -Wpedantic`; `-fopenmp` is always added. Release
builds produce new versioned executables, preserving the historical pinned
historical pinned native executables.

## Verify

```sh
hyqec-check verify proof.cov.gz --threads 4 \
  --expected-group GROUP_SHA256 --expected-instance INSTANCE_SHA256 \
  --expected-domain P --expected-cap 17
```

Replace the expected identities with actual 64-digit lowercase hashes. The group
hash identifies the stabilizer group regardless of generator order; the instance
hash binds exact ordered instance bytes. Embedded hashes are always validated.
Expected values bind acceptance to the caller's intended code, domain and bound;
otherwise the verdict identifies the claim actually checked.

| Command | Generator weight | Column degree | Module equivalent |
|---|---:|---:|---|
| `hyqec-check` | ≤32 | ≤64 | `python3 -m hyqec_check` |
| `hyqec-check-wide` | ≤128 | ≤128 | `python3 -m hyqec_check_wide` |

Both profiles use container `stabcert-bnb-coverage/1`, search
`irreducible-pauli-dfs/1` and trace `poly64-state-rules/1`. They implement the
rules in **BNB_CERT_SPEC.md revision 1.1**; the original profile deliberately
retains the narrower pre-1.1 limits. Other limits are `n ≤ 4096`, `m ≤ 4096`,
`k > 0`, `cap ≤ 46`, `1 ≤ split ≤ 47`, and counters ≤ `2^63 - 1`.
The specification is maintained with hyqec-verify, document revision
`cb00ad7eee48512240d1c484aa22afd441a7708d`. Its revision-1.1 limit expansion
changes neither search rules nor wire format.

Always use the Python CLI for untrusted certificates. Native executables accept
a private, already validated model and are not standalone public verifiers.

## Verdicts and partial checks

Verification writes one JSON object to stdout; progress goes to stderr.
Acceptance requires **both exit 0 and `certificate_verified: true`**.

| Exit | Meaning |
|---:|---|
| 0 | Full certificate accepted |
| 1 | Certificate rejected, malformed input, resource limit, or I/O failure |
| 2 | Successful partial check; no global distance bound established |
| 3 | CLI usage error |

`--help` and `--version` also exit 0 but do not check a certificate. Development
commands `validate-reference-code` and `reference-search` operate on checker-owned
fixtures, report `certificate_verified: false`, and do not accept certificates.

```sh
hyqec-check verify proof.cov.gz --threads 4 --sample 1000 --seed 42
hyqec-check verify proof.cov.gz --threads 4 --range 0:1000
```

Ranges are half-open. Partial modes validate the whole top frontier, task IDs
and paths, and footer consistency, but replay only selected subtrees. They
return `status: partial` even if selection happens to cover all tasks. Unchecked
task summaries cannot establish the full bound. JSON includes the checker version,
limits profile, specification revision, identities, domain/cap, task and node
counts, global trace and timings.

## Resource controls

Raw and gzip inputs are detected by magic. Proof bodies stream in 64 KiB chunks
through an OS pipe with backpressure; decompressed proofs are never spooled to
disk. The native reader has a 64 KiB buffer. Early record rejection stops decoding.
For v1, full acceptance requires logical EOF, all gzip members and valid
checksums; reserved gzip flag bits and trailing zero padding remain tolerated.
For v2, exactly one gzip member is accepted: additional members, padding, bad
CRC/length trailers, and logical bytes after COMPLETE are rejected.

| Option | Default | Bounds |
|---|---:|---|
| `--max-header-bytes` | 134217728 (128 MiB) | Decoded header bytes before JSON parsing |
| `--max-proof-bytes` | 1099511627776 (1 TiB) | Decoded proof records, excluding header |
| `--max-permutation-work` | 10000000 | `permutation_count * n * max(1, m)` |
| `--scratch-dir` | System temporary directory | Validated model and canonical frontier spool location |
| `--threads` | Required, 1–384 | V1 replay threads; v2 N workers plus reader for N > 1 |

The permutation budget is checked before instance algebra/permutation validation.
Header dimensions must later match the parsed instance exactly, so understated
dimensions cannot bypass it. This work estimate is not a wall-time bound; the
header limit also bounds parsing input.

The default 10,000,000 budget rejects even a single permutation when `n * m`
exceeds 10,000,000, although those dimensions can be within the supported limits.
For example, `n = m = 4096` requires 16,777,216 units for one permutation.
Raise the budget to at least `permutation_count * n * max(1, m)` when you intend
to spend that validation work:

```sh
hyqec-check verify proof.cov.gz --threads 4 --max-permutation-work 16777216
```

The same option works with `hyqec-check-wide`. This example permits one
permutation at `n = m = 4096`; multiple permutations require a proportionally
larger budget. Raising the budget retains all algebra and permutation checks
and may increase validation time.

V1 native replay holds at most 4096 task descriptions per batch. V2 uses a
bounded task backlog and checks top/FULL recorded decisions in the worker pool. Scratch still
includes a canonical frontier spool, proportional to validated frontier tasks;
the proof limit is **not** a total scratch or CPU-time quota. Header/algebra data
and per-thread search state occupy memory. Choose limits and resources for the
certificate.

## Tests and evidence

`make test` runs self-contained unittest tests from a clean checkout without a
prover installation, production artifacts, network or GPU. Cases cover published
vectors, brute-force tiny-code distances, wide boundaries, forged logical leaves,
parser/transport failures and partial modes. The fixture writer shares checker
algebra/hash/trace utilities; the string/group-closure oracle and published vectors
provide additional checks.

After the build and test commands above:

```sh
make ubsan ubsan-wide
HYQEC_CHECK_REFERENCE_ENGINE="$PWD/build/hyqec-check-reference-ubsan" \
HYQEC_CHECK_NATIVE_ENGINE="$PWD/build/hyqec-check-native-v1.3.0-ubsan" \
HYQEC_CHECK_WIDE_ENGINE="$PWD/build/hyqec-check-wide-native-v1.3.0-ubsan" \
  python3 -m unittest discover -s tests -v
```

Version 1.3.0 passes all 48 tests from the unpacked sdist against a fresh
installed wheel and all 246 artifact replays. Native wheel contents match the
1.2.0 engines byte for byte. The 1.2.0 native core passed a full 452 cap17 gzip
replay: all 2,282,025 tasks and 124,648,548,270 nodes, with the historical trace
unchanged. Version 1.3.0 preserves these native source files and changes the
package/import/CLI names.

The independent code review covered the earlier
v1 binary (`24133140…`) and v1.1.0 wide binary (`7ca4a184…`). Relative to those
reviewed releases, 1.2.0 adds a streaming reader with bounded proof transport
and no decompressed-proof scratch file, the native `Reader` seek guard,
structured rejections, distinct usage/partial exit codes and a permutation-work
cap; 1.3.0 adds the package/import/CLI rename. Canonical frontier scratch still
scales with validated task count. The later releases passed the acceptance
checks above, including the full 452 rerun under 1.2.0 and 246 artifact replays
under each release; acceptance runs are separate evidence from a code review.

The independent **1.2.0/1.3.0 delta review is complete**. It reviewed tag
`v1.3.0` (`126de1a`) and the released wheel (`bc0c9760…`), found **no blocker**,
confirmed the reported fixes, and passed 40/40 delta-specific checks on that
wheel. A separate rename and packaging review also confirmed artifact/native
byte identity and installed CLI behavior.

See [CHANGELOG](CHANGELOG.md) and [release report](REPORT.md). Historical production
verdicts belong to their exact executables, not to every future build. Earlier
logs and artifacts remain at Git revision `c49b4ef`, outside release distributions.
Copyright (C) 2026 Tencent. Licensed under Apache-2.0; see [LICENSE](LICENSE)
and [NOTICE](NOTICE).
