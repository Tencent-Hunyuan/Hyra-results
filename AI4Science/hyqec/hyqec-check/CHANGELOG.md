# Changelog

## 2.0.2 — 2026-10-28 (local release)

- Move internal conformance and process reports out of distributed contents;
  retain a public validation summary and neutral implementation descriptions.
- Document the current node/VERIFY runner for plain source exports without Git
  metadata. Distinguish historical adapters and optional external data or
  baseline drivers; remove references to excluded local evidence paths.
- Include the unchanged v1 specification and companion vectors at the paths
  cited by the frozen source and specification.
- Update distribution/Python v2 version metadata to 2.0.2. Preserve verifier
  logic, native bytes, frozen v1 profiles and licence notices.

## 2.0.1 — 2026-10-28 (local release)

- Update the project copyright to `Copyright (C) 2026 Tencent` under Apache-2.0.
  Preserve the licence text, third-party notices and historical release evidence.
- Clarify that the 888-search comparison covers node totals and that an S1
  filter check replays `J2` on the unmodified parity system.
- Remove development identifiers and machine paths from shipped documentation;
  exclude development history, review archives and their private harness adapter.
- Bump distribution/Python v2 version metadata to 2.0.1; preserve all native
  executable bytes and frozen v1 profiles. Rebuild and test both installed
  package formats, then tag locally; no push or upload.

## 2.0.0 — 2026-10-27 (local release)

- Add the separate `hyqec-check-v2` profile for BNB_CERT_SPEC_v2 `2.0-rc1`,
  with rule sets `greedy`, `s0`, `s0s1` and `s0s1s2`. Independently verify every
  prune, including S1 filter justifications (`J_S1F`) and exact S2 exceptions.
- Validate v2 identifiers and strict single-member gzip transport. Preserve the
  v1/v1.1 native sources and historical executables, v1 artifact compatibility,
  frozen profile version 1.3.0 and v1 transport behavior.
- Optimise allocation, syndrome and linear-algebra work, and run top-tree/FULL
  decision checks in the worker pool. Acceptance still waits for every check.
  For N > 1, `--threads N` means N workers plus a native reader.
- Complete exported certificate/vector conformance, 888 full searches, local
  and sanitizer tests, and native-payload/frozen-v1 package comparisons.
- Independent specification and code review: no blocker/no should-fix through `93deceb`. Author
  review by the independent clean-room implementer: corrected canonical S1 order and strict runner confirmed with
  every canonical vector field; no weakened VERIFY check found.
- Credits: independent clean-room implementation and conformance, performance
  optimisation, independent specification and code review, and exported
  certificate/data preparation. Release coordination: Minbo Gao.
- Build and tag locally; no push or package upload.

### Earlier v1 review scope

Review scope, clarified 2026-10-23: independent code review covered the earlier v1 (`24133140…`) and v1.1.0 wide (`7ca4a184…`) binaries.
The later 1.2.0 changes comprise bounded streaming proof transport without a
decompressed-proof scratch file, the native `Reader` seek guard, structured
rejections, distinct usage/partial exit codes and a permutation-work cap;
1.3.0 adds the rename. Canonical frontier scratch still scales with task count.
Release acceptance passed, including a full 452 gzip rerun under 1.2.0 and
246 artifact replays under each of 1.2.0 and 1.3.0.

The independent 1.2.0/1.3.0 delta review is **complete**: **no blocker**,
reported fixes confirmed, and 40/40 delta-specific checks passed on the released
1.3.0 wheel. The separate rename and packaging review also confirmed artifact
and native byte identity, together with installed CLI behavior.

The default permutation-work cap can reject a single permutation when
`n * m > 10000000`. Raise `--max-permutation-work` to at least
`permutation_count * n * max(1, m)` for a larger validation budget; see the
[resource-control example](README.md#resource-controls).

## 1.3.0 — 2026-10-23

- Rename distribution to `hyqec-check`, imports to `hyqec_check` and
  `hyqec_check_wide`, and commands to `hyqec-check` and `hyqec-check-wide`.
- Set NOTICE to `Copyright 2026 Minbo Gao`; the standard Apache license
  appendix template remains unchanged.
- Preserve all historical wire-format identifiers, certificate bytes and native
  verifier source; package names are not protocol identifiers.
- Use `HYQEC_CHECK_*` test/tool settings, retaining `BNBCHECK_*` as deprecated
  aliases for this release with the new names taking precedence.
- Rebuild and test the renamed package independently from a clean Git archive.

## 1.2.0 — 2026-10-23

- Installable Linux source and platform wheel packages include the C++20/OpenMP
  executables, both CLIs, Apache-2.0 LICENSE and NOTICE.
- Raw and gzip proof records stream through a bounded pipe; no full decompressed
  proof is written to scratch before checking. The proof-byte limit is enforced
  while streaming. Acceptance requires decoder EOF and valid gzip checksums;
  early native rejection stops decompression.
- Corrupt deflate and excessive JSON nesting produce structured JSON rejections.
- CLI usage errors exit 3; successful partial checks remain exit 2.
- Permutation validation work is bounded before algebra/permutation checks by
  `permutation_count * n * max(1, m)` (default 10,000,000).
- Native changes only allow a zero-offset nonseekable input stream; search rules
  are unchanged. Release builds do not overwrite historical pinned binaries.

## 1.1.0 — 2026-10-23

- Wide profile for specification revision 1.1 supports generator weight and
  column degree up to 128, with widened child counts and path-byte handling.
- Independent replay passed 96 wide artifacts, 72 original smoke artifacts
  and four real wide certificate claims.

## Original checker — 2026-10-22

- Independent algebra, orbit, frontier, subtree, trace and footer verification
  with generator weight ≤32 and column degree ≤64.
- Full 452, 488 and Lee production verification with the pinned original binary.
  Historical research evidence is preserved at Git revision `c49b4ef`.
