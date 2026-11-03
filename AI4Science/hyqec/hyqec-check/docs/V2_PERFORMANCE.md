# v2 checker performance work (first optimisation round)

Base: `24ad05e` (the independent draft-4 checker). Every commit below changes one thing, keeps every decision exact, and
leaves VERIFY reading only the node labels, the instance (rows and their column transpose) and J. The v1 sources
and binaries are untouched (`docs/V1_FROZEN_SHA256.json`).

## Why the base was slow

1-thread profile of `24ad05e` on 488-123-22 c16 s0s1s2, 3000 sampled tasks (130,951 nodes, 54 s):

| part | share | cause |
|---|---|---|
| S2 decision | 55% | std::map basis; every one of ~3n actions fully reduced; map of remainders; O(m^2) functionals |
| VERIFY (S1, S2, S0) | 33% | per prune: n column vectors rebuilt, a std::set F(g) for all m generators, full syndrome |
| S1 decision, packing, System | 10% | per-action and per-value allocations, n-sized masks per condition |

Under `greedy`, every PACKING prune built the same full Local just to run VERIFY_PACKING. That made v2-greedy
about 4x slower per node than the frozen v1 checker on the identical tree; after 4c225bf/7dbafd0 the gap is 1.45x,
mostly VERIFY_PACKING itself.

The task loop also waited at a barrier after each batch of 4096 tasks. The top tree (and FULL transcripts) ran
decide and VERIFY on the single reading thread. On 452-115-18 c13 that serial part took 124 of 295 s at 8 threads.

## Changes

| commit | change |
|---|---|
| 141e6a3 | S2: exact membership through a kappa fingerprint filter (kappa vanishes on W, so a member always passes); flat echelon basis |
| a381b09 | VERIFY: F(g) as a mask over packed variables, s(g) from lab on demand, derivation steps scan only owners of supp(g) |
| 6b1a5e1 | S2 decision and VERIFY_S2: fingerprint pairs through a hash table instead of a sort |
| b6ac375 | tasks dispatched as they are read (OpenMP tasks, bounded backlog), no batch barrier; earliest failure in stream order reported |
| 9591917 | top tree and FULL transcripts: each ENTER record is a claim, the walk proceeds on it, decide+VERIFY run in the team on a copy of the entry state and must reproduce the claim |
| 6a5447c | decider flip sets from the instance columns, no per-value lists |
| 718366c | VERIFY_S1: a variable's killer list is checked once per node |
| 49a62ef, 6999afe | S1 decision: ANCH, Q* and test 6.2 without sets, sorts or n-sized masks per condition |
| 8398529, 3ee5a45 | VERIFY_S1: ANCH from per-qubit anchor labels; KILL reasons tested against E's column |
| a6c2d75, dd77d11 | packing candidates visited in place; S2 fingerprints of a qubit's three labels in one column pass |
| 1591af6 | backlog below LLVM libomp's inline-execution threshold |
| c151b73, 730d314 | S2 basis in leading-bit echelon form; tagged fingerprint table slots |
| 4c225bf | rule 2: VERIFY_PACKING on its own (one pass over the listed rows, n-bit set on the stack), no Local |
| 7dbafd0 | rule set as an enum; greedy packings verified straight from the decision; branch generator from the active bitset |
| 6df319a | the reading thread gets its own team slot, so `--threads T` replays with T workers (T + 1 native threads for T > 1; allow for it in CPU limits) |
| 405caeb, a7f64f5 | VERIFY_S2: all labels of a qubit fingerprinted in one column pass; sigma(P_v) from the fixed qubits' columns |
| 7fdd699, 7d355fa, c96595e | J_S1: first-killer lists shared by the KILL reasons; reasons kept as a sorted vector (FlatMap), merged lookup in VERIFY |
| 3ea36f2, 793b78e, 4039328 | S1 decision: flip bits from the column entry in hand; K(c) enumerated, not stored; test 6.2 marks indexed by generator |
| fd21304, 0e10ade, d11b914 | reserved value/action lists; S2 arrays filled by index; active list from the bitset, packing on the stack |
| cdd0047 | task replay recomputes only the branch generator at its ancestors (their full decisions are checked in the top walk) |

## Soundness of the deferred decisions (9591917)

The trusted base is unchanged: dead tests, the entry fingerprint, weight and active count, child enumeration,
dead-suffix cuts, frontier cuts and all count/trace bookkeeping still run inline on the reading thread. Only the
recomputation of the node's rule, PK or branch generator (and the VERIFY inside it) moves to a task. That task runs
`decide()` on an exact copy of the entry state and compares rule, packing and generator with the record. The
OpenMP region ends only after every task. A verdict is printed only if no check failed, so an accepted certificate
had every node decided and verified exactly as before. If a claim is wrong, the walk may follow it into records that
do not match the true tree. Every such path ends in a reject, because the claim's own check fails. The walker
range-checks a claimed generator before using it.

The earliest failure in stream order is reported, so the verdict and message do not depend on scheduling. All 24
exported invalid certificates give the same rejection message as `24ad05e` at 1, 4 and 16 threads.

## Validation per commit

Per-commit validation used the following checks. The completed release results
are summarised in [PUBLIC_VALIDATION.md](PUBLIC_VALIDATION.md):
- test-v2, GCC UBSan + wrong-J, Clang ASan+UBSan with forced-zero kappa;
- the 24,000-node experiment (0 unsound) and the 24,000-node comparison with a `24ad05e` driver (0 differences);
- the exported set: 21 ACCEPT with exact counts and traces, 24 REJECT, v1 pairs;
- frozen v1 hashes;
- a large-n stratum (`n` = 1365, 1366, 2048, 4095, 4096) of random DFS descents, compared with a `24ad05e` driver;
- the medium certificates (traces = MANIFEST);
- the independent review adversarial J harness, run against `24ad05e` and the commit on the same J.

## Throughput (shared CCD, so wall times are indicative; instructions:u is the A/B metric)

1-thread benchmark, 488-123-22 c16 s0s1s2, 3000 sampled tasks (130,951 nodes): 608 G user instructions and 53.1 s at
`24ad05e`, 27.2 G at `d11b914`.

Whole checks, 1 thread, at `cdd0047` (core shared with another job's SMT sibling): 452-115-18 c13 114.5 s
(663 G instructions), 488-123-22 c16 284 s (1361 G). At `24ad05e` the 488 check would take about 2,900 s at 1
thread (410 us per node on the sample). For comparison, the producer's count-only 1-thread runs take 37.6 s and
65-89 s.

These measurements do not establish scaling on otherwise idle cores. The
completed release timings in [README.md](../README.md) also use shared CPU
resources and should not be interpreted as isolated scaling results.
