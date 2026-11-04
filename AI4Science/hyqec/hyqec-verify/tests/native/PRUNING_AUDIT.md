# Pruning acceptance tests

Run the CPU CI test with `pytest tests/native/test_pruning.py -m 'not gpu'`.
It is not marked slow, so the regular CPU CI includes it. A manual GPU job is
defined in `.github/workflows/pruning-gpu.yml`; on an allocated local device,
set `CUDA_VISIBLE_DEVICES` and `HYQEC_TEST_GPU=1`, then run
`pytest tests/native/test_pruning.py::test_s1_reproducers_gpu`.

The audit uses 2,000 deterministic random stabilizer codes, 200 at each size
5–14, with random P/X/Z domains, random caps, random Clifford circuits,
generator recombinations, dependent rows and occasional zero rows. Logical
dimension is sampled from 1–4 so that exhaustive enumeration remains cheap.
Nine additional cases cover the independent S1 reproducers in all domains, and
the preserved review inputs in `fixtures/pruning/review/` (the first 40 trials
of the S1 reviewer's S1 differential and the S0 reviewer's GAC-incompleteness code; hashes in
`provenance.json`) add three cases each, one per domain, at random caps.
Every case runs all shipped policies at its random cap and at cap n, comparing
the minimum with exhaustive binary-algebra enumeration of N(S) minus S.

The instrumented test build calls a hook at each actual WEIGHT, PACKING,
S0, S1, S2, dead-child/root and dead-suffix return, before any domain is
restored. The opt-in parity policies (`greedy+s0`, `greedy+s0s1`,
`greedy+s0s1s2`, docs/SOUNDNESS.md 5.1) are audited like the others and must
show positive `s0` / `s1` / `s2` counters; their node-level review cases are
in `rule_cases.cpp` (`test_parity_rules.py`). They exist only in the CPU
rules build (`-DSTABCERT_RULES`), which the auditor uses for every policy, so
the GPU reproducer job runs only the GPU-capable policies. The
hook checks every discarded domain against independently enumerated logicals.
It then reconstructs all incremental state from those exact domains and
re-searches with packing disabled at the cut's current cap. This baseline
retains structural weight/dead cuts; the independent enumeration checks those
too. Copying the original cached state would hide stale-cache defects, so the
baseline deliberately rebuilds it. No cuts are sampled or capped.

The oracle implements its own binary elimination, stabilizer membership,
commutation nullspace and exhaustive enumeration in `cut_audit.cpp`. It also
checks the supplied logical basis spans the full quotient; it does not use
the prover's candidate, packing or logical predicates for its verdict. The
released input loader and search are the systems under test. Production
builds expand the hook to a no-op and have no audit fields or runtime branch.

Any future policy must be added to `POLICIES` in `test_pruning.py` and the
policy loop in `cut_audit.cpp`, and every new cut site must invoke the hook.
Acceptance requires positive counters for all applicable cut kinds, plus
the counterexample regressions. Set `HYQEC_AUDIT_SANITIZE=1` to compile
the CPU auditor with AddressSanitizer and UndefinedBehaviorSanitizer.

Large-n and buffer-limit strata (`test_parity_rules.py`, all six rules build variants
1024 | 4096 x w32 | w64 | w128; details and the limit table in `port/SCRATCH_AUDIT.md`):

- `test_planted_stratum_*`: node level. Planted completions make any prune a false prune; every node is also
  evaluated on a reused and a fresh state, and the scratch is checked after every call. The `slow` version
  requires every reachable capacity limit of the variant to be hit at full size.
- `test_cut_audit_large_every_variant` (`slow`): `cut_audit.cpp --large` at n just above 2048 and up to the
  variant's size limit. Every cut is re-searched without the rule, and distances must equal the reviewed greedy
  search.
- `test_large_n_rule_state_reuse` and `test_wide_s1_filter_cleanup_regression`: review finding 01 and its
  boundary sweep.
- The 2132-case audit runs twice, at default capacities and with every table shrunk (`CAPACITY_FLAGS`).

These are implementation checks over finite inputs, not a proof that a new
rule is sound. A new rule also needs its own mathematical justification and
independent review before certificate use.
