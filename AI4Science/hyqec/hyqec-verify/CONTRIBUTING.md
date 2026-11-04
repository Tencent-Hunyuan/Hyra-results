# Contributing

## Soundness rules (non-negotiable)

* Every lower bound must come from a search or proof whose rules are written down in
  `docs/SOUNDNESS.md`; engines report evidence, `hyqec_verify/core.py` checks witnesses.
* **New pruning** (bounds, caches, symmetry reductions, heuristics that skip work) must pass the
  differential cut test before it can be enabled for certificates or default runs: every subtree the
  rule cuts is re-explored without the rule under the same cap and must contain no nontrivial logical
  of weight <= cap (`tests/native/test_pruning.py::test_differential_every_cut_on_2000_random_codes`, documented in
`tests/native/PRUNING_AUDIT.md`;
  its self-test injects an unsafe cut and must fail). The audit must cover every buffer-size regime of every
  build variant the rule is compiled into: every fixed-size table, touched list or cache needs a bound and
  cleanup argument, every reachable capacity limit must be crossed at full size in each variant (counted and
  asserted), the small-code audit must also pass with all tables shrunk, and between-call state must be
  checked after every rule call (docs/SOUNDNESS.md section 5 *Adding pruning*; for S0/S1/S2
  `tests/native/port/SCRATCH_AUDIT.md`). Agreement with reference distances alone is **not** sufficient
  (see `tests/inputs/lifted_cache_counterexample.json`).
* Changes to the CNF encoder (`engines/sat.py`) need the exact semantic test (`tests/test_sat.py`)
  extended, and change the encoding identifier if the CNF bytes change.
* Changes to the B&B search, instance or certificate format must keep CPU and GPU node counts and
  coverage record bytes identical and update `docs/BNB_CERT_SPEC.md` (and the native interface notes) by agreement.

## Tests

```sh
pip install -e ".[test]"
hyqec-verify build && hyqec-verify tools install
pytest -q -m "not gpu and not slow"      # GPU tests: pytest -m gpu on a CUDA machine
```

Long experiments must run from a pinned snapshot (`git archive` into a fresh venv), never from an
editable checkout that is being edited.
