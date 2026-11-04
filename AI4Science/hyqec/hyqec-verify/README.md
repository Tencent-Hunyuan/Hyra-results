# hyqec-verify

**Certified distances of quantum stabilizer codes, general (non-CSS) and CSS.**

hyqec-verify finds light logical operators (upper bounds) and *proves* that nothing
lighter exists (lower bounds), for any stabilizer code given by Pauli
generators. Its main use is the exact Pauli-weight distance of non-CSS codes, where
the usual tools give only heuristic upper bounds. Every result is
backed by evidence you can re-check:

| evidence | proves | how to re-check |
|---|---|---|
| witness | `d <= w` | `hyqec-verify witness` / `hyqec-verify verify` (linear algebra, milliseconds) |
| SAT + DRAT proof | `d >= W+1` | `hyqec-verify verify` regenerates the CNF from the code and re-runs the bundled `drat-trim` (optionally LRAT + `lrat-check`) |
| branch-and-bound run | `d >= W+1` (or exact d) | with `--bnb-cert`: a coverage certificate (docs/BNB_CERT_SPEC.md) that an independent checker re-executes (`hyqec-verify verify --bnb-checker CMD`); without it, `hyqec-verify verify --rerun-bnb` re-executes the search and compares node counts |
| brute force | exact d for small codes | re-run |

What exactly is proved, and what has to be trusted for each kind of evidence:
[docs/SOUNDNESS.md](docs/SOUNDNESS.md).

## Names and file formats

hyqec-verify was developed under the working name *stabcert*; the distribution, import package and
command were renamed for release (`hyqec-verify`, `hyqec_verify`, `hyqec-verify`). **File formats keep
their historical `stabcert-*` ids** (`stabcert-certificate/1`, `stabcert-pauli-cnf/1`, `stabcert-group/1`,
`stabcert-bnb-instance 1`, `stabcert-bnb-coverage/1` / `SBCOV1`, the checkpoint header `stabcert-bnb-ckpt 2`,
the input format id `stabcert-json` and the certificate field `stabcert_version`), and so do the native
engine names (`stabcert-bnb`, `stabcert-ris`, ...) and their compile-time identifiers: they are hashed
into existing certificates, coverage files, checkpoints and the bundles. Version-2 checkpoints
add integrity checks in 0.2.0 and refuse version-1 files. Environment variables are
`HYQEC_*`; the old `STABCERT_*` names still work as deprecated aliases in 0.1.x and 0.2.x (with a warning), and
the cache moved to `~/.cache/hyqec-verify`. Historical records (development run logs, the bundles,
imported docs) use the old name.

## Results

Measured during development; the run records, the benchmark and applicability-study data and the release
bundles are not part of this package (the companion checker `hyqec-check` is a separate package).

* **Independent reference suite** (a separate reference corpus, n <= 63): 300 cases, i.e. 270 reference presentations
  (233 with an independent-oracle distance; 37 with only a reference interval: 13 oracle bounds and 24
  board claims, each with a checked witness) plus 30 adversarial inputs. Some presentations repeat
  across subsets (289 distinct ids), so these are not 270 inequivalent codes. Each engine alone
  (`--brute-limit 0`): SAT 263 PASS / 37 COMPATIBLE / 0 FAIL; branch-and-bound the same. PASS = the
  233 oracle distances reproduced exactly and the 30 adversarial inputs handled as expected;
  COMPATIBLE = the result lies in the reference interval (interval consistency only).
* **Generality:** of the 184 non-CSS codes on the qldpc-challenge board, the engines alone (no input
  witness, no brute force) give the exact distance for 151 on CPU (300 s, 8 threads) and 13 more
  with GPU branch-and-bound (30 min, 1 GPU): 164 / 184.
* **Independently checked distance certificates** (B&B coverage certificates re-executed by a
  separate implementation; the certificates are distributed separately as release bundles):
  [[452,115,18]] (board entry), [[488,123,22]] (our construction), and Lee et al.'s [[452,115,18]]
  plain fold o04829 (an independent re-certification of the d=18 value Lee et al. list for this instance). [[336,10,28]]: exact per a count-only
  prover run, no certificate.
* **Finding is fast, proving is the work:** minimum-weight logicals of these codes are found in
  1-15 s on 16 CPU threads; proving nothing lighter exists takes minutes to hours, e.g. with the
  public CLI and shipped defaults on 6 H20 GPUs (count-only runs): [[452,115,18]] cap 17 in 120 s,
  [[488,123,22]] cap 21 in 2,517 s.
* **Head-to-head with dist-m4ri v0.14.7** (same matrix, target and symmetry; 64 + 64 threads on
  disjoint cores at the same time): on the board's [[452,115,18]] hyqec-verify proves "no logical <= 17"
  in 1,234 s, while dist-m4ri certifies only <= 13 within its 2 h cap; on [[488,123,22]] at cap 13,
  hyqec-verify 0.48 s vs dist-m4ri 876 s. On small dense codes and CSS one-side searches dist-m4ri is faster: of the eight
  codetables codes measured, by 1.2x-109x on seven (the largest ratio, [[49,17,7]], has dist-m4ri at 0.07 s vs
  hyqec-verify 7.61 s; e.g. [[50,13,8]] at cap 7: dist-m4ri 0.84 s vs hyqec-verify 86 s), and [[41,2,11]] is a tie. Details and caveats:
  [docs/benchmarks.md](docs/benchmarks.md); B&B ablation: [docs/ablation.md](docs/ablation.md); prior art: [docs/related_work.md](docs/related_work.md).
* SAT certificates of an earlier tool reproduced with hyqec-verify's own encoder and checker: 6/6.

## Install

```sh
pip install .            # pure Python + NumPy; no compiler needed at install time
hyqec-verify tools install   # optional: builds pinned Kissat and CaDiCaL (needs git, make, a C compiler)
hyqec-verify build           # optional: compiles native CPU engines (C++ compiler; B&B also needs zlib)
hyqec-verify build bnb-cuda ris-cuda --cuda-arch native   # optional: GPU engines (needs nvcc)
hyqec-verify doctor          # what is available
```

Python >= 3.10. The DRAT/LRAT proof checkers (drat-trim, lrat-check; MIT) are
bundled as source and compiled on first use with the system C compiler. Native
engines are compiled on first use into `~/.cache/hyqec-verify` (override with
`HYQEC_CACHE`; set `HYQEC_NO_BUILD=1` to forbid compiling). B&B CPU and CUDA builds
require zlib headers and library (`zlib1g-dev` on Debian/Ubuntu, `zlib-devel` on Fedora,
or `conda install zlib`). `doctor` reports missing native build dependencies when no
matching binary is cached; `build` gives the same installation guidance. Installation
and import do not run these compiler probes.

## Quick start

```sh
hyqec-verify info code.json
hyqec-verify distance code.json --engines bnb --budget 3600 -o cert.json      # exact d if it fits the budget
hyqec-verify distance code.json --engines bnb --cross-check sat -o cert.json  # two independent proofs
hyqec-verify distance css.json --side X                                        # CSS X-distance
hyqec-verify verify cert.json --rerun-bnb                                      # re-check everything, re-running B&B
# without --rerun-bnb (or --bnb-checker for a coverage certificate) a B&B lower bound is withheld:
# verify reports the bounds it could check, claims_reproduced=false, exit 3
hyqec-verify distance code.json --engines bnb --bnb-cert -o cert.json         # B&B with a coverage certificate
hyqec-verify verify cert.json --bnb-checker auto --threads 64     # independent check with hyqec-check
# --bnb-checker also takes a checker executable, a checker source directory, or a command template:
#   "hyqec-check verify {file} --threads 64 --expected-group {group} --expected-domain {domain} \
#    --expected-cap {cap} --expected-instance {instance}"
hyqec-verify lower code.json --engine bnb --bound 17 --bnb-cert -o cert.json  # one bound + the input file's witness
hyqec-verify wrap-coverage code.json run.cov.gz -o cert.json                  # coverage made by the native engine directly
hyqec-verify lower code.json --bound 9 --engine sat --cubes 40 --sat-workers 32   # prove d >= 10
hyqec-verify upper code.json --engine gpu --seconds 600                        # GPU upper-bound search
hyqec-verify witness code.json --witness '{"X": [0, 3], "Z": [3, 7]}'
```

```python
import hyqec_verify as sc

lc = sc.load("code.json")           # LoadedCode: .code, claims (k, d, witnesses) kept separately
code = lc.code                       # StabilizerCode: n, k, logicals, is_logical(), group_hash(), ...
res = sc.distance(code, engines=("bnb", "sat"), budget=600)
print(res.status, res.lower, res.upper)
```

**What `hyqec-verify distance` does, in order** (every step's evidence is recorded, and the
output names its source: `lower_bound_by`, `upper_bound_by`):

1. verified witnesses in the input file become upper bounds (`--no-input-witness` turns this off);
2. an upper-bound search (random information sets): native CPU until it stalls (3 s without a lighter
   logical) or, with `--gpu`, for at most 5 s; then with `--gpu` the GPU for the rest of the time (without a GPU the CPU continues with
   new seeds); it stops early only on reaching the target; Python if nothing native is built;
3. if exhaustive search needs at most `--brute-limit` candidates (default 2,000,000), it answers
   the question by itself (`--brute-limit 0` turns this off);
4. the requested `--engines` prove lower bounds from the current lower bound up to upper-1;
5. `--cross-check` engines re-prove the final bound independently.

So on small codes, or with a good input witness, `--engines sat` may never need to run SAT.
If you name `--engines`, `"requested_engines_proved_lower_bound"` is true only when one of them
proved the final lower bound. Otherwise it is false and hyqec-verify prints a warning, including when the
lower bound is only the trivial d >= 1; for an exact d = 1 (nothing to prove) it is false with a note
instead of a warning. To benchmark or test one
engine, use `--brute-limit 0 --no-input-witness`.

Exit codes: `0` result produced, `1` input rejected or usage error, `2`
inconclusive (bounds only), `3` **inconsistency** (an engine contradicts a
verified witness or another engine; never silently resolved). For `hyqec-verify verify`, `3` also means
that the certificate's claimed result was not reproduced by the evidence checks selected; this
includes a valid B&B record verified without `--rerun-bnb` / `--bnb-checker`, whose lower bound is
then withheld rather than trusted.

## Input formats

* qldpc-challenge JSON: `code_type: "stabilizer"` with `checks.S = [{"X": [...], "Z": [...]}, ...]`,
  or `code_type: "CSS"` with `checks.X`, `checks.Z` (lists of supports). Claimed `k` and
  `distance` witnesses are read as claims and checked.
* generator JSON (format id `stabcert-json`): `{"n": 5, "generators": [{"X": [0, 3], "Z": [1, 2]}, ...]}` or
  `"generators": ["XZZXI", "IXZZX", ...]`.
* `.npz`: `S` (m x 2n, rows `(x|z)`), or `A`, `B` (`S = (A|B)`), or `HX`, `HZ`.
* text: one Pauli per line, dense (`XZZXI`) or sparse/stim-like (`X0 Z1 Z2 X3`, `X0*Z1`).

Generators may be dependent. Phases are ignored; phase keys are rejected.

## Engines

* **Upper bounds**: random information sets with qubit-aligned elimination (a Y
  costs one qubit, not two coordinates), in pure Python (always available),
  native C++/OpenMP, and CUDA (with a Markov chain over information sets).
  Every reported operator is re-verified.
* **Branch-and-bound** (C++/OpenMP, optional CUDA executor): exhaustive search
  over irreducible Pauli supports with a disjoint-candidate packing bound and
  symmetry roots from verified qubit automorphisms; checkpoint/resume; optional
  coverage certificates (docs/BNB_CERT_SPEC.md) for an independent checker. The recommended
  checker is **hyqec-check** (formerly stabcert-bnbcheck), a separate implementation shipped as a
  companion package: `hyqec-check` for generator weight <= 32 and column degree <= 64, `hyqec-check-wide`
  for the wide CPU builds, and (hyqec-check >= 2.0.0) `hyqec-check-v2` for coverage version 2, including the
  S0/S1/S2 rule sets; `--bnb-checker auto` picks the right one. hyqec-check 2.0 implements BNB_CERT_SPEC_v2
  2.0-rc1 independently of the prover; its independent review found no blocker and no should-fix. The checker (then stabcert-bnbcheck
  v1 / v1.1.0) was reviewed independently: no soundness issue. The reviewer replayed the production
  [[452,115,18]] cap-17 certificate in full (2,282,025 tasks; accepted) and only sampled
  [[488,123,22]] cap 21 (3,000 tasks) and Lee et al.'s o04829 cap 17 (1,000 tasks), 0 mismatches; the
  full 488 and o04829 acceptances are the checker author's complete runs with the same reviewed binary. The released hyqec-check 1.2.0/1.3.0 differ from those reviewed
  binaries by the review's robustness changes and the rename; they passed release acceptance, including a
  full re-run of the 452 cap-17 certificate; a delta review of 1.2.0/1.3.0
  (independent delta review) found no blocker and no new finding, confirmed the
  earlier robustness findings S1, N1, N2 and N4 fixed, and re-ran its harnesses on the released wheel with
  the same outcomes.
  hyqec-verify accepts a coverage bound only if the checker exits 0 and reports `certificate_verified: true`.
  The search algorithm is not new: it follows the irreducible-cluster principle of
  Dumer, Kovalev and Pryadko (IEEE Trans. Inf. Theory 63, 4675 (2017), arXiv:1611.07164)
  and is essentially the general-stabilizer branch-and-bound of Lee et al.,
  arXiv:2609.30069, Appendix B.2 (branch on an anticommuting check, forbid earlier
  choices, stop at zero syndrome, disjoint-candidate bound). What hyqec-verify adds is
  engineering and evidence: verified symmetry, CPU and GPU executors that explore the
  identical tree on runs that find no logical (a find lowers the shared cap, so find-mode work
  depends on the schedule), checkpointing, coverage certificates checked by an independent
  implementation, and SAT/DRAT cross-checks.
* **Pruning policies** (opt-in): `lower --packing greedy+s0 | greedy+s0s1 | greedy+s0s1s2` adds the
  parity rules S0/S1/S2 (docs/SOUNDNESS.md section 5.1) to the default `greedy` packing bound. They run
  only in a separate CPU rules build (no GPU). With `--bnb-cert` they write a **coverage certificate
  version 2** (docs/BNB_CERT_SPEC_v2.md, revision 2.0-rc1), which reaches the *independently checked*
  level only when a v2 checker, normally the companion **hyqec-check >= 2.0.0** (`hyqec-check-v2`), accepts it; `verify` binds
  the instance, domain, cap, rule set and spec revision of the header and of the checker's verdict to the
  certificate (docs/SOUNDNESS.md 5.2). Without that checker the record counts only through
  `verify --rerun-bnb` (same prover) and is labelled not independently checked. The default policy, its
  search tree, node counts and version-1 certificates are unchanged.

  ```sh
  hyqec-verify lower code.json --engine bnb --bound 17 --packing greedy+s0s1s2 --bnb-cert -o cert.json
  pip install 'hyqec-check>=2.0.0'   # companion checker (provides hyqec-check and hyqec-check-v2)
  hyqec-verify verify cert.json --bnb-checker auto --threads 16
  # -> "lower bound by bnb coverage certificate v2 (rules s0s1s2, spec 2.0-rc1), policy greedy+s0s1s2,
  #     independently checked by hyqec-check-v2 (reported version 2.0.1)"
  ```
* **Where B&B is the right tool.** In an applicability study (measured 2026-10-27 on 48 codes, 2 threads, with
  the default `greedy` search of a pre-release build; its tree and node counts are unchanged in 0.3.0), each +1 of the proved cap grows the search tree by x2.67-x6.51
  on 29 sparse codes (surface/toric, colour, hyperbolic, hypergraph-product, BB and board non-CSS codes) but
  by x7.56-x16.68 (median x11.04) on 9 small dense codetables codes (n <= 53), where the packing bound saves a
  median 1.06x. Of the eight codetables codes measured head-to-head, dist-m4ri is 1.2x-109x faster on seven;
  [[41,2,11]] is a tie (per-row results:
  [docs/benchmarks.md](docs/benchmarks.md)). Use B&B for sparse codes; for small dense codes another tool is faster.
* **SAT/DRAT**: a small in-package CNF encoder (commutation XORs, nontriviality
  against a logical basis, sequential-counter weight bound, optional orbit
  roots), Kissat or CaDiCaL, every UNSAT checked by drat-trim; cube-and-conquer
  with a checked coverage proof for parallel runs.
* **Brute force**: exhaustive enumeration for small codes.

## Reproducing the benchmarks

Head-to-head against other exact tools: [docs/benchmarks.md](docs/benchmarks.md) (raw data and
scripts in a separate benchmark repository). Native micro-benchmarks (GPU RIS vs the CPU RIS of the
qldpc-challenge verifier, B&B CPU vs GPU node rates) are kept in the development repository, not in this package.

## License

Apache-2.0 (see LICENSE). Bundled third-party source: drat-trim and lrat-check
(MIT, see NOTICE). hyqec-verify reads the code JSON format of the Unitary Foundation
qldpc-challenge repository; no code from that repository is included.
