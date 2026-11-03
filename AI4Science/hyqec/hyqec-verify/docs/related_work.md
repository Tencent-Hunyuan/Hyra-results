# Related work: distance finding with `codeDistance` and `dist-m4ri`

> Imported from the separate benchmark/prior-art repository at revision d15b65c
> (2026-10-23). Paths such as `bench/...` and `literature/` refer to that repository, which holds the raw
> results, scripts and run logs; it is not part of this package. Internal references were removed for publication.
> *stabcert* in this document is the development name of hyqec-verify (renamed for release).

**Summary.** Two pieces of prior art matter most.

* **Lee, Okada, Maskara, Kasai and Zhou, arXiv:2609.30069, App. B.** This is the
  algorithm stabcert's B&B uses. It already contains the branching rule, the packing
  bound, the irreducible-support leaf and orbit roots for general stabilizer codes,
  and its pruning bounds are a superset of ours (§3.1).
* **Webster, Jacob and Higgott, arXiv:2603.22532.** A broad survey and benchmark
  (classical codes, CSS and non-CSS stabilizer codes, circuits; about two dozen exact
  and heuristic methods) with the open package `codeDistance`. Its best exact LDPC
  method is the C program `dist-m4ri`. The current dist-m4ri release (v0.14.7, Oct
  2026) is much faster than the version the paper used and is the strongest exact
  competitor we measured.

stabcert does not offer a new search algorithm. It offers evidence and execution:

* a verified witness for every upper bound;
* lower bounds proved either by a DRAT-checked SAT proof or by a deterministic,
  re-runnable B&B run. On runs that find no logical, the CPU and GPU executions visit
  identical trees. A find lowers the shared cap, so find-mode work depends on the
  schedule (SOUNDNESS.md §5);
* symmetry from automorphism groups the package verifies itself;
* checkpointing and a certificate verifier.

None of codeDistance, dist-m4ri or Lee et al.'s artifact emits a lower-bound proof
that a third party can check without re-running the search. On speed the picture is
mixed, and §5 states it plainly:

* dist-m4ri is faster on small dense codes and on CSS single-side distances;
* our B&B is faster on Pauli-weight distances of the LDPC codes we measured.

Section numbers, table numbers and line references below are to the v2 text. Facts
about software were read from the source at the stated revisions.

## 1. What the prior work provides

**Paper (arXiv:2603.22532v2).**

* Benchmarks exact methods: Magma Brouwer–Zimmermann (BZ), Gurobi and SCIP MIP,
  CLISAT MaxSAT, and the `m4riCC` connected-cluster algorithm. It also benchmarks
  heuristic methods: `m4riRW`/QDistRnd random information sets, QDistEvol, BP-OSD,
  and Stim undetectable-error searches. Datasets: codetables.de GF(2) and non-CSS
  codes, hyperbolic surface and colour codes, lifted-product codes, BB codes,
  quantum Tanner codes, and circuits (Table 1; §5).
* Hardware: one 40-core node with 8 GB per core. Exact methods were capped at 8 h,
  heuristics at 10,000 trials (§5.1).
* Non-CSS codes are mapped to binary codes. Methods use a 2-block symplectic
  representation, a 3-block representation `(x|z|x⊕z)`, or the 4-block
  Bravyi–Terhal–Leemhuis representation (§3.2.1–3.2.3). The paper finds both
  accuracy and run time "very sensitive to the choice of block encoding" (§5.3.1,
  Fig. 5).
* "Exact" means a method that terminates with the optimum. In the paper's words:
  "algorithms that are able to certify that a given solution is the global optimum"
  (§4.4). The paper does not discuss proof objects or independent re-checking.
* Recommendations for non-CSS codetables codes: Magma (exact) and QDistRnd
  (heuristic). For BB codes: m4riCC, Gurobi and MIP-SCIP (exact) and QDistEvol
  (heuristic) (Tables 2/30, 3/31).

**Package `codeDistance`** (github.com/m-webster/codeDistancePYPI, revision
`a4afe9c`, 28 May 2026, version 0.0.6; `LICENSE` file MIT, `pyproject.toml` says
"GNUv3").

* Entry points: `codeDistance(H, L, tB, method, params)`, `CSScodeDistance`,
  `circuitDistance`.
* Returns a dict with `d`, an example codeword `L`, trial counts `R`/`T` and a
  progress string (README, "Output Format").
* Open methods implemented in Python: `BZDistMW` (BZ with Gray codes, 2/3/4-block),
  `QDistRndMW`, `QDistEvol`, `MeetMiddleMW`, `connectedClusterMW`, `decoderDist`
  (BP-OSD/LSD), `MIPDist` (OR-Tools, SCIP by default) and `pySATDist`.
* Wrappers for external tools: `dist_m4ri_CC`/`dist_m4ri_RW`, Magma, GAP QDistRnd,
  Qubitserf, a MaxSAT binary, Gurobi and Stim.

**`dist-m4ri`** (github.com/QEC-pages/dist-m4ri; GPL, version 2 or later). This C program
provides the paper's best exact method for LDPC codes (`m4riCC`). Two revisions
matter:

* The paper-era revision (we used `d48d361`, 21 Mar 2026) is single-threaded. When
  stopped early it returns no partial lower bound (§4.3.2: "partial lower bound
  results are not available").
* The current release, v0.14.7 (`bed621e`, 15 Oct 2026), is multithreaded. Its
  default `method=3` runs CC and RW together ("bracketing") and reports a certified
  `dmin` together with `dmax`. It accepts a `start=` column list for codes with a
  known symmetry (README §2, "Restricting the CC search"); the tool does not verify
  that the symmetry exists and prints a warning instead. This release postdates the
  paper and changes its conclusions about m4riCC (§5 below).

**What dist-m4ri's "certified dmin" means.** We read `introduction.md` ("How it works:
CC algorithm") and the source (`cc_walk` in `src/dist_m4ri.c`, `start_CC_recurs` in
`src/dist_cc.c`).

* **What it proves.** Round w enumerates every cluster of w columns that starts at its
  smallest column and is grown through the first unsatisfied check. A round with no
  codeword sets `dmin = w + 1`.
* **Completeness argument.** This is the DKP irreducibility argument: a proper subset
  of a minimum-weight codeword has a nonzero syndrome, and the support of its first
  unsatisfied check contains another position of the codeword. The pruning is a
  syndrome-weight bound (Lee's b1) plus an exact match at the last level. For non-CSS
  codes the input is the 3-block matrix. A minimum-weight nontrivial vector uses at
  most one of a qubit's three columns, so `dmin` bounds the Pauli distance.
* **What a third party gets.** Three integers on stdout (`dmin dmax rw_steps`) and log
  lines on stderr. Codewords can be exported (`outC`), but they only support `dmax`.
  There is no proof object, coverage record or replayable task list for `dmin`; it can
  only be checked by re-running the program.
* **Expert options.** With `start=`, `cbeg/cend` or `noscan`, `dmin` is conditional (a
  `WARNING` is printed). For `start=`, the condition is that the user-supplied
  symmetry exists, which the program does not check.

**Licence.** dist-m4ri is GPL, version 2 or later (`LICENSE`, `CITATION.cff`). stabcert
(Apache-2.0) could call a separately installed `dist_m4ri` binary as an optional
external engine or cross-check through a subprocess, as it already does for Kissat.
It should not vendor or link dist-m4ri's source, since that would put the combined
work under the GPL. This is our reading, not legal advice.

## 2. Capability comparison

| | codeDistance (+ dist-m4ri) | stabcert |
|---|---|---|
| classical linear codes | yes | no |
| syndrome-extraction circuits / Stim DEMs | yes (`circuitDistance`; dist-m4ri reads DEMs) | no |
| CSS X-/Z-distance separately | yes | yes (`--side X/Z`) |
| non-CSS Pauli-weight distance | via 2/3/4-block binary encodings | native Pauli representation: one variable or label per qubit, Y counts once |
| exact methods | BZ (Python; Magma if licensed), CC (dist-m4ri), MIP (SCIP; Gurobi if licensed), MaxSAT | B&B (C++/OpenMP, CUDA), SAT (Kissat/CaDiCaL), brute force |
| heuristic upper bounds | QDistRnd (C, GAP, Python), QDistEvol, BP-OSD/LSD, Stim searches, meet-in-the-middle | qubit-aligned RIS (Python, C++/OpenMP, CUDA with a Markov chain over information sets) |
| biased or weighted error models | some methods (Table 1, "Error Models" column) | no (Pauli weight only) |
| GPU | no | B&B executor and RIS |
| multithreading | dist-m4ri v0.14 (CC+RW); MIP solver threads | B&B (OpenMP), SAT cube-and-conquer workers, RIS |
| symmetry reduction | dist-m4ri `start=` list, supplied by the user and not checked | orbit roots from automorphisms that the core re-verifies; the permutations are stored in the certificate |
| checkpoint / resume of exact runs | no (MIP/Magma return partial bounds at a time limit) | B&B checkpoint bound to instance, cap, split, delta and binary hash (`stabcert lower --checkpoint`) |
| lower bound reported when stopped early | Magma, MIP and SAT: partial bounds (§5.1); current dist-m4ri: `dmin` | yes: the largest proven bound `W+1` of the cap ladder |
| re-checkable lower-bound certificate | **no** | DRAT proof checked by bundled drat-trim, optionally LRAT + lrat-check; B&B: deterministic record re-run by `stabcert verify --rerun-bnb` |
| independent witness check | optional `LOCheck` printout (off by default) | always (core re-verifies every operator); `stabcert witness` |
| cross-check of two engines | no | `--cross-check sat`; any disagreement exits with code 3 |

Notes on the table, from our reading of the source:

* In `codeDistance`, `MIPDist` calls `solver.Solve()` and returns
  `round(Objective().Value())` without reading the solver status
  (`distance.py`, `MIPDist`). A run that hits `maxTime` therefore returns its
  incumbent with nothing that marks it as unproven. The caller has to compare the
  run time with the cap, as the paper does in §5.1.
* The 2-block `BZDistMW` path returns its codeword with the X and Z halves swapped.
  The reported `d` is correct; we checked this on [[30,9,6]] by swapping the halves
  back.
* With `L=None` (the batch driver's default for codetables codes), version 0.0.6
  raises `NameError: getLogicalPaulis` for non-CSS input. The cause is a circular
  import between `distance.py` and `code_library.py`. Our benchmark binds the
  missing names at import time and does not modify their files
  (`bench/cd_child.py`).

## 3. Algorithmic relation

### 3.1 stabcert B&B, the irreducible-cluster family, and Lee et al. App. B.2

**The algorithm is not new.** stabcert's B&B belongs to the irreducible-cluster
branch-and-bound family. That family starts with Dumer, Kovalev and Pryadko [23]: a
minimum-weight codeword has a Tanner-connected, irreducible support, so supports can
be grown through unsatisfied checks. It is the error-enumeration family of the paper
(§4.3.2). The closest published description of what we implement is Appendix B.2 of
Lee, Okada, Maskara, Kasai and Zhou, arXiv:2609.30069 (Sept 2026), "Symmetry-Informed
Branch and Bound for Minimum-Distance Verification", for general stabilizer codes.
Component by component:

| | Lee et al. App. B.2 | stabcert B&B (SOUNDNESS.md §5) |
|---|---|---|
| leaf | zero syndrome, then a nontriviality test; DKP irreducibility | Lemma 1 (irreducible support), the same principle |
| branching | "the unsatisfied check with the fewest available actions", anticommuting labels grouped by qubit, "only the tried actions are forbidden in later branches" | the anticommuting generator with the fewest children; earlier candidates restricted to commuting labels. This is the same partition |
| pruning | b1 = ⌈\|s\|/Δ⌉, b2 (disjoint available sets), b3 (coverage), and the parent-distance rule (61) for halved codes | greedy b2 only, plus "dead" nodes. **A strict subset of theirs** |
| symmetry | cyclic translation roots for CPM lifts | orbit roots of any qubit-permutation group, each permutation verified by the core and stored in the certificate (a modest generalization) |

Adding their b1 and b3 to our engine was measured separately. It gave 1.40–1.46×
fewer nodes but ran 1.45–1.53× slower in wall time on 452 caps 14/15 and 488 cap 16
(measured during development). So we kept the simpler bound.

**How Lee et al. certify a distance.** We read their data repository
(github.com/ultra-high-rate-qec/design-principles-data, revision 306b31d, MIT).

Their record for o04829, a [[452,115,18]] plain fold of [[904,230,20]], lists three
pieces of evidence:

* **CSS parent exclusion.** X side, through weight 18, on n = 904. It used
  quasi-cyclic roots and 4 threads, took 239 s and visited 3.7e8 states, with "greedy
  plus exact component packing" (exact on components of ≤ 12 checks).
* **Folded non-CSS exclusion through Pauli weight 17.** It uses their parent-distance
  rule (61) and further options (action packing, dynamic/family capacity, type
  budget, orbit order). The search is split into a 270-shard prefix partition
  (≤ 1,200 s per shard, 1 thread), with a verified partition replay and coverage files.
* **A weight-18 witness.**

Their record describes the evidence as "not SAT/DRAT proof traces". Their published
(portable) validation layer re-checks algebra, partition coverage, options and receipt
provenance of the shard records, but is, in their words, "not an independent rerun of
the exhaustive search". The CSS verifier source is published
(about 1.8k lines of C++). The solver used for the 452 build (`solver/solver.cpp`)
is not, and the published `sparse_cpm_distance_core.hpp` does not match the hash in
the 452 build record. We found no
GPU code. The code we certify is the qldpc-challenge board entry, a different
[[452,115,18]] instance (a reversing fold of [[904,230,22]]) with the same parameters
as Lee et al.'s o04829, which they list as exact.

**What stabcert adds is engineering and evidence, not a new search:**

* a written soundness proof for the exact variant implemented, including the
  restricted X/Z domains;
* deterministic task splitting with path replay, so the CPU and CUDA executors visit
  identical trees on no-find runs (node counts checked equal; a find lowers the shared
  cap, so find-mode work depends on the schedule);
* checkpoint/resume bound to the instance, cap, split and binary;
* verified symmetry groups;
* an independent SAT/DRAT engine for cross-checks;
* B&B coverage certificates that a separately written checker re-executes task by task.
  The checker is `stabcert-bnbcheck`, built from the certificate specification
  without access to the prover code. For the qldpc-challenge board's [[452,115,18]] at
  cap 17 it accepted all 2,282,025 tasks, 1.246e11 nodes, with 0 mismatches, in
  1,526 s on 128 threads.
  Lee et al.'s published validation layer checks partition coverage and provenance
  of their shard records but does not re-run the search.

**Comparison with dist-m4ri's connected-cluster implementation** (v0.14.7, `cc_walk`
in `src/dist_m4ri.c`, which is what codeDistance's `m4riCC` calls):

| | dist-m4ri CC | stabcert B&B |
|---|---|---|
| object | binary column sets; non-CSS codes enter as the 3n-column 3-block matrix (one column per single-qubit X, Y or Z error; codeDistance's `encode2block(.., tB=3)`) | Pauli labels per qubit (decided / restricted to `{I,P}` / free); Y is one label |
| branching | every column of the first unsatisfied row that is larger than the start column; siblings are not excluded in later branches, so a cluster can be reached along several extension orders | the B.2 partition above: each operator lies in at most one subtree |
| Y on a qubit | three columns; a cluster may hold both X_q and Z_q (cost 2) where Y_q costs 1 | at most one label per qubit |
| pruning | syndrome weight ≤ (remaining weight) × (max column weight), i.e. b1; exact last-level match by hash | b2 (greedy disjoint packing) |
| symmetry | `start=` columns asserted by the user; dmin is valid "under a code symmetry only" | verified orbit roots |
| parallelism | threads over start columns or work units; CC shares threads with RW | split tree; OpenMP or CUDA executor, identical trees on no-find runs |
| stopped early | `dmin` = last completed round + 1 | largest completed cap of the ladder; checkpoint |

The benchmark (§5) shows that dist-m4ri's bitset-and-hash implementation is much
faster than our B&B on CSS LDPC codes and on small dense codes. We have not isolated
which of the differences above accounts for which part of the gap.

### 3.2 vs Brouwer–Zimmermann

BZ (§4.2.2) enumerates **codewords** of a generator matrix over several
information sets. It stops when the sum of per-information-set lower bounds meets
the best weight found. For non-CSS codes it runs on a binary image: the 2-block
"pseudo-generator" method of White and Grassl [9], used by Magma, or the 3-block
image, faster in `BZDistMW` (§5.3.1, Fig. 5a; also [10]).

BZ's cost depends on the dimension of the normalizer (`n+k`) and on information-set
coverage. The cost of B&B/CC depends on generator and column weight (sparsity) and
on d. This is why BZ (Magma) wins on the dense, small codetables codes in the paper
(Table 8) and in our runs, and loses on LDPC families (Tables 10–16). stabcert has no
BZ engine.

### 3.3 vs MIP / MaxSAT

The paper's MIP (§4.4.1) and MaxSAT (§4.4.2, App. A) formulations work on the
3-block DEM. The variables are 3n error mechanisms, with integer slack (MIP) or
Tseitin XOR clauses (MaxSAT) for parity, and they *optimize* the weight. Gurobi also
supports a 2-block symplectic-weight objective.

stabcert's SAT engine (SOUNDNESS.md §4) solves *decision* questions "is there a
nontrivial logical of weight ≤ W". It has per-qubit variables `x_q, z_q` and
`u_q = x_q ∨ z_q`, so Y counts once. It uses commutation XORs, nontriviality against
a logical basis, a sequential-counter cardinality bound, and optional orbit-root
clauses. Each UNSAT answer comes with a DRAT proof that the bundled drat-trim checks.
Optimization solvers give no comparable artifact: an optimal MIP or MaxSAT answer
cannot be re-checked without trusting the solver.

## 4. Certificates: who can re-check what

| result | codeDistance / dist-m4ri | Lee et al. artifact (design-principles-data, revision 306b31d) | stabcert |
|---|---|---|---|
| upper bound d ≤ w | an example codeword is returned; checking it is left to the user (`LOCheck` prints three checks) | witness supports stored in the distance records | the witness is verified by the core; `stabcert witness` |
| lower bound d ≥ W+1 | solver or program termination; no artifact | completed exhaustive prefix exclusions split into shards, with receipts, a verified partition replay and coverage files. Their published validation layer re-checks algebra, coverage and provenance of the shard records but, in their words, is "not an independent rerun of the exhaustive search" and the records are "not SAT/DRAT proof traces" | SAT: CNF hash + DRAT proof, re-checked by `stabcert verify` (the CNF is regenerated from the code and compared). B&B: instance, permutations, source and binary hashes, node counts; `--rerun-bnb` re-executes the search. A coverage certificate (task list + traces) is re-executed task by task by the independent checker `stabcert-bnbcheck` |
| trusted for a lower bound | the solver or program (MIP, MaxSAT, BZ, CC implementations) | their solver. The non-CSS solver source is published only as hashes; the CSS verifier source is published | SAT: `core.py`, the CNF encoder (≈150 lines) and drat-trim. B&B: `core.py`, `core.h` and its driver |

A B&B record is reproducible, but it is not a proof object, because trees reach
1e11–1e14 nodes. Checking one means re-running the search, which our independent
checker does task by task. The same holds for Lee et al.'s exclusions, whose
published validation layer does not re-run them, and for dist-m4ri's rounds. Only SAT/DRAT evidence can be checked without re-running a search, and SAT
reaches only part of the range B&B covers. This limitation is stated in SOUNDNESS.md
§5.

## 5. Head-to-head benchmark (details: [benchmarks.md](benchmarks.md))

Every tool got the same matrix, the same target ("no logical of Pauli weight ≤ W"),
the same symmetry and the same threads; the metric is wall time to that certified
bound. Headline rows ran both tools at the same time on disjoint physical cores, 64 +
64 threads.

* **Large non-CSS LDPC codes ([[452,115,18]], [[488,123,22]]).** On these two codes
  stabcert reached the target cap in 1,234 s / 0.48 s, where dist-m4ri reached ≤ 13 in
  2 h / took 876 s. Details:
  * The board's [[452,115,18]] to "≤ 17": stabcert 1,234 s; dist-m4ri v0.14.7 does
    not reach it in 2 h (certifies ≤ 13).
  * [[488,123,22]] to "≤ 13" (same verified symmetry): stabcert 0.48 s vs dist-m4ri
    876 s.
  * On one H20, stabcert proves 452 ≤ 17 in 487 s and 488 ≤ 21 (d ≥ 22) in 4.0 h,
    beyond the 2 h cap.
* **Mid-size Pauli distances.** BB [[144,12,12]] (Pauli): stabcert 0.38 s vs dist-m4ri
  14.6 s. [[100,23,11]]: stabcert 10.6 s vs dist-m4ri 37.8 s.
* **Small dense codetables codes and CSS one-side distances.** dist-m4ri is faster on seven of the
  eight codetables codes measured, by 1.2× to 109× ([[49,17,7]]: 0.07 s vs 7.61 s). Codetables [[41,2,11]] is a tie.
* **codeDistance's own open exact methods** (Python BZ, paper-era m4riCC, SCIP) did
  not finish 452/488 in 1 h. At equal threads (1), their Python BZ is faster than our
  B&B on 6 of 8 finished small codetables codes. With 22 threads the B&B is faster on
  all but [[41,2,11]] ([benchmarks.md](benchmarks.md) §4).
* **Lee et al.'s own [[452,115,18]] (o04829).**
  * Our prover proves ≤ 17 in 1,281 s on one H20 and 324 s on four, without their
    translation roots and without their parent-distance rule (61).
  * Our independent checker re-executed the resulting coverage certificate in
    4,090 s on 128 CPU threads.
  * Their recorded evidence is 270 single-thread shards (≤ 1,200 s each); actual
    shard times are not public.

## 6. What they have that stabcert lacks

* Classical codes and circuit (DEM) distances, including the circuit filtering of
  §5.5.
* Many heuristics, including QDistEvol. On BB, lifted-product and quantum Tanner
  codes it had the best success rate per trial (Tables 15, 17, 19).
* Weighted and biased error models for several methods.
* BZ, which suits dense small codes, and MIP/MaxSAT optimization with any-time
  incumbents.
* A large published benchmark corpus with per-code results (`paper results/*.xlsx`).
* Through current dist-m4ri: the fastest exact search we measured on CSS LDPC codes,
  and an any-time `dmin`/`dmax` bracket on all cores.

## 7. References (BibTeX)

```bibtex
@misc{WebsterJacobHiggott2026,
  author        = {Webster, Mark and Jacob, Abraham and Higgott, Oscar},
  title         = {Distance-Finding Algorithms for Quantum Codes and Circuits},
  year          = {2026},
  eprint        = {2603.22532},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  note          = {Version 2, 14 August 2026}
}
@misc{codeDistance,
  author       = {Webster, Mark and Jacob, Abraham and Higgott, Oscar},
  title        = {{codeDistance}: distance algorithms for classical codes, quantum codes and quantum circuits},
  howpublished = {\url{https://github.com/m-webster/codeDistancePYPI}},
  note         = {Version 0.0.6, revision a4afe9c},
  year         = {2026}
}
@misc{distm4ri,
  author       = {Pryadko, Leonid P. and Zeng, Weilei},
  title        = {{dist-m4ri}: distance of a classical or quantum {CSS} code},
  howpublished = {\url{https://github.com/QEC-pages/dist-m4ri}},
  note         = {Version 0.14.7, 2026-10-15},
  year         = {2026}
}
@article{DumerKovalevPryadko2017,
  author  = {Dumer, Ilya and Kovalev, Alexey A. and Pryadko, Leonid P.},
  title   = {Distance Verification for Classical and Quantum {LDPC} Codes},
  journal = {IEEE Transactions on Information Theory},
  volume  = {63},
  number  = {7},
  pages   = {4675--4686},
  year    = {2017},
  doi     = {10.1109/TIT.2017.2690381},
  eprint  = {1611.07164},
  archivePrefix = {arXiv}
}
@misc{Lee2026Design,
  author        = {Lee, Jong Yeon and Okada, Koki and Maskara, Nishad and Kasai, Kenta and Zhou, Hengyun},
  title         = {Design Principles for Ultra-High-Rate Quantum Codes},
  year          = {2026},
  eprint        = {2609.30069},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  note          = {Appendix B: Symmetry-Informed Branch and Bound for Minimum-Distance Verification}
}
@inproceedings{WhiteGrassl2006,
  author    = {White, Greg and Grassl, Markus},
  title     = {A New Minimum Weight Algorithm for Additive Codes},
  booktitle = {2006 IEEE International Symposium on Information Theory},
  pages     = {1119--1123},
  year      = {2006},
  doi       = {10.1109/ISIT.2006.261958}
}
```
