# Head-to-head benchmark: stabcert vs dist-m4ri / codeDistance (final, 2026-10-23)

> Imported from the separate benchmark/prior-art repository at revision d15b65c
> (2026-10-23); generated there by `bench/doc/make_benchmark_md.py` from `bench/h2h/results.jsonl`. Paths such as `bench/...` and `literature/` refer to that repository, which holds the raw
> results, scripts and run logs; it is not part of this package. Internal references were removed for publication.
> *stabcert* in this document is the development name of hyqec-verify (renamed for release).

**Summary.** Every tool got the same matrix, the same target ("prove there is no
nontrivial logical of Pauli weight ≤ W"), the same symmetry information and the same
number of threads; the measure is wall time to that certified bound. The headline rows
ran **both tools at the same time on disjoint physical cores** (64 + 64 threads, idle
SMT siblings).

* **Large non-CSS LDPC codes ([[452,115,18]], [[488,123,22]]).** On these two codes
  stabcert reached the target cap in 1,234 s / 0.48 s, where dist-m4ri reached ≤ 13
  in 2 h / took 876 s. Details:
  * On the qldpc-challenge board's [[452,115,18]], stabcert proves "no logical ≤ 17"
    in **1,234 s** on 64 threads. Running at the same time on its own 64
    cores, dist-m4ri v0.14.7 (Oct 2026; the strongest exact competitor we measured, much faster than the
    version in the codeDistance paper) does **not reach it within its 2 h cap**: it certifies only
    ≤ 13. Its last round (w = 13) alone took
    4,394 s, and each round costs about 8× the previous
    one.
  * On [[488,123,22]] at the cap both can finish (≤ 13, same verified 61-cycle
    symmetry): 0.48 s vs 876 s.
  * On one H20 GPU, stabcert proves 452 ≤ 17 in 487 s, and 488 ≤ 20 in
    2,652 s and ≤ 21 (d ≥ 22) in 4.0 h.
    The 488 cap-21 run exceeds the 2 h head-to-head cap.
* **Mid-size Pauli-distance instances.** BB [[144,12,12]] as a Pauli
  distance: stabcert 0.38 s vs dist-m4ri 14.6 s. [[100,23,11]]: stabcert 10.6 s vs
  dist-m4ri 37.8 s (22 threads,
  Appendix A).
* **Small dense codes and CSS one-side searches.** dist-m4ri is faster on seven
  of the eight codetables codes measured, by 1.2× to 109× (22 threads each; the largest ratio, [[49,17,7]], has dist-m4ri at 0.07 s vs
  stabcert 7.61 s).
  For example, codetables [[50,13,8]] at cap 7: dist-m4ri 0.84 s vs stabcert 86 s,
  and BB144 Z side: dist-m4ri 0.04 s vs stabcert 0.23 s. Codetables [[41,2,11]] is a
  tie: dist-m4ri 3,852 s vs stabcert 3,778 s.
* **codeDistance's own exact methods** (Python BZ, paper-era m4riCC, SCIP MIP) did
  not finish any 452/488 run in 1 h. Their Python BZ against our B&B on small dense
  codes is compared per row in §4.
* **Lee et al.'s own [[452,115,18]] (o04829).**
  * Our prover: ≤ 17 in 1,281 s on 1 H20, and in 324 s on 4 H20
    (the coverage production run).
  * The independent checker re-executed all 254,720 tasks (299,674,975,243 nodes,
    0 mismatches) in 4,090 s on 128 CPU threads. That is a CPU cost for the same tree,
    but it is a checker run, not our prover.
  * Caveats are on the row (§2).

Speed is not our only difference. Every stabcert lower bound comes with a re-checkable
record, and none of the other tools produces one ([related_work.md](related_work.md) §4).

## 1. Setup (fairness rules)

* **Machine.** 2× AMD EPYC 9K84 (2×96 cores, 384 threads), 2.2 TB RAM, 8× NVIDIA H20
  (driver 535.247.01, CUDA 12.9). The machine was shared, with a 1-minute load average
  of about 100–310 on 384 threads.
  * CPU runs are pinned with `taskset` to physical cores. On this machine logical CPU c
    and c+192 are the two hyperthreads of one physical core.
  * **Headline rows (§2, tag `conc64`).** Both tools ran at the same time, 64 threads
    each, on disjoint physical cores: dist-m4ri on 0–63, stabcert on 96–159. The SMT
    siblings of those cores (192–255, 288–351) were idle (1–8% busy). All other users'
    jobs were pinned to 64–95, 160–191, 256–287 and 352–383 from 02:19 on. Load,
    affinity and per-core busy % were snapshotted at each pair's start and end
    (`bench/h2h/snapshots.jsonl`).
  * **Second table (§3, tag `conc`).** Concurrent 22 + 22 threads on 128–149 /
    152–173, run while other workers' jobs were not pinned away from our sibling
    threads.
  * **Appendix A.** Rows that are not concurrent pairs (earlier exploration, cap
    ladders, single-tool rows). They are labelled by their overlap value, and both
    tools' cores and load are listed.
  * A first 64+64 attempt on 452 (02:10) is not used. Another worker's 128-thread job
    sat on the SMT siblings of all dist-m4ri cores from 02:12:04 to 02:16:19, against
    only half of stabcert's. It was stopped at 820 s and is kept in
    `bench/h2h/results-cancelled.jsonl`.
  * GPU runs used one H20 (GPU 7) with 4 host threads.
* **Same matrix.** Every row records the instance JSON sha256 and the stabilizer group
  hash.
  * stabcert reads the generator JSON. dist-m4ri gets the 3-block matrices
    [Sz|Sx|Sx⊕Sz] of the same generators and of the logical basis computed by the
    stabcert core. This is exactly how codeDistance's `dist_m4ri_CC` wrapper builds
    them (`encode2block(.., tB=3)`).
  * CSS Z side: HX and HZ (`finH`/`finG`).
  * Exception (Appendix A only): the first, non-concurrent 64-thread dist-m4ri run on
    452 was started before `h2h.py` existed. Its matrices were built by codeDistance's
    own helpers from the same generators (noted in its record).
* **Same target.** dist-m4ri `method=2` (exhaustive connected-cluster rounds only; its
  random-window threads cannot raise a lower bound) with `dstop=W+1`, so it stops
  once `dmin ≥ W+1`. stabcert runs its B&B at cap W (`engines.bnb.run_bnb`). "Reached"
  means the tool reported the certified bound itself.
* **Same symmetry.** Either none for both, or the same verified automorphism group for
  both: stabcert as orbit roots; dist-m4ri as `start=` = the 3-block columns of every
  orbit representative. With `start=`, dist-m4ri prints its own warning that `dmin` is
  then valid only under the symmetry. 452 has no qubit automorphisms, so it is run
  without symmetry by both tools.
* **Same threads** within a row: 22, 16 or 64, stated per row. GPU runs are a
  separate column.
* **Time.** Every headline time is **wall** time (`wall_s` in the records).
  * For dist-m4ri the sum of its own CC round times is shown alongside.
  * For stabcert the engine time (`sec` of the native `done` line) is in the raw
    record.
  * Preprocessing (loading, logical basis, symmetry detection, root order) is
    recorded separately in `prep`, outside the headline wall times.
    Figures are computed from the same `bench/h2h/results.jsonl` (81 rows) as the tables in
    this document:
    * dist-m4ri prep (load + core logical basis + symmetry + writing its .mtx matrices):
      * all 31 rows: median 0.77 s, max 6.0 s (`m4ri-ct-30-9-6-P-cap5-symnone-t22`);
      * `conc64` + `conc` 6 rows: median 1.61 s, max 2.8 s (`m4ri-nc-452-115-18-P-cap17-symnone-t22-conc`);
    * stabcert prep (load + logical basis + symmetry + root order):
      * all 49 rows: median 0.18 s, max 4.5 s (`bnb-nc-452-115-18-P-cap14-symnone-t16`);
      * `conc64` + `conc` 6 rows: median 0.37 s, max 2.3 s (`bnb-nc-452-115-18-P-cap17-symnone-t22-conc`);
  * One-time native builds are excluded.
* **Caps.** 2 h per (tool, instance, cap). The single exception is labelled.
* **Versions.**
  * dist-m4ri master `bed621e` (v0.14.7, binary sha256 in records).
  * stabcert `d598657` for the small rows; `abc1def` (same tree: identical node
    counts on ct-30) for the 452, 488 and o04829 rows. Each row records the
    revision.
  * codeDistance `a4afe9c` (v0.0.6) with the paper-era dist-m4ri `d48d361`.
* **Raw data.**
  * `bench/h2h/results.jsonl`: one JSON per run, with commands, versions, load,
    matrix hashes and the tail of the log.
  * Per-run directories are in `bench/h2h/runs/`.
  * `bench/h2h/table.py` regenerates §3.
  * The codeDistance and 1 h-cap batch records are in `bench/results/*.json`;
    `bench/summarize.py` regenerates §4.

## 2. Headline rows: 64 + 64 threads, concurrent, disjoint physical cores (tag `conc64`)

| instance | target | dist-m4ri master, 64 thr (wall) | stabcert B&B, 64 thr (wall) | stabcert B&B, 1× H20 (wall) |
|---|---|---|---|---|
| [[452,115,18]] board (no automorphisms; neither tool uses symmetry) | no logical ≤ 17 | **not reached in 2 h**: certified ≤ 13 (w=11: 66 s, w=12: 560 s, w=13: 4,394 s) | **1,233.6 s** (124,648,548,270 nodes) | **486.7 s** |
| [[488,123,22]] (same verified 61-cycle for both, 8 orbits) | no logical ≤ 13 | **875.8 s** | **0.48 s** (16,669,230 nodes) | — |
| [[488,123,22]] (same symmetry) | no logical ≤ 21 (d ≥ 22) | not run at 64+64 (dropped). Evidence: the non-concurrent 64-thread run reached only ≤ 13 in 2 h with round 14 ~100% claimed; concurrent 22+22 cap 14 not reached in 2 h (§3) | 64 thr: cap 17 in 58.5 s (5.99e9 nodes, 1.03e8 nodes/s; Appendix A); cap 21 not rerun on CPU (measured earlier: 13,604 s on 192 threads, an earlier internal measurement) | caps 17/18/19/20: 21.7 / 96.1 / 475 / 2,652 s; **cap 21: 14,454 s = 4.0 h** (3,157,706,215,314 nodes = ledger count; *exceeds the 2 h cap*, approved) |
| Lee et al.'s own [[452,115,18]] o04829 (plain fold; trivial qubit-permutation group) | no logical ≤ 17 | — | not run (see the checker cost next) | **1,281 s** (299,674,975,243 nodes); 4× H20: 324 s (coverage production, stabcert 5045cac) |

**o04829 row, read with these caveats:**
* Independent re-execution: the independent checker (`stabcert-bnbcheck`, not our prover)
  re-executed all 254,720 tasks, 299,674,975,243 nodes, 0 mismatches, in 4,090 s on
  128 CPU threads (`stabcert-bnbcheck/results/production-lee-cap17/verdict.json`).
* Different resources: our runs are on GPUs; Lee et al.'s evidence is 270
  single-thread shards with `max_solver_seconds = 1200`, plus a 239 s 4-thread
  parent exclusion.
* Their per-shard times are not public ("complete search evidence is kept outside
  this repository"), so only the bound ≤ 270 × 1,200 s ≈ 90 single-thread hours can be
  derived.
* They used translation/orbit roots and their parent-distance rule (61). We used
  neither: the translation acts on folded qubits as permutation × local Hadamard,
  which the release does not use.

Per-row cores, load at start and overlap for the `conc64` rows:

| instance | side | sym | cap W | tag | threads | dist-m4ri CPU (wall) | stabcert CPU (wall) | stabcert 1× H20 (wall) | cores m4ri / bnb | load at start m4ri / bnb | overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nc-452-115-18 | P | none | 17 | conc64 | 64 | not reached at cap after 7200 s; certified ≤ 13 | **1233.63 s** (engine 1233.53 s, 124,648,548,270 nodes) |  | 0-63 / 96-159 | 216 / 216 | 1.00 |
| nc-488-123-22 | P | cyclic | 13 | conc64 | 64 | **875.84 s** (CC 875.75 s) | **0.48 s** (engine 0.39 s, 16,669,230 nodes) |  | 0-63 / 96-159 | 287 / 287 | 1.00 |

## 3. Second table: 22 + 22 threads, concurrent (tag `conc`)

Both tools started together on disjoint cores (128–149 / 152–173). Other workers'
jobs were not yet pinned away from our sibling threads during these runs; the load is
listed per row. The 488 cap-14 (dist-m4ri) and cap-17 (stabcert) runs were one
concurrent pair.

| instance | side | sym | cap W | tag | threads | dist-m4ri CPU (wall) | stabcert CPU (wall) | stabcert 1× H20 (wall) | cores m4ri / bnb | load at start m4ri / bnb | overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nc-452-115-18 | P | none | 17 | conc | 22 | not reached at cap after 7200 s; certified ≤ 12 | **4071.15 s** (engine 4071.01 s, 124,648,548,270 nodes) |  | 128-149 / 152-173 | 264 / 264 | 1.00 |
| nc-488-123-22 | P | cyclic | 12 | conc | 22 | **280.75 s** (CC 280.63 s) | **0.93 s** (engine 0.76 s, 4,106,478 nodes) |  | 128-149 / 152-173 | 280 / 280 | 1.00 |
| nc-488-123-22 | P | cyclic | 13 | conc | 22 | **2542.43 s** (CC 2542.25 s) | **0.84 s** (engine 0.74 s, 16,669,230 nodes) |  | 128-149 / 152-173 | 271 / 271 | 1.00 |
| nc-488-123-22 | P | cyclic | 14 | conc | 22 | not reached at cap after 7200 s; certified ≤ 13 |  |  | 128-149 / - | 170 / - | - |
| nc-488-123-22 | P | cyclic | 17 | conc | 22 |  | **197.62 s** (engine 197.40 s, 5,986,583,611 nodes) |  | - / 152-173 | - / 170 | - |

## 4. codeDistance's own methods (1 thread, 1 h cap) and reproduction of the paper

From the 1 h-cap batch (`bench/results`). "paper" columns come from the authors'
per-code spreadsheet `paper results/03 codeTables_QECC.xlsx` (40-core node, 8 h cap).
Times exclude Python import and numba JIT (≈ 14 s per process).

| code | paper Magma | paper m4riCC | paper SCIP | BZDistMW | m4riCC (paper-era dist-m4ri) | SCIP MIP | dist-m4ri master, 1 thr |
|---|---|---|---|---|---|---|---|
| [[30,9,6]] | 6 (0 s) | 6 (8 s) | 6 (27 s) | 6 (0.9 s) | 6 (4.0 s) | 6 (15 s) | 6 (0.02 s) |
| [[32,5,8]] | 8 (0 s) | 8 (7,170 s) | 8 (639 s) | 8 (12 s) | 8 (3,426 s) | 8 (489 s) | 8 (1.4 s) |
| [[35,4,9]] | 9 (0 s) | no result | 9 (7,688 s) | 9 (95 s) | unfinished 1 h | 9 (3,381 s) | 9 (42 s) |
| [[41,2,11]] | 11 (1 s) | no result | ≤11 @8 h | 11 (291 s) | unfinished | ≤11 @1 h | [10,11] @1 h |
| [[49,17,7]] | 7 (5 s) | 7 (8,296 s) | 7 (2,656 s) | 7 (255 s) | unfinished | 7 (2,397 s) | 7 (1.1 s) |
| [[50,13,8]] | 8 (38 s) | no result | 8 (12,838 s) | 8 (1,295 s) | unfinished | ≤9 @1 h | 8 (11 s) |
| [[53,6,11]] | 11 (847 s) | no result | ≤11 @8 h | unfinished | unfinished | ≤11 @1 h | [9,11] |
| BB [[144,12,12]] (Z) | | | | unfinished | 12 (4.2 s) | 12 (97 s) | 12 (0.12 s) |
| [[452,115,18]] | | | | unfinished 1 h | no result 1 h | incumbent obj. 26 at 1 h (decoded witness weight 24) | [11,22] (stopped itself at 660 s) |
| [[488,123,22]] | | | | unfinished 1 h | no result 1 h | incumbent obj. 69 (witness weight 63) | [11,26] |

**Reproduction.** Every exact result that codeDistance's methods produced here
equals the paper's value. Time-limited MIP incumbents are upper bounds and are
marked ≤.


**Python BZ (`BZDistMW`, 1 thread, time to exact d; excludes import and JIT) vs
stabcert, per row.** Two stabcert numbers are given:

* (a) `stabcert distance --engines bnb` at **1 thread**, end to end. This includes a
  fixed 10 s RIS phase and the cap ladder.
* (b) the B&B engine alone at cap d−1 on **22 threads** (from §3, wall).

| code | BZDistMW 1 thr | stabcert (a) 1 thr, e2e | stabcert (b) 22 thr, cap d−1 | verdict |
|---|---|---|---|---|
| [[30,9,6]] | 0.94 s | 21.1 s | 0.81 s | BZ faster at equal threads; about even against 22 threads |
| [[34,17,5]] | 32.7 s | 15.3 s | 0.18 s | stabcert faster |
| [[32,5,8]] | 11.8 s | 71.3 s | 2.48 s | BZ faster at equal threads |
| [[33,7,8]] | 38.1 s | 120 s | 5.19 s | BZ faster at equal threads |
| [[35,4,9]] | 94.7 s | 640 s | 17.3 s | BZ faster at equal threads |
| [[41,2,11]] | **291 s** | not finished in 1 h ([9,11]) | 3,778 s | **BZ clearly faster**, even against 22 threads |
| [[49,17,7]] | 255 s | 155 s | 7.61 s | stabcert faster |
| [[50,13,8]] | 1,295 s | 1,761 s | 86.1 s | BZ faster at equal threads; stabcert faster with 22 threads |
| [[53,6,11]], [[80,7,15]] | not finished in 1 h | B&B rejected at the time (generator weight > 32); the later dense W64 build was not benchmarked here | — | open |

At equal threads (1), BZ is faster on 6 of 8 finished small codes. Part of stabcert's
1-thread time is its fixed 10 s upper-bound phase. With 22 threads the B&B is faster on
every code except [[41,2,11]] ([[30,9,6]] is about even: 0.81 s vs 0.94 s).

**Observations about codeDistance v0.0.6:**
* A circular-import NameError breaks non-CSS input with `L=None`. We worked around it
  without editing their files.
* `BZDistMW` with 2-block input returns its witness with X and Z swapped.
* `MIPDist` returns the time-limited incumbent's objective as `d` without a status
  flag. For non-CSS input that objective counts 3-block columns and can exceed the
  decoded witness's Pauli weight (26 vs 24 on 452).

## 5. Heuristic upper bounds (3 seeds, single thread, equal time to QDistEvol's 10,000 trials)

| code | QDistEvol | QDistRndMW | m4riRW (paper-era) | stabcert CPU RIS, same seconds |
|---|---|---|---|---|
| ct codes n ≤ 53 | all at true d | all at true d | all at true d | all at true d |
| [[80,7,15]] | 16/16/16 | 15/16/16 | 15/16/16 | **15/15/15** |
| [[452,115,18]] | 21/21/20 | 26/**18**/21 | 55/60/28 | **18**/21/21 |
| [[488,123,22]] | **22/22**/24 | 30/36/49 | 64/66/76 | **22/22**/25 |

At equal single-thread time, QDistEvol and our CPU RIS are comparable, and both are
clearly better than m4riRW. Our GPU RIS was benchmarked separately during development: for example, 488 reaches 22 on every seed in about 5 s. Its
heuristic witnesses are re-verified by the core but carry no lower-bound meaning.

## 6. Where they win, and open caveats

* **Small dense codes** (codetables n ≤ 50): of the eight codetables codes measured head-to-head,
  dist-m4ri master is 1.2–109× faster than our B&B on seven; [[41,2,11]] is a tie. Their Python BZ is compared per row in §4. Magma, not run here because it is
  commercial, is faster still according to the paper.
* **CSS one-side distances** (BB Z side): dist-m4ri is faster.
* **Pruning.** Lee et al.'s pruning is richer than ours. We did not run their solver:
  its non-CSS source is not public.
* **Load.** The machine was shared. Load is recorded per row, and the headline pairs
  ran on disjoint, otherwise idle physical cores.
* **Not measured:**
  * our own CPU-only prover run on o04829 (the independent checker's
    4,090 s on 128 threads is cited instead, labelled as a checker);
  * codetables [[53,6,11]] / [[80,7,15]] with the dense (W64) B&B build. The pair was
    started and then stopped to stay within the CPU allocation;
  * an ablation of the B&B switches is in [ablation.md](ablation.md). Without the packing bound, node
    counts grow 1,400–5,200×; verified symmetry gives 3.7–18× fewer nodes; Fiedler root
    order gives ~1.7× fewer nodes than file order on 452.

## Appendix A. Non-concurrent rows (exploration, cap ladders, GPU column, single-tool rows)

All times are wall seconds. "overlap" < 1 means the two tools did not run at the same
time; see each row's cores and load.

| instance | side | sym | cap W | tag | threads | dist-m4ri CPU (wall) | stabcert CPU (wall) | stabcert 1× H20 (wall) | cores m4ri / bnb | load at start m4ri / bnb | overlap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bb-144-12-12 | P | cyclic | 11 | - | 22 | **9.76 s** (CC 9.73 s) | **0.37 s** (engine 0.21 s, 1,103,775 nodes) |  | 152-173 / 152-173 | 209 / 229 | 0.00 |
| bb-144-12-12 | P | none | 11 | - | 22 | **14.61 s** (CC 14.58 s) | **0.38 s** (engine 0.31 s, 4,074,223 nodes) | **1.32 s** (engine 1.03 s, 4,074,223 nodes) | 152-173 / 152-173 | 188 / 209 | 0.00 |
| bb-144-12-12 | Z | cyclic | 11 | - | 22 | **0.24 s** (CC 0.02 s) | **0.40 s** (engine 0.18 s, 57,003 nodes) |  | 152-173 / 152-173 | 175 / 181 | 0.00 |
| bb-144-12-12 | Z | none | 11 | - | 22 | **0.04 s** (CC 0.01 s) | **0.23 s** (engine 0.15 s, 218,904 nodes) |  | 152-173 / 152-173 | 169 / 169 | 0.00 |
| bb-72-12-6 | P | none | 5 | - | 22 | **0.03 s** (CC 0.00 s) | **0.13 s** (engine 0.06 s, 1,917 nodes) |  | 152-173 / 152-173 | 169 / 169 | 1.00 |
| bb-72-12-6 | Z | cyclic | 5 | - | 22 | **0.03 s** (CC 0.00 s) | **0.13 s** (engine 0.06 s, 72 nodes) |  | 152-173 / 152-173 | 169 / 169 | 0.00 |
| bb-72-12-6 | Z | none | 5 | - | 22 | **0.03 s** (CC 0.00 s) | **0.13 s** (engine 0.06 s, 335 nodes) |  | 152-173 / 152-173 | 162 / 162 | 0.00 |
| ct-30-9-6 | P | none | 5 | - | 22 | **0.17 s** (CC 0.00 s) | **0.81 s** (engine 0.40 s, 1,567,391 nodes) |  | 128-149 / 152-173 | 140 / 140 | 1.00 |
| ct-32-5-8 | P | none | 7 | - | 22 | **0.08 s** (CC 0.06 s) | **2.48 s** (engine 2.44 s, 119,309,586 nodes) |  | 152-173 / 128-149 | 138 / 138 | 1.00 |
| ct-33-7-8 | P | none | 7 | - | 22 | **0.10 s** (CC 0.08 s) | **5.19 s** (engine 5.14 s, 255,045,190 nodes) |  | 152-173 / 152-173 | 138 / 138 | 0.00 |
| ct-34-17-5 | P | none | 4 | - | 22 | **0.15 s** (CC 0.00 s) | **0.18 s** (engine 0.13 s, 492,420 nodes) |  | 152-173 / 128-149 | 139 / 138 | 0.00 |
| ct-35-4-9 | P | none | 8 | - | 22 | **1.41 s** (CC 1.39 s) | **17.34 s** (engine 17.24 s, 877,975,955 nodes) |  | 128-149 / 128-149 | 144 / 144 | 0.00 |
| ct-41-2-11 | P | none | 10 | - | 22 | **3851.67 s** (CC 3851.64 s) | **3778.03 s** (engine 3777.98 s, 124,454,455,329 nodes) |  | 128-149 / 152-173 | 268 / 266 | 1.00 |
| ct-49-17-7 | P | none | 6 | - | 22 | **0.07 s** (CC 0.05 s) | **7.61 s** (engine 7.52 s, 352,862,949 nodes) |  | 152-173 / 152-173 | 150 / 150 | 0.00 |
| ct-50-13-8 | P | none | 7 | - | 22 | **0.84 s** (CC 0.61 s) | **86.13 s** (engine 86.07 s, 3,369,213,915 nodes) |  | 152-173 / 128-149 | 159 / 162 | 0.00 |
| lee-452-115-18-o04829 | P | none | 16 | - | - |  |  | **216.15 s** (engine 215.89 s, 64,696,430,757 nodes) | - / - | - / - | - |
| lee-452-115-18-o04829 | P | none | 17 | - | - |  |  | **1281.34 s** (engine 1280.97 s, 299,674,975,243 nodes) | - / - | - / - | - |
| nc-100-23-11 | P | cyclic | 10 | - | 22 | **35.93 s** (CC 35.86 s) | **11.71 s** (engine 11.68 s, 376,523,434 nodes) |  | 152-173 / 128-149 | 264 / 263 | 1.00 |
| nc-100-23-11 | P | none | 10 | - | 22 | **37.82 s** (CC 37.78 s) | **10.61 s** (engine 10.40 s, 376,523,434 nodes) | **2.25 s** (engine 1.96 s, 376,523,434 nodes) | 152-173 / 128-149 | 229 / 264 | 0.00 |
| nc-150-62-7 | P | cyclic | 6 | - | 22 | **0.04 s** (CC 0.01 s) | **0.13 s** (engine 0.09 s, 282,397 nodes) |  | 128-149 / 128-149 | 270 / 270 | 0.00 |
| nc-150-62-7 | P | none | 6 | - | 22 | **0.03 s** (CC 0.01 s) | **0.13 s** (engine 0.10 s, 282,397 nodes) |  | 128-149 / 128-149 | 270 / 270 | 0.00 |
| nc-216-4-10 | P | cyclic | 9 | - | 22 | **0.27 s** (CC 0.24 s) | **0.18 s** (engine 0.14 s, 294,236 nodes) |  | 128-149 / 128-149 | 268 / 268 | 0.00 |
| nc-216-4-10 | P | none | 9 | - | 22 | **0.14 s** (CC 0.11 s) | **0.24 s** (engine 0.15 s, 1,090,794 nodes) |  | 128-149 / 128-149 | 270 / 268 | 0.00 |
| nc-452-115-18 | P | none | 13 | - | 16 |  | **15.24 s** (engine 15.09 s, 365,923,578 nodes) |  | - / 128-143 | - / 288 | - |
| nc-452-115-18 | P | none | 14 | - | 16 |  | **66.75 s** (engine 66.58 s, 1,513,941,247 nodes) |  | - / 128-143 | - / 278 | - |
| nc-452-115-18 | P | none | 15 | - | 16 |  | **266.70 s** (engine 266.53 s, 6,424,364,897 nodes) |  | - / 128-143 | - / 294 | - |
| nc-452-115-18 | P | none | 16 | - | 16 |  | **1223.69 s** (engine 1223.56 s, 27,961,897,747 nodes) | **92.09 s** (engine 91.79 s, 27,961,897,747 nodes) | - / 152-167 | - / 241 | - |
| nc-452-115-18 | P | none | 17 | - | 16 | not reached at cap after 7200 s; certified ≤ 12 | **5206.06 s** (engine 5205.90 s, 124,648,548,270 nodes) |  | 128-143 / 152-167 | 218 / 218 | 1.00 |
| nc-452-115-18 | P | none | 17 | - | 64 | not reached at cap after 7200 s; certified ≤ 13 | **1281.74 s** (engine 1281.65 s, 124,648,548,270 nodes) | **486.68 s** (engine 486.39 s, 124,648,548,270 nodes) | 0-47,64-79 / 0-47,64-79 | 102.25 / 189 | 0.00 |
| nc-488-123-22 | P | cyclic | 14 | - | 16 |  | **3.41 s** (engine 3.11 s, 69,678,289 nodes) |  | - / 128-143 | - / 309 | - |
| nc-488-123-22 | P | cyclic | 15 | - | 16 |  | **15.78 s** (engine 15.61 s, 299,421,414 nodes) |  | - / 128-143 | - / 309 | - |
| nc-488-123-22 | P | cyclic | 16 | - | 16 |  | **63.35 s** (engine 63.23 s, 1,322,235,538 nodes) |  | - / 128-143 | - / 293 | - |
| nc-488-123-22 | P | cyclic | 17 | - | 16 |  | **268.06 s** (engine 267.84 s, 5,986,583,611 nodes) |  | - / 128-143 | - / 283 | - |
| nc-488-123-22 | P | cyclic | 17 | - | 64 |  | **58.48 s** (engine 58.37 s, 5,986,583,611 nodes) | **21.66 s** (engine 21.22 s, 5,986,583,611 nodes) | - / 0-47,64-79 | - / 113 | - |
| nc-488-123-22 | P | cyclic | 18 | - | - |  |  | **96.10 s** (engine 95.65 s, 27,770,455,736 nodes) | - / - | - / - | - |
| nc-488-123-22 | P | cyclic | 19 | - | - |  |  | **475.37 s** (engine 475.01 s, 131,759,691,896 nodes) | - / - | - / - | - |
| nc-488-123-22 | P | cyclic | 20 | - | - |  |  | **2651.92 s** (engine 2651.64 s, 638,602,122,823 nodes) | - / - | - / - | - |
| nc-488-123-22 | P | cyclic | 21 | - | 16 | cancelled (killed to free cores) after 4341 s; certified ≤ 13 |  |  | 152-167 / - | 285 / - | - |
| nc-488-123-22 | P | cyclic | 21 | - | 64 | not reached at cap after 7200 s; certified ≤ 13 |  |  | 80-111,113,115-127,174-191 / - | 214 / - | - |
| nc-488-123-22 | P | cyclic | 21 | exceeds-2h-cap | - |  |  | **14454.40 s** (engine 14454.44 s, 3,157,706,215,314 nodes) | - / - | - / - | - |
| nc-488-123-22 | P | none | 21 | - | 16 | cancelled (killed to free cores) after 2856 s; certified ≤ 11 |  |  | 128-143 / - | 295 / - | - |


The cancelled SMT-contended attempt (02:10, not used anywhere above) is in
`bench/h2h/results-cancelled.jsonl`. dist-m4ri had certified ≤ 12 when it was stopped
at 820 s.

*Generated by `bench/doc/make_benchmark_md.py` from `bench/h2h/results.jsonl` at
commit 2f4e540 (plus working-tree rows, if any).*
