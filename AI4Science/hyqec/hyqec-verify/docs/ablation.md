# Ablation of the stabcert B&B: packing bound, symmetry roots, root order (partial, 2026-10-22)

> Imported from the separate benchmark/prior-art repository at revision d891577
> (2026-10-23). Paths such as `bench/...` refer to that repository, which holds the raw
> results and instances; it is not part of this package. Internal references were removed for publication.
> *stabcert* in this document is the development name of hyqec-verify (renamed for release).

Engine: the stabcert native B&B of 2026-10-22 (`--bound greedy|none`). Every run uses fixed instance bytes written by
stabcert's own writer in variants `sym-file` (verified cyclic group, orbit representatives in file order),
`nosym-file` (trivial group, qubits in file order) and `nosym-fiedler` (trivial group, Fiedler order). Each run
proves "no nontrivial logical of weight <= cap" and none found one, so node counts are deterministic and comparable
across executors (CPU = GPU tree). Times depend on executor and threads (stated per cell); compare **node counts**.
`none` keeps the weight, dead-node and leaf pruning and drops only the disjoint-candidate packing bound. 900 s
limit per run. Raw: `bench/ablation/abl.jsonl`; instances and hashes: `bench/ablation/inst/`.

## Findings (node-count ratios)

* **The packing bound is essential.**
  * Turning it off multiplies nodes by about 1,900× on [[142,2,18]] at cap 13,
    1,440× on [[134,2,17]] at cap 13, 1,730× on [[177,3,16]] at cap 12, and
    4,500–5,200× on BB [[144,12,12]] at cap 11.
  * On [[452,115,18]] at cap 13 and [[488,123,22]] at cap 14, the run without it does
    not finish in 900 s. With it, those runs take 11–34 s and 16 s.
* **Verified symmetry roots matter on cyclic codes.**
  * [[488,123,22]] (61-cycle, 8 orbits): 18× fewer nodes at caps 14 and 15.
  * [[142,2,18]] (one orbit): 5.8× at cap 14.
  * BB144 (24 orbits): 3.7× at cap 11.
  * [[134,2,17]] (one orbit): 4.7× at cap 16.
  * [[177,3,16]]: no-symmetry rows pending.
  * [[452,115,18]] has no qubit automorphisms.
  * Caveat: these ratios compare `sym-file` against `nosym-fiedler`, the package
    default without symmetry. Symmetry and root order therefore both differ between
    the two runs.
* **Root order matters less.** Fiedler order beats file order by about 1.7× on
  [[452,115,18]] (caps 13–15: 1.71×, 1.68×, 1.66×).

Not yet run: [[177,3,16]] no-symmetry rows and packing-off at cap ≥ 13, and
[[452,115,18]] packing-off at cap ≤ 12. These are queued for after the 488 check, at
≤ 16 threads.

## Table

| instance variant | cap | packing on (greedy) | packing off (none) |
|---|---|---|---|
| 134-nosym-fiedler | 13 | 165,483,161 (6.6 s, cpu 22 thr) | — |
| 134-nosym-fiedler | 14 | 623,328,521 (20.7 s, cpu 22 thr) | — |
| 134-nosym-fiedler | 15 | 2,365,430,335 (94.5 s, cpu 22 thr) | — |
| 134-nosym-fiedler | 16 | 9,057,722,862 (303.3 s, cpu 22 thr) | — |
| 134-sym-file | 13 | 29,825,514 (6.8 s, cpu 22 thr) | 42,911,341,325 (363.1 s, cpu 22 thr) |
| 134-sym-file | 14 | 118,924,898 (23.9 s, cpu 22 thr) | timeout 600 s |
| 134-sym-file | 15 | 475,785,359 (86.1 s, cpu 22 thr) | — |
| 134-sym-file | 16 | 1,912,583,799 (152.6 s, cpu 22 thr) | — |
| 142-nosym-fiedler | 14 | 553,035,783 (23.4 s, cpu 12 thr) | — |
| 142-nosym-fiedler | 15 | 2,029,758,703 (90.4 s, cpu 12 thr) | — |
| 142-nosym-fiedler | 16 | 7,518,104,403 (440.7 s, cpu 12 thr) | — |
| 142-sym-file | 13 | 24,620,137 (4.1 s, cpu 32 thr) | 46,394,804,660 (231.0 s, cpu 32 thr) |
| 142-sym-file | 14 | 94,605,986 (16.2 s, cpu 32 thr) | — |
| 142-sym-file | 15 | 364,603,286 (62.9 s, cpu 32 thr) | — |
| 142-sym-file | 16 | 1,411,455,324 (117.7 s, cpu 32 thr) | — |
| 142-sym-file | 17 | 5,480,209,939 (196.4 s, cpu 32 thr) | — |
| 177-sym-file | 12 | 6,704,055 (1.0 s, cpu 22 thr) | 11,595,026,271 (113.4 s, cpu 22 thr) |
| 177-sym-file | 13 | 26,062,430 (3.7 s, cpu 22 thr) | — |
| 177-sym-file | 14 | 101,885,359 (18.4 s, cpu 22 thr) | — |
| 177-sym-file | 15 | 400,101,952 (74.7 s, cpu 22 thr) | — |
| 452-nosym-fiedler | 13 | 365,923,578 (10.8 s, gpu 4 thr) | timeout 900 s |
| 452-nosym-fiedler | 14 | 1,513,941,247 (15.2 s, gpu 4 thr) | — |
| 452-nosym-fiedler | 15 | 6,424,364,897 (35.2 s, gpu 4 thr) | — |
| 452-nosym-file | 13 | 625,098,978 (33.4 s, cpu 12 thr) | timeout 900 s |
| 452-nosym-file | 14 | 2,547,442,540 (134.4 s, cpu 12 thr) | — |
| 452-nosym-file | 15 | 10,660,850,080 (588.9 s, cpu 12 thr) | — |
| 488-nosym-fiedler | 14 | 1,282,026,178 (72.8 s, cpu 12 thr) | — |
| 488-nosym-fiedler | 15 | 5,314,037,970 (298.6 s, cpu 12 thr) | — |
| 488-nosym-fiedler | 16 | timeout 900 s | — |
| 488-sym-file | 14 | 69,678,289 (15.6 s, cpu 12 thr) | timeout 900 s |
| 488-sym-file | 15 | 299,421,414 (73.0 s, cpu 12 thr) | — |
| 488-sym-file | 16 | 1,322,235,538 (184.4 s, cpu 12 thr) | — |
| bb144-nosym-fiedler | 9 | 314,013 (0.1 s, cpu 12 thr) | 420,225,211 (5.6 s, cpu 12 thr) |
| bb144-nosym-fiedler | 10 | 1,156,938 (0.1 s, cpu 12 thr) | 2,815,989,065 (40.6 s, cpu 12 thr) |
| bb144-nosym-fiedler | 11 | 4,074,223 (0.2 s, cpu 12 thr) | 18,569,755,345 (283.8 s, cpu 12 thr) |
| bb144-sym-file | 9 | 84,139 (0.0 s, cpu 12 thr) | 130,034,829 (2.9 s, cpu 12 thr) |
| bb144-sym-file | 10 | 311,635 (0.0 s, cpu 12 thr) | 873,890,926 (26.5 s, cpu 12 thr) |
| bb144-sym-file | 11 | 1,103,775 (0.1 s, cpu 12 thr) | 5,786,021,825 (98.1 s, cpu 12 thr) |
