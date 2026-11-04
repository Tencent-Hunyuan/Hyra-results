# Fixed-size scratch in the S0/S1/S2 rules: bounds and cleanup (all build variants)

Context: review finding 01. The reviewed sources (the rule author `frozen/v2-s1fix-2f08f4/core.h`, the S2 author
`frozen/v1-c54147fb/fast.inc`) were reviewed at NMAX = MMAX = 1024, WMAX = 32, CMAX = 64. The package also
builds the rules at NMAX/MMAX 4096 and WMAX 64/128, CMAX 128. S1's touched-qubit list `tqs[CSP_MAXCH * 2]`
(2048 entries) could overflow once NMAX > 2048. Its cleanup then missed entries, and a later call read them,
giving a false S1 prune. This file records, for every fixed-size array in `native/bnb/s012.inc` and
`native/bnb/s2.inc`, why its index stays in range at the widest limits, and how entries that must be clean
between calls are restored.

Limits used below:
- NMAX, MMAX <= 4096; WMAX <= 128; CMAX <= 128.
- The search cap is at most DMAXW - 2 = 46, so the packing sizes r, h satisfy r, h <= 46 < DMAX = 72.
- A packed variable has at most 2 * WMAX = 256 values.
- `CSP_MAXCH` 1024, `CSP_MAXLG` 2048, `CSP_MAXOCC` 6144, `AMAX` 96, `S2_HT` 8192 are the defaults; the
  capacity-boundary cut audit shrinks them (see below).

"Guard" means the code checks the bound before writing and on failure **gives up without pruning** or
**drops a constraint**, either of which only weakens the rule. "Cross-call" marks state that must be clean
when a call starts.

## s012.inc (S0, S1, MIS packing)

| array | index / value range | why in range | cross-call cleanup |
|---|---|---|---|
| `gmap[MMAX]` int16 | g < m <= MMAX; values -1, -2, lid < CSP_MAXLG = 2048 | fits int16 | cross-call (-1). Set to -2 for packed rows inside unit_core / s1 and reset before every return. GAC resets via `lg[0..nlg)`, which lists every entry it set (lid and lg are written together). Checked by the scratch hook. |
| `lg, tgt [CSP_MAXLG]`, `occs[CSP_MAXLG+1]` | nlg | guard `nlg >= CSP_MAXLG -> res = 0` (no prune) | per call |
| `cvar, alive [CSP_MAXCH]`, `cfs[CSP_MAXCH+1]` | nch <= CSP_MAXCH | guard `nch >= CSP_MAXCH -> res = 0`; cvar holds i < r <= 46 (uint8) | per call |
| `cfl, occ [CSP_MAXOCC]` | nfl; occ total = nfl | guard `nfl >= CSP_MAXOCC -> res = 0` | per call |
| `vst, na, packed, vfirst, wip [DMAX+2]` | <= r + 1 <= 47 | cap bound; vst/vfirst values <= 46 * 256 < 65536 (uint16); na widened to uint16 ([port]) | per call |
| `adj[64]`, `lst[64]` (mis_pack) | k <= nA <= 64 | guard `nA > 64 -> return -1` (greedy packing kept) | per call |
| `qmk[NMAX]` uint64 | q < n | - | cross-call (0). Every set entry is listed once in `tq` (when it is first set) and cleared from it. Checked by the hook. |
| `tq, tm [64*WMAX]` | distinct candidate qubits of <= 64 rows of weight <= W | <= 64 * W | per call |
| `lst, ord [AMAX]`, `hist[WMAX+2]` (greedy) | na0 < AMAX; gcnt <= W | guard `na0 < AMAX` (truncated packing, sound) | per call |
| `qown[NMAX]` uint8 | values i + 1 <= 47 | - | cross-call (0). own_setup writes candidate positions of packed rows; own_clear clears all positions of the same rows (a superset), on every path. Checked by the hook. |
| `qgl[NMAX]` uint8 | q < n | - | write-only: own_setup assigns it, nothing reads it (kept as in the reviewed source); not decision state, not cleared, not checked (the rules reviewer) |
| `gstamp[MMAX]` uint32, `gfn, gfa, gfb [MMAX]` | stamps; gfa < 46 | curstamp wrap resets gstamp | stamp-invalidated (`++curstamp` per unit pass and after restoring qown: the the S1 reviewer fix) |
| `kil[CSP_MAXCH]`, `kst[CSP_MAXCH+1]` | nch | `kil` written only if nch < CSP_MAXCH; S1 runs only if `vfirst[h] <= CSP_MAXCH` | per call |
| `k2[CSP_MAXOCC]` | nk2 | guard; `nk2 >= CSP_MAXOCC -> res = 0` | per call |
| `need_g, need_v [AMAX]` (local), `sneed_g, sneed_v, zl [AMAX]` | nneed, nzl | guards `< AMAX`. A dropped need or zero entry only removes a killer or E-condition (weaker) | per call |
| `kmask[MMAX]` uint32 | killers g < m; bits i < 31 and bit 31 | `h > 31 -> res = 0` | cross-call (0). Every first-set entry is listed in `kt` and cleared. Checked by the hook. |
| `kt[CSP_MAXCH + AMAX]` | distinct killers (<= recorded values <= CSP_MAXCH) + zl (<= AMAX) | - | per call |
| `base[CSP_MAXCH]` | nb <= values of one variable | - | per call |
| `ef[CMAX]` | nef <= qc[q] <= C <= CMAX | - | per call |
| `qcnt, qlast [NMAX]` uint8 | need <= 46 + 96 = 142 < 256 | - | cross-call (0). **FIXED** ([port], finding 01): every qubit is recorded in `s1_tq[NMAX]` when it is first dirtied (at most once per call), and all of them are cleared. The reviewed `tqs[CSP_MAXCH*2]` recorded only level-1 qubits, up to 2048. Checked by the hook. |
| `s1_tq[NMAX]` | <= n distinct qubits | one entry per qubit per call | per call |
| `qlstamp[NMAX*4]` uint32 | q * 4 + l | - | stamp-invalidated (estamp; wrap resets) |

Not decision state (members of `struct Dfs` from s012.inc that no rule decision reads): the statistics counters
`csp_calls`, `csp_prunes`, `csp_unit`, `s1_calls`, `s1_prunes`, `s1_cands`, `s1_fast` (only incremented) and `last_np`
(an output of csp_prune). Per-call lengths with their arrays: `nk2`, `nzl`, `nneed_s` (sneed), `ntq` (local).

## s2.inc (S2)

| array | range | why | cleanup |
|---|---|---|---|
| `basis[MMAX][NW]`, `be[MMAX]`, `key[MMAX]` | pivot rows g < m | - | epoch-invalidated (`be[g] == epoch`; wrap resets) |
| `hk, he, hhead [S2_HT]` (`HT` = S2_HT, `NW`) | <= 3n distinct fingerprints | guard `3n > S2_HT/2 -> return 0` (no prune); load <= 1/2, so linear probing terminates | epoch-invalidated (`he`); `hhead` set when its slot is first used in an epoch |
| `aq, al, anext [S2_HT]` | action chain, `na` <= 3n <= S2_HT/2 entries | one entry per action per call | per call (rewritten before read) |
| `target, base, v, u [NW]` (local) | NW = ceil(MMAX/64) | - | per call |

`S2Scratch` is `thread_local`, about 2 MB at MMAX = 4096 (`basis`). glibc reserves static TLS inside each
thread's stack. If a stack is too small, thread creation fails and the process exits with an error; it never
prunes wrongly.

## Capacity limits and where each is crossed at full size

Each guard carries a test-only counter, `STABCERT_RULE_LIMIT(name)`. The generator emits it as
`#ifdef STABCERT_RULE_LIMIT <hooked line> #else <reviewed line> #endif`, so production builds preprocess to the
unhooked source; this is verified byte for byte, preprocessed and compiled, for all six rules variants.
`rule_planted.cpp` (node level, planted completions) crosses every limit marked x in every variant:

| limit (counter) | guard | 1024-w32 / w64 / w128 | 4096-w32 / w64 / w128 |
|---|---|---|---|
| `gac_maxch` | GAC choices > CSP_MAXCH | x | x |
| `gac_maxlg` | GAC local generators > CSP_MAXLG (2048) | unreachable (m <= 1024) | x |
| `gac_maxocc` | GAC occurrences > CSP_MAXOCC | x | x |
| `greedy_amax` | \|A_v\| > AMAX (greedy list truncated) | x | x |
| `mis_nA_gt_64` | \|A_v\| > 64 (mis_pack declines) | x | x |
| `kil_maxch` | S1 value records > CSP_MAXCH (kil not written) | x | x |
| `s1_records_maxch` | S1 declines: vfirst[h] > CSP_MAXCH | x | x |
| `need_amax` | unit need list > AMAX | x | x |
| `zl_amax` | unit zl list > AMAX | x | x |
| `s1_k2_maxocc` | S1 killer list > CSP_MAXOCC | x | x |
| `s1_bitmask_h_gt_31` | S1 declines: h > 31 | x | x |
| `s1_touched_gt_2048` | S1 filter touched > 2048 qubits (finding 01 regime) | unreachable (n <= 1024) | x |
| `s2_table` | S2 declines: 3n > S2_HT/2 | unreachable (n <= 1365) | x |
| `s2_h_range` | S2: h < 0 or h > DMAX | unreachable (h <= 46 < 72) | unreachable |
| `s2_empty_domain` | S2: packed row without a value | unreachable (live packing) | unreachable |

`high_guard_cases.cpp` (the rules reviewer's deterministic boundary cases) hits `s1_bitmask_h_gt_31` (h = 32),
`mis_nA_gt_64` and `greedy_amax` (|A_v| = 97), `need_amax`, `zl_amax`, and the `s2_table` boundary (no hit at
n = 1365, hit at n = 1366), all at witnessed nodes that must not be pruned.
The search-level strata (cut audit `--large` in every variant, the reuse fuzzer) cross a subset of these during
real searches at caps 2-3. The shrunk-table configuration crosses, on small codes, only the guards of the
overridable tables (see below); the fixed-constant guards and the n > 2048 regime are crossed by the planted-node
stratum and `high_guard_cases.cpp` only.

## How this is tested

- `cut_audit.cpp` defines `STABCERT_RULE_SCRATCH_HOOK`. After every rule pass it checks that `qcnt`, `qlast`,
  `qmk`, `qown`, `gmap`, `kmask` and the packing bitmap are clean (about 1.1e7 checks per audit run).
- `test_pruning.py` runs the audit twice: at the default capacities, and in the **capacity** configuration
  (`CSP_MAXCH=6 CSP_MAXLG=8 CSP_MAXOCC=16 AMAX=6 S2_HT=64`). There the guards of the overridable tables (choices,
  local generators, occurrences, AMAX lists and greedy truncation, the S2 table) are taken on n <= 14 codes, and
  every cut is still re-searched and checked against enumeration. The fixed-constant guards (`h > 31` in S1,
  `|A_v| > 64` in mis_pack, S2's `h > DMAX`) and the n > 2048 regime cannot occur at n <= 14. The first two, and
  every other limit at full size, are crossed and counted by the planted-node stratum in each variant (table
  above); `h > DMAX` is unreachable.
- Against the old S1 filter, the capacity configuration's scratch check reports dirty `qcnt` entries (the
  finding 01 class). At full capacity the old code leaves only `qlast` on qubits outside level 1 dirty. That is
  harmless, because such a qubit's `qcnt` stays below `need`, but the new code cleans it as well.
- `wide_s1_regression.cpp` / `test_parity_rules.py::test_wide_s1_filter_cleanup_regression`: the rules reviewer's n = 2177
  node pair at NMAX/MMAX 4096, WMAX/CMAX 128. It fails on the old code (S1 prune, 128 stale entries) and
  passes on the fixed code.
- Large-n stratum, at NMAX/MMAX 4096 and WMAX/CMAX 128:
  - `rule_reuse_fuzz.cpp` (`test_large_n_rule_state_reuse`) evaluates random node sequences on one reused search
    state and on a fresh state. It uses the rules reviewer's block family and variants (n up to 2401, rows of weight 60-128,
    n <= 1365 shapes so S2 is active), runs over 200k calls, and requires equal decisions and clean scratch after
    every call. Against the old filter it reports dirty scratch on the 40x60 and 17x128 shapes.
  - `cut_audit.cpp --large` (`test_cut_audit_large_codes`) runs on codes beyond the enumeration oracle (the rules reviewer's
    family and toric(33), n > 2048, P/X, caps 2-3). Every cut is re-searched without the rule from rebuilt state,
    and each policy's distance must equal the reviewed greedy search's.
- Stamp wraparound: the reuse fuzzer starts `curstamp` and `estamp` up to 63 below 2^32 - 1 at the start of every
  node run, so the wrap-and-reset code runs (`gstamp` / `qlstamp` cleared) and must still agree with a fresh state.
  S2's `epoch` uses the same pattern in a function-local thread_local object that the tests cannot reach; it is
  covered by inspection (`if(++epoch==0){memset(be..); memset(he..); epoch=1;}`).
- The cut counts at default capacities and the 452 cap-12 node counts are unchanged by the fix.
