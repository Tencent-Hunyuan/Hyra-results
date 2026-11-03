#!/usr/bin/env python3
"""Regenerate native/bnb/s012.inc and native/bnb/s2.inc from the reviewed sources (docs/SOUNDNESS.md 5.1).

    gen_s012.py CORE_H FAST_INC OUTDIR

CORE_H   = the reviewed S0 + fixed S1 core frozen/v2-s1fix-2f08f4/core.h (sha256 2f08f408...);
FAST_INC = the reviewed S2 core s2/frozen/v1-c54147fb/fast.inc (sha256 126594f8...).
Mechanical, auditable transformation; every change is marked [port]. Running it on the reviewed inputs
must reproduce both files byte for byte (reviewers: diff OUTDIR against src/hyqec_verify/native/bnb).
Not part of the package or its tests (the reviewed sources live outside this repository)."""
import hashlib
import os
import sys
CORE_SHA = "2f08f40898470ff19c9671d3d063dbb352d3e1876c3ffca7a38abd706204290d"
FAST_SHA = "126594f8ec94d66b9480862cb86be6cd475ed454cbc9781f4c3da777e6132c74"
core, fast, outdir = sys.argv[1:4]
# --no-hooks: omit the test-only STABCERT_RULE_LIMIT calls; production rules builds must compile to the same bytes.
HOOKS = "--no-hooks" not in sys.argv[4:]
assert hashlib.sha256(open(core, "rb").read()).hexdigest() == CORE_SHA, "not the reviewed core.h"
assert hashlib.sha256(open(fast, "rb").read()).hexdigest() == FAST_SHA, "not the reviewed fast.inc"
L = open(core).read().split('\n')
start = next(i for i, l in enumerate(L) if '---- slack-0 parity CSP' in l)
gac2 = next(i for i, l in enumerate(L) if 'Structure-free GAC for the S0 CSP' in l)
cspp = next(i for i, l in enumerate(L) if l.startswith('  HD int csp_prune(const Code& c) {'))
enter = next(i for i, l in enumerate(L) if '// Node evaluation. Returns 1 if a frame was pushed' in l)
a = '\n'.join(L[start:gac2]).rstrip() + '\n'
b = '\n'.join(L[cspp:enter]).rstrip() + '\n'
def rep(s, old, new):
    assert s.count(old) == 1, (s.count(old), old[:90])
    return s.replace(old, new)
a = rep(a, "  unsigned long long tcyc[6], tcnt[6], sub[3], nsub[3], tsub0;\n", "")
a = rep(a, "uint16_t vst[DMAX + 2]; uint8_t alive[CSP_MAXCH]; uint8_t na[DMAX + 2];",
        "uint16_t vst[DMAX + 2]; uint8_t alive[CSP_MAXCH]; uint16_t na[DMAX + 2];   // [port] na widened: up to 2*WMAX values per variable")
a = rep(a, "#ifdef MINDEG\n", "")
a = rep(a, "    }\n#endif\n    // upper bound: the generators sharing", "    }\n    // upper bound: the generators sharing")
a = rep(a, "#ifdef S1\n    if (ub < r - 1 || ub <= lb) return lb;\n#else\n    if (ub < r || ub <= lb) return lb;\n#endif\n",
        "    if (ub < (c.prune >= 2 ? r - 1 : r) || ub <= lb) return lb;   // [port] runtime form of the S1 build's condition\n")
a = rep(a, "    uint64_t adj[64]; uint16_t tq[64 * WMAX]; uint64_t tm[64 * WMAX]; int nt = 0;",
        "    uint64_t adj[64]; int nt = 0;   // [port] tq/tm moved to members (stack size with wide WMAX)")
a = rep(a, "  uint64_t qmk[NMAX];   // scratch, all zero between calls\n",
        "  uint64_t qmk[NMAX];   // scratch, all zero between calls\n  uint16_t tq[64 * WMAX]; uint64_t tm[64 * WMAX];   // [port] mis_pack scratch (were stack arrays)\n")
i = a.index("  // can a packed variable other than i (at a qubit other than qx) flip g?")
j = a.index("  uint32_t gstamp[MMAX], curstamp;")
a = a[:i] + "  // [port] unused helper flippers() dropped\n" + a[j:]
a = rep(a, "  long long s1_calls, s1_prunes, s1_cands, x_calls, x_prunes;", "  long long s1_calls, s1_prunes, s1_cands;")
# [port] S1 qubit filter cleanup (review finding 01, wide builds): the reviewed list tqs[CSP_MAXCH * 2] recorded only
# level-1 qubits and only up to 2048 of them, while qcnt/qlast were updated for every touched qubit. With NMAX > 2048
# an overflow left dirty entries that a later call read (false S1 prune). Now every qubit is recorded when it is first
# dirtied in this call (at most once, so NMAX entries suffice) and all of them are cleaned. Where the old list did not
# overflow the decisions are identical (only qcnt == need counts, and such qubits were level-1 qubits).
a = rep(a, "uint8_t qcnt[NMAX], qlast[NMAX];", "uint8_t qcnt[NMAX], qlast[NMAX]; uint16_t s1_tq[NMAX];")
a = rep(a, "int need = 0, ntq = 0; uint16_t tqs[CSP_MAXCH * 2];", "int need = 0, ntq = 0;   // [port] touched list is the member s1_tq[NMAX]")
a = rep(a, "if (qlast[q] == need) continue; if (qcnt[q] == need - 1) { qcnt[q] = need; if (need == 1 && ntq < CSP_MAXCH * 2) tqs[ntq++] = (uint16_t)q; } qlast[q] = need; }",
        "if (qlast[q] == need) continue; if (!qlast[q] && !qcnt[q]) s1_tq[ntq++] = (uint16_t)q; if (qcnt[q] == need - 1) qcnt[q] = need; qlast[q] = need; }   // [port] record on first dirtying")
a = rep(a, "if (qcnt[q] == need - 1) { qcnt[q] = need; if (need == 1 && ntq < CSP_MAXCH * 2) tqs[ntq++] = (uint16_t)q; } }\n",
        "if (qcnt[q] == need - 1) { if (!qlast[q] && !qcnt[q]) s1_tq[ntq++] = (uint16_t)q; qcnt[q] = need; } }   // [port] record on first dirtying\n")
a = rep(a, "for (int t = 0; t < ntq; t++) { int q = tqs[t]; if (qcnt[q] == need && lab[q] >= LFREE) any = 1; qcnt[q] = 0; qlast[q] = 0; }",
        "for (int t = 0; t < ntq; t++) { int q = s1_tq[t]; if (qcnt[q] == need && lab[q] >= LFREE) any = 1; qcnt[q] = 0; qlast[q] = 0; }")
a = rep(a, "        // (qubits only touched by later variables were never recorded: they cannot have qcnt == need)\n        if (ntq >= CSP_MAXCH * 2) any = 1;           // overflow: be conservative\n",
        "        // [port] every dirtied qubit is in s1_tq (no overflow is possible), so all of them are clean again here\n")
# [port] test-only capacity-limit counters. Each hooked line is emitted as
#   #ifdef STABCERT_RULE_LIMIT / <line with the counter call> / #else / <the line exactly as reviewed> / #endif
# so every production build (which never defines STABCERT_RULE_LIMIT) preprocesses to the unhooked source.
def hook(text, old, new):
    lines = text.split("\n")
    hit = [k for k, l in enumerate(lines) if old in l]
    assert len(hit) == 1, (len(hit), old[:80])
    k = hit[0]
    lines[k] = "#ifdef STABCERT_RULE_LIMIT\n" + lines[k].replace(old, new) + "\n#else\n" + lines[k] + "\n#endif"
    return "\n".join(lines)
if HOOKS:
    L = lambda name: f'STABCERT_RULE_LIMIT("{name}");'
    a = hook(a, "    if (nA > 64) return -1;", "    if (nA > 64) { " + L("mis_nA_gt_64") + " return -1; }")
    a = hook(a, "if (gfn[g] == 0) { res = 1; if (rec && nzl < AMAX) zl[nzl++] = (uint16_t)g; }",
             "if (gfn[g] == 0) { res = 1; if (rec) { if (nzl < AMAX) zl[nzl++] = (uint16_t)g; else " + L("zl_amax") + " } }")
    a = hook(a, "else if (gfn[g] == 1 && nneed < AMAX) {",
             "else if (gfn[g] == 1 && nneed >= AMAX) " + L("need_amax") + " else if (gfn[g] == 1) {")
    a = hook(a, "if (rec) { if (nch < CSP_MAXCH) kil[nch] = (uint16_t)(killer < 0 ? 0xffff : killer); nch++; }",
             "if (rec) { if (nch < CSP_MAXCH) kil[nch] = (uint16_t)(killer < 0 ? 0xffff : killer); else " + L("kil_maxch") + " nch++; }")
    a = hook(a, "    if (unit_core(c, h, nullptr, 0, -1, 1) && vfirst[h] <= CSP_MAXCH) {",
             "    int unit_wiped = unit_core(c, h, nullptr, 0, -1, 1); if (unit_wiped && vfirst[h] > CSP_MAXCH) " + L("s1_records_maxch") + " if (unit_wiped && vfirst[h] <= CSP_MAXCH) {")
    a = hook(a, "      if (nk2 >= CSP_MAXOCC) res = 0;", "      if (nk2 >= CSP_MAXOCC) { " + L("s1_k2_maxocc") + " res = 0; }")
    a = hook(a, "      if (h > 31) res = 0;", "      if (h > 31) { " + L("s1_bitmask_h_gt_31") + " res = 0; }")
    a = hook(a, "        for (int t = 0; t < ntq; t++) { int q = s1_tq[t];",
             "        if (ntq > 2048) " + L("s1_touched_gt_2048") + " for (int t = 0; t < ntq; t++) { int q = s1_tq[t];")
    b = hook(b, "    int na0 = 0;", "    int na0 = 0; if (nA > AMAX) " + L("greedy_amax"))
    b = hook(b, "          if (nch >= CSP_MAXCH) { res = 0; break; }", "          if (nch >= CSP_MAXCH) { " + L("gac_maxch") + " res = 0; break; }")
    b = hook(b, "if (lid < 0) { if (nlg >= CSP_MAXLG) { res = 0; break; }", "if (lid < 0) { if (nlg >= CSP_MAXLG) { " + L("gac_maxlg") + " res = 0; break; }")
    b = hook(b, "            if (nfl >= CSP_MAXOCC) { res = 0; break; }", "            if (nfl >= CSP_MAXOCC) { " + L("gac_maxocc") + " res = 0; break; }")
a = a.replace("  HD int ", "  int ").replace("  HD void ", "  void ")
b = rep(b, "#ifdef TIMING\n    tsub0 = __rdtsc();\n#endif\n", "")
b = rep(b, "na[i] = (uint8_t)(vst[i + 1] - vst[i]);", "na[i] = (uint16_t)(vst[i + 1] - vst[i]);   // [port] matches widened na[]")
b = rep(b, "#if defined(MISPACK) && !defined(FUSED)\n    np = mis_pack(c, r, 0);\n#endif\n", "")
b = rep(b, "#ifdef TIMING\n    unsigned long long tm0 = __rdtsc(); sub[0] += tm0 - tsub0; nsub[0]++;\n#endif\n"
           "#if defined(MISPACK) && defined(FUSED)\n    if (np < r) { int m = mis_pack(c, r, np); if (m >= 0) np = m;\n"
           "#ifdef TIMING\n      sub[1] += __rdtsc() - tm0; nsub[1]++;\n#endif\n    }\n#endif\n"
           "#ifdef XB\n    if (np < r - 1 && np >= 1 && r - np <= XMAXS) return xbound(c, np, r - np);\n#endif\n"
           "#ifdef S1\n    if (np == r - 1 && r >= S1MINR) return s1_refute(c, np);\n#endif\n",
        "    if (np < r) { int m = mis_pack(c, r, np); if (m >= 0) np = m; }\n"
        "    if (c.prune >= 3 && np == r - 2) return s2_extra(c, *this, np) ? 4 : 0;   // [port] S2 dispatch as in the reviewed S2 core c54147fb\n"
        "    if (c.prune >= 2 && np == r - 1 && r >= 2) return s1_refute(c, np) ? 3 : 0;   // [port] S1 with S1MINR=2\n")
b = rep(b, "    if (np > r) { csp_prunes++; return 1; }", "    if (np > r) { csp_prunes++; return 1; }   // packing exceeds the budget: the same decision as bound()")
b = rep(b, "#ifdef UNITFIRST\n    {\n#ifdef TIMING\n    unsigned long long tu0 = __rdtsc();\n#endif\n    int ur = unit_refute(c, r);\n"
           "#ifdef TIMING\n    sub[2] += __rdtsc() - tu0; nsub[2]++;\n#endif\n    if (ur) { csp_prunes++; csp_unit++; return 1; }\n    }\n#endif\n"
           "#ifdef GAC2\n    if (gac2(c, r, nullptr, 0, -1)) { csp_prunes++; return 1; }\n    return 0;\n#endif\n"
           "#ifdef UNITFIRST\n#ifdef UNITONLY\n    return 0;\n#endif\n#endif\n",
        "    {\n    int ur = unit_refute(c, r);\n    if (ur) { csp_prunes++; csp_unit++; return 2; }\n    }\n")
b = rep(b, "    if (res > 0) csp_prunes++;\n    return res > 0;", "    if (res > 0) csp_prunes++;\n    return res > 0 ? 2 : 0;")
b = rep(b, "  HD int csp_prune(const Code& c) {",
        "  // [port] returns 0 = no prune, 1 = packing > r, 2 = S0 (unit or GAC), 3 = S1, 4 = S2\n  int csp_prune(const Code& c) {")
hdr = """// S0 / S1 (and the S2 dispatch) for the opt-in --bound policies greedy+s0, greedy+s0s1, greedy+s0s1s2.
// Included INSIDE struct Dfs (core.h). Host-only code; never called from device code.
// GENERATED by tests/native/port/gen_s012.py from the reviewed core frozen/v2-s1fix-2f08f4/core.h (sha256 2f08f408...),
// which was reviewed built with -DCSP0 -DMISPACK -DUNITFIRST -DFUSED -DMINDEG -DS1 -DS1MINR=2 (the S2 core c54147fb adds
// -DS2). Every change is marked [port]: compile-time flags became checks of c.prune with the same order and
// semantics; csp_prune reports which rule cut; na[] width and mis_pack scratch for wide variants; statistics/timing
// instrumentation and unused experimental code (GAC2, xbound, flippers) dropped.
// Soundness: docs/SOUNDNESS.md section 5.1.
"""
out = hdr + a + "\n" + b
# [port] comments carried from the reviewed source: reviewer/author ids replaced by role words
out = rep(out, '// ---- slack-0 parity CSP (r-' '119; NOTES.md "Rule S0").', '// ---- slack-0 parity CSP (rule S0).')
out = rep(out, "invalidate it (r-" "125 finding, stale-cache false prune)", "invalidate it (review finding: stale-cache false prune)")
bad = [l for l in out.split('\n') if l.lstrip().startswith('#') and 'CSP_MAX' not in l and '#endif' not in l
       and l.strip() not in ('#ifdef STABCERT_RULE_LIMIT', '#else')]
assert not bad, bad
body = a + b
assert 'TIMING' not in body and 'STATS' not in body and 'xbound' not in body and 'gac2' not in body and '__rdtsc' not in body
open(os.path.join(outdir, 's012.inc'), 'w').write(out)
s2 = open(fast).read()
s2 = rep(s2, "#ifdef S2AUDIT\nstatic thread_local long long s2_audit_calls=0,s2_audit_cuts=0;\n#endif\n", "")
s2 = rep(s2, "#ifdef S2AUDIT\n  ++s2_audit_calls;s2_audit_cuts+=result;\n#endif\n", "")
assert 'S2AUDIT' not in s2 and '#if' not in s2
# [port] S2 exact quotient membership (owner decision, coverage spec 2.0): an extra set E survives iff
# sigma(E) XOR tau lies in the span W EXACTLY. The reviewed 64-bit fingerprint (keys annihilating W) is kept as a
# necessary filter: membership implies equal fingerprints. Every fingerprint match is then confirmed by reducing
# against the stored basis, so a fingerprint collision no longer lets a non-member survive. The hash entries become
# chains, so that every action with a given fingerprint is checked. Decisions differ from the reviewed S2 only where
# a fingerprint collided.
s2 = rep(s2, "  uint64_t basis[MMAX][NW], key[MMAX], hk[HT];\n  uint32_t epoch=0, be[MMAX]={}, he[HT]={};\n  int hq[HT]; bool hm[HT];\n",
         "  uint64_t basis[MMAX][NW], key[MMAX], hk[HT];\n  uint32_t epoch=0, be[MMAX]={}, he[HT]={};\n"
         "  int hhead[HT], aq[HT], al[HT], anext[HT]; // [port] chains of actions per fingerprint (at most 3n <= HT/2)\n")
s2 = rep(s2, "  int lookup(uint64_t k)const{int p=mix(k)&(HT-1);while(he[p]==epoch&&hk[p]!=k)p=(p+1)&(HT-1);return p;}\n",
         "  int lookup(uint64_t k)const{int p=mix(k)&(HT-1);while(he[p]==epoch&&hk[p]!=k)p=(p+1)&(HT-1);return p;}\n"
         "  // [port] sigma of action (q,l): the generators it flips\n"
         "  static void action(const Code&c,int q,int l,uint64_t*v,int nw){for(int k=0;k<nw;++k)v[k]=0;for(int j=0;j<c.qc[q];++j)if(c.ql[q*c.C+j]!=l){int g=c.qg[q*c.C+j];v[g>>6]^=1ull<<(g&63);}}\n"
         "  // [port] exact membership in W: reduce by the lowest-pivot basis (as in the insertion loop) and test for zero\n"
         "  bool inW(uint64_t*v,int nw)const{int word=0;while(true){while(word<nw&&!v[word])++word;if(word==nw)return true;int pivot=word*64+__builtin_ctzll(v[word]);\n"
         "      if(be[pivot]!=epoch)return false;for(int k=word;k<nw;++k)v[k]^=basis[pivot][k];}}\n")
OLD_TAIL = (
"    uint64_t goal=0;for(int k=0;k<nw;++k)for(uint64_t b=target[k];b;b&=b-1)goal^=key[k*64+__builtin_ctzll(b)];\n"
"    if(!goal)return 0; // E empty survives this necessary condition.\n"
"    for(int q=0;q<c.n;++q){int a=st.lab[q];if(a<LFREE)continue;\n"
"      for(int l=1;l<=3;++l){if(a>LFREE&&a-LFREE!=l)continue;uint64_t z=0;\n"
"        for(int j=0;j<c.qc[q];++j)if(c.ql[q*c.C+j]!=l)z^=key[c.qg[q*c.C+j]];\n"
"        if(z==goal)return 0; // One extra survives.\n"
"        int p=lookup(z^goal);if(he[p]==epoch&&(hq[p]!=q||hm[p]))return 0;\n"
"        p=lookup(z);if(he[p]!=epoch){he[p]=epoch;hk[p]=z;hq[p]=q;hm[p]=false;}else if(hq[p]!=q)hm[p]=true;\n"
"      }\n"
"    }\n"
"    return 1;")
NEW_TAIL = (
"    uint64_t goal=0;for(int k=0;k<nw;++k)for(uint64_t b=target[k];b;b&=b-1)goal^=key[k*64+__builtin_ctzll(b)];\n"
"    uint64_t v[NW],u[NW];\n"
"    // [port] E empty survives iff tau is in W (goal == 0 is necessary; confirm exactly)\n"
"    if(!goal){for(int k=0;k<nw;++k)v[k]=target[k];if(inW(v,nw))return 0;}\n"
"    int na=0;\n"
"    for(int q=0;q<c.n;++q){int a=st.lab[q];if(a<LFREE)continue;\n"
"      for(int l=1;l<=3;++l){if(a>LFREE&&a-LFREE!=l)continue;uint64_t z=0;\n"
"        for(int j=0;j<c.qc[q];++j)if(c.ql[q*c.C+j]!=l)z^=key[c.qg[q*c.C+j]];\n"
"        // [port] one extra (q,l) survives iff sigma(q,l) XOR tau is in W\n"
"        if(z==goal){action(c,q,l,v,nw);for(int k=0;k<nw;++k)v[k]^=target[k];if(inW(v,nw))return 0;}\n"
"        // [port] a pair with an earlier action on a different qubit survives iff sigma(a) XOR sigma(b) XOR tau is in W\n"
"        int p=lookup(z^goal);\n"
"        if(he[p]==epoch)for(int b=hhead[p];b>=0;b=anext[b]){if(aq[b]==q)continue;\n"
"          action(c,q,l,v,nw);action(c,aq[b],al[b],u,nw);for(int k=0;k<nw;++k)v[k]^=u[k]^target[k];if(inW(v,nw))return 0;}\n"
"        p=lookup(z);if(he[p]!=epoch){he[p]=epoch;hk[p]=z;hhead[p]=-1;}\n"
"        aq[na]=q;al[na]=l;anext[na]=hhead[p];hhead[p]=na;na++;\n"
"      }\n"
"    }\n"
"    return 1;")
s2 = rep(s2, OLD_TAIL, NEW_TAIL)
s2 = rep(s2, "// The linear fingerprint may collide. Collisions ACCEPT extra candidates;\n// therefore failure to find an E of size <=2 still proves infeasibility.\n",
         "// [port] The linear fingerprint is only a filter: every match is confirmed by exact membership in W,\n"
         "// so the decision is independent of the keys. Failure to find an E of size <=2 proves infeasibility.\n")
if HOOKS:
    s2 = hook(s2, "if(h<0||h>DMAX||c.n*3>HT/2)return 0;",
              'if(h<0||h>DMAX||c.n*3>HT/2){STABCERT_RULE_LIMIT(c.n*3>HT/2?"s2_table":"s2_h_range");return 0;}')
    s2 = hook(s2, "if(first)return 0; //", 'if(first){STABCERT_RULE_LIMIT("s2_empty_domain");return 0;} //')
# [port] hash-table size overridable (only by the capacity-boundary build of the cut audit); default 8192 as reviewed
s2 = rep(s2, "static const int NW=(MMAX+63)/64, HT=8192;", "static const int NW=(MMAX+63)/64, HT=S2_HT;")
s2 = rep(s2, "#include <cstring>\n", "#include <cstring>\n#ifndef S2_HT\n#define S2_HT 8192   // [port] hash-table size (reviewed value)\n#endif\n")
s2 = ("// S2 (slack two) for the opt-in --bound policy greedy+s0s1s2. Included after struct Dfs (core.h); host-only.\n"
      "// Ported verbatim from the reviewed S2 core, frozen/v1-c54147fb/fast.inc (sha256 126594f8...).\n"
      "// [port] changes: this header comment; the S2AUDIT statistics counters removed; S2_HT; exact quotient membership\n"
      "// (fingerprint matches confirmed by reduction modulo W). Soundness: docs/SOUNDNESS.md 5.1.\n" + s2)
open(os.path.join(outdir, 's2.inc'), 'w').write(s2)
print(len(out.split('\n')), 'and', len(s2.split('\n')), 'lines')
