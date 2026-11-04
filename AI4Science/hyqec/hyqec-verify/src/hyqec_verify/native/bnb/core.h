// pauli_cert core: exhaustive branch & bound over irreducible Pauli clusters.
// Shared verbatim by the CPU (OpenMP) and GPU (CUDA) drivers, so both explore
// exactly the same tree (node counts must agree; checked in validate.py).
// The algorithm and its soundness argument are written out in README.md.
#pragma once
#include <stdint.h>

// Test builds may inspect every discarded domain before it is restored. Release
// builds have no callback, state field, or runtime branch for this instrumentation.
#ifndef STABCERT_CUT_HOOK
#define STABCERT_CUT_HOOK(code, state, reason) ((void)0)
#endif
#ifndef STABCERT_CHECKPOINT_HOOK
#define STABCERT_CHECKPOINT_HOOK(point) ((void)0)
#endif

#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif

HD int __builtin_ctz_hd(uint32_t x) {
#ifdef __CUDA_ARCH__
  return __ffs(x) - 1;
#else
  return __builtin_ctz(x);
#endif
}

#ifndef NMAX
#define NMAX 1024      // qubits
#endif
#ifndef MMAX
#define MMAX 1024      // generators
#endif
#ifndef WMAX
#define WMAX 32        // generator weight
#endif
#ifndef CMAX
#define CMAX 64        // qubit column degree
#endif
#if WMAX != 32 && WMAX != 64 && WMAX != 128
#error "WMAX must be 32, 64 or 128"
#endif
#if CMAX < 1 || CMAX > 128
#error "CMAX must be in 1..128"
#endif
#if defined(__CUDACC__) && (WMAX != 32 || CMAX != 64)
#error "wide generator/column variants are CPU-only"
#endif
#define DMAX 72        // DFS depth (= max weight explored)
#ifndef AMAX            // (overridable only by the capacity-boundary test build of the cut audit)
#define AMAX 96        // anticommuting gens considered by the packing bound (truncation is sound)
#endif

// qubit states
#define LFREE 4        // undecided, domain {I,X,Y,Z}
// 0 = I, 1 = X, 2 = Z, 3 = Y (decided); 5,6,7 = restricted to {I, lab-4}

struct Code {
  int n, m, W, C, nl;
  int domain = 0;      // 0 = P, 1 = X-only, 2 = Z-only
  bool greedy = true;
  bool trace_enabled = false;
  bool coverage_bound = false; // experimental b1/b3, outside coverage certificate v1
#ifdef STABCERT_RULES
  int prune = 0;       // opt-in S0/S1/S2 rules (s012.inc, s2.inc): 0 off, 1 s0, 2 s0s1, 3 s0s1s2. Only in the
                       // CPU rules build (-DSTABCERT_RULES); outside coverage certificate v1, exported only as
                       // coverage certificate v2 (docs/BNB_CERT_SPEC_v2.md).
#endif
  int flip_degree = 0;
  const uint16_t* gq;   // m*W  qubit of generator g, position p
  const uint8_t* gl;    // m*W  Pauli label of g at that qubit
  const uint8_t* gw;    // m    weight of g
  const uint16_t* qg;   // n*C  generators containing qubit q
  const uint8_t* ql;    // n*C  label of that generator at q
  const uint8_t* qc;    // n    column degree
  const uint8_t* qp;    // n*C  position of q in the support of that generator (GPU kernel only)
  const uint8_t* L;     // nl*n logical basis labels (0 = I)
};

HD int cand(int lab, int gl) { return lab == LFREE || (lab > LFREE && lab - LFREE != gl); }
HD int nonI(int lab) { return lab >= 1 && lab <= 3; }

// BNB_CERT_SPEC.md section 8. Only diagnostic hash arithmetic wraps.
HD uint64_t trace_mix(uint64_t x) {
  x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  return x ^ (x >> 31);
}
HD uint64_t trace_salt(int q, int a) { return trace_mix((uint64_t)8*q + a + 1); }
HD uint64_t trace_word(int w, int na, int rule, int g = 65535, int h = 0) {
  return (uint64_t)w | ((uint64_t)na << 8) | ((uint64_t)rule << 24)
       | ((uint64_t)g << 32) | ((uint64_t)h << 48);
}
HD uint64_t trace_append(uint64_t hash, uint64_t fp, uint64_t packed) {
  return (hash * UINT64_C(0x100000001b3) + fp) * UINT64_C(0x100000001b3) + packed;
}

struct Frame {
  uint16_t g;        // branching generator
  uint8_t pos;       // current candidate position in g's support
  uint8_t li;        // next label option index at pos
  uint8_t acur;      // state of qubit at pos before the current child label
  uint8_t pad[3];
#if WMAX == 32
  uint32_t modmask;  // positions this frame restricted (free->{I,g}) or set to I (restricted->I)
  uint64_t savedP;   // original P (2 bits/pos) for restricted->I
#else
  uint8_t saved[WMAX]; // original undecided state; zero means unmodified
#endif

  HD void clear_changes() {
#if WMAX == 32
    modmask = 0; savedP = 0;
#else
    for (int p = 0; p < WMAX; p++) saved[p] = 0;
#endif
  }
  HD void remember(int p, int a) {
#if WMAX == 32
    modmask |= 1u << p;
    if (a != LFREE) savedP |= (uint64_t)(a - LFREE) << (2*p);
#else
    saved[p] = (uint8_t)a;
#endif
  }
  HD bool changed(int p) const {
#if WMAX == 32
    return (modmask >> p) & 1;
#else
    return saved[p] != 0;
#endif
  }
  HD int original(int p, int current) const {
#if WMAX == 32
    return current == 0 ? LFREE + ((savedP >> (2*p)) & 3) : LFREE;
#else
    return saved[p];
#endif
  }
};

#ifdef STABCERT_RULES
struct Dfs;
inline int s2_extra(const Code&, Dfs&, int);   // s2.inc
// Test builds may define STABCERT_RULE_LIMIT(name) to count capacity-limit hits (s012.inc, s2.inc); without it
// the rule sources preprocess to exactly their unhooked form.
#endif

struct Dfs {
  uint8_t lab[NMAX];
  uint8_t gcnt[MMAX];          // # candidate qubits of g
  uint32_t sb[(MMAX + 31) / 32]; // anticommutation bit of g (the set A)
  int dead;                    // # g with s=1, cnt=0
  int nA;                      // |A|
  int w;                       // # decided non-I qubits
  int nsupp;
  uint16_t supp[DMAX + 2];
  int depth;
  Frame fr[DMAX + 2];
  uint32_t used[(NMAX + 31) / 32];
  int cap;
  long long nodes;
  uint64_t state_fp, trace;
  int last_rule, last_g, last_h;

  HD void init(const Code& c, int cap0) {
    for (int q = 0; q < c.n; q++) lab[q] = LFREE + c.domain;
    for (int g = 0; g < c.m; g++) {
      gcnt[g] = 0;
      for (int p = 0; p < c.gw[g]; p++)
        gcnt[g] += cand(LFREE + c.domain, c.gl[g * c.W + p]);
    }
    for (int i = 0; i < (c.m + 31) / 32; i++) sb[i] = 0;
    for (int i = 0; i < (c.n + 31) / 32; i++) used[i] = 0;
    dead = 0; nA = 0; w = 0; nsupp = 0; depth = 0; cap = cap0; nodes = 0; last_found_w = 0; nwit = 0; aborted = 0;
    state_fp = trace = 0;
    if (c.trace_enabled) for (int q = 0; q < c.n; q++) state_fp ^= trace_salt(q, lab[q]);
#ifdef STABCERT_RULES
    if (c.prune) {          // s012.inc scratch: all-zero / -1 between calls
      for (int g = 0; g < c.m; g++) { gmap[g] = -1; gstamp[g] = 0; kmask[g] = 0; }
      for (int q = 0; q < c.n; q++) { qmk[q] = 0; qown[q] = 0; qcnt[q] = qlast[q] = 0; }
      for (int q = 0; q < c.n * 4; q++) qlstamp[q] = 0;
      curstamp = 0; estamp = 0; s1_fast = 0;
      csp_calls = csp_prunes = csp_unit = 0; s1_calls = s1_prunes = s1_cands = 0;
    }
#endif
  }

  HD int sbit(int g) const { return (sb[g >> 5] >> (g & 31)) & 1; }

  // set qubit q from state a to state b
  HD void change(const Code& c, int q, int a, int b) {
    if (c.trace_enabled) state_fp ^= trace_salt(q, a) ^ trace_salt(q, b);
    int deg = c.qc[q];
    for (int j = 0; j < deg; j++) {
      int g = c.qg[q * c.C + j], gl = c.ql[q * c.C + j];
      int s = sbit(g), cn = gcnt[g];
      int wd = s && cn == 0;
      cn += cand(b, gl) - cand(a, gl);
      int fa = nonI(a) && a != gl, fb = nonI(b) && b != gl;
      if (fa != fb) { s ^= 1; sb[g >> 5] ^= 1u << (g & 31); nA += s ? 1 : -1; }
      gcnt[g] = (uint8_t)cn;
      dead += (s && cn == 0) - wd;
    }
    lab[q] = (uint8_t)b;
    if (nonI(b) && !nonI(a)) { w++; supp[nsupp++] = (uint16_t)q; }
    else if (nonI(a) && !nonI(b)) { w--; nsupp--; }
  }

  HD int anyA(const Code& c) const { return nA > 0; }

  // nontrivial iff anticommutes with some element of the logical basis
  HD int nontrivial(const Code& c) const {
    for (int l = 0; l < c.nl; l++) {
      int p = 0;
      for (int i = 0; i < nsupp; i++) {
        int q = supp[i], b = c.L[l * c.n + q];
        if (b && b != lab[q]) p ^= 1;
      }
      if (p) return 1;
    }
    return 0;
  }

  // branching generator: anticommuting g with the fewest children
  // (free candidate -> 2 labels, restricted candidate -> 1); ties -> lowest index
  HD int pick(const Code& c) const {
    int best = -1, bc = 1 << 30;
    for (int i = 0; i < (c.m + 31) / 32; i++) {
      uint32_t word = sb[i];
      while (word) {
        int g = i * 32 + __builtin_ctz_hd(word); word &= word - 1;
        int ch = 0;
        for (int p = 0; p < c.gw[g]; p++) {
          int lq = lab[c.gq[g * c.W + p]], gl = c.gl[g * c.W + p];
          if (lq == LFREE) ch += 2; else if (lq > LFREE && lq - LFREE != gl) ch += 1;
        }
        if (ch < bc) { bc = ch; best = g; }
      }
    }
    return best;
  }

  // lower bound on the number of further non-I qubits: greedy packing of
  // anticommuting generators with pairwise disjoint candidate sets.
  // Returns as soon as w + h > cap (the caller only needs that comparison).
  HD int bound(const Code& c, uint16_t* selected = nullptr) {
    if (!c.greedy) return 0;
    if (w + nA <= cap) return nA > 0 ? 1 : 0;   // h <= |A|: no prune possible (decision unchanged)
    uint16_t lst[AMAX], ord[AMAX];
    int hist[WMAX + 2];
    for (int i = 0; i <= c.W + 1; i++) hist[i] = 0;
    int na = 0;
    for (int i = 0; i < (c.m + 31) / 32 && na < AMAX; i++) {
      uint32_t word = sb[i];
      while (word && na < AMAX) {
        int g = i * 32 + __builtin_ctz_hd(word); word &= word - 1;
        lst[na++] = (uint16_t)g; hist[gcnt[g] + 1]++;
      }
    }
    for (int i = 1; i <= c.W + 1; i++) hist[i] += hist[i - 1];
    for (int i = 0; i < na; i++) ord[hist[gcnt[lst[i]]]++] = lst[i];   // stable: (cnt, g) order
    int h = 0;
    for (int i = 0; i < na; i++) {
      if (w + h + (na - i) <= cap) break;     // cannot exceed the budget any more (decision unchanged)
      int g = ord[i];
      int ok = 1;
      for (int p = 0; p < c.gw[g]; p++) {
        int q = c.gq[g * c.W + p];
        if (cand(lab[q], c.gl[g * c.W + p]) && ((used[q >> 5] >> (q & 31)) & 1)) { ok = 0; break; }
      }
      if (!ok) continue;
      if (selected) selected[h] = (uint16_t)g;
      h++;
      for (int p = 0; p < c.gw[g]; p++) {
        int q = c.gq[g * c.W + p];
        if (cand(lab[q], c.gl[g * c.W + p])) used[q >> 5] |= 1u << (q & 31);
      }
      if (w + h > cap) break;
    }
    if (h) for (int i = 0; i < (c.n + 31) / 32; i++) used[i] = 0;
    return h;
  }

  HD int first_cand(const Code& c, int g, int from) const {
    for (int p = from; p < c.gw[g]; p++)
      if (cand(lab[c.gq[g * c.W + p]], c.gl[g * c.W + p])) return p;
    return -1;
  }

  // Each active generator needs at least one future syndrome flip. A qubit
  // can cover at most c_q active generators under any allowed label. Taking
  // the t largest c_q therefore bounds what t new support qubits can cover.
  HD int syndrome_coverage_bound(const Code& c) const {
    if (!c.coverage_bound) return 0;
    if (!c.flip_degree) return cap + 1;
    int b1 = (nA + c.flip_degree - 1) / c.flip_degree;
    if (w + b1 > cap) return b1;
    int hist[CMAX + 1];
    for (int i=0;i<=c.C;i++) hist[i]=0;
    for (int q=0;q<c.n;q++) if (lab[q]>=LFREE) {
      int active=0, labels[4]={0,0,0,0};
      for (int j=0;j<c.qc[q];j++) if (sbit(c.qg[q*c.C+j])) {
        active++; labels[c.ql[q*c.C+j]]++;
      }
      int cover=0;
      if (lab[q]==LFREE) {
        for (int l=1;l<=3;l++) if (active-labels[l]>cover) cover=active-labels[l];
      } else cover=active-labels[lab[q]-LFREE];
      hist[cover]++;
    }
    int needed=nA, support=0;
    for (int cover=c.C;cover>0 && needed>0;cover--) {
      int take=(needed+cover-1)/cover;
      if (take>hist[cover]) take=hist[cover];
      support+=take; needed-=take*cover;
    }
    if (needed>0) return cap+1;
    return support>b1 ? support : b1;
  }

  // label options at a candidate qubit in state a for generator label gl
  HD int options(int a, int gl, int* opt) const {
    if (a == LFREE) { int k = 0; for (int l = 1; l <= 3; l++) if (l != gl) opt[k++] = l; return 2; }
    opt[0] = a - LFREE; return 1;
  }

  // after all children at position p: free -> {I, gl}; restricted -> I
  HD void modify(const Code& c, Frame& F, int p) {
    int q = c.gq[F.g * c.W + p], gl = c.gl[F.g * c.W + p], a = lab[q];
    F.remember(p, a);
    if (a == LFREE) change(c, q, a, LFREE + gl);
    else change(c, q, a, 0);
  }

  HD void unmodify(const Code& c, Frame& F) {
    for (int p = c.gw[F.g] - 1; p >= 0; p--) {
      if (!F.changed(p)) continue;
      int q = c.gq[F.g * c.W + p], a = lab[q];
      int orig = F.original(p, a);
      change(c, q, a, orig);
    }
    F.clear_changes();
  }

  // Node evaluation. Returns 1 if a frame was pushed (node expanded).
  // *found set to weight when a nontrivial logical (in N(S)) is reached.
  HD void record(const Code& c, int rule, int g = 65535, int h = 0) {
    last_rule = rule; last_g = g; last_h = h;
    if (c.trace_enabled) trace = trace_append(trace, state_fp, trace_word(w, nA, rule, g, h));
  }
  HD int enter(const Code& c, int* found) {
    nodes++;
    if (!anyA(c)) {
      int nt = nontrivial(c);
      record(c, nt ? 4 : 0);
      if (nt) { *found = w; on_found(c, w); if (w - 1 < cap) cap = w - 1; }
      return 0;
    }
    if (w + 1 > cap) { STABCERT_CUT_HOOK(c, *this, "weight"); record(c, 1); return 0; }
    int h = bound(c);
    if (c.coverage_bound && w+h<=cap) {
      int extra=syndrome_coverage_bound(c); if(extra>h)h=extra;
    }
    if (w + h > cap) { STABCERT_CUT_HOOK(c, *this, "packing"); record(c, 2, 65535, h); return 0; }
    int g = pick(c);
    record(c, 3, g);
    Frame& F = fr[depth++];
    F.g = (uint16_t)g; F.pos = (uint8_t)first_cand(c, g, 0); F.li = 0; F.clear_changes(); F.acur = 0;
    return 1;
  }

  // Produce the next child of the top frame. Returns 1 with the child state
  // applied, or 0 when the frame is exhausted (modifications still applied).
  HD int next_child(const Code& c) {
    Frame& F = fr[depth - 1];
    while (true) {
      int p = F.pos;
      int q = c.gq[F.g * c.W + p], gl = c.gl[F.g * c.W + p], a = lab[q];
      int opt[2]; int no = options(a, gl, opt);
      if (F.li < no) {
        int l = opt[F.li++];
        F.acur = (uint8_t)a;
        change(c, q, a, l);
        if (!dead) return 1;
        STABCERT_CUT_HOOK(c, *this, "dead");
        change(c, q, l, a);
        continue;
      }
      modify(c, F, p);
      if (dead) { STABCERT_CUT_HOOK(c, *this, "dead-suffix"); return 0; }
      int np = first_cand(c, F.g, p + 1);
      if (np < 0) return 0;
      F.pos = (uint8_t)np; F.li = 0;
    }
  }

  HD void undo_child(const Code& c) {
    Frame& F = fr[depth - 1];
    int q = c.gq[F.g * c.W + F.pos];
    change(c, q, lab[q], F.acur);
  }

  // Explore the subtree of the current node; stops when the stack returns to base.
  // Returns minimum found weight (or 1<<30). gcap: optional external cap reader.
  int aborted;
  template <class CapFn>
  HD int run(const Code& c, CapFn capfn, long long budget = (1LL << 62)) {
#ifdef STABCERT_RULES
    if (c.prune) return run_rules(c, capfn, budget);   // opt-in S0/S1/S2 policies (end of this struct)
#endif
    int base = depth, best = 1 << 30, f = -1;
    aborted = 0;
    if (!enter(c, &f)) { if (f >= 0 && f < best) best = f; return best; }
    long long chk = 0;
    while (depth > base) {
      if (nodes > budget) { aborted = 1; return best; }
      if (((++chk) & 1023) == 0) { int gc = capfn(); if (gc < cap) cap = gc; }
      if (next_child(c)) {
        f = -1;
        int e = enter(c, &f);
        if (f >= 0 && f < best) best = f;
        if (!e) undo_child(c);
        continue;
      }
      unmodify(c, fr[depth - 1]);
      depth--;
      if (depth > base) undo_child(c);
    }
    return best;
  }

  // hook: witness capture (driver copies supp/lab when called)
  int last_found_w, nwit;
  uint16_t wq[DMAX + 2];
  uint8_t wl[DMAX + 2];
  HD void on_found(const Code& c, int f) {
    if (last_found_w > 0 && f >= last_found_w) return;
    STABCERT_CHECKPOINT_HOOK("find-detected");
    last_found_w = f; nwit = nsupp;
    for (int i = 0; i < nsupp; i++) { wq[i] = supp[i]; wl[i] = lab[supp[i]]; }
    STABCERT_CHECKPOINT_HOOK("witness-captured");
  }

  // root: orbits < oi forced to I, rep labelled l. Returns 0 if dead.
  HD int root(const Code& c, const int* orbit, int rep, int oi, int l) {
    for (int q = 0; q < c.n; q++) if (orbit[q] < oi) change(c, q, lab[q], 0);
    change(c, rep, lab[rep], l);
    if (dead) STABCERT_CUT_HOOK(c, *this, "dead");
    return !dead;
  }

  // Replay a path of (pos, label index) decisions from the root node.
  // Returns 0 if the path is inconsistent (must not happen for generated paths).
  HD int replay(const Code& c, const uint8_t* path, int len) {
    int f = -1;
    for (int t = 0; t < len; t++) {
      int pos = path[t] >> 1, li = path[t] & 1;
      uint64_t saved_trace = trace;
#ifdef STABCERT_RULES
      if (!enter_any(c, &f)) return 0;
#else
      if (!enter(c, &f)) return 0;
#endif
      trace = saved_trace;
      nodes--;                               // replayed nodes belong to the generator
      Frame& F = fr[depth - 1];
      while (F.pos != pos) {
        modify(c, F, F.pos);
        if (dead) { STABCERT_CUT_HOOK(c, *this, "dead-suffix"); return 0; }
        int np = first_cand(c, F.g, F.pos + 1);
        if (np < 0) return 0;
        F.pos = (uint8_t)np;
      }
      F.li = (uint8_t)li;
      if (!next_child(c)) return 0;
      if (F.pos != pos || F.li != li + 1) return 0;
    }
    return 1;
  }

#ifdef STABCERT_RULES
  // ---- Opt-in S0/S1/S2 policies (docs/SOUNDNESS.md 5.1), compiled only into the CPU rules build
  // (-DSTABCERT_RULES; native variant suffix "-rules"). Every default build preprocesses to the
  // unchanged search. enter_rules()/run_rules() are enter()/run() with the fused rule pass (csp_prune)
  // in place of bound(), as in the reviewed -DFUSED build; callers select them with c.prune.
#include "s012.inc"
  HD int enter_any(const Code& c, int* found) {
    if (c.prune) return enter_rules(c, found);
    return enter(c, found);
  }
  int enter_rules(const Code& c, int* found) {
    nodes++;
    if (!anyA(c)) {
      int nt = nontrivial(c);
      record(c, nt ? 4 : 0);
      if (nt) { *found = w; on_found(c, w); if (w - 1 < cap) cap = w - 1; }
      return 0;
    }
    if (w + 1 > cap) { STABCERT_CUT_HOOK(c, *this, "weight"); record(c, 1); return 0; }
    int why = csp_prune(c);
#ifdef STABCERT_RULE_SCRATCH_HOOK
    STABCERT_RULE_SCRATCH_HOOK(c, *this);   // test builds: the cross-call rule scratch must be clean again
#endif
    // h_field = |PK| (BNB_CERT_SPEC_v2 section 12): r + 1, r, r - 1, r - 2; packed[0..h) is the packing PK
    int r = cap - w;
    if (why == 1) { STABCERT_CUT_HOOK(c, *this, "packing"); record(c, 2, 65535, r + 1); return 0; }
    if (why == 2) { STABCERT_CUT_HOOK(c, *this, "s0"); record(c, 5, 65535, r); return 0; }
    if (why == 3) { STABCERT_CUT_HOOK(c, *this, "s1"); record(c, 6, 65535, r - 1); return 0; }
    if (why == 4) { STABCERT_CUT_HOOK(c, *this, "s2"); record(c, 7, 65535, r - 2); return 0; }
    int g = pick(c);
    record(c, 3, g);
    Frame& F = fr[depth++];
    F.g = (uint16_t)g; F.pos = (uint8_t)first_cand(c, g, 0); F.li = 0; F.clear_changes(); F.acur = 0;
    return 1;
  }
  template <class CapFn>
  int run_rules(const Code& c, CapFn capfn, long long budget) {
    int base = depth, best = 1 << 30, f = -1;
    aborted = 0;
    if (!enter_rules(c, &f)) { if (f >= 0 && f < best) best = f; return best; }
    long long chk = 0;
    while (depth > base) {
      if (nodes > budget) { aborted = 1; return best; }
      if (((++chk) & 1023) == 0) { int gc = capfn(); if (gc < cap) cap = gc; }
      if (next_child(c)) {
        f = -1;
        int e = enter_rules(c, &f);
        if (f >= 0 && f < best) best = f;
        if (!e) undo_child(c);
        continue;
      }
      unmodify(c, fr[depth - 1]);
      depth--;
      if (depth > base) undo_child(c);
    }
    return best;
  }
#endif
};

#ifdef STABCERT_RULES
#include "s2.inc"
#endif
