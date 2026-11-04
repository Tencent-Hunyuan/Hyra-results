// Warp-cooperative GPU executor: one DFS per warp, the 32 lanes share the work of
// every node (generator updates, the anticommuting set, the packing bound, the
// branching choice). It explores exactly the tree of core.h (same branching
// rule, same bound, same tie-breaking, same AMAX truncation), so per-task node
// counts must equal the CPU executor's; validate.py checks this.
// Requires W <= 32 and C <= 64. All per-DFS state lives in shared memory.
#pragma once
#include "core.h"

#define FULLM 0xffffffffu
#ifndef DMAXW
#define DMAXW 48
#endif
#define LMAX MMAX

__device__ inline unsigned warp_sum(unsigned v) {
#if __CUDA_ARCH__ >= 800
  return __reduce_add_sync(FULLM, v);
#else
  for (int d = 16; d; d >>= 1) v += __shfl_xor_sync(FULLM, v, d);
  return v;
#endif
}
__device__ inline unsigned warp_min(unsigned v) {
#if __CUDA_ARCH__ >= 800
  return __reduce_min_sync(FULLM, v);
#else
  for (int d = 16; d; d >>= 1) { unsigned u = __shfl_xor_sync(FULLM, v, d); v = u < v ? u : v; }
  return v;
#endif
}

struct WarpDfs {
  const Code* cp;
  uint32_t* sb; uint32_t* used; uint32_t* cmk; uint32_t* fmk; uint16_t* lst; uint16_t* supp; uint16_t* pk; Frame* fr; uint8_t* lab; uint8_t* gcnt;
  int lane, n, m, W, C, nw;
  int w, nA, dead, nsupp, depth, cap, na, err;
  long long nodes;
  uint64_t state_fp, trace; // used only by lane zero
  int last_found_w, nwit;
  int* coverage_hist;

  __host__ __device__ static size_t bytes(int n, int m, bool coverage_bound = false) {
    size_t b = 0;
    b += 8 * (DMAXW + 2) * 3;                      // Frame (24 B, 8-aligned) first
    b += 8 * m;                                    // candidate / free position masks
    b += 4 * ((m + 31) / 32) + 4 * ((n + 31) / 32);
    b += 2 * LMAX + 2 * (DMAXW + 2) + 2 * (DMAXW + 2);
    b += n + m;
    if (coverage_bound) { b=(b+3)&~(size_t)3; b+=4*(CMAX+1); }
    return (b + 15) & ~(size_t)15;
  }
  __device__ void bind(uint8_t* base, const Code* c) {
    cp = c; n = c->n; m = c->m; W = c->W; C = c->C; nw = (m + 31) / 32;
    uint8_t* p = base;
    fr = (Frame*)p; p += 8 * (DMAXW + 2) * 3;
    cmk = (uint32_t*)p; p += 4 * m;
    fmk = (uint32_t*)p; p += 4 * m;
    sb = (uint32_t*)p; p += 4 * nw;
    used = (uint32_t*)p; p += 4 * ((n + 31) / 32);
    lst = (uint16_t*)p; p += 2 * LMAX;
    supp = (uint16_t*)p; p += 2 * (DMAXW + 2);
    pk = (uint16_t*)p; p += 2 * (DMAXW + 2);
    lab = p; p += n;
    gcnt = p; p += m;
    coverage_hist = c->coverage_bound ? (int*)(((uintptr_t)p+3)&~(uintptr_t)3) : nullptr;
    lane = threadIdx.x & 31;
  }

  __device__ __forceinline__ int sbit(int g) const { return (sb[g >> 5] >> (g & 31)) & 1; }

  // root (orbit oi, rep labelled l): equivalent to core.h root()
  __device__ int root(const int* orbit, int rep, int oi, int l, int cap0) {
    const Code& c = *cp;
    for (int q = lane; q < n; q += 32) lab[q] = orbit[q] < oi ? 0 : LFREE + c.domain;
    for (int i = lane; i < (n + 31) / 32; i += 32) used[i] = 0;
    for (int i = lane; i < nw; i += 32) sb[i] = 0;
    __syncwarp();
    for (int g = lane; g < m; g += 32) {
      uint32_t mk = 0, fm = 0;
      for (int p = 0; p < c.gw[g]; p++) {
        int a = lab[c.gq[g * W + p]];
        if (cand(a, c.gl[g * W + p])) mk |= 1u << p;
        if (a == LFREE) fm |= 1u << p;
      }
      gcnt[g] = (uint8_t)__popc(mk); cmk[g] = mk; fmk[g] = fm;
    }
    __syncwarp();
    w = 0; nA = 0; dead = 0; nsupp = 0; depth = 0; cap = cap0; nodes = 0; err = 0; last_found_w = 0; nwit = 0;
    state_fp = trace = 0;
    if (c.trace_enabled && lane == 0)
      for (int q = 0; q < n; q++) state_fp ^= trace_salt(q, lab[q]);
    change(rep, LFREE + c.domain, l);
    return !dead;
  }

  __device__ void change(int q, int a, int b) {
    const Code& c = *cp;
    if (c.trace_enabled && lane == 0) state_fp ^= trace_salt(q, a) ^ trace_salt(q, b);
    int deg = c.qc[q], dd = 0, dA = 0;
    for (int j = lane; j < deg; j += 32) {     // C <= 64: at most two generators per lane
      int g = c.qg[q * C + j], gl = c.ql[q * C + j];
      int s = sbit(g), cn = gcnt[g];
      int wd = s && cn == 0;
      cn += cand(b, gl) - cand(a, gl);
      int fa = nonI(a) && a != gl, fb = nonI(b) && b != gl;
      if (fa != fb) { atomicXor(&sb[g >> 5], 1u << (g & 31)); s ^= 1; dA += s ? 1 : -1; }
      gcnt[g] = (uint8_t)cn;
      uint32_t bit = 1u << c.qp[q * C + j];
      cmk[g] = cand(b, gl) ? (cmk[g] | bit) : (cmk[g] & ~bit);
      fmk[g] = (b == LFREE) ? (fmk[g] | bit) : (fmk[g] & ~bit);
      dd += (s && cn == 0) - wd;
    }
    dead += warp_sum((unsigned)dd);
    nA += warp_sum((unsigned)dA);
    if (lane == 0) lab[q] = (uint8_t)b;
    if (nonI(b) && !nonI(a)) { if (lane == 0) supp[nsupp] = (uint16_t)q; nsupp++; w++; }
    else if (nonI(a) && !nonI(b)) { nsupp--; w--; }
    __syncwarp();
  }

  __device__ int nontrivial() {
    const Code& c = *cp;
    int any = 0;
    for (int l0 = 0; l0 < c.nl; l0 += 32) {
      int l = l0 + lane, p = 0;
      if (l < c.nl) for (int i = 0; i < nsupp; i++) { int q = supp[i], b = c.L[l * n + q]; if (b && b != lab[q]) p ^= 1; }
      if (__ballot_sync(FULLM, p)) { any = 1; break; }
    }
    return any;
  }

  __device__ void collectA() {
    int tot = 0;
    for (int base = 0; base < nw; base += 32) {
      int i = base + lane;
      uint32_t word = i < nw ? sb[i] : 0;
      int cw = __popc(word), inc = cw;
      for (int off = 1; off < 32; off <<= 1) { int t = __shfl_up_sync(FULLM, inc, off); if (lane >= off) inc += t; }
      int pos = tot + inc - cw;
      while (word) { int g = i * 32 + __ffs(word) - 1; word &= word - 1; if (pos < LMAX) lst[pos] = (uint16_t)g; pos++; }
      tot += __shfl_sync(FULLM, inc, 31);
    }
    na = tot;
    if (na > LMAX) err = 1;
    __syncwarp();
  }

  __device__ int pick() {
    const Code& c = *cp;
    unsigned best = 0xffffffffu;
    int lim = na < LMAX ? na : LMAX;
    for (int i = lane; i < lim; i += 32) {
      int g = lst[i], ch = __popc(cmk[g]) + __popc(fmk[g]);
      unsigned key = ((unsigned)ch << 16) | (unsigned)g;
      if (key < best) best = key;
    }
    best = warp_min(best);
    return best & 0xffff;
  }

  __device__ int bound() {
    const Code& c = *cp;
    if (!c.greedy) return 0;
    if (w + nA <= cap) return nA > 0 ? 1 : 0;
    int nb = na < AMAX ? na : AMAX;
    // keys (cnt, g) of the first AMAX anticommuting generators, <= 3 per lane;
    // processing them by repeated warp-min is the sorted order of core.h
    unsigned key[3];
#pragma unroll
    for (int k = 0; k < 3; k++) { int i = lane + 32 * k; key[k] = i < nb ? (((unsigned)gcnt[lst[i]] << 16) | lst[i]) : 0xffffffffu; }
    int h = 0;
    for (int done = 0; done < nb; done++) {
      if (w + h + (nb - done) <= cap) break;          // cannot exceed the budget any more
      unsigned km = key[0] < key[1] ? key[0] : key[1]; km = km < key[2] ? km : key[2];
      km = warp_min(km);
#pragma unroll
      for (int k = 0; k < 3; k++) if (key[k] == km) key[k] = 0xffffffffu;
      int g = km & 0xffff;
      int q = 0, cd = (cmk[g] >> lane) & 1;
      if (cd) q = c.gq[g * W + lane];
      int conf = cd && ((used[q >> 5] >> (q & 31)) & 1);
      if (__ballot_sync(FULLM, conf)) continue;
      if (cd) atomicOr(&used[q >> 5], 1u << (q & 31));
      h++;
      __syncwarp();
      if (w + h > cap) break;
    }
    __syncwarp();
    if (h) { for (int i = lane; i < (n + 31) / 32; i += 32) used[i] = 0; __syncwarp(); }
    return h;
  }

  __device__ int first_cand(int g, int from) {
    const Code& c = *cp;
    uint32_t bal = from >= 32 ? 0u : (cmk[g] & ~((1u << from) - 1u));
    return bal ? __ffs(bal) - 1 : -1;
  }

  __device__ int syndrome_coverage_bound() {
    const Code& c=*cp;
    if (!c.coverage_bound) return 0;
    if (!c.flip_degree) return cap+1;
    int b1=(nA+c.flip_degree-1)/c.flip_degree;
    if (w+b1>cap) return b1;
    for(int i=lane;i<=c.C;i+=32) coverage_hist[i]=0;
    __syncwarp();
    for(int q=lane;q<n;q+=32) if(lab[q]>=LFREE) {
      int active=0, labels[4]={0,0,0,0};
      for(int j=0;j<c.qc[q];j++) if(sbit(c.qg[q*C+j])) {
        active++; labels[c.ql[q*C+j]]++;
      }
      int cover=0;
      if(lab[q]==LFREE) {
        for(int l=1;l<=3;l++) if(active-labels[l]>cover) cover=active-labels[l];
      } else cover=active-labels[lab[q]-LFREE];
      atomicAdd(&coverage_hist[cover],1);
    }
    __syncwarp();
    int result=0;
    if(lane==0) {
      int needed=nA,support=0;
      for(int cover=C;cover>0&&needed>0;cover--) {
        int take=(needed+cover-1)/cover;
        if(take>coverage_hist[cover])take=coverage_hist[cover];
        support+=take;needed-=take*cover;
      }
      result=needed>0?cap+1:(support>b1?support:b1);
    }
    return __shfl_sync(FULLM,result,0);
  }

  __device__ void on_found() {
    if (last_found_w > 0 && w >= last_found_w) return;
    last_found_w = w; nwit = nsupp;
  }

  // Fast path (na <= 96): one pass computes, per anticommuting generator, its
  // candidate mask, sort key (cnt, g) and child count. The packing is the same
  // greedy as core.h bound() (sorted order, take if disjoint from the taken
  // candidates, stop once w + h > cap) in the equivalent form "repeatedly take
  // the smallest-key generator not conflicting with the taken set".
  // Returns 1 if the node is pruned; else sets *gp to the branching generator.
  __device__ int bound_pick(int* gp) {
    const Code& c = *cp;
    unsigned key[3], pkey = 0xffffffffu; uint32_t cm[3];
#pragma unroll
    for (int k = 0; k < 3; k++) {
      int i = lane + 32 * k;
      key[k] = 0xffffffffu; cm[k] = 0;
      if (i < na) {
        int g = lst[i], ch = 0; uint32_t mk = 0;
        for (int p = 0; p < c.gw[g]; p++) {
          int lq = lab[c.gq[g * W + p]], gl = c.gl[g * W + p];
          if (lq == LFREE) { ch += 2; mk |= 1u << p; } else if (lq > LFREE && lq - LFREE != gl) { ch += 1; mk |= 1u << p; }
        }
        cm[k] = mk;
        if (i < AMAX) key[k] = ((unsigned)gcnt[g] << 16) | (unsigned)g;
        unsigned kk = ((unsigned)ch << 16) | (unsigned)g;
        if (kk < pkey) pkey = kk;
      }
    }
    pkey = warp_min(pkey);
    *gp = pkey & 0xffff;
    if (w + nA <= cap) return 0;                 // h <= |A| cannot exceed the budget
    int h = 0, pruned = 0;
    while (true) {
      if (h > 0) {
#pragma unroll
        for (int k = 0; k < 3; k++) if (key[k] != 0xffffffffu) {
          int g = key[k] & 0xffff; uint32_t mk = cm[k];
          while (mk) { int p = __ffs(mk) - 1; mk &= mk - 1; int q = c.gq[g * W + p]; if ((used[q >> 5] >> (q & 31)) & 1) { key[k] = 0xffffffffu; break; } }
        }
      }
      unsigned km = key[0] < key[1] ? key[0] : key[1]; km = km < key[2] ? km : key[2];
      km = warp_min(km);
      if (km == 0xffffffffu) break;
      int gs = km & 0xffff;
#pragma unroll
      for (int k = 0; k < 3; k++) if (key[k] == km) key[k] = 0xffffffffu;
      if (lane < c.gw[gs]) { int q = c.gq[gs * W + lane]; if (cand(lab[q], c.gl[gs * W + lane])) atomicOr(&used[q >> 5], 1u << (q & 31)); }
      if (lane == 0) pk[h] = (uint16_t)gs;
      h++;
      __syncwarp();
      if (w + h > cap) { pruned = 1; break; }
    }
    for (int i = 0; i < h; i++) {
      int g = pk[i];
      if (lane < c.gw[g]) { int q = c.gq[g * W + lane]; atomicAnd(&used[q >> 5], ~(1u << (q & 31))); }
    }
    __syncwarp();
    return pruned;
  }

  __device__ void record(int rule, int g = 65535, int h = 0) {
    if (cp->trace_enabled && lane == 0)
      trace = trace_append(trace, state_fp, trace_word(w, nA, rule, g, h));
  }
  __device__ int enter(int* found) {
    nodes++;
    if (nA == 0) {
      int nt = nontrivial();
      record(nt ? 4 : 0);
      if (nt) { *found = w; on_found(); if (w - 1 < cap) cap = w - 1; }
      return 0;
    }
    if (w + 1 > cap) { record(1); return 0; }
    collectA();
    int h = bound();
    if (cp->coverage_bound && w+h<=cap) {
      int extra=syndrome_coverage_bound(); if(extra>h)h=extra;
    }
    if (w + h > cap) { record(2, 65535, h); return 0; }
    int g = pick();
    record(3, g);
    int fc = first_cand(g, 0);
    if (depth >= DMAXW) { err = 2; return 0; }
    if (lane == 0) { Frame& F = fr[depth]; F.g = (uint16_t)g; F.pos = (uint8_t)fc; F.li = 0; F.modmask = 0; F.savedP = 0; F.acur = 0; }
    depth++;
    __syncwarp();
    return 1;
  }

  __device__ void modify(int p) {
    const Code& c = *cp;
    Frame& F = fr[depth - 1];
    int g = F.g;
    int q = c.gq[g * W + p], gl = c.gl[g * W + p], a = lab[q];
    __syncwarp();
    if (a == LFREE) change(q, a, LFREE + gl);
    else {
      if (lane == 0) F.savedP |= (uint64_t)(a - LFREE) << (2 * p);
      change(q, a, 0);
    }
    if (lane == 0) F.modmask |= 1u << p;
    __syncwarp();
  }

  __device__ void unmodify() {
    const Code& c = *cp;
    Frame& F = fr[depth - 1];
    int g = F.g; uint32_t mm = F.modmask; uint64_t sp = F.savedP;
    __syncwarp();
    for (int p = c.gw[g] - 1; p >= 0; p--) {
      if (!((mm >> p) & 1)) continue;
      int q = c.gq[g * W + p], a = lab[q];
      int orig = (a == 0) ? (int)(LFREE + ((sp >> (2 * p)) & 3)) : LFREE;
      change(q, a, orig);
    }
    if (lane == 0) { F.modmask = 0; F.savedP = 0; }
    __syncwarp();
  }

  __device__ int next_child() {
    const Code& c = *cp;
    while (true) {
      Frame& F = fr[depth - 1];
      int g = F.g, p = F.pos, li = F.li;
      int q = c.gq[g * W + p], gl = c.gl[g * W + p], a = lab[q];
      int no = (a == LFREE) ? 2 : 1;
      if (li < no) {
        int l;
        if (a == LFREE) { int k = 0; l = 0; for (int x = 1; x <= 3; x++) if (x != gl) { if (k == li) l = x; k++; } }
        else l = a - LFREE;
        __syncwarp();
        if (lane == 0) { F.li = (uint8_t)(li + 1); F.acur = (uint8_t)a; }
        __syncwarp();
        change(q, a, l);
        if (!dead) return 1;
        change(q, l, a);
        continue;
      }
      modify(p);
      if (dead) return 0;
      int np = first_cand(g, p + 1);
      if (np < 0) return 0;
      if (lane == 0) { F.pos = (uint8_t)np; F.li = 0; }
      __syncwarp();
    }
  }

  __device__ void undo_child() {
    const Code& c = *cp;
    Frame& F = fr[depth - 1];
    int q = c.gq[F.g * W + F.pos], ac = F.acur, cur = lab[q];
    __syncwarp();
    change(q, cur, ac);
  }

  __device__ int replay(const uint8_t* path, int len) {
    int f = -1;
    for (int t = 0; t < len; t++) {
      int pos = path[t] >> 1, li = path[t] & 1;
      uint64_t saved_trace = trace;
      if (!enter(&f)) return 0;
      trace = saved_trace;
      nodes--;
      while (true) {
        int fp = fr[depth - 1].pos;
        if (fp == pos) break;
        modify(fp);
        if (dead) return 0;
        int np = first_cand(fr[depth - 1].g, fp + 1);
        if (np < 0) return 0;
        if (lane == 0) fr[depth - 1].pos = (uint8_t)np;
        __syncwarp();
      }
      if (lane == 0) fr[depth - 1].li = (uint8_t)li;
      __syncwarp();
      if (!next_child()) return 0;
      if (fr[depth - 1].pos != pos || fr[depth - 1].li != li + 1) return 0;
    }
    return 1;
  }

  int aborted;
  __device__ int run(volatile int* gcap, long long budget) {
    int base = depth, best = 1 << 30, f = -1;
    aborted = 0;
    if (!enter(&f)) { if (f >= 0 && f < best) best = f; return best; }
    long long chk = 0;
    while (depth > base) {
      if (err) return best;
      if (nodes > budget) { aborted = 1; return best; }
      if (((++chk) & 1023) == 0) { int gc = __shfl_sync(FULLM, *gcap, 0); if (gc < cap) cap = gc; }
      if (next_child()) {
        f = -1;
        int e = enter(&f);
        if (f >= 0 && f < best) best = f;
        if (!e) undo_child();
        continue;
      }
      unmodify();
      depth--;
      if (depth > base) undo_child();
    }
    return best;
  }
};

__global__ void kern_warp(Code c, int cap0, const int* orbit, const int* reps, const uint32_t* troot, const uint8_t* tlen,
                          const uint8_t* tpath, int stride, long long t0, long long t1, unsigned long long* next, int* gcap,
                          long long* tnodes, uint64_t* ttraces, int* replay_fail, int* bw, int* bn, int* errflag, size_t wbytes, long long budget, int stopfirst) {
  extern __shared__ __align__(16) uint8_t smem[];
  int wid = threadIdx.x >> 5;
  int gw_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  WarpDfs st;
  st.bind(smem + wid * wbytes, &c);
  volatile int* vcap = gcap;
  int mybest = 1 << 30;
  while (true) {
    long long t = 0;
    if (st.lane == 0) t = (long long)atomicAdd(next, 1ULL) + t0;
    t = __shfl_sync(FULLM, t, 0);
    if (t >= t1) break;
    int oi = troot[t] >> 2, l = troot[t] & 3;
    long long nd = 0;
    if (st.root(orbit, reps[oi], oi, l, cap0)) {
      if (!st.replay(tpath + t * stride, tlen[t])) { if (st.lane == 0) atomicAdd(replay_fail, 1); }
      else {
        int gc = __shfl_sync(FULLM, *vcap, 0);
        st.cap = gc < cap0 ? gc : cap0;
        int b = st.run(vcap, budget);
        if (b < (1 << 30)) {
          if (st.lane == 0) atomicMin(gcap, stopfirst ? 0 : b - 1);
          if (st.last_found_w > 0 && st.last_found_w < mybest) {
            mybest = st.last_found_w;
            // the host re-runs this task on the CPU core to recover the witness
            if (st.lane == 0) { bw[gw_id] = mybest; bn[gw_id] = (int)t; }
          }
        }
        if (st.aborted) st.nodes = -1;
      }
      nd = st.nodes;
      if (st.err && st.lane == 0) atomicOr(errflag, st.err);
    }
    if (st.lane == 0) {
      tnodes[t - t0] = nd;
      if (c.trace_enabled) ttraces[t - t0] = st.trace;
    }
  }
}
