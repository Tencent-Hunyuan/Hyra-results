/*
 * pauli_ris -- GPU Pauli-weight distance search for general (non-CSS)
 * stabilizer codes.  Upper bounds only; every operator reported here is
 * re-verified on the CPU by pauli_ris.py before it is believed.
 *
 * Search space: rows of a basis W of a subspace V of F_2^{2n}, written as
 * Pauli vectors (x | z) (normally V = N(S) = ker(B | A), the normalizer).
 * A vector is scored by its Pauli weight |supp x U supp z| (a Y counts once)
 * and is a nontrivial logical iff it has odd overlap with some row of LS
 * (LS = logical basis with halves swapped, so overlap = symplectic product).
 *
 * Algorithm: qubit-aligned information-set decoding with a Markov chain of
 * information sets (Canteaut-Chabaud style).  Each CUDA block owns one copy
 * of W in shared memory.
 *   init:  Gauss-Jordan on W with columns visited QUBIT BY QUBIT in a random
 *          qubit order (x_q and z_q consecutively), so the information set is
 *          a union of whole qubits.  A logical of Pauli weight w is then a row
 *          when its support is one pivot qubit plus free qubits only, with
 *          probability ~ f^(w-1), f = free-qubit fraction -- independent of
 *          how many of its letters are Y.  Bit-level RIS on the doubled code
 *          needs all |supp x|+|supp z| bits free instead.
 *   step:  swap one free qubit with one pivot qubit (a rank-2 update: the
 *          two pivot rows of the leaving qubit are re-solved on the entering
 *          qubit's two columns; all other rows are updated by at most two
 *          row XORs).  Cost O(m * words) per new information set instead of
 *          O(m^2 * words) for a fresh elimination.
 *   check: every row of a pivot qubit pair (r_x, r_z) and r_x ^ r_z (the Y
 *          combination on the pivot qubit) is scored after each change.
 * Every `steps` chain steps the block re-initialises from a fresh random
 * qubit order.
 *
 * Qubits whose z (or x) column is identically zero on V are handled as
 * one-column qubits (class 1); class-1 and class-2 qubits are swapped within
 * their class.  Qubits left with one of two columns pivoted at init ("half")
 * stay fixed until the next init.
 *
 * Input file (little-endian):  magic "PAULIRS2", int32 n, nq, m, nl,
 *   uint8 colnz[2n] (1 if the column is nonzero on V), then W as m rows of
 *   2nq + ceil(nl/64) words: x words, z words (LSB-first), then the row's
 *   logical syndrome (bit l = overlap parity with LS row l), host-computed.
 * Output: JSON lines on stdout.
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <string>
#include <vector>

#include "input.h"
#include "../common/platform.h"

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err_ = (call);                                          \
        if (err_ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,          \
                    __LINE__, cudaGetErrorString(err_));                    \
            exit(3);                                                        \
        }                                                                   \
    } while (0)

#define INF_ROW 0x7fffffff

struct Params {
    int n, nq, rs, m, nl, nsw, nwa;
    int steps;          /* chain steps per round (between re-inits) */
    int rounds;         /* rounds per launch per block */
    int report_le;      /* ring-buffer threshold (0: off) */
    int ring_cap;
    int no_chain;       /* 1: plain RIS, re-init every round, no swaps */
    int hamming;        /* 1: score |x| + |z| (bits) instead of the Pauli weight */
    u64 seed;
    int gmem;
};

struct Globals {
    const u64* W;
    const uint8_t* colnz;
    int* gbest;
    u64* hits;          /* [n+1] nontrivial hits per weight */
    int* slotflag;      /* [n+1] */
    u64* slotvec;       /* [(n+1) * 2nq] first vector found per weight */
    int* ringcount;
    int* ringw;
    u64* ringvec;       /* [ring_cap * 2nq] */
    u64* stats;         /* 0 steps, 1 inits, 2 trivial checks, 3 failed swaps */
    u64* scratch;       /* global-memory matrices, one m*rs slab per block (gmem mode) */
};

__device__ __forceinline__ u64 splitmix64(u64 x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

__device__ __forceinline__ u64 xs64(u64* s) {
    u64 x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *s = x;
    return x;
}

__device__ __forceinline__ int col_word(int c, int n, int nq) {
    return c < n ? (c >> 6) : nq + ((c - n) >> 6);
}
__device__ __forceinline__ u64 col_mask(int c, int n) {
    return 1ULL << ((c < n ? c : c - n) & 63);
}
__device__ __forceinline__ int getbit(const u64* row, int c, int n, int nq) {
    return (row[col_word(c, n, nq)] & col_mask(c, n)) ? 1 : 0;
}

/* Column visited at position t of the init: qubit qperm[t>>1], x or z first
   by a per-init hash bit, so each qubit's two columns are consecutive. */
__device__ __forceinline__ int col_of(int t, const int16_t* qperm, int n, int salt) {
    int q = qperm[t >> 1];
    int first = (((unsigned)q * 2654435761u ^ (unsigned)salt) >> 7) & 1;
    return ((t & 1) ^ first) ? n + q : q;
}

/* Each row carries its logical syndrome (overlap parity with every LS row)
   in words [nw, nw + nsw); the map is linear, so row operations keep it
   exact.  Nontrivial iff the syndrome is nonzero.  b may be null. */
__device__ __forceinline__ int is_nontrivial(const u64* a, const u64* b, int nw, int nsw) {
    u64 acc = 0;
    if (b) for (int i = nw; i < nw + nsw; i++) acc |= a[i] ^ b[i];
    else   for (int i = nw; i < nw + nsw; i++) acc |= a[i];
    return acc != 0;
}

__device__ void report(const Params& P, const Globals& G, int w,
                       const u64* a, const u64* b) {
    int nw = 2 * P.nq;
    atomicAdd(&G.hits[w], 1ULL);
    atomicMin(G.gbest, w);
    if (atomicCAS(&G.slotflag[w], 0, 1) == 0) {
        u64* dst = G.slotvec + (size_t)w * nw;
        for (int i = 0; i < nw; i++) dst[i] = b ? (a[i] ^ b[i]) : a[i];
        __threadfence();
        atomicExch(&G.slotflag[w], 2);
    }
    /* the ring counter only advances while below capacity (it would
       overflow int32 within a minute on small codes otherwise) */
    if (P.report_le > 0 && w <= P.report_le &&
        *(volatile int*)G.ringcount < P.ring_cap) {
        int idx = atomicAdd(G.ringcount, 1);
        if (idx >= 0 && idx < P.ring_cap) {
            u64* dst = G.ringvec + (size_t)idx * nw;
            for (int i = 0; i < nw; i++) dst[i] = b ? (a[i] ^ b[i]) : a[i];
            G.ringw[idx] = w;
        }
    }
}

/* Score a slot (one or two rows) and its Y combination; report hits. */
__device__ void check_slot(const Params& P, const Globals& G, const u64* A,
                           const u64* B, int thresh, u64* trivial) {
    int nq = P.nq, nw = 2 * nq;
    int wa = 0, wb = 0, wab = 0;
    for (int i = 0; i < nq; i++) {
        u64 ax = A[i], az = A[nq + i];
        wa += P.hamming ? __popcll(ax) + __popcll(az) : __popcll(ax | az);
        if (B) {
            u64 bx = B[i], bz = B[nq + i];
            if (P.hamming) {
                wb += __popcll(bx) + __popcll(bz);
                wab += __popcll(ax ^ bx) + __popcll(az ^ bz);
            } else {
                wb += __popcll(bx | bz);
                wab += __popcll((ax ^ bx) | (az ^ bz));
            }
        }
    }
    if (wa > 0 && wa <= thresh) {
        if (is_nontrivial(A, nullptr, nw, P.nsw)) report(P, G, wa, A, nullptr);
        else (*trivial)++;
    }
    if (B) {
        if (wb > 0 && wb <= thresh) {
            if (is_nontrivial(B, nullptr, nw, P.nsw)) report(P, G, wb, B, nullptr);
            else (*trivial)++;
        }
        if (wab > 0 && wab <= thresh) {
            if (is_nontrivial(A, B, nw, P.nsw)) report(P, G, wab, A, B);
            else (*trivial)++;
        }
    }
}

extern "C" __global__ void pauli_chain_kernel(Params P, Globals Gl, int launch) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int n = P.n, nq = P.nq, rs = P.rs, m = P.m, nwa = P.nwa;
    const int tid = threadIdx.x, T = blockDim.x;

    /* shared layout */
    u64* G;
    u64* buf;
    if (P.gmem) { G = Gl.scratch + (size_t)blockIdx.x * m * rs; buf = (u64*)smem; }
    else { G = (u64*)smem; buf = G + (size_t)m * rs; }
    int* ctrl = (int*)(buf + 2 * rs);                     /* 32 ints */
    int16_t* piv = (int16_t*)(ctrl + 32);                 /* m */
    int16_t* colrow = piv + m;                            /* 2n */
    int16_t* qperm = colrow + 2 * n;                      /* n */
    int16_t* fl = qperm + n;                              /* 2 * n: free lists */
    int16_t* pl = fl + 2 * n;                             /* 2 * n: pivot lists */
    int16_t* sa = pl + 2 * n;                             /* m: slot row a */
    int16_t* sb = sa + m;                                 /* m: slot row b */
    /* ctrl: 0..2 found ring, 3 nslots, 4 thresh, 5 r1, 6 r2, 7 cin1, 8 cin2,
       9 ok, 10/11 nf[0..1], 12/13 np[0..1] */

    u64 rng = splitmix64(P.seed ^ splitmix64(((u64)launch << 32) ^ (u64)blockIdx.x));
    u64 trivial = 0, failed = 0, steps_done = 0, inits = 0;

    for (int round = 0; round < P.rounds; round++) {
        /* ---------------- init: qubit-aligned Gauss-Jordan ---------------- */
        for (size_t i = tid; i < (size_t)m * nwa; i += T) {
            size_t r = i / nwa, w = i % nwa;
            G[r * rs + w] = __ldg(Gl.W + i);
        }
        for (int s = tid; s < m; s += T) piv[s] = -1;
        if (tid == 0) {
            for (int i = 0; i < n; i++) qperm[i] = (int16_t)i;
            for (int i = n - 1; i > 0; i--) {
                int j = (int)(xs64(&rng) % (u64)(i + 1));
                int16_t t = qperm[i]; qperm[i] = qperm[j]; qperm[j] = t;
            }
            ctrl[0] = INF_ROW; ctrl[1] = INF_ROW; ctrl[2] = INF_ROW;
            ctrl[4] = *(volatile int*)Gl.gbest;
            if (P.report_le > ctrl[4]) ctrl[4] = P.report_le;
        }
        u64 salt = 0;
        if (tid == 0) salt = xs64(&rng);
        /* broadcast salt via ctrl slots 14/15 */
        if (tid == 0) { ctrl[14] = (int)(salt & 0x7fffffff); }
        __syncthreads();
        int csalt = ctrl[14];
        {
            int c0 = col_of(0, qperm, n, csalt);
            for (int s = tid; s < m; s += T)
                if (getbit(G + (size_t)s * rs, c0, n, nq)) atomicMin(&ctrl[0], s);
        }
        __syncthreads();
        int npiv = 0;
        for (int t = 0; t < 2 * n && npiv < m; t++) {
            int c = col_of(t, qperm, n, csalt);
            int r = ctrl[t % 3];
            int has_next = (t + 1 < 2 * n);
            int cn = has_next ? col_of(t + 1, qperm, n, csalt) : 0;
            if (tid == 0) ctrl[(t + 2) % 3] = INF_ROW;
            if (r != INF_ROW) {
                const u64* pr = G + (size_t)r * rs;
                int cw = col_word(c, n, nq);
                u64 cm = col_mask(c, n);
                for (int s = tid; s < m; s += T) {
                    if (s == r) continue;
                    u64* row = G + (size_t)s * rs;
                    if (row[cw] & cm)
                        for (int i = 0; i < nwa; i++) row[i] ^= pr[i];
                }
                if (tid == r % T) piv[r] = (int16_t)c;
                npiv++;
            }
            if (has_next) {
                for (int s = tid; s < m; s += T) {
                    if (piv[s] >= 0 || s == r) continue;
                    if (getbit(G + (size_t)s * rs, cn, n, nq)) atomicMin(&ctrl[(t + 1) % 3], s);
                }
            }
            __syncthreads();
        }
        /* bookkeeping (thread 0): colrow, qubit lists, slots */
        if (tid == 0) {
            for (int c = 0; c < 2 * n; c++) colrow[c] = -1;
            for (int s = 0; s < m; s++) if (piv[s] >= 0) colrow[piv[s]] = (int16_t)s;
            int nf0 = 0, nf1 = 0, np0 = 0, np1 = 0, ns = 0;
            for (int q = 0; q < n; q++) {
                int cx = Gl.colnz[q], cz = Gl.colnz[n + q];
                int cls = cx + cz;
                if (cls == 0) continue;
                int px = colrow[q] >= 0, pz = colrow[n + q] >= 0;
                int npq = px + pz;
                if (npq == 0) {
                    if (cls == 1) fl[nf0++] = (int16_t)q; else fl[n + nf1++] = (int16_t)q;
                } else if (npq == cls) {
                    if (cls == 1) {
                        pl[np0++] = (int16_t)q;
                        sa[ns] = colrow[px ? q : n + q]; sb[ns] = -1; ns++;
                    } else {
                        pl[n + np1++] = (int16_t)q;
                        sa[ns] = colrow[q]; sb[ns] = colrow[n + q]; ns++;
                    }
                } else {                       /* half qubit: fixed singleton */
                    sa[ns] = colrow[px ? q : n + q]; sb[ns] = -1; ns++;
                }
            }
            ctrl[3] = ns; ctrl[10] = nf0; ctrl[11] = nf1; ctrl[12] = np0; ctrl[13] = np1;
        }
        __syncthreads();
        inits++;
        const int ns = ctrl[3];
        {
            int thresh = ctrl[4];
            for (int j = tid; j < ns; j += T) {
                const u64* A = G + (size_t)sa[j] * rs;
                const u64* B = sb[j] >= 0 ? G + (size_t)sb[j] * rs : nullptr;
                check_slot(P, Gl, A, B, thresh, &trivial);
            }
        }
        __syncthreads();
        if (P.no_chain) continue;

        /* ---------------- chain of qubit swaps ---------------- */
        for (int step = 0; step < P.steps; step++) {
            if (tid == 0) {
                int nf0 = ctrl[10], nf1 = ctrl[11], np0 = ctrl[12], np1 = ctrl[13];
                int ok = 0;
                int e0 = (nf0 > 0 && np0 > 0) ? nf0 : 0;
                int e1 = (nf1 > 0 && np1 > 0) ? nf1 : 0;
                if (e0 + e1 > 0) {
                    int cls = ((int)(xs64(&rng) % (u64)(e0 + e1)) < e1) ? 2 : 1;
                    int nf = cls == 2 ? nf1 : nf0, np_ = cls == 2 ? np1 : np0;
                    int16_t* F = cls == 2 ? fl + n : fl;
                    int16_t* Pp = cls == 2 ? pl + n : pl;
                    int fi = (int)(xs64(&rng) % (u64)nf);
                    int qin = F[fi];
                    for (int tries = 0; tries < 16 && !ok; tries++) {
                        int pi = (int)(xs64(&rng) % (u64)np_);
                        int qout = Pp[pi];
                        if (cls == 2) {
                            int r1 = colrow[qout], r2 = colrow[n + qout];
                            const u64* R1 = G + (size_t)r1 * rs;
                            const u64* R2 = G + (size_t)r2 * rs;
                            int a = getbit(R1, qin, n, nq), b = getbit(R1, n + qin, n, nq);
                            int c = getbit(R2, qin, n, nq), d = getbit(R2, n + qin, n, nq);
                            if ((a & d) ^ (b & c)) {
                                /* new r1 = d r1 + b r2 ; new r2 = c r1 + a r2 */
                                for (int i = 0; i < nwa; i++) {
                                    u64 x1 = R1[i], x2 = R2[i];
                                    buf[i] = (d ? x1 : 0) ^ (b ? x2 : 0);
                                    buf[rs + i] = (c ? x1 : 0) ^ (a ? x2 : 0);
                                }
                                colrow[qout] = -1; colrow[n + qout] = -1;
                                colrow[qin] = (int16_t)r1; colrow[n + qin] = (int16_t)r2;
                                piv[r1] = (int16_t)qin; piv[r2] = (int16_t)(n + qin);
                                F[fi] = (int16_t)qout; Pp[pi] = (int16_t)qin;
                                ctrl[5] = r1; ctrl[6] = r2; ctrl[7] = qin; ctrl[8] = n + qin;
                                ok = 1;
                            }
                        } else {
                            int cout = colrow[qout] >= 0 ? qout : n + qout;
                            int cin = Gl.colnz[qin] ? qin : n + qin;
                            int r1 = colrow[cout];
                            const u64* R1 = G + (size_t)r1 * rs;
                            if (getbit(R1, cin, n, nq)) {
                                for (int i = 0; i < nwa; i++) buf[i] = R1[i];
                                colrow[cout] = -1; colrow[cin] = (int16_t)r1;
                                piv[r1] = (int16_t)cin;
                                F[fi] = (int16_t)qout; Pp[pi] = (int16_t)qin;
                                ctrl[5] = r1; ctrl[6] = -1; ctrl[7] = cin; ctrl[8] = -1;
                                ok = 1;
                            }
                        }
                    }
                }
                if (!ok) failed++;
                ctrl[9] = ok;
                if ((step & 15) == 0) {
                    int gb = *(volatile int*)Gl.gbest;
                    ctrl[4] = P.report_le > gb ? P.report_le : gb;
                }
            }
            __syncthreads();
            if (ctrl[9]) {
                const int r1 = ctrl[5], r2 = ctrl[6], c1 = ctrl[7], c2 = ctrl[8];
                const int thresh = ctrl[4];
                const int w1 = col_word(c1, n, nq);
                const u64 m1 = col_mask(c1, n);
                const int w2 = c2 >= 0 ? col_word(c2, n, nq) : 0;
                const u64 m2 = c2 >= 0 ? col_mask(c2, n) : 0;
                for (int j = tid; j < ns; j += T) {
                    int rows[2] = {sa[j], sb[j]};
                    int changed = 0;
                    for (int h = 0; h < 2; h++) {
                        int r = rows[h];
                        if (r < 0) continue;
                        u64* row = G + (size_t)r * rs;
                        if (r == r1) {
                            for (int i = 0; i < nwa; i++) row[i] = buf[i];
                            changed = 1;
                        } else if (r == r2) {
                            for (int i = 0; i < nwa; i++) row[i] = buf[rs + i];
                            changed = 1;
                        } else {
                            int s1 = (row[w1] & m1) ? 1 : 0;
                            int s2 = (c2 >= 0 && (row[w2] & m2)) ? 1 : 0;
                            if (s1 | s2) {
                                if (s1 && s2) for (int i = 0; i < nwa; i++) row[i] ^= buf[i] ^ buf[rs + i];
                                else if (s1)  for (int i = 0; i < nwa; i++) row[i] ^= buf[i];
                                else          for (int i = 0; i < nwa; i++) row[i] ^= buf[rs + i];
                                changed = 1;
                            }
                        }
                    }
                    if (changed) {
                        const u64* A = G + (size_t)rows[0] * rs;
                        const u64* B = rows[1] >= 0 ? G + (size_t)rows[1] * rs : nullptr;
                        check_slot(P, Gl, A, B, thresh, &trivial);
                    }
                }
                steps_done++;
            }
            __syncthreads();
        }
    }
    /* stats */
    atomicAdd(&Gl.stats[2], trivial);
    if (tid == 0) {
        atomicAdd(&Gl.stats[0], steps_done);
        atomicAdd(&Gl.stats[1], inits);
        atomicAdd(&Gl.stats[3], failed);
    }
}

/* ------------------------------------------------------------------ */
/*  Host                                                               */
/* ------------------------------------------------------------------ */

static double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void print_vec(const u64* v, int n, int nq) {
    printf("\"x\":[");
    int first = 1;
    for (int q = 0; q < n; q++)
        if ((v[q >> 6] >> (q & 63)) & 1) { printf(first ? "%d" : ",%d", q); first = 0; }
    printf("],\"z\":[");
    first = 1;
    for (int q = 0; q < n; q++)
        if ((v[nq + (q >> 6)] >> (q & 63)) & 1) { printf(first ? "%d" : ",%d", q); first = 0; }
    printf("]");
}

static void usage(const char* a0) {
    fprintf(stderr,
        "usage: %s <input.prs> [--seconds S] [--max-steps N] [--steps-per-round K]\n"
        "        [--rounds R] [--threads T] [--seed S] [--target D] [--report-le W]\n"
        "        [--ring-cap C] [--blocks B] [--no-chain] [--device D] [--quiet]\n", a0);
    exit(2);
}

static int main_impl(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--version") {
        stabcert::version("stabcert-ris-cuda", "{\"NMAX\":16000,\"global_memory_fallback\":true}");
        return 0;
    }
    const char* path = nullptr;
    double seconds = 10.0;
    long long max_steps = 0;
    int steps_per_round = -1, rounds = 1, threads = 0, target = -1, report_le = 0;
    int ring_cap = 4096, blocks = 0, no_chain = 0, device = 0, quiet = 0, hamming = 0, force_gmem = 0;
    u64 seed = 1;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nxt = [&]() { if (i + 1 >= argc) usage(argv[0]); return argv[++i]; };
        if (a == "--seconds") seconds = ris_real(nxt());
        else if (a == "--max-steps") max_steps = ris_integer(nxt());
        else if (a == "--steps-per-round") steps_per_round = ris_int(nxt());
        else if (a == "--rounds") rounds = ris_int(nxt());
        else if (a == "--threads") threads = ris_int(nxt());
        else if (a == "--seed") seed = (u64)ris_integer(nxt());
        else if (a == "--target") target = ris_int(nxt());
        else if (a == "--report-le") report_le = ris_int(nxt());
        else if (a == "--ring-cap") ring_cap = ris_int(nxt());
        else if (a == "--blocks") blocks = ris_int(nxt());
        else if (a == "--no-chain") no_chain = 1;
        else if (a == "--hamming") hamming = 1;
        else if (a == "--gmem") force_gmem = 1;
        else if (a == "--device") device = ris_int(nxt());
        else if (a == "--quiet") quiet = 1;
        else if (a[0] == '-') usage(argv[0]);
        else path = argv[i];
    }
    if (!path) usage(argv[0]);
    if (max_steps < 0 || rounds < 1 || steps_per_round < -1 || threads < 0 || threads > 1024 || (threads && threads%32) || ring_cap < 1 || blocks < 0 || device < 0 || report_le < 0)
        throw std::invalid_argument("invalid rounds/steps/threads/ring-cap/blocks/device/report-le");
    RisInput input = read_ris(path);
    int n = input.n, nq = input.nq, m = input.m, nl = input.nl;
    int nw = input.nw, nsw = input.nsw, nwa = input.stride;
    auto& colnz = input.colnz; auto& W = input.rows;

    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int rs = nwa | 1;   /* odd row stride: fewer bank conflicts */
    size_t shmem = (size_t)m * rs * 8 + 2 * rs * 8 + 32 * 4 +
                   2 * (size_t)(m + 2 * n + n + 2 * n + 2 * n + 2 * m);
    shmem = (shmem + 15) & ~(size_t)15;
    int max_shmem = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_shmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    int gmem = 0;
    if ((int)shmem > max_shmem || force_gmem) {
        gmem = 1;
        shmem = 2 * rs * 8 + 32 * 4 + 2 * (size_t)(m + 2 * n + n + 2 * n + 2 * n + 2 * m);
        shmem = (shmem + 15) & ~(size_t)15;
        if ((int)shmem > max_shmem) {
            fprintf(stderr, "n=%d needs %zu bytes shared memory even in global-memory mode (device max %d)\n", n, shmem, max_shmem);
            return 2;
        }
    }
    CUDA_CHECK(cudaFuncSetAttribute(pauli_chain_kernel,
               cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem));
    if (threads <= 0) {
        int slots = (m + 1) / 2 + 8;
        threads = ((slots + 31) / 32) * 32;
        if (threads > 256) threads = 256;
        if (threads < 32) threads = 32;
    }
    int per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, pauli_chain_kernel,
                                                             threads, shmem));
    if (per_sm < 1) { fprintf(stderr, "kernel does not fit\n"); return 2; }
    if (gmem && per_sm > 2) per_sm = 2;
    if (blocks <= 0) blocks = per_sm * prop.multiProcessorCount;
    if (steps_per_round < 0) steps_per_round = 20 * m;

    Params P;
    P.n = n; P.nq = nq; P.rs = rs; P.m = m; P.nl = nl; P.nsw = nsw; P.nwa = nwa;
    P.steps = steps_per_round; P.rounds = rounds; P.report_le = report_le;
    P.ring_cap = ring_cap; P.no_chain = no_chain; P.seed = seed; P.hamming = hamming;
    P.gmem = gmem;

    Globals Gd;
    u64 *dW, *dhits, *dslotvec, *dringvec, *dstats;
    uint8_t* dcolnz;
    int *dbest, *dslotflag, *dringcount, *dringw;
    CUDA_CHECK(cudaMalloc(&dW, W.size() * 8));
    CUDA_CHECK(cudaMalloc(&dcolnz, 2 * n));
    CUDA_CHECK(cudaMalloc(&dbest, 4));
    const int WMAX = hamming ? 2 * n : n;
    CUDA_CHECK(cudaMalloc(&dhits, (2 * n + 1) * 8));
    CUDA_CHECK(cudaMalloc(&dslotflag, (2 * n + 1) * 4));
    CUDA_CHECK(cudaMalloc(&dslotvec, (size_t)(2 * n + 1) * nw * 8));
    CUDA_CHECK(cudaMalloc(&dringcount, 4));
    CUDA_CHECK(cudaMalloc(&dringw, (size_t)ring_cap * 4));
    CUDA_CHECK(cudaMalloc(&dringvec, (size_t)ring_cap * nw * 8));
    CUDA_CHECK(cudaMalloc(&dstats, 8 * 8));
    CUDA_CHECK(cudaMemcpy(dW, W.data(), W.size() * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dcolnz, colnz.data(), 2 * n, cudaMemcpyHostToDevice));
    int init_best = WMAX + 1;
    CUDA_CHECK(cudaMemcpy(dbest, &init_best, 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dhits, 0, (2 * n + 1) * 8));
    CUDA_CHECK(cudaMemset(dslotflag, 0, (2 * n + 1) * 4));
    CUDA_CHECK(cudaMemset(dringcount, 0, 4));
    CUDA_CHECK(cudaMemset(dstats, 0, 64));
    Gd.W = dW; Gd.colnz = dcolnz; Gd.gbest = dbest; Gd.hits = dhits;
    Gd.slotflag = dslotflag; Gd.slotvec = dslotvec; Gd.ringcount = dringcount;
    Gd.ringw = dringw; Gd.ringvec = dringvec; Gd.stats = dstats;
    Gd.scratch = nullptr;
    if (gmem) CUDA_CHECK(cudaMalloc(&Gd.scratch, (size_t)blocks * m * rs * 8));

    printf("{\"event\":\"start\",\"n\":%d,\"m\":%d,\"nl\":%d,\"nq\":%d,\"threads\":%d,"
           "\"blocks\":%d,\"blocks_per_sm\":%d,\"shmem\":%zu,\"steps_per_round\":%d,"
           "\"rounds\":%d,\"no_chain\":%d,\"seed\":%llu,\"device\":\"%s\",\"gmem\":%d}\n",
           n, m, nl, nq, threads, blocks, per_sm, shmem, steps_per_round, rounds,
           no_chain, seed, prop.name, gmem);
    fflush(stdout);

    double t0 = now_s();
    int launch = 0, best = WMAX + 1, reported_best = WMAX + 1;
    u64 stats[8] = {0};
    while (now_s() - t0 < seconds) {
        pauli_chain_kernel<<<blocks, threads, shmem>>>(P, Gd, launch);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        launch++;
        CUDA_CHECK(cudaMemcpy(stats, dstats, 64, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&best, dbest, 4, cudaMemcpyDeviceToHost));
        if (best < reported_best) {
            std::vector<u64> witness(nw);
            CUDA_CHECK(cudaMemcpy(witness.data(), dslotvec + (size_t)best*nw, nw*8, cudaMemcpyDeviceToHost));
            printf("{\"event\":\"witness\",\"w\":%d,", best);
            print_vec(witness.data(), n, nq); printf("}\n"); fflush(stdout);
            reported_best = best;
        }
        double el = now_s() - t0;
        if (!quiet)
            printf("{\"event\":\"progress\",\"t\":%.2f,\"launches\":%d,\"steps\":%llu,"
                   "\"inits\":%llu,\"best\":%d}\n", el, launch, stats[0], stats[1],
                   best <= WMAX ? best : -1);
        fflush(stdout);
        if (target >= 0 && best < target) break;
        if (el >= seconds) break;
        if (max_steps > 0 && (no_chain || !steps_per_round ? stats[1] : stats[0] + stats[3]) >= (u64)max_steps) break;
    }
    double el = now_s() - t0;
    std::vector<u64> hits(WMAX + 1);
    std::vector<int> sflag(WMAX + 1);
    CUDA_CHECK(cudaMemcpy(hits.data(), dhits, (WMAX + 1) * 8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sflag.data(), dslotflag, (WMAX + 1) * 4, cudaMemcpyDeviceToHost));
    std::vector<u64> svec((size_t)(WMAX + 1) * nw);
    CUDA_CHECK(cudaMemcpy(svec.data(), dslotvec, svec.size() * 8, cudaMemcpyDeviceToHost));
    int rc = 0;
    CUDA_CHECK(cudaMemcpy(&rc, dringcount, 4, cudaMemcpyDeviceToHost));
    if (report_le > 0 && rc > 0) {
        int cnt = rc < ring_cap ? rc : ring_cap;
        std::vector<int> rw(cnt);
        std::vector<u64> rv((size_t)cnt * nw);
        CUDA_CHECK(cudaMemcpy(rw.data(), dringw, cnt * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(rv.data(), dringvec, rv.size() * 8, cudaMemcpyDeviceToHost));
        for (int i = 0; i < cnt; i++) {
            printf("{\"event\":\"ring\",\"w\":%d,", rw[i]);
            print_vec(rv.data() + (size_t)i * nw, n, nq);
            printf("}\n");
        }
    }
    printf("{\"event\":\"done\",\"interface\":1,\"seconds\":%.3f,\"launches\":%d,\"steps\":%llu,"
           "\"inits\":%llu,\"attempted_steps\":%llu,\"trivial_checks\":%llu,\"failed_swaps\":%llu,\"best\":%d,"
           "\"ring_total\":%d,\"hits_by_weight\":{", el, launch, stats[0], stats[1], stats[0]+stats[3], stats[2], stats[3],
           best <= WMAX ? best : -1, rc);
    int first = 1;
    for (int w = 0; w <= WMAX; w++)
        if (hits[w]) { printf(first ? "\"%d\":%llu" : ",\"%d\":%llu", w, hits[w]); first = 0; }
    printf("}}\n");
    return 0;
}

int main(int argc, char** argv) {
    try { return main_impl(argc, argv); }
    catch (const std::invalid_argument& e) { fprintf(stderr, "input error: %s\n", e.what()); return 2; }
    catch (const std::out_of_range& e) { fprintf(stderr, "input error: %s\n", e.what()); return 2; }
    catch (const std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
}
