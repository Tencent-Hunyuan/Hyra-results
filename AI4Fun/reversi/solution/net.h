// ============================================================================
// net.h - the policy+value network for the "az_mcts" AlphaZero Reversi solution,
// plus the feature extraction shared by self-play (emit features), the trainer
// (fits weights to those features), and the match bot (uses the trained weights).
//
// WHY THIS SHAPE (the CPU-AlphaZero constraint). On a bare `g++ -O2` match host a
// dense / conv net forward costs micro-to-milliseconds, capping MCTS at a few
// thousand sims - too few to out-search a depth-12 alpha-beta. The only CPU-fast
// strong Othello evaluator is a PER-PHASE LINEAR PATTERN MODEL (Logistello /
// Edax lineage): the joint occupancy of a small cell-window indexes a learned
// weight, summed over all symmetric instances. One eval is ~100-200 ns, so MCTS
// gets tens of thousands of sims per move. We therefore build the AlphaZero net on
// pattern features:
//
//   * VALUE head  v(s) in (-1,1): per-phase scalar pattern tables
//     + a few dense scalar features, summed and tanh-squashed. Trained on the
//     AlphaZero target z (game result from the side-to-move perspective).
//   * POLICY head P(s,a): PUCT priors over legal moves from a learned
//     per-(phase,square) bias plus a handful of cheap, board-derived MOVE features
//     (flips, corner-capture, opponent mobility after the move, …). Trained on the
//     MCTS visit-count distribution (the pi . log p term).
//
// This is a genuinely different ALGORITHM from the classic pattern-eval + alpha-beta
// Othello engines (e.g. ridge-regressed or TD(lambda) pattern values searched with
// alpha-beta). Here the policy improvement operator is PUCT MCTS, targets are visit
// counts + outcomes, and the search at match time is MCTS, not alpha-beta - while the
// exact endgame solver in board.h handles the last plies where MCTS would otherwise
// be weakest.
//
// PARITY-SAFE TRAINING. Self-play emits the EXACT integer pattern feature ids and
// float move features that this file computes; the PyTorch trainer fits weights to
// those features and recomputes NOTHING from the board. So the only way C++ and
// Python can disagree is the trivial "sum table[id]" arithmetic, which net_selfcheck
// verifies. The bot then uses these same net.h routines, so bot == trainer by
// construction.
// ============================================================================
#ifndef AZ_MCTS_NET_H
#define AZ_MCTS_NET_H

#include "board.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>

namespace az {

// ───────────────────────── pattern feature system ──────────────────────────
// Pattern machinery: a pattern TYPE has one shared weight table; its 8
// dihedral images (deduped by ordered tuple) all read/write that one table, so
// the table learns one orientation and generalises across the board.
static const int NPHASE = 10;       // game phases by disc count
static const int MAXK = 10;         // longest pattern (cells)
static const int MAXMV = 34;        // max legal moves per position (Othello ceiling)

struct PatType {
    int k;
    int size;                       // 3^k
    int ninst;
    int cell[16][MAXK];
    int pw[MAXK];                   // base-3 place values
};

static std::vector<PatType> g_pat;
static int g_npat = 0;
static int g_pat_off[NPHASE][32];   // offset of each type's table within a phase block
static int g_phase_block = 0;       // total table entries per phase
static int g_ninst_total = 0;       // sum of instances over all types (feature count)
static const int MAXFEAT = 256;     // safe upper bound on per-position feature ids

static inline int dih(int t, int r, int c) {
    int nr = r, nc = c;
    switch (t) {
        case 0: nr = r;     nc = c;     break;
        case 1: nr = c;     nc = 7 - r; break;
        case 2: nr = 7 - r; nc = 7 - c; break;
        case 3: nr = 7 - c; nc = r;     break;
        case 4: nr = r;     nc = 7 - c; break;
        case 5: nr = 7 - c; nc = 7 - r; break;
        case 6: nr = 7 - r; nc = c;     break;
        case 7: nr = c;     nc = r;     break;
    }
    return nr * 8 + nc;
}

static void add_pattern(const std::vector<int> &base) {
    PatType pt;
    pt.k = (int)base.size();
    pt.size = 1; for (int i = 0; i < pt.k; i++) pt.size *= 3;
    for (int j = 0; j < pt.k; j++) { int p = 1; for (int q = 0; q < pt.k - 1 - j; q++) p *= 3; pt.pw[j] = p; }
    std::vector<std::vector<int>> inst;
    for (int t = 0; t < 8; t++) {
        std::vector<int> cur(pt.k);
        for (int j = 0; j < pt.k; j++) { int r = base[j] / 8, c = base[j] % 8; cur[j] = dih(t, r, c); }
        bool dup = false; for (auto &e : inst) if (e == cur) { dup = true; break; }
        if (!dup) inst.push_back(cur);
    }
    pt.ninst = (int)inst.size();
    for (int i = 0; i < pt.ninst; i++) for (int j = 0; j < pt.k; j++) pt.cell[i][j] = inst[i][j];
    g_pat.push_back(pt);
}

static inline std::vector<int> diag_cells(int sr, int sc, int len, int dr, int dc) {
    std::vector<int> v; for (int i = 0; i < len; i++) v.push_back((sr + i * dr) * 8 + (sc + i * dc)); return v;
}

static void pat_init() {
    if (!g_pat.empty()) return;
    add_pattern({0,1,2,3,4,5,6,7, 9, 14});            // edge + 2X
    add_pattern({0,1,2,3,4, 8,9,10,11,12});           // 2x5 corner block
    add_pattern({0,1,2, 8,9,10, 16,17,18});           // 3x3 corner block
    add_pattern({8,9,10,11,12,13,14,15});             // row 1
    add_pattern({16,17,18,19,20,21,22,23});           // row 2
    add_pattern({24,25,26,27,28,29,30,31});           // row 3
    add_pattern(diag_cells(0,0,8,1,1));               // main diagonals
    add_pattern(diag_cells(0,1,7,1,1));
    add_pattern(diag_cells(0,2,6,1,1));
    add_pattern(diag_cells(0,3,5,1,1));
    add_pattern(diag_cells(0,4,4,1,1));
    g_npat = (int)g_pat.size();
    g_ninst_total = 0;
    for (int t = 0; t < g_npat; t++) g_ninst_total += g_pat[t].ninst;
    // per-phase table layout
    for (int ph = 0; ph < NPHASE; ph++) {
        int total = 0;
        for (int t = 0; t < g_npat; t++) { g_pat_off[ph][t] = total; total += g_pat[t].size; }
        g_phase_block = total;
    }
}

static inline int phase_of(int discs) {
    int ply = discs - 4; if (ply < 0) ply = 0;
    int ph = ply * NPHASE / 61;
    if (ph >= NPHASE) ph = NPHASE - 1;
    return ph;
}

// ───────────────────────── dense value features ────────────────────────────
static const int NDENSE = 4;
static inline void dense_feats(u64 P, u64 O, float out[NDENSE]) {
    int mP = popc(gen_moves(P, O)), mO = popc(gen_moves(O, P));
    u64 e = ~(P | O);
    u64 adjE = AZ_SH_E(e)|AZ_SH_W(e)|AZ_SH_S(e)|AZ_SH_N(e)|AZ_SH_SE(e)|AZ_SH_SW(e)|AZ_SH_NE(e)|AZ_SH_NW(e);
    u64 adjP = AZ_SH_E(P)|AZ_SH_W(P)|AZ_SH_S(P)|AZ_SH_N(P)|AZ_SH_SE(P)|AZ_SH_SW(P)|AZ_SH_NE(P)|AZ_SH_NW(P);
    u64 adjO = AZ_SH_E(O)|AZ_SH_W(O)|AZ_SH_S(O)|AZ_SH_N(O)|AZ_SH_SE(O)|AZ_SH_SW(O)|AZ_SH_NE(O)|AZ_SH_NW(O);
    out[0] = (float)(mP - mO);                          // actual mobility
    out[1] = (float)(popc(e & adjO) - popc(e & adjP));  // potential mobility
    out[2] = (float)(popc(e) & 1);                      // parity (odd empties)
    out[3] = (float)(popc(P & adjE) - popc(O & adjE));  // frontier (mine - opp)
}

// Collect the value pattern feature ids for a position into `ids` (GLOBAL ids with
// the phase offset baked in, so the trainer can use one big embedding table).
// Returns the count. ph = phase_of(discs). Layout: global = ph*g_phase_block +
// g_pat_off[ph][type] + index.
static inline int value_feature_ids(u64 P, u64 O, int ph, int *ids) {
    int n = 0;
    int base = ph * g_phase_block;
    for (int t = 0; t < g_npat; t++) {
        const PatType &pt = g_pat[t];
        int toff = base + g_pat_off[ph][t];
        for (int i = 0; i < pt.ninst; i++) {
            const int *cc = pt.cell[i];
            int idx = 0;
            for (int j = 0; j < pt.k; j++) {
                int c = cc[j];
                int d = ((P >> c) & 1) ? 1 : (((O >> c) & 1) ? 2 : 0);
                idx += d * pt.pw[j];
            }
            ids[n++] = toff + idx;
        }
    }
    return n;
}

// ───────────────────────── policy move features ────────────────────────────
// Cheap, board-derived features for a single legal move `sq` (P to move). These
// feed the linear policy head (+ a per-(phase,square) bias). Kept small and exact
// so self-play can emit them and the trainer fits weights directly.
static const int NMOVEF = 6;
static const u64 XMASK = (1ULL<<9)|(1ULL<<14)|(1ULL<<49)|(1ULL<<54);
static const u64 CMASK = 0x8100000000000081ULL; // corners
static inline void move_feats(u64 P, u64 O, int sq, float out[NMOVEF]) {
    u64 m = 1ULL << sq;
    u64 f = flips_for(P, O, m);
    u64 nP = P | m | f, nO = O & ~f;
    int oppMob = popc(gen_moves(nO, nP));        // opponent replies after our move
    int myNext = popc(gen_moves(nP, nO));        // (rare same-side; usually opp moves)
    out[0] = (float)popc(f) * 0.1f;              // #flipped (scaled)
    out[1] = (m & CMASK) ? 1.f : 0.f;            // captures a corner
    out[2] = (m & XMASK) ? 1.f : 0.f;            // sits on an X-square (usually bad)
    out[3] = (float)oppMob * 0.1f;               // restricts opponent? (lower better)
    out[4] = (float)myNext * 0.1f;
    out[5] = (float)SQW[sq] * 0.01f;             // static positional value
}

// ───────────────────────── weights ─────────────────────────────────────────
// weights.bin format (little-endian, host order; magic 'A','Z','N','1'):
//   int32 nphase, int32 npat, int32 phase_block, int32 ndense, int32 nmovef
//   float32 vscale
//   value: nphase*phase_block float32   (pattern tables, concatenated per phase)
//   value dense: nphase*ndense float32
//   value bias:  nphase float32
//   policy bias: nphase*64 float32      (per-(phase,square) prior bias)
//   policy w:    nmovef float32         (shared move-feature weights)
//   policy temp: 1 float32              (softmax temperature, >0)
struct Net {
    bool ok = false;
    int nphase = NPHASE, npat = 0, phase_block = 0, ndense = NDENSE, nmovef = NMOVEF;
    float vscale = 1.0f;
    std::vector<float> vtab;            // nphase*phase_block
    std::vector<float> vdense;          // nphase*ndense
    std::vector<float> vbias;           // nphase
    std::vector<float> pbias;           // nphase*64
    std::vector<float> pw;              // nmovef
    float ptemp = 1.0f;
};
static Net g_net;

[[maybe_unused]] static bool net_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    char magic[4];
    if (fread(magic,1,4,f)!=4 || magic[0]!='A'||magic[1]!='Z'||magic[2]!='N'||magic[3]!='1'){ fclose(f); return false; }
    int nph,np,pb,nd,nm; float vs;
    if (fread(&nph,4,1,f)!=1||fread(&np,4,1,f)!=1||fread(&pb,4,1,f)!=1||fread(&nd,4,1,f)!=1||fread(&nm,4,1,f)!=1||fread(&vs,4,1,f)!=1){fclose(f);return false;}
    if (nph!=NPHASE || np!=g_npat || pb!=g_phase_block || nd!=NDENSE || nm!=NMOVEF){ fclose(f); return false; }
    g_net.nphase=nph; g_net.npat=np; g_net.phase_block=pb; g_net.ndense=nd; g_net.nmovef=nm; g_net.vscale=vs;
    g_net.vtab.assign((size_t)nph*pb,0.f);
    g_net.vdense.assign((size_t)nph*nd,0.f);
    g_net.vbias.assign(nph,0.f);
    g_net.pbias.assign((size_t)nph*64,0.f);
    g_net.pw.assign(nm,0.f);
    bool okr = true;
    okr &= fread(g_net.vtab.data(),4,g_net.vtab.size(),f)==g_net.vtab.size();
    okr &= fread(g_net.vdense.data(),4,g_net.vdense.size(),f)==g_net.vdense.size();
    okr &= fread(g_net.vbias.data(),4,g_net.vbias.size(),f)==g_net.vbias.size();
    okr &= fread(g_net.pbias.data(),4,g_net.pbias.size(),f)==g_net.pbias.size();
    okr &= fread(g_net.pw.data(),4,g_net.pw.size(),f)==g_net.pw.size();
    okr &= fread(&g_net.ptemp,4,1,f)==1;
    fclose(f);
    if (!okr) return false;
    if (g_net.ptemp <= 1e-3f) g_net.ptemp = 1.0f;
    g_net.ok = true;
    return true;
}

// Write `nt` to weights.bin (format above). Returns false on I/O error.
[[maybe_unused]] static bool net_save(const Net &nt, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    char magic[4] = {'A','Z','N','1'};
    int nph = NPHASE, np = g_npat, pb = g_phase_block, nd = NDENSE, nm = NMOVEF;
    float vs = nt.vscale, pt = nt.ptemp;
    bool ok = true;
    ok &= fwrite(magic,1,4,f)==4;
    ok &= fwrite(&nph,4,1,f)==1; ok &= fwrite(&np,4,1,f)==1; ok &= fwrite(&pb,4,1,f)==1;
    ok &= fwrite(&nd,4,1,f)==1;  ok &= fwrite(&nm,4,1,f)==1; ok &= fwrite(&vs,4,1,f)==1;
    ok &= fwrite(nt.vtab.data(),4,nt.vtab.size(),f)==nt.vtab.size();
    ok &= fwrite(nt.vdense.data(),4,nt.vdense.size(),f)==nt.vdense.size();
    ok &= fwrite(nt.vbias.data(),4,nt.vbias.size(),f)==nt.vbias.size();
    ok &= fwrite(nt.pbias.data(),4,nt.pbias.size(),f)==nt.pbias.size();
    ok &= fwrite(nt.pw.data(),4,nt.pw.size(),f)==nt.pw.size();
    ok &= fwrite(&pt,4,1,f)==1;
    fclose(f);
    return ok;
}

// Load into an explicit Net (not the global). Mirrors net_load.
[[maybe_unused]] static bool net_load_into(Net &nt, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    char magic[4];
    if (fread(magic,1,4,f)!=4 || magic[0]!='A'||magic[1]!='Z'||magic[2]!='N'||magic[3]!='1'){ fclose(f); return false; }
    int nph,np,pb,nd,nm; float vs;
    if (fread(&nph,4,1,f)!=1||fread(&np,4,1,f)!=1||fread(&pb,4,1,f)!=1||fread(&nd,4,1,f)!=1||fread(&nm,4,1,f)!=1||fread(&vs,4,1,f)!=1){fclose(f);return false;}
    if (nph!=NPHASE || np!=g_npat || pb!=g_phase_block || nd!=NDENSE || nm!=NMOVEF){ fclose(f); return false; }
    nt.nphase=nph; nt.npat=np; nt.phase_block=pb; nt.ndense=nd; nt.nmovef=nm; nt.vscale=vs;
    nt.vtab.assign((size_t)nph*pb,0.f); nt.vdense.assign((size_t)nph*nd,0.f);
    nt.vbias.assign(nph,0.f); nt.pbias.assign((size_t)nph*64,0.f); nt.pw.assign(nm,0.f);
    bool okr = true;
    okr &= fread(nt.vtab.data(),4,nt.vtab.size(),f)==nt.vtab.size();
    okr &= fread(nt.vdense.data(),4,nt.vdense.size(),f)==nt.vdense.size();
    okr &= fread(nt.vbias.data(),4,nt.vbias.size(),f)==nt.vbias.size();
    okr &= fread(nt.pbias.data(),4,nt.pbias.size(),f)==nt.pbias.size();
    okr &= fread(nt.pw.data(),4,nt.pw.size(),f)==nt.pw.size();
    okr &= fread(&nt.ptemp,4,1,f)==1;
    fclose(f);
    if (!okr) return false;
    if (nt.ptemp <= 1e-3f) nt.ptemp = 1.0f;
    nt.ok = true;
    return true;
}

// ───────────────────────── forward ─────────────────────────────────────────
// All forward routines take an explicit `const Net&` so the self-play "actor" net
// and the "learner" net can be evaluated concurrently by different threads with
// no shared mutable state. Convenience wrappers using the global g_net follow.

// value in (-1,1), side-to-move perspective.
static inline float net_value(const Net &nt, u64 P, u64 O) {
    int discs = popc(P | O);
    int ph = phase_of(discs);
    const float *tab = nt.vtab.data() + (size_t)ph * nt.phase_block;
    double s = nt.vbias[ph];
    int ids[MAXFEAT];
    int n = value_feature_ids(P, O, ph, ids);
    int base = ph * g_phase_block;
    for (int i = 0; i < n; i++) s += tab[ids[i] - base];
    float df[NDENSE]; dense_feats(P, O, df);
    const float *dw = nt.vdense.data() + (size_t)ph * NDENSE;
    for (int i = 0; i < NDENSE; i++) s += dw[i] * df[i];
    double v = std::tanh(s * nt.vscale);
    return (float)v;
}

// policy logit for a single legal move.
static inline float net_policy_logit(const Net &nt, u64 P, u64 O, int ph, int sq) {
    float mf[NMOVEF]; move_feats(P, O, sq, mf);
    double l = nt.pbias[(size_t)ph * 64 + sq];
    for (int i = 0; i < NMOVEF; i++) l += nt.pw[i] * mf[i];
    return (float)l;
}

// Fill priors[] (over the legal moves in `sqs`, count m) with a softmax of the
// policy logits / temperature. priors sum to 1.
static inline void net_policy(const Net &nt, u64 P, u64 O, const int *sqs, int m, float *priors) {
    int discs = popc(P | O);
    int ph = phase_of(discs);
    float t = nt.ptemp;
    float maxl = -1e30f;
    for (int i = 0; i < m; i++) { priors[i] = net_policy_logit(nt, P, O, ph, sqs[i]) / t; if (priors[i] > maxl) maxl = priors[i]; }
    double z = 0;
    for (int i = 0; i < m; i++) { priors[i] = std::exp(priors[i] - maxl); z += priors[i]; }
    if (z <= 0) { for (int i = 0; i < m; i++) priors[i] = 1.0f / m; return; }
    for (int i = 0; i < m; i++) priors[i] = (float)(priors[i] / z);
}

// ── global-g_net convenience wrappers (used by the match bot + benchmarks) ──
static inline float net_value(u64 P, u64 O) { return net_value(g_net, P, O); }
static inline float net_policy_logit(u64 P, u64 O, int ph, int sq) { return net_policy_logit(g_net, P, O, ph, sq); }
static inline void net_policy(u64 P, u64 O, const int *sqs, int m, float *priors) { net_policy(g_net, P, O, sqs, m, priors); }

static inline void net_init() {
    pat_init();
    board_init();
    end_tt_init();
}

} // namespace az

#endif // AZ_MCTS_NET_H
