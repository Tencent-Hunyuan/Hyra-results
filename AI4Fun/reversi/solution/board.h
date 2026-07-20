// ============================================================================
// board.h - portable bitboard Reversi (Othello, 8x8) core for the "az_mcts"
// AlphaZero solution: move generation, flips, an EXACT endgame disc-difference
// solver, and a few shared helpers. No BMI2 / inline asm - only
// __builtin_popcountll / ctzll and plain shifts - so it builds and runs under a
// bare `g++ -O2 -std=c++17` on any x86-64 match host (the Botzone build flags).
//
// Board convention:
//   idx = x*8 + y, x = row (0..7), y = col (0..7), bit `idx` set in a u64.
//   Everything is from the SIDE-TO-MOVE perspective: P = mover's discs,
//   O = opponent's discs. Black's start discs sit at (3,4)=28 and (4,3)=35;
//   White's at (3,3)=27 and (4,4)=36.
//
// The bitboard gen_moves / flips_for use a standard directional-fill bitboard
// technique; the rules they encode were cross-checked move for move against
// referee.py over thousands of self-play games (0 illegal / turn / final-count
// mismatches). The exact endgame solver here is the AlphaZero
// bot's "great equalizer" vs a deep alpha-beta opponent: once few squares remain
// the game is solved EXACTLY (final disc difference), so the MCTS never has to
// out-read a depth-12 alpha-beta in the phase where exactness, not heuristics,
// decides the game.
// ============================================================================
#ifndef AZ_MCTS_BOARD_H
#define AZ_MCTS_BOARD_H

#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <chrono>

namespace az {

typedef uint64_t u64;
typedef uint32_t u32;

static const int N = 8;
static const int NN = 64;

// ───────────────────────── bitboard primitives ─────────────────────────────
static const u64 NOTA = 0xfefefefefefefefeULL; // ~col0 (block W-wrap)
static const u64 NOTH = 0x7f7f7f7f7f7f7f7fULL; // ~col7 (block E-wrap)
#define AZ_SH_E(b)  (((b) << 1) & NOTA)
#define AZ_SH_W(b)  (((b) >> 1) & NOTH)
#define AZ_SH_S(b)  ((b) << 8)
#define AZ_SH_N(b)  ((b) >> 8)
#define AZ_SH_SE(b) (((b) << 9) & NOTA)
#define AZ_SH_SW(b) (((b) << 7) & NOTH)
#define AZ_SH_NE(b) (((b) >> 7) & NOTA)
#define AZ_SH_NW(b) (((b) >> 9) & NOTH)

static inline int popc(u64 b) { return __builtin_popcountll(b); }
static inline int lsb(u64 b)  { return __builtin_ctzll(b); }

// All legal moves for P (empties that flank an opponent line ending at a P disc).
static inline u64 gen_moves(u64 P, u64 O) {
    u64 e = ~(P | O), mv = 0, t;
    t = O & AZ_SH_E(P);  t |= O & AZ_SH_E(t);  t |= O & AZ_SH_E(t);  t |= O & AZ_SH_E(t);  t |= O & AZ_SH_E(t);  t |= O & AZ_SH_E(t);  mv |= e & AZ_SH_E(t);
    t = O & AZ_SH_W(P);  t |= O & AZ_SH_W(t);  t |= O & AZ_SH_W(t);  t |= O & AZ_SH_W(t);  t |= O & AZ_SH_W(t);  t |= O & AZ_SH_W(t);  mv |= e & AZ_SH_W(t);
    t = O & AZ_SH_S(P);  t |= O & AZ_SH_S(t);  t |= O & AZ_SH_S(t);  t |= O & AZ_SH_S(t);  t |= O & AZ_SH_S(t);  t |= O & AZ_SH_S(t);  mv |= e & AZ_SH_S(t);
    t = O & AZ_SH_N(P);  t |= O & AZ_SH_N(t);  t |= O & AZ_SH_N(t);  t |= O & AZ_SH_N(t);  t |= O & AZ_SH_N(t);  t |= O & AZ_SH_N(t);  mv |= e & AZ_SH_N(t);
    t = O & AZ_SH_SE(P); t |= O & AZ_SH_SE(t); t |= O & AZ_SH_SE(t); t |= O & AZ_SH_SE(t); t |= O & AZ_SH_SE(t); t |= O & AZ_SH_SE(t); mv |= e & AZ_SH_SE(t);
    t = O & AZ_SH_SW(P); t |= O & AZ_SH_SW(t); t |= O & AZ_SH_SW(t); t |= O & AZ_SH_SW(t); t |= O & AZ_SH_SW(t); t |= O & AZ_SH_SW(t); mv |= e & AZ_SH_SW(t);
    t = O & AZ_SH_NE(P); t |= O & AZ_SH_NE(t); t |= O & AZ_SH_NE(t); t |= O & AZ_SH_NE(t); t |= O & AZ_SH_NE(t); t |= O & AZ_SH_NE(t); mv |= e & AZ_SH_NE(t);
    t = O & AZ_SH_NW(P); t |= O & AZ_SH_NW(t); t |= O & AZ_SH_NW(t); t |= O & AZ_SH_NW(t); t |= O & AZ_SH_NW(t); t |= O & AZ_SH_NW(t); mv |= e & AZ_SH_NW(t);
    return mv;
}

// Discs flipped by placing single-bit m for P (m must be a legal-move bit).
static inline u64 flips_for(u64 P, u64 O, u64 m) {
    u64 f = 0, line, cur;
    line = 0; cur = AZ_SH_E(m);  while (cur & O) { line |= cur; cur = AZ_SH_E(cur);  } if (cur & P) f |= line;
    line = 0; cur = AZ_SH_W(m);  while (cur & O) { line |= cur; cur = AZ_SH_W(cur);  } if (cur & P) f |= line;
    line = 0; cur = AZ_SH_S(m);  while (cur & O) { line |= cur; cur = AZ_SH_S(cur);  } if (cur & P) f |= line;
    line = 0; cur = AZ_SH_N(m);  while (cur & O) { line |= cur; cur = AZ_SH_N(cur);  } if (cur & P) f |= line;
    line = 0; cur = AZ_SH_SE(m); while (cur & O) { line |= cur; cur = AZ_SH_SE(cur); } if (cur & P) f |= line;
    line = 0; cur = AZ_SH_SW(m); while (cur & O) { line |= cur; cur = AZ_SH_SW(cur); } if (cur & P) f |= line;
    line = 0; cur = AZ_SH_NE(m); while (cur & O) { line |= cur; cur = AZ_SH_NE(cur); } if (cur & P) f |= line;
    line = 0; cur = AZ_SH_NW(m); while (cur & O) { line |= cur; cur = AZ_SH_NW(cur); } if (cur & P) f |= line;
    return f;
}

// Apply legal move at single-bit m: P becomes opponent-to-move (returns new P,O
// already swapped to the next mover's perspective).
static inline void play_swap(u64 &P, u64 &O, u64 m) {
    u64 f = flips_for(P, O, m);
    u64 nP = P | m | f;
    u64 nO = O & ~f;
    P = nO; O = nP;           // swap perspective to the next mover
}

// Start position from BLACK's perspective (Black to move first).
static inline void start_black(u64 &P, u64 &O) {
    P = (1ULL << 28) | (1ULL << 35);   // (3,4),(4,3) black
    O = (1ULL << 27) | (1ULL << 36);   // (3,3),(4,4) white
}

// ───────────────────────── square ordering ─────────────────────────────────
// Corner-first positional order, used to order moves for alpha-beta pruning in
// the endgame solver (corners are the highest-leverage squares so trying them
// first prunes the most). A standard corner-weighted positional (SQW) table.
static const int SQW[64] = {
    120, -20, 20,  5,  5, 20, -20, 120,
    -20, -40, -5, -5, -5, -5, -40, -20,
     20,  -5, 15,  3,  3, 15,  -5,  20,
      5,  -5,  3,  3,  3,  3,  -5,   5,
      5,  -5,  3,  3,  3,  3,  -5,   5,
     20,  -5, 15,  3,  3, 15,  -5,  20,
    -20, -40, -5, -5, -5, -5, -40, -20,
    120, -20, 20,  5,  5, 20, -20, 120,
};
static int g_order[64];
static const u64 CORNERS = 0x8100000000000081ULL;

static inline void board_init() {
    std::vector<int> v(64);
    for (int i = 0; i < 64; i++) v[i] = i;
    std::stable_sort(v.begin(), v.end(), [](int a, int b) { return SQW[a] > SQW[b]; });
    for (int i = 0; i < 64; i++) g_order[i] = v[i];
}

// ───────────────────────── exact endgame solver ────────────────────────────
// Returns the EXACT final disc difference (mover - opponent) under optimal play.
// fail-soft negamax + alpha-beta, with the standard Othello endgame accelerators:
//   * move ordering by OPPONENT MOBILITY ("fewest-replies-first"): try moves that
//     most restrict the opponent first - by far the strongest endgame ordering
//     heuristic, prunes the tree dramatically (more than corner-first alone);
//   * a small transposition table (keyed on P,O), used in the deeper region;
//   * "fast-first" near the leaves: at <=6 empties, skip ordering overhead and
//     just iterate the move bits (the subtree is tiny);
//   * PVS-style null-window scout on non-first moves.
// `passed` tracks a prior pass so two passes in a row ends the game.
//
// DEADLINE SAFETY: a global deadline (g_solve_deadline) + g_solve_abort flag let
// the match bot attempt a deep solve and BAIL OUT cleanly if it would blow the
// per-turn budget - the caller then falls back to the MCTS/heuristic move, so the
// bot never forfeits on time. Self-play / analysis can disable the deadline.
//
// Scale: returns disc diff in [-64, 64].

struct EndTT {
    u64 key;
    int8_t lower, upper;   // disc-diff bounds
    int8_t depth;          // empties at which computed
    int8_t best;           // best move square (for ordering), -1 if none
};
static const int END_TTBITS = 22;
static const u64 END_TTSIZE = 1ULL << END_TTBITS;
static const u64 END_TTMASK = END_TTSIZE - 1;
static std::vector<EndTT> g_end_tt;

// deadline machinery (off by default; match bot turns it on)
static std::chrono::steady_clock::time_point g_solve_t0;
static double g_solve_limit = 0.0;     // seconds; 0 = no limit
static bool g_solve_abort = false;
static long long g_solve_nodes = 0;
static inline void solve_set_deadline(double seconds) {
    g_solve_limit = seconds;
    g_solve_t0 = std::chrono::steady_clock::now();
    g_solve_abort = false;
    g_solve_nodes = 0;
}
static inline bool solve_time_up() {
    if (g_solve_limit <= 0) return false;
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - g_solve_t0).count() > g_solve_limit;
}

static inline u64 smix(u64 x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
static inline u64 hkey(u64 P, u64 O) { return smix(P) ^ smix(O + 0x123456789ULL); }

static inline void end_tt_init() {
    if (g_end_tt.empty()) g_end_tt.assign(END_TTSIZE, EndTT{0, 0, 0, 0, -1});
}
static inline void end_tt_clear() {
    end_tt_init();
    std::fill(g_end_tt.begin(), g_end_tt.end(), EndTT{0, 0, 0, 0, -1});
}

// final disc diff (placed discs; matches referee "more stones wins").
static inline int final_diff(u64 P, u64 O) {
    return popc(P) - popc(O);
}

static int solve(u64 P, u64 O, int alpha, int beta, bool passed, int empties);

// Order moves of `moves` for P by opponent mobility after the move (ascending:
// fewest opponent replies first). Writes squares into ord[0..n) and returns n.
static inline int order_by_mobility(u64 P, u64 O, u64 moves, int *ord) {
    int n = 0;
    int mobScore[32];
    u64 t = moves;
    while (t) {
        int sq = lsb(t); t &= t - 1;
        u64 m = 1ULL << sq;
        u64 f = flips_for(P, O, m);
        u64 nP = P | m | f, nO = O & ~f;
        // opponent (nO) is to move next; fewer of its moves = better for us
        int om = popc(gen_moves(nO, nP));
        // also nudge corners to the front (corner captures are near-always good)
        int s = om * 4 - ((m & CORNERS) ? 3 : 0);
        ord[n] = sq; mobScore[n] = s; n++;
    }
    // insertion sort (n is small, <=~20)
    for (int i = 1; i < n; i++) {
        int sq = ord[i], sc = mobScore[i], j = i - 1;
        while (j >= 0 && mobScore[j] > sc) { ord[j+1]=ord[j]; mobScore[j+1]=mobScore[j]; j--; }
        ord[j+1] = sq; mobScore[j+1] = sc;
    }
    return n;
}

static int solve(u64 P, u64 O, int alpha, int beta, bool passed, int empties) {
    if (empties == 0) return final_diff(P, O);
    if ((++g_solve_nodes & 4095) == 0 && solve_time_up()) { g_solve_abort = true; return 0; }

    u64 moves = gen_moves(P, O);
    if (!moves) {
        if (passed) return final_diff(P, O);
        return -solve(O, P, -beta, -alpha, true, empties);
    }

    // near the leaves: tiny subtree, skip ordering + TT overhead.
    if (empties <= 6) {
        int best = -127;
        u64 t = moves;
        while (t) {
            int sq = lsb(t); t &= t - 1;
            u64 m = 1ULL << sq, f = flips_for(P, O, m);
            u64 nP = P | m | f, nO = O & ~f;
            int v = -solve(nO, nP, -beta, -alpha, false, empties - 1);
            if (g_solve_abort) return 0;
            if (v > best) best = v;
            if (best > alpha) alpha = best;
            if (alpha >= beta) break;
        }
        return best;
    }

    // TT probe
    EndTT *en = &g_end_tt[hkey(P, O) & END_TTMASK];
    u64 key = hkey(P, O);
    int ttBest = -1;
    if (en->key == key && en->depth == (int8_t)empties) {
        if (en->lower >= beta) return en->lower;
        if (en->upper <= alpha) return en->upper;
        if (en->lower > alpha) alpha = en->lower;
        if (en->upper < beta) beta = en->upper;
        ttBest = en->best;
    }

    int a0 = alpha, best = -127, bestSq = -1;
    int ord[32];
    int n = order_by_mobility(P, O, moves, ord);

    // bring TT best to front
    if (ttBest >= 0) {
        for (int i = 0; i < n; i++) if (ord[i] == ttBest) { std::swap(ord[0], ord[i]); break; }
    }

    for (int i = 0; i < n; i++) {
        int sq = ord[i];
        u64 m = 1ULL << sq, f = flips_for(P, O, m);
        u64 nP = P | m | f, nO = O & ~f;
        int v;
        if (i == 0) {
            v = -solve(nO, nP, -beta, -alpha, false, empties - 1);
        } else {
            v = -solve(nO, nP, -alpha - 1, -alpha, false, empties - 1);   // scout
            if (!g_solve_abort && v > alpha && v < beta)
                v = -solve(nO, nP, -beta, -alpha, false, empties - 1);    // re-search
        }
        if (g_solve_abort) return 0;
        if (v > best) { best = v; bestSq = sq; }
        if (best > alpha) alpha = best;
        if (alpha >= beta) break;
    }

    en->key = key; en->depth = (int8_t)empties; en->best = (int8_t)bestSq;
    if (best <= a0)        { en->upper = (int8_t)best; en->lower = -64; }
    else if (best >= beta) { en->lower = (int8_t)best; en->upper = 64; }
    else                   { en->lower = en->upper = (int8_t)best; }
    return best;
}

// Public entry: exact disc difference (mover - opponent) under optimal play.
static inline int solve_endgame(u64 P, u64 O) {
    int empties = popc(~(P | O));
    return solve(P, O, -64, 64, false, empties);
}

// Pick the exact-best move in the endgame (square index), or -1 if forced pass.
// Returns -1 and sets outDiff if no legal move. If the solve aborts on the
// deadline, sets *aborted (if given) and returns the best move found so far.
static inline int solve_endgame_move(u64 P, u64 O, int &outDiff, bool *aborted = nullptr) {
    int empties = popc(~(P | O));
    u64 moves = gen_moves(P, O);
    if (aborted) *aborted = false;
    if (!moves) { outDiff = -solve(O, P, -64, 64, true, empties); return -1; }
    int ord[32];
    int n = order_by_mobility(P, O, moves, ord);
    int alpha = -64, beta = 64, best = -127, bestSq = ord[0];
    for (int i = 0; i < n; i++) {
        int sq = ord[i];
        u64 m = 1ULL << sq, f = flips_for(P, O, m);
        u64 nP = P | m | f, nO = O & ~f;
        int v;
        if (i == 0) v = -solve(nO, nP, -beta, -alpha, false, empties - 1);
        else {
            v = -solve(nO, nP, -alpha - 1, -alpha, false, empties - 1);
            if (!g_solve_abort && v > alpha && v < beta)
                v = -solve(nO, nP, -beta, -alpha, false, empties - 1);
        }
        if (g_solve_abort) { if (aborted) *aborted = true; break; }
        if (v > best) { best = v; bestSq = sq; if (v > alpha) alpha = v; }
    }
    outDiff = best;
    return bestSq;
}

} // namespace az

#endif // AZ_MCTS_BOARD_H
