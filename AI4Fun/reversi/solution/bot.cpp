// ============================================================================
// bot.cpp - Botzone 8x8 Reversi match bot for the az_mcts AlphaZero solution.
//
// Engine: net.h (pattern policy+value net) + mcts.h (PUCT MCTS) + board.h (exact
// endgame solver). This file is the Botzone "simple interaction" shell + the
// turn-level move policy:
//   1. read the whole turn block, reconstruct the board by replaying the move
//      history IN ORDER (Reversi flips are order-dependent), determine our colour;
//   2. if it's a forced pass (no legal move) → print "-1 -1";
//   3. ENDGAME: if empties <= ENDGAME_EMPTIES, run the EXACT disc-difference solver
//      with a wall deadline; if it finishes, play the proven-best move (this is the
//      "great equalizer" - in the last plies we play perfectly, neutralizing a deep
//      alpha-beta opponent). If the solve would blow the budget it aborts and we
//      fall through to MCTS, so we never forfeit;
//   4. MIDGAME: run PUCT-MCTS for a wall-clock budget and play the most-visited
//      move. Falls back to the net policy / any legal move if anything is missing,
//      so a missing weights.bin still produces a legal game.
//
// TIMING: the referee meters wall-clock PER TURN (cpp ceiling 1.0s, first turn
// 2.0s) from process spawn. We self-limit conservatively so the slowest move stays
// safely under the ceiling even with the ~30ms weight-load + a cold first turn.
// Override budgets with REVERSI_TLIMIT (per-move seconds) for self-test / equal-
// time matches.
//
// weights.bin (the trained net) is shipped beside the bot via solution.json
// "files" and loaded by relative path; if absent the engine falls back to a warm
// hand-eval-like prior so the bot is still legal (just much weaker).
// ============================================================================
#include "mcts.h"
#include "trainer.h"   // for net_init_params warm fallback
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <tuple>
#include <chrono>
using namespace az;

static const int BLACK = 1, WHITE = 2;

// switch to exact endgame solve at/below this many empty squares. Measured safe
// (median ~160ms, p90 ~470ms, max ~620ms single-core -O2) - comfortably under the
// 1.0s ceiling, with a deadline-abort fallback for the rare worst case.
static const int ENDGAME_EMPTIES = 16;

int main() {
    net_init();
    if (!net_load("weights.bin")) {
        // fallback: warm hand-eval-like prior so the bot still plays legally.
        net_init_params(g_net, true);
    }

    // ── ANCHOR MODE (deterministic, machine-independent) ─────────────────────
    // A sibling "anchor.cfg" makes this bot a REPRODUCIBLE yardstick: a FIXED MCTS
    // sim count with the wall-clock deadline disabled, so it plays identically on
    // any machine. Present ONLY in a benchmarking bundle (the referee runs each
    // opponent with cwd = its own dir); a shipped solution has no anchor.cfg,
    // so it falls through to the wall-clock match mode below. Keys: sims=<N>
    // (>0 enables anchor mode), endgame=<E> (exact-solve threshold in empties).
    long anchor_sims = 0;
    int anchor_endgame = ENDGAME_EMPTIES;
    if (FILE *cf = fopen("anchor.cfg", "r")) {
        char line[128]; long v;
        while (fgets(line, sizeof line, cf)) {
            if (sscanf(line, "sims=%ld", &v) == 1) anchor_sims = v;
            else if (sscanf(line, "endgame=%ld", &v) == 1) anchor_endgame = (int)v;
        }
        fclose(cf);
    }
    const bool anchor_mode = anchor_sims > 0;

    // ── read the whole turn block from stdin (ints) ──
    std::vector<long long> v;
    { long long x; while (scanf("%lld", &x) == 1) v.push_back(x); }
    if (v.empty()) { printf("-1 -1\n"); return 0; }
    int n = (int)v[0]; size_t idx = 1;
    std::vector<std::pair<int,int>> reqs, resps;
    for (int i = 0; i < n; i++) {
        if (idx + 1 >= v.size()) break;
        reqs.push_back({(int)v[idx], (int)v[idx+1]}); idx += 2;
        if (i < n - 1) {
            if (idx + 1 >= v.size()) break;
            resps.push_back({(int)v[idx], (int)v[idx+1]}); idx += 2;
        }
    }
    if (reqs.empty()) { printf("-1 -1\n"); return 0; }

    int me = (reqs[0].first == -1 && reqs[0].second == -1) ? BLACK : WHITE;
    int opp = (me == BLACK) ? WHITE : BLACK;

    u64 black = (1ULL<<28)|(1ULL<<35);   // (3,4),(4,3)
    u64 white = (1ULL<<27)|(1ULL<<36);   // (3,3),(4,4)

    // rebuild the timeline of (colour, x, y) in play order, then apply.
    std::vector<std::tuple<int,int,int>> tl;
    if (me == BLACK) {
        for (int i = 0; i < n; i++) {
            if (i < (int)resps.size()) tl.push_back(std::make_tuple(me, resps[i].first, resps[i].second));
            if (i + 1 < n)             tl.push_back(std::make_tuple(opp, reqs[i+1].first, reqs[i+1].second));
        }
    } else {
        for (int i = 0; i < n; i++) {
            tl.push_back(std::make_tuple(opp, reqs[i].first, reqs[i].second));
            if (i < (int)resps.size()) tl.push_back(std::make_tuple(me, resps[i].first, resps[i].second));
        }
    }
    for (auto &t : tl) {
        int c = std::get<0>(t), x = std::get<1>(t), y = std::get<2>(t);
        if (x < 0 || y < 0) continue;            // pass
        int sq = x * 8 + y;
        u64 m = 1ULL << sq;
        if (c == BLACK) { u64 f = flips_for(black, white, m); black |= m|f; white &= ~f; }
        else            { u64 f = flips_for(white, black, m); white |= m|f; black &= ~f; }
    }

    u64 P = (me == BLACK) ? black : white;   // side-to-move = us
    u64 O = (me == BLACK) ? white : black;

    u64 moves = gen_moves(P, O);
    if (!moves) { printf("-1 -1\n"); return 0; }   // forced pass

    bool firstMove = resps.empty();
    // self-limit (override with REVERSI_TLIMIT). First turn ceiling is 2.0s; normal
    // 1.0s. We leave headroom for the ~30ms net load + startup. In anchor mode the
    // wall clock is irrelevant (fixed sims), and we ignore the env so host speed /
    // REVERSI_TLIMIT can never perturb the deterministic yardstick.
    double tlimit = firstMove ? 1.50 : 0.80;
    if (!anchor_mode) { const char *e = getenv("REVERSI_TLIMIT"); if (e) { double x = atof(e); if (x > 0) tlimit = x; } }

    auto t_start = std::chrono::steady_clock::now();
    int empties = popc(~(P | O));
    int bestMove = -1;

    // ── ENDGAME: exact disc-difference solve ──
    // Anchor mode disables the deadline (solve_set_deadline(0.0) ⇒ never aborts) so
    // the last plies are solved EXACTLY and deterministically on any machine. Match mode
    // keeps the deadline-abort fallback so it never forfeits on time.
    int endgame_thresh = anchor_mode ? anchor_endgame : ENDGAME_EMPTIES;
    if (empties <= endgame_thresh) {
        end_tt_clear();
        // reserve most of the budget for the solve; if it aborts we still have time
        // for a shallow MCTS fallback.
        solve_set_deadline(anchor_mode ? 0.0 : tlimit * 0.85);
        int diff; bool aborted = false;
        int mv = solve_endgame_move(P, O, diff, &aborted);
        if (!aborted && mv >= 0) {
            printf("%d %d\n", mv / 8, mv % 8);
            return 0;
        }
        // else: fall through to MCTS for the rest of the budget
    }

    // ── MIDGAME: PUCT-MCTS ──
    {
        MCTS mc(0x12345 ^ (P * 2654435761ULL) ^ O, &g_net);
        MCTSParams pm;
        pm.cpuct = 1.5f; pm.add_noise = false;
        if (anchor_mode) {
            pm.sims = (int)anchor_sims;         // fixed count, no wall clock → deterministic
            pm.time_limit = 0.0;
        } else {
            pm.sims = 250000;                   // high cap; the deadline ends it
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
            pm.time_limit = (tlimit * 0.92) - elapsed;   // remaining budget for search
            if (pm.time_limit < 0.02) pm.time_limit = 0.02;
        }
        MCTSResult r = mc.search(P, O, pm);
        bestMove = r.bestMove;
    }

    if (bestMove < 0) {  // safety: any legal move
        for (int k = 0; k < 64; k++) { int sq = g_order[k]; if (moves & (1ULL << sq)) { bestMove = sq; break; } }
    }
    printf("%d %d\n", bestMove / 8, bestMove % 8);
    return 0;
}
