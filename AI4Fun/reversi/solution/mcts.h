// ============================================================================
// mcts.h - single-threaded PUCT Monte-Carlo Tree Search for the az_mcts solution.
//
// This is the policy-improvement operator that makes this an AlphaZero engine, not
// an alpha-beta: each move runs N simulations of select → expand → evaluate →
// backup, guided by the net's value (leaf eval) and policy (PUCT priors). The
// match host pins each bot to ONE core, so the search is single-threaded;
// self-play parallelism comes from running many games at once, not from threads
// inside one search. (At ~7 µs/leaf this still gives tens of thousands of sims in
// a 0.5 s turn - a strong search.)
//
// Othello specifics handled here:
//   * FORCED PASS: a node with no legal move but a non-terminal opponent has a
//     single edge "pass" (sq = -1) leading to the opponent to move. A node is
//     TERMINAL iff neither side can move; its value is sign(disc diff) in {-1,0,1}
//     (matching the AlphaZero outcome target z and the referee's "more stones
//     wins / equal is a draw").
//   * Every edge (real move OR pass) swaps the side to move, so backup negates the
//     value at each step uniformly.
//
// The search uses the net's value at leaves; the match bot switches to the EXACT
// endgame solver (board.h) once few squares remain, so MCTS never has to out-read
// a deep alpha-beta in the phase where exactness decides the game.
// ============================================================================
#ifndef AZ_MCTS_MCTS_H
#define AZ_MCTS_MCTS_H

#include "net.h"
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace az {

struct MCTSParams {
    int sims = 800;
    float cpuct = 1.5f;
    float dir_alpha = 0.30f;        // Dirichlet noise concentration (root)
    float dir_eps = 0.25f;          // noise weight (0 = none)
    bool add_noise = false;         // self-play sets true at the root
    float fpu = 0.0f;               // first-play-urgency value for unvisited edges
    double time_limit = 0.0;        // wall-clock budget (s); 0 = pure sim count.
                                    // when >0, search stops at min(sims, deadline),
                                    // checking the clock every 256 sims. `sims`
                                    // should be set high (a cap) when timing.
};

struct MCTSResult {
    int bestMove = -1;              // most-visited root move (sq, or -1 = pass)
    int nm = 0;
    int mv[MAXMV];
    int visits[MAXMV];
    float rootValue = 0.f;          // net/search value at the root (side-to-move)
};

// xorshift RNG (deterministic, fast) for Dirichlet noise + move sampling.
struct MctsRng {
    uint64_t s;
    MctsRng(uint64_t seed) : s(seed ? seed : 0x9e3779b97f4a7c15ULL) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    float uf() { return (float)((next() >> 11) * (1.0 / 9007199254740992.0)); } // [0,1)
    // gamma(alpha,1) via Marsaglia-Tsang (alpha may be <1)
    float gamma(float a) {
        if (a < 1.0f) {
            float u = uf(); if (u < 1e-9f) u = 1e-9f;
            return gamma(a + 1.0f) * std::pow(u, 1.0f / a);
        }
        float d = a - 1.0f / 3.0f, c = 1.0f / std::sqrt(9.0f * d);
        for (;;) {
            float x, v;
            do { // normal via Box-Muller
                float u1 = uf(), u2 = uf();
                if (u1 < 1e-9f) u1 = 1e-9f;
                x = std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
                v = 1.0f + c * x;
            } while (v <= 0.0f);
            v = v * v * v;
            float u = uf();
            if (u < 1.0f - 0.0331f * x * x * x * x) return d * v;
            if (std::log(u) < 0.5f * x * x + d * (1.0f - v + std::log(v))) return d * v;
        }
    }
};

struct MNode {
    u64 P, O;
    int nm = 0;
    bool expanded = false;
    bool terminal = false;
    float term = 0.f;               // terminal value (side-to-move perspective)
    int sumN = 0;
    int16_t mv[MAXMV];
    float pr[MAXMV];
    int N[MAXMV];
    float W[MAXMV];
    int ch[MAXMV];                  // child node index, -1 if not created
};

class MCTS {
public:
    std::vector<MNode> pool;
    MctsRng rng;
    const Net *net;                 // net used for value/policy (defaults to g_net)
    MCTS(uint64_t seed = 1, const Net *n = nullptr) : rng(seed), net(n ? n : &g_net) {}

    // legal moves (sq list); if none and opponent can move → a single pass edge.
    // terminal flag set if neither side can move.
    static int gen_edges(u64 P, u64 O, int16_t *out, bool &terminal) {
        u64 mv = gen_moves(P, O);
        terminal = false;
        if (!mv) {
            if (!gen_moves(O, P)) { terminal = true; return 0; }
            out[0] = -1; return 1;          // forced pass
        }
        int n = 0; u64 t = mv;
        while (t) { out[n++] = (int16_t)lsb(t); t &= t - 1; }
        return n;
    }

    int new_node(u64 P, u64 O) {
        pool.emplace_back();
        MNode &nd = pool.back();
        nd.P = P; nd.O = O;
        for (int i = 0; i < MAXMV; i++) nd.ch[i] = -1;
        bool term;
        nd.nm = gen_edges(P, O, nd.mv, term);
        nd.terminal = term;
        if (term) nd.term = (float)((popc(P) > popc(O)) ? 1 : (popc(P) < popc(O) ? -1 : 0));
        return (int)pool.size() - 1;
    }

    // expand: fill priors + return leaf value (side-to-move perspective).
    float expand(int idx) {
        MNode &nd = pool[idx];
        nd.expanded = true;
        if (nd.terminal) return nd.term;
        if (nd.nm == 1 && nd.mv[0] == -1) {                 // pass node
            nd.pr[0] = 1.0f; nd.N[0] = 0; nd.W[0] = 0.f;
            return net_value(*net, nd.P, nd.O);
        }
        int nm = nd.nm < MAXMV ? nd.nm : MAXMV;
        int sqs[MAXMV] = {};                        // value-init: keeps the policy
        float pri[MAXMV] = {};                       // read provably-defined for all i
        for (int i = 0; i < nm; i++) sqs[i] = nd.mv[i];
        net_policy(*net, nd.P, nd.O, sqs, nm, pri);
        for (int i = 0; i < nm; i++) { nd.pr[i] = pri[i]; nd.N[i] = 0; nd.W[i] = 0.f; }
        return net_value(*net, nd.P, nd.O);
    }

    void add_dirichlet(int idx, float alpha, float eps) {
        MNode &nd = pool[idx];
        if (nd.nm <= 1) return;
        float g[MAXMV], sum = 0;
        for (int i = 0; i < nd.nm; i++) { g[i] = rng.gamma(alpha); sum += g[i]; }
        if (sum <= 0) return;
        for (int i = 0; i < nd.nm; i++) nd.pr[i] = (1 - eps) * nd.pr[i] + eps * (g[i] / sum);
    }

    int select_child(const MNode &nd, float cpuct, float fpu) const {
        float sq = std::sqrt((float)std::max(1, nd.sumN));
        int best = 0; float bestScore = -1e30f;
        for (int i = 0; i < nd.nm; i++) {
            float q = nd.N[i] > 0 ? nd.W[i] / nd.N[i] : fpu;
            float u = cpuct * nd.pr[i] * sq / (1 + nd.N[i]);
            float s = q + u;
            if (s > bestScore) { bestScore = s; best = i; }
        }
        return best;
    }

    MCTSResult search(u64 P, u64 O, const MCTSParams &pm) {
        pool.clear();
        size_t reserveN = (size_t)pm.sims + 16;
        if (reserveN > 300000) reserveN = 300000;   // cap for high time-limited caps
        pool.reserve(reserveN);
        int root = new_node(P, O);
        float rv = expand(root);
        if (pm.add_noise) add_dirichlet(root, pm.dir_alpha, pm.dir_eps);

        int pathNode[80]; int pathEdge[80];
        std::chrono::steady_clock::time_point t0;
        bool timed = pm.time_limit > 0.0;
        if (timed) t0 = std::chrono::steady_clock::now();
        for (int s = 0; s < pm.sims; s++) {
            if (timed && (s & 255) == 0 && s > 0) {
                double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (el > pm.time_limit) break;
            }
            int cur = root, depth = 0;
            float leafVal;
            for (;;) {
                MNode &nd = pool[cur];
                if (nd.terminal) { leafVal = nd.term; break; }
                int e = select_child(nd, pm.cpuct, pm.fpu);
                pathNode[depth] = cur; pathEdge[depth] = e; depth++;
                if (nd.ch[e] < 0) {
                    // create + evaluate the new leaf
                    u64 nP, nO;
                    if (nd.mv[e] == -1) { nP = nd.O; nO = nd.P; }     // pass
                    else { nP = nd.P; nO = nd.O; play_swap(nP, nO, 1ULL << nd.mv[e]); }
                    int c = new_node(nP, nO);
                    pool[pathNode[depth-1]].ch[e] = c;                // (pool may have realloc'd; re-index)
                    leafVal = expand(c);
                    break;
                }
                cur = nd.ch[e];
                if (depth >= 78) { leafVal = pool[cur].terminal ? pool[cur].term : net_value(*net, pool[cur].P, pool[cur].O); break; }
            }
            // backup (negate at every step - each edge swaps side to move)
            float v = leafVal;
            for (int d = depth - 1; d >= 0; d--) {
                v = -v;
                MNode &nd = pool[pathNode[d]];
                nd.W[pathEdge[d]] += v;
                nd.N[pathEdge[d]] += 1;
                nd.sumN += 1;
            }
        }

        MCTSResult r;
        MNode &rt = pool[root];
        r.nm = rt.nm; r.rootValue = rv;
        int bestN = -1;
        for (int i = 0; i < rt.nm; i++) {
            r.mv[i] = rt.mv[i]; r.visits[i] = rt.N[i];
            if (rt.N[i] > bestN) { bestN = rt.N[i]; r.bestMove = rt.mv[i]; }
        }
        return r;
    }

    // Sample a move index from visit counts with temperature tau (tau→0 = greedy).
    int sample_move(const MCTSResult &r, float tau) {
        if (tau < 1e-3f) {
            int best = 0; for (int i = 1; i < r.nm; i++) if (r.visits[i] > r.visits[best]) best = i;
            return best;
        }
        float w[MAXMV], sum = 0;
        for (int i = 0; i < r.nm; i++) { w[i] = std::pow((float)r.visits[i] + 1e-9f, 1.0f / tau); sum += w[i]; }
        float x = rng.uf() * sum, acc = 0;
        for (int i = 0; i < r.nm; i++) { acc += w[i]; if (x <= acc) return i; }
        return r.nm - 1;
    }
};

} // namespace az

#endif // AZ_MCTS_MCTS_H
