// rules_check.cpp - emit random self-play games as referee-style move histories,
// and self-check the endgame solver timing. Two modes:
//   ./rules_check games N SEED      -> prints N games, one per line: "B x y B x y ..."
//                                      (color letter then move, -1 -1 for a pass),
//                                      so a Python harness can replay them through
//                                      referee.py and confirm 0 rule mismatches.
//   ./rules_check endbench EMPTIES T -> from random positions with EMPTIES empties,
//                                      time T exact endgame solves; prints
//                                      mean/median/max milliseconds.
#include "board.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>
using namespace az;

struct RNG {
    u64 s;
    RNG(u64 seed) : s(seed ? seed : 0x9e3779b9ULL) {}
    u64 next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    int below(int n) { return (int)(next() % (u64)n); }
};

// Play one fully-random game from the start, recording moves with colours.
// Returns list of (colour, sq) with sq=-1 for a pass. colour: 1=Black,2=White.
static void random_game(RNG &rng, std::vector<std::pair<int,int>> &moves) {
    u64 P, O; start_black(P, O);   // P = black to move
    int mover = 1;                 // 1=black,2=white
    bool passed = false;
    moves.clear();
    while (true) {
        u64 mv = gen_moves(P, O);
        if (!mv) {
            moves.push_back({mover, -1});
            if (passed) break;
            passed = true;
            std::swap(P, O);
            mover = 3 - mover;
            continue;
        }
        passed = false;
        // pick a random legal move
        int cnt = popc(mv);
        int pick = rng.below(cnt);
        int sq = -1;
        u64 t = mv;
        for (int i = 0; i <= pick; i++) { sq = lsb(t); t &= t - 1; }
        moves.push_back({mover, sq});
        play_swap(P, O, 1ULL << sq);
        mover = 3 - mover;
    }
}

int main(int argc, char **argv) {
    board_init();
    end_tt_init();
    if (argc < 2) { fprintf(stderr, "usage: games N SEED | endbench EMPTIES T\n"); return 1; }

    if (!strcmp(argv[1], "games")) {
        int Ngames = argc > 2 ? atoi(argv[2]) : 100;
        u64 seed = argc > 3 ? strtoull(argv[3], 0, 10) : 12345;
        RNG rng(seed);
        std::vector<std::pair<int,int>> mv;
        for (int g = 0; g < Ngames; g++) {
            random_game(rng, mv);
            for (size_t i = 0; i < mv.size(); i++) {
                int c = mv[i].first, sq = mv[i].second;
                int x = sq < 0 ? -1 : sq / 8, y = sq < 0 ? -1 : sq % 8;
                printf("%c %d %d%s", c == 1 ? 'B' : 'W', x, y,
                       i + 1 < mv.size() ? " " : "\n");
            }
        }
        return 0;
    }

    if (!strcmp(argv[1], "endbench")) {
        int targetEmpties = argc > 2 ? atoi(argv[2]) : 20;
        double seconds = argc > 3 ? atof(argv[3]) : 5.0;
        RNG rng(99991);
        std::vector<double> ms;
        auto t0 = std::chrono::steady_clock::now();
        int solved = 0;
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) {
            // random-play down to targetEmpties empties
            u64 P, O; start_black(P, O);
            bool passed = false;
            while (popc(~(P | O)) > targetEmpties) {
                u64 mv = gen_moves(P, O);
                if (!mv) { if (passed) break; passed = true; std::swap(P, O); continue; }
                passed = false;
                int cnt = popc(mv), pick = rng.below(cnt), sq = -1; u64 t = mv;
                for (int i = 0; i <= pick; i++) { sq = lsb(t); t &= t - 1; }
                play_swap(P, O, 1ULL << sq);
            }
            if (popc(~(P | O)) != targetEmpties) continue;
            end_tt_clear();
            auto s0 = std::chrono::steady_clock::now();
            volatile int d = solve_endgame(P, O);
            (void)d;
            auto s1 = std::chrono::steady_clock::now();
            ms.push_back(std::chrono::duration<double, std::milli>(s1 - s0).count());
            solved++;
        }
        std::sort(ms.begin(), ms.end());
        double sum = 0; for (double v : ms) sum += v;
        printf("empties=%d solved=%d mean=%.2fms median=%.2fms p90=%.2fms max=%.2fms\n",
               targetEmpties, solved, ms.empty() ? 0 : sum / ms.size(),
               ms.empty() ? 0 : ms[ms.size() / 2],
               ms.empty() ? 0 : ms[(size_t)(ms.size() * 0.9)],
               ms.empty() ? 0 : ms.back());
        return 0;
    }
    fprintf(stderr, "unknown mode\n");
    return 1;
}
