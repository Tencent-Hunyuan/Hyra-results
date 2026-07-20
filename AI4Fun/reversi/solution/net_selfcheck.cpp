// net_selfcheck.cpp - report architecture sizes and BENCHMARK the per-leaf eval
// cost (value + policy over legal moves), which is the MCTS per-simulation cost
// that decides how many sims/move we get on a bare g++ -O2 match host.
#include "net.h"
#include <cstdio>
#include <chrono>
#include <random>
using namespace az;

int main() {
    net_init();
    // sizes
    long long vtab = (long long)NPHASE * g_phase_block;
    printf("npat=%d phase_block=%d  value_table_params=%lld (%.2f MB f32)\n",
           g_npat, g_phase_block, vtab, vtab * 4.0 / 1e6);
    long long total = vtab + (long long)NPHASE*NDENSE + NPHASE + (long long)NPHASE*64 + NMOVEF + 1;
    printf("total_params=%lld (%.2f MB f32)\n", total, total*4.0/1e6);
    int ninst_sum = 0;
    for (int t=0;t<g_npat;t++){ printf("  pat%d k=%d size=%d ninst=%d\n", t, g_pat[t].k, g_pat[t].size, g_pat[t].ninst); ninst_sum += g_pat[t].ninst; }
    printf("value features per position = %d (instance sums)\n", ninst_sum);

    // give weights small random values so forward does real work
    g_net.ok=true;
    g_net.vtab.assign(vtab, 0.001f);
    g_net.vdense.assign((size_t)NPHASE*NDENSE, 0.01f);
    g_net.vbias.assign(NPHASE, 0.0f);
    g_net.pbias.assign((size_t)NPHASE*64, 0.0f);
    g_net.pw.assign(NMOVEF, 0.1f);
    g_net.ptemp = 1.0f;

    // build a set of midgame positions by random self-play
    std::mt19937_64 rng(12345);
    std::vector<std::pair<u64,u64>> pos;
    for (int g=0; g<2000; g++) {
        u64 P,O; start_black(P,O);
        int steps = 8 + (rng()%30);
        bool passed=false;
        for (int s=0;s<steps;s++){
            u64 mv=gen_moves(P,O);
            if(!mv){ if(passed)break; passed=true; std::swap(P,O); continue; }
            passed=false;
            int cnt=popc(mv), pick=rng()%cnt, sq=-1; u64 t=mv;
            for(int i=0;i<=pick;i++){sq=lsb(t);t&=t-1;}
            play_swap(P,O,1ULL<<sq);
        }
        if (gen_moves(P,O)) pos.push_back({P,O});
    }
    printf("benchmark positions: %zu\n", pos.size());

    // benchmark: value only
    {
        auto t0=std::chrono::steady_clock::now();
        volatile double acc=0; int reps=200;
        for(int r=0;r<reps;r++) for(auto&pr:pos) acc+=net_value(pr.first,pr.second);
        auto t1=std::chrono::steady_clock::now();
        double ns = std::chrono::duration<double,std::nano>(t1-t0).count()/(reps*(double)pos.size());
        printf("value-only:  %.0f ns/call  -> %.0f k/s\n", ns, 1e6/ns);
    }
    // benchmark: full leaf eval (value + policy over legal moves) = MCTS per-sim cost
    {
        auto t0=std::chrono::steady_clock::now();
        volatile double acc=0; int reps=200;
        for(int r=0;r<reps;r++) for(auto&pr:pos){
            u64 P=pr.first,O=pr.second;
            acc+=net_value(P,O);
            int sqs[40], m=0; u64 mv=gen_moves(P,O), t=mv;
            while(t){sqs[m++]=lsb(t);t&=t-1;}
            float pri[40]; net_policy(P,O,sqs,m,pri);
            for(int i=0;i<m;i++) acc+=pri[i];
        }
        auto t1=std::chrono::steady_clock::now();
        double ns = std::chrono::duration<double,std::nano>(t1-t0).count()/(reps*(double)pos.size());
        printf("leaf(v+pol): %.0f ns/call  -> %.0f k sims/s  -> %.0f sims in 0.5s, %.0f in 1.0s\n",
               ns, 1e6/ns, 0.5*1e9/ns, 1.0*1e9/ns);
    }
    return 0;
}
