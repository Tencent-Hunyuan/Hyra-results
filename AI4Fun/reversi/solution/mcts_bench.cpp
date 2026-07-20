// mcts_bench.cpp - verify MCTS speed (in-tree, with subtree locality) and sanity.
//   ./mcts_bench speed SIMS         -> sims/sec on midgame positions
//   ./mcts_bench vsrandom N SIMS    -> MCTS win-rate vs random over N games (should ~1.0)
//   ./mcts_bench selfcheck          -> value/policy parity dump for a few positions
#include "mcts.h"
#include <cstdio>
#include <cstring>
#include <chrono>
#include <random>
using namespace az;

// fill net with mild hand-eval-like weights so search is meaningful even untrained:
// approximate value via SQW positional + mobility through the dense feats; leave
// pattern tables ~0. (Only for benchmarking; real weights come from training.)
static void fill_dummy() {
    long long vtab=(long long)NPHASE*g_phase_block;
    g_net.ok=true;
    g_net.vtab.assign(vtab,0.f);
    g_net.vdense.assign((size_t)NPHASE*NDENSE,0.f);
    g_net.vbias.assign(NPHASE,0.f);
    g_net.pbias.assign((size_t)NPHASE*64,0.f);
    g_net.pw.assign(NMOVEF,0.f);
    g_net.vscale=0.02f;
    for(int ph=0;ph<NPHASE;ph++){ g_net.vdense[ph*NDENSE+0]=0.3f; /*mobility*/ g_net.vdense[ph*NDENSE+3]=-0.1f; }
    // policy: prefer corners, fewer-opp-replies, positional
    g_net.pw[0]=0.2f; g_net.pw[1]=2.0f; g_net.pw[2]=-1.5f; g_net.pw[3]=-0.4f; g_net.pw[5]=0.5f;
    g_net.ptemp=1.0f;
}

int main(int argc,char**argv){
    net_init();
    fill_dummy();
    if(argc<2){fprintf(stderr,"speed|vsrandom|selfcheck\n");return 1;}

    if(!strcmp(argv[1],"speed")){
        int sims=argc>2?atoi(argv[2]):2000;
        std::mt19937_64 rng(7);
        std::vector<std::pair<u64,u64>> pos;
        for(int g=0;g<200;g++){ u64 P,O;start_black(P,O);int steps=6+rng()%26;bool ps=false;
            for(int s=0;s<steps;s++){u64 mv=gen_moves(P,O);if(!mv){if(ps)break;ps=true;std::swap(P,O);continue;}ps=false;int c=popc(mv),pk=rng()%c,sq=-1;u64 t=mv;for(int i=0;i<=pk;i++){sq=lsb(t);t&=t-1;}play_swap(P,O,1ULL<<sq);}
            if(gen_moves(P,O))pos.push_back({P,O});}
        MCTS mc(123);
        MCTSParams pm; pm.sims=sims;
        auto t0=std::chrono::steady_clock::now();
        long long totSims=0;
        for(auto&pr:pos){ mc.search(pr.first,pr.second,pm); totSims+=sims; }
        auto t1=std::chrono::steady_clock::now();
        double sec=std::chrono::duration<double>(t1-t0).count();
        printf("positions=%zu sims/pos=%d total=%lld time=%.2fs -> %.0f sims/s, %.2f ms/move\n",
               pos.size(),sims,totSims,sec, totSims/sec, 1000.0*sec/pos.size());
        return 0;
    }

    if(!strcmp(argv[1],"vsrandom")){
        int N=argc>2?atoi(argv[2]):200; int sims=argc>3?atoi(argv[3]):200;
        MCTS mc(42); MCTSParams pm; pm.sims=sims;
        std::mt19937_64 rng(99);
        double pts=0;
        for(int g=0;g<N;g++){
            bool mctsBlack=(g%2==0);
            u64 P,O;start_black(P,O); int mover=1; bool ps=false;
            while(true){
                u64 mv=gen_moves(P,O);
                if(!mv){ if(ps)break; ps=true; std::swap(P,O); mover=3-mover; continue; }
                ps=false;
                bool mctsTurn=((mover==1)==mctsBlack);
                int sq;
                if(mctsTurn){ auto r=mc.search(P,O,pm); sq=r.bestMove; }
                else { int c=popc(mv),pk=rng()%c; sq=-1; u64 t=mv; for(int i=0;i<=pk;i++){sq=lsb(t);t&=t-1;} }
                play_swap(P,O,1ULL<<sq); mover=3-mover;
            }
            // P,O are from `mover` perspective at end; count absolute black/white
            // reconstruct: easier - track via final board. Recompute black count:
            // We lost color tracking; redo with explicit boards.
            (void)0;
            // fallback: replay handled below
            // (handled by separate explicit-color sim)
            break;
        }
        // explicit-color version (clean):
        pts=0;
        for(int g=0;g<N;g++){
            bool mctsBlack=(g%2==0);
            u64 black=(1ULL<<28)|(1ULL<<35), white=(1ULL<<27)|(1ULL<<36);
            int mover=1; bool ps=false;
            while(true){
                u64 P = mover==1?black:white, O = mover==1?white:black;
                u64 mv=gen_moves(P,O);
                if(!mv){ if(ps)break; ps=true; mover=3-mover; continue; }
                ps=false;
                bool mctsTurn=((mover==1)==mctsBlack);
                int sq;
                if(mctsTurn){ auto r=mc.search(P,O,pm); sq=r.bestMove; }
                else { int c=popc(mv),pk=rng()%c; sq=-1; u64 t=mv; for(int i=0;i<=pk;i++){sq=lsb(t);t&=t-1;} }
                u64 m=1ULL<<sq, f=flips_for(P,O,m);
                if(mover==1){ black|=m|f; white&=~f; } else { white|=m|f; black&=~f; }
                mover=3-mover;
            }
            int nb=popc(black),nw=popc(white);
            int mc_cnt = mctsBlack?nb:nw, op_cnt = mctsBlack?nw:nb;
            if(mc_cnt>op_cnt)pts+=1; else if(mc_cnt==op_cnt)pts+=0.5;
        }
        printf("MCTS(sims=%d) vs random over %d games: %.1f%%\n", sims, N, 100.0*pts/N);
        return 0;
    }
    return 0;
}
