// ============================================================================
// train.cpp - AlphaZero self-play trainer for the az_mcts Reversi solution.
//
// One self-contained process per worker machine. Threads:
//   * ACTOR threads (N-1): endlessly self-play games with PUCT-MCTS guided by a
//     shared, lock-free-read "actor" net snapshot; push each visited position as a
//     training sample (board, MCTS visit policy, eventual game result z) into a
//     shared ring replay buffer. Dirichlet root noise + temperature(τ=1 for the
//     first OPENING_T plies, then greedy) give exploration / diverse games.
//   * LEARNER thread (1): repeatedly draws random minibatches from the buffer and
//     applies AdaGrad SGD (trainer.h) to the "learner" net. Every PUBLISH_GAMES
//     games it atomically publishes a fresh actor-net snapshot (so self-play tracks
//     the improving net, AlphaZero-style). Every ARENA_GAMES it ARENA-GATES: the
//     learner net plays the current champion; promote (save weights.bin + become
//     the actor) only on >= ARENA_WINRATE, so a regressed net never ships.
//
// Lock-free reads: actor threads read an immutable Net snapshot via an atomic
// shared_ptr; the learner swaps in a new snapshot when publishing. The replay
// buffer uses a coarse mutex (push/sample are cheap vs. an MCTS game, so it's not
// a bottleneck). Determinism is per-thread RNG-seeded; wall-clock only bounds the
// run length (this is training, not timed match play).
//
// CLI: ./train --out weights.bin --log train.log --seconds S --threads T
//              --sims K --buffer B --lr X --champion path [--init path] [--warm]
// Emits to --log: periodic "games=.. samples=.. vloss=.. ploss=.. arena=.. champ=.."
// so a driver script can tail progress and harvest weights.bin.
// ============================================================================
#include "mcts.h"
#include "trainer.h"
#include <atomic>
#include <thread>
#include <mutex>
#include <memory>
#include <vector>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <random>
#include <string>
#include <cmath>
using namespace az;

// ───────────────────────── replay sample ───────────────────────────────────
struct Sample {
    u64 P, O;
    float z;                 // game result, side-to-move perspective
    uint8_t nm;
    int8_t mv[MAXMV];
    float pi[MAXMV];
};

struct Replay {
    std::vector<Sample> buf;
    size_t cap, head = 0;
    bool full = false;
    std::mutex mu;
    std::atomic<long long> total{0};
    void init(size_t c) { cap = c; buf.resize(c); }
    void push(const Sample &s) {
        std::lock_guard<std::mutex> lk(mu);
        buf[head] = s; head = (head + 1) % cap;
        if (head == 0) full = true;
        total.fetch_add(1, std::memory_order_relaxed);
    }
    size_t size() const { return full ? cap : head; }
    // copy a random minibatch into out (returns count)
    int sample(Sample *out, int k, MctsRng &rng) {
        std::lock_guard<std::mutex> lk(mu);
        size_t n = full ? cap : head;
        if (n == 0) return 0;
        for (int i = 0; i < k; i++) out[i] = buf[rng.next() % n];
        return k;
    }
};

// ───────────────────────── globals ─────────────────────────────────────────
static Replay g_replay;
static std::shared_ptr<const Net> g_actor;       // current self-play net (atomic via shared_ptr ops)
static std::mutex g_actor_mu;
static std::atomic<bool> g_run{true};
static std::atomic<long long> g_games{0};
static std::atomic<long long> g_moves{0};

static std::shared_ptr<const Net> actor_get() {
    std::lock_guard<std::mutex> lk(g_actor_mu);
    return g_actor;
}
static void actor_set(std::shared_ptr<const Net> n) {
    std::lock_guard<std::mutex> lk(g_actor_mu);
    g_actor = std::move(n);
}

// ───────────────────────── self-play actor ─────────────────────────────────
struct Cfg {
    int sims = 200;
    int threads = 16;
    long long seconds = 600;
    size_t buffer = 4000000;
    float cpuct = 1.5f;
    float dir_alpha = 0.30f, dir_eps = 0.25f;
    int opening_t = 12;          // plies sampled at τ=1 (exploration), then greedy
    int batch = 1024;
    float lr = 0.02f;
    int publish_games = 300;     // refresh actor snapshot every N games (per learner view)
    int arena_every = 2000;      // arena-gate every N games
    int arena_games = 200;
    float arena_winrate = 0.53f;
    uint64_t seed = 1;
    char out[512] = "weights.bin";
    char logp[512] = "train.log";
    char champ[512] = "";        // champion to gate against / broadcast (driver-managed)
    char init[512] = "";         // initial weights to start from
    char champ_in[512] = "";     // external (global) champion to inject if it beats local
    bool warm = true;
};
static Cfg C;

// Play one self-play game with the given actor net; push samples to replay.
static void play_one(const Net &net, uint64_t gseed) {
    MCTS mc(gseed, &net);
    MCTSParams pm; pm.sims = C.sims; pm.cpuct = C.cpuct;
    pm.add_noise = true; pm.dir_alpha = C.dir_alpha; pm.dir_eps = C.dir_eps;

    // per-game record of (P,O, visit policy, side-to-move sign vs black)
    struct Rec { u64 P, O; uint8_t nm; int8_t mv[MAXMV]; float pi[MAXMV]; int stm; };
    std::vector<Rec> recs;
    recs.reserve(64);

    u64 black = (1ULL<<28)|(1ULL<<35), white = (1ULL<<27)|(1ULL<<36);
    int mover = 1; bool passed = false; int ply = 0;
    while (true) {
        u64 P = mover==1 ? black : white;
        u64 O = mover==1 ? white : black;
        u64 mv = gen_moves(P, O);
        if (!mv) {
            if (passed) break;
            if (!gen_moves(O, P)) break;   // both stuck (shouldn't happen if !passed)
            passed = true; mover = 3 - mover; continue;
        }
        passed = false;
        MCTSResult r = mc.search(P, O, pm);
        // record visit policy (normalized)
        Rec rec; rec.P = P; rec.O = O; rec.nm = (uint8_t)r.nm; rec.stm = mover;
        int tot = 0; for (int i = 0; i < r.nm; i++) tot += r.visits[i];
        for (int i = 0; i < r.nm; i++) { rec.mv[i] = (int8_t)r.mv[i]; rec.pi[i] = tot>0 ? (float)r.visits[i]/tot : 1.0f/r.nm; }
        recs.push_back(rec);
        // pick move: τ=1 for opening, greedy after
        float tau = (ply < C.opening_t) ? 1.0f : 0.0f;
        int ci = mc.sample_move(r, tau);
        int sq = r.mv[ci];
        if (sq >= 0) {
            u64 m = 1ULL<<sq, f = flips_for(P, O, m);
            if (mover==1){ black|=m|f; white&=~f; } else { white|=m|f; black&=~f; }
        }
        mover = 3 - mover; ply++;
        if (ply > 70) break;   // safety
    }
    // game result
    int nb = popc(black), nw = popc(white);
    int winner = nb>nw ? 1 : (nw>nb ? 2 : 0);   // 0 = draw
    for (auto &rec : recs) {
        float z = winner==0 ? 0.f : (winner==rec.stm ? 1.f : -1.f);
        Sample s; s.P = rec.P; s.O = rec.O; s.z = z; s.nm = rec.nm;
        memcpy(s.mv, rec.mv, sizeof(int8_t)*rec.nm);
        memcpy(s.pi, rec.pi, sizeof(float)*rec.nm);
        g_replay.push(s);
    }
    g_games.fetch_add(1, std::memory_order_relaxed);
    g_moves.fetch_add((long long)recs.size(), std::memory_order_relaxed);
}

static void actor_thread(uint64_t tseed) {
    MctsRng rng(tseed);
    while (g_run.load(std::memory_order_relaxed)) {
        auto net = actor_get();
        if (!net) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); continue; }
        play_one(*net, rng.next());
    }
}

// ───────────────────────── arena (learner vs champion) ─────────────────────
// Plays `games` between net A and net B from varied random openings, returns A's
// win-rate (draws=0.5). Each side uses greedy MCTS with `sims` sims.
static double arena(const Net &A, const Net &B, int games, int sims, uint64_t seed) {
    double pts = 0; int played = 0;
    MCTSParams pm; pm.sims = sims; pm.add_noise = false;
    for (int g = 0; g < games; g++) {
        bool aBlack = (g % 2 == 0);
        MCTS ma(seed*1000003ULL + g*2 + 1, &A), mb(seed*7919ULL + g*2 + 2, &B);
        u64 black=(1ULL<<28)|(1ULL<<35), white=(1ULL<<27)|(1ULL<<36);
        int mover=1; bool passed=false;
        // random opening plies for spread
        MctsRng orng(seed ^ (0x9e3779b97f4a7c15ULL*(g+1)));
        int oplies = 2 + (orng.next()%6);
        for (int op=0; op<oplies; op++) {
            u64 P=mover==1?black:white, O=mover==1?white:black;
            u64 mv=gen_moves(P,O); if(!mv){ if(passed)break; passed=true; mover=3-mover; continue; } passed=false;
            int c=popc(mv),pk=orng.next()%c,sq=-1; u64 t=mv; for(int i=0;i<=pk;i++){sq=lsb(t);t&=t-1;}
            u64 m=1ULL<<sq,f=flips_for(P,O,m); if(mover==1){black|=m|f;white&=~f;}else{white|=m|f;black&=~f;} mover=3-mover;
        }
        passed=false;
        while (true) {
            u64 P=mover==1?black:white, O=mover==1?white:black;
            u64 mv=gen_moves(P,O);
            if(!mv){ if(passed)break; if(!gen_moves(O,P))break; passed=true; mover=3-mover; continue; }
            passed=false;
            bool aTurn = ((mover==1)==aBlack);
            MCTS &mc = aTurn ? ma : mb;
            auto r = mc.search(P,O,pm);
            int sq=r.bestMove;
            if(sq>=0){ u64 m=1ULL<<sq,f=flips_for(P,O,m); if(mover==1){black|=m|f;white&=~f;}else{white|=m|f;black&=~f;} }
            mover=3-mover;
        }
        int nb=popc(black),nw=popc(white);
        int ac = aBlack?nb:nw, oc=aBlack?nw:nb;
        if(ac>oc)pts+=1; else if(ac==oc)pts+=0.5; played++;
    }
    return played? pts/played : 0.0;
}

// ───────────────────────── learner thread ──────────────────────────────────
static void learner_thread() {
    FILE *lg = fopen(C.logp, "a");
    auto t0 = std::chrono::steady_clock::now();

    Net learner; net_init_params(learner, C.warm);
    if (C.init[0]) { Net tmp; if (net_load_into(tmp, C.init)) learner = tmp; }
    Net champion = learner;            // best-so-far
    if (C.champ[0]) { Net tmp; if (net_load_into(tmp, C.champ)) champion = tmp; }

    Grads grads; grads.init();
    TrainCfg tc; tc.lr_v = C.lr; tc.lr_p = C.lr;

    // publish initial actor = champion (so self-play starts from the best net)
    actor_set(std::make_shared<const Net>(champion));
    // also write an initial weights.bin so the harvester always finds one
    net_save(champion, C.out);

    std::vector<Sample> mb(C.batch);
    MctsRng rng(C.seed ^ 0xabcdef);
    long long steps = 0;
    long long last_publish_games = 0, last_arena_games = 0;
    double vacc = 0, pacc = 0; long long lacc = 0;

    // wait for the buffer to warm up
    while (g_run.load() && g_replay.size() < (size_t)std::max(2000, C.batch*4))
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

    while (g_run.load()) {
        int k = g_replay.sample(mb.data(), C.batch, rng);
        if (k == 0) { std::this_thread::sleep_for(std::chrono::milliseconds(20)); continue; }
        for (int i = 0; i < k; i++) {
            int idmv[MAXMV]; for (int j=0;j<mb[i].nm;j++) idmv[j]=mb[i].mv[j];
            auto sl = train_step(learner, grads, tc, mb[i].P, mb[i].O, mb[i].z,
                                 mb[i].nm, idmv, mb[i].pi);
            vacc += sl.vl; pacc += sl.pl; lacc++;
        }
        steps++;

        long long gnow = g_games.load();
        // publish actor snapshot periodically (track the improving net)
        if (gnow - last_publish_games >= C.publish_games) {
            last_publish_games = gnow;
            actor_set(std::make_shared<const Net>(learner));
        }
        // arena-gate: promote champion only if learner beats it
        if (gnow - last_arena_games >= C.arena_every) {
            last_arena_games = gnow;

            // ── external champion injection (cross-worker global best) ──
            // A driver process periodically drops a global champion at champ_in. If
            // it is present and BEATS our local champion in arena, adopt it as both
            // the local champion AND the learner (seeding this worker from the global
            // best). This propagates the strongest net across all workers, bounding
            // population divergence - the centralized-AlphaZero approximation.
            // We mtime-gate via a sidecar marker so we only test a fresh drop once.
            if (C.champ_in[0]) {
                FILE *cf = fopen(C.champ_in, "rb");
                if (cf) {
                    fclose(cf);
                    Net ext;
                    if (net_load_into(ext, C.champ_in)) {
                        double wr_ext = arena(ext, champion, C.arena_games, C.sims, (uint64_t)(steps+555));
                        if (wr_ext >= 0.55) {            // clearly better → adopt
                            champion = ext;
                            learner = ext;               // reseed learner from global best
                            grads.init();                // reset AdaGrad accumulators
                            net_save(champion, C.out);
                            actor_set(std::make_shared<const Net>(champion));
                            if (lg) { fprintf(lg, "INJECT global champion (wr=%.3f vs local)\n", wr_ext); fflush(lg); }
                        }
                    }
                    // consume the drop so we don't re-test it: rename to .used
                    std::string used = std::string(C.champ_in) + ".used";
                    rename(C.champ_in, used.c_str());
                }
            }

            double wr = arena(learner, champion, C.arena_games, C.sims, (uint64_t)(steps+12345));
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
            bool promoted = false;
            if (wr >= C.arena_winrate) {
                champion = learner;
                net_save(champion, C.out);
                actor_set(std::make_shared<const Net>(champion));
                promoted = true;
            }
            if (lg) {
                fprintf(lg, "t=%.0fs games=%lld samples=%lld steps=%lld vloss=%.4f ploss=%.4f arena=%.3f %s\n",
                        el, gnow, g_replay.total.load(), steps,
                        lacc? vacc/lacc:0, lacc? pacc/lacc:0, wr, promoted?"PROMOTED":"");
                fflush(lg);
            }
            vacc = pacc = 0; lacc = 0;
        }
    }
    // final save of the champion (best gated net)
    net_save(champion, C.out);
    if (lg) { fprintf(lg, "FINAL games=%lld champion saved to %s\n", g_games.load(), C.out); fclose(lg); }
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        auto eq = [&](const char*f){ return !strcmp(argv[i],f); };
        if (eq("--sims")&&i+1<argc) C.sims=atoi(argv[++i]);
        else if (eq("--threads")&&i+1<argc) C.threads=atoi(argv[++i]);
        else if (eq("--seconds")&&i+1<argc) C.seconds=atoll(argv[++i]);
        else if (eq("--buffer")&&i+1<argc) C.buffer=(size_t)atoll(argv[++i]);
        else if (eq("--batch")&&i+1<argc) C.batch=atoi(argv[++i]);
        else if (eq("--lr")&&i+1<argc) C.lr=atof(argv[++i]);
        else if (eq("--cpuct")&&i+1<argc) C.cpuct=atof(argv[++i]);
        else if (eq("--dir-alpha")&&i+1<argc) C.dir_alpha=atof(argv[++i]);
        else if (eq("--dir-eps")&&i+1<argc) C.dir_eps=atof(argv[++i]);
        else if (eq("--opening-t")&&i+1<argc) C.opening_t=atoi(argv[++i]);
        else if (eq("--publish-games")&&i+1<argc) C.publish_games=atoi(argv[++i]);
        else if (eq("--arena-every")&&i+1<argc) C.arena_every=atoi(argv[++i]);
        else if (eq("--arena-games")&&i+1<argc) C.arena_games=atoi(argv[++i]);
        else if (eq("--arena-winrate")&&i+1<argc) C.arena_winrate=atof(argv[++i]);
        else if (eq("--seed")&&i+1<argc) C.seed=strtoull(argv[++i],0,10);
        else if (eq("--out")&&i+1<argc) strncpy(C.out,argv[++i],511);
        else if (eq("--log")&&i+1<argc) strncpy(C.logp,argv[++i],511);
        else if (eq("--champion")&&i+1<argc) strncpy(C.champ,argv[++i],511);
        else if (eq("--init")&&i+1<argc) strncpy(C.init,argv[++i],511);
        else if (eq("--champ-in")&&i+1<argc) strncpy(C.champ_in,argv[++i],511);
        else if (eq("--warm")) C.warm=true;
        else if (eq("--no-warm")) C.warm=false;
    }
    net_init();
    g_replay.init(C.buffer);

    int nactors = std::max(1, C.threads - 1);
    std::vector<std::thread> actors;
    std::thread learner(learner_thread);
    for (int i = 0; i < nactors; i++) actors.emplace_back(actor_thread, C.seed*131 + i*0x9e37 + 1);

    auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count() < C.seconds)
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    g_run.store(false);
    for (auto &t : actors) t.join();
    learner.join();
    return 0;
}
