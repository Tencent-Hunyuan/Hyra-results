#!/bin/bash
# ============================================================================
# Pipeline:
#   1. compile the engine (board.h + net.h + mcts.h + trainer.h) into train + bot;
#   2. SELF-PLAY TRAIN for a wall-clock window (TRAIN_SECONDS) - AlphaZero loop with
#      arena-gated promotion (train.cpp); produces a fresh weights.bin;
#   3. GATE: arena the fresh-trained net vs the committed, pre-trained baseline
#      (weights_baseline.bin) ON THIS MACHINE; ship whichever is stronger, and NEVER
#      below the baseline (a short/slow machine falls back to the verified baseline
#      rather than shipping an under-converged net);
#   4. pack solution.json = {source: inlined self-contained bot.cpp, language: cpp,
#      files: {weights.bin: base64}}.
#
# The heavy lifting (a converged net) is done once offline and committed as
# weights_baseline.bin; solve.sh can RETRAIN to (try to) beat it given enough
# wall budget, but never regresses below it.
# ============================================================================
set -e
cd "$(dirname "$0")"
HERE="$(pwd)"
SEED_ROOT="$(cd .. && pwd)"

CXX=${CXX:-g++}
CXXFLAGS="-O3 -std=c++17 -w -pipe -pthread -march=native"
SUBMIT_CXXFLAGS="-O2 -std=c++17 -w -pipe"   # the exact submission build flags (portable)
JOBS=${JOBS:-$(nproc)}
# Self-play training wall budget. Derive it from the overall time budget
# (TIME_BUDGET_SEC, default 1h), reserving MARGIN for compile + the fresh-vs-baseline
# arena gate + pack + verify, so a longer (e.g. 2h) run genuinely RETRAINS rather
# than only repackaging the committed baseline. A VALIDATE=1 debug pre-flight skips
# training entirely (just compiles + smoke-tests + packs the baseline, staying inside
# its short budget).
MARGIN=${MARGIN:-600}
BUDGET=${TIME_BUDGET_SEC:-3600}
_DEF_TRAIN=$(awk "BEGIN{b=$BUDGET-$MARGIN; print (b>0)?int(b):0}")
[ "${VALIDATE:-0}" = "1" ] && _DEF_TRAIN=0
TRAIN_SECONDS=${TRAIN_SECONDS:-$_DEF_TRAIN}    # self-play training wall budget
SIMS=${SIMS:-160}                       # self-play sims/move
GATE_GAMES=${GATE_GAMES:-160}           # arena games (fresh vs baseline)
GATE_SIMS=${GATE_SIMS:-200}             # arena sims/move
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "=== [1/4] compile (portable engine; no external libs) ==="
$CXX $CXXFLAGS train.cpp -o "$WORK/train"
# build the bot with the SUBMIT flags so the smoke test reflects the submission build
python3 pack.py /dev/null "$WORK/sol_probe.json" >/dev/null 2>&1 || true
$CXX $SUBMIT_CXXFLAGS bot.cpp -o "$WORK/bot"
# host arena to gate fresh vs baseline (reuses the engine)
cat > "$WORK/arena.cpp" <<'EOF'
#include "mcts.h"
#include <cstdio>
#include <cstdlib>
using namespace az;
static double play_pair(const Net&A,const Net&B,int g,int sims,uint64_t seed){
  bool aB=(g%2==0); MCTS ma(seed*1000003ULL+g*2+1,&A),mb(seed*7919ULL+g*2+2,&B);
  u64 black=(1ULL<<28)|(1ULL<<35),white=(1ULL<<27)|(1ULL<<36); int mover=1; bool ps=false;
  MctsRng o(seed^(0x9e3779b97f4a7c15ULL*(g+1))); int op=2+(o.next()%6);
  for(int k=0;k<op;k++){u64 P=mover==1?black:white,O=mover==1?white:black;u64 mv=gen_moves(P,O);if(!mv){if(ps)break;ps=true;mover=3-mover;continue;}ps=false;int c=popc(mv),pk=o.next()%c,sq=-1;u64 t=mv;for(int i=0;i<=pk;i++){sq=lsb(t);t&=t-1;}u64 m=1ULL<<sq,f=flips_for(P,O,m);if(mover==1){black|=m|f;white&=~f;}else{white|=m|f;black&=~f;}mover=3-mover;}
  ps=false; MCTSParams pm;pm.sims=sims;pm.add_noise=false;
  while(true){u64 P=mover==1?black:white,O=mover==1?white:black;u64 mv=gen_moves(P,O);if(!mv){if(ps)break;if(!gen_moves(O,P))break;ps=true;mover=3-mover;continue;}ps=false;bool at=((mover==1)==aB);MCTS&mc=at?ma:mb;auto r=mc.search(P,O,pm);int sq=r.bestMove;if(sq>=0){u64 m=1ULL<<sq,f=flips_for(P,O,m);if(mover==1){black|=m|f;white&=~f;}else{white|=m|f;black&=~f;}}mover=3-mover;}
  int nb=popc(black),nw=popc(white);int ac=aB?nb:nw,oc=aB?nw:nb;return ac>oc?1.0:(ac==oc?0.5:0.0);}
int main(int argc,char**argv){net_init();Net A,B;if(!net_load_into(A,argv[1]))return 1;if(!net_load_into(B,argv[2]))return 1;int g=atoi(argv[3]),s=atoi(argv[4]);uint64_t sd=argc>5?strtoull(argv[5],0,10):42;double p=0;for(int i=0;i<g;i++)p+=play_pair(A,B,i,s,sd);printf("%.4f\n",p/g);return 0;}
EOF
$CXX $CXXFLAGS -I"$HERE" "$WORK/arena.cpp" -o "$WORK/arena"
echo "compiled OK"

echo "=== smoke: legal opening move ==="
( cd "$HERE" && printf "1\n-1 -1\n" | REVERSI_TLIMIT=0.3 "$WORK/bot" | head -1 )
echo "smoke OK"

echo "=== [2/4] self-play train (<= ${TRAIN_SECONDS}s, ${JOBS} threads, sims ${SIMS}) ==="
if [ "$TRAIN_SECONDS" -gt 0 ]; then
    INIT=""
    [ -f weights_baseline.bin ] && INIT="--init weights_baseline.bin"
    "$WORK/train" --out "$WORK/weights.bin" --log "$WORK/train.log" \
        --seconds "$TRAIN_SECONDS" --threads "$JOBS" --sims "$SIMS" \
        --buffer 6000000 --batch 1024 --lr 0.03 --arena-every 5000 --arena-games 80 \
        --arena-winrate 0.53 --seed 7 $INIT --warm || true
    tail -3 "$WORK/train.log" 2>/dev/null || true
else
    echo "  (TRAIN_SECONDS=0 - skipping self-play; the gate below ships the committed baseline)"
fi

echo "=== [3/4] gate: ship the stronger of {fresh, baseline} ==="
SHIP="$WORK/weights.bin"
if [ -f weights_baseline.bin ] && [ -f "$WORK/weights.bin" ]; then
    WR=$("$WORK/arena" "$WORK/weights.bin" weights_baseline.bin "$GATE_GAMES" "$GATE_SIMS" 4242 2>/dev/null | tail -1)
    echo "  fresh vs baseline arena win-rate: ${WR:-?}"
    KEEP=$(python3 -c "print('fresh' if float('${WR:-0}')>=0.52 else 'baseline')")
    if [ "$KEEP" = "baseline" ]; then SHIP="weights_baseline.bin"; fi
    echo "  shipping: $KEEP"
elif [ -f weights_baseline.bin ]; then
    SHIP="weights_baseline.bin"; echo "  no fresh weights -> baseline"
fi

echo "=== [4/4] pack solution.json ==="
python3 pack.py "$SHIP" "$SEED_ROOT/solution.json"

echo "=== verify packed solution compiles standalone (submission flags) & runs ==="
python3 - "$SEED_ROOT" "$WORK" <<'PY'
import sys, os, json, base64, subprocess
seed_root, work = sys.argv[1], sys.argv[2]
sol = json.load(open(os.path.join(seed_root, "solution.json")))
d = os.path.join(work, "verify"); os.makedirs(d, exist_ok=True)
open(os.path.join(d, "bot.cpp"), "w").write(sol["source"])
for name, b64 in (sol.get("files") or {}).items():
    open(os.path.join(d, name), "wb").write(base64.b64decode(b64))
subprocess.run(["g++","-O2","-std=c++17","-w","-pipe",
                os.path.join(d,"bot.cpp"),"-o",os.path.join(d,"bot")], check=True)
out = subprocess.run([os.path.join(d,"bot")], input="1\n-1 -1\n",
                     capture_output=True, text=True, cwd=d, timeout=8).stdout
assert len(out.split()) >= 2, f"bad first move: {out!r}"
print(f"  packed solution OK; Black opening = {out.split()[0]} {out.split()[1]}")
PY
echo "=== done: $SEED_ROOT/solution.json ==="
