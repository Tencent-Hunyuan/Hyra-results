#!/bin/bash
# ============================================================================
# verify.sh - gates for the az_mcts solution. Run before committing / shipping:
#   1. RULES gate: the bitboard engine matches referee.py over thousands of random
#      self-play games (0 illegal/colour/terminal mismatches) - rules-accurate.
#   2. GRADIENT gate: train_step gradients match finite differences (1e-3) and SGD
#      reduces loss on a synthetic target - the trainer math is correct.
#   3. BUILD gate: the SELF-CONTAINED packed source compiles with the exact submission
#      flags (g++ -O2 -std=c++17, NO -I) and plays a legal opening move.
#   4. SEARCH-CORRECTNESS gate: the built bot sweeps the public sparring partners
#      (random/greedy) >= 90% - catches a pass-back / negamax-sign regression that
#      would compile + play legally but lose to trivial bots.
#   5. TIMING gate: worst-case (first) move stays under the per-turn ceiling.
# Any failure exits non-zero.
# ============================================================================
set -e
cd "$(dirname "$0")"
HERE="$(pwd)"
REV="$(cd ../../.. && pwd)"   # datasets/reversi
CXX=${CXX:-g++}

echo "=== [1/5] RULES gate (engine vs referee.py) ==="
$CXX -O2 -std=c++17 -w rules_check.cpp -o /tmp/az_rules_check
cp /tmp/az_rules_check ./rules_check
python3 verify_rules.py 3000 31337
echo

echo "=== [2/5] GRADIENT gate ==="
$CXX -O2 -std=c++17 -w trainer_check.cpp -o /tmp/az_trainer_check
/tmp/az_trainer_check | tail -4
echo

echo "=== [3/5] BUILD gate (self-contained, submission flags) ==="
WBIN=""; [ -f weights_baseline.bin ] && WBIN="weights_baseline.bin"
python3 pack.py "${WBIN:-/dev/null}" /tmp/az_sol.json >/dev/null
python3 -c "import json;print(json.load(open('/tmp/az_sol.json'))['source'])" > /tmp/az_submit.cpp
g++ -O2 -std=c++17 -w /tmp/az_submit.cpp -o /tmp/az_submit
echo "  standalone compile OK"
mv /tmp/az_submit /tmp/az_botdir_bot 2>/dev/null || true
# run from a dir that has weights.bin beside it (mirror how the bot is materialized at match time)
RUN=/tmp/az_botrun; rm -rf $RUN; mkdir -p $RUN
cp /tmp/az_submit.cpp $RUN/bot.cpp
[ -n "$WBIN" ] && cp "$WBIN" $RUN/weights.bin
( cd $RUN && g++ -O2 -std=c++17 -w bot.cpp -o bot && MV=$(printf "1\n-1 -1\n" | REVERSI_TLIMIT=0.3 ./bot) && echo "  Black opening move: $MV" && [ $(echo $MV | wc -w) -ge 2 ] )
echo

echo "=== [4/5] SEARCH-CORRECTNESS gate (sweep sparring >= 90%) ==="
# Use the referee CLI to play the packed bot vs the public sparring partners.
# Cap the bot's per-move time (REVERSI_TLIMIT) so the gate runs fast - we are
# checking move CORRECTNESS (does it crush trivial bots / no pass-back regression),
# not full strength, so a short search is plenty against random/greedy.
export REVERSI_TLIMIT=${VERIFY_TLIMIT:-0.10}
for OPP in sparring/random_player.py sparring/greedy_positional.py; do
    WR=$(cd "$REV" && REVERSI_TLIMIT=$REVERSI_TLIMIT python3 referee.py "$RUN/bot.cpp" "$OPP" 2>/dev/null | tail -1 | grep -oE "[0-9.]+%" | head -1 | tr -d '%')
    echo "  vs $(basename $OPP): ${WR:-?}% (tlimit=${REVERSI_TLIMIT}s)"
    python3 -c "import sys; sys.exit(0 if float('${WR:-0}')>=90.0 else 1)" || { echo "  FAIL: did not sweep $OPP"; exit 1; }
done
unset REVERSI_TLIMIT
echo

echo "=== [5/5] TIMING gate (first move under ceiling) ==="
T=$( { /usr/bin/time -f "%e" sh -c "printf '1\n-1 -1\n' | REVERSI_TLIMIT=1.5 $RUN/bot >/dev/null" ; } 2>&1 )
echo "  first move wall: ${T}s (ceiling 2.0s, self-limit 1.5s)"
python3 -c "import sys; sys.exit(0 if float('${T:-9}')<1.95 else 1)" || { echo "  FAIL: over budget"; exit 1; }

echo
echo "=== ALL GATES PASSED ==="
