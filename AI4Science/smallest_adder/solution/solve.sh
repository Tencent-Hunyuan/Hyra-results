#!/bin/bash
set -e
cd "$(dirname "$0")"
mkdir -p .run

# --- Write the proven 16-parameter arc-decoder FIRST (auto_acc=1.00000 on the
#     seed-2025 protocol). Committing this baseline up front guarantees a valid
#     solution.json always exists even if the 15-parameter search below crashes
#     or is interrupted.
cp base16.json solution.json
echo "[baseline] committed proven 16p solution.json"

export OMP_NUM_THREADS=${CPUS:-8}
export MKL_NUM_THREADS=${CPUS:-8}

# --- Search for a 15-parameter model (one additional genuine phi tie). It only
#     overwrites the 16-parameter baseline on a re-verified 15-unique-parameter
#     model that clears the accuracy bar with a margin, scored under a fixed
#     thread count for reproducibility.
python train.py || echo "[warn] train.py exited non-zero; keeping current solution.json"

test -f solution.json && echo "solution.json present"
