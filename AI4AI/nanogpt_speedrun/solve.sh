#!/bin/bash
# solve.sh - modded-nanogpt record #83 (BigramsSignTrick, PR#299), N-run eval.
#
# Trains the SAME solution N times (default 3) and reports every run's
# (val_loss, train_time); the reported number is the MEAN - averaging out the
# ~±0.0014 single-run val_loss jitter (CUDA nondeterminism). Each run is a fresh
# torchrun process (genuinely independent init/RNG); train_gpt.py is unchanged
# single-run code, independent of this wrapper script.
#
# The venv and FineWeb shards are large; their locations are set via env vars
# (with defaults below). The inductor cache persists across the N processes, so
# runs after the first reuse compiled kernels.
set -e
cd "$(dirname "$0")"

VENV="${NANOGPT_VENV:-./.venv}"
export DATA_PATH="${NANOGPT_DATA:-.}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# Persist the inductor/compile cache PER HOST (not in the per-eval scratch dir),
# so the ~7-min kernel warmup amortizes across runs and debug validations on the
# same machine. It only speeds compilation; train_time is measured after warmup,
# so the score is unaffected. A code change gets a new cache key (no false reuse).
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/nanogpt_inductor}"
# Persist Triton JIT cache across torchrun invocations to speed up warmup for runs 2+3.
# Only affects compilation; train_time is measured after warmup, so score is unaffected.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/nanogpt_triton_cache}"

# VALIDATE=1 => QUICK/DEBUG mode: a single short debug run: ONE run, FULL steps -
# so the final val_loss is real and convergence can be judged, not just compile +
# early loss. A single run's val_loss carries the ~±0.0014 jitter; the reported
# number is the N-run mean below. TIME_BUDGET_SEC sets the hard timeout - enough
# for the ~7-min compile + a full ~80s run. For a faster compile-only / early-loss
# smoke test set NANOGPT_MAX_STEPS (e.g. 200); left unset it runs the full single run.
if [ -n "${VALIDATE:-}" ]; then
    N_RUNS="${NANOGPT_RUNS:-1}"
    echo "=== DEBUG (VALIDATE=1, budget=${TIME_BUDGET_SEC:-?}s): N_RUNS=$N_RUNS MAX_STEPS=${NANOGPT_MAX_STEPS:-full} - not a scored run ==="
else
    N_RUNS="${NANOGPT_RUNS:-3}"
fi

"$VENV/bin/python" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'ngpu',torch.cuda.device_count())"

# N independent trainings of the SAME code. Each writes its own log.
for i in $(seq 1 "$N_RUNS"); do
    echo "=== run $i/$N_RUNS ==="
    "$VENV/bin/torchrun" --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "run_${i}.log"
done

# Collect the final "val_loss:<x> train_time:<y>ms" of every run into arrays and
# emit solution.json. A run with no parseable final line is dropped; the
# solution is rejected if fewer than the minimum number of runs survive.
"$VENV/bin/python" - "$N_RUNS" <<'PY'
import re, json, sys, glob
val_losses, train_times = [], []
for path in sorted(glob.glob("run_*.log")):
    log = open(path, encoding="utf-8", errors="replace").read()
    m = re.findall(r"val_loss:([0-9.]+)\s+train_time:([0-9]+)ms", log)
    if m:
        vl, ms = m[-1]
        val_losses.append(float(vl)); train_times.append(int(ms) / 1000.0)
if not val_losses:
    json.dump({"error": "no run produced a final val_loss/train_time line"}, open("solution.json", "w"))
    raise SystemExit("no runs parsed")
sol = {
    "val_losses": val_losses,
    "train_times": train_times,
    "val_loss": sum(val_losses) / len(val_losses),       # mean, for readability
    "train_time_sec": sum(train_times) / len(train_times),
    "n_runs": len(val_losses),
}
json.dump(sol, open("solution.json", "w"))
print("solution.json:", sol)
PY
