#!/bin/bash
cd "$(dirname "$0")"
mkdir -p .run

export SEED="${SEED:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export NANOCHAT_DATA_DIR="${NANOCHAT_DATA_DIR:-${NANOCHAT_DATA:-$HOME/.cache/autoresearch/data}}"

export USE_AVALANCHE_TRIGRAM="${USE_AVALANCHE_TRIGRAM:-1}"
export TRIGRAM_TABLE_SIZE="${TRIGRAM_TABLE_SIZE:-27000000}"
export BIGRAM_TABLE_SIZE="${BIGRAM_TABLE_SIZE:-16777216}"
export BIGRAM_FACTORED="${BIGRAM_FACTORED:-1}"
export TRIGRAM_LR="${TRIGRAM_LR:-0.45}"
export TRIGRAM_LR_FLOOR="${TRIGRAM_LR_FLOOR:-0.60}"
export BIGRAM_LR="${BIGRAM_LR:-0.65}"
export BIGRAM_VE_LAYERS="${BIGRAM_VE_LAYERS:-}"

export USE_TRIGRAM_FREQ_LR="${USE_TRIGRAM_FREQ_LR:-1}"
export TRIGRAM_FREQ_LR_K="${TRIGRAM_FREQ_LR_K:-16.0}"

export USE_BIGRAM_FREQ_LR="${USE_BIGRAM_FREQ_LR:-1}"
export BIGRAM_FREQ_LR_K="${BIGRAM_FREQ_LR_K:-128.0}"
export BIGRAM_FREQ_LR_FLOOR="${BIGRAM_FREQ_LR_FLOOR:-0.0}"

export LONG_LAYERS="${LONG_LAYERS:-}"

export USE_AVALANCHE_BIGRAM="${USE_AVALANCHE_BIGRAM:-0}"

export USE_INTRADOC_POS="${USE_INTRADOC_POS:-1}"
export POS_BUCKET_LR="${POS_BUCKET_LR:-0.15}"

export USE_LM_HEAD_BIAS="${USE_LM_HEAD_BIAS:-0}"
export LM_HEAD_BIAS_INIT="${LM_HEAD_BIAS_INIT:-1}"
export LM_HEAD_BIAS_LR="${LM_HEAD_BIAS_LR:-0.03}"

export USE_ATTN_SCALE="${USE_ATTN_SCALE:-1}"
export ATTN_SCALE_LR="${ATTN_SCALE_LR:-0.005}"

export USE_LAYER_READOUT="${USE_LAYER_READOUT:-1}"
export LAYER_READOUT_LR="${LAYER_READOUT_LR:-0.01}"
export USE_READOUT_NORM="${USE_READOUT_NORM:-0}"

export USE_DENSE_DWA="${USE_DENSE_DWA:-0}"
export DENSE_DWA_LR="${DENSE_DWA_LR:-0.01}"
export DENSE_DWA_K="${DENSE_DWA_K:-8}"

export USE_BIGRAM_INDUCTION="${USE_BIGRAM_INDUCTION:-1}"
export BIGRAM_INDUCTION_LR="${BIGRAM_INDUCTION_LR:-0.6}"
export INDUCTION_ORDER="${INDUCTION_ORDER:-backoff}"
export INDUCTION_DOCMASK="${INDUCTION_DOCMASK:-1}"
export USE_INDUCTION_DIST="${USE_INDUCTION_DIST:-1}"
export INDUCTION_DIST_LR="${INDUCTION_DIST_LR:-0.25}"
export NUM_INDUCTION_DIST_BUCKETS="${NUM_INDUCTION_DIST_BUCKETS:-32}"
export USE_UNIGRAM_INDUCTION="${USE_UNIGRAM_INDUCTION:-1}"
export UNIGRAM_INDUCTION_LR="${UNIGRAM_INDUCTION_LR:-0.2}"
export UNIGRAM_INDUCTION_DOCMASK="${UNIGRAM_INDUCTION_DOCMASK:-1}"
export COMPILE_LOOKUPS="${COMPILE_LOOKUPS:-1}"

export USE_BIGRAM_RECALL="${USE_BIGRAM_RECALL:-0}"
export BIGRAM_RECALL_LR="${BIGRAM_RECALL_LR:-0.4}"

pick_python () {
  local candidates=(
    "$NANOCHAT_VENV/bin/python"
    python3
    python
  )
  for PY in "${candidates[@]}"; do
    [ -n "$PY" ] || continue
    command -v "$PY" >/dev/null 2>&1 || [ -x "$PY" ] || continue
    if "$PY" -c "import torch; from flash_attn.cute import flash_attn_func; from flash_attn.cute.interface import _flash_attn_bwd; assert torch.cuda.is_available()" >/dev/null 2>&1; then
      echo "$PY"; return 0
    fi
  done
  for PY in python3 python; do
    if "$PY" -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
      echo "$PY"; return 0
    fi
  done
  echo "python3"
}

PY="$(pick_python)"
echo "Selected python: $PY"
echo "USE_INTRADOC_POS=$USE_INTRADOC_POS  USE_TRIGRAM_FREQ_LR=$USE_TRIGRAM_FREQ_LR  TRIGRAM_FREQ_LR_K=$TRIGRAM_FREQ_LR_K  TRIGRAM_LR=$TRIGRAM_LR  TRIGRAM_LR_FLOOR=$TRIGRAM_LR_FLOOR"
echo "USE_BIGRAM_FREQ_LR=$USE_BIGRAM_FREQ_LR  BIGRAM_FREQ_LR_K=$BIGRAM_FREQ_LR_K  BIGRAM_FREQ_LR_FLOOR=$BIGRAM_FREQ_LR_FLOOR  BIGRAM_LR=$BIGRAM_LR"
echo "USE_ATTN_SCALE=$USE_ATTN_SCALE ATTN_SCALE_LR=$ATTN_SCALE_LR  USE_LAYER_READOUT=$USE_LAYER_READOUT LAYER_READOUT_LR=$LAYER_READOUT_LR USE_READOUT_NORM=$USE_READOUT_NORM  (USE_LM_HEAD_BIAS=$USE_LM_HEAD_BIAS off)"
echo "USE_DENSE_DWA=$USE_DENSE_DWA DENSE_DWA_LR=$DENSE_DWA_LR DENSE_DWA_K=$DENSE_DWA_K"
echo "USE_BIGRAM_INDUCTION=$USE_BIGRAM_INDUCTION BIGRAM_INDUCTION_LR=$BIGRAM_INDUCTION_LR INDUCTION_ORDER=$INDUCTION_ORDER INDUCTION_DOCMASK=$INDUCTION_DOCMASK  USE_INDUCTION_DIST=$USE_INDUCTION_DIST INDUCTION_DIST_LR=$INDUCTION_DIST_LR COMPILE_LOOKUPS=$COMPILE_LOOKUPS"
echo "USE_UNIGRAM_INDUCTION=$USE_UNIGRAM_INDUCTION UNIGRAM_INDUCTION_LR=$UNIGRAM_INDUCTION_LR  (USE_BIGRAM_RECALL=$USE_BIGRAM_RECALL off)"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cap', torch.cuda.get_device_capability())" 2>/dev/null || echo "(torch/cuda probe failed)"

set -e
"$PY" train.py 2>&1 | tee .run/train.log

"$PY" - <<'PY'
import re, json, sys
log = open(".run/train.log", encoding="utf-8", errors="replace").read()
def grab(key):
    m = re.search(rf"^{key}:\s+([0-9.]+)\s*$", log, re.MULTILINE)
    return m.group(1) if m else None
vb = grab("val_bpb"); ts = grab("training_seconds"); ns = grab("num_steps")
if vb is None:
    json.dump({"error": "no val_bpb in training log - check .run/train.log"}, open("solution.json", "w"))
    print("ERROR: could not parse val_bpb from training log", file=sys.stderr); sys.exit(1)
result = {"val_bpb": float(vb),
          "training_seconds": float(ts) if ts else None,
          "num_steps": int(float(ns)) if ns else None}
json.dump(result, open("solution.json", "w"))
print("solution.json:", json.dumps(result))
PY
