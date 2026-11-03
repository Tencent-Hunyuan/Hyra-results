#!/bin/bash
# Small end-to-end examples (CPU, about a minute). Needs hyqec-verify and hyqec-check installed.
#   bash examples/run_examples.sh [OUT_DIR]        THREADS=4 by default
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CODES="$HERE/../hyqec-verify/examples"
OUT=${1:-out}; T=${THREADS:-4}
mkdir -p "$OUT"; cd "$OUT"
run() { echo; echo "\$ $*"; "$@"; }

# 1. [[5,1,3]] perfect code (text format): exact distance by branch-and-bound with a coverage
#    certificate, then the certificate is re-executed by the independent checker hyqec-check.
run hyqec-verify info "$CODES/five_qubit.txt"
run hyqec-verify distance "$CODES/five_qubit.txt" --engines bnb --bnb-cert --brute-limit 0 --threads "$T" -o five.cert.json
run hyqec-verify verify five.cert.json --bnb-checker auto --threads "$T"

# 2. [[25,1,5]] XZZX surface code (non-CSS, qldpc-challenge JSON): same, engines alone
#    (no brute force, no input witness).
run hyqec-verify distance "$CODES/xzzx_surface_5.json" --engines bnb --bnb-cert --brute-limit 0 --no-input-witness \
    --threads "$T" -o xzzx.cert.json
run hyqec-verify verify xzzx.cert.json --bnb-checker auto --threads "$T"

# 3. [[32,2,4]] toric code (CSS): X-distance only, and the full Pauli distance.
run hyqec-verify distance "$CODES/toric_4.json" --side X --engines bnb --bnb-cert --brute-limit 0 --threads "$T" -o toric-X.cert.json
run hyqec-verify verify toric-X.cert.json --bnb-checker auto --threads "$T"

# 4. SAT with a DRAT proof checked by drat-trim (needs `hyqec-verify tools install`), as a
#    cross-check of branch-and-bound: two lower-bound proofs in one certificate. The B&B coverage
#    certificate is re-executed by the separately written hyqec-check; the DRAT proof is checked by
#    drat-trim, but its CNF encoding is hyqec-verify's own.
if hyqec-verify doctor 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if "path" in d.get("kissat",{}) or "path" in d.get("cadical",{}) else 1)'; then
  run hyqec-verify distance "$CODES/xzzx_surface_5.json" --engines bnb --bnb-cert --cross-check sat --brute-limit 0 \
      --no-input-witness --threads "$T" -o xzzx.bnb+sat.json
  # --json lists every evidence item: the B&B certificate (checked by hyqec-check) and the DRAT proof
  run hyqec-verify verify xzzx.bnb+sat.json --bnb-checker auto --threads "$T" --json
else
  echo; echo "(skipping SAT example: no SAT solver; run 'hyqec-verify tools install')"
fi

# 5. Python API.
run python3 - "$CODES/xzzx_surface_5.json" <<'PY'
import sys, hyqec_verify as hv
code = hv.load(sys.argv[1]).code
res = hv.distance(code, engines=("bnb",), budget=60)
print("python API:", res.status, res.lower, res.upper)
PY
echo; echo "all examples finished; outputs in $(pwd)"
