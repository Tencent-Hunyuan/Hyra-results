#!/bin/bash
# Prove a lower bound on the qldpc-challenge board's [[452,115,18]] (non-CSS) with a coverage
# certificate, and check the certificate independently.
#   bash examples/board_452.sh [OUT_DIR] [CAP]     proves "no logical of weight <= CAP", i.e. d >= CAP+1
# CAP 12: ~11 s + ~12 s check on 6 CPU threads; each +1 costs ~4x more; CAP 17 (d = 18) needs GPUs or hours.
# The code file is downloaded from the board (Apache-2.0) at a pinned commit and checked by sha256.
set -euo pipefail
OUT=${1:-out}; CAP=${2:-12}; T=${THREADS:-4}
URL=https://raw.githubusercontent.com/unitaryfoundation/qldpc-challenge/2f1f65ea2bdf4b745da62f9ab01d71d41074379e/codes/452-115-18.json
SHA=326032ad2bc866512151779690024e3c6db42ba21cfd5677657d803d18f94790
mkdir -p "$OUT"; cd "$OUT"
[ -f 452-115-18.json ] || curl -sSfL -o 452-115-18.json "$URL"
echo "$SHA  452-115-18.json" | sha256sum -c -
hyqec-verify info 452-115-18.json
time hyqec-verify lower 452-115-18.json --engine bnb --bound "$CAP" --bnb-cert --threads "$T" \
    --workdir "work-452-cap$CAP" -o "452-cap$CAP.cert.json"
time hyqec-verify verify "452-cap$CAP.cert.json" --bnb-checker auto --threads "$T"
