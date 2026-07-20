#!/bin/bash
set -e
cd "$(dirname "$0")"
# The scorer needs only numpy and the Python standard library.
python3 search.py
# safety: ensure a valid solution.json exists; else fall back to the best bundled seed.
if [ ! -s solution.json ]; then
  cp seed1.json solution.json
fi
