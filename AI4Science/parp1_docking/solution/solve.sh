#!/bin/bash
# PARP1 DOCKSTRING joint-objective solver.
# Deps (rdkit, vina, meeko, gemmi) are usually already available; only
# pip-install if missing. Then run the docking-based selector.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -c "import rdkit, vina, meeko" 2>/dev/null || \
    pip install --quiet rdkit vina meeko gemmi

python solve.py
