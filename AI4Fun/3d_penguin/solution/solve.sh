#!/bin/bash
set -e
cd "$(dirname "$0")"
python3 - <<'PY'
import json, pathlib
src = pathlib.Path("build_script.py").read_text(encoding="utf-8")
pathlib.Path("solution.json").write_text(
    json.dumps({"build_script": src}), encoding="utf-8")
print("wrote solution.json (%d chars of build_script)" % len(src))
PY
