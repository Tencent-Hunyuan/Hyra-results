"""Run one worker's outer loop: ``python -m hyprover.execution <worker_dir>``.

This is how ``hyprover start`` launches each worker (detached, own process group).
"""

from __future__ import annotations

import sys

from .loop import main

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m hyprover.execution <worker_dir>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
