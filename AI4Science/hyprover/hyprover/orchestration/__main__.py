"""Run the hyprover CLI: ``python -m hyprover.orchestration <verb> …`` (bin/hyprover execs this)."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
