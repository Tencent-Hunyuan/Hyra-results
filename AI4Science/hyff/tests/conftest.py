"""Test the source checkout unless HYFF_TEST_INSTALLED=1 (then the installed package is used)."""
import os
import sys
from pathlib import Path

if os.environ.get('HYFF_TEST_INSTALLED') != '1':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
# Test helpers (tests/helpers) are imported as a top-level package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
