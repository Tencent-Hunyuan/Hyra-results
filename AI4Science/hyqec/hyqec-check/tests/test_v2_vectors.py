"""The conformance-vector runner must fail when its driver fails or drops answers (negative tests)."""
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "v2-vectors.py"


class VectorRunnerTests(unittest.TestCase):
    def corpus(self, directory):
        code = {"code": "T0", "n": 2, "m": 1, "rows": [[0, 1, 1, 1]]}
        node = {"id": "N0", "code": "T0", "set": "test", "rules": "s0s1", "cap": 1, "lab": "44",
                "expect": {"decision": "BRANCH", "branch_generator": 0, "np": None, "PK": None, "limits_crossed": []}}
        for name, row in (("codes.jsonl.gz", code), ("nodes.jsonl.gz", node)):
            with gzip.open(directory / name, "wt") as f:
                f.write(json.dumps(row) + "\n")

    def run_with(self, script):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.corpus(directory)
            driver = directory / "driver"
            driver.write_text("#!/bin/sh\n" + script)
            driver.chmod(0o755)
            env = dict(os.environ, NODES_ONLY="1")
            return subprocess.run([sys.executable, str(RUNNER), str(directory), str(driver)],
                                  capture_output=True, text=True, env=env, timeout=60).returncode

    def test_failing_driver_fails_the_run(self):
        self.assertNotEqual(self.run_with("exit 7\n"), 0)

    def test_missing_answers_fail_the_run(self):
        self.assertNotEqual(self.run_with("cat > /dev/null\nexit 0\n"), 0)


if __name__ == "__main__":
    unittest.main()
