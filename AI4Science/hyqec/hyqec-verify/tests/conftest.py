import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# test opt-ins under their pre-rename names (STABCERT_TEST_GPU, STABCERT_TEST_BNBCHECK, ...) still work
for _k in list(os.environ):
    if _k.startswith("STABCERT_TEST_") and _k not in ("STABCERT_TEST_KILL_POINT", "STABCERT_TEST_PARTIAL_FIND_BYTES",
                                                      "STABCERT_TEST_FSYNC_FAULT"):
        os.environ.setdefault("HYQEC_" + _k[len("STABCERT_"):], os.environ[_k])
