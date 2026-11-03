"""Compare two checker-owned native node drivers on the fixed 24,000-node corpus.

PYTHONPATH=tests:. python3 scripts/v2-compare-nodes.py BASELINE_DRIVER CURRENT_DRIVER
This is differential regression evidence, not an independent correctness oracle.
"""
import hashlib
import json
import os
import random
import sys

from cert_factory import Fixture
from v2_factory import NodeDriver
import oracle

rng = random.Random(171)
digest = hashlib.sha256()
compared = 0
for n in (5, 6, 7, 8, 9):
    for instance in range(4):
        rows = []
        group = {'I' * n}
        while len(group) < 2 ** (n - 1):
            g = ''.join(rng.choices('IXZY', k=n))
            if g not in group and all(oracle.commutes(g, h) for h in rows):
                rows.append(g)
                group = oracle.closure(n, rows)
        probe = Fixture(n, rows, 0, 1)
        drivers = []
        try:
            for path in sys.argv[1:3]:
                os.environ['HYQEC_CHECK_V2_NODE_ENGINE'] = path
                drivers.append(NodeDriver(probe, 's0s1s2', 4))
            for trial in range(1200):
                lab = [rng.choice([0, 4, 5, 6, 7]) for q in range(n)]
                for q in rng.sample(range(n), rng.randrange(1, 4)):
                    lab[q] = rng.randrange(1, 4)
                results = [driver.query(lab) for driver in drivers]
                for result in results:
                    result.pop('mutations_rejected', None)
                    for key in ('s0', 's1', 's2', 'limits_crossed'):  # vector intermediates, newer drivers only
                        result.pop(key, None)
                assert results[0] == results[1], (n, rows, lab, results)
                digest.update(json.dumps([n, rows, lab, results[0]], sort_keys=True).encode() + b'\n')
                compared += 1
        finally:
            for driver in drivers:
                driver.close()
print(json.dumps({'compared': compared, 'mismatches': 0, 'equal_stream_sha256': digest.hexdigest()}, sort_keys=True))
