"""V2 coverage/justification tests; independent brute-force completion oracle."""
import copy
import gzip
import itertools
import io
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest

from hyqec_check.algebra import ValidationError
from hyqec_check_v2.verify import DEFAULT_ENGINE, verify
from cert_factory import Fixture
from v2_factory import NodeDriver, V2Fixture
from test_certificate import FIVE
import oracle

RULES = ('greedy', 's0', 's0s1', 's0s1s2')


def matches(lab, pauli):
    return all(a == b if a < 4 else True if a == 4 else b in (0, a - 4)
               for a, b in zip(lab, pauli))


class V2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'proof'
        self.engine = os.environ.get('HYQEC_CHECK_V2_ENGINE', str(DEFAULT_ENGINE))

    def check(self, fixture, **kw):
        self.path.write_bytes(fixture.encode() if isinstance(fixture, Fixture) else fixture)
        return verify(self.path, threads=kw.pop('threads', 1), engine=self.engine, **kw)

    def reject(self, fixture, **kw):
        try:
            verdict = self.check(fixture, **kw)
        except (ValidationError, ValueError, EOFError, OSError):
            return
        self.assertFalse(verdict['certificate_verified'], verdict)
        self.assertEqual(verdict['status'], 'rejected', verdict)

    def test_published_nine_node_trace_and_v1_compatibility(self):
        for rules, split, full in itertools.product(RULES, (1, 2), (False, True)):
            f = V2Fixture(4, ['XXXX', 'ZZZZ'], 1, split, rules=rules, full=full)
            for data in (f.encode(), gzip.compress(f.encode(), mtime=0)):
                v = self.check(data)
                self.assertTrue(v['certificate_verified'], v)
                self.assertEqual(v['total_nodes'], 9)
                self.assertEqual(v['global_trace_hex'], '05dc5a17c64e8a73')
        f = Fixture(4, ['XXXX', 'ZZZZ'], 1, 1)
        # Old entry point/transport behavior remains active for old artifacts.
        v = self.check(gzip.compress(f.encode()[:80]) + gzip.compress(f.encode()[80:]))
        self.assertTrue(v['certificate_verified'])
        self.assertEqual(v['checker_version'], '1.3.0')
        f.header.update(rules='s0s1s2', rules_version=987)
        v = self.check(f)
        self.assertTrue(v['certificate_verified'])
        self.assertNotIn('rules', v)
        self.assertNotIn('rules_version', v)
        self.assertEqual(v['spec_revision'], '1.1')

    def test_tiny_distances_and_fabricated_logical_transcripts(self):
        cases = [(4, ['XXXX', 'ZZZZ']), (5, FIVE)]
        cases += [(n, rows) for n, rows in oracle.random_codes(repetitions=1)
                  if len(oracle.closure(n, rows)) < 2**n]
        for n, rows in cases:
            for domain, rules in itertools.product(('P', 'X', 'Z'), RULES):
                d = oracle.distance(n, rows, domain)
                cap = n if d is None else d - 1
                for split, full in ((1, False), (2, True), (cap + 1, False)):
                    f = V2Fixture(n, rows, cap, split, domain, rules=rules, full=full)
                    v = self.check(f, threads=2)
                    self.assertTrue(v['certificate_verified'], v)
                    self.assertEqual(v['total_nodes'], f.footer.fields[-2])
                    self.assertEqual(int(v['global_trace']), f.footer.fields[-1])
                if d is not None:
                    self.reject(V2Fixture(n, rows, d, 1, domain, rules=rules, allow_logical=True, full=True))

    def test_every_prune_against_exhaustive_completions_and_wrong_justifications(self):
        examples = json.loads((Path(__file__).with_name('v2_prune_examples.json')).read_text())
        seen = set()
        rng = random.Random(171)
        for rule, e in examples.items():
            probe = Fixture(e['n'], e['rows'], 0, 1)
            driver = NodeDriver(probe, 's0s1s2', e['cap'])
            try:
                out = driver.query(e['lab'])
                for key, value in e['expected'].items():
                    self.assertEqual(out[key], value)
                self.assertGreaterEqual(out['mutations_rejected'], 2)
                completions = [tuple('IXZY'.index(c) for c in p) for p in oracle.all_paulis(e['n'])
                               if sum(c != 'I' for c in p) <= e['cap'] and
                               all(oracle.commutes(p, g) for g in e['rows'])]
                for lab in [e['lab']] + [[rng.choice((0, 1, 2, 3, 4, 5, 6, 7)) for _ in range(e['n'])]
                                        for _ in range(700)]:
                    if sum(1 <= a <= 3 for a in lab) >= e['cap']:
                        continue
                    out = driver.query(lab)
                    if 'error' in out:
                        self.assertIn('logical leaf', out['error'])
                    if out.get('rule') in (2, 5, 6, 7):
                        seen.add(out['rule'])
                        self.assertFalse(any(matches(lab, p) for p in completions), (lab, out))
            finally:
                driver.close()
        self.assertEqual(seen, {2, 5, 6, 7})

    def test_hints_identifiers_claims_and_record_tampering(self):
        f = V2Fixture(5, FIVE, 2, 3, rules='s0s1s2', full=True)
        prune = next(i for i, r in enumerate(f.top) if r.tag == 0x10 and r.fields[2] in (2, 5, 6, 7))
        for key, value in (('rules', 'unknown'), ('rules_version', 2), ('rules_version', True),
                           ('rules_version', 1.0), ('search', 'irreducible-pauli-dfs/1'),
                           ('trace', 'poly64-state-rules/1'), ('format', 'stabcert-bnb-coverage/1'),
                           ('cap', 3), ('domain', 'X'), ('rules', 'greedy')):
            bad = copy.deepcopy(f); bad.header[key] = value; self.reject(bad)
        for field, value in ((6, 0), (7, 9999), (2, 4), (3, 0)):
            bad = copy.deepcopy(f); bad.top[prune].fields[field] = value; self.reject(bad)
        f = V2Fixture(5, FIVE, 2, 1, full=True)
        for mutation in ('task', 'count', 'path', 'trace', 'full-hint'):
            bad = copy.deepcopy(f)
            if mutation == 'task': bad.task_records.pop()
            if mutation == 'count': bad.task_records[0][-1].fields[-2] += 1
            if mutation == 'path': bad.task_records[0][-1].fields[1] += 1
            if mutation == 'trace': bad.footer.fields[-1] ^= 1
            if mutation == 'full-hint':
                r = next(r for task in bad.task_records for r in task if r.tag == 0x10 and r.fields[2] == 5)
                r.fields[7] = 9999
            self.reject(bad)
        for opts in (dict(mode='sample', sample=0), dict(mode='range', start=0, stop=1)):
            v = self.check(f, **opts); self.assertEqual(v['status'], 'partial'); self.assertFalse(v['certificate_verified'])
        self.reject(f, expected_cap=1)

    def test_strict_transport_and_resource_limits(self):
        f = V2Fixture(5, FIVE, 2, 1)
        raw = f.encode(); gz = gzip.compress(raw, mtime=0)
        length = int.from_bytes(raw[8:16], 'little'); proof = len(raw) - 16 - length
        for data in (raw, gz):
            self.assertTrue(self.check(data, max_header=length, max_proof_bytes=proof)['certificate_verified'])
            self.reject(data, max_header=length - 1)
            self.reject(data, max_proof_bytes=proof - 1)
        badcrc = bytearray(gz); badcrc[-8] ^= 1
        for data in (raw[:-1], raw + b'\0', gz[:-1], bytes(badcrc), gz + b'\0', gz + b'x',
                     gz + gzip.compress(b'', mtime=0), gzip.compress(raw[:100], mtime=0) + gzip.compress(raw[100:], mtime=0),
):
            self.reject(data)
        named = io.BytesIO()
        with gzip.GzipFile(fileobj=named, mode='wb', filename='transport-name', mtime=123) as stream:
            stream.write(raw)
        self.assertTrue(self.check(named.getvalue())['certificate_verified'])
        # Numeric types, duplicate header keys, and noncanonical LEB128.
        hb = json.dumps(f.header).encode(); hb = hb[:-1] + b',"rules":"s0"}'
        self.reject(raw[:8] + struct.pack('<Q', len(hb)) + hb + raw[16 + length:])
        start = 16 + length
        self.reject(raw[:start + 1] + b'\x80\x00' + raw[start + 2:])

    def test_s1_variable_giveup_is_after_qubit_filter(self):
        # Commuting all-Z equations, with a private fixed X and 31/32 forced
        # X variables. First 96 active rows omit the extra Z anchor, bypassing
        # MIS (>64 active rows). The extra action contradicts a zero-target row.
        for h, empty_filter, expected in ((31, False, 6), (32, False, 3), (32, True, 6)):
            n = h + 3 + int(empty_filter)
            def row(qs):
                return ''.join('Z' if q in qs else 'I' for q in range(n))
            base = [row({0, i}) for i in range(1, h + 1)]
            rows = (base * 4)[:96] + [row({0, h + 1}), row({h + 1})]
            if empty_filter:
                rows.append(row({0, h + 2}))
            driver = NodeDriver.from_rows(n, rows, 's0s1', h + 2)
            try:
                out = driver.query([1] + [5] * (n - 2) + [0])
                self.assertEqual(out['np'], h, out)
                self.assertEqual(out['rule'], expected, out)
            finally:
                driver.close()

    def test_prop_capacity_boundaries(self):
        # Two packed variables make incompatible parity requirements. UNIT
        # sees two flippers at each constraint; PROP determines the first
        # variable and then refutes the second. Disjoint fillers adjust limits.
        def case(h, repeats=1, extra_pairs=0, extra_repeat=False, free=0):
            sizes = [2, 2] + [127] * (h - 2)
            n = sum(sizes) + 2  # one fixed qubit and one isolated logical
            blocks = []
            first = 1
            for size in sizes:
                blocks.append(list(range(first, first + size))); first += size
            def zrow(qs):
                return ''.join('Z' if q in qs else 'I' for q in range(n))
            base = [zrow({0, *b}) for b in blocks]
            rows = base + [zrow(set(blocks[0] + blocks[1][:1])),
                           zrow({0, *blocks[0], *blocks[1][:1]})]
            rows += base[2:] * (repeats - 1)
            if extra_repeat:
                rows.append(base[2])
            pairs = itertools.islice(itertools.product(blocks[2], blocks[3]), extra_pairs)
            rows += [zrow({a, b}) for a, b in pairs]
            driver = NodeDriver.from_rows(n, rows, 's0', h + 1)
            try:
                return driver.query([1] + [4] * free + [5] * (sum(sizes) - free) + [0])
            finally:
                driver.close()
        for options, expected in ((dict(h=10), 5), (dict(h=11), 3),
                                  (dict(h=10, repeats=6), 5),
                                  (dict(h=10, repeats=6, extra_repeat=True), 3),
                                  (dict(h=10, extra_pairs=2036), 5),
                                  (dict(h=10, extra_pairs=2037), 3),
                                  (dict(h=10, free=4), 5), (dict(h=10, free=5), 3)):
            with self.subTest(options=options):
                out = case(**options)
                self.assertEqual(out.get('rule'), expected, out)
                self.assertFalse(out.get('wiped'), out)
                self.assertFalse(out.get('zeros'), out)

    def test_s1_choice_and_killer_capacity_limits(self):
        for h, copies, extra_killers, free, expected in ((1, 48, 0, 0, 6), (1, 49, 0, 0, 3),
                (1, 48, 47, 0, 6), (1, 48, 48, 0, 3), (8, 1, 0, 8, 6), (8, 1, 0, 9, 3),
                (8, 1, 0, 0, 6), (9, 1, 0, 0, 3)):
            n = 127 * h + 3
            def row(qs):
                return ''.join('Z' if q in qs else 'I' for q in range(n))
            blocks = [list(range(1 + i * 127, 1 + (i + 1) * 127)) for i in range(h)]
            base = [row({0, *b}) for b in blocks]
            extra = n - 2
            rows = base + [base[0]] + [row({*blocks[0], extra})] * copies + [row({extra})]
            if extra_killers:
                rows += [row({*blocks[0][:extra_killers], extra})]
            driver = NodeDriver.from_rows(n, rows, 's0s1', h + 2)
            try:
                out = driver.query([1] + [4] * free + [5] * (127 * h + 1 - free) + [0])
                self.assertEqual(out.get('rule'), expected, (h, copies, out))
                self.assertEqual(out.get('np'), h, out)
            finally:
                driver.close()

    def test_mis_threshold_64_vertex_boundary(self):
        # The minimum-candidate greedy picks the center of a star; the
        # minimum-degree improvement picks its two disjoint leaves.
        for count, np, pk in ((64, 2, [1, 2]), (65, 1, [0])):
            rows = ['ZZZIIIII', 'ZZIZZIII', 'ZIZIIZZI']
            rows += [rows[0]] * (count - 3)
            driver = NodeDriver.from_rows(8, rows, 's0', 3)
            try:
                out = driver.query([1] + [5] * 6 + [0])
                self.assertEqual(out.get('np'), np, out)
                self.assertEqual(out.get('PK'), pk, out)
            finally:
                driver.close()

    def test_s2_qubit_limit(self):
        e = json.loads(Path(__file__).with_name('v2_prune_examples.json').read_text())['7']
        for n, expected in ((1365, 7), (1366, 3)):
            rows = [g + 'I' * (n - e['n']) for g in e['rows']]
            driver = NodeDriver.from_rows(n, rows, 's0s1s2', e['cap'])
            try:
                out = driver.query(e['lab'] + [0] * (n - e['n']))
                self.assertEqual(out['rule'], expected, out)
            finally:
                driver.close()

    def test_draft3_published_verify_examples(self):
        self.assertTrue(all(oracle.commutes('XIXZZ', g) for g in FIVE))
        driver = NodeDriver(Fixture(5, FIVE, 0, 1), 's0s1', 4)
        try:
            engine = os.environ.get('HYQEC_CHECK_V2_NODE_ENGINE', 'build/v2-node-driver')
            run = subprocess.run([engine, str(Path(driver.temp.name) / 'model'), '--draft3-verify'],
                                 text=True, capture_output=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertEqual(json.loads(run.stdout), dict(accepted=4, rejected=10,
                             modified_empty_records=True, modified_order=True))
        finally:
            driver.close()

    def test_draft3_unit_records_continue_after_zero_and_truncate_only_lists(self):
        n = 100
        def row(qs):
            return ''.join('Z' if q in qs else 'I' for q in range(n))
        rows = [row({0, 1})] + [row({0, q}) for q in range(2, 99)] + [row({1})]
        driver = NodeDriver.from_rows(n, rows, 's0', 2, mode='--unit-records')
        try:
            out = driver.query([1] + [5] * 98 + [0])
            self.assertEqual(out, dict(zeros=list(range(1, 97)), wiped=[0], first=[[98]], needs=[], refuted=True))
        finally:
            driver.close()
        # First 95 need entries are duplicates. The 97th need entry is omitted,
        # so q1 survives even though using all needs would wipe the variable.
        rows = ['ZZZI'] * 96 + ['ZZII', 'ZIZI']
        driver = NodeDriver.from_rows(4, rows, 's0', 2, mode='--unit-records')
        try:
            out = driver.query([1, 5, 5, 0])
            self.assertEqual(out, dict(zeros=[], wiped=[], first=[[-1, 96]],
                                      needs=[[g, 0] for g in range(1, 97)], refuted=False))
        finally:
            driver.close()

    def test_v1_greedy_v2_record_bytes_identical(self):
        for split, full, domain in itertools.product((1, 2, 3), (False, True), ('P', 'X', 'Z')):
            one = Fixture(5, FIVE, 2, split, domain, full=full).encode()
            two = V2Fixture(5, FIVE, 2, split, domain, rules='greedy', full=full).encode()
            records = lambda raw: raw[16 + int.from_bytes(raw[8:16], 'little'):]
            self.assertEqual(records(one), records(two))

    def test_cli_version_and_wrong_j_test_build(self):
        p = subprocess.run([sys.executable, '-m', 'hyqec_check_v2', '--version'], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0); self.assertIn('2.0.2', p.stdout)
        engine = os.environ.get('HYQEC_CHECK_V2_WRONG_J_ENGINE')
        if engine:
            f = V2Fixture(5, FIVE, 2, 1)
            self.path.write_bytes(f.encode())
            v = verify(self.path, threads=1, engine=engine)
            self.assertEqual(v['status'], 'rejected')
            self.assertIn('justification failed', v['mismatches'][0])
