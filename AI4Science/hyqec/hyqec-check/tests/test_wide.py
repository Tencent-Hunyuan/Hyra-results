"""Spec-only wide-limit checks, with a full-string tiny-code distance oracle."""

import copy
import gzip
import json
from hyqec_check.engines import environment
import subprocess
import sys

from hyqec_check.algebra import ValidationError
from hyqec_check.verify import verify as original_verify
from hyqec_check_wide.verify import DEFAULT_ENGINE, verify
from cert_factory import Fixture
import oracle
import test_certificate as original_tests


def ghz_with_logical_block(width, split=2, full=False, reverse=False):
    # A GHZ stabilizer state tensor the [[5,1,3]] code still has distance 3.
    # Its all-X generator exercises every support position. Cyclic permutations
    # reduce the roots and are independently validated by the checker.
    n = width + 5
    rows = ['X' * width + 'I' * 5]
    for q in range(width - 1):
        row = ['I'] * n
        row[q] = row[(q + 1) % width] = 'Z'
        rows.append(''.join(row))
    rows += ['I' * width + row for row in original_tests.FIVE]
    cycle = list(range(1, width)) + [0] + list(range(width, n))
    logical_cycle = list(range(width)) + list(range(width + 1, n)) + [width]
    return Fixture(n, rows, 2, split, permutations=[cycle, logical_cycle],
                   full=full, reverse_orbits=reverse)


class WideTests(original_tests.CertificateTests):
    # Inherited tests replay all original certificate cases with the new engine.
    engine = environment('WIDE_ENGINE', str(DEFAULT_ENGINE))

    def check(self, fixture, **kwargs):
        self.path.write_bytes(fixture.encode() if isinstance(fixture, Fixture) else fixture)
        return verify(self.path, threads=kwargs.pop('threads', 2), engine=self.engine, **kwargs)

    def test_wide_rows_and_unsigned_path_boundaries(self):
        for width in (33, 64, 65, 127, 128):
            fixture = ghz_with_logical_block(width)
            paths = {byte for _, path, _ in fixture.tasks for byte in path}
            self.assertIn(2 * (width - 1) + 1, paths)
            for data in (fixture.encode(), gzip.compress(fixture.encode(), mtime=0)):
                with self.subTest(width=width, compressed=data[:2] == b'\x1f\x8b'):
                    result = self.check(data)
                    self.assertTrue(result['certificate_verified'], result)
                    self.assertEqual(result['total_nodes'], fixture.footer.fields[-2])
                    self.assertEqual(int(result['global_trace']), fixture.footer.fields[-1])
        # Explicit transcripts and reversed orbits exercise restoration and a
        # fully free, inactive wide row (256 child options) at the first root.
        for split, full, reverse in ((1, False, True), (1, True, False), (2, True, True), (3, False, False)):
            result = self.check(ghz_with_logical_block(128, split, full, reverse))
            self.assertTrue(result['certificate_verified'], result)

    def test_wide_degree_against_exhaustive_distance(self):
        cases = [(4, ['XXXX', 'ZZZZ']), (5, original_tests.FIVE)]
        cases += [(n, rows) for n, rows in oracle.random_codes(repetitions=2)
                  if rows and n > 1]
        tested = 0
        for n, rows in cases:
            maximum = max(sum(row[q] != 'I' for row in rows) for q in range(n))
            if not maximum:
                continue
            expanded = rows * (128 // maximum) + ['I' * n]
            probe = Fixture(n, expanded, 0, 1)
            if probe.code.k == 0:
                continue
            self.assertGreater(max(sum(row[q] != 'I' for row in expanded) for q in range(n)), 64)
            for domain in ('P', 'X', 'Z'):
                # Duplicate rows do not alter the explicit group closure.
                distance = oracle.distance(n, rows, domain)
                cap = n if distance is None else distance - 1
                for split in sorted({1, 2, cap + 1}):
                    for full in (False, True):
                        with self.subTest(n=n, domain=domain, split=split, full=full):
                            result = self.check(Fixture(n, expanded, cap, split, domain, full=full))
                            self.assertTrue(result['certificate_verified'], result)
                if distance is not None:
                    # A fabricated self-consistent transcript must fail when
                    # replay discovers the exhaustively known logical witness.
                    self.reject(Fixture(n, expanded, distance, 1, domain, allow_logical=True))
                tested += 1
        self.assertGreaterEqual(tested, 6)

    def test_wide_tampering_and_partial_modes(self):
        fixture = ghz_with_logical_block(128)
        task = next(i for i, (_, path, _) in enumerate(fixture.tasks) if 255 in path)
        for replacement in (127, 254, 0):
            bad = copy.deepcopy(fixture)
            path = bad.task_records[task][-1].fields[3]
            bad.task_records[task][-1].fields[3] = tuple(replacement if p == 255 else p for p in path)
            self.reject(bad)
        for kind in ('top', 'task', 'footer', 'cap'):
            bad = copy.deepcopy(fixture)
            if kind == 'top':
                bad.top.pop()
            elif kind == 'task':
                bad.task_records[task][-1].fields[-1] ^= 1
            elif kind == 'footer':
                bad.footer.fields[-1] ^= 1
            else:
                bad.header['cap'] = 3
            self.reject(bad)
        for options in (dict(mode='range', start=task, stop=task + 1),
                        dict(mode='sample', sample=5, seed=132)):
            result = self.check(fixture, **options)
            self.assertEqual(result['status'], 'partial')
            self.assertFalse(result['certificate_verified'])

    def test_wide_limit_rejection_and_original_isolation(self):
        for degree in (65, 127, 128):
            fixture = Fixture(4, ['XXXX'] * (degree - 1) + ['ZZZZ'], 1, 1)
            self.assertTrue(self.check(fixture)['certificate_verified'])
            with self.assertRaises(ValidationError):
                original_verify(self.path, threads=1)
        self.reject(Fixture(4, ['XXXX'] * 128 + ['ZZZZ'], 0, 1))
        # Construct an over-limit row without any search.
        self.reject(Fixture(130, ['X' * 129 + 'I'], 0, 1))
        fixture = ghz_with_logical_block(128)
        self.path.write_bytes(fixture.encode())
        with self.assertRaises(ValidationError):
            original_verify(self.path, threads=1)

    def test_wide_cli_version_and_claim_binding(self):
        version = subprocess.run([sys.executable, '-m', 'hyqec_check_wide', '--version'],
                                 capture_output=True, text=True, check=True)
        self.assertIn('1.3.0 (spec 1.1)', version.stdout)
        self.path.write_bytes(ghz_with_logical_block(128).encode())
        command = [sys.executable, '-m', 'hyqec_check_wide', 'verify', str(self.path), '--threads', '1']
        for options, exitcode in (([], 0), (['--sample', '0'], 2), (['--expected-cap', '1'], 1)):
            result = subprocess.run(command + options, capture_output=True, text=True)
            self.assertEqual(result.returncode, exitcode, result.stdout)
            verdict = json.loads(result.stdout)
            self.assertEqual(verdict['checker_version'], '1.3.0')
            self.assertEqual(verdict['spec_revision'], '1.1')
            self.assertEqual(verdict['certificate_verified'], exitcode == 0)
