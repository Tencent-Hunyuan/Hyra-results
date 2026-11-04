"""Release transport/resource regressions, exercised through both public CLIs."""

import copy
import gzip
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from hyqec_check.algebra import ValidationError
from hyqec_check.certificate import MAGIC, validate_header
from hyqec_check.verify import verify
from hyqec_check_wide.verify import verify as verify_wide
from cert_factory import Fixture, block_product_fixture


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "proof"
        self.fixture = Fixture(4, ["XXXX", "ZZZZ"], 1, 1)
        self.raw = self.fixture.encode()

    def cli(self, module, data, *options):
        self.path.write_bytes(data)
        result = subprocess.run([sys.executable, "-m", module, "verify", str(self.path),
                                 "--threads", "1", *options], capture_output=True, text=True, timeout=20)
        self.assertNotIn("Traceback", result.stderr)
        return result.returncode, json.loads(result.stdout)

    def test_structured_gzip_and_nested_json_rejection(self):
        nested = b"[" * 2000 + b"0" + b"]" * 2000
        # BFINAL=1, BTYPE=3 is an invalid deflate block type.
        invalid_deflate = bytes.fromhex("1f8b0800000000000003") + b"\x07" + bytes(8)
        for module in ("hyqec_check", "hyqec_check_wide"):
            for data in (invalid_deflate, MAGIC + struct.pack("<Q", len(nested)) + nested):
                with self.subTest(module=module, data=data[:16]):
                    code, verdict = self.cli(module, data)
                    self.assertEqual(code, 1)
                    self.assertEqual(verdict["status"], "rejected")
                    self.assertFalse(verdict["certificate_verified"])

    def test_usage_is_distinct_from_partial(self):
        for module in ("hyqec_check", "hyqec_check_wide"):
            for options in (("--range", "-1:2"), ("--range", "2:1"), ("--sample", "-1"),
                            ("--threads", "0"), ("--unknown-option",), ("--range", "0:1:2"),
                            ("--max-proof-bytes", "-1")):
                with self.subTest(module=module, options=options):
                    code, verdict = self.cli(module, self.raw, *options)
                    self.assertEqual(code, 3)
                    self.assertEqual(verdict["status"], "usage_error")
                    self.assertFalse(verdict["certificate_verified"])
            code, verdict = self.cli(module, self.raw, "--sample", "0")
            self.assertEqual(code, 2)
            self.assertEqual(verdict["status"], "partial")

    def test_exact_proof_and_header_limits_raw_and_gzip(self):
        header_size, = struct.unpack("<Q", self.raw[8:16])
        proof_size = len(self.raw) - 16 - header_size
        for check in (verify, verify_wide):
            for data in (self.raw, gzip.compress(self.raw)):
                self.path.write_bytes(data)
                verdict = check(self.path, threads=1, max_header=header_size, max_proof_bytes=proof_size)
                self.assertTrue(verdict["certificate_verified"])
                for kwargs in ({"max_header": header_size - 1}, {"max_proof_bytes": proof_size - 1},
                               {"max_proof_bytes": 0}):
                    with self.assertRaises(ValidationError):
                        check(self.path, threads=1, **kwargs)

    def test_members_trailers_and_checksum_before_acceptance(self):
        # Body exceeds native buffering and crosses several pipe writes.
        raw, _ = block_product_fixture(240)
        header_size, = struct.unpack("<Q", raw[8:16])
        split = 16 + header_size
        self.assertGreater(len(raw) - split, 65536)
        members = gzip.compress(raw[:split]) + gzip.compress(raw[split:])
        broken_crc = bytearray(members)
        broken_crc[-8] ^= 1
        for module in ("hyqec_check", "hyqec_check_wide"):
            code, verdict = self.cli(module, members)
            self.assertEqual(code, 0)
            self.assertTrue(verdict["certificate_verified"])
            for data in (bytes(broken_crc), members[:-1], members + gzip.compress(b"extra"),
                         members + b"garbage", members + gzip.compress(raw)):
                with self.subTest(module=module, suffix=data[-10:]):
                    code, verdict = self.cli(module, data)
                    self.assertEqual(code, 1)
                    self.assertFalse(verdict["certificate_verified"])

    def test_gzip_bomb_rejected_after_bounded_decoding(self):
        header_size, = struct.unpack("<Q", self.raw[8:16])
        with gzip.open(self.path, "wb") as output:
            output.write(self.raw[:16 + header_size])
            for _ in range(64):
                output.write(bytes(1 << 20))
        read = gzip.GzipFile.read
        for check in (verify, verify_wide):
            decoded = 0

            def measured(stream, size=-1):
                nonlocal decoded
                data = read(stream, size)
                decoded += len(data)
                self.assertFalse(list(Path(self.temp.name).rglob("proof-records.bin")))
                return data

            with patch.object(gzip.GzipFile, "read", measured):
                verdict = check(self.path, threads=1, scratch_dir=self.temp.name)
            self.assertEqual(verdict["status"], "rejected")
            self.assertIn("wrong FRONTIER tag", verdict["mismatches"][0])
            self.assertLess(decoded, 512 * 1024)
            self.assertEqual(list(Path(self.temp.name).iterdir()), [self.path])

    def test_permutation_work_precheck_and_no_dimension_bypass(self):
        header = copy.deepcopy(self.fixture.header)
        header["permutations"] = [list(range(4))] * 10
        with patch("hyqec_check.certificate.parse_instance", side_effect=AssertionError("expensive parse")):
            with self.assertRaisesRegex(ValidationError, "work limit"):
                validate_header(header, max_permutation_work=79)
        validate_header(header, max_permutation_work=80)
        header["n"] = 1
        with self.assertRaisesRegex(ValidationError, "disagrees"):
            validate_header(header, max_permutation_work=20)
        for module in ("hyqec_check", "hyqec_check_wide"):
            header = copy.deepcopy(self.fixture.header)
            header["permutations"] = [list(range(4))]
            encoded = json.dumps(header).encode()
            old_size, = struct.unpack("<Q", self.raw[8:16])
            data = MAGIC + struct.pack("<Q", len(encoded)) + encoded + self.raw[16 + old_size:]
            for work, expected in ((7, 1), (8, 0)):
                code, verdict = self.cli(module, data, "--max-permutation-work", str(work))
                self.assertEqual(code, expected, verdict)
