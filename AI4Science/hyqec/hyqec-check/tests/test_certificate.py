import copy
import gzip
import hashlib
import json
from hyqec_check.engines import environment
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from hyqec_check.algebra import ValidationError
from hyqec_check.certificate import MAGIC, parse_instance, validate_header
from hyqec_check.trace import BASE, concatenate, entry, fingerprint, mix64, packed
from hyqec_check.verify import DEFAULT_ENGINE, verify
from cert_factory import Fixture, block_product_fixture
import oracle

VECTORS = Path(__file__).parent / 'spec_vectors.json'
FIVE = ['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ']
SHOR = ['ZZIIIIIII','IZZIIIIII','IIIZZIIII','IIIIZZIII','IIIIIIZZI','IIIIIIIZZ','XXXXXXIII','IIIXXXXXX']


class CertificateTests(unittest.TestCase):
    engine = environment('NATIVE_ENGINE', str(DEFAULT_ENGINE))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'proof'

    def check(self, fixture, **kwargs):
        self.path.write_bytes(fixture.encode() if isinstance(fixture, Fixture) else fixture)
        return verify(self.path, threads=kwargs.pop('threads', 2), engine=self.engine, **kwargs)

    def reject(self, fixture, **kwargs):
        try:
            result = self.check(fixture, **kwargs)
        except (ValidationError, OSError, EOFError):
            return
        self.assertEqual(result['status'], 'rejected', result)
        self.assertFalse(result['certificate_verified'])

    def test_published_vectors(self):
        vectors = json.loads(VECTORS.read_text())
        self.assertEqual(BASE, int(vectors['trace_base'], 16))
        for argument, expected in vectors['mix_vectors'].items():
            self.assertEqual(mix64(int(argument)), int(expected, 16))
        total = (0, 0)
        for record in vectors['records']:
            self.assertEqual(fingerprint(record['lab']), int(record['state_fp'], 16))
            if record['node_count']:
                active = 2 if record['root'] % 3 == 2 else 1
                summary = entry(record['lab'], 1, active, 1)
                self.assertEqual(packed(1, active, 1), int(record['packed'], 16))
                self.assertEqual(summary[1], int(record['trace'], 16))
                total = concatenate(total, summary)
        self.assertEqual(total, (9, int(vectors['global_trace'], 16)))
        for split in (1, 2):
            for full in (False, True):
                fixture = Fixture(4, ['XXXX', 'ZZZZ'], 1, split, full=full,
                                  logicals=['XXII', 'XIXI', 'ZZII', 'ZIZI'])
                result = self.check(fixture)
                self.assertTrue(result['certificate_verified'])
                self.assertEqual(result['total_nodes'], 9)
                self.assertEqual(int(result['global_trace']), total[1])
                self.assertEqual(result['tasks_total'], 12 if split == 1 else 0)

    def test_deterministic_split_and_full_transcripts(self):
        # Shor's code exercises genuine disjoint packing and early suffixes.
        for n, rows, cap, sym in [(5, FIVE, 2, [[1,2,3,4,0]]), (9, SHOR, 2, [])]:
            for domain in ('P','X','Z'):
                common = None
                for split in (1, 2, 3):
                    for full in (False, True):
                        with self.subTest(n=n, domain=domain, split=split, full=full):
                            f = Fixture(n, rows, cap, split, domain, sym, full=full)
                            result = self.check(f, threads=4)
                            self.assertTrue(result['certificate_verified'], result)
                            current = (result['total_nodes'], result['global_trace'])
                            if common is not None:
                                self.assertEqual(current, common)
                            common = current

    def test_random_code_lower_bounds_against_exhaustive_distance(self):
        for n, rows in oracle.random_codes(repetitions=3):
            f = Fixture(n, rows, 0, 1)
            if f.code.k == 0:
                self.reject(f)
                continue
            for domain in ('P','X','Z'):
                distance = oracle.distance(n, rows, domain)
                cap = n if distance is None else distance - 1
                for split in sorted({1, cap + 1, min(47, cap + 2)}):
                    result = self.check(Fixture(n, rows, cap, split, domain, reverse_orbits=True))
                    self.assertTrue(result['certificate_verified'], result)
                if distance is not None:
                    self.reject(Fixture(n, rows, distance, 1, domain, allow_logical=True))

    def test_transport_and_truncation(self):
        raw = Fixture(5, FIVE, 2, 2).encode()
        compressed = gzip.compress(raw, mtime=0)
        self.assertTrue(self.check(compressed)['certificate_verified'])
        for data in (raw[:-1], raw[:16], raw[:7], compressed[:-1], compressed[:-8],
                     raw+b'x', gzip.compress(raw+b'x'), compressed+gzip.compress(b'x')):
            self.reject(data)
        self.reject(raw, max_header=1)
        self.reject(compressed, max_proof_bytes=1)

    def test_tampered_task_top_hint_and_claim(self):
        base = Fixture(5, FIVE, 2, 2)
        self.assertGreater(len(base.task_records), 0)
        mutations = []
        f=copy.deepcopy(base); f.task_records.pop(0); mutations.append(f)
        f=copy.deepcopy(base); f.task_records[0][-1].fields[-2]+=1; mutations.append(f)
        f=copy.deepcopy(base); f.task_records[0][-1].fields[-1]^=1; mutations.append(f)
        f=copy.deepcopy(base); f.task_records[0][-1].fields[3]=(63,); mutations.append(f)
        f=copy.deepcopy(base); f.top.pop(0); mutations.append(f)
        f=copy.deepcopy(base); f.top[0].fields[1]=1; mutations.append(f)
        f=copy.deepcopy(base); f.top[0].fields[3]^=1; mutations.append(f)
        f=copy.deepcopy(base); f.header['cap']=1; mutations.append(f)
        f=copy.deepcopy(base); f.header['k']=2; mutations.append(f)
        f=copy.deepcopy(base); f.header['permutations']=[[1,1,2,3,4]]; mutations.append(f)
        f=copy.deepcopy(base); f.footer.fields[-1]^=1; mutations.append(f)
        shor=Fixture(9, SHOR, 2, 3, orbit_order=[1,0,2,4,3,5,7,6,8])
        self.assertTrue(self.check(shor)['certificate_verified'])
        pack=next(i for i,r in enumerate(shor.top) if r.tag==0x10 and r.fields[2]==2)
        f=copy.deepcopy(shor); f.top[pack].fields[-1]=1000; mutations.append(f)
        dead=next(i for i,r in enumerate(shor.top) if r.tag in (0x11,0x12))
        f=copy.deepcopy(shor); f.top[dead].fields[2]=1000; mutations.append(f)
        suffix=next(i for i,r in enumerate(shor.top) if r.tag==0x12)
        f=copy.deepcopy(shor); f.top.pop(suffix); mutations.append(f)
        for i, fixture in enumerate(mutations):
            with self.subTest(mutation=i):
                self.reject(fixture)
        self.reject(base, expected_cap=1)
        self.reject(base, expected_domain='X')
        self.reject(base, expected_group='0'*64)
        self.reject(base, expected_instance='0'*64)
        result=self.check(base, expected_cap=2, expected_group=base.header['group_sha256'], expected_domain='P')
        self.assertTrue(result['certificate_verified'])

    def test_sample_and_range_are_partial(self):
        f=Fixture(5, FIVE, 2, 2)
        for mode, kwargs, count in [('sample',dict(sample=5, seed=116),5), ('range',dict(start=2,stop=4),2)]:
            first=self.check(f,mode=mode,**kwargs)
            second=self.check(f,mode=mode,**kwargs)
            self.assertEqual(first['status'],'partial')
            self.assertFalse(first['certificate_verified'])
            self.assertEqual(first['tasks_checked'],count)
            self.assertEqual(first['replayed_task_nodes'],second['replayed_task_nodes'])
        bad=copy.deepcopy(f); bad.task_records.pop()
        self.reject(bad,mode='sample',sample=0)
        self.reject(f,mode='range',start=0,stop=100000)
        full=Fixture(5,FIVE,2,2,full=True)
        result=self.check(full,mode='range',start=1,stop=3)
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['tasks_checked'],2)

    def test_cli_expected_cap_rejects_higher_and_lower(self):
        self.path.write_bytes(Fixture(5, FIVE, 2, 2).encode())
        command = [sys.executable, '-m', 'hyqec_check', 'verify', str(self.path), '--threads', '1']
        for requested in (1, 2, 3):
            with self.subTest(expected_cap=requested):
                result = subprocess.run(command + ['--expected-cap', str(requested)],
                                        capture_output=True, text=True)
                verdict = json.loads(result.stdout)
                if requested == 2:
                    self.assertEqual(result.returncode, 0, verdict)
                    self.assertTrue(verdict['certificate_verified'])
                else:
                    self.assertEqual(result.returncode, 1, verdict)
                    self.assertFalse(verdict['certificate_verified'])
                    self.assertIn('validated cap does not match requested value', verdict['mismatches'])

    def test_five_qubit_cap_edits_rejected_by_independent_replay(self):
        # This code has d=3: accepting its cap2 proof as cap3 would be unsound.
        self.assertEqual(oracle.distance(5, FIVE, 'P'), 3)
        for split in (1, 2, 3):
            for full in (False, True):
                base = Fixture(5, FIVE, 2, split, full=full)
                self.assertTrue(self.check(base, threads=1)['certificate_verified'])
                original = base.encode()
                original_payload = original[16 + struct.unpack('<Q', original[8:16])[0]:]
                for edited_cap in (1, 3):
                    with self.subTest(split=split, full=full, edited_cap=edited_cap):
                        edited = copy.deepcopy(base)
                        edited.header['cap'] = edited_cap
                        raw = edited.encode()
                        self.assertEqual(raw[16 + struct.unpack('<Q', raw[8:16])[0]:], original_payload)
                        # Match the edited header deliberately: rejection must come
                        # from replay, not the expected-cap equality check.
                        verdict = self.check(raw, threads=1, expected_cap=edited_cap)
                        self.assertEqual(verdict['status'], 'rejected', verdict)
                        self.assertFalse(verdict['certificate_verified'])
                        self.assertTrue(verdict['mismatches'], verdict)
                        self.assertNotIn('validated cap does not match requested value', verdict['mismatches'])

    def test_cross_word_coordinates_and_group_identity(self):
        # Appending fixed-|0> ancillas preserves the five-qubit code's distance.
        # This exercises X/Z coordinates on both sides of multiple 64-bit words.
        for n in (65, 130):
            rows = [row + 'I' * (n - 5) for row in FIVE]
            rows += ['I' * q + 'Z' + 'I' * (n - q - 1) for q in range(5, n)]
            result = self.check(Fixture(n, rows, 2, 2))
            self.assertTrue(result['certificate_verified'], result)
        f = Fixture(4, ['XXXX','ZZZZ'], 1, 2)
        expected = hashlib.sha256(b'stabcert-group/1 n=4 rank=2\n\x0f\xf0').hexdigest()
        self.assertEqual(f.header['group_sha256'], expected)
        alternative = Fixture(4, ['YYYY','ZZZZ','IIII','XXXX'], 1, 2)
        self.assertEqual(alternative.header['group_sha256'], expected)
        self.assertNotEqual(alternative.header['instance_sha256'], f.header['instance_sha256'])
        self.assertTrue(self.check(alternative)['certificate_verified'])

    def test_streaming_batch_boundary(self):
        raw, expected = block_product_fixture()
        result = self.check(raw, threads=4)
        self.assertTrue(result['certificate_verified'], result)
        self.assertEqual(result['tasks_total'], 4200)
        self.assertEqual(result['total_nodes'], 3150)
        self.assertEqual((result['total_nodes'], int(result['global_trace'])), expected)
        partial = self.check(raw, threads=4, mode='range', start=4094, stop=4101)
        self.assertEqual(partial['tasks_checked'], 7)
        self.assertEqual(partial['status'], 'partial')
        self.assertEqual(partial['global_trace'], result['global_trace'])

    def test_cli_exit_status_distinguishes_partial_from_acceptance(self):
        fixture = Fixture(4, ['XXXX', 'ZZZZ'], 1, 1)
        self.path.write_bytes(fixture.encode())
        command = [sys.executable, '-m', 'hyqec_check', 'verify', str(self.path), '--threads', '1']
        for selection, expected_exit, status in [([], 0, 'accepted'),
                                                 (['--sample', '0'], 2, 'partial'),
                                                 (['--range', '0:1'], 2, 'partial')]:
            result = subprocess.run(command + selection, capture_output=True, text=True)
            self.assertEqual(result.returncode, expected_exit, result.stdout)
            verdict = json.loads(result.stdout)
            self.assertEqual(verdict['status'], status)
            self.assertEqual(verdict['certificate_verified'], expected_exit == 0)
        self.path.write_bytes(fixture.encode()[:-1])
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(json.loads(result.stdout)['certificate_verified'])

    def test_strict_binary_and_header_parsing(self):
        f=Fixture(4,['XXXX','ZZZZ'],1,2)
        header=json.dumps(f.header).encode()
        prefix=MAGIC+struct.pack('<Q',len(header))+header
        records=b''.join(r.encode() for r in f.top+[f.top_end,f.footer])
        # Root parent_ref=0 encoded nonminimally, followed by the rest unchanged.
        self.reject(prefix+records[:1]+b'\x80\x00'+records[2:])
        self.reject(prefix+records[:1]+b'\xff'*10+records[2:])
        dup=b'{"cap":1,'+header[1:]
        self.reject(MAGIC+struct.pack('<Q',len(dup))+dup+records)
        for override in ({'cap':True},{'n':4.0},{'amax':95},{'instance_sha256':'f'*64},
                         {'group_sha256':'f'*64},{'domain':'X'}):
            bad=copy.deepcopy(f); bad.header.update(override); self.reject(bad)
        # Header provenance is informational; changing it does not affect proof.
        f.header['provenance']['binary_sha256']='a'*64
        self.assertTrue(self.check(f)['certificate_verified'])

    def test_instance_syntax_and_logical_basis(self):
        f=Fixture(4,['XXXX','ZZZZ','IIII'],1,2)
        text=f.header['instance_text']
        good=parse_instance(text)
        legacy='\n'.join(text.splitlines()[1:-1])+'\n'
        self.assertEqual(parse_instance(legacy).group_sha256,good.group_sha256)
        for bad in (text+'rootmask 1',text.replace('4 3 4','+4 3 4',1),text+'\0',text+'\u00a0'):
            with self.assertRaises(ValidationError):
                parse_instance(bad)
        broken=copy.deepcopy(f.header)
        # Replace a quotient row by identity while updating exact-byte identity.
        lines=broken['instance_text'].splitlines(); lines[5]='0'
        broken['instance_text']='\n'.join(lines)+'\n'
        broken['instance_sha256']=hashlib.sha256(broken['instance_text'].encode()).hexdigest()
        with self.assertRaises(ValidationError):
            validate_header(broken)


if __name__=='__main__':
    unittest.main()
