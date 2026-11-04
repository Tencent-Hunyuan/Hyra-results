"""Full sequential replays and wall timings of the data-only medium export.

Run from the checkout: python3 scripts/v2-check-medium.py EXPORT_DIR RESULT_DIR
Uses eight native threads by default; the v1 twin is checked for byte identity.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def field(line, name):
    return re.search(r'\b' + name + r'=(\S+)', line).group(1)


def header(stream):
    magic = stream.read(8)
    length = int.from_bytes(stream.read(8), 'little')
    assert length < 128 << 20
    return magic, json.loads(stream.read(length))


def pair_identity(export):
    line = next(line for line in (export / 'v1-pair/MANIFEST.txt').read_text().splitlines()
                if line.startswith('488-'))
    one = export / 'v1-pair' / line.split()[0]
    two = (one.parent / field(line, 'pairs_with')).resolve()
    hashes = re.findall(r'\bsha256=(\S+)', line)
    assert sha(one) == hashes[0] and sha(two) == hashes[1]
    digest = hashlib.sha256()
    size = 0
    with gzip.open(one, 'rb') as left, gzip.open(two, 'rb') as right:
        lm, lh = header(left)
        rm, rh = header(right)
        assert lm == b'SBCOV1\r\n' and rm == b'SBCOV2\r\n'
        assert rh['rules'] == 'greedy'
        for key in ('format', 'search', 'trace', 'rules', 'rules_version', 'provenance'):
            lh.pop(key, None)
            rh.pop(key, None)
        assert lh == rh
        while True:
            a, b = left.read(1 << 20), right.read(1 << 20)
            assert a == b
            if not a:
                break
            digest.update(a)
            size += len(a)
    assert digest.hexdigest() == field(line, 'record_stream_sha256')
    assert size == int(field(line, 'record_stream_bytes'))
    return dict(v1_file=one.name, v1_sha256=hashes[0], v2_file=two.name, v2_sha256=hashes[1],
                record_stream_bytes=size, record_stream_sha256=digest.hexdigest(),
                nodes=int(field(line, 'nodes')), trace=field(line, 'trace'),
                record_stream_identical=True, v1_full_replay=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export_dir', type=Path)
    parser.add_argument('result_dir', type=Path)
    parser.add_argument('--threads', type=int, default=8, choices=range(1, 17))
    args = parser.parse_args()
    export, out = args.export_dir.resolve(), args.result_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = export / 'medium/MANIFEST.txt'
    snapshot = manifest.read_bytes()
    (out / 'medium-MANIFEST.txt').write_bytes(snapshot)
    cpu_model = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                      if line.startswith('model name')), 'unknown')
    report = dict(checker_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  native_sha256=sha(Path('build/hyqec-check-v2-native-v2.0.0')),
                  threads=args.threads, cpu_model=cpu_model,
                  runner_affinity_at_start=sorted(os.sched_getaffinity(0)),
                  manifest_sha256=hashlib.sha256(snapshot).hexdigest(),
                  pair=pair_identity(export), replays=[])
    summary = out / 'medium-conformance.json'
    summary.write_text(json.dumps(report, indent=2) + '\n')
    print('v1/v2 record identity verified: ' + json.dumps(report['pair']), flush=True)
    for line in snapshot.decode().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        name = line.split()[0]
        path = manifest.parent / name
        expected_sha = field(line, 'sha256')
        assert sha(path) == expected_sha
        with gzip.open(path, 'rb') as stream:
            magic, h = header(stream)
        assert magic == b'SBCOV2\r\n'
        assert h['split'] == int(field(line, 'split')) and h['detail'] == field(line, 'detail')
        command = [sys.executable, '-m', 'hyqec_check_v2', 'verify', str(path),
                   '--threads', str(args.threads), '--expected-instance', field(line, 'instance_sha256'),
                   '--expected-domain', field(line, 'domain'), '--expected-cap', field(line, 'cap')]
        print('START ' + name + ' ' + time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), flush=True)
        started = time.perf_counter()
        with (out / (name + '.stdout.json')).open('w') as stdout, (out / (name + '.stderr.log')).open('w') as stderr:
            process = subprocess.run(command, stdout=stdout, stderr=stderr)
        elapsed = time.perf_counter() - started
        verdict = json.loads((out / (name + '.stdout.json')).read_text())
        expected = dict(nodes=int(field(line, 'nodes')), tasks=int(field(line, 'tasks')),
                        trace=field(line, 'trace'), rules=field(line, 'rules'))
        matched = (process.returncode == 0 and verdict.get('status') == 'accepted'
                   and verdict.get('certificate_verified') is True
                   and verdict.get('total_nodes') == expected['nodes']
                   and verdict.get('tasks_checked') == verdict.get('tasks_total') == expected['tasks']
                   and verdict.get('global_trace_hex') == expected['trace']
                   and verdict.get('rules') == expected['rules'])
        item = dict(file=name, sha256=expected_sha, expected=expected, command=command,
                    process_wall_seconds=elapsed, returncode=process.returncode, verdict=verdict, matched=matched)
        report['replays'].append(item)
        if expected['rules'] == 'greedy':
            report['pair']['trace_matches_recomputed_v2'] = (
                report['pair']['nodes'] == verdict.get('total_nodes')
                and report['pair']['trace'] == verdict.get('global_trace_hex'))
        summary.write_text(json.dumps(report, indent=2) + '\n')
        print('FINISH ' + json.dumps(item), flush=True)
        if not matched:
            return 1
    assert report['replays'], 'no selected certificate in manifest'
    return 0


if __name__ == '__main__':
    sys.exit(main())
