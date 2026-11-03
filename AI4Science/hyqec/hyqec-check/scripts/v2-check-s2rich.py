"""Full sequential conformance replays of the data-only S2-heavy exports.

Run from the checkout, pinned with taskset:
python3 scripts/v2-check-s2rich.py EXPORT_CERTS_DIR RESULT_DIR --threads 4
Record census is a claim inventory; only the public checker verifies proofs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time


audit = runpy.run_path(str(Path(__file__).with_name('v2-check-exported-certs.py')))
field = audit['field']


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            digest.update(data)
    return digest.hexdigest()


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def frozen():
    expected = json.loads(Path('docs/V1_FROZEN_SHA256.json').read_text())
    changed = [name for name, digest in expected.items() if sha(Path(name)) != digest]
    assert not changed, changed
    return dict(files_checked=len(expected), changed=changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export_dir', type=Path)
    parser.add_argument('result_dir', type=Path)
    parser.add_argument('--threads', type=int, default=4, choices=range(1, 17))
    parser.add_argument('--only', help='replay only this manifest filename, for per-run CPU coordination')
    args = parser.parse_args()
    affinity = sorted(os.sched_getaffinity(0))
    assert set(affinity) <= {180, 181, 372, 373}, affinity
    export, out = args.export_dir.resolve() / 's2rich', args.result_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary = out / 'conformance.json'
    assert not summary.exists(), 'use a fresh result directory to preserve earlier evidence'
    manifest = (export / 'MANIFEST.txt').read_bytes()
    (out / 'MANIFEST.txt').write_bytes(manifest)
    (out / 'README.txt').write_bytes((export / 'README.txt').read_bytes())
    entries = [line for line in manifest.decode().splitlines() if line.strip() and not line.startswith('#')]
    assert len(entries) == 5
    if args.only:
        entries = [line for line in entries if line.split()[0] == args.only]
        assert len(entries) == 1, 'filename not present in manifest'
    native = Path('build/hyqec-check-v2-native-v2.0.0')
    report = dict(checker_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  native_sha256=sha(native), runner_sha256=sha(Path(__file__)),
                  inventory_sha256=sha(Path(__file__).with_name('v2-check-exported-certs.py')),
                  threads=args.threads, affinity=affinity, export=str(export), started_utc=utc(),
                  selected_files=[line.split()[0] for line in entries],
                  cpu_model=next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                                 if line.startswith('model name')),
                  manifest_sha256=hashlib.sha256(manifest).hexdigest(),
                  frozen_v1_before=frozen(), replays=[])
    summary.write_text(json.dumps(report, indent=2) + '\n')
    for line in entries:
        name = line.split()[0]
        path = export / name
        digest = sha(path)
        assert digest == field(line, 'sha256'), name
        header, raw, start = audit['unpack'](path)
        assert raw[:8] == b'SBCOV2\r\n'
        _, census, footer = audit['inventory'](raw, start, keep_records=False)
        del raw
        expected_census = {}
        for source, context in [('top_rules', 'top'), ('transcript_rules', 'task')]:
            encoded = field(line, source)
            if encoded != 'n/a':
                for pair in encoded.split(','):
                    rule, count = pair.split(':')
                    if int(count):
                        expected_census[f'{context}:rule{rule}'] = int(count)
        assert census == expected_census, (name, census, expected_census)
        assert header['split'] == int(field(line, 'split'))
        assert header['detail'] == field(line, 'detail')
        expected = dict(nodes=int(field(line, 'nodes')), tasks=int(field(line, 'tasks')),
                        trace=field(line, 'trace'), rules=field(line, 'rules'),
                        instance=field(line, 'instance_sha256'), domain=field(line, 'domain'),
                        cap=int(field(line, 'cap')))
        command = [sys.executable, '-m', 'hyqec_check_v2', 'verify', str(path), '--threads', str(args.threads),
                   '--expected-instance', expected['instance'], '--expected-domain', expected['domain'],
                   '--expected-cap', str(expected['cap'])]
        started_utc = utc()
        print('START ' + name + ' ' + started_utc, flush=True)
        started = time.perf_counter()
        with (out / (name + '.stdout.json')).open('w') as stdout, (out / (name + '.stderr.log')).open('w') as stderr:
            process = subprocess.run(command, stdout=stdout, stderr=stderr)
        elapsed = time.perf_counter() - started
        finished_utc = utc()
        verdict = json.loads((out / (name + '.stdout.json')).read_text())
        matched = (process.returncode == 0 and verdict.get('status') == 'accepted'
                   and verdict.get('certificate_verified') is True and verdict.get('selection') == 'all'
                   and verdict.get('total_nodes') == expected['nodes'] == footer['total_nodes']
                   and verdict.get('tasks_checked') == verdict.get('tasks_total') == expected['tasks'] == footer['tasks']
                   and verdict.get('global_trace_hex') == expected['trace'] == footer['global_trace_hex']
                   and verdict.get('gen_nodes') == footer['gen_nodes']
                   and verdict.get('task_nodes') == footer['task_nodes']
                   and verdict.get('rules') == expected['rules'])
        item = dict(file=name, sha256=digest, expected=expected, command=command,
                    detail=header['detail'], requested_threads=args.threads, effective_threads=verdict.get('threads'),
                    started_utc=started_utc, finished_utc=finished_utc, affinity=sorted(os.sched_getaffinity(0)),
                    process_wall_seconds=elapsed, returncode=process.returncode,
                    census=census, census_matches_manifest=True, footer=footer, verdict=verdict, matched=matched)
        report['replays'].append(item)
        summary.write_text(json.dumps(report, indent=2) + '\n')
        print('FINISH ' + json.dumps(item), flush=True)
        if not matched:
            return 1
    report.update(frozen_v1_after=frozen(), finished_utc=utc(), all_matched=True)
    assert sha(native) == report['native_sha256']
    summary.write_text(json.dumps(report, indent=2) + '\n')
    print(f'ALL {len(entries)} SELECTED CERTIFICATES MATCHED', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
