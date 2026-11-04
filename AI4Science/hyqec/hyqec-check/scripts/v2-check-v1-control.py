"""Time the frozen wide-v1 checker on the exported 488 greedy twin."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export_dir', type=Path)
    parser.add_argument('result_dir', type=Path)
    parser.add_argument('--threads', type=int, default=8, choices=range(1, 17))
    args = parser.parse_args()
    root, out = args.export_dir.resolve(), args.result_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = root / 'v1-pair/MANIFEST.txt'
    line = next(line for line in manifest.read_text().splitlines() if line.startswith('488-'))
    name = line.split()[0]
    path = manifest.parent / name
    value = lambda key: re.search(r'\b' + key + r'=(\S+)', line).group(1)
    assert digest(path) == value('sha256')
    frozen = json.loads(Path('docs/V1_FROZEN_SHA256.json').read_text())
    for filename, expected in frozen.items():
        assert digest(Path(filename)) == expected, filename
    command = [sys.executable, '-m', 'hyqec_check_wide', 'verify', str(path),
               '--threads', str(args.threads), '--expected-domain', 'P', '--expected-cap', '16']
    report = dict(checker_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  command=command, affinity=sorted(os.sched_getaffinity(0)), threads=args.threads,
                  native_sha256=frozen['build/hyqec-check-wide-native-v1.3.0'],
                  certificate_sha256=digest(path), manifest_sha256=digest(manifest),
                  expected_nodes=int(value('nodes')), expected_trace=value('trace'),
                  started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    print('START ' + json.dumps(report), flush=True)
    started = time.perf_counter()
    with (out / 'v1-control.stdout.json').open('w') as stdout, (out / 'v1-control.stderr.log').open('w') as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr)
    report['process_wall_seconds'] = time.perf_counter() - started
    verdict = json.loads((out / 'v1-control.stdout.json').read_text())
    report.update(verdict=verdict, returncode=result.returncode)
    report['matched'] = (result.returncode == 0 and verdict.get('certificate_verified') is True
                         and verdict.get('checker_version') == '1.3.0'
                         and verdict.get('total_nodes') == report['expected_nodes']
                         and verdict.get('global_trace_hex') == report['expected_trace']
                         and verdict.get('tasks_checked') == verdict.get('tasks_total') == 168774
                         and 'rules' not in verdict)
    for filename, expected in frozen.items():
        assert digest(Path(filename)) == expected, filename
    (out / 'v1-control.json').write_text(json.dumps(report, indent=2) + '\n')
    print('FINISH ' + json.dumps(report), flush=True)
    return 0 if report['matched'] else 1


if __name__ == '__main__':
    sys.exit(main())
