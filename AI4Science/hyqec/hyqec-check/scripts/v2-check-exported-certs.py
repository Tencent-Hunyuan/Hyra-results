"""Audit the data-only small certificate export using the public checker CLI.

Usage: python3 scripts/v2-check-exported-certs.py EXPORT_DIR RESULT_DIR
The record scanner inventories claims/positions; it is not a proof verifier.
The public checker independently replays every top node and task.
"""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def sha(data):
    return hashlib.sha256(data).hexdigest()


def field(line, name):
    return re.search(r'\b' + name + r'=(\S+)', line).group(1)


def unpack(path):
    raw = gzip.decompress(path.read_bytes())
    start = 16 + int.from_bytes(raw[8:16], 'little')
    return json.loads(raw[16:start]), raw, start


def inventory(raw, start, keep_records=True):
    at = start
    records = []
    census = collections.Counter()
    task = None
    local_event = 0
    footer = None

    def integer():
        nonlocal at
        value = shift = 0
        while True:
            b = raw[at]
            at += 1
            value |= (b & 127) << shift
            if b < 128:
                return value
            shift += 7
            assert shift < 70

    def fixed():
        nonlocal at
        assert at + 8 <= len(raw)
        value = int.from_bytes(raw[at:at+8], 'little')
        at += 8
        return value

    while at < len(raw):
        begin = at
        tag = raw[at]
        at += 1
        record = dict(offset=begin, record_offset=begin-start, tag=f'{tag:02x}',
                      task=task, event=local_event)
        if tag == 0x10:
            record.update(parent=integer(), edge=integer())
            rule = raw[at]
            at += 1
            record.update(rule=rule, fingerprint=fixed(), weight=integer(), active=integer())
            if rule == 2 or rule in (5, 6, 7):
                record['packing'] = [integer() for _ in range(integer())]
            elif rule == 3:
                record['generator'] = integer()
            census[f'{"top" if task is None else "task"}:rule{rule}'] += 1
            local_event += 1
        elif tag in (0x11, 0x12, 0x13):
            integer(); integer(); integer(); fixed()
            local_event += 1
        elif tag == 0x1f:
            record.update(events=integer(), tasks=integer(), nodes=integer(), trace=f'{fixed():016x}')
        elif tag == 0x20:
            record.update(task_id=integer(), root=integer())
            length = integer()
            record['path'] = list(raw[at:at+length])
            at += length
            record.update(nodes=integer(), trace=f'{fixed():016x}')
        elif tag == 0x21:
            task = integer()
            local_event = 0
            record['task'] = task
        elif tag == 0x22:
            record['events'] = integer()
            task = None
        elif tag == 0x7f:
            footer = dict(tasks=integer(), gen_nodes=integer(), task_nodes=integer(),
                          total_nodes=integer(), global_trace_hex=f'{fixed():016x}')
            record.update(footer)
            assert at == len(raw), 'data follows COMPLETE'
        else:
            raise ValueError(f'unknown tag {tag:02x} at {begin}')
        record['end_offset'] = at
        if keep_records:
            records.append(record)
    assert footer is not None
    return records, dict(census), footer


def run(path, out, bindings=None):
    command = [sys.executable, '-m', 'hyqec_check_v2', 'verify', str(path), '--threads', '1']
    if bindings:
        for arg in ('instance', 'domain', 'cap'):
            command.extend([f'--expected-{arg}', str(bindings[arg])])
    process = subprocess.run(command, capture_output=True, text=True, timeout=180)
    out.with_suffix('.stdout.json').write_text(process.stdout)
    out.with_suffix('.stderr.log').write_text(process.stderr)
    return process.returncode, json.loads(process.stdout)


def main():
    export = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    report = dict(checker_revision=revision, export=str(export), accepts=[], rejects=[], v1_pairs=[])
    report['native_sha256'] = {str(p): sha(p.read_bytes()) for p in (
        Path('build/hyqec-check-v2-native-v2.0.0'), Path('build/hyqec-check-wide-native-v1.3.0'))}
    report['manifest_sha256'] = {}
    cache = {}
    failures = []
    for relative in ('MANIFEST.txt', 'invalid/MANIFEST.txt', 'v1-pair/MANIFEST.txt'):
        manifest = export / relative
        if not manifest.exists():
            continue
        report['manifest_sha256'][relative] = sha(manifest.read_bytes())
        (output / relative.replace('/', '-')).write_bytes(manifest.read_bytes())
        for line in manifest.read_text().splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            name = line.split()[0]
            path = manifest.parent / name
            data = path.read_bytes()
            assert sha(data) == field(line, 'sha256'), name
            result = dict(file=str(path.relative_to(export)), sha256=sha(data))
            if relative == 'MANIFEST.txt':
                header, raw, start = unpack(path)
                records, census, footer = inventory(raw, start)
                cache[name] = (header, raw, start, records, footer)
                (output / (name + '.records.json')).write_text(json.dumps(records, indent=2) + '\n')
                expected = dict(instance=field(line, 'instance_sha256'), domain=field(line, 'domain'),
                                cap=int(field(line, 'cap')))
                code, verdict = run(path, output / name, expected)
                matches = (code == 0 and verdict.get('certificate_verified') is True
                           and verdict.get('status') == 'accepted'
                           and verdict.get('total_nodes') == int(field(line, 'nodes')) == footer['total_nodes']
                           and verdict.get('tasks_checked') == verdict.get('tasks_total')
                           == int(field(line, 'tasks')) == footer['tasks']
                           and verdict.get('global_trace_hex') == footer['global_trace_hex']
                           and verdict.get('gen_nodes') == footer['gen_nodes']
                           and verdict.get('task_nodes') == footer['task_nodes']
                           and verdict.get('rules') == field(line, 'rules')
                           and header['split'] == int(field(line, 'split'))
                           and header['detail'] == field(line, 'detail'))
                runtime_trace = re.search(r'\btrace=(\S+)', line)
                if runtime_trace:
                    result['expected_runtime_trace'] = runtime_trace.group(1)
                    matches = matches and verdict.get('global_trace_hex') == runtime_trace.group(1)
                result.update(expected=expected, footer=footer, census=census, verdict=verdict, matched=matches)
                report['accepts'].append(result)
            elif relative.startswith('invalid/'):
                defect = re.search(r'defect="(.*)" verdict=', line).group(1)
                code, verdict = run(path, output / ('invalid-' + name))
                matches = code == 1 and verdict.get('status') == 'rejected' and not verdict.get('certificate_verified')
                result.update(defect=defect, verdict=verdict, matched=matches)
                # Locate the first altered logical byte in the base record stream.
                # For transport-only corruption, report its first physical byte.
                base = field(line, 'base')
                _, base_raw, start, records, _ = cache[base]
                try:
                    raw = gzip.decompress(data)
                    first = next((i for i, (a, b) in enumerate(zip(raw, base_raw)) if a != b),
                                 min(len(raw), len(base_raw)))
                    result['logical_bytes_identical_to_base'] = raw == base_raw
                    if raw != base_raw:
                        result['first_changed_logical_offset'] = first
                        result['base_record_at_change'] = next((r for r in records if r['offset'] <= first < r['end_offset']), None)
                except (OSError, EOFError):
                    base_data = (export / base).read_bytes()
                    result['first_changed_transport_offset'] = next(
                        (i for i, (a, b) in enumerate(zip(data, base_data)) if a != b), min(len(data), len(base_data)))
                report['rejects'].append(result)
            else:
                pair = (path.parent / field(line, 'pairs_with')).resolve()
                if pair.name not in cache:
                    # The medium pair is handled by v2-check-medium.py; do not
                    # turn this small-set audit into a billion-node replay.
                    print('SKIP medium pair: ' + name, flush=True)
                    continue
                assert sha(pair.read_bytes()) == re.findall(r'\bsha256=(\S+)', line)[1]
                header, raw, start = unpack(path)
                other, paired_raw, paired_start = unpack(pair)
                record_bytes = raw[start:]
                identical = (record_bytes == paired_raw[paired_start:]
                             and sha(record_bytes) == field(line, 'record_stream_sha256')
                             and len(record_bytes) == int(field(line, 'record_stream_bytes')))
                # Only format identifiers, rule fields and informational provenance may differ.
                for key in ('format', 'search', 'trace', 'rules', 'rules_version', 'provenance'):
                    header.pop(key, None)
                    other.pop(key, None)
                code, verdict = run(path, output / ('v1-' + name))
                footer = cache[pair.name][4]
                matches = (identical and header == other and code == 0
                           and verdict.get('certificate_verified') is True
                           and verdict.get('global_trace_hex') == footer['global_trace_hex']
                           and verdict.get('total_nodes') == footer['total_nodes']
                           and verdict.get('tasks_checked') == verdict.get('tasks_total') == footer['tasks']
                           and verdict.get('checker_version') == '1.3.0'
                           and 'rules' not in verdict)
                runtime_trace = re.search(r'\btrace=(\S+)', line)
                if runtime_trace:
                    result['expected_runtime_trace'] = runtime_trace.group(1)
                    matches = matches and verdict.get('global_trace_hex') == runtime_trace.group(1)
                result.update(record_stream_identical=identical, record_stream_sha256=sha(record_bytes),
                              record_stream_bytes=len(record_bytes), verdict=verdict, matched=matches)
                report['v1_pairs'].append(result)
            if not matches:
                failures.append(result['file'])
            print(json.dumps(dict(file=result['file'], matched=matches, status=verdict.get('status'),
                                  nodes=verdict.get('total_nodes'), trace=verdict.get('global_trace_hex'),
                                  reasons=verdict.get('mismatches'))), flush=True)
            report['failures'] = failures
            (output / 'conformance.json').write_text(json.dumps(report, indent=2) + '\n')
    assert len(report['accepts']) == 21 and len(report['rejects']) == 24
    print(json.dumps(dict(accepts=len(report['accepts']), rejects=len(report['rejects']),
                          v1_pairs=len(report['v1_pairs']), failures=failures)), flush=True)
    return bool(failures)


if __name__ == '__main__':
    sys.exit(main())
