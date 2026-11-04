"""Audit all data-only node, VERIFY and search vectors with the baseline checker.

Run from the checkout with PYTHONPATH=. and taskset. The private C++ adapter
includes production code; it is not a new public certificate format or verifier.
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from hyqec_check_v2.certificate import write_native_model
from hyqec_check_wide.certificate import parse_instance


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def rows(path):
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def raw_model(path, code):
    n, data = code['n'], code['rows']
    lines = [f'{n} {len(data)} 0 46 1 P frontier s0s1s2']
    lines += [str(len(row) // 2) + ''.join(f' {x}' for x in row) for row in data]
    lines += [' '.join(map(str, range(n))), str(n) + ' ' + ' '.join(map(str, range(n)))]
    path.write_text('\n'.join(lines) + '\n')


def compare(expected, actual, prefix=''):
    failures = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [dict(field=prefix, expected=expected, actual=actual)]
        for key, value in expected.items():
            if key == 'survivor_informative':
                continue  # S2 witnesses need not be the same: spec 10.1, vector README.
            failures += compare(value, actual.get(key), prefix + '.' + key if prefix else key)
    elif prefix == 'limits_crossed':
        if not isinstance(actual, list) or sorted(expected) != sorted(actual):
            failures.append(dict(field=prefix, expected=expected, actual=actual))
    elif expected != actual:
        failures.append(dict(field=prefix, expected=expected, actual=actual))
    return failures


def integer_list(values):
    return [len(values), *values]


def derivation(steps):
    out = [len(steps)]
    for step in steps:
        if step[0] == 'ELIM': out += [0, step[1], step[2]]
        elif step[0] == 'END_WIPED': out += [1, -1, step[1]]
        elif step[0] == 'END_ODD': out += [2, step[1], -1]
        else: raise ValueError(step)
    return out


def justification(case):
    j = case['J']
    out = [case['rule'], *integer_list(j['PK']), *derivation(j['J2'])]
    anchor = j['anchor']
    out += [int(anchor[0] == 'Z'), anchor[1], *integer_list(anchor[2] if anchor[0] == 'WIPED' else [])]
    out += [len(j['reasons'])]
    kinds = {'PACKED': 0, 'KILL': 1, 'ZGEN': 2, 'ODD': 3, 'DER': 4}
    for action, reason in j['reasons']:
        kind = reason[0]
        out += [*action, kinds[kind], reason[1] if kind != 'DER' else -1]
        out += integer_list(reason[2] if kind == 'KILL' else [])
        out += derivation(reason[1] if kind == 'DER' else [])
    return ' '.join(map(str, out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export_dir', type=Path)
    parser.add_argument('result_dir', type=Path)
    parser.add_argument('--phase', choices=['all', 'nodes', 'verify', 'searches'], default='all')
    parser.add_argument('--threads', type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    export, out = args.export_dir.resolve(), args.result_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary = out / 'conformance.json'
    assert not summary.exists(), 'use a fresh result directory'
    affinity = sorted(os.sched_getaffinity(0))
    assert set(affinity) <= {180, 181, 372, 373}, affinity
    driver = Path('build/v2-vector-driver').resolve()
    manifest = (export / 'SHA256SUMS').read_bytes()
    (out / 'SHA256SUMS').write_bytes(manifest)
    (out / 'README.md').write_bytes((export / 'README.md').read_bytes())
    hashes = {}
    for line in manifest.decode().splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip('*')
        assert sha(export / name) == expected, name
        hashes[name] = expected
    fixed = json.loads(Path('docs/V1_FROZEN_SHA256.json').read_text())
    assert all(sha(Path(name)) == expected for name, expected in fixed.items())
    report = dict(started_utc=utc(), checker_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  implementation_revision='24ad05e', driver_sha256=sha(driver),
                  driver_source_sha256=sha(Path('tests/v2_vector_driver.cpp')),
                  runner_sha256=sha(Path(__file__)),
                  production_native_sha256=sha(Path('build/hyqec-check-v2-native-v2.0.0')),
                  export=str(export), input_sha256=hashes, manifest_sha256=hashlib.sha256(manifest).hexdigest(),
                  affinity=affinity, threads=args.threads, phases={}, failures=[], frozen_v1_before=len(fixed),
                  noncanonical_field_not_compared='s2.survivor_informative',
                  diagnostic_method='test-only reconstruction from checker objects; actual decisions and J verification use production code')
    models = out / 'models'
    models.mkdir()

    def save():
        summary.write_text(json.dumps(report, indent=2) + '\n')

    save()
    if args.phase in ('all', 'nodes'):
        started = time.perf_counter()
        codes = {c['code']: c for c in rows(export / 'codes.jsonl.gz')}
        grouped = collections.defaultdict(list)
        for case in rows(export / 'nodes.jsonl.gz'): grouped[case['code']].append(case)
        phase = dict(cases=0, matched=0, failures=0, sets=collections.Counter(), decisions=collections.Counter())
        report['phases']['nodes'] = phase
        with (out / 'nodes-results.jsonl').open('w') as log:
            for code, cases in grouped.items():
                model = models / (code + '.model')
                raw_model(model, codes[code])
                command = [str(driver), str(model), 'node']
                payload = ''.join(f"{c['cap']} {c['rules']} {c['lab']}\n" for c in cases)
                process = subprocess.run(command, input=payload, capture_output=True, text=True)
                actual = [json.loads(line) for line in process.stdout.splitlines()]
                assert process.returncode == 0 and len(actual) == len(cases), (code, process.returncode, process.stderr, len(actual), len(cases))
                for case, result in zip(cases, actual):
                    differences = compare(case['expect'], result)
                    if case.get('completion_exists') and result['decision'] != 'BRANCH':
                        differences.append(dict(field='completion_exists', expected='BRANCH', actual=result['decision']))
                    item = dict(id=case['id'], code=code, set=case['set'], expected=case['expect'], actual=result,
                                matched=not differences, differences=differences)
                    log.write(json.dumps(item) + '\n')
                    phase['cases'] += 1;phase['matched'] += not differences;phase['failures'] += bool(differences)
                    phase['sets'][case['set']] += 1;phase['decisions'][result['decision']] += 1
                    if differences: report['failures'].append(dict(phase='nodes', id=case['id'], differences=differences))
                if phase['cases'] % 1000 < len(cases):
                    print('NODES ' + json.dumps(phase), flush=True);save()
        phase['wall_seconds'] = time.perf_counter() - started
        save();print('NODES COMPLETE ' + json.dumps(phase), flush=True)
    if args.phase in ('all', 'verify'):
        phase = dict(cases=0, matched=0, results=[])
        report['phases']['verify'] = phase
        for case in rows(export / 'verify.jsonl'):
            model = models / (case['id'] + '.model')
            raw_model(model, case['code_rows'])
            payload = f"{case['cap']} {case['rules']} {case['lab']} " + justification(case) + '\n'
            command = [str(driver), str(model), 'verify']
            process = subprocess.run(command, input=payload, capture_output=True, text=True)
            assert process.returncode == 0, process.stderr
            result = json.loads(process.stdout)
            matched = result['verdict'] == case['expect']
            item = dict(id=case['id'], expected=case['expect'], actual=result, why=case['why'], matched=matched,
                        command=command, bridge_input=payload)
            phase['results'].append(item);phase['cases'] += 1;phase['matched'] += matched
            if not matched: report['failures'].append(dict(phase='verify', **item))
        save();print('VERIFY COMPLETE ' + json.dumps(phase), flush=True)
    if args.phase in ('all', 'searches'):
        cases = rows(export / 'searches.jsonl')
        phase = dict(cases=0, matched=0, expected_cases=len(cases), expected_total_nodes=sum(c['total_nodes'] for c in cases),
                     matched_total_nodes=0, results=[])
        report['phases']['searches'] = phase
        cache = {}
        for index, case in enumerate(cases):
            instance_path = export / case['instance']
            if case['instance'] not in cache:
                instance = parse_instance(instance_path.read_text())
                assert instance.instance_sha256 == case['instance_sha256']
                # Singleton orbits may appear in any supplied order; that order
                # controls roots and must be preserved (spec sections 2 and 3).
                orbits = [[q for q, oi in enumerate(instance.orbit_indices) if oi == i]
                          for i in range(len(instance.representatives))]
                instance.code.validate_orbit_order([], orbits, instance.representatives)
                cache[case['instance']] = instance
            instance = cache[case['instance']]
            assert instance.instance_sha256 == case['instance_sha256']
            model = models / f'search-{index:04d}.model'
            write_native_model(model, dict(cap=case['cap'], rules=case['rules'], split=1, detail='frontier'), instance)
            command = [str(driver), str(model), 'search', str(args.threads)]
            print('SEARCH START ' + json.dumps(dict(index=index, **case)), flush=True)
            started_utc = utc();started = time.perf_counter()
            process = subprocess.run(command, capture_output=True, text=True)
            elapsed = time.perf_counter() - started
            (out / f'search-{index:04d}.stdout.json').write_text(process.stdout)
            (out / f'search-{index:04d}.stderr.log').write_text(process.stderr)
            result = json.loads(process.stdout) if process.stdout.strip() else {}
            matched = process.returncode == 0 and result.get('total_nodes') == case['total_nodes'] and result.get('logical_found') is False
            item = dict(index=index, expected=case, actual=result, matched=matched, command=command,
                        process_wall_seconds=elapsed, started_utc=started_utc, finished_utc=utc(), returncode=process.returncode)
            phase['results'].append(item);phase['cases'] += 1;phase['matched'] += matched
            if matched: phase['matched_total_nodes'] += case['total_nodes']
            else: report['failures'].append(dict(phase='searches', **item))
            save();print('SEARCH FINISH ' + json.dumps(item), flush=True)
    assert all(sha(Path(name)) == expected for name, expected in fixed.items())
    assert sha(Path('build/hyqec-check-v2-native-v2.0.0')) == report['production_native_sha256']
    report.update(finished_utc=utc(), all_matched=not report['failures'], frozen_v1_after=len(fixed))
    save();print('FINISHED ' + json.dumps(dict(all_matched=report['all_matched'], failures=len(report['failures']))), flush=True)
    return bool(report['failures'])


if __name__ == '__main__':
    raise SystemExit(main())
