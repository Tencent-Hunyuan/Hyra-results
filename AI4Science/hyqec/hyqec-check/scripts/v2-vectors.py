"""Run the exported conformance vectors through the native node driver.

  PYTHONPATH=tests:. python3 scripts/v2-vectors.py VECTORS_DIR [DRIVER]
  (NODES_ONLY=1 skips verify.jsonl, for drivers without the --verify-j mode.)

verify.jsonl: each (node, J) goes to VERIFY through the driver's --verify-j mode and must give the expected verdict.
nodes.jsonl.gz: each node's decision, np, PK, branch generator, the dispatched rule's recorded intermediates (s0, s1,
s2, except S2's informative survivor) and the limits crossed must equal the expected values.
"""
import collections, gzip, json, os, subprocess, sys, tempfile
from pathlib import Path

vectors = Path(sys.argv[1])
driver = sys.argv[2] if len(sys.argv) > 2 else 'build/v2-node-driver'
DECISION = {2: 'PACKING', 3: 'BRANCH', 5: 'S0', 6: 'S1', 7: 'S2'}
STEP = {'ELIM': 0, 'END_WIPED': 1, 'END_ODD': 2}
REASON = {'PACKED': 0, 'KILL': 1, 'ZGEN': 2, 'ODD': 3, 'DER': 4}


def model_file(directory, n, rows, rules, cap):
    path = Path(directory) / 'model'
    lines = [f'{n} {len(rows)} 0 {cap} 1 P frontier {rules}']
    lines += [' '.join([str(len(r) // 2)] + [str(x) for x in r]) for r in rows]
    lines += [' '.join(map(str, range(n))), str(n) + ' ' + ' '.join(map(str, range(n))), '']
    path.write_text('\n'.join(lines))
    return path


def steps(d):
    out = [len(d)]
    for st in d:
        if st[0] == 'END_WIPED': out += [STEP[st[0]], -1, st[1]]
        elif st[0] == 'END_ODD': out += [STEP[st[0]], st[1], -1]
        else: out += [STEP[st[0]], st[1], st[2]]
    return out


def lst(xs): return [len(xs)] + list(xs)


def j_tokens(v):
    j = v['J']; t = [int(c) for c in v['lab']] + [v['rule']] + lst(j['PK']) + steps(j['J2'])
    if j.get('form') == 'S1F':
        t += [1] + lst(j['I'])
        for k in j['K']: t += lst(k)
        t += lst(j['ZL'])
    else:
        a = j['anchor']
        t += [0, 1 if a[0] == 'Z' else 0, a[1]] + lst(a[2] if a[0] == 'WIPED' else [])
        t += [len(j['reasons'])]
        for (q, l), r in j['reasons']:
            kind = REASON[r[0]]
            if r[0] == 'KILL': t += [q, l, kind, r[1]] + lst(r[2]) + [0]
            elif r[0] == 'DER': t += [q, l, kind, -1, 0] + steps(r[1])
            else: t += [q, l, kind, r[1], 0, 0]
    return ' '.join(map(str, t))


def run(model, lines, mode=None):
    """One driver run; it must exit 0 and answer every input line with exactly one JSON line."""
    cmd = [driver, str(model)] + ([mode] if mode else [])
    p = subprocess.run(cmd, input='\n'.join(lines) + '\n', capture_output=True, text=True, timeout=7200)
    if p.returncode != 0:
        raise SystemExit(f'driver failed with exit {p.returncode}: {p.stderr[-500:]}')
    out = [json.loads(x) for x in p.stdout.splitlines()]
    if len(out) != len(lines):
        raise SystemExit(f'driver answered {len(out)} of {len(lines)} inputs')
    return out


report = dict(verify=[], verify_failures=0, nodes=0, node_failures=0, failures=[], traces=collections.Counter())
STEPS = ['6.1', '6.2', '6.3', '6.4', '6.5', '6.6']
with tempfile.TemporaryDirectory() as tmp:
    for line in ([] if os.environ.get('NODES_ONLY') else (vectors / 'verify.jsonl').read_text().splitlines()):
        v = json.loads(line)
        model = model_file(tmp, v['code_rows']['n'], v['code_rows']['rows'], v['rules'], v['cap'])
        got = run(model, [j_tokens(v)], '--verify-j')[0]['verify']
        ok = got == v['expect']
        report['verify'].append(dict(id=v['id'], expect=v['expect'], got=got, ok=ok))
        report['verify_failures'] += not ok
        print(json.dumps(report['verify'][-1]), flush=True)
    codes = {}
    with gzip.open(vectors / 'codes.jsonl.gz', 'rt') as f:
        for line in f:
            c = json.loads(line); codes[c['code']] = c
    groups = collections.defaultdict(list)
    with gzip.open(vectors / 'nodes.jsonl.gz', 'rt') as f:
        for line in f:
            v = json.loads(line); groups[(v['code'], v['rules'], v['cap'])].append(v)
    for (code, rules, cap), vs in groups.items():
        c = codes[code]
        model = model_file(tmp, c['n'], c['rows'], rules, cap)
        outs = run(model, [' '.join(v['lab']) for v in vs])
        for v, out in zip(vs, outs):
            e = v['expect']; bad = []
            if 'error' in out: bad.append(('error', out['error']))
            else:
                if DECISION.get(out['rule']) != e['decision']: bad.append(('decision', out['rule'], e['decision']))
                if e.get('np') is not None and out.get('np') != e['np']: bad.append(('np', out.get('np'), e['np']))
                if e.get('PK') is not None and out.get('PK') != e['PK']: bad.append(('PK', out.get('PK'), e['PK']))
                if e['decision'] == 'BRANCH' and out['generator'] != e['branch_generator']:
                    bad.append(('branch_generator', out['generator'], e['branch_generator']))
                if e['decision'] in ('PACKING', 'S0', 'S1', 'S2') and out['packing'] != e['PK']:
                    bad.append(('record PK', out['packing'], e['PK']))
                if v.get('completion_exists') and out['rule'] != 3: bad.append(('pruned a node with a completion',))
                # Every recorded intermediate of the dispatched rule (5.2), and the limits crossed.
                for rule in ('s0', 's1', 's2'):
                    want = dict(e[rule]) if rule in e else None
                    if want is not None:
                        want.pop('survivor_informative', None)  # informative only (10.1: any valid E)
                        if 'tests' in want: want['tests'] = [want['tests'][k] for k in STEPS]
                        report['traces'][rule] += 1
                    if out.get(rule) != want: bad.append((rule, out.get(rule), want))
                if out.get('limits_crossed') != sorted(e['limits_crossed']):
                    bad.append(('limits_crossed', out.get('limits_crossed'), e['limits_crossed']))
            report['nodes'] += 1
            if bad:
                report['node_failures'] += 1
                if len(report['failures']) < 50: report['failures'].append(dict(id=v['id'], bad=bad))
        print(json.dumps(dict(code=code, rules=rules, cap=cap, nodes=len(vs), failures_so_far=report['node_failures'])), flush=True)
expected_verify = len((vectors / 'verify.jsonl').read_text().splitlines()) if not os.environ.get('NODES_ONLY') else 0
with gzip.open(vectors / 'nodes.jsonl.gz', 'rt') as f:
    expected_nodes = sum(1 for line in f if line.strip())
if len(report['verify']) != expected_verify or report['nodes'] != expected_nodes:
    report['node_failures'] += 1
    report['failures'].append(dict(id='COUNT', bad=[('verify', len(report['verify']), expected_verify), ('nodes', report['nodes'], expected_nodes)]))
print(json.dumps({k: report[k] for k in ('verify_failures', 'nodes', 'node_failures', 'traces')}))
print(json.dumps(report['failures'][:10]))
sys.exit(1 if report['verify_failures'] or report['node_failures'] else 0)
