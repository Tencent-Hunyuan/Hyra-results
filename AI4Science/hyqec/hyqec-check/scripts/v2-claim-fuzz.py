"""Claim-tampering stratum for the deferred top-tree / FULL-transcript decisions.

For each exported certificate: extract the validated native model and record stream, then rewrite single
ENTER records (rule byte, branch generator, packing entries, rule swaps with consistent re-encoding), deep in
the top tree and inside FULL transcripts. Every mutant must be rejected by the native checker with the same
message at every thread count, and the sanitizer build must stay clean.

Usage (from the checkout): python3 scripts/v2-claim-fuzz.py EXPORT_CERTS_DIR RELEASE_BIN SANITIZE_BIN OUTDIR
                           [mutants_per_cert] [threads,...]
Environment: CPUSET (taskset list), CERTS (comma list relative to EXPORT_CERTS_DIR; large certificates are tampered
early in the top tree or inside early FULL transcripts, so the walk runs far ahead of the deferred checks).
"""
import sys, os, json, random, subprocess, tempfile, shutil
from pathlib import Path

export_dir, release, sanitize, outdir = sys.argv[1:5]
per = int(sys.argv[5]) if len(sys.argv) > 5 else 40
threads = [int(t) for t in (sys.argv[6] if len(sys.argv) > 6 else '1,2,8,16').split(',')]
cpus = os.environ.get('CPUSET', '0')
checkout = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(checkout)); sys.path.insert(0, str(checkout / 'scripts'))
from hyqec_check_v2.certificate import prepare_certificate, write_native_model
import v2_records as inv
export = Path(export_dir)
out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
rng = random.Random(176)


def leb(v):
    b = bytearray()
    while True:
        c = v & 127; v >>= 7
        if v: b.append(c | 128)
        else: b.append(c); return bytes(b)


def enter_bytes(r):
    b = bytearray([0x10]) + leb(r['parent']) + leb(r['edge']) + bytes([r['rule']])
    b += r['fingerprint'].to_bytes(8, 'little') + leb(r['weight']) + leb(r['active'])
    if r['rule'] == 2 or r['rule'] in (5, 6, 7):
        b += leb(len(r['packing'])) + b''.join(leb(g) for g in r['packing'])
    elif r['rule'] == 3:
        b += leb(r['generator'])
    return bytes(b)


def mutate(r, m, rules):
    r = dict(r); kinds = []
    prunes = [5] + ([6] if rules in ('s0s1', 's0s1s2') else []) + ([7] if rules == 's0s1s2' else [])
    if r['rule'] == 3:
        kinds += ['gen-other', 'gen-range', 'branch-to-prune', 'branch-to-weight']
    if r['rule'] in (2, 5, 6, 7):
        kinds += ['pk-entry', 'pk-drop', 'prune-to-branch', 'prune-swap']
    if r['rule'] in (0, 1):
        kinds += ['terminal-to-branch']
    kinds += ['rule-byte']
    k = rng.choice(kinds)
    if k == 'gen-other': r['generator'] = (r['generator'] + rng.randrange(1, m)) % m
    elif k == 'gen-range': r['generator'] = m + rng.randrange(0, 3)
    elif k == 'branch-to-prune':
        r['rule'] = rng.choice(prunes + [2]); r['packing'] = [rng.randrange(m) for _ in range(rng.randrange(1, 6))]
    elif k == 'branch-to-weight': r['rule'] = 1
    elif k == 'pk-entry':
        p = list(r['packing']); p[rng.randrange(len(p))] = rng.randrange(m); r['packing'] = p
    elif k == 'pk-drop': r['packing'] = r['packing'][:-1]
    elif k == 'prune-to-branch': r['rule'] = 3; r['generator'] = rng.randrange(m)
    elif k == 'prune-swap': r['rule'] = rng.choice([x for x in prunes + [2] if x != r['rule']] or [2])
    elif k == 'terminal-to-branch': r['rule'] = 3; r['generator'] = rng.randrange(m)
    elif k == 'rule-byte':
        return k, None, rng.choice([x for x in range(9) if x != r['rule']])
    return k, r, None


def run(binary, model, rec, t):
    spool = tempfile.mktemp()
    env = dict(os.environ, ASAN_OPTIONS='detect_leaks=1', UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
    p = subprocess.run(['taskset', '-c', cpus, binary, str(model), str(rec), '0', spool, str(t), 'all', '0', '0', '0', '0', '1'],
                       capture_output=True, text=True, env=env, timeout=1800)
    if os.path.exists(spool): os.remove(spool)
    try: v = json.loads(p.stdout)
    except json.JSONDecodeError: v = {'status': 'no-verdict'}
    bad = 'Sanitizer' in p.stderr or 'runtime error' in p.stderr
    return v, p.returncode, bad, p.stderr[-2000:] if bad else ''


summary = dict(mutants=0, rejected_all=0, not_rejected=[], sanitizer=[], message_differences=[], kinds={})
names = os.environ.get('CERTS')
lines = names.split(',') if names else [l for l in (export / 'MANIFEST.txt').read_text().splitlines()
                                         if l.strip() and not l.startswith('#')]
for line in lines:
    name = line.split()[0]
    work = Path(tempfile.mkdtemp())
    with prepare_certificate(export / name, max_header=1 << 30, max_permutation_work=1 << 40) as (header, instance, stream):
        write_native_model(work / 'm.model', header, instance)
        chunks = []
        while True:
            c = stream.read(1 << 20)
            if not c: break
            chunks.append(c)
        raw = b''.join(chunks)
    m = int(header['m']); rules = header['rules']
    records, _, _ = inv.inventory(raw, 0)
    enters = [r for r in records if r['tag'] == '10']
    if not enters: shutil.rmtree(work); continue
    # bias towards deep records: the later half of the stream and FULL transcripts
    if names:  # large certificates: early, so the walker runs far ahead; FULL: inside early transcripts
        inside = [r for r in enters if r['task'] is not None]
        pool = inside[len(inside) // 100: len(inside) // 5] if inside else enters[len(enters) // 50: len(enters) // 5]
    else:
        pool = enters[len(enters) // 2:] + [r for r in enters if r['task'] is not None]
    for k in range(per):
        r = rng.choice(pool)
        kind, new, byte = mutate(r, m, rules)
        a, b = r['offset'], r['end_offset']
        if new is not None:
            data = raw[:a] + enter_bytes(new) + raw[b:]
        else:
            rb = a + 1 + len(leb(r['parent'])) + len(leb(r['edge']))
            data = raw[:rb] + bytes([byte]) + raw[rb + 1:]
        if data == raw: continue
        (work / 'mut.rec').write_bytes(data)
        summary['mutants'] += 1; summary['kinds'][kind] = summary['kinds'].get(kind, 0) + 1
        verdicts = {}
        for t in threads:
            v, rc, bad, err = run(release, work / 'm.model', work / 'mut.rec', t)
            verdicts[f'release@{t}'] = (v.get('status'), (v.get('mismatches') or [''])[0])
        v, rc, bad, err = run(sanitize, work / 'm.model', work / 'mut.rec', max(threads))
        verdicts[f'sanitize@{max(threads)}'] = (v.get('status'), (v.get('mismatches') or [''])[0])
        if bad: summary['sanitizer'].append(dict(cert=name, kind=kind, offset=a, err=err))
        statuses = {s for s, _ in verdicts.values()}
        if statuses == {'rejected'}: summary['rejected_all'] += 1
        else: summary['not_rejected'].append(dict(cert=name, kind=kind, offset=a, verdicts=verdicts))
        if len({msg for _, msg in verdicts.values()}) > 1:
            summary['message_differences'].append(dict(cert=name, kind=kind, offset=a, verdicts=verdicts))
    shutil.rmtree(work)
    print(name, json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in summary.items()}), flush=True)
(out / 'claim_fuzz.json').write_text(json.dumps(summary, indent=1))
print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in summary.items()}))
