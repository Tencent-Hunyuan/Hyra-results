"""Deferred-decision stress: swap two PK entries of one S0/S1/S2 claim early in a top tree (the stream stays well
formed, so only the pooled decide+VERIFY can see it), run the native checker at several thread counts, and
report the verdict and how far reading got before the failure stopped it (stderr diagnostic).
Usage (from the checkout): python3 scripts/v2-far-ahead.py CERT FRACTION NATIVE_BIN THREADS...
Environment: CPUSET (taskset list)."""
import os, re, subprocess, sys, tempfile
from pathlib import Path
checkout = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(checkout)); sys.path.insert(0, str(checkout / 'scripts'))
from hyqec_check_v2.certificate import prepare_certificate, write_native_model
import v2_records

cert, frac, binary = Path(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
work = Path(tempfile.mkdtemp())
with prepare_certificate(cert, max_header=1 << 30, max_permutation_work=1 << 40) as (header, instance, stream):
    write_native_model(work / 'model', header, instance)
    chunks = []
    while True:
        c = stream.read(1 << 20)
        if not c: break
        chunks.append(c)
raw = b''.join(chunks)
recs, _, _ = v2_records.inventory(raw, 0)
top = [r for r in recs if r['tag'] == '10' and r['task'] is None]
r = next(x for x in top[int(len(top) * frac):] if x['rule'] in (5, 6, 7) and len(x['packing']) >= 2)


def leb(v):
    b = bytearray()
    while True:
        c = v & 127; v >>= 7
        if v: b.append(c | 128)
        else: b.append(c); return bytes(b)


p = list(r['packing']); p[0], p[1] = p[1], p[0]
body = (bytearray([0x10]) + leb(r['parent']) + leb(r['edge']) + bytes([r['rule']]) + r['fingerprint'].to_bytes(8, 'little')
        + leb(r['weight']) + leb(r['active']) + leb(len(p)) + b''.join(leb(g) for g in p))
(work / 'records').write_bytes(raw[:r['offset']] + bytes(body) + raw[r['end_offset']:])
print(f"tampered top ENTER event {r['event']} (rule {r['rule']}) of {len(top)} top entries")
for t in sys.argv[4:]:
    spool = work / 'spool'
    q = subprocess.run(['taskset', '-c', os.environ.get('CPUSET', '0'), binary, str(work / 'model'), str(work / 'records'),
                        '0', str(spool), t, 'all', '0', '0', '0', '0', '1'], capture_output=True, text=True)
    pos = re.findall(r'rejected at stream position.*', q.stderr)
    sanitizer = 'Sanitizer' in q.stderr or 'runtime error' in q.stderr
    print(f"threads {t}: {q.stdout.strip()[:120]} | {pos[0] if pos else '?'} | sanitizer report: {sanitizer}")
