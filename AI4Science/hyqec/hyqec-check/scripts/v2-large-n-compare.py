"""Large-n stratum: base (24ad05e) vs new node driver on padded hypergraph-product codes.
Crosses n = 1365/1366 (S2), 2048, 4096 (max n); m up to ~2000; caps up to 46 so h reaches 31/32 when packings allow.
Usage: PYTHONPATH=tests:. python3 scripts/v2-large-n-compare.py BASE_DRIVER NEW_DRIVER QUERIES_PER_CASE
This is differential regression evidence across the large-n limits, not an independent oracle."""
import sys, os, random, json, collections, hashlib
from v2_factory import NodeDriver

def hgp_rows(n1, m1, w, rng, n_total):
    H = [sorted(rng.sample(range(n1), w)) for _ in range(m1)]
    nq = n1 * n1 + m1 * m1
    assert nq <= n_total
    A = lambda i, j: i * n1 + j          # left block n1 x n1
    B = lambda a, b: n1 * n1 + a * m1 + b  # right block m1 x m1
    rows = []
    for a in range(m1):           # X checks: (a, j)
        for j in range(n1):
            r = ['I'] * n_total
            for i in H[a]: r[A(i, j)] = 'X'
            for b in range(m1):
                if j in H[b]: r[B(a, b)] = 'X'
            rows.append(''.join(r))
    for i in range(n1):           # Z checks: (i, b)
        for b in range(m1):
            r = ['I'] * n_total
            for j in H[b]: r[A(i, j)] = 'Z'
            for a in range(m1):
                if i in H[a]: r[B(a, b)] = 'Z'
            rows.append(''.join(r))
    return rows

base, new, per = sys.argv[1], sys.argv[2], int(sys.argv[3])
rng = random.Random(176)
cases = [(1365, 24, 18, 3), (1366, 24, 18, 3), (2048, 30, 22, 3), (4095, 40, 30, 4), (4096, 24, 16, 4)]
stats = collections.Counter(); digest = hashlib.sha256(); mism = []
for n, n1, m1, w in cases:
    rows = hgp_rows(n1, m1, w, rng, n)
    for rules in ('s0s1s2', 's0s1'):
        for cap in (8, 12, 16, 24):
            drivers = []
            for exe in (base, new):
                os.environ['HYQEC_CHECK_V2_NODE_ENGINE'] = exe
                drivers.append(NodeDriver.from_rows(n, rows, rules, cap))
            try:
                done = 0
                while done < per:
                    # random root-to-terminal descent through the real child enumeration (spec 3)
                    lab = [4] * n; lab[rng.randrange(n)] = rng.randrange(1, 4)
                    while done < per:
                        out = [d.query(lab) for d in drivers]
                        done += 1
                        key = 'n%d-%s-c%d-rule%s' % (n, rules, cap, out[1].get('rule', ('err:' + out[1]['error'][:20]) if 'error' in out[1] else 'dead'))
                        stats[key] += 1
                        if out[0] != out[1]:
                            stats['MISMATCH'] += 1
                            if len(mism) < 5: mism.append((n, rules, cap, out))
                        digest.update(json.dumps(out[1], sort_keys=True).encode())
                        if out[1].get('rule') != 3: break
                        g = rows[out[1]['generator']]
                        cands = [q for q in range(n) if g[q] != 'I' and (lab[q] == 4 or (lab[q] > 4 and lab[q] - 4 != 'IXZY'.index(g[q])))]
                        k = rng.randrange(len(cands))
                        for q in cands[:k]:
                            gq = 'IXZY'.index(g[q]); lab[q] = 4 + gq if lab[q] == 4 else 0
                        q = cands[k]; gq = 'IXZY'.index(g[q])
                        lab[q] = rng.choice([l for l in (1, 2, 3) if l != gq]) if lab[q] == 4 else lab[q] - 4
            finally:
                for d in drivers: d.close()
            print(n, rules, cap, dict((k, v) for k, v in stats.items() if k.startswith('n%d-%s-c%d-' % (n, rules, cap))), flush=True)
print(json.dumps(dict(mismatches=stats['MISMATCH'], total=sum(v for k, v in stats.items() if k != 'MISMATCH'), sha256=digest.hexdigest())))
for m in mism: print(str(m)[:1500])
