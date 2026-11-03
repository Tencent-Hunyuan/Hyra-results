"""Every exported certificate (top level, invalid/, medium/, s2rich/) at several thread counts: identical verdict,
counts, traces and message, and agreement with the MANIFEST counts and traces.
Usage (from the checkout): python3 scripts/v2-thread-determinism.py ENGINE EXPORT_CERTS_DIR OUT.json [threads]"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hyqec_check_v2.verify import verify
from hyqec_check.algebra import ValidationError
engine, exp, outp = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
threads = [int(t) for t in (sys.argv[4] if len(sys.argv) > 4 else '1,2,8,16').split(',')]
keys = ('status', 'certificate_verified', 'tasks_checked', 'tasks_total', 'gen_nodes', 'task_nodes', 'total_nodes', 'global_trace_hex', 'mismatches')
rows = []; diffs = 0
manifests = [m for m in [('MANIFEST.txt', ''), ('invalid/MANIFEST.txt', 'invalid/'), ('medium/MANIFEST.txt', 'medium/'), ('s2rich/MANIFEST.txt', 's2rich/')] if (exp / m[0]).exists()]
for man, sub in manifests:
    for line in (exp / man).read_text().splitlines():
        if not line.strip() or line.startswith('#'): continue
        name = sub + line.split()[0]
        res = {}
        for t in threads:
            try:
                v = verify(exp / name, threads=t, engine=engine)
                res[t] = {k: v.get(k) for k in keys}
            except (ValidationError, ValueError, EOFError, OSError) as e:
                res[t] = {'python_reject': str(e)}
        same = all(res[t] == res[threads[0]] for t in threads)
        import re
        mn = re.search(r'\bnodes=(\d+)', line); mt = re.search(r'\btrace=(\w+)', line)
        manifest_ok = None
        if mn and mt and 'invalid' not in sub:
            manifest_ok = res[threads[0]].get('total_nodes') == int(mn.group(1)) and res[threads[0]].get('global_trace_hex') == mt.group(1) and res[threads[0]].get('certificate_verified') is True
        elif 'invalid' in sub:
            manifest_ok = all(r.get('status') == 'rejected' or 'python_reject' in r for r in res.values())
        diffs += not same
        rows.append(dict(cert=name, identical=same, manifest_ok=manifest_ok, result=res[threads[0]], **({} if same else {'all': res})))
        print(json.dumps(dict(cert=name, identical=same, manifest_ok=manifest_ok, status=res[threads[0]].get('status'), nodes=res[threads[0]].get('total_nodes'), trace=res[threads[0]].get('global_trace_hex'))), flush=True)
bad = sum(r['manifest_ok'] is False for r in rows)
json.dump(dict(engine=engine, threads=threads, certs=len(rows), differences=diffs, manifest_failures=bad, rows=rows), open(outp, 'w'), indent=1)
print(json.dumps(dict(certs=len(rows), differences=diffs, manifest_failures=bad)))
