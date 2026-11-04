"""Exact distance of a non-CSS code with two independent proofs, then re-check the certificate.

    python examples/api_example.py examples/xzzx_surface_5.json
"""
import sys

import hyqec_verify as sc

path = sys.argv[1] if len(sys.argv) > 1 else "examples/xzzx_surface_5.json"
lc = sc.load(path)
code = lc.code
print(code, "CSS" if code.is_css else "non-CSS")

# lower bounds from the SAT/DRAT engine (brute_limit=0: skip the brute-force shortcut for small codes)
res = sc.distance(code, engines=("sat",), budget=300, ris_seconds=5, brute_limit=0)
print(res.status, "d =", res.upper if res.status == "exact" else (res.lower, res.upper))
for ev in res.evidence:
    print("  ", ev["method"], ev.get("proves", ev.get("weight")))

cert = sc.build_certificate(code, res, {"path": path})
sc.write_certificate(cert, "xzzx5-cert.json")   # copies the DRAT proofs next to it
res.cleanup()                                         # remove the temporary engine files
check = sc.verify_certificate("xzzx5-cert.json")
print("verify:", check["status"], check["verified_lower"], "<= d <=", check["verified_upper"])
