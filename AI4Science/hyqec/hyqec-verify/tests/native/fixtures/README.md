# Legacy checkpoint rejection fixture

`legacy-resume-five.inst` and `legacy-resume-cap2.ckpt` are byte-for-byte copies
of an independent referee's five-qubit resume counterexample (2026-10-22). The old
`pauli_cert_cpu` completed cap 2 in 46 nodes, then accepted the headerless checkpoint
at cap 3 and returned the same 46-node no-find result, despite the code having
distance 3; the invalid cap-4 resume was confirmed as well. These files contain
input and search data, not a dependency on the old executable.

| Fixture | SHA-256 |
|---|---|
| `legacy-resume-five.inst` | `c5f93c2e8a224176c1817aed29864138deabb8a5cd1fd5e5e569f83bd2401bc1` |
| `legacy-resume-cap2.ckpt` | `a8ccd70c79a24ec019bcdfe6329076628cdc1e9e9cec08901d669340533bafd1` |

The tests require rejection with exit 6 and byte-preservation of the
checkpoint, both for legacy files and for cap escalation on a new native
checkpoint. A fresh search at cap 3 or 4 must find distance 3.
