# Validation scope

The coverage-v2 implementation was checked against specification `2.0-rc1`
(SHA256 `408806c15fbbfc8597fec20f4c0a0861846dc56cec027d214b984de3c59f43ee`).
Independent specification and code review found no blocker or should-fix issue
through native revision `93deceb`. An implementation review also confirmed the
canonical S1 action order, strict vector runner and independent VERIFY checks.

## Completed conformance

- All 56 exported artifacts matched: 29 v2 accepts, 24 expected rejects and
  three v1 pairs. Accepted artifacts matched exact node counts, task counts
  and traces; all paired record streams matched after their headers.
- All 42,250 node vectors matched every canonical field, including 6,899 S1
  evaluations. All seven explicit VERIFY cases, V000–V006, matched.
- All 888 complete searches matched their expected node totals:
  2,396,942,083 nodes. These vectors did not supply expected search traces;
  agreement on complete-search traces is not claimed.
- The full 64-test suite, 14 GCC UBSan tests and 14 Clang ASan+UBSan tests
  passed, including deliberately incorrect justifications and forced S2
  fingerprint collisions. Strict-runner invalid-output controls rejected.
- 24,000 exhaustive-completion probes found no unsound prune; another 24,000
  baseline comparisons found no decision, packing or UNIT mismatch.

Version 2.0.0 release gates additionally installed both the wheel and source
package in fresh environments. Each passed 64 installed tests, all 56 exported
artifacts and 244 v1 profile checks (72 narrow, 172 wide). All 30 frozen source
and binary hashes and 19 frozen v1 package payload hashes matched.

## Documentation releases 2.0.1 and 2.0.2

Version 2.0.1 changes copyright, documentation, source-distribution selection
and version metadata. Native source and all four native executables are
unchanged. The native v2 executable SHA256 is
`436de764c93cc4778650a1a99f09ff1ca3ee7b8a759a73b7811314f85fec31ac`.
The Python v2 profile differs only in its version constant; the existing CLI
version test expectation is updated accordingly.

Version 2.0.2 moves internal conformance and process reports out of distributed
contents and replaces remaining development branch names with neutral descriptions.
The [current vector instructions](V2.md#run-current-node-and-verify-vectors) work
from plain source exports and replace an unavailable historical build target.
Historical source records remain separate from this public results summary.
The [v1 specification](BNB_CERT_SPEC.md) and its
[arithmetic/example vectors](BNB_CERT_VECTORS.json) are included unchanged;
the specification's original validation-status prose describes its revision date,
while the completed release results are reported above.

The short release gates rebuild both package formats, run all 64 installed
tests for each, compare native bytes, verify frozen hashes and scan the shipped
contents. Full certificate, vector, search and sanitizer results above carry
forward under unchanged implementation and executable bytes.

Detailed historical evidence is retained in the development repository and
is not included in publication artifacts. Earlier baseline measurements in
other documentation describe their stated revisions; they do not supersede
the current conformance results summarized here. Timing comparisons use shared
CPU resources and do not claim an otherwise idle host.
