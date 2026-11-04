# Verification scope and format compatibility

Full acceptance means that the identified stabilizer code has no nontrivial
logical Pauli of weight at most the immutable certificate cap in its stated
P, X or Z domain. It establishes a lower bound of cap + 1 when that domain
contains a logical operator. It does not alone establish exact distance.

The checker independently validates generator commutation, stabilizer rank,
logical quotient span, every supplied automorphism and the exact orbit
partition. It reconstructs the top traversal, every canonical task path and
all selected subtree searches. Full acceptance additionally requires every
task, all record fields, counts, traces, footer and EOF to pass. Encountering a
nontrivial logical leaf rejects the claim, regardless of any supplied trace.
The trace is noncryptographic and is not a substitute for independent search.

Partial sample/range checks do not establish the full lower bound, even when
their selected set happens to include every task. Expected group/instance,
domain and cap options bind the checked certificate to the caller's question.
See README and the written BNB_CERT_SPEC.md revision 1.1 for the supported
limits and search conventions; historical implementation review notes are
under docs/history.

**File formats keep their historical `stabcert-*` ids.** The 1.3.0 product rename
does not alter `stabcert-group/1`, `stabcert-bnb-instance 1`,
`stabcert-bnb-coverage/1`, `SBCOV1`, search/trace identifiers, or the reference
fixture ID `bnbcheck-reference-code/0`. These names participate in identities
and parsing, not in Python package discovery. Existing certificate bytes and
hashes remain valid after the rename. Native verifier sources are unchanged
from 1.2.0; the rename introduces no change to the mathematical checking rules.
