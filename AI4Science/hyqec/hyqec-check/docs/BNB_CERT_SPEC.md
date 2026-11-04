# B&B coverage certificate, container version 1

Specification revision 1.1 (supported-limit expansion), 2026-10-22. The
CPU/CUDA exporter and independent checker have passed the original-limit
smoke artifacts and full 452 cap17 replay. Revision 1.1 increases supported
generator weight and column degree without changing the search rules, trace,
container or record layout. Independent wide-limit validation is pending.
The independent checker must be implemented from this specification, without
reading the prover's search implementation. Ambiguities are specification bugs;
report them before substituting an implementation-dependent interpretation.

A successful check establishes that no nontrivial logical Pauli of weight at
most `cap` exists within `domain`. The certificate explicitly covers the upper
search tree and names every frontier subtree. The checker independently
executes every frontier subtree, proves it contains no logical witness, and
compares its count and trace. A count or trace alone is **not a proof**.

The trace is deliberately noncryptographic. Even a malicious collision cannot
justify accepting a subtree in which the checker finds a logical operator.
SHA-256 authenticates artifact identities; it does not replace search checking.
This format makes the computation re-checkable, not succinct: checking a
trillion-node certificate can still require a trillion-node computation.

## 1. Conventions and supported questions

All indices are zero-based. A Pauli label is `0=I, 1=X, 2=Z, 3=Y`; its low bit
is x and its next bit is z. Multiplication, with phase ignored, XORs labels.
Two nonzero single-qubit labels anticommute exactly when they differ. Multi-qubit
commutation is the parity of those local anticommutations. Pauli weight counts
nonzero labels, including Y once.

The supplied ordered generator list spans S. It may contain dependent or zero
rows. Rows must commute pairwise. Let `rank` be their binary rank and
`k=n-rank`; this certificate format requires `k>0`. Its ordered logical list
has exactly `nl=2k` rows. Every logical row must commute with S and the rank
of the generators plus logical rows must be `n+k`. These conditions ensure
that the list is a basis of N(S)/S. The checker must verify these algebraic
conditions independently. Recomputing its own quotient basis is allowed for
validation, but the ordered supplied basis defines the instance bytes.

Domains:

* `P`: all Pauli labels allowed initially; root labels are 1,2,3 in that order.
* `X`: only I/X allowed initially; root label is 1.
* `Z`: only I/Z allowed initially; root label is 2.

X/Z statements are mathematically meaningful for any isotropic input. The
package's side-certificate API restricts issuance to CSS presentations; that
product restriction does not change the native search semantics here.

`cap` is a nonnegative integer and remains **immutable** throughout certificate
production and checking. Finding a nontrivial logical invalidates a proposed
lower certificate; there is no min-mode cap reduction in this protocol.

Revision-1.1 limits are `1<=n<=4096`, `0<=m<=4096`, generator weight at most
128, column degree at most 128, `0<=cap<=46`, and
`1<=split<=47`. Logical rows can have weight up to n. All counters must fit in
`0..2^63-1`; overflow is an error, never wraparound. Hash arithmetic alone wraps
modulo `2^64`.

Column degree counts supplied generator rows containing the qubit, including
dependent or duplicate rows separately. A zero generator contributes no
column incidence.

The initial specification supported generator weight at most 32 and column
degree at most 64. Its validated independent checker, revision
`ff641d76669bd23b3a83474df0dc50333ee2ae9b` in `stabcert-bnbcheck`, correctly
rejects instances above those limits. Expanded inputs require a checker that
explicitly implements specification revision 1.1; no independently validated
wide-capable checker revision is recorded yet. Compatibility means that a
wide-capable checker also accepts valid original-limit artifacts, not that an
older checker accepts newly supported inputs. CPU builds support the expanded
limits; the CUDA executor retains generator weight 32 and column degree 64.

## 2. Self-contained instance and identities

The header embeds `instance_text`, the **exact ASCII bytes** read by the
prover, encoded as a JSON string. The checker hashes its decoded bytes with
SHA-256 and compares `instance_sha256`, then parses them as below. The
certificate does not require access to the original input file.

The normal instance syntax is:

```
stabcert-bnb-instance 1
n m nl
m generator rows
nl logical rows
n orbit indices
no rep_0 ... rep_(no-1)
domain P|X|Z
```

An operator row is `r q_0 l_0 ... q_(r-1) l_(r-1)`. Its nonzero labels are in
1..3, qubits are strictly ascending, and `r` is its support length. Empty rows
are written as `0`. Tokens are separated by ASCII whitespace. A headerless
legacy instance starts with `n m nl`, omits the final domain line, and means P.
No optional rootmask or additional trailing token is permitted. Embedded NUL,
non-ASCII bytes, signed dimensions, invalid indices, and duplicate qubits are
invalid. Decimal integer tokens may have leading zeroes but no sign. Instance
hashes distinguish different whitespace or equivalent generating sets.

The header also gives `group_sha256`, which deliberately ignores generator
order and choice of basis:

1. Encode each Pauli row as the nonnegative integer
   `sum(x_q*2^q + z_q*2^(n+q))`.
2. Compute the fully reduced binary row basis using **highest set bit** as
   pivot. Each pivot bit is zero in every other row. Sort rows by increasing
   pivot index. These are unique for the row space with this convention.
3. Hash ASCII `stabcert-group/1 n=<n> rank=<rank>\n`, with decimal integers,
   followed by each sorted row in exactly `ceil(2n/8)` little-endian bytes.

Both hashes must match. The exact instance binds search order; the group hash
allows callers to recognize the same stabilizer group in another presentation.
`source_sha256` and `binary_sha256` in provenance are informational identities,
not conditions under which the independent checker assumes correctness. The
prover receives instance/binary digests from a Python caller that re-hashes the
files on every invocation; the checker computes hashes over embedded content
itself and does not trust the caller's assertion.

## 3. Permutations, orbits, roots

The header supplies `permutations`, each a list of n integers representing
`q -> permutation[q]`. The checker verifies each is a bijection and maps every
generator into the row space S. Because a permutation is invertible, this
proves it preserves S. No graph-coloring tool or automorphism finder is trusted.

The connected components of the undirected edges `(q, permutation[q])`, over
all listed permutations, must be **exactly** the orbit partition in the
instance. Component names may be reordered, but input orbit indices must be
contiguous `0..no-1`. Each representative must belong to its corresponding
orbit. With an empty permutation list all orbits must be singletons. Coarser
unjustified orbits are forbidden.

Roots are ordered first by orbit index, then by the numeric allowed root
label. `root_ordinal` is the position in this complete ordered root list,
including roots that turn out dead. It is **not** the bit-packed root value
used by any implementation. To construct root `(oi,l)`:

1. Give every qubit its initial domain state (section 4).
2. Set all qubits in orbits `<oi` to I.
3. Set `rep_oi` to l.

Every other qubit retains its initial domain. For `cap=0`, the root list for
search purposes is empty: the only weight-zero Pauli is I, which is in S.
For positive cap, all declared roots must be covered, even when dead.

Why roots suffice: a minimum logical can be permuted so that a qubit in its
least occupied orbit lies at that orbit's representative. The permutation
preserves S, Pauli weight, and the X/Z restricted domains.

## 4. Node state and derived quantities

The semantic state is an array `lab[0..n-1]` with these meanings:

| state | domain |
|---:|---|
| 0,1,2,3 | the fixed singleton I,X,Z,Y respectively |
| 4 | {I,X,Z,Y} |
| 5 | {I,X} |
| 6 | {I,Z} |
| 7 | {I,Y} |

Initial states are 4 for P, 5 for X, 6 for Z. A fixed non-I assignment has
state 1..3. Restricted states 5..7 are **undecided**, not present support.
The partial Pauli `P_v` contains exactly the fixed non-I labels; `w` is their
count. A completion chooses a label in every current domain.

For each generator g:

* `s(g)` is its symplectic product with `P_v` (0 or 1).
* A support position p, containing qubit q and generator label a, is a
  **candidate** iff `lab[q]==4`, or `lab[q]>4 and lab[q]-4 != a`.
* `C(g)` is that candidate-qubit set; `cnt(g)=|C(g)|`.

The active list A consists of all g with `s(g)=1`, in increasing **input row
index**, including dependent rows if active. A state is dead iff some active
generator has no candidate. Its canonical dead hint is the **least such
input generator index**.

For a nondead state with A empty, `P_v` is a logical witness iff it
anticommutes with at least one supplied logical-basis row. Equivalently, it
is outside S. The checker should establish this using its own row-space test;
if it also uses the supplied logical list, the results must agree. No
anticommuting generator alone implies nontriviality.

## 5. Deterministic rules and node count

Dead states are rejected **before entry**. A root or child rejected as dead
contributes zero nodes and zero trace words. A nondead entry increments the
node count by one, then applies these rules in exactly this order:

1. **LEAF (rule 0):** A is empty. If `P_v` is nontrivial, reject the lower
   certificate. Otherwise it is a stabilizer leaf: stop, without extending it.
2. **WEIGHT (rule 1):** A is nonempty and `w+1>cap`. Stop. At least one further
   non-I qubit would be needed to commute with an active generator.
3. **PACKING (rule 2):** run the algorithm below. If `w+h>cap`, stop with the
   selected generator list as the hint.
4. **BRANCH (rule 3):** choose the branching generator below and enumerate
   children exactly as in section 6.

A discovered logical can be logged internally as rule 4 for debugging, but
rule 4 is forbidden in an accepted certificate.

Stopping at a stabilizer leaf needs an irreducibility argument: choose a
minimum-weight logical in the domain. It cannot contain a nonempty proper
sub-Pauli in S, since multiplying by that sub-Pauli deletes its support and
leaves a lower-weight logical in the same domain. Along its branch, every
nonempty proper partial assignment therefore has an active generator: if it
commuted with S and were outside S, it would itself be a lower-weight logical;
if it were in S, the support-deletion argument would apply. Both contradict
minimality. At
each such generator the completion has a first anticommuting candidate, so
section 6 includes the appropriate child. Dead cuts and the two bounds cannot
remove this completion. Thus failure to extend stabilizer leaves does not
omit every minimum logical. This is the coverage argument; the search does
not enumerate every normalizer Pauli.

### 5.1 Packing bound, including early exits

The certificate fixes `AMAX=96`.

```
if w + len(A) <= cap:
    do not construct a packing; do not prune
else:
    L = first min(96,len(A)) indices of A
    order L by (cnt(g), g), increasing
    used = empty qubit set
    selected = []
    for i = 0 .. len(L)-1:
        if w + len(selected) + (len(L)-i) <= cap:
            break
        g = L[i]
        if C(g) intersects used:
            continue
        append g to selected
        used = used union C(g)
        if w + len(selected) > cap:
            break
    h = len(selected)
    prune iff w+h>cap
```

The hint lists selected generators in **selection order**, not sorted again.
It is the exact canonical list from this algorithm. All candidate sets are
computed from the node's entry domains. The checker verifies their pairwise
disjointness, the active condition, and canonical selection. Truncating A to
96 only weakens the lower bound. The early-exit `w+len(A)<=cap` is a no-prune
shortcut; it is not a one-generator packing hint.

### 5.2 Branch generator

For every active generator g, count children as two for every free candidate
qubit and one for every restricted candidate qubit. Choose the g minimizing
`(child_count(g), g)` lexicographically. The lowest input row index breaks all
ties. Branch choice uses **all** active generators, not only the first 96.

## 6. Exact child order and dead suffix cuts

Let g be the chosen generator and let its candidate positions in the parent
entry state be `p_0 < ... < p_(c-1)`. Positions are zero-based positions in the
ordered support of g, not qubit indices. Since rows are sorted, this also
orders candidate qubits increasingly. Start a mutable working state equal to
the parent entry state. For each candidate position p in this order:

1. Let q be its qubit and a its current domain state.
2. If `a==4`, options are the two labels in `{1,2,3} minus {g_q}`, in increasing
   numeric order. Otherwise the only option is `a-4`.
3. For each option, set q to it. Check deadness. If dead, emit/check an
   uncounted dead-child cut; otherwise visit that child recursively. Restore
   q to a after the child, including all its descendant changes.
4. Restrict q to labels commuting with `g_q`: state 4 becomes `4+g_q`;
   any candidate restricted state becomes 0. This restriction persists when
   considering later candidate positions.
5. If the resulting working state is dead, emit/check one **dead suffix**
   cut and stop all later positions. Otherwise continue to the next position.
   Emit this cut even when p was the final candidate position; the deadness
   test precedes the test for another candidate.

The parent itself is restored when all positions finish. Earlier candidate
restrictions are part of each later child's state. Never branch independently
from the unchanged parent at every candidate.

An edge decision is `(p,li)` where li is the index in that position's option
list (0 or 1). Its byte encoding is `(p<<1)|li`, in 0..255 for support positions
0..127 (the original weight-32 limit used only 0..63). A restricted position
only permits li=0. A dead suffix cut identifies p **after** its commuting
restriction; it is not an additional child and does not increment node count.
No extension remains after the last candidate either: a commuting completion
of active g must differ at at least one candidate.

For a dead suffix, let g' be the active generator without candidates after
the restriction. All remaining undecided qubits are outside its support or
have domains within {I,g'_q}. Later assignments cannot change its syndrome;
restrictions only shrink domains. Thus every later child and descendant is
dead, which justifies cutting the whole suffix.

## 7. Canonical split, frontier and task IDs

`split` S means **support weight**, not a count of stack frames including the
root. A root has weight 1. Each child adds one fixed non-I qubit, so a task
path of length L has weight `L+1`.

For S=1, every declared positive-cap root is a frontier task **before** its
deadness is tested. This preserves the original unsplit task convention:
dead roots are zero-node tasks. For S>1, test root deadness first and then
walk the top tree by sections 5–6. Whenever a nondead child has weight S,
cut it before entry and emit a frontier record. All lower-weight nodes are
entered normally. Leaves/prunes/dead cuts can stop above S. If S exceeds cap,
there can be no frontier tasks; the complete search is explicit in the top.

Task IDs are contiguous from 0 in **frontier appearance order in root-ordered
DFS**. This order does not depend on threads, devices, budgets, chunks, or
completion order. The task path is the full sequence of encoded decisions
from its declared root to that frontier state. The task node has not yet
contributed to generation count or trace. The task's search includes its
entry, all descendants, and no ancestor.

Replaying a task path reconstructs domains and branching decisions under the
original immutable cap. Replay nodes are not counted or hashed a second time.
Every path decision must match the recomputed branching generator, candidate
and label option. A failed replay is a hard error. A supplied complete state
must never replace path verification.

`gen_nodes` counts only entered top-tree nodes. `task_nodes` is the sum of
counts of all canonical frontier tasks. Their sum must equal an unsplit
root-ordered search under the same cap.

## 8. Per-node trace, fixed unsigned arithmetic

This trace is diagnostic, not cryptographic. All operations in this section
are unsigned 64-bit; multiplication/addition wrap modulo `2^64`, right shifts
are logical. Define:

```
mix64(x):
    x = (x XOR (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x XOR (x >> 27)) * 0x94d049bb133111eb
    return x XOR (x >> 31)

salt(q, a) = mix64(8*q + a + 1)
state_fp(lab) = XOR over q=0..n-1 of salt(q, lab[q])
B = 0x00000100000001b3
```

All qubits contribute, including fixed I and undecided/restricted qubits. A
state change from a to b at q updates the fingerprint by XORing `salt(q,a)`
and `salt(q,b)`. Precomputing salts is permitted. This avoids scanning every
qubit at every node without weakening the checker's obligation to reconstruct
the actual domains.

Every **counted entry** contributes exactly two 64-bit words, in this order:

1. Its entry-state fingerprint, before applying any child restrictions.
2. The packed word below, after deciding its terminal/branch rule.

```
packed = w
       OR (len(A) << 8)
       OR (rule << 24)
       OR (g_field << 32)
       OR (h_field << 48)
```

`w` occupies bits0..7; `len(A)` bits8..23; `rule` bits24..31;
`g_field` bits32..47; `h_field` bits48..55; bits56..63 are zero.
For BRANCH, `g_field` is its selected generator and `h_field=0`.
For PACKING, `g_field=65535` and `h_field` is the selected packing size.
For LEAF or WEIGHT, `g_field=65535` and `h_field=0`. An unpruned packing's
intermediate size is not encoded. Record words themselves are numeric values;
the recurrence does not process their constituent bytes.

Start each task trace at H=0. For each word v in its root-first DFS entry
sequence, `H = H*B + v`. Dead children, dead suffixes, replayed ancestors,
frontier markers, scheduler activity and thread IDs contribute no words.
A dead S=1 root task has node count0 and trace0.

The pair `(node_count,H)` is composable. For entry sequences A followed by C:

```
count(A || C) = count(A) + count(C)
H(A || C) = H(A) * B^(2*count(C)) + H(C)   modulo 2^64
```

This enables refinement without changing the logical trace. `global_trace`
is the same recurrence on the entire unsplit root-ordered forest. In contrast,
`top_trace` skips every frontier subtree and hashes only generation entries.

`BNB_CERT_VECTORS.json` supplies standalone arithmetic vectors and a complete
nine-node example. It was generated from explicit label arrays, without
importing the prover. Hex strings there are fixed-width human-readable numbers;
binary record hashes use little-endian 8-byte integers.

## 9. Budget re-splitting is not a new certificate frontier

The certificate always records the original frontier of section 7. Runtime
refinement is an implementation detail. When a task exceeds a node budget:

* Discard its entire partial node count and trace.
* Re-expand it into an ordered sequence of generation-entry segments and
  deeper subtree placeholders. Preserve their exact DFS interleaving.
* Execute/refine those deeper subtrees as necessary.
* Concatenate the generation segments and completed deeper summaries in
  their original DFS order using section 8. The result is the original task's
  single count and trace.

Summing all generation counts and then appending all child traces is **wrong**:
pruned or leaf generation nodes can lie between child placeholders. Completion
order must not influence composition. Nested refinements follow the same rule.
Refinement replay ancestors never contribute again. No aborted task is treated
as proved; it must ultimately resolve into completely explored pieces.

The final file has one record per **canonical** task, regardless of how many
physical scheduler tasks were created. Runtime counters such as aborted pieces
or extra physical tasks belong in JSON progress/provenance, outside the proof
record stream. Existing count-only checkpoints cannot supply missing traces.
Version1 initially rejects combining `--emit-cert` with `--ckpt`; resumable
proof production requires a later explicitly versioned checkpoint extension.

## 10. Container and header

The logical stream starts with 8 bytes `SBCOV1\r\n`, then an unsigned
little-endian u64 header-byte length, then that many UTF-8 bytes containing one
JSON object. Duplicate JSON keys are forbidden. Normative fields:

```
{
  "format": "stabcert-bnb-coverage/1",
  "search": "irreducible-pauli-dfs/1",
  "trace": "poly64-state-rules/1",
  "detail": "frontier",
  "instance_text": "<exact instance ASCII>",
  "instance_sha256": "<64 lowercase hex>",
  "group_sha256": "<64 lowercase hex>",
  "n": 4, "m": 2, "nl": 4, "k": 2,
  "cap": 1, "domain": "P", "split": 2,
  "amax": 96,
  "permutations": [],
  "provenance": {
    "source_sha256": "<64 lowercase hex>",
    "binary_sha256": "<64 lowercase hex>"
  }
}
```

Numeric header values must be JSON integers, not booleans, floats or strings.
Dimensions/domain must agree with the parsed embedded instance and algebra.
The requested lower claim is `cap+1`. A consumer citing this artifact MUST
check `group_sha256 == desired_code.group_hash()`, `domain == desired_domain`,
and `cap+1 == claimed_lower_bound` against the validated header. Equality of
trace or node count cannot replace this binding to the intended question.

Gzip transport is supported by wrapping the entire logical stream in a gzip
member. For deterministic compression, writers use mtime0 and an empty
original filename. Readers accept raw or gzip by magic, never by extension
alone, and must reject data after the logical footer even inside additional
gzip members. The uncompressed proof records are canonical; gzip compression
level is not part of search identity. Zstandard is reserved for a future
optional transport, not required by version1.

CPU and GPU runs with the same instance/cap/split must have **identical proof
record bytes**. Their provenance binary hashes necessarily differ. Therefore
literal identity of whole files is not a valid requirement; compare normalized
headers excluding provenance and the complete uncompressed record stream.
Wall-clock times, hardware and executor names go in an external run record.

## 11. Binary proof records

After the header, records have a one-byte tag followed by the specified
fields. `U` means canonical unsigned LEB128 (least-significant groups of seven
bits first; high bit means continuation). Reject nonminimal encodings,
encodings longer than10bytes, or a tenth byte greater than1. Fixed `u64` means
eight little-endian bytes. No implicit alignment or padding is present.

Top events have implicit IDs0,1,... in record order, including all types below.
`parent_ref=0` denotes a root; otherwise it is `parent_event_id+1` and must name
a previous BRANCH entry. For a root, `edge` is `root_ordinal`. Otherwise edge
is the encoded `(p,li)` decision. The parent and edge are checked against the
expected DFS walk; naming a valid earlier parent is not enough to reorder or
omit its children.

| tag | name | fields after tag |
|---|---|---|
| 0x10 | ENTER | `parent_ref:U edge:U rule:u8 state_fp:u64 w:U active_count:U`, then rule-specific fields |
| 0x11 | DEAD_CHILD_OR_ROOT | `parent_ref:U edge:U dead_generator:U state_fp:u64` |
| 0x12 | DEAD_SUFFIX | `parent_ref:U position:U dead_generator:U state_fp:u64` |
| 0x13 | FRONTIER | `parent_ref:U edge:U task_id:U state_fp:u64` |
| 0x1f | TOP_END | `event_count:U task_count:U gen_nodes:U top_trace:u64` |

ENTER rule fields: none for LEAF/WEIGHT; for PACKING,
`packing_size:U` followed by that many generator indices `U` in selection
order; for BRANCH, one `generator:U`. Only rules0..3 are permitted.
DEAD_SUFFIX always has nonzero parent_ref and uses the state after the named
position's commuting restriction. It terminates the unvisited suffix for
that parent. Every supplied fingerprint and hint is recomputed.

The checker drives the deterministic top traversal and consumes exactly one
expected record for each event. It rejects a missing, extra, out-of-order or
wrong-rule record, an invalid parent, a mismatching hint, or an unjustified
frontier cut. This checks coverage rather than merely validating individual
prune hints. At TOP_END all roots must have been completely covered. Its
counts and top trace must equal the independently reconstructed values.

Next, task results appear in exactly increasing canonical ID order:

| tag | name | fields after tag |
|---|---|---|
| 0x20 | TASK_RESULT | `task_id:U root_ordinal:U path_len:U path_bytes[path_len] node_count:U trace:u64` |

There must be exactly `task_count` results, including zero-node dead-root
tasks at split1. IDs, root and complete path must match the corresponding
frontier. The checker independently executes the subtree, rejects any logical
leaf, and compares count and trace. It may parallelize verification, but it
must eventually check every ID. Sampling cannot yield an accepted lower bound.

The final record is:

| tag | name | fields after tag |
|---|---|---|
| 0x7f | COMPLETE | `task_count:U gen_nodes:U task_nodes:U total_nodes:U global_trace:u64` |

All fields must match independently accumulated counts/traces,
`total_nodes=gen_nodes+task_nodes`, and the stream must end immediately after
this record. Missing footer means an incomplete artifact, not a weaker proof.
The global trace inserts each task's entry stream at its frontier position in
the top walk, using the composable summaries. A checker can spool top entry
segments between frontier markers, then combine them with ordered task results
without keeping the whole top tree in memory.

## 12. Optional fully explicit mode

With header `detail="full"`, the top section is unchanged. For each task, in
canonical task order, emit:

1. `TASK_BEGIN` tag0x21 with `task_id:U`.
2. Its entire uncut subtree using ENTER, DEAD_CHILD_OR_ROOT and DEAD_SUFFIX
   records. IDs and parent references are local to this task, starting at0.
   The task-relative root uses parent_ref0 and edge0; it is the already
   reconstructed frontier state, not a newly constructed global root.
   No FRONTIER record is allowed inside this transcript.
3. `TASK_END` tag0x22 with `event_count:U`.
4. Its normal TASK_RESULT record.

A dead split1 task has one local DEAD_CHILD_OR_ROOT record and zero entered
nodes. The checker validates every event against its own deterministic walk
and compares the normal summary. This mode is intended for small regression
cases and explanation; it can be enormous on production trees. There is no
trusted omission or unchecked prune hint in full mode.

## 13. Exporter interface and atomic completion

Proposed native options:

```
--emit-cert FILE
--cert-metadata FILE.json
--cert-detail frontier|full
```

The metadata file supplies at least `group_sha256` and the verified candidate
`permutations`. Python computes the group hash and supplies permutations from
its verified orbit object. Native emission may carry these assertions, but
the independent checker always validates them. For nonsingleton input orbits,
missing permutations are an error; the exporter must not silently claim
unjustified symmetry. Existing `--instance-sha256`/`--binary-sha256` arguments
are required with certificate emission.

`--emit-cert` is incompatible with `--stop-first` and version1 count-only
`--ckpt`. Node budgets and internal re-splitting remain supported. If a logical
is found, the run returns its witness in ordinary JSON output and **does not
publish a completed lower-bound artifact**. The exporter stages the stream at
a temporary path, writes COMPLETE only after all tasks resolve and all replay
checks pass, finishes compression, then atomically publishes the final path.
Interruption, replay failure, overflow, disk failure, compression failure or
unresolved tasks must leave no apparently complete final file. Progress and
performance JSON remain outside the proof stream.

The raw record writer must be streaming. Gzip may be supplied by the package's
Python standard library or an external `gzip -n -c` process; this is transport,
not a new cryptographic implementation. Standalone raw output is supported.
The exact native compression integration is an implementation decision pending
review; it does not change any logical bytes consumed by the checker.

For20million canonical tasks, compact records use variable-length paths and
counts, not full n-qubit states per task. At split5, a task path is four bytes;
top frontier plus result records are typically tens of bytes per task before
compression. Writers/checkers must be able to spool paths and summaries to
disk; they must not require materializing a fully explicit subtree or the
entire decompressed certificate in RAM. Record counts/header lengths must be
resource-checked before allocation; input values never authorize unbounded
allocations silently.

## 14. Independent-checker acceptance checklist

The checker must, in order or an equivalent dependency order:

1. Parse transport/container strictly and validate version identifiers.
2. Hash/parse embedded instance; validate dimensions, isotropy, logical basis,
   group hash, domain and immutable cap.
3. Verify every permutation and the exact orbit partition/representatives.
4. Reconstruct/check every top event in deterministic DFS order, including
   uncounted dead children and suffix cuts, and generate the canonical frontier.
5. Verify every canonical task by full independent search (or its complete
   explicit transcript); reject any nontrivial logical, count/trace mismatch,
   missing task, duplicate task, wrong root/path, or unresolved refinement.
6. Validate the footer and reject trailing bytes. Only then accept
   `no nontrivial <domain> logical of Pauli weight <=cap`.

Acceptance soundness rests on completeness of the rules plus a faithful
independent checker. The completeness argument is also given in
`docs/SOUNDNESS.md` section 5. The independent checker must not reuse the
prover's implementation of these rules.

Version 1 covers exactly those rules. The opt-in parity rules S0/S1/S2
(`--bound greedy+s0 | greedy+s0s1 | greedy+s0s1s2`, `docs/SOUNDNESS.md`
section 5.1) and the experimental `greedy+b1b3` bounds are outside it: the
engine refuses `--emit-cert` with them, and their internal trace rule codes
(5, 6, 7 for S0, S1, S2) are not version-1 rule values. Admitting a new rule
needs a spec revision with its exact definition and hint encoding, and
support in the independent checker.

Changing cap, domain, generator order, logical list, orbit order, representative,
path byte, parent ID, prune hint, task count, trace or footer must either be
rejected or result in a separately valid fully checked proof of the new
question. Binary/source hashes are not a substitute for this checking.

## 15. Small complete example and required validation

Use n4, generators XXXX/ZZZZ, logical rows XXII/XIXI/ZZII/ZIZI, trivial orbits
ordered0,1,2,3, domainP, cap1. This is a [[4,2,2]] code. The first three roots'
qubits each permit three nondead entries pruned by WEIGHT; the final qubit's
three roots are dead. Thus the canonical unsplit count is9.

* split1:12 canonical tasks,9 nodes in tasks,0 generation nodes. Tasks9,10,11
  are dead-root tasks with count0/trace0.
* split2:0 canonical tasks,9 generation nodes,3 explicit dead-root cuts.
* Both have global trace `05dc5a17c64e8a73`.

The companion vectors list each root's domains, state fingerprint, packed
record and task trace. Additional required tests before release: CPU/GPU
payload identity; different internal budgets/refinement depths producing the
same payload; task/edge/hint/footer tampering; truncated gzip/raw input;
cap0; dependent/zero generators; P/X/Z; independently generated small codes;
orbit partitions finer or coarser than the permutation components (reject);
logical lists with a row in S, so the required rank n+k fails (reject);
revision-1.1 positions31/32/63/64/127 and path byte255; column degrees64/65/128
including duplicate rows; generator weight129 and column degree129 (reject);
and the complete452 cap17 and488 cap21 runs checked against their no-proof
baseline counts. A certificate is not accepted merely because its count
matches an old ledger.
