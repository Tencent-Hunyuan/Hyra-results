# B&B coverage certificate, container version 2

Specification revision 2.0-rc1, 2026-10-27. This document is self-contained. An independent checker must be
implemented from it alone. Ambiguities are specification bugs: report them before substituting an interpretation.

A successful check establishes that **no nontrivial logical Pauli of weight at most `cap` exists within `domain`**.
Version 2 extends version 1 (the version-1 specification, revision 1.1) by three optional pruning rules, S0, S1
and S2. Every version-1 rule that a version-2 checker needs is restated here.

Under the `greedy` rule set, a version-2 artifact has exactly the version-1 search, trace and record semantics: its
record stream after the header is byte-identical to the version-1 record stream for the same instance, cap,
split, permutations and detail. Only the magic, the header identifiers and the `rules`/`rules_version` fields
differ. A version-1 artifact keeps its version-1 meaning. A version-2 checker MUST also implement the version-1
profile.

**How a version-2 certificate is checked.** Sections 0-5 define the search: a deterministic depth-first search
over partial Pauli assignments. At every node, a pure function of the node state decides LEAF, WEIGHT, a prune
(PACKING, S0, S1, S2) or BRANCH. The certificate lists:
- the explicit top part of the tree;
- every frontier subtree, with its node count and trace.

The checker re-executes every frontier subtree with its own implementation of these functions, and compares
counts and traces exactly.

The checker also runs a short **verification** at every prune it takes. Each prune rule defines a
**justification object J** (sections 6.3, 8.2, 9.2, 10.2). The checker computes J itself while deciding, and checks
it with the rule's VERIFY procedure. VERIFY uses only the node state and the instance, never the deciding code's
intermediate results. A prune whose J fails VERIFY is a checker error and the certificate is rejected.

Soundness of an accepted certificate therefore rests on:
- the VERIFY procedures;
- the lemmas of section 11;
- the coverage argument of section 4;
- the **trusted base**: the checker code that has no VERIFY, exactly as in version 1. This is:
  - container, instance and header parsing, and the hashes;
  - permutation and orbit validation, and root construction;
  - the node state (s, candidates, A), the deadness test, LEAF with its logical test, and WEIGHT;
  - BRANCH: the generator choice, the child enumeration, the restrictions and the dead-suffix cuts;
  - frontier and task-path replay, and the count and trace bookkeeping.

It does not rest on the prune deciders (PACKING_SELECT, UNIT, PROP, the S1 decision, the S2 decision). These can
only cause a mismatch (rejection) if implemented wrongly.

The trace is deliberately noncryptographic. A count or trace is not a proof: the checker executes every subtree.

## 0. Versions and header identifiers

| header field | value |
|---|---|
| `format` | `"stabcert-bnb-coverage/2"` |
| `search` | `"irreducible-pauli-dfs/2"` |
| `trace` | `"poly64-state-rules/2"` |
| `rules` | one of `"greedy"`, `"s0"`, `"s0s1"`, `"s0s1s2"` |
| `rules_version` | `1` |

`rules` selects the pruning rule set:
- `greedy`: only the rules of sections 3 and 5.1;
- `s0`: adds S0;
- `s0s1`: adds S0 and S1;
- `s0s1s2`: adds S0, S1 and S2.

The rule set of a node is fixed for the whole certificate. `rules_version` identifies this document's rule
definitions. A checker MUST reject an unknown `format`, `search`, `trace`, `rules` or `rules_version`.

**Version-1 profile.** A version-1 artifact has magic `SBCOV1\r\n` and the identifiers `stabcert-bnb-coverage/1`,
`irreducible-pauli-dfs/1` and `poly64-state-rules/1`.
- Canonical version-1 writer output has no `rules` or `rules_version` field. This describes writers; it is not a
  reader rejection rule.
- Version-1 readers ignore unknown header fields. A version-1 certificate is always checked with greedy-only
  decisions, so a `rules` field in a version-1 header has no effect: any extra prune mismatches the re-decided
  trace.
- Consumers MUST NEVER label evidence from a version-1 header's `rules` field.

It is checked exactly as a version-2 `greedy` artifact, with two differences:
- ENTER records permit only rules 0-3;
- the transport is the version-1 one: gzip may be followed by further members, and every logical byte after
  COMPLETE, in any member, is rejected.

All other header fields are as in version 1: `instance_text`, `instance_sha256`, `group_sha256`, `n`, `m`, `nl`,
`k`, `cap`, `domain` (`"P"`, `"X"` or `"Z"`), `split`, `amax` (= 96), `permutations` (a list of integer lists),
`provenance` (an object with `source_sha256` and `binary_sha256`, informational only), and `detail` (`"frontier"`
or `"full"`).
- Numeric header values MUST be JSON integers: not booleans, floats or strings.
- Duplicate JSON keys are forbidden.
- `n`, `m`, `nl`, `k` and `domain` must agree with the parsed embedded instance and its algebra.

**Consumer binding.** The certified claim is the lower bound `cap + 1`. A consumer citing an artifact MUST check,
against the validated header:
- `group_sha256 == desired_code.group_hash()`;
- `domain == desired_domain`;
- `cap + 1 == claimed_lower_bound`.

Equality of a trace or a node count cannot replace this binding to the intended question. `rules` does not enter
the claim: the checker re-executes with the declared rule set.

## 1. Conventions, supported questions, limits

All indices are zero-based. A Pauli label is `0=I, 1=X, 2=Z, 3=Y`. Two nonzero single-qubit labels anticommute
exactly when they differ. Multi-qubit commutation is the parity of the local anticommutations. Pauli weight counts
the nonzero labels.

**Generators.** The supplied ordered generator list spans the stabilizer group S. It may contain dependent, zero
or duplicate rows. Rows must commute pairwise. Let `rank` be their binary rank and `k = n - rank`; this format
requires `k > 0`.

**Logicals.** The ordered logical list has exactly `nl = 2k` rows. Every logical row must commute with S, and the
rank of the generators plus the logical rows must be `n + k`. The checker verifies these conditions itself.

**Domains:**
- `P`: all labels allowed initially; root labels 1, 2, 3 in that order.
- `X`: only I or X allowed; root label 1.
- `Z`: only I or Z allowed; root label 2.

`cap` is a nonnegative integer and is immutable during production and checking. Finding a nontrivial logical
invalidates the certificate.

**Limits:** `1 <= n <= 4096`, `0 <= m <= 4096`, generator weight at most 128, column degree at most 128,
`0 <= cap <= 46`, `1 <= split <= 47`.
- Column degree counts the supplied rows containing the qubit, duplicates separately; zero rows contribute no
  incidence.
- All counters must fit in `0..2^63-1`. Overflow is an error, never wraparound.
- Only the trace arithmetic wraps, modulo 2^64.

## 2. Instance, identities, permutations, orbits, roots

These are identical to version 1. They are restated here so that this document is self-contained.

**Instance.** The header embeds `instance_text`: the exact ASCII bytes of the instance, as a JSON string. The
checker hashes the decoded bytes with SHA-256, compares them with `instance_sha256`, then parses:

```
stabcert-bnb-instance 1
n m nl
m generator rows
nl logical rows
n orbit indices
no rep_0 ... rep_(no-1)
domain P|X|Z
```

Operator rows:
- An operator row is `r q_0 l_0 ... q_(r-1) l_(r-1)`, with labels in 1..3, strictly ascending qubits, and `r` the
  support length. An empty row is written `0`.
- A row's "support positions" are 0..r-1 in this order.

Token rules:
- Tokens are separated by ASCII whitespace.
- A headerless legacy instance starts with `n m nl`, omits the domain line, and means P.
- No other trailing token is allowed.
- Decimal tokens may have leading zeroes but no sign.
- Invalid: embedded NUL, non-ASCII bytes, signed dimensions, invalid indices, and duplicate qubits in a row.
- Instance hashes distinguish different whitespace or equivalent generating sets.

**group_sha256** (independent of generator order and basis):
1. Encode each row as the integer `sum(x_q*2^q + z_q*2^(n+q))`.
2. Compute the fully reduced row basis with the highest set bit as pivot (each pivot bit zero in every other row),
   sorted by increasing pivot.
3. Hash ASCII `stabcert-group/1 n=<n> rank=<rank>\n`, followed by each row in `ceil(2n/8)` little-endian bytes.

Both hashes must match.

**Permutations.** `permutations` lists permutations `q -> perm[q]`. Each must be a bijection mapping every generator
into the row space S.

**Orbits.** The connected components of the edges `(q, perm[q])` over all listed permutations must be exactly the
instance's orbit partition. Orbit indices are contiguous `0..no-1`, and `rep_oi` must lie in orbit oi. With no
permutations, all orbits are singletons.

Singleton orbits do not imply the identity orbit order. The orbit index of each qubit is read from the instance,
and it determines the root order and the qubits set to I at each root.

**Roots** are ordered by orbit index, then by allowed root label. To build root (oi, l):
1. Give every qubit its initial domain state.
2. Set every qubit in an orbit `< oi` to I.
3. Set `rep_oi` to l.

Every other qubit keeps its initial domain. For `cap = 0` there are no roots: the only weight-zero Pauli is I, which
is in S. For positive cap every declared root is covered, even when dead. `root_ordinal` is the position in this
complete ordered root list.

**Why roots suffice.** Let oi be the least orbit index on which a minimum logical L is non-I. An element of the group
generated by the listed permutations maps a qubit of L's support in orbit oi to `rep_oi`. The permutation preserves S, the Pauli weight and
the X/Z domains. The image of L is then a completion of the root (oi, l) with oi that orbit and l its label at the
representative: it is I on every orbit `< oi`.

## 3. Node state, derived quantities, dead states, LEAF, WEIGHT, BRANCH, child order

**State.** `lab[0..n-1]` with:

| state | domain |
|---|---|
| 0, 1, 2, 3 | the fixed singleton I, X, Z, Y |
| 4 | {I, X, Z, Y} (free) |
| 5, 6, 7 | {I, X}, {I, Z}, {I, Y} (restricted) |

- Initial states are 4 (P), 5 (X) or 6 (Z). A qubit with state 4..7 is **undecided**.
- `P_v` = the fixed non-I labels; `w` = their number.
- `dom(q)` = the set of labels of `lab[q]`.

**Per generator g:**
- `s(g)` is the commutation parity of g with `P_v`.
- A support position p (qubit q, generator label `a = g_q`) is a **candidate** iff `lab[q] == 4`, or
  `lab[q] > 4 and lab[q]-4 != a`.
- `C(g)` is the candidate-qubit set; `cnt(g) = |C(g)|`.

**Active list and dead states.**
- `A` = all g with `s(g) = 1`, in increasing input row index.
- A state is **dead** iff some active generator has no candidate. Its dead hint is the least such generator index.

**Node count and rules.** Dead states are rejected before entry; they contribute zero nodes and zero trace words.
A nondead entry increments the node count by one, then applies, in this order:
1. **LEAF (rule 0):** `A` is empty. If `P_v` is a nontrivial logical (anticommutes with some logical row;
   equivalently it is outside S), the certificate is invalid. Otherwise stop.
2. **WEIGHT (rule 1):** `A` is nonempty and `w + 1 > cap`. Stop.
3. **Pruning rules** (section 5), per `rules`: possibly stop with rule 2, 5, 6 or 7.
4. **BRANCH (rule 3):** for every active g, let `child_count(g)` = 2 per free candidate + 1 per restricted
   candidate. Choose the g minimising `(child_count(g), g)`, over ALL active generators.

**Child order.** Let the branch generator g have candidate positions `p_0 < ... < p_(c-1)`, all in the parent's
entry state. Start a working state equal to the parent. For each p in order:
1. Let q be its qubit and a its current state. Options:
   - if `a == 4`: the two labels of {1,2,3} \ {g_q}, ascending;
   - otherwise: the single label `a-4`.
2. For each option: set q to it. If dead, emit/check an uncounted dead-child cut; otherwise visit the child
   recursively. Then restore q to a.
3. Restrict q to labels commuting with g_q: state 4 becomes `4 + g_q`; a candidate restricted state becomes 0.
   This persists for later positions.
4. If the working state is now dead, emit/check one dead-suffix cut and stop (also after the last position).

An edge decision is `(p, li)`, encoded as the byte `(p<<1) | li`.

## 4. Coverage theorem (why the search, with its prune rules, misses no minimum logical)

Let L* be a minimum-weight nontrivial logical within the domain. By the root argument, after a permutation
preserving S, L* is a completion of some root. For any node v on L*'s branch, write `L* = P_v (x) R`, with R on
undecided qubits.

- (a) v is not dead. Each active g needs an odd number of R's qubits whose L*-label differs from g_q; those are
  candidates.
- (b) If `A` is empty, then `P_v` is in N(S). Write `M = P_v`. M is a nonempty restriction of L*: it agrees with
  L* on `supp(M)`, which is a subset of `supp(L*)`. If `supp(M)` were a proper subset, then:
  - either M is not in S, and M is a nontrivial logical lighter than L*;
  - or M is in S. Then `L* M` is in N(S) and not in S, because L* is not in S. So it is a logical, within the
    domain, since each domain is closed under products. L* and M agree on `supp(M)`, so
    `supp(L* M) = supp(L*) \ supp(M)`, and `L* M` is lighter than L*.

  Either way L* would not be minimal. Hence `P_v = L*`, which is reported.
- (c) WEIGHT does not stop v: `w < wt(L*) <= cap`.
- (d) No prune rule of section 5 stops v. Every prune rule here stops a node only when **no zero-syndrome completion
  of weight <= cap exists** (section 11). L* is such a completion.
- (e) For the branch generator, the first candidate whose L*-label anticommutes with g_q selects exactly one child
  containing L*.

So following L* ends at (b), and an accepted certificate proves `d_domain >= cap + 1`.

## 5. Pruning rules at a node: dispatch

For `rules = greedy` this is version 1's PACKING rule (section 5.1). For the other rule sets it is the **fused
packing pass** of section 6 followed by the rules of sections 8-10. In what follows, `r = cap - w` (r >= 1 after
WEIGHT) and `R = |A|`.

Notation:
- In sections 4 and 11, R instead denotes the undecided part of a completion.
- In cost lines, `wmax` is the maximum generator weight and `cmax` the maximum column degree.
- `W` denotes the S2 span (section 10).
- The split parameter is written `split`; `S` is the stabilizer group.

### 5.1 `rules = greedy` (version 1 PACKING, rule 2)

```
if w + R <= cap: no prune
else:
    L = the first min(96, R) entries of A (A is in increasing index order);
        then order L by (cnt(g), g) increasing
    used = {}; selected = []
    for i = 0 .. len(L)-1:
        if w + len(selected) + (len(L) - i) <= cap: break
        g = L[i]; if C(g) intersects used: continue
        append g to selected; used = used U C(g)
        if w + len(selected) > cap: break
    prune (rule 2) iff w + len(selected) > cap; the hint is `selected`
```

### 5.2 `rules` in {s0, s0s1, s0s1s2}

```
if R < r: no prune (none of the rules below is applied)
(np, PK) = PACKING_SELECT(state)                         section 6.1
if np > r:                                    prune, rule 2 (PACKING), packing PK
elif rules == s0s1s2 and np == r - 2:         prune iff S2(state, PK)   (rule 7)    section 10
elif rules in {s0s1, s0s1s2} and np == r - 1 and r >= 2:
                                              prune iff S1(state, PK)   (rule 6)    section 9
elif np == r:                                 prune iff S0(state, PK)   (rule 5)    section 8
else: no prune
```

Branches are tested in exactly this order. For example, with rules s0s1s2 and np = r-2, only S2 is evaluated.
The condition `r >= 2` of the S1 branch is redundant, since np >= 1 at every live node with `R >= r >= 1`.
When a rule does not prune, the node continues with BRANCH. A rule that "gives up" (a capacity limit, section 13)
does not prune.

`PK` is the packing returned by PACKING_SELECT; its order is defined in section 6.1. Its length is always
`min(np, r + 1)`:
- if `np <= r`, PK has exactly np entries, and rule 5, 6 or 7 uses all of them (np = r, r - 1, r - 2);
- if `np > r`, PK has exactly r + 1 entries, and rule 2 uses exactly these r + 1 generators. np itself may exceed
  r + 1; it is then used only in the comparison `np > r`.

PK is part of the record hint (section 12).

## 6. Packing selection (PACKING_SELECT) and the PACKING rule

### 6.1 Decision

All quantities are on the node's entry state.

```
PACKING_SELECT:
  # (i) greedy packing, collecting at most r+1 generators
  L = the first min(96, R) entries of A (A is in increasing index order);
      then order L by (cnt(g), g) increasing
  used = {}; PK = []
  for g in L:
      if len(PK) == r + 1: break
      if C(g) intersects used: continue
      append g; used = used U C(g)
  np = len(PK)
  if np >= r: return (np, PK)
  # (ii) threshold search over A, only if R <= 64
  if R > 64: return (np, PK)
  return MIS_THRESHOLD(np, PK)
```

**MIS_THRESHOLD.**
- Let `V = (a_0, ..., a_(R-1))` be A in increasing index order. Vertex j is `a_j`.
- Vertices j and j' are adjacent iff `C(a_j)` and `C(a_j')` share a qubit (j != j').
- For a vertex set X and vertex u, `deg_X(u) = |adjacent(u) cap X|`.
- `MINDEG(X)`: the vertex u in X minimising `(deg_X(u), u)`. Scanning u in increasing order and stopping at the
  first u with degree 0 gives the same result.
- `T(r) = r - 1` if `rules` is s0s1 or s0s1s2; `T(r) = r` if `rules` is s0.

```
MIS_THRESHOLD(lb, PK):
  # (a) min-degree greedy independent set
  X = V; Sg = {}
  while X nonempty: u = MINDEG(X); Sg = Sg U {u}; X = X \ ({u} U adjacent(u))
  if |Sg| > lb:
      PK = the first min(|Sg|, r+1) generators of Sg in increasing index order
      if |Sg| >= r: return (|Sg|, PK)
      lb = |Sg|
  # (b) clique-cover upper bound
  Q = the sequence of distinct qubits q in the order of first appearance when scanning
      a_0, a_1, ... and, within each generator, its candidate positions in support order
  for q in Q: M(q) = { j : q in C(a_j) }
  U = V; cover = 0
  while U nonempty and cover < r:
      choose q in Q maximising |M(q) cap U|; the first such q in Q order on ties; b = that size
      if b <= 1: cover = cover + |U|; U = {}; break
      U = U \ M(q); cover = cover + 1
  ub = cover if U is empty else r
  if ub < T(r) or ub <= lb: return (lb, PK)
  # (c) branch and bound, stopping at the target ub
  best = lb; bs = none
  SEARCH(X = V, cur = 0, chosen = {})
  if bs is none: return (lb, PK)
  PK = the first min(best, r+1) generators of bs in increasing index order
  return (best, PK)

SEARCH(X, cur, chosen):
  if best >= ub: return
  if X empty: if cur > best: best = cur; bs = chosen.  return
  if cur + |X| <= best: return
  u = MINDEG(X); d = deg_X(u)
  SEARCH(X \ ({u} U adjacent(u)), cur + 1, chosen U {u})
  if d > 0: SEARCH(X \ {u}, cur, chosen)
```

Notes:
- `best` and `bs` are global to one MIS_THRESHOLD call. The early return `best >= ub` is checked at entry of every
  SEARCH call, including after the first branch returns.
- The selected packing PK keeps the greedy order in (i). It is in increasing index order when it comes from (a) or
  (c).
- In (a), PK is replaced only if `|Sg| > lb`. Otherwise PK remains the greedy list.
- The returned `np` can exceed r from (i) (np = r + 1), from (a) (|Sg| > r), or from (c): ub can exceed r through
  the `b <= 1` step, and then `best` can exceed r. In every case PK has length `min(np, r + 1)`.

### 6.2 PACKING rule (rule 2)

The PACKING rule applies iff `np > r`: stop with rule 2, packing PK, of length r + 1.

### 6.3 Justification J_P and VERIFY_PACKING

`J_P = PK[0 .. np'-1]`, with np' = np for rules 5-7 and np' = r + 1 for rule 2.

VERIFY_PACKING(J_P, required size):
- every listed generator is active;
- the listed generators are pairwise distinct;
- their candidate sets are pairwise disjoint (mark qubits);
- `|J_P| = required size`: r + 1 (rule 2), r (rule 5), r - 1 (rule 6), or r - 2 (rule 7).

Cost: O(sum of the listed generators' weights).

Lemma P (section 11.1): if VERIFY_PACKING passes with size > r, no zero-syndrome completion of weight <= cap exists.

## 7. Common notions for S0, S1 and S2

Fix a node state and a packing `PK = (g_0, ..., g_(h-1))`: the first h entries of PACKING_SELECT's list. Packed
variable i belongs to `g_i`, and the variable index is the position in PK.

**Values.** The value list `D_i` is built by scanning the support positions p of `g_i` in increasing order. For each
candidate position (qubit q, generator label `a = g_i,q`), take each label `l = 1, 2, 3` in increasing order with:
- `l != a`;
- `l` in `dom(q)`: any label if `lab[q] == 4`, only `lab[q]-4` if restricted.

Each such pair `(q, l)` is a value, and values keep this canonical order.

**Flips and syndromes.**
- A pair `(q, l)` **flips** generator g iff q is in `supp(g)` and `l != g_q`.
- `flips(q,l)` is the set of generators flipped, listed in increasing generator index; this is also the order of the
  generators containing q, the "column order".
- `sigma(q,l)` is the same set viewed as an m-bit vector.

**Ownership.** `own(q) = i` if q is in `C(g_i)`, and undefined otherwise. Candidate sets are disjoint, so this is
well defined.

**Flipper set.** For a generator g, `F(g)` is the set of packed variables i such that some support position of g
has a qubit q with `own(q) = i` and some label of `dom(q)` other than I differs from `g_q`. Equivalently, some value
in `D_i` flips g.

**Modified variants.** A *modified* system has an extra flip set EF and an excluded qubit qe. In it:
- the targets are `t(g) = s(g) XOR [g in EF]`;
- `F(g)` is computed with qe treated as unowned;
- the value lists exclude every value at qe: the modified `D_i` is `D_i` without its values `(qe, l)`. A modified
  `D_i` can be empty (when `C(g_i) = {qe}`). Such a variable has no alive value and is wiped (7.1).

The unmodified system has EF empty and no qe.

**Lemma S0, used throughout.** If `|PK| = r` and a zero-syndrome completion R of weight <= r exists, then R consists
of exactly one value `x_i` in `D_i` per packed variable and nothing else, and for every generator g,
`#{i : x_i flips g} = s(g) (mod 2)`. This is the "parity system on PK"; a modified system replaces `s(g)` by `t(g)`.
The proof is in section 11.2.

### 7.1 UNIT (the unit pass)

`UNIT(PK, EF, qe)` returns a refutation flag. In recording mode it also returns the records used by S1: the recorded
zero list Z, the need list N, the first killer of every value of every packed variable, and the list of wiped
variables. All flipper sets are computed on the node's entry domains, with only the modification above. Removing a
value never changes another value's status ("single pass, original domains").

1. **Odd scan.** Let ODD be the sequence formed by:
   - first, the active generators in increasing index, skipping packed ones, keeping those with `t(g) = 1`;
   - then the generators of EF, in EF order, with `s(g) = 0`, skipping packed ones. These have `t(g) = 1`.

   For each g in ODD, in order:
   - if `F(g)` is empty, g is a **zero generator**. Append it to the recorded zero list Z if Z has fewer than 96
     entries;
   - if `|F(g)| = 1`, append `(g, the variable)` to the need list N if N has fewer than 96 entries.

   The scan covers all of ODD, also after a zero generator has been found.
2. **Killers.** For every packed variable i in increasing order, and every value `c = (q, l)` of `D_i` (the modified
   `D_i` in a modified system) in canonical order, the **first killer** `fk(c)` is the first of the following that
   applies:
   1. **Column killer.** Scan the generators g containing q in column order, skipping `g_i`. Take the first g with
      `[c flips g] != t(g)` and `F(g)` a subset of `{i}`.
   2. **Need killer.** Scan N in order, among entries for variable i. Take the first generator g that c does not
      flip.

   If neither applies, c is **alive**. Every value of every variable gets a first killer or is alive. This holds
   also when a zero generator exists, or when an earlier variable is wiped.
3. A variable is **wiped** if none of its values is alive. In particular, a variable with an empty (modified) value
   list is wiped.
4. `UNIT` is **refuted** iff some zero generator exists or some variable is wiped.
   - The refutation counts every zero generator. Only the recorded list Z is truncated, to its first 96 entries, so
     Z is nonempty iff a zero generator exists.
   - The refutation flag does not depend on recording. A non-recording UNIT may stop at the first zero generator or
     wiped variable.

### 7.2 PROP (the propagation fixpoint used by S0)

`PROP(PK)` works on the unmodified system with `h = r`.

1. **Choices.** The choices are all values of all packed variables, in variable then canonical order.
   - Give up (no prune) if there are more than 1024 choices.
   - Each choice c has its flip list `flips(c)`.
   - Give up if the number of distinct generators occurring in flip lists exceeds 2048, or if the total length of
     all flip lists exceeds 6144.
2. If some active generator occurs in no flip list, PROP is refuted.
3. **Fixpoint.** Every choice starts alive. Repeat until nothing changes, for every generator g occurring in some
   flip list (the order is immaterial, Lemma 11.3):
   - for each variable v with at least one alive choice flipping g: if all of v's alive choices flip g, v
     contributes 1; otherwise v is mixed for g. Variables with no alive choice flipping g contribute 0;
   - let `p` be the XOR of the contributions;
   - two or more mixed variables: do nothing for g;
   - no mixed variable: PROP is refuted if `p != s(g)`;
   - exactly one mixed variable mv: kill every alive choice of mv whose flip bit for g differs from `s(g) XOR p`.
     If mv has no alive choice left, PROP is refuted.

   Unless refuted, PROP is not refuted at the fixpoint.

## 8. S0 (rule 5), for `|PK| = r`

### 8.1 Decision

`S0` prunes iff `UNIT(PK, {}, none)` is refuted, or PROP did not give up and `PROP(PK)` is refuted.

### 8.2 Justification J_S0: a derivation

J_S0 = (PK, a derivation over the parity system on PK). A **derivation** works on alive sets `AL_i`, which start as
the system's value lists (in a modified system, without the values at qe). It is a sequence of steps:
- **ELIM(g, v).** Every packed variable `j != v` must be *determined* for g over the current alive sets. That is,
  either its alive set is nonempty and all its alive values flip g ("all flip", contribution 1), or none of its
  alive values flips g (contribution 0). **A variable with an empty alive set is determined for every g, with
  contribution 0.** Let `p` be the XOR of the contributions of all `j != v`. The step removes every
  alive value of v whose flip bit for g differs from `t(g) XOR p`. It must remove at least one value.
- **END_WIPED(v).** `AL_v` is empty.
- **END_ODD(g).** Every packed variable is determined for g, and `p != t(g)`, where `p` is the XOR of the
  contributions of all variables.

A derivation ends with exactly one END step, and its last step is the END.

**How a checker obtains J_S0 from its own decision:**
- **UNIT refuted, Z nonempty:** the derivation `END_ODD(Z[0])`. No variable has a value flipping `Z[0]`, so every
  variable is determined with contribution 0, and `t = 1`.
- **UNIT refuted with a wiped variable i** (any wiped variable; the choice is the checker's): for the values c of i
  in canonical order, if c is still alive, emit `ELIM(fk(c), i)`; then `END_WIPED(i)`.
  - Every other variable is determined for `fk(c)`, with contribution 0: it has no value flipping `fk(c)`.
  - The step removes c: for a column killer, c's flip bit differs from `t`; for a need killer, `t = 1` and c does
    not flip.
  - A variable with an empty modified value list gives `END_WIPED(i)` directly.
- **PROP refuted:** the killing actions of the fixpoint, in the order the checker performed them, each as
  `ELIM(g, mv)`; then `END_WIPED(mv)` or `END_ODD(g)`.
  - An active generator in no flip list gives `END_ODD(g)` directly.
  - A checker that kills one choice at a time may emit one ELIM per kill batch for the same (g, mv). Every emitted
    step must remove at least one value.

### 8.3 VERIFY_S0

1. Run VERIFY_PACKING(PK, r).
2. Build the value lists `D_i` from the definitions in section 7, independently of the decider.
3. Replay the derivation: for ELIM, recompute "determined" and the flip bits from the current alive sets; for END,
   check its condition.
4. Accept iff every step is valid and the derivation ends in an END.

**Cost:** at most `sum |D_i| <= 2 wmax r` ELIM steps, since each candidate qubit has at most 2 values. Each step costs
O(weight(g) + sum over j of |AL_j cap supp(g)|), using the ownership map. The worst case is O(wmax x sum |D_i|).

## 9. S1 (rule 6), for `|PK| = r - 1 = h`, `r >= 2`, rule set s0s1 or s0s1s2

### 9.1 Decision

The steps are evaluated in exactly this order, and the first step that decides ends S1.

1. **Recorded pass.** Compute `U = UNIT(PK, {}, none)` in recording mode (7.1: all first killers, all wiped
   variables, the recorded Z and N).
   - If U is not refuted, no prune.
   - If `sum |D_i| > 1024`, no prune.
2. **Base list B.**
   - If Z is nonempty: `B = (Z[0])`.
   - Otherwise: let bv be the wiped variable with the fewest values, lowest index on ties. B is the list of
     distinct first killers `fk(c)`, for c in `D_bv` in canonical order, keeping first occurrences.
3. **Qubit filter.** Let `Q*` be the undecided qubits q such that:
   - for every wiped variable i, q is in the support of `fk(c)` for some value c of i; and
   - q is in `supp(z)` for every recorded `z` in Z (at most 96 entries).

   A condition over an empty set is vacuous: with no wiped variable, only the Z condition applies. If `Q*` is empty,
   **prune**. This prune is taken **before** the limits of steps 4 and 5: it also applies when the killer total is at
   least 6144 or `h > 31`.
4. **All killers.** For every wiped variable i and every value c of i (in variable order, then canonical order), the
   list `K(c)` contains, in this order:
   - every generator g containing c's qubit, in column order, with `g != g_i`, `[c flips g] != s(g)`, and `F(g)` a
     subset of `{i}`;
   - then every generator of U's recorded need list N (at most 96 entries) whose variable is i and that c does not
     flip, in N's order.

   If the total length of all `K(c)` lists is **at least** 6144, no prune.
5. If `h > 31`, no prune.
6. **E enumeration.** Let:
   - `KM(i)` = the set `{fk(c) : c in D_i}` for each wiped variable i;
   - `ZS` = the set of recorded Z entries.

   For each g in B, in order; each support position of g, in order, whose qubit q is undecided; and each label
   `l = 1, 2, 3` in increasing order with `l` in `dom(q)` and `l != g_q`: skip `(q, l)` if already considered in this
   S1 decision. Otherwise let `E = (q, l)` and `EF = flips(q, l)`, and test, in order:
   1. if EF contains a packed generator: skip E;
   2. if, for some wiped i, `EF cap KM(i)` is empty: skip E;
   3. if ZS is not a subset of EF: skip E;
   4. if some g2 in EF has `s(g2) = 0` and `F(g2)` empty (unmodified, qe not excluded): skip E;
   5. if, for some wiped variable i, no value c of i has `K(c)` a subset of EF: skip E;
   6. otherwise compute `UNIT(PK, EF, q)`. If it is NOT refuted: **no prune**, and stop.
7. If no E stopped the enumeration: **prune**.

### 9.2 Justification J_S1

An S1 prune has exactly one justification form, chosen by the step of 9.1 that pruned:
- a **filter prune** (9.1 step 3) is justified by `J_S1F` (9.2.1, verified by 9.3.1);
- an **enumeration prune** (9.1 step 7) is justified by `J_S1E` (9.2.2, verified by 9.3.2).

In both forms, **J2** is a derivation refuting the unmodified parity system on PK, obtained from U as in 8.2 (from
Z[0] or from any wiped variable).

#### 9.2.1 Filter prunes: J_S1F

`J_S1F = (PK, J2, I, (K_i)_{i in I}, ZL)`:
- **I:** a list of distinct packed variables.
- **K_i:** for each i in I, a killer list with **exactly one entry per value of `D_i`**, in D's canonical order. Its
  j-th entry belongs to the j-th value of `D_i`.
- **ZL:** a list of generators.

**How a checker obtains J_S1F.** From U, the recorded pass of 9.1 step 1:
- I = the wiped variables, in increasing order;
- `K_i = (fk(c) for c in D_i)`. A column killer satisfies K1, and a need killer satisfies K2 (9.3);
- ZL = the recorded zero list Z.

The qubit set computed by VERIFY_S1F (9.3.1 step 4) is then exactly `Q*`, which is empty at a filter prune.

#### 9.2.2 Enumeration prunes: J_S1E

`J_S1E = (PK, J2, ANCHOR, REASONS)`:
- **ANCHOR:** either `("Z", z)` with `z = Z[0]`, or `("WIPED", bv, (k_c for c in D_bv))` with `k_c = fk(c)`. It is the
  same choice as the base list B. The WIPED list has exactly one entry per value of `D_bv`, in D's canonical order.
- **REASONS:** one reason for every action of `ANCH` (defined in 9.3.2), keyed by the action. The checker obtains it
  from the test that excluded the action. Every `KILL` list has exactly one entry per value of `D_i`, in D's
  canonical order:

| step | reason |
|---|---|
| step 6.1 | `PACKED(g)`, g a packed generator in EF |
| step 6.2 | `KILL(i, (fk(c) for c in D_i))` |
| step 6.3 | `ZGEN(z)`, z in ZS not in EF |
| step 6.4 | `ODD(g2)` |
| step 6.5 | `KILL(i, (k_c ...))`, with for each c a member of `K(c)` not in EF |
| step 6.6 | `DER(derivation)` for the modified system with EF and qe = q, obtained from `UNIT(PK, EF, q)` as in 8.2 |

### 9.3 VERIFY_S1

A **killer** of a value `c = (q, l)` of `D_i`, on the unmodified system, is a generator k satisfying one of:
- (K1) k contains q, `k != g_i`, `[c flips k] != s(k)`, and `F(k)` a subset of `{i}`;
- (K2) `s(k) = 1`, k is not packed, `F(k) = {i}`, and c does not flip k.

K2's "not packed" is redundant, since a packed k with `F(k) = {i}` is `g_i`, which every value of `D_i` flips. A
VERIFY may still check it.

A **Z-generator** is an unpacked generator z with `s(z) = 1` and `F(z)` empty. Check it by scanning supp(z) with
the ownership map.

#### 9.3.1 VERIFY_S1F (filter prunes)

1. VERIFY_PACKING(PK, r - 1).
2. Replay J2 on the unmodified parity system (8.3).
3. Check the lists:
   - the entries of I are distinct packed variables;
   - there is exactly one `K_i` per entry of I;
   - every `K_i` has **exactly `|D_i|` entries**, otherwise reject;
   - the j-th entry of `K_i` is a killer of the j-th value of `D_i`;
   - every entry of ZL is a Z-generator.
4. Compute `Q_J`: the undecided qubits q such that
   - for every i in I, q lies in supp(k) for some entry k of `K_i`; and
   - q lies in supp(z) for every z in ZL.

   A condition over an empty list is vacuous.
5. Accept iff all checks pass and `Q_J` is empty.

**The list lengths and alignment are load-bearing** (as for the WIPED anchor of 9.3.2). Soundness needs a verified
killer for whichever value a completion uses in `D_i`. With a shorter list, `Q_J` can miss the extra action's
qubit. Example: at the 5-qubit-code node `lab = [4, 0, 1, 2, 4]`, cap 4, with PK = [g1], an empty `K_0` gives an
empty `Q_J`, yet the completion `X_0 Z_4` exists.

**Cost:** J2 as for S0; `sum over i in I of |D_i|` killer checks, O(wmax) each; and one pass over the listed
supports. This is the same order as the decider's filter.

#### 9.3.2 VERIFY_S1E (enumeration prunes)

1. VERIFY_PACKING(PK, r - 1).
2. Replay J2 on the unmodified parity system (8.3).
3. Check ANCHOR:
   - **Z anchor:** z is a Z-generator.
   - **WIPED anchor:** the list has **exactly `|D_bv|` entries**. Otherwise reject. Its j-th entry `k_c` belongs to
     the j-th value c of `D_bv` in canonical order. Each `k_c` is a killer of c (K1 or K2, with bv in place of i).

     The list length is load-bearing. A shorter list would give a smaller ANCH that can miss a real extra action.
4. Compute `ANCH`:
   - Z anchor: all actions `(q, l)` with q undecided, l in `dom(q)` non-I, that flip z;
   - WIPED anchor: the union, over c in `D_bv`, of the actions that flip `k_c`.

   Every action of ANCH must have a reason, and reasons for other actions are ignored.
5. Check each reason for its action E:
   - **`PACKED(g)`:** g is packed and E flips g.
   - **`KILL(i, (k'_c)_c)`:** i is a packed variable, and the list has **exactly `|D_i|` entries**, otherwise reject.
     The j-th entry belongs to the j-th value c of `D_i` in canonical order. Each `k'_c` is a killer of c (K1 or K2,
     with i in place of bv), and E flips none of the `k'_c`.
   - **`ZGEN(z')`:** z' is a Z-generator, and E does not flip z'.
   - **`ODD(g2)`:** `s(g2) = 0`, E flips g2, and `F(g2)` is empty.
   - **`DER(d)`:** replay d on the modified system with `EF = flips(E)` and qe = E's qubit.
6. Accept iff all checks pass.

ANCH covers every possible extra action. An action E outside ANCH flips no listed anchor generator, so by the
killer lemma (11.4) it cannot be the extra element of a case-(b) completion: a Z anchor must be flipped by E, and
for a WIPED anchor E must flip `k_c` for `c = x_bv`. Hence VERIFY needs no check outside ANCH.

**Cost:**
- J2 is as for S0;
- ANCHOR: `|D_bv|` killer checks, O(wmax) each;
- ANCH: at most `|D_bv| x wmax x 3` actions;
- per reason: O(wmax) for PACKED, ZGEN and ODD; `|D_i|` killer checks for KILL; an S0-sized replay for DER.

The decider performs the same work.

## 10. S2 (rule 7), for `|PK| = r - 2 = h`, rule set s0s1s2

### 10.1 Decision (exact quotient membership)

1. If `n > 1365`, no prune. If some `D_i` is empty, no prune (this cannot occur at a live node). The bound
   `n <= 1365` is part of the rule definition: it is the largest n with `3n <= 4096`, the decider's action-table
   size.
2. **Base values and span.** Let `b_i` be the first value of `D_i`. Define:
   - `W = span{ sigma(c) XOR sigma(b_i) : c in D_i, all i }`, over GF(2)^m;
   - `tau = sigma(P_v) XOR XOR_i sigma(b_i)`, where `sigma(P_v)` is the active-generator indicator.
3. An **action** is `(q, l)` with q undecided and l in `dom(q)` non-I (on any undecided qubit, including packed
   candidates).
4. S2 **does not prune** iff at least one of the following holds:
   - `tau` is in W;
   - for some action a, `sigma(a) XOR tau` is in W;
   - for some actions a, a' on distinct qubits, `sigma(a) XOR sigma(a') XOR tau` is in W.
5. Otherwise S2 prunes. The decision is a set property, independent of evaluation order.

### 10.2 Justification J_S2: functionals vanishing on W

`J_S2 = (PK, kappa, EXC)`:
- `kappa`: a 64-bit word per generator, defining the 64 linear functionals `kappa(v) = XOR over g in v of kappa[g]`.
- `EXC`: a list of exceptions `(E, y)`. E is one of: empty, an action, or a pair of actions on distinct qubits. y is
  an m-bit vector.

**How a checker obtains J_S2.** Any kappa vanishing on W works. A convenient choice:
- reduce the difference vectors to an echelon basis;
- give every non-pivot generator a pseudo-random key;
- back-substitute the pivot keys so that every basis row has key 0.

E's that the decision rejected despite `kappa(sigma(E)) = kappa(tau)` (a fingerprint collision, about 2^-64 per
comparison) become exceptions. For each, the checker solves `y . b = 0` for all basis rows b and
`y . (sigma(E) XOR tau) = 1`; the decision guarantees such a y exists.

The keys affect neither the decision (it is exact, 10.1) nor the trace. A checker SHOULD derive them from a fixed,
documented seed, so that a VERIFY failure can be reproduced.

### 10.3 VERIFY_S2

1. VERIFY_PACKING(PK, r - 2). Rebuild `D_i`, `b_i`, `tau` and the action list from the definitions.
2. **Orthogonality:** for every i and every c in `D_i`: `kappa(sigma(c)) = kappa(sigma(b_i))`. For every exception
   `(E, y)`: `y . sigma(c) = y . sigma(b_i)` for every value, and `y . (sigma(E) XOR tau) = 1`.
3. **Refutation**, with `gamma = kappa(tau)`:
   - E empty: `gamma != 0`, or E is an exception;
   - every action a: `kappa(sigma(a)) != gamma`, or `(a)` is an exception;
   - every pair a, a' on distinct qubits with `kappa(sigma(a)) XOR kappa(sigma(a')) = gamma` is an exception. Find
     these pairs with a hash map from fingerprint to actions, in O(#actions) expected time.
4. Accept iff all checks pass.

**Cost:** O(sum |D_i| x cmax) for orthogonality, O(#actions x cmax) for the fingerprints, plus O(m) per exception. This is
the same order as the decision.

## 11. Lemmas and proofs

Throughout, v is a node, `r = cap - w`, and R is the undecided part of a zero-syndrome completion `L = P_v (x) R`
with `wt(R) <= r`. "Zero-syndrome" means L commutes with every generator. Hence, for every generator g, R
anticommutes with g iff `P_v` does, i.e. iff g is active (`s(g) = 1`).

### 11.1 Lemma P (packing)

Let `PK` be pairwise-disjoint-candidate active generators. Then `wt(R) >= |PK|`.

*Proof.* For each `g_i` in PK, R anticommutes with `g_i`. So an odd number (>= 1) of R's qubits in `supp(g_i)` carry
a label differing from `g_i,q`. Such a qubit is undecided and its label is in its domain, so it is a candidate of
`g_i`: it lies in `C(g_i)`. For a free qubit this is immediate. A restricted qubit `{I, b}` carries R's label b,
and `b != g_i,q`, which is exactly the candidate condition `lab[q] - 4 != g_i,q`. The sets `C(g_i)` are disjoint,
so these qubits are distinct. ∎

Consequently `|PK| > r` means no completion of weight <= cap exists. This justifies rule 2 (VERIFY_PACKING with
size r + 1).

### 11.2 Lemma S0 (tight packing)

If `|PK| = r`, then R has exactly one qubit in each `C(g_i)` and none elsewhere, and its label `x_i` there
anticommutes with `g_i,q`. So `x_i` is in `D_i`. Moreover, for every generator g,
`#{i : x_i flips g} = s(g) (mod 2)`.

*Proof.* By 11.1, R needs at least one qubit in each `C(g_i)`, and `wt(R) <= r = |PK|` forces exactly one in each
and none outside. In `C(g_i)`, R's single qubit must anticommute with `g_i` (odd count). Finally, R anticommutes
with g iff an odd number of R's qubits flip g, and R's qubits are exactly the `x_i`. ∎

Every solution of this "parity system" assigns `x_i` in `D_i`. The parity system of a modified variant (EF, qe)
adds `flips(E)` to the targets and forbids qe.

### 11.3 Lemma ELIM (derivations) and PROP

In a derivation step `ELIM(g, v)`, every solution `(x_j)` of the parity system with all `x_j` in the current alive
sets `AL_j` keeps `x_v` in `AL_v` after the step.

*Proof.* If some `AL_j` is empty, no such solution exists and the claim is vacuous; this is why an empty alive set
may be given contribution 0. Otherwise, for `j != v`, `x_j` is in `AL_j` and j is determined for g, so
`[x_j flips g]` equals j's contribution. Hence `[x_v flips g] = t(g) XOR p`, and the step only removes values with
the other flip bit. ∎

By induction over the steps, every solution stays inside the alive sets. END_WIPED leaves some variable with no
alive value, so no solution exists. END_ODD: every variable is determined for g, so every solution's parity at g is
p, which differs from `t(g)`. No solution exists. With Lemma S0, a verified derivation proves that no completion of
weight <= cap exists (S0). For S1 case (a) and case (b), see 11.4.

PROP's kills are ELIM steps, and its refutations are END steps. Whether PROP refutes does not depend on the order
in which generators are processed.

*Monotonicity.* Let A' be alive sets with `A' ⊆ A` pointwise, every variable nonempty in A'.
- A refutation "no mixed variable, parity != s(g)" enabled at A is also enabled at A'. Determined variables stay
  determined with the same contribution.
- Let a kill of choice x of v via g be enabled at A, with x alive in A'. Then at A' either the same kill is
  enabled (v still mixed), or v is determined with x's (wrong) flip bit, and the refutation at g is enabled. The
  case "v determined with the correct parity" cannot occur, since x is alive.

*Order independence.* Let F be the alive sets at the end of some non-refuting maximal sequence; no kill or
refutation is enabled at F. Take any other sequence. By induction it never kills a choice of F: its current state
contains F, and a kill of some x in F would, by monotonicity, enable a kill or a refutation at F. For the same
reason it never refutes; an emptied variable would require killing a choice of F. So it also ends without
refutation, at alive sets containing F, and by symmetry at F. Hence either every order refutes or none does, and
every non-refuting order reaches the same fixpoint.

### 11.4 Lemma S1 and the killer lemma

Let `|PK| = h = r - 1`. Then either:
- **(a)** R is a solution of the parity system on PK (weight h); or
- **(b)** `R = {x_0..x_(h-1)} U {E}` with `x_i` in `D_i`, and E an action (undecided qubit, label in its domain,
  non-I) whose qubit is used by no `x_i`, such that:
  - E flips no packed generator;
  - for every generator g, `#{i : x_i flips g} + [E flips g] = s(g) (mod 2)`.

*Proof.* By 11.1, R has at least one qubit in each `C(g_i)`, and `wt(R) <= h + 1` leaves at most one more qubit E.
- If there is none, then (a) holds: the S0 argument with h in place of r.
- If E lies outside every `C(g_i)`, it flips no packed generator, since only candidates can flip.
- If E is a second qubit in some `C(g_j)`, then exactly one of the two qubits there anticommutes with `g_j` (odd
  count). Call that one `x_j`; E does not flip `g_j`.

The parity follows as in 11.2. ∎

**Killers.** Use the unmodified parity system and original domains. A generator k is a killer of a value
`c = (q, l)` of variable i if:
- (K1) k contains q, `k != g_i`, `[c flips k] != s(k)`, and `F(k)` is a subset of `{i}`; or
- (K2) `s(k) = 1`, k unpacked, `F(k) = {i}`, and c does not flip k.

A Z-generator is an unpacked k with `s(k) = 1` and `F(k)` empty.

**Killer lemma.** In a case-(b) completion with `x_i = c`, E flips every killer of c and every Z-generator.

*Proof.* At k, the parity is `sum_j [x_j flips k] + [E flips k] = s(k)`. Every `x_j` lies in `D_j`, so `x_j` can
flip k only if j is in `F(k)`.
- (K1): `F(k)` is a subset of `{i}`, so the sum without E is `[c flips k] != s(k)`; hence E flips k.
- (K2): only `x_i = c` could flip k, and it does not, while `s(k) = 1`; hence E flips k.
- Z-generator: no `x_j` flips k and `s(k) = 1`; hence E flips k. ∎

**Reasons.** Each VERIFY_S1E reason excludes its action E from case (b):
- `PACKED`: contradicts "E flips no packed generator".
- `KILL(i, ...)`: whichever value `x_i = c` is used, its listed killer is not flipped by E. This contradicts the
  killer lemma.
- `ZGEN`: contradicts the killer lemma.
- `ODD(g2)`: `s(g2) = 0`, E flips g2, and no `x_j` can flip g2, so the parity at g2 is 1, not 0.
- `DER`: a verified derivation on the modified system (targets `s XOR flips(E)`, values at E's qubit forbidden)
  excludes every case-(b) completion with this E, by 11.3 applied to the system stated in 11.4(b).

**Lemma S1-J (enumeration form).** If VERIFY_S1E accepts, no completion of weight <= cap exists.
- J2 refutes (a).
- Any case-(b) E must flip the Z anchor, or the anchor killer `k_c` of `c = x_bv`, by the killer lemma. Hence E is
  in ANCH, and its reason excludes it. ∎

**Lemma S1-F (filter form).** If VERIFY_S1F accepts, no completion of weight <= cap exists.

*Proof.* Suppose a completion R of weight <= cap exists. By Lemma S1, R is of case (a) or case (b).
- Case (a): R is a solution of the parity system on PK. J2 is a verified derivation, so by Lemma ELIM (11.3) no
  such solution exists.
- Case (b): `R = {x_0, ..., x_(h-1)} U {E}`, with `x_i` in `D_i` and E an action on an undecided qubit `q_E`.
  - For each i in I: let `c = x_i`. This is some value of `D_i`. Because `K_i` has exactly one entry per value of
    `D_i` and each entry was verified as a killer of its own value, the entry `k` for this c is a killer of c. By
    the killer lemma, E flips k. Flipping requires `q_E in supp(k)`, so `q_E` satisfies the condition of 9.3.1
    step 4 for i.
  - For each z in ZL: z is a Z-generator, so by the killer lemma E flips z, and `q_E in supp(z)`.
  - Hence `q_E` is an undecided qubit satisfying every condition of 9.3.1 step 4, i.e. `q_E` is in `Q_J`. This
    contradicts `Q_J` being empty.

So no completion exists. ∎

The proof uses the exact length and alignment of every `K_i`: the killer of `x_i` must be present whatever value
`x_i` takes. Any list I of packed variables and any list ZL of Z-generators give a sound justification. Smaller
lists only enlarge `Q_J`.

### 11.5 Lemma S2 and Lemma S2-J

Let `|PK| = h = r - 2`.

**Lemma S2.** There are representatives `x_i` in `D_i` and a set E of 0, 1 or 2 actions on distinct undecided
qubits such that `XOR_i sigma(x_i) XOR sigma(E) = sigma(P_v)`. Consequently `sigma(E) XOR tau` is in W.

*Proof.*
- **Decomposition.** By 11.1, each `C(g_i)` contains an odd number of R's qubits anticommuting with `g_i`. Choose
  one of them, with its label, as `x_i`. The choices lie in disjoint sets, so they are distinct and in `D_i`. The
  rest, E, has at most `wt(R) - h <= 2` actions on distinct qubits other than the `x_i`'s.
- **Syndrome.** R anticommutes exactly with the active generators, so `XOR sigma(R) = sigma(P_v)`.
- **Membership.** `XOR_i sigma(x_i) XOR XOR_i sigma(b_i)` is in W, so `sigma(E) XOR tau` is in W. ∎

The decision of 10.1 tests membership exactly: E empty, every action, every pair on distinct qubits. A prune
therefore implies no completion of weight <= cap.

**Lemma S2-J.** If VERIFY_S2 accepts, no completion of weight <= cap exists.
- Orthogonality makes kappa and each exception functional y vanish on W.
- If `sigma(E) XOR tau` is in W, every such functional takes equal values on `sigma(E)` and `tau`. But VERIFY
  requires, for every E, either a kappa mismatch or an exception y with `y . (sigma(E) XOR tau) = 1`.
- So no E satisfies Lemma S2. ∎

### 11.6 Soundness of the checker's acceptance

A checker that runs VERIFY at every prune it takes accepts a prune only if the corresponding lemma's hypotheses are
verified. By section 4, the accepted search misses no minimum logical. The deciding functions influence only which
prunes are attempted, and therefore the node counts and traces. Implementation errors there lead to mismatches,
not to unsound acceptance.

This argument assumes a correct trusted base (introduction): parsing, hashes, permutations and orbits, roots, node
state and deadness, LEAF with its logical test, WEIGHT, BRANCH with its child enumeration and dead-suffix cuts, and
frontier/task replay with the count and trace bookkeeping. This is the same base that version 1 trusts. These parts
have no VERIFY and MUST be implemented exactly as specified.

## 12. Trace, records, hints

**State fingerprint and recurrence** (as version 1). All arithmetic is unsigned 64-bit:

```
mix64(x): x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9; x = (x ^ (x >> 27)) * 0x94d049bb133111eb; return x ^ (x >> 31)
salt(q, a) = mix64(8*q + a + 1)
state_fp(lab) = XOR over q of salt(q, lab[q])
B = 0x00000100000001b3
```

**Words per entry.** Every counted entry contributes two words: its entry-state fingerprint, then

`packed = w | (len(A) << 8) | (rule << 24) | (g_field << 32) | (h_field << 48)`

| rule | g_field | h_field |
|---|---|---|
| 0 (LEAF), 1 (WEIGHT) | 65535 | 0 |
| 2 (PACKING) under `greedy` | 65535 | the selected size (version 1) |
| 2 (PACKING) under the other rule sets | 65535 | `|PK| = r + 1` |
| 3 (BRANCH) | the branch generator | 0 |
| 5, 6, 7 (S0, S1, S2) | 65535 | `|PK|` = r, r - 1, r - 2 respectively |

**Recurrence.** Start each task trace at `H = 0`; for each word v, `H = H*B + v`.
- Dead children, dead suffixes, replayed ancestors and frontier markers contribute nothing.
- Composition: `count(A||C) = count(A) + count(C)`, `H(A||C) = H(A) * B^(2 count(C)) + H(C)`.

**Records.** These are the version-1 binary records, with the following changes:
- **ENTER (tag 0x10):** `parent_ref:U edge:U rule:u8 state_fp:u64 w:U active_count:U`, then:
  - rule 2: `packing_size:U` and that many generator indices in PK order. Under `greedy`, the version-1
    `selected` list.
  - rule 3: `generator:U`.
  - rules 5, 6, 7: `packing_size:U` and the generator indices of PK in PK order.
  - rules 0 and 1: nothing.
- Rules 5, 6 and 7 are permitted only under rule sets that contain them.
- The checker recomputes PK at every top-tree prune and rejects unless it is identical to the record. This is the
  only hint in the certificate; justifications J are never stored.
- The other records (DEAD_CHILD_OR_ROOT 0x11, DEAD_SUFFIX 0x12, FRONTIER 0x13, TOP_END 0x1f, TASK_RESULT 0x20,
  COMPLETE 0x7f, TASK_BEGIN 0x21, TASK_END 0x22) are the version-1 records, restated in section 14.
- Inside FULL-mode task transcripts, ENTER records carry the same fields and the same recomputation duty.

## 13. Limits and give-up behaviour

| constant | value | effect when exceeded |
|---|---|---|
| greedy list | first 96 active generators | longer A is truncated (sound: any subset gives a valid packing) |
| MIS_THRESHOLD | `R <= 64` | otherwise not run (PK = greedy list) |
| UNIT zero list / need list | 96 entries recorded each | only the records are truncated. The refutation counts every zero generator (7.1). A truncated need list gives fewer need killers: weaker, sound. S1 uses the recorded Z (Q*, ZS, step 6.3) |
| PROP choices / local generators / occurrences | > 1024 / > 2048 / > 6144 | PROP gives up: no S0 prune from PROP |
| S1 recorded values | `sum |D_i| > 1024` | no S1 prune |
| S1 all-killer total | >= 6144 | no S1 prune, except the filter prune of 9.1 step 3, which is evaluated first |
| S1 variables | `h > 31` | no S1 prune, except the filter prune of 9.1 step 3, which is evaluated first |
| S2 qubits | `n > 1365` | no S2 prune |

A give-up never prunes. Every limit is evaluated from the node state alone, so the decision stays a pure function.
The comparisons are exactly as written: S1's killer total gives up at **>= 6144**, while PROP's occurrence total
gives up at **> 6144**. The difference is intended and normative.
VERIFY procedures have no limits of their own: their inputs are bounded by the same constants.

## 14. Split, frontier, tasks, refinement, container, records (as version 1)

**Split.** `split` is a support weight; a root has weight 1.
- For `split = 1`, every declared root is a frontier task before its deadness test. A dead root is a zero-node task.
- For `split > 1`: test root deadness, then walk the top tree. Cut every nondead child of weight `split` before
  entry, and emit a FRONTIER record.

**Tasks.** Task IDs are contiguous from 0 in frontier appearance order. A task path is the full decision sequence
from its root.
- Replay reconstructs domains under the immutable cap. Every decision must match the recomputed branch generator,
  candidate and option.
- `gen_nodes` counts top-tree entries, and `task_nodes` the sum of task counts.

**Refinement.** Internal budget re-splitting must compose segment summaries in DFS order (section 12 composition).
The certificate records only the canonical frontier.

**Container.** 8 bytes `SBCOV2\r\n`, a little-endian u64 header length, then that many UTF-8 bytes of one JSON
object, the header (section 0), then the records.
- `U` is canonical unsigned LEB128: least-significant groups of seven bits first, high bit = continuation. Reject
  nonminimal encodings, encodings longer than 10 bytes, and a tenth byte greater than 1.
- `u64` is eight little-endian bytes. There is no alignment or padding.
- **Gzip transport (normative for version 2).** Writers produce exactly one gzip member with mtime 0 and no
  original file name. Readers detect gzip by its magic bytes, never by file extension. A version-2 reader MUST
  reject:
  - any byte after the end of the first gzip member, including further members and zero padding;
  - a member whose CRC-32 or length trailer does not match;
  - any logical byte after COMPLETE.

  Readers need not check the other gzip header fields (mtime, name, OS, extra flags). Version-1 artifacts keep the
  version-1 transport rules unchanged.
- **Byte identity.** Producers with equal instance, cap, split, permutations, detail and rules produce identical
  record bytes. To compare two artifacts, compare their headers without `provenance`, plus the complete
  uncompressed record stream. Under `greedy`, CPU and GPU producers must agree in this sense. The rule sets s0,
  s0s1 and s0s1s2 have CPU producers only.

**Record stream.** Records are a one-byte tag followed by the fields below.
- Top events have implicit IDs 0, 1, ... in record order, counting every record type of the top section.
- `parent_ref = 0` denotes a root; its `edge` is `root_ordinal`. Otherwise `parent_ref = parent_event_id + 1`, and
  it must name an earlier BRANCH entry. The edge is then the encoded decision `(p << 1) | li`.
- The checker checks parent and edge against its own DFS walk. Naming a valid earlier parent is not enough to
  reorder or omit children.

| tag | name | fields |
|---|---|---|
| 0x10 | ENTER | as section 12 |
| 0x11 | DEAD_CHILD_OR_ROOT | `parent_ref:U edge:U dead_generator:U state_fp:u64` |
| 0x12 | DEAD_SUFFIX | `parent_ref:U position:U dead_generator:U state_fp:u64` |
| 0x13 | FRONTIER | `parent_ref:U edge:U task_id:U state_fp:u64` |
| 0x1f | TOP_END | `event_count:U task_count:U gen_nodes:U top_trace:u64` |
| 0x20 | TASK_RESULT | `task_id:U root_ordinal:U path_len:U path_bytes[path_len] node_count:U trace:u64` |
| 0x7f | COMPLETE | `task_count:U gen_nodes:U task_nodes:U total_nodes:U global_trace:u64` |

- `dead_generator` is the dead hint of section 3: the least active generator with no candidate. `state_fp` is the
  fingerprint of the dead state. Every supplied fingerprint and hint is recomputed.
- **DEAD_SUFFIX** always has a nonzero parent_ref. Its `position` is the candidate position p whose commuting
  restriction (section 3, child order step 3) made the working state dead. Its `state_fp` is that of the state
  after the restriction. It terminates the unvisited suffix of that parent's children. It is not a child and does
  not count as a node.
- The checker drives the deterministic top traversal and consumes exactly one expected record per event. It
  rejects a missing, extra, out-of-order or wrong-rule record, an invalid parent, a mismatching hint, or an
  unjustified frontier cut. At TOP_END every root must be completely covered, and its counts and top trace must
  equal the reconstructed values.
- Task results appear in increasing ID order: exactly `task_count` of them, including zero-node dead-root tasks at
  `split = 1`. IDs, root and complete path must match the corresponding frontier. The checker executes every task
  subtree, rejects any logical leaf, and compares count and trace. It may parallelise, but it must check every ID.
  Sampling cannot yield an accepted lower bound.
- COMPLETE: every field must match the independently accumulated values, and `total_nodes = gen_nodes +
  task_nodes`. The global trace inserts each task's entry stream at its frontier position in the top walk, using
  the composition rule of section 12. The stream ends immediately after COMPLETE. A missing footer means an
  incomplete artifact, not a weaker proof.

**FULL mode** (`detail = "full"`). The top section is unchanged. For each task, in canonical task order:
1. `TASK_BEGIN` (tag 0x21) with `task_id:U`.
2. The task's entire uncut subtree, as ENTER, DEAD_CHILD_OR_ROOT and DEAD_SUFFIX records.
   - Event IDs and parent references are local to the task, starting at 0.
   - The task root uses `parent_ref 0` and `edge 0`. It is the reconstructed frontier state, not a newly built
     global root.
   - No FRONTIER record is allowed inside a transcript.
3. `TASK_END` (tag 0x22) with `event_count:U`.
4. Its normal TASK_RESULT record.

A dead `split = 1` task has one local DEAD_CHILD_OR_ROOT record and zero entered nodes.

**Producer requirements** (normative for exporters; a checker cannot observe them):
- A producer stages the stream at a temporary path. It writes COMPLETE only after every task has resolved and every
  replay check has passed, then finishes compression, then publishes the final path atomically.
- Interruption or any failure leaves no apparently complete final file.
- If a logical is found, no lower-bound artifact is published.
- Certificate emission is incompatible with stop-at-first-logical runs and with count-only checkpoint/resume runs.

## 15. Checker acceptance checklist

1. Parse the container strictly. Validate format, search, trace, rules and rules_version.
2. Hash and parse the instance. Validate dimensions, isotropy, the logical basis, the group hash, the domain and the
   immutable cap.
3. Verify the permutations and the exact orbit partition.
4. Walk the top tree in deterministic DFS order. At every entry, decide with the rule set and compare the record,
   including the recomputed PK hint. **At every prune, compute J and run its VERIFY.**
5. Execute every task, deciding and **verifying at every prune**. Reject any nontrivial logical, any count or trace
   mismatch, and any missing or extra task.
6. Validate the footer and reject trailing bytes. Only then accept "no nontrivial <domain> logical of weight
   <= cap". A consumer then binds this to its question (section 0, consumer binding).

The checker MUST NOT reuse the prover's implementation. VERIFY MUST be implemented separately from the deciding
code.

## 16. Conformance vectors (to be supplied as data with revision 2.0)

Planned:
- **Per-node vectors.** Each gives the state, the decision, PK, and the spec-defined intermediate quantities: the
  UNIT records, the wiped variables, Z, the S1 stopping step, and the S2 rank of W and outcome. They cover:
  - every rule and every give-up limit;
  - S2 at n = 1365 and 1366;
  - S1 at h = 32, both with `Q*` empty (filter prune, 9.1 step 3) and with `Q*` nonempty (no prune, step 5);
  - more than 96 active generators, where "truncate, then order" and "order, then truncate" (5.1, 6.1) give
    different packings;
  - a graph on which np > r comes from SEARCH (6.1 (c));
  - planted-completion nodes (no prune allowed).
- **VERIFY vectors**, as (node, J, expected verdict):
  - the 5-qubit-code node `lab = [4, 0, 1, 2, 4]`, cap 4 (generators XZZXI, IXZZX, XIXZZ, ZXIXZ; PK = [g1]). It has
    a weight-4 completion `X_0 Z_4`. A WIPED anchor or KILL list with fewer than `|D_i|` entries MUST be rejected
    (9.3.2), and so MUST a `J_S1F` whose `K_0` is empty (9.3.1);
  - a valid `J_S1F` at a filter prune (accept); and `J_S1F` lists containing an entry that is not a killer of its
    value, or a shifted (misaligned) list (reject);
  - the 5-qubit-code node `lab = [0, 1, 0, 4, 1]`, cap 4, PK = [g0], E = (3, X). The modified `D_0` is empty, and
    `DER = END_ODD(g3)` MUST be accepted (empty alive set, contribution 0, 8.2).
- The [[4,2,2]] example of version 1, extended to every rule set.
- A byte-identity vector: one instance emitted as version 1 and as version-2 `greedy`, with equal record streams.
- Adversarial artifacts: edited PK hints, a wrong rules id, cap or domain edits, dropped tasks.
