"""Trusted core: stabilizer codes as GF(2) subspaces of Pauli vectors.

Everything an engine reports (witnesses, automorphisms) is re-checked here, and
every lower-bound engine takes its code data (generators, logical basis) from
here. Keep this module small, dependency-free and easy to audit.

Conventions (fixed; certificates depend on them):

* A Pauli on ``n`` qubits, phases ignored, is a Python ``int`` of ``2n`` bits:
  bit ``q`` is ``x_q`` and bit ``n + q`` is ``z_q`` (X = x, Z = z, Y = x and z).
* Two Paulis commute iff their symplectic product
  ``x.z' + z.x'`` is 0 mod 2, computed as ``popcount(u & swap(v)) & 1``.
* Pauli weight = number of qubits with a non-identity label (a Y counts once).
* The stabilizer group ``S`` is the span of the supplied generators (which may
  be dependent); ``N(S)`` is its symplectic complement (the Paulis commuting
  with every generator); the distance is ``min wt(P), P in N(S) minus S``.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

__all__ = [
    "CodeError", "GF2Basis", "StabilizerCode", "weight", "symplectic", "swap_halves",
    "pauli_from_xz", "xz_from_pauli", "kernel",
]


class CodeError(ValueError):
    """Malformed input or a code that does not satisfy the stabilizer axioms."""


def popcount(v: int) -> int:
    return v.bit_count()


def swap_halves(v: int, n: int) -> int:
    mask = (1 << n) - 1
    return ((v & mask) << n) | (v >> n)


def symplectic(u: int, v: int, n: int) -> int:
    """Symplectic product of two Paulis (0 = commute, 1 = anticommute)."""
    return popcount(u & swap_halves(v, n)) & 1


def weight(v: int, n: int) -> int:
    """Pauli weight: qubits where x or z is set."""
    return popcount((v | (v >> n)) & ((1 << n) - 1))


def support(v: int, n: int) -> list[int]:
    s = (v | (v >> n)) & ((1 << n) - 1)
    return [q for q in range(n) if (s >> q) & 1]


def _index_list(lst, n: int, what: str) -> list[int]:
    if not isinstance(lst, (list, tuple)):
        raise CodeError(f"{what}: expected a list of qubit indices")
    out = []
    for q in lst:
        # bool is an int subclass; reject it (and floats, strings) explicitly
        if type(q) is not int or q < 0 or q >= n:
            raise CodeError(f"{what}: qubit index {q!r} not an int in [0, {n})")
        out.append(q)
    if len(set(out)) != len(out):
        raise CodeError(f"{what}: repeated qubit index")
    return out


def pauli_from_xz(x: Iterable[int], z: Iterable[int], n: int, what: str = "operator") -> int:
    """Pauli from its X-support and Z-support lists (a qubit in both is a Y)."""
    v = 0
    for q in _index_list(list(x), n, what + " X"):
        v |= 1 << q
    for q in _index_list(list(z), n, what + " Z"):
        v |= 1 << (n + q)
    return v


def xz_from_pauli(v: int, n: int) -> dict:
    """``{"X": [...], "Z": [...]}`` form of a Pauli (sorted supports)."""
    return {"X": [q for q in range(n) if (v >> q) & 1],
            "Z": [q for q in range(n) if (v >> (n + q)) & 1]}


class GF2Basis:
    """Echelon basis of a GF(2) subspace of ints; pivot = highest set bit.

    ``rows[p]`` has highest bit ``p``. Rows are kept fully reduced (no row has a
    bit at another row's pivot), so the representation of a subspace is unique.
    """

    def __init__(self, rows: Iterable[int] = ()):
        self.rows: dict[int, int] = {}
        self._pivmask = 0
        for v in rows:
            self.add(v)

    def __len__(self) -> int:
        return len(self.rows)

    def reduce(self, v: int) -> int:
        """Reduce v modulo the span; 0 iff v is in the span.

        Rows are fully reduced, so XORing row p never changes v at another pivot:
        the pivots to apply are exactly the pivot bits set in the input."""
        hits = v & self._pivmask
        while hits:
            low = hits & -hits
            v ^= self.rows[low.bit_length() - 1]
            hits ^= low
        return v

    def contains(self, v: int) -> bool:
        return self.reduce(v) == 0

    def add(self, v: int) -> bool:
        """Add v to the span. Returns True iff the dimension grew."""
        v = self.reduce(v)
        if not v:
            return False
        p = v.bit_length() - 1
        # keep full reduction: clear bit p from every other row
        for q, r in self.rows.items():
            if (r >> p) & 1:
                self.rows[q] = r ^ v
        self.rows[p] = v
        self._pivmask |= 1 << p
        return True

    def canonical(self) -> list[int]:
        """The reduced basis, sorted by pivot (unique for the subspace)."""
        return [self.rows[p] for p in sorted(self.rows)]

    def kernel(self, width: int) -> list[int]:
        """Basis of {v in GF(2)^width : popcount(v & r) even for every row r}."""
        out = []
        for j in range(width):
            if j in self.rows:
                continue
            # free column j: v = e_j + sum of pivots p whose (reduced) row has bit j
            v = 1 << j
            for p, r in self.rows.items():
                if (r >> j) & 1:
                    v |= 1 << p
            out.append(v)
        return out


def kernel(functionals: Sequence[int], width: int) -> list[int]:
    """Basis of the common kernel of the linear functionals v -> popcount(v & f) mod 2."""
    return GF2Basis(functionals).kernel(width)


class StabilizerCode:
    """A stabilizer code given by generator Paulis (ints, see module doc).

    Construction validates: ``n >= 1``, every generator is a ``2n``-bit int,
    every pair of generators commutes. ``k = n - rank``. A logical basis of
    ``N(S)/S`` (2k operators) is computed and audited.
    """

    def __init__(self, n: int, generators: Sequence[int], *, name: str | None = None):
        if type(n) is not int or n < 1:
            raise CodeError(f"n must be a positive int, got {n!r}")
        gens = list(generators)
        for i, g in enumerate(gens):
            if type(g) is not int or g < 0 or g >> (2 * n):
                raise CodeError(f"generator {i} is not a {2 * n}-bit Pauli vector")
        self.n = n
        self.name = name
        self.generators: list[int] = gens
        # isotropy: every pair commutes (checked on the given rows, which span S)
        sw = [swap_halves(g, n) for g in gens]
        for i, g in enumerate(gens):
            for j in range(i + 1, len(gens)):
                if popcount(g & sw[j]) & 1:
                    raise CodeError(f"generators {i} and {j} anticommute")
        self._span = GF2Basis(gens)
        self.rank = len(self._span)
        self.k = n - self.rank
        self._swapped = sw
        self._logicals: list[int] | None = None

    # -- constructors -------------------------------------------------------
    @classmethod
    def from_xz(cls, rows: Sequence[dict], n: int, **kw) -> "StabilizerCode":
        """Generators as ``[{"X": [...], "Z": [...]}, ...]`` (challenge stabilizer form)."""
        if not isinstance(rows, (list, tuple)):
            raise CodeError("generators: expected a list")
        gens = []
        for i, r in enumerate(rows):
            if not isinstance(r, dict) or set(r) - {"X", "Z"}:
                raise CodeError(f"generator {i}: expected {{'X': [...], 'Z': [...]}}")
            gens.append(pauli_from_xz(r.get("X", []), r.get("Z", []), n, f"generator {i}"))
        return cls(n, gens, **kw)

    @classmethod
    def css(cls, hx: Sequence[Sequence[int]], hz: Sequence[Sequence[int]], n: int, **kw) -> "StabilizerCode":
        """CSS code from X-check and Z-check supports (lists of qubit indices)."""
        gens = [pauli_from_xz(r, [], n, f"X check {i}") for i, r in enumerate(hx)]
        gens += [pauli_from_xz([], r, n, f"Z check {i}") for i, r in enumerate(hz)]
        return cls(n, gens, **kw)

    # -- basic properties ---------------------------------------------------
    @property
    def is_css(self) -> bool:
        """True iff every *given* generator is pure X or pure Z."""
        mask = (1 << self.n) - 1
        return all(not (g & mask) or not (g >> self.n) for g in self.generators)

    def weight(self, v: int) -> int:
        return weight(v, self.n)

    def in_stabilizer(self, v: int) -> bool:
        return self._span.contains(v)

    def commutes_with_stabilizer(self, v: int) -> bool:
        return not any(popcount(v & s) & 1 for s in self._swapped)

    def is_logical(self, v: int) -> bool:
        """True iff v is a nontrivial logical: in N(S) and not in S."""
        if type(v) is not int or v <= 0 or v >> (2 * self.n):
            return False
        return self.commutes_with_stabilizer(v) and not self.in_stabilizer(v)

    def witness_problem(self, v: int) -> str | None:
        """None if v is a nontrivial logical, else "malformed", "not_commuting" or
        "inside_stabilizer" (which includes the identity: it is an element of S)."""
        if type(v) is not int or v < 0 or v >> (2 * self.n):
            return "malformed"
        if v == 0:
            return "inside_stabilizer"
        if not self.commutes_with_stabilizer(v):
            return "not_commuting"
        if self.in_stabilizer(v):
            return "inside_stabilizer"
        return None

    def check_witness(self, v: int) -> int:
        """Weight of v if it is a nontrivial logical, else CodeError."""
        if not self.is_logical(v):
            raise CodeError("operator is not a nontrivial logical (outside N(S) or inside S)")
        return self.weight(v)

    def require_logicals(self) -> None:
        if self.k < 1:
            raise CodeError(f"k = {self.k}: no logical qubits, distance undefined")

    # -- normalizer and logical basis ---------------------------------------
    def normalizer_basis(self) -> list[int]:
        """Basis of N(S) (dimension n + k)."""
        return kernel(self._swapped, 2 * self.n)

    @property
    def logicals(self) -> list[int]:
        """2k Paulis in N(S), independent modulo S (a basis of N(S)/S), audited."""
        if self._logicals is None:
            self._logicals = self._compute_logicals()
        return self._logicals

    def _compute_logicals(self) -> list[int]:
        n, k = self.n, self.k
        K = self.normalizer_basis()
        if len(K) != n + k:
            raise AssertionError("normalizer dimension != n + k")
        span = GF2Basis(self._span.canonical())
        # prefer light representatives (smaller CNF / instance files); any choice is valid
        L = [v for v in sorted(K, key=lambda v: (weight(v, n), v)) if span.add(v)]
        if len(L) != 2 * k:
            raise AssertionError("logical basis size != 2k")
        if any(not self.commutes_with_stabilizer(v) for v in L):
            raise AssertionError("logical representative outside N(S)")
        # the symplectic pairing on N(S)/S is nondegenerate: its Gram matrix has full rank
        gram = [sum(symplectic(a, b, n) << j for j, b in enumerate(L)) for a in L]
        if len(GF2Basis(gram)) != 2 * k:
            raise AssertionError("degenerate logical pairing matrix")
        return L

    def is_nontrivial_via_logicals(self, v: int) -> bool:
        """For v in N(S): v not in S iff v anticommutes with some logical basis element."""
        return any(symplectic(v, l, self.n) for l in self.logicals)

    # -- restricted ("side") questions ----------------------------------------
    def x_part_mask(self) -> int:
        return (1 << self.n) - 1

    def z_part_mask(self) -> int:
        return ((1 << self.n) - 1) << self.n

    # -- identity -------------------------------------------------------------
    def canonical_rows(self) -> list[int]:
        return self._span.canonical()

    def group_hash(self) -> str:
        """sha256 of the canonical (fully reduced echelon) basis of S.

        Equal for any generating set of the same group (same qubit labels)."""
        h = hashlib.sha256()
        h.update(f"stabcert-group/1 n={self.n} rank={self.rank}\n".encode())
        width = (2 * self.n + 7) // 8
        for r in self.canonical_rows():
            h.update(r.to_bytes(width, "little"))
        return h.hexdigest()

    def to_xz(self) -> list[dict]:
        return [xz_from_pauli(g, self.n) for g in self.generators]

    # -- symmetry ---------------------------------------------------------------
    def permute(self, v: int, perm: Sequence[int]) -> int:
        """Image of Pauli v under the qubit permutation q -> perm[q]."""
        n = self.n
        out = 0
        for q in range(n):
            if (v >> q) & 1:
                out |= 1 << perm[q]
            if (v >> (n + q)) & 1:
                out |= 1 << (n + perm[q])
        return out

    def is_automorphism(self, perm: Sequence[int]) -> bool:
        """True iff the qubit permutation maps S onto S.

        perm must be a bijection of range(n); it suffices that every generator's
        image lies in S (a bijection preserves dimension, so pi(S) = S)."""
        n = self.n
        if len(perm) != n or any(type(p) is not int for p in perm) or sorted(perm) != list(range(n)):
            return False
        return all(self._span.contains(self.permute(g, perm)) for g in self.generators)

    def __repr__(self) -> str:
        nm = f" {self.name!r}" if self.name else ""
        return f"<StabilizerCode{nm} n={self.n} k={self.k} generators={len(self.generators)}>"
