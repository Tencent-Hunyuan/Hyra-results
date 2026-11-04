"""Exhaustive search for small codes: every Pauli of weight w = 1, 2, ... in turn.

A Pauli P is a nontrivial logical iff it commutes with every generator and
anticommutes with some element of the logical basis (core.StabilizerCode docs).
Each single-qubit Pauli (q, l) gets a combined syndrome bit vector
(commutation with the generators | with the logical basis); P is a nontrivial
logical iff the XOR of its single-qubit vectors has zero generator part and a
nonzero logical part. The search enumerates supports in lexicographic order
with all 3^w label choices, so it is exhaustive by construction. Work grows as
sum_w C(n, w) 3^w, so this is for small codes and cross-checks only.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb

from ..core import CodeError, StabilizerCode, swap_halves

__all__ = ["BruteResult", "brute_force", "brute_work"]

# labels: 1 = X (x bit), 2 = Z (z bit), 3 = Y (both)
_DOMAINS = {"P": (1, 2, 3), "X": (1,), "Z": (2,)}


@dataclass
class BruteResult:
    max_weight: int          # every Pauli of weight <= max_weight (in the domain) was examined
    found: int | None        # weight of the lightest nontrivial logical found (None: none <= max_weight)
    witness: int | None
    examined: int


def brute_work(n: int, max_weight: int, domain: str = "P") -> int:
    a = len(_DOMAINS[domain])
    return sum(comb(n, w) * a ** w for w in range(1, max_weight + 1))


def brute_force(code: StabilizerCode, max_weight: int, *, domain: str = "P",
                work_limit: int = 50_000_000, deadline: float | None = None) -> BruteResult:
    """Find the lightest nontrivial logical of weight <= max_weight, or prove none exists.

    Stops at the first weight w with a nontrivial logical (so ``found`` is the
    minimum over the domain). Raises CodeError if the work estimate exceeds the limit.
    deadline (time.monotonic() value): stop early; the result then covers only the
    weights that were completed (``max_weight`` is lowered accordingly)."""
    import time
    code.require_logicals()
    n = code.n
    labels = _DOMAINS[domain]
    max_weight = min(max_weight, n)
    if brute_work(n, max_weight, domain) > work_limit:
        raise CodeError(f"brute force too large: {brute_work(n, max_weight, domain)} > {work_limit}")
    gens = code.generators
    L = code.logicals
    m = len(gens)
    gsw = [swap_halves(g, n) for g in gens]
    lsw = [swap_halves(l, n) for l in L]
    vec = {}
    for q in range(n):
        for lab in labels:
            p = ((lab & 1) << q) | (((lab >> 1) & 1) << (n + q))
            s = 0
            for i, g in enumerate(gsw):
                if (p & g).bit_count() & 1:
                    s |= 1 << i
            for j, l in enumerate(lsw):
                if (p & l).bit_count() & 1:
                    s |= 1 << (m + j)
            vec[q, lab] = (s, p)
    gmask = (1 << m) - 1
    examined = 0

    for w in range(1, max_weight + 1):
        # depth-first over supports q_1 < ... < q_w and labels
        stack = [(0, 0, 0, 0)]       # (next qubit, depth, syndrome, pauli)
        while stack:
            start, depth, s, p = stack.pop()
            if depth == w:
                examined += 1
                if deadline is not None and not (examined & 0xFFFF) and time.monotonic() > deadline:
                    return BruteResult(w - 1, None, None, examined)
                if not (s & gmask) and (s >> m):
                    if not code.is_logical(p) or code.weight(p) != w:
                        raise AssertionError("brute force: inconsistent logical test")
                    return BruteResult(w, w, p, examined)
                continue
            # leave room for the remaining w - depth - 1 qubits
            for q in range(n - (w - depth) + 1 - 1, start - 1, -1):
                for lab in labels:
                    sv, pv = vec[q, lab]
                    stack.append((q + 1, depth + 1, s ^ sv, p | pv))
    return BruteResult(max_weight, None, None, examined)
