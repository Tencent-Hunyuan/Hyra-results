"""Upper bounds by random information sets (RIS), pure Python + NumPy.

Search space: a basis of N(S) (or of any subspace of it). Each trial picks a
random qubit order, interleaves the columns as (x_q, z_q) per qubit and brings
the basis to reduced echelon form in that column order, so the information set
is a union of whole qubits ("qubit-aligned"): a logical of Pauli weight w is a
reduced row, or the sum of the two rows pivoting on one qubit, once all but one
of its qubits lie outside the pivot qubits, whatever its X/Y/Z mix. Light rows
are also combined in pairs. Every row carries its "logical syndrome" (products
with the logical basis) as extra columns, so a candidate is a nontrivial
logical iff its syndrome is nonzero.

Nothing here is trusted: every reported operator is re-verified with
``StabilizerCode.check_witness`` before it is returned.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..core import CodeError, StabilizerCode, swap_halves

__all__ = ["RisResult", "ris_upper_bound"]

_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


@dataclass
class RisResult:
    weight: int | None
    witness: int | None
    trials: int
    seconds: float
    seed: int
    hits_by_weight: dict = field(default_factory=dict)
    engine: str = "ris-python"


def _popcount_rows(words: np.ndarray) -> np.ndarray:
    return _POP8[words.view(np.uint8)].reshape(words.shape[0], -1).sum(axis=1)


def _to_bits(v: int, nbits: int) -> np.ndarray:
    raw = v.to_bytes((nbits + 7) // 8, "little")
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")[:nbits]


def ris_upper_bound(code: StabilizerCode, *, seconds: float = 10.0, trials: int | None = None,
                    seed: int = 0, target: int | None = None, domain: str = "P",
                    pair_depth: int = 16, patience: float | None = None) -> RisResult:
    """Lightest nontrivial logical found in the time/trial budget (re-verified).

    domain "X"/"Z": search only X-type/Z-type logicals (CSS per-side distances), with
    the search space restricted to the X-type/Z-type part of N(S).
    target: stop as soon as a logical of weight < target is found.
    patience: stop after this many seconds without an improvement."""
    code.require_logicals()
    n = code.n
    rng = np.random.default_rng(seed)
    L = code.logicals
    lsw = [swap_halves(l, n) for l in L]
    if domain == "P":
        basis = code.normalizer_basis()
    elif domain in ("X", "Z"):
        basis = _side_basis(code, domain)
        if not basis:
            return RisResult(None, None, 0, 0.0, seed)
    else:
        raise ValueError(f"unknown domain {domain!r}")
    r = len(basis)
    nl = len(L)
    # bit matrix: 2n Pauli columns (x_0..x_{n-1}, z_0..z_{n-1}) then nl syndrome columns
    B = np.zeros((r, 2 * n + nl), dtype=np.uint8)
    for i, v in enumerate(basis):
        B[i, :2 * n] = _to_bits(v, 2 * n)
        for j, l in enumerate(lsw):
            B[i, 2 * n + j] = (v & l).bit_count() & 1
    t0 = time.monotonic()
    best_w, best_v = None, None
    hits: dict[int, int] = {}
    t = 0

    nonlocal_improved = [t0]

    def record(rows_bits):
        nonlocal best_w, best_v
        if rows_bits.shape[0] == 0:
            return
        x, z, syn = rows_bits[:, :n], rows_bits[:, n:2 * n], rows_bits[:, 2 * n:]
        w = (x | z).sum(axis=1)
        good = syn.any(axis=1) & (w > 0)
        for i in np.flatnonzero(good):
            wi = int(w[i])
            hits[wi] = hits.get(wi, 0) + 1
            if best_w is None or wi < best_w:
                v = int.from_bytes(np.packbits(rows_bits[i, :2 * n], bitorder="little").tobytes(), "little")
                code.check_witness(v)          # raises if the engine produced garbage
                if code.weight(v) != wi:
                    raise AssertionError("RIS weight mismatch")
                best_w, best_v = wi, v
                nonlocal_improved[0] = time.monotonic()

    while True:
        if trials is not None and t >= trials:
            break
        if trials is None and time.monotonic() - t0 > seconds:
            break
        if target is not None and best_w is not None and best_w < target:
            break
        if patience is not None and trials is None and time.monotonic() - nonlocal_improved[0] > patience:
            break
        t += 1
        order = rng.permutation(n)
        cols = np.empty(2 * n, dtype=np.int64)
        cols[0::2] = order
        cols[1::2] = order + n
        R, piv_rows, piv_qubit = _rref(B, cols)
        record(R)
        # sums of the two rows pivoting on the same qubit (a Y-type pivot)
        pq = np.asarray(piv_qubit)
        if len(pq):
            uq, inv, cnt = np.unique(pq, return_inverse=True, return_counts=True)
            for qi in np.flatnonzero(cnt == 2):
                a, b = np.flatnonzero(inv == qi)[:2]
                record((R[piv_rows[a]] ^ R[piv_rows[b]])[None, :])
        if pair_depth > 1 and R.shape[0] > 1:
            w = (R[:, :n] | R[:, n:2 * n]).sum(axis=1)
            light = np.argsort(w, kind="stable")[:pair_depth]
            sub = R[light]
            for a in range(len(light) - 1):
                record(sub[a] ^ sub[a + 1:])
    return RisResult(best_w, best_v, t, time.monotonic() - t0, seed, dict(sorted(hits.items())))


def _rref(B: np.ndarray, cols: np.ndarray):
    """Reduced echelon form of B pivoting in the column order ``cols`` (copy)."""
    R = B.copy()
    r = R.shape[0]
    n2 = len(cols)
    n = n2 // 2
    used = np.zeros(r, dtype=bool)
    piv_rows, piv_qubit = [], []
    nrank = 0
    for c in cols:
        if nrank == r:
            break
        col = R[:, c]
        cand = np.flatnonzero(col.astype(bool) & ~used)
        if cand.size == 0:
            continue
        p = cand[0]
        used[p] = True
        others = np.flatnonzero(col.astype(bool))
        others = others[others != p]
        if others.size:
            R[others] ^= R[p]
        piv_rows.append(p)
        piv_qubit.append(int(c % n))
        nrank += 1
    return R, piv_rows, piv_qubit


def _side_basis(code: StabilizerCode, domain: str) -> list[int]:
    """Basis of the X-type (or Z-type) part of N(S)."""
    from ..core import GF2Basis
    n = code.n
    gsw = [swap_halves(g, n) for g in code.generators]
    # functionals restricted to the allowed half
    if domain == "X":
        funcs = [g & ((1 << n) - 1) for g in gsw]          # x-part pairs with generator z-part
        ker = GF2Basis(funcs).kernel(n)
        return [v for v in ker]
    funcs = [g >> n for g in gsw]
    ker = GF2Basis(funcs).kernel(n)
    return [v << n for v in ker]
