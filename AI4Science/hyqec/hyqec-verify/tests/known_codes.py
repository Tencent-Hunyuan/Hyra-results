"""Small textbook codes with known parameters (for tests)."""
from hyqec_verify.core import StabilizerCode
from hyqec_verify.io import parse_pauli


def from_strings(strs, name=None):
    gens = [parse_pauli(s)[0] for s in strs]
    return StabilizerCode(len(strs[0]), gens, name=name)


def cyclic(base, rows):
    return [base[-i:] + base[:-i] if i else base for i in range(rows)]


def rotated_surface(L):
    """Rotated surface code [[L^2, 1, L]] (CSS)."""
    n = L * L
    idx = lambda r, c: r * L + c
    hx, hz = [], []
    for r in range(-1, L):
        for c in range(-1, L):
            qs = [idx(rr, cc) for rr in (r, r + 1) for cc in (c, c + 1) if 0 <= rr < L and 0 <= cc < L]
            if (r + c) % 2 == 0:
                # X plaquettes: bulk + top/bottom boundary halves
                if len(qs) == 4 or (len(qs) == 2 and r in (-1, L - 1)):
                    hx.append(qs)
            else:
                if len(qs) == 4 or (len(qs) == 2 and c in (-1, L - 1)):
                    hz.append(qs)
    return StabilizerCode.css(hx, hz, n, name=f"surface-{L}")


def toric(L):
    """Toric code [[2L^2, 2, L]] (CSS)."""
    n = 2 * L * L
    h = lambda r, c: (r % L) * L + (c % L)
    v = lambda r, c: L * L + (r % L) * L + (c % L)
    hx = [[h(r, c), h(r, c - 1), v(r, c), v(r - 1, c)] for r in range(L) for c in range(L)]
    hz = [[h(r, c), h(r + 1, c), v(r, c), v(r, c + 1)] for r in range(L) for c in range(L)]
    return StabilizerCode.css(hx, hz, n, name=f"toric-{L}")


def known():
    """(code, k, d, dX, dZ) — dX/dZ None when not a CSS question."""
    out = [
        (from_strings(cyclic("XZZXI", 4), "5-qubit"), 1, 3, None, None),
        (from_strings(["XXXX", "ZZZZ"], "4-2-2"), 2, 2, 2, 2),
        (from_strings(["IIIXXXX", "IXXIIXX", "XIXIXIX", "IIIZZZZ", "IZZIIZZ", "ZIZIZIZ"], "steane"), 1, 3, 3, 3),
        (from_strings(["ZZIIIIIII", "IZZIIIIII", "IIIZZIIII", "IIIIZZIII", "IIIIIIZZI", "IIIIIIIZZ",
                       "XXXXXXIII", "IIIXXXXXX"], "shor"), 1, 3, 3, 3),
        # [[8,3,3]] (Gottesman)
        (from_strings(["XXXXXXXX", "ZZZZZZZZ", "IXIXYZYZ", "IXZYIXZY", "IYXZXZIY"], "8-3-3"), 3, 3, None, None),
        (rotated_surface(3), 1, 3, 3, 3),
        (rotated_surface(5), 1, 5, 5, 5),
        (toric(3), 2, 3, 3, 3),
        (toric(4), 2, 4, 4, 4),
    ]
    return out
