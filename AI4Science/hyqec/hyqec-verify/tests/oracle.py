"""Independent brute-force oracle for tests (shares no code with hyqec-verify).

Paulis are tuples of per-qubit labels 0=I, 1=X, 2=Z, 3=Y. The group is built
by explicit closure, the normalizer by scanning all 4^n Paulis. Only for n <= 8.
"""
from __future__ import annotations

import itertools
import random


def mul(p, q):
    # label bits are (x, z): the product (phases ignored) xors them
    return tuple(a ^ b for a, b in zip(p, q))


def commute(p, q):
    return sum(1 for a, b in zip(p, q) if a and b and a != b) % 2 == 0


def wt(p):
    return sum(1 for a in p if a)


def group(gens, n):
    G = {tuple([0] * n)}
    for g in gens:
        G |= {mul(h, g) for h in G}
    return G


def analyse(gens, n):
    """(isotropic, k, distance or None, normalizer, group)."""
    iso = all(commute(a, b) for a, b in itertools.combinations(gens, 2))
    G = group(gens, n)
    rank = len(G).bit_length() - 1
    k = n - rank
    if not iso:
        return False, k, None, None, G
    N = [p for p in itertools.product(range(4), repeat=n) if all(commute(p, g) for g in gens)]
    logicals = [p for p in N if p not in G]
    d = min((wt(p) for p in logicals), default=None)
    return True, k, d, N, G


def side_distance(gens, n, label):
    """Min weight of a nontrivial logical using only labels {0, label}."""
    G = group(gens, n)
    best = None
    for bits in itertools.product((0, label), repeat=n):
        if bits in G or not any(bits):
            continue
        if all(commute(bits, g) for g in gens):
            w = wt(bits)
            best = w if best is None else min(best, w)
    return best


# --- random stabilizer codes via random Clifford circuits -------------------

def _apply(p, gate, a, b=None):
    p = list(p)
    x = [l & 1 for l in p]
    z = [l >> 1 for l in p]
    if gate == "H":
        x[a], z[a] = z[a], x[a]
    elif gate == "S":
        z[a] ^= x[a]
    elif gate == "CX":
        x[b] ^= x[a]
        z[a] ^= z[b]
    return tuple(xi | (zi << 1) for xi, zi in zip(x, z))


def random_code(n, k, rng: random.Random, depth=None, extra_rows=0):
    """Random [[n,k]] stabilizer code: Z on n-k qubits, scrambled by a random Clifford."""
    gens = []
    for i in range(n - k):
        p = [0] * n
        p[i] = 2
        gens.append(tuple(p))
    for _ in range(depth or 4 * n):
        g = rng.choice(["H", "S", "CX"])
        if g == "CX":
            a, b = rng.sample(range(n), 2)
            gens = [_apply(p, g, a, b) for p in gens]
        else:
            a = rng.randrange(n)
            gens = [_apply(p, g, a) for p in gens]
    # recombine and add dependent rows: same group, different generating set
    for _ in range(2 * len(gens)):
        i, j = rng.sample(range(len(gens)), 2) if len(gens) > 1 else (0, 0)
        if i != j:
            gens[i] = mul(gens[i], gens[j])
    for _ in range(extra_rows):
        if gens:
            gens.append(mul(rng.choice(gens), rng.choice(gens)))
    rng.shuffle(gens)
    return gens


def to_int(p, n):
    v = 0
    for q, l in enumerate(p):
        if l & 1:
            v |= 1 << q
        if l & 2:
            v |= 1 << (n + q)
    return v


def from_int(v, n):
    return tuple(((v >> q) & 1) | (((v >> (n + q)) & 1) << 1) for q in range(n))
