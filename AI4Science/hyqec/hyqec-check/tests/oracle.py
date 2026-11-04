"""Tiny-code oracle using strings, explicit group closure and full enumeration.

Does not import hyqec_check or use binary elimination/symplectic bit arithmetic.
"""

import itertools
import random


def multiply(a, b):
    out = []
    for x, y in zip(a, b):
        if x == "I":
            out.append(y)
        elif y == "I":
            out.append(x)
        elif x == y:
            out.append("I")
        else:
            out.append(next(p for p in "XYZ" if p != x and p != y))
    return "".join(out)


def commutes(a, b):
    return sum(x != "I" and y != "I" and x != y for x, y in zip(a, b)) % 2 == 0


def closure(n, generators):
    group = {"I" * n}
    for row in generators:
        group |= {multiply(row, element) for element in group}
    return group


def all_paulis(n, domain="P"):
    return map("".join, itertools.product("IXYZ" if domain == "P" else "I" + domain, repeat=n))


def logicals(n, generators, domain="P"):
    group = closure(n, generators)
    return {p for p in all_paulis(n, domain)
            if p not in group and all(commutes(p, g) for g in generators)}


def distance(n, generators, domain="P"):
    return min((sum(c != "I" for c in p) for p in logicals(n, generators, domain)), default=None)


def random_codes(seed=116, repetitions=8):
    rng = random.Random(seed)
    for n in range(1, 6):
        for _ in range(repetitions):
            rows = []
            desired_rank = rng.randrange(n + 1)
            group = {"I" * n}
            while len(group) < 2 ** desired_rank:
                proposal = "".join(rng.choices("IXYZ", k=n))
                if proposal not in group and all(commutes(proposal, g) for g in rows):
                    rows.append(proposal)
                    group = closure(n, rows)
            # Explicit dependent/identity inputs; ordering deliberately retained.
            rows += rng.sample(sorted(group), min(len(group), rng.randrange(3)))
            rng.shuffle(rows)
            yield n, rows
