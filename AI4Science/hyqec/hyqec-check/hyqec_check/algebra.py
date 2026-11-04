"""Binary algebra implemented from the mathematical definitions.

Coordinates 0..n-1 are X, coordinates n..2n-1 are Z. Our elimination uses
lowest-bit pivots; it intentionally imposes no prover-specific serialization.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence


class ValidationError(ValueError):
    pass


def integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValidationError(f"{name} must be an integer >= {minimum}")
    return value


class Basis:
    def __init__(self, rows: Iterable[int] = ()):
        self.pivots: dict[int, int] = {}
        for row in rows:
            self.add(row)

    def reduce(self, value: int) -> int:
        for pivot in sorted(self.pivots):
            if value & (1 << pivot):
                value ^= self.pivots[pivot]
        return value

    def add(self, value: int) -> bool:
        integer(value, "binary vector")
        value = self.reduce(value)
        if value == 0:
            return False
        pivot = (value & -value).bit_length() - 1
        for other, row in self.pivots.items():
            if row & (1 << pivot):
                self.pivots[other] = row ^ value
        self.pivots[pivot] = value
        return True

    @property
    def rows(self) -> tuple[int, ...]:
        return tuple(self.pivots[p] for p in sorted(self.pivots))

    def __len__(self) -> int:
        return len(self.pivots)


def kernel(equations: Iterable[int], width: int) -> tuple[int, ...]:
    integer(width, "width")
    basis = Basis(equations)
    if any(row >> width for row in basis.rows):
        raise ValidationError("equation exceeds specified vector width")
    result = []
    for free in range(width):
        if free in basis.pivots:
            continue
        value = 1 << free
        for pivot, row in basis.pivots.items():
            if row & (1 << free):
                value |= 1 << pivot
        result.append(value)
    return tuple(result)


def symplectic(a: int, b: int, n: int) -> int:
    mask = (1 << n) - 1
    return (((a & mask) & (b >> n)).bit_count()
            + ((b & mask) & (a >> n)).bit_count()) & 1


def weight(value: int, n: int) -> int:
    return ((value | (value >> n)) & ((1 << n) - 1)).bit_count()


def pauli(text: str, n: int) -> int:
    if not isinstance(text, str) or len(text) != n:
        raise ValidationError(f"Pauli must be a string of length {n}")
    value = 0
    for q, label in enumerate(text):
        if label not in "IXYZ":
            raise ValidationError(f"invalid Pauli label {label!r} at qubit {q}")
        if label in "XY":
            value |= 1 << q
        if label in "ZY":
            value |= 1 << (n + q)
    return value


def pauli_string(value: int, n: int) -> str:
    integer(value, "Pauli vector")
    if value >> (2 * n):
        raise ValidationError("Pauli exceeds specified vector width")
    return "".join("IXZY"[((value >> q) & 1) | (((value >> (q + n)) & 1) << 1)]
                   for q in range(n))


@dataclass
class Code:
    n: int
    generators: tuple[int, ...]
    stabilizers: Basis
    normalizer: tuple[int, ...]
    logicals: tuple[int, ...]

    @classmethod
    def from_strings(cls, n: int, rows: Sequence[str]) -> "Code":
        integer(n, "n", 1)
        if not isinstance(rows, (list, tuple)):
            raise ValidationError("generators must be a list of Pauli strings")
        generators = tuple(pauli(row, n) for row in rows)
        for i, a in enumerate(generators):
            for j, b in enumerate(generators[:i]):
                if symplectic(a, b, n):
                    raise ValidationError(f"generators {j} and {i} anticommute")
        stabilizers = Basis(generators)
        mask = (1 << n) - 1
        normalizer = kernel(((g >> n) | ((g & mask) << n)
                             for g in stabilizers.rows), 2 * n)
        extension = Basis(stabilizers.rows)
        logicals = tuple(v for v in normalizer if extension.add(v))
        result = cls(n, generators, stabilizers, normalizer, logicals)
        # These checks make a failed algebra invariant a refusal, never a proof.
        if result.k < 0 or len(normalizer) != n + result.k:
            raise ValidationError("invalid stabilizer/normalizer dimensions")
        if len(logicals) != 2 * result.k:
            raise ValidationError("invalid quotient dimension")
        if any(symplectic(v, g, n) for v in normalizer for g in generators):
            raise ValidationError("computed normalizer violates commutation")
        gram = Basis(sum(symplectic(a, b, n) << j for j, b in enumerate(logicals))
                     for a in logicals)
        if len(gram) != 2 * result.k:
            raise ValidationError("logical symplectic form is singular")
        return result

    @property
    def k(self) -> int:
        return self.n - len(self.stabilizers)

    def is_logical(self, value: int) -> bool:
        integer(value, "Pauli vector")
        if value >> (2 * self.n):
            raise ValidationError("Pauli exceeds code width")
        return (all(symplectic(value, g, self.n) == 0 for g in self.generators)
                and self.stabilizers.reduce(value) != 0)

    def validate_permutation(self, permutation: Sequence[int]) -> tuple[int, ...]:
        if (not isinstance(permutation, (list, tuple)) or len(permutation) != self.n
                or any(type(q) is not int for q in permutation)
                or set(permutation) != set(range(self.n))):
            raise ValidationError("qubit permutation must be a bijection of 0..n-1")
        for i, g in enumerate(self.generators):
            moved = 0
            for q, destination in enumerate(permutation):
                moved |= ((g >> q) & 1) << destination
                moved |= ((g >> (q + self.n)) & 1) << (destination + self.n)
            if self.stabilizers.reduce(moved):
                raise ValidationError(f"permutation does not preserve generator {i}")
        return tuple(permutation)

    def orbits(self, permutations: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
        if not isinstance(permutations, (tuple, list)):
            raise ValidationError("permutations must be a list")
        parent = list(range(self.n))

        def root(q: int) -> int:
            while parent[q] != q:
                parent[q] = parent[parent[q]]
                q = parent[q]
            return q

        for permutation in permutations:
            for q, target in enumerate(self.validate_permutation(permutation)):
                parent[root(q)] = root(target)
        groups: dict[int, list[int]] = {}
        for q in range(self.n):
            groups.setdefault(root(q), []).append(q)
        return tuple(sorted((tuple(qs) for qs in groups.values()), key=lambda qs: qs[0]))

    def validate_orbit_order(self, permutations: Sequence[Sequence[int]],
                             orbits: Sequence[Sequence[int]],
                             representatives: Sequence[int]) -> None:
        actual = self.orbits(permutations)
        if not isinstance(orbits, (tuple, list)) or not isinstance(representatives, (tuple, list)):
            raise ValidationError("orbits and representatives must be lists")
        if len(orbits) != len(actual) or len(representatives) != len(actual):
            raise ValidationError("orbit/representative count differs from computed orbits")
        normalized = []
        for orbit, representative in zip(orbits, representatives):
            if (not isinstance(orbit, (list, tuple)) or not orbit
                    or any(type(q) is not int for q in orbit)
                    or len(set(orbit)) != len(orbit)):
                raise ValidationError("invalid orbit members")
            if type(representative) is not int or representative not in orbit:
                raise ValidationError("representative is not a member of its orbit")
            normalized.append(tuple(sorted(orbit)))
        if sorted(normalized) != sorted(actual):
            raise ValidationError("claimed orbits differ from verified automorphism orbits")
