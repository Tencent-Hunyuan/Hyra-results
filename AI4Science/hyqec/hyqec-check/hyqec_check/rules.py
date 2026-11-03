"""Specification-independent semantic obligations for future frontier checking.

These are mathematical node/domain operations, not certificate deserialization
or deterministic policy checks. Both will be added from BNB_CERT_SPEC.md.
"""

from dataclasses import dataclass

from .algebra import Code, ValidationError, integer, symplectic, weight


VALID_DOMAINS = frozenset({1, 2, 4, 8, 3, 5, 9, 15})


@dataclass(frozen=True)
class Node:
    # Label codes I=0 X=1 Z=2 Y=3; each bit denotes an allowed label.
    domains: tuple[int, ...]

    def validate(self, code: Code) -> None:
        if (len(self.domains) != code.n
                or any(type(a) is not int or a not in VALID_DOMAINS for a in self.domains)):
            raise ValidationError("invalid node domains")

    def decided(self, code: Code) -> int:
        self.validate(code)
        result = 0
        for q, allowed in enumerate(self.domains):
            if allowed.bit_count() != 1:
                continue
            p = allowed.bit_length() - 1
            result |= (p & 1) << q
            result |= ((p >> 1) & 1) << (q + code.n)
        return result

    def active(self, code: Code) -> tuple[int, ...]:
        value = self.decided(code)
        return tuple(i for i, row in enumerate(code.generators) if symplectic(value, row, code.n))


def generator_label(code: Code, generator: int, qubit: int) -> int:
    row = code.generators[generator]
    return ((row >> qubit) & 1) | (((row >> (code.n + qubit)) & 1) << 1)


def candidates(code: Code, node: Node, generator: int) -> dict[int, tuple[int, ...]]:
    node.validate(code)
    integer(generator, "generator index")
    if generator >= len(code.generators):
        raise ValidationError("generator index out of range")
    result = {}
    for q, allowed in enumerate(node.domains):
        g = generator_label(code, generator, q)
        if not g or allowed.bit_count() == 1:
            continue
        options = tuple(p for p in (1, 2, 3) if p != g and allowed & (1 << p))
        if options:
            result[q] = options
    return result


def verify_dead_hint(code: Code, node: Node, generator: int) -> None:
    # Validate index even when Python booleans would otherwise compare as ints.
    remaining = candidates(code, node, generator)
    if generator not in node.active(code):
        raise ValidationError("dead hint generator is not active")
    if remaining:
        raise ValidationError("dead hint generator still has a candidate")


def verify_packing_hint(code: Code, node: Node, cap: int, packing: tuple[int, ...]) -> None:
    integer(cap, "cap")
    active = node.active(code)
    used_generators = set()
    occupied = set()
    for g in packing:
        current = candidates(code, node, g)
        if g not in active or g in used_generators:
            raise ValidationError("packing generators must be distinct and active")
        if not current:
            raise ValidationError("packing uses a dead generator; require a dead hint")
        if occupied.intersection(current):
            raise ValidationError("packing candidate sets overlap")
        used_generators.add(g)
        occupied.update(current)
    if weight(node.decided(code), code.n) + len(used_generators) <= cap:
        raise ValidationError("packing does not prove weight greater than cap")


def verify_stabilizer_leaf(code: Code, node: Node) -> None:
    value = node.decided(code)
    if node.active(code):
        raise ValidationError("leaf still has active generators")
    # Use membership, not a certificate-provided logical basis or syndrome.
    if code.stabilizers.reduce(value):
        raise ValidationError("leaf contains a nontrivial logical")


def children(code: Code, node: Node, generator: int,
             qubit_order: tuple[int, ...]) -> tuple[Node, ...]:
    available = candidates(code, node, generator)
    if generator not in node.active(code) or not available:
        raise ValidationError("branch generator must be active with a candidate")
    if (any(type(q) is not int for q in qubit_order)
            or len(qubit_order) != len(available) or set(qubit_order) != set(available)):
        raise ValidationError("branch order must contain every candidate exactly once")
    prior = list(node.domains)
    result = []
    for q in qubit_order:
        for p in available[q]:
            child = prior.copy()
            child[q] = 1 << p
            result.append(Node(tuple(child)))
        prior[q] &= 1 | (1 << generator_label(code, generator, q))
    return tuple(result)
