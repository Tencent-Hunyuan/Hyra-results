import itertools
import unittest

from hyqec_check.algebra import Code, ValidationError
from hyqec_check.rules import Node, VALID_DOMAINS, candidates, children, verify_dead_hint, verify_packing_hint, verify_stabilizer_leaf
import oracle


def completions(node):
    return {"".join("IXZY"[p] for p in ps) for ps in itertools.product(
        *(tuple(p for p in range(4) if allowed & (1 << p)) for allowed in node.domains))}


class RulesTests(unittest.TestCase):
    def test_first_anticommuting_qubit_partition_exhaustively(self):
        code = Code.from_strings(3, ["XYZ"])
        for domains in itertools.product(sorted(VALID_DOMAINS), repeat=3):
            node = Node(domains)
            if not node.active(code):
                continue
            available = candidates(code, node, 0)
            if not available:
                verify_dead_hint(code, node, 0)
                self.assertFalse(any(oracle.commutes(p, "XYZ") for p in completions(node)))
                continue
            for order in (tuple(available), tuple(reversed(available))):
                child_nodes = children(code, node, 0, order)
                covered = set()
                for child in child_nodes:
                    actual = completions(child)
                    self.assertFalse(covered & actual, "child cases must be disjoint")
                    covered |= actual
                parent = completions(node)
                self.assertLessEqual(covered, parent)
                expected = {p for p in parent if any(p[q] not in ("I", "XYZ"[q]) for q in available)}
                self.assertEqual(covered, expected)
                self.assertLessEqual({p for p in parent if oracle.commutes(p, "XYZ")}, covered)

    def test_hint_tampering(self):
        code = Code.from_strings(4, ["ZZII", "IIZZ"])
        node = Node((2, 15, 2, 15))  # decided X at q=0,2; free q=1,3
        verify_packing_hint(code, node, 3, (0, 1))
        for cap, packing in [(4, (0, 1)), (3, (0,)), (3, (0, 0)), (3, (0, 2)), (3, (True, 0))]:
            with self.assertRaises(ValidationError):
                verify_packing_hint(code, node, cap, packing)
        with self.assertRaises(ValidationError):
            verify_dead_hint(code, node, 0)
        dead = Node((2, 1, 2, 15))
        verify_dead_hint(code, dead, 0)
        with self.assertRaises(ValidationError):
            verify_dead_hint(code, dead, 1)
        overlapping = Code.from_strings(3, ["ZZI", "ZIZ"])
        with self.assertRaises(ValidationError):
            verify_packing_hint(overlapping, Node((15, 2, 2)), 3, (0, 1))
        with self.assertRaises(ValidationError):
            children(code, node, 0, ())
        with self.assertRaises(ValidationError):
            children(code, node, 0, (1, 1))
        with self.assertRaises(ValidationError):
            children(code, node, 0, (3,))

    def test_leaf_membership(self):
        code = Code.from_strings(2, ["ZZ"])
        verify_stabilizer_leaf(code, Node((4, 4)))
        with self.assertRaises(ValidationError):
            verify_stabilizer_leaf(code, Node((2, 2)))
        with self.assertRaises(ValidationError):
            verify_stabilizer_leaf(code, Node((2, 15)))


if __name__ == "__main__":
    unittest.main()
