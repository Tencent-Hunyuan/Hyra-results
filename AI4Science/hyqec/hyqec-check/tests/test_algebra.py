import json
from pathlib import Path
import tempfile
import unittest

from hyqec_check.algebra import Basis, Code, ValidationError, kernel, pauli, pauli_string, symplectic, weight
from hyqec_check.input import read_reference_instance, strict_json
import oracle


class AlgebraTests(unittest.TestCase):
    def test_exhaustive_against_group_closure(self):
        for n, rows in oracle.random_codes():
            with self.subTest(n=n, rows=rows):
                code = Code.from_strings(n, rows)
                group = oracle.closure(n, rows)
                self.assertEqual(2 ** len(code.stabilizers), len(group))
                normalizer = {p for p in oracle.all_paulis(n)
                              if all(oracle.commutes(p, g) for g in rows)}
                generated_normalizer = oracle.closure(n, [pauli_string(v, n) for v in code.normalizer])
                self.assertEqual(normalizer, generated_normalizer)
                extended = oracle.closure(n, rows + [pauli_string(v, n) for v in code.logicals])
                self.assertEqual(extended, normalizer)
                for p in oracle.all_paulis(n):
                    value = pauli(p, n)
                    self.assertEqual(pauli_string(value, n), p)
                    self.assertEqual(weight(value, n), sum(c != "I" for c in p))
                    expected = p in normalizer and p not in group
                    self.assertEqual(code.is_logical(value), expected)
                    if p in normalizer:
                        self.assertEqual(any(symplectic(value, l, n) for l in code.logicals), expected)

    def test_permutations_and_orbits(self):
        code = Code.from_strings(5, ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"])
        cycle = [1, 2, 3, 4, 0]
        self.assertEqual(code.orbits([cycle]), ((0, 1, 2, 3, 4),))
        self.assertEqual(code.orbits([]), ((0,), (1,), (2,), (3,), (4,)))
        code.validate_orbit_order([cycle], [[4, 3, 2, 1, 0]], [2])
        invalid = [[0, 1], [0, 1, 2, 3, 3], [True, 0, 2, 3, 4], [0, 1, 2, 3, 5]]
        for permutation in invalid:
            with self.assertRaises(ValidationError):
                code.validate_permutation(permutation)
        with self.assertRaises(ValidationError):
            Code.from_strings(2, ["ZI"]).validate_permutation([1, 0])
        for orbits, reps in [([[0, 1, 2], [3, 4]], [0, 3]), ([[0, 1, 2, 3, 4, 4]], [0]),
                             ([[0, 1, 2, 3, 4]], [5]), ([[0, 1, 2, 3, True]], [0])]:
            with self.assertRaises(ValidationError):
                code.validate_orbit_order([cycle], orbits, reps)

    def test_rejection(self):
        for n, rows in [(1, ["X", "Z"]), (True, []), (0, []), (2, ["X"]),
                        (1, ["x"]), (1, [1]), (1, "X"), (1, ["-"])]:
            with self.assertRaises(ValidationError):
                Code.from_strings(n, rows)
        with self.assertRaises(ValidationError):
            kernel([4], 2)
        with self.assertRaises(ValidationError):
            Basis([-1])

    def test_parser(self):
        data = {"format": "bnbcheck-reference-code/0", "n": 2, "k": 1,
                "generators": ["ZZ"], "permutations": [[1, 0]]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "code.json"
            path.write_text(json.dumps(data))
            self.assertEqual(read_reference_instance(path).orbits, [[0, 1]])
            for override in [{"k": 2}, {"k": True}, {"phase": 1}, {"orbits": [[0], [1]]},
                             {"permutations": None}, {"orbits": None}]:
                path.write_text(json.dumps(data | override))
                with self.assertRaises(ValidationError):
                    read_reference_instance(path)
        for raw in [b'{"n":1,"n":2}', b'{"n":NaN}', b'\xff', b'{']:
            with self.assertRaises(ValidationError):
                strict_json(raw)


if __name__ == "__main__":
    unittest.main()
