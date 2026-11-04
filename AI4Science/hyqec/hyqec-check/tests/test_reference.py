from hyqec_check.engines import environment
import unittest

from hyqec_check.algebra import Code
from hyqec_check.input import Instance
from hyqec_check.reference import DEFAULT_ENGINE, search
import oracle


def instance(n, rows, permutations=(), reverse=False):
    code = Code.from_strings(n, rows)
    orbits = list(map(list, code.orbits(list(permutations))))
    if reverse:
        orbits.reverse()
    return Instance(code, list(permutations), orbits, [o[-1] for o in orbits], "local-test")


class ReferenceTests(unittest.TestCase):
    engine = environment("REFERENCE_ENGINE", str(DEFAULT_ENGINE))

    def test_known_distances(self):
        examples = [
            (5, ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"], 3, [[1, 2, 3, 4, 0]]),
            (4, ["XXXX", "ZZZZ"], 2, [[1, 2, 3, 0]]),
            (3, ["ZZI", "IZZ"], 1, [[1, 2, 0]]),
            (2, ["XX", "ZZ"], None, [[1, 0]]),
            (2, [], 1, [[1, 0]]),
        ]
        for n, rows, expected, permutations in examples:
            self.assertEqual(oracle.distance(n, rows), expected)
            for domain in ("P", "X", "Z"):
                d = oracle.distance(n, rows, domain)
                for cap in range(n + 1):
                    for symmetry in ([], permutations):
                        with self.subTest(n=n, domain=domain, cap=cap, symmetry=symmetry):
                            result = search(instance(n, rows, symmetry), cap, domain, engine=self.engine)
                            self.assertEqual(result["minimum_found"], d if d is not None and d <= cap else None)
                            for root in result["roots"]:
                                self.assertEqual(root["nodes"], sum(root[k] for k in ("dead", "leaves", "bound", "branches")))
                            self.assertFalse(result["certificate_verified"])

    def test_random_codes_against_exhaustive_oracle(self):
        for n, rows in oracle.random_codes(repetitions=5):
            for domain in ("P", "X", "Z"):
                d = oracle.distance(n, rows, domain)
                for cap in sorted({0, n, max(0, (d or n) - 1), d or n}):
                    with self.subTest(n=n, rows=rows, domain=domain, cap=cap):
                        result = search(instance(n, rows, reverse=True), cap, domain, engine=self.engine)
                        self.assertEqual(result["minimum_found"], d if d is not None and d <= cap else None)

    def test_thread_count_does_not_change_tree(self):
        code = instance(5, ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"])
        serial = search(code, 3, engine=self.engine)
        parallel = search(code, 3, threads=4, engine=self.engine)
        self.assertEqual(serial["roots"], parallel["roots"])


if __name__ == "__main__":
    unittest.main()
