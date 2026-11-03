"""The per-process compiled-engine budget (two construction attempts per Python process)."""
import os
import unittest
from unittest import mock

from hyff.engines import compiler_budget


class CompilerBudgetTest(unittest.TestCase):
    def ledger(self):
        return compiler_budget._ProcessAttemptLedger()

    def test_two_attempts_then_refusal(self):
        ledger = self.ledger()
        self.assertEqual(ledger.reserve('presto'), 1)
        self.assertEqual(ledger.reserve('largo'), 2)
        with self.assertRaises(compiler_budget.CompiledEngineProcessLimitError) as caught:
            ledger.reserve('presto')
        self.assertEqual((caught.exception.reason, caught.exception.used, caught.exception.limit),
                         ('attempt_limit', 2, 2))

    def test_invalid_route_names(self):
        for route in ('', 'Presto', '1presto', 'pres to', None, 'x' * 65):
            with self.subTest(route=route), self.assertRaises(ValueError):
                self.ledger().reserve(route)

    def test_forked_child_is_refused(self):
        ledger = self.ledger()
        with mock.patch.object(os, 'getpid', return_value=ledger._owner_pid + 1):
            with self.assertRaises(compiler_budget.CompiledEngineProcessLimitError) as caught:
                ledger.reserve('presto')
        self.assertEqual(caught.exception.reason, 'forked_process')


if __name__ == '__main__':
    unittest.main()
