"""CPU checks of the MD failure policy (recovery on/off, NVE); no model or GPU."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyff import cli
from hyff.md import simulation as md
from hyff.md.checked_phases import PhaseCheckError


class RecoveryPolicyTest(unittest.TestCase):
    def make_run(self, *, recovery='off', tau_ps=.5):
        run = object.__new__(md.MDRun)
        run.cfg = md.RunConfig(system='unused.npz', model='presto', steps=1,
                               timestep_fs=1.5, recovery=recovery, tau_ps=tau_ps)
        run.stepper = SimpleNamespace(position=torch.tensor([[1., 2., 3.]], dtype=torch.float64),
                                      velocity=torch.tensor([[.1, .2, .3]], dtype=torch.float64))
        run.result = dict(status='running', recoveries=[])
        run.periodic_molecules = None
        run.dt = 1.5
        run.reserve = 1.5
        run.driver_base_steps = 0
        run.rebuild_integrator = Mock(side_effect=AssertionError('rebuild forbidden'))
        run.save = Mock()
        run.flush = Mock()
        return run

    def fail_run(self, run, error):
        run.advance = Mock(side_effect=error)
        generator = torch.Generator(device='cpu')
        with patch.object(md.torch, 'Generator', return_value=generator), \
             patch.object(md.torch.cuda, 'synchronize'), \
             patch.object(md.torch, 'randn', side_effect=AssertionError('redraw forbidden')) as redraw, \
             patch.object(run, 'recover', wraps=run.recover) as recover:
            with self.assertRaises(PhaseCheckError) as caught:
                run.run()
        self.assertIs(caught.exception, error)
        self.assertEqual(run.result['status'], 'failed')
        self.assertEqual(run.result['recoveries'], [])
        self.assertEqual(run.reserve, 1.5)
        self.assertEqual(run.dt, 1.5)
        self.assertEqual(run.fallback_remaining, 0)
        torch.testing.assert_close(run.stepper.position, torch.tensor([[1., 2., 3.]], dtype=torch.float64), rtol=0, atol=0)
        torch.testing.assert_close(run.stepper.velocity, torch.tensor([[.1, .2, .3]], dtype=torch.float64), rtol=0, atol=0)
        run.rebuild_integrator.assert_not_called()
        redraw.assert_not_called()
        recover.assert_not_called()
        return run

    def test_nvt_off_rejects_both_recovery_actions(self):
        for phase, native in [('post_check', {}), ('neighbor_rebuild', {'error_bits': 1})]:
            with self.subTest(phase=phase):
                run = self.make_run()
                error = PhaseCheckError([1], phase, 0)
                error.details['native_neighbor_status'] = native
                self.fail_run(run, error)
                self.assertTrue(run.result['recovery_disabled'])
                self.assertFalse(run.result['nve_interrupted'])
                run.flush.assert_called_once()

    def test_reporting_error_preserves_original_phase_error(self):
        run = self.make_run()
        run.flush.side_effect = OSError('report failed')
        self.fail_run(run, PhaseCheckError([1], 'post_check', 0))
        self.assertEqual(run.result['failure_report_error_type'], 'OSError')

    def test_direct_recover_off_also_has_no_actions(self):
        run = self.make_run()
        error = PhaseCheckError([1], 'post_check', 0)
        with self.assertRaises(PhaseCheckError) as caught:
            run.recover(error)
        self.assertIs(caught.exception, error)
        self.assertEqual(run.result['status'], 'failed')
        self.assertEqual(run.result['recoveries'], [])
        run.rebuild_integrator.assert_not_called()
        run.save.assert_not_called()

    def test_nve_default_still_fails_without_recovery(self):
        run = self.make_run(recovery='on', tau_ps=0)
        self.fail_run(run, PhaseCheckError([1], 'post_check', 0))
        self.assertTrue(run.result['nve_interrupted'])
        self.assertFalse(run.result['recovery_disabled'])

    def test_default_nvt_still_selects_recovery(self):
        run = self.make_run(recovery='on')
        run.advance = Mock(side_effect=PhaseCheckError([1], 'post_check', 0))
        sentinel = RuntimeError('recovery selected')
        run.recover = Mock(side_effect=sentinel)
        generator = torch.Generator(device='cpu')
        with patch.object(md.torch, 'Generator', return_value=generator), \
             patch.object(md.torch.cuda, 'synchronize'):
            with self.assertRaises(RuntimeError) as caught:
                run.run()
        self.assertIs(caught.exception, sentinel)
        run.recover.assert_called_once()

    def test_cli_forwarding(self):
        for flag, expected in [([], 'on'), (['--no-recovery'], 'off')]:
            with self.subTest(expected=expected), patch.object(md, 'MDRun') as make:
                make.return_value.run.return_value = dict(status='complete')
                code = cli.main(['md', '--model', 'presto', '--system', 'unused.npz',
                                 '--output', 'unused-output', *flag])
                self.assertEqual(code, 0)
                self.assertEqual(make.call_args.args[0].recovery, expected)
                make.return_value.setup.assert_called_once()
                make.return_value.run.assert_called_once()
                make.return_value.close.assert_called()

    def test_config_validation(self):
        self.assertEqual(md.RunConfig(system='x', model='presto').recovery, 'on')
        with self.assertRaises(ValueError):
            md.RunConfig(system='x', model='presto', recovery='invalid')


if __name__ == '__main__':
    unittest.main()
