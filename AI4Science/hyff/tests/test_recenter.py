"""CPU checks: periodic recentering is refused with an active isolated latent-charge term."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from hyff.kernels.long_range.backends import _KIND
from hyff.md.simulation import MDRun, RunConfig
from hyff.models.latent_charges import LatentChargeEnergy


class RecenterLatentChargeTest(unittest.TestCase):
    def make(self, scale, interval=1., *, nested=True, kind=None):
        model = torch.nn.Module()
        if scale is not None:
            holder = torch.nn.Module() if nested else model
            holder.add_module('latent_charge_energy', LatentChargeEnergy(2))
            with torch.no_grad():
                holder.latent_charge_energy.scale.fill_(scale)
            if kind is not None:
                holder.latent_charge_energy.__dict__[_KIND] = kind
            if nested:
                model.add_module('nested', holder)
        run = object.__new__(MDRun)
        run.cfg = RunConfig(system='unused.npz', model='largo', recenter_interval_ps=interval)
        run.engine = None
        run.result = {'status': 'setup'}
        made = dict(model=model, batch=object(), engine=SimpleNamespace(close=Mock()))
        return run, made

    def test_nonzero_scale_rejected_and_engine_closed(self):
        for nested in (False, True):
            for kind in (None, 'native', 'dense', 'tree'):
                for scale in (1., -1., 1e-30):
                    with self.subTest(nested=nested, kind=kind, scale=scale):
                        run, made = self.make(scale, nested=nested, kind=kind)
                        with self.assertRaisesRegex(ValueError, 'raw unwrapped coordinates'):
                            run._adopt_engine(made)
                        self.assertEqual(run.result['status'], 'failed')
                        self.assertIsNone(run.engine)
                        made['engine'].close.assert_called_once()

    def test_active_spme_allowed_on_top_level_or_nested_module(self):
        for nested in (False, True):
            for scale in (1., -1., 1e-30):
                with self.subTest(nested=nested, scale=scale):
                    run, made = self.make(scale, nested=nested, kind='spme')
                    run._adopt_engine(made)
                    self.assertEqual(run.result['status'], 'setup')
                    self.assertIs(run.engine, made['engine'])
                    self.assertIs(run.model, made['model'])
                    self.assertIs(run.batch, made['batch'])
                    made['engine'].close.assert_not_called()

    def test_spme_sibling_does_not_hide_active_isolated_term(self):
        run, made = self.make(1., nested=False, kind='spme')
        isolated = torch.nn.Module()
        isolated.add_module('latent_charge_energy', LatentChargeEnergy(2))
        with torch.no_grad():
            isolated.latent_charge_energy.scale.fill_(1.)
        made['model'].add_module('isolated_sibling', isolated)
        with self.assertRaisesRegex(ValueError, 'LR_RECENTER_REQUIRES_PERIODIC_LR'):
            run._adopt_engine(made)
        self.assertEqual(run.result['status'], 'failed')
        self.assertIsNone(run.engine)
        made['engine'].close.assert_called_once()

    def test_renamed_active_latent_term_is_not_skipped(self):
        run, made = self.make(None)
        term = LatentChargeEnergy(2)
        with torch.no_grad():
            term.scale.fill_(1.)
        made['model'].add_module('renamed_lr', term)
        with self.assertRaisesRegex(ValueError, 'recenter_interval_ps'):
            run._adopt_engine(made)
        self.assertEqual(run.result['status'], 'failed')
        self.assertIsNone(run.engine)
        made['engine'].close.assert_called_once()

    def test_modulelist_active_latent_term_is_not_skipped(self):
        run, made = self.make(None)
        term = LatentChargeEnergy(2)
        with torch.no_grad():
            term.scale.fill_(1.)
        nested = torch.nn.Module()
        nested.add_module('terms', torch.nn.ModuleList([term]))
        made['model'].add_module('nested', nested)
        with self.assertRaisesRegex(ValueError, 'recenter_interval_ps'):
            run._adopt_engine(made)
        self.assertEqual(run.result['status'], 'failed')
        self.assertIsNone(run.engine)
        made['engine'].close.assert_called_once()

    def test_close_failure_preserves_clear_rejection(self):
        run, made = self.make(1.)
        made['engine'].close.side_effect = RuntimeError('close failed')
        with self.assertRaisesRegex(ValueError, 'recenter_interval_ps'):
            run._adopt_engine(made)
        self.assertEqual(run.result['status'], 'failed')

    def test_zero_scale_and_absent_term_preserve_recenter(self):
        for nested in (False, True):
            for scale in (0., None):
                with self.subTest(nested=nested, scale=scale):
                    run, made = self.make(scale, nested=nested)
                    run._adopt_engine(made)
                    self.assertIs(run.engine, made['engine'])
                    self.assertIs(run.model, made['model'])
                    self.assertIs(run.batch, made['batch'])
                    made['engine'].close.assert_not_called()

    def test_omitted_recenter_does_not_inspect_model(self):
        run, made = self.make(1., interval=None)
        made['model'].modules = Mock(side_effect=AssertionError('inspection forbidden'))
        run._adopt_engine(made)
        self.assertIs(run.engine, made['engine'])
        made['model'].modules.assert_not_called()
        made['engine'].close.assert_not_called()


if __name__ == '__main__':
    unittest.main()
