"""Weight file and architecture JSON pairing (symlinks, checksums); synthetic bytes, no real model."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from hyff import registry, weights
from hyff.models import factory


@contextmanager
def cwd(path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class SidecarTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='hyff_sidecar_')
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.storage = self.work / 'storage'
        self.owned = self.work / 'selected'
        self.storage.mkdir()
        self.owned.mkdir()
        self.payload = self.storage / 'payload.safetensors'
        self.data = b'Invented sidecar-test bytes; this is not a model export.'
        self.payload.write_bytes(self.data)
        self.digest = hashlib.sha256(self.data).hexdigest()
        self.selected = self.owned / 'weights.safetensors'
        self.selected.symlink_to(self.payload)

    def config(self, path, marker='hyff-efm-charges-w672c1', **extra):
        value = dict(format=factory.FORMAT, model_id=marker,
                     model_config={factory.CHARGE_BOUND_KEY: 6.824962139129639},
                     weights_sha256=self.digest)
        value.update(extra)
        path.with_suffix('.json').write_text(json.dumps(value))
        return value

    def test_selected_sidecar_wins_when_both_exist(self):
        expected = self.config(self.selected)
        self.config(self.payload, marker='target-must-not-be-selected')
        payload, config = weights.read_config(self.selected)
        self.assertEqual(payload, self.payload)
        self.assertEqual(config, expected)

    def test_target_sidecar_missing_or_invalid_does_not_matter(self):
        expected = self.config(self.selected)
        for invalid in (None, 'invalid JSON'):
            with self.subTest(target=invalid):
                if invalid is not None:
                    self.payload.with_suffix('.json').write_text(invalid)
                self.assertEqual(weights.read_config(self.selected)[1], expected)

    def test_missing_selected_sidecar_never_falls_back(self):
        self.config(self.payload)
        with self.assertRaises(FileNotFoundError):
            weights.read_config(self.selected)

    def test_bad_selected_sidecar_never_falls_back(self):
        self.config(self.payload)
        for value in ('{invalid', json.dumps({'format': 'training-state'})):
            self.selected.with_suffix('.json').write_text(value)
            with self.assertRaises(ValueError):
                weights.read_config(self.selected)

    def test_legacy_format_spelling_is_accepted(self):
        for legacy in factory.LEGACY_FORMATS:
            expected = self.config(self.selected, format=legacy)
            self.assertEqual(weights.read_config(self.selected)[1], expected)

    def test_checksum_authenticates_payload(self):
        self.config(self.selected, weights_sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            weights.read_config(self.selected)

    def test_dangling_symlink_refuses(self):
        self.config(self.selected)
        self.payload.unlink()
        with self.assertRaises(FileNotFoundError):
            weights.read_config(self.selected)
        with self.assertRaises(FileNotFoundError):
            registry.resolve('largo', self.selected)

    def test_relative_selected_path_and_relative_symlink(self):
        self.selected.unlink()
        self.selected.symlink_to(Path('../storage/payload.safetensors'))
        expected = self.config(self.selected)
        with cwd(self.work):
            resolved = registry.resolve(None, Path('selected/weights.safetensors'))
            self.assertEqual(Path(resolved.weights), self.selected)
            self.assertEqual(resolved.name, 'largo')
            self.assertEqual(resolved.sha256, self.digest)
            self.assertEqual(weights.read_config(resolved.weights)[1], expected)

    def test_chain_keeps_outer_sidecar(self):
        middle = self.owned / 'middle.safetensors'
        middle.symlink_to(self.payload)
        self.selected.unlink()
        self.selected.symlink_to(middle)
        expected = self.config(self.selected)
        self.config(middle, marker='intermediate-must-not-be-selected')
        self.assertEqual(weights.read_config(self.selected), (self.payload, expected))

    def test_explicit_weights_keep_selected_path_and_bypass_discovery(self):
        expected = self.config(self.selected)
        with mock.patch.dict(os.environ, {'HYFF_HOME': str(self.work / 'not-used')}):
            result = registry.resolve(None, self.selected)
        self.assertEqual((result.weights, result.sha256), (str(self.selected), self.digest))
        self.assertEqual(weights.read_config(result.weights)[1], expected)

    def test_non_symlink_pair_is_unchanged(self):
        expected = self.config(self.payload)
        self.assertEqual(weights.read_config(self.payload), (self.payload, expected))

    def test_load_model_receives_selected_architecture_and_resolved_payload(self):
        import torch
        expected = self.config(self.selected)
        model = torch.nn.Linear(2, 1, bias=False)
        state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        with mock.patch.object(factory, 'build_model', return_value=model) as build, \
                mock.patch.object(weights, 'read_weights', return_value=state) as read:
            loaded, config = weights.load_model(self.selected, 'cpu')
        read.assert_called_once_with(self.payload)
        build.assert_called_once_with({k: v for k, v in expected.items() if k != 'weights_sha256'})
        self.assertIs(loaded, model)
        self.assertEqual(config, expected)


if __name__ == '__main__':
    unittest.main()
