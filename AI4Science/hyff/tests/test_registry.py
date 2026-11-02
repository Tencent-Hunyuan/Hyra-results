"""Checkpoint discovery, model identification and default engine options (synthetic indexes, CPU)."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from hyff import registry
from hyff.models import factory

PRESTO_JSON = Path(registry.__file__).resolve().parent / 'checkpoints' / 'presto' / 'weights.json'


def sha(data):
    return hashlib.sha256(data).hexdigest()


# registry.py lives in <root>/src/hyff in a source checkout.
_SOURCE_CHECKOUT = Path(registry.__file__).resolve().parents[1].name == 'src'


class DiscoveryTests(unittest.TestCase):
    """Installed weights: the cache directory ($HYFF_HOME/checkpoints), then a source checkout; explicit weights."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='hyff_registry_')
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        if not PRESTO_JSON.is_file():
            self.skipTest('the source checkout has no checkpoints/presto/weights.json')

    def setup_index(self, payload=b'fake presto weights', digest=None):
        """A packaged index whose presto row lists synthetic weights; returns the patcher context."""
        config = json.loads(PRESTO_JSON.read_text())
        config['weights_sha256'] = sha(payload)
        architecture = json.dumps(config).encode()
        package = self.work / 'package'
        (package / 'presto').mkdir(parents=True)
        (package / 'presto' / 'weights.json').write_bytes(architecture)
        index = dict(format='hyff-checkpoints',
                     release=dict(repository='example/repo', tag='v0', url='https://example.invalid/{repo}/{tag}/{file}'),
                     models=dict(presto=dict(
                         weights='presto/weights.safetensors', sha256=digest or sha(payload),
                         defaults=registry.DEFAULT_OPTIONS['presto'],
                         files=[dict(path='presto/weights.json', size=len(architecture), sha256=sha(architecture)),
                                dict(path='presto/weights.safetensors', size=len(payload),
                                     sha256=digest or sha(payload), asset='presto-weights.safetensors')])))
        (package / 'index.json').write_text(json.dumps(index))
        self.payload, self.architecture = payload, architecture
        return mock.patch.object(registry, 'PACKAGE_CHECKPOINTS', package)

    def home(self, name):
        """A HYFF_HOME holding checkpoints/presto/weights.{safetensors,json}."""
        root = self.work / name
        directory = root / 'checkpoints' / 'presto'
        directory.mkdir(parents=True)
        (directory / 'weights.safetensors').write_bytes(self.payload)
        (directory / 'weights.json').write_bytes(self.architecture)
        return root

    def environment(self, home):
        return mock.patch.dict(os.environ, {'HYFF_HOME': str(home)})

    def test_hyff_home_precedes_source_checkout(self):
        with self.setup_index():
            home = self.home('home')
            with self.environment(home):
                self.assertEqual(registry.checkpoint_dirs()[0], home / 'checkpoints')
                # A source checkout adds <root>/checkpoints after the cache; an installed package does not.
                self.assertEqual(len(registry.checkpoint_dirs()), 2 if _SOURCE_CHECKOUT else 1)
                selected = registry.resolve('presto')
        self.assertEqual(Path(selected.weights), home / 'checkpoints/presto/weights.safetensors')
        self.assertEqual(selected.sha256, sha(b'fake presto weights'))
        self.assertEqual((selected.name, selected.engine), ('presto', 'presto'))

    def test_cache_dir_defaults_to_user_cache(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('HYFF_HOME', None)
            self.assertEqual(registry.cache_dir(), Path('~/.cache/hyff/checkpoints').expanduser())
            source = Path(registry.__file__).resolve().parents[2] / 'checkpoints'
            expected = [registry.cache_dir()] + ([source] if _SOURCE_CHECKOUT else [])
            self.assertEqual(registry.checkpoint_dirs(), expected)

    def test_incomplete_directory_falls_through_to_the_next(self):
        with self.setup_index():
            partial = self.work / 'partial' / 'checkpoints'
            (partial / 'presto').mkdir(parents=True)
            (partial / 'presto' / 'weights.json').write_bytes(self.architecture)   # weights missing
            full = self.home('full') / 'checkpoints'
            with mock.patch.object(registry, 'checkpoint_dirs', return_value=[partial, full]):
                self.assertEqual(registry.installed('presto'), full)
            with mock.patch.object(registry, 'checkpoint_dirs', return_value=[partial]):
                self.assertIsNone(registry.installed('presto'))
                self.assertIsNone(registry.bundled_weights('presto'))

    def test_missing_weights_are_installed_on_first_use(self):
        with self.setup_index():
            empty = self.work / 'empty'
            with self.environment(empty), \
                    mock.patch.object(registry, 'checkpoint_dirs', return_value=[empty / 'checkpoints']), \
                    mock.patch.object(registry, 'install', side_effect=RuntimeError('net')) as fetch:
                with self.assertRaisesRegex(RuntimeError, 'net'):
                    registry.resolve('presto')
            fetch.assert_called_once_with('presto')

    def test_invalid_packaged_index_is_refused(self):
        package = self.work / 'bad'
        package.mkdir()
        (package / 'index.json').write_text(json.dumps({'format': 'other'}))
        with mock.patch.object(registry, 'PACKAGE_CHECKPOINTS', package), \
                self.assertRaisesRegex(ValueError, 'not a HyFF checkpoint index'):
            registry.resolve('presto')

    def test_installed_checksum_mismatch_is_refused(self):
        with self.setup_index(digest='0' * 64):
            home = self.home('mismatch')
            with self.environment(home), self.assertRaisesRegex(ValueError, 'Checksum mismatch.*hyff download'):
                registry.resolve('presto')

    def test_index_cannot_point_outside_the_model_directory(self):
        with self.setup_index():
            index = registry.PACKAGE_CHECKPOINTS / 'index.json'
            data = json.loads(index.read_text())
            for bad in ('../outside.safetensors', 'vivace/weights.safetensors', '/abs/weights.safetensors'):
                data['models']['presto']['files'][1]['path'] = bad
                index.write_text(json.dumps(data))
                with self.subTest(path=bad), self.assertRaisesRegex(ValueError, 'inside the model directory'):
                    registry.model_files('presto')

    def test_explicit_weights_bypass_the_index(self):
        with self.setup_index(digest='0' * 64):    # a broken installed entry is never consulted
            home = self.home('home')
            explicit = self.work / 'mine.safetensors'
            explicit.write_bytes(b'explicit')
            config = json.loads(PRESTO_JSON.read_text())
            config['weights_sha256'] = sha(b'explicit')
            explicit.with_suffix('.json').write_text(json.dumps(config))
            with self.environment(home):
                selected = registry.resolve(None, explicit)
                self.assertEqual((selected.name, selected.engine), ('presto', 'presto'))
                self.assertEqual(selected.sha256, sha(b'explicit'))
                self.assertEqual(registry.resolve(str(explicit)).weights, str(explicit))
                # A directory holding one checkpoint selects it.
                self.assertEqual(registry.resolve(None, self.work / 'home/checkpoints/presto').name, 'presto')

    def test_identify_maps_explicit_weights_to_models(self):
        with self.setup_index():
            home = self.home('home')
            unknown = self.work / 'unknown.safetensors'
            unknown.write_bytes(b'unknown')
            config = json.loads(PRESTO_JSON.read_text())
            config['model_config'] = dict(config['model_config'], hidden=48)
            config['weights_sha256'] = sha(b'unknown')
            unknown.with_suffix('.json').write_text(json.dumps(config))
            charges = self.work / 'charges.safetensors'
            charges.write_bytes(b'charges')
            charges.with_suffix('.json').write_text(json.dumps(dict(
                format=factory.FORMAT, model_id='hyff-efm-charges-w672c1', model_config={})))
            with self.environment(home):
                self.assertEqual(registry.identify(home / 'checkpoints/presto/weights.safetensors'), 'presto')
                self.assertIsNone(registry.identify(unknown))
                self.assertEqual(registry.identify(charges), 'largo')
                unidentified = registry.resolve(None, unknown)
                self.assertIsNone(unidentified.name)
                self.assertEqual(unidentified.engine, 'reference')
                selected = registry.resolve(None, charges)
                self.assertEqual((selected.name, selected.engine), ('largo', 'largo'))
                self.assertEqual(selected.options, {k: v for k, v in registry.DEFAULT_OPTIONS['largo'].items()
                                                    if k != 'engine'})
                with self.assertRaisesRegex(ValueError, 'holds largo weights'):
                    registry.resolve('presto', charges)

    def test_legacy_model_ids_identify_like_current_ids(self):
        for old, new in factory._LEGACY_MODEL_IDS.items():
            with self.subTest(model_id=old):
                self.assertEqual(factory.factory_id(old), factory.factory_id(new))
        legacy = [old for old, new in factory._LEGACY_MODEL_IDS.items() if new == 'hyff-l2']
        with self.setup_index():
            home = self.home('home')
            path = self.work / 'legacy.safetensors'
            shutil.copyfile(home / 'checkpoints/presto/weights.safetensors', path)
            config = json.loads((home / 'checkpoints/presto/weights.json').read_text())
            config.update(model_id=legacy[0], format=factory.LEGACY_FORMATS[0])
            path.with_suffix('.json').write_text(json.dumps(config))
            with self.environment(home):
                self.assertEqual(registry.identify(path), 'presto')


class NameAndOptionTests(unittest.TestCase):
    def test_aliases_are_silent_and_complete(self):
        self.assertEqual(sorted(registry.MODEL_ALIASES.values()), sorted(registry.MODELS))
        for old, new in registry.MODEL_ALIASES.items():
            self.assertEqual(registry.model_name(old), new)
            self.assertNotIn(old, registry.MODELS)
        for name in registry.MODELS:
            self.assertEqual(registry.model_name(name), name)
        self.assertIsNone(registry.model_name(None))
        self.assertIsNone(registry.model_name('unknown'))

    def test_engine_names(self):
        for name in registry.ENGINES:
            self.assertEqual(registry.engine_name(name), name)
        self.assertIsNone(registry.engine_name(None))
        with self.assertRaisesRegex(ValueError, 'Unknown engine'):
            registry.engine_name('fastest')

    def test_default_options(self):
        expected = {
            'presto': dict(engine='presto', precision='tf32x3', statistics='fp32'),
            'vivace': dict(engine='vivace', precision='tf32x3', statistics='fp32'),
            'moderato': dict(engine='moderato', precision='tf32x3', statistics='fp32'),
            'andante': dict(engine='andante', precision='fp32', statistics='fp64', edge_reserve=1.05,
                            capacity_policy='auto'),
            'largo': dict(engine='largo', precision='tf32x3', statistics='fp64', edge_reserve=1.05,
                          long_range='auto', capacity_policy='auto'),
        }
        self.assertEqual(registry.DEFAULT_OPTIONS, expected)

    def test_options_validation(self):
        check = registry._check_options
        good = [('presto', dict(precision='tf32x3', statistics='fp64')),
                ('vivace', dict(precision='fp32', statistics='fp64')),
                ('andante', dict(precision='fp32', statistics='fp64', edge_reserve=1.0, capacity_policy='tiled')),
                ('largo', dict(precision='tf32x3', statistics='fp64', long_range='spme', conservation='fp64',
                               capacity_policy='partial')),
                ('reference', dict(precision='fp32', statistics='fp64'))]
        for engine, options in good:
            with self.subTest(engine=engine, options=options):
                self.assertEqual(check(engine, dict(options)), options)
        bad = [('presto', dict(precision='fp16', statistics='fp32')),
               ('presto', dict(precision='tf32x3', statistics='fp16')),
               ('andante', dict(precision='tf32x3', statistics='fp64')),
               ('largo', dict(precision='fp32', statistics='fp64')),
               ('andante', dict(precision='fp32', statistics='fp32')),
               ('vivace', dict(precision='fp32', statistics='fp32')),
               ('reference', dict(precision='tf32x3', statistics='fp64')),
               ('presto', dict(precision='tf32x3', statistics='fp32', edge_reserve=1.05)),
               ('andante', dict(precision='fp32', statistics='fp64', edge_reserve=0.9)),
               ('andante', dict(precision='fp32', statistics='fp64', edge_reserve=True)),
               ('andante', dict(precision='fp32', statistics='fp64', edge_reserve=float('nan'))),
               ('andante', dict(precision='fp32', statistics='fp64', long_range='dense')),
               ('andante', dict(precision='fp32', statistics='fp64', capacity_policy='partial')),
               ('largo', dict(precision='tf32x3', statistics='fp64', long_range='exact')),
               ('moderato', dict(precision='tf32x3', statistics='fp32', conservation='fp64'))]
        for engine, options in bad:
            with self.subTest(engine=engine, options=options), self.assertRaises(ValueError):
                check(engine, dict(options))

    def test_overrides_win_without_mutating_defaults(self):
        path = registry.bundled_weights('presto')
        if path is None:
            self.skipTest('presto weights are not installed')
        before = json.dumps(registry.DEFAULT_OPTIONS, sort_keys=True)
        selected = registry.resolve('presto', statistics='fp64')
        self.assertEqual(selected.options, dict(precision='tf32x3', statistics='fp64'))
        self.assertEqual(registry.resolve('presto', precision=None).options,
                         dict(precision='tf32x3', statistics='fp32'))
        self.assertEqual(registry.resolve('presto', engine='reference').options,
                         dict(precision='fp32', statistics='fp64'))
        self.assertEqual(json.dumps(registry.DEFAULT_OPTIONS, sort_keys=True), before)
        with self.assertRaises(ValueError):
            registry.resolve('presto', precision='fp16')

    def test_unknown_model_and_bad_weight_suffix(self):
        with self.assertRaisesRegex(ValueError, 'Unknown model'):
            registry.resolve('allegro')
        with self.assertRaisesRegex(ValueError, 'Pass a model name'):
            registry.resolve()
        with tempfile.NamedTemporaryFile(suffix='.bin') as handle:
            with self.assertRaisesRegex(ValueError, 'safetensors'):
                registry.resolve('presto', handle.name)
        with tempfile.NamedTemporaryFile(suffix='.json') as handle:
            with self.assertRaisesRegex(ValueError, 'index.json'):
                registry.resolve('presto', handle.name)


if __name__ == '__main__':
    unittest.main()
