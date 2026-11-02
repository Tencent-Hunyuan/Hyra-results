"""Sharded checkpoints, the release manifest, mirror URLs and resumable verified downloads (CPU, synthetic data).

The download tests serve files from a local HTTP server on 127.0.0.1 with HTTP Range support.
"""
import hashlib
import http.server
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from hyff import download, registry, weights
from hyff.models import factory

PACKAGE = Path(registry.__file__).resolve().parent / 'checkpoints'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_sharded(directory, state, groups, *, digests=True, name='weights'):
    """Write a sharded checkpoint with an architecture JSON; returns the index path."""
    directory.mkdir(parents=True, exist_ok=True)
    weight_map, files = {}, []
    for k, group in enumerate(groups, 1):
        file = f'{name}-{k:05d}-of-{len(groups):05d}.safetensors'
        save_file({key: state[key] for key in group}, str(directory / file))
        weight_map.update({key: file for key in group})
        files.append(file)
    metadata = dict(total_size=sum(v.numel() * v.element_size() for v in state.values()))
    if digests:
        metadata['sha256'] = {file: sha((directory / file).read_bytes()) for file in files}
    index = directory / f'{name}.safetensors.index.json'
    index.write_text(json.dumps(dict(metadata=metadata, weight_map=weight_map)))
    (directory / f'{name}.json').write_text(json.dumps(dict(format=factory.FORMAT, model_id='hyff-l2', model_config={})))
    return index


class ShardedLoaderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='hyff_shards_')
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        generator = torch.Generator().manual_seed(3)
        self.state = {'a.weight': torch.randn(4, 3, generator=generator),
                      'b.weight': torch.randn(5, generator=generator, dtype=torch.float64),
                      'c.index': torch.arange(7)}
        self.index = write_sharded(self.work / 'ckpt', self.state, [['a.weight'], ['b.weight', 'c.index']])

    def assertSameState(self, loaded):
        self.assertEqual(set(loaded), set(self.state))
        for key, value in self.state.items():
            self.assertEqual(loaded[key].dtype, value.dtype)
            self.assertTrue(torch.equal(loaded[key], value))

    def test_index_and_directory_entries(self):
        self.assertTrue(weights.is_index(self.index))
        self.assertEqual(weights.entry_path(self.index), self.index)
        self.assertEqual(weights.entry_path(self.work / 'ckpt'), self.index)
        self.assertEqual(weights.config_path(self.index), self.work / 'ckpt' / 'weights.json')
        self.assertEqual([p.name for p in weights.shard_paths(self.index)],
                         ['weights-00001-of-00002.safetensors', 'weights-00002-of-00002.safetensors'])
        files = weights.checkpoint_files(self.index)
        self.assertEqual(files[:2], [self.work / 'ckpt' / 'weights.json', self.index])
        self.assertEqual(weights.checkpoint_bytes(self.index), sum(p.stat().st_size for p in files))

    def test_sharded_round_trip_is_bitwise(self):
        self.assertSameState(weights.read_weights(self.index))

    def test_single_file_directory(self):
        single = self.work / 'single'
        single.mkdir()
        save_file(self.state, str(single / 'model.safetensors'))
        (single / 'model.json').write_text('{}')
        self.assertEqual(weights.entry_path(single), single / 'model.safetensors')
        self.assertSameState(weights.read_weights(weights.entry_path(single)))

    def test_directory_without_checkpoint_is_refused(self):
        empty = self.work / 'empty'
        empty.mkdir()
        with self.assertRaises(FileNotFoundError):
            weights.entry_path(empty)

    def test_shard_checksum_is_enforced(self):
        shard = self.work / 'ckpt' / 'weights-00002-of-00002.safetensors'
        save_file({'b.weight': self.state['b.weight'] + 1, 'c.index': self.state['c.index']}, str(shard))
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            weights.read_weights(self.index)

    def test_missing_shard_and_wrong_weight_map(self):
        index = json.loads(self.index.read_text())
        index['weight_map']['extra'] = index['weight_map']['a.weight']
        self.index.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, 'weight_map'):
            weights.read_weights(self.index)
        index['weight_map'].pop('extra')
        index['weight_map']['a.weight'] = 'missing.safetensors'
        index['metadata'].pop('sha256')
        self.index.write_text(json.dumps(index))
        with self.assertRaises((FileNotFoundError, ValueError)):
            weights.read_weights(self.index)

    def test_total_size_is_checked(self):
        index = json.loads(self.index.read_text())
        index['metadata']['total_size'] += 4
        self.index.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, 'total_size'):
            weights.read_weights(self.index)

    def test_shard_names_must_stay_next_to_the_index(self):
        for bad in ('../x.safetensors', 'sub/x.safetensors', 'x.bin'):
            index = dict(metadata={}, weight_map={'a.weight': bad})
            self.index.write_text(json.dumps(index))
            with self.subTest(file=bad), self.assertRaisesRegex(ValueError, 'next to the index'):
                weights.read_shard_index(self.index)

    def test_read_config_and_resolve_accept_sharded_checkpoints(self):
        config = json.loads((self.work / 'ckpt' / 'weights.json').read_text())
        config['weights_sha256'] = sha(self.index.read_bytes())
        (self.work / 'ckpt' / 'weights.json').write_text(json.dumps(config))
        self.assertEqual(weights.read_config(self.index)[1], config)
        self.assertEqual(weights.read_config(self.work / 'ckpt')[0], self.index)
        config['weights_sha256'] = '0' * 64
        (self.work / 'ckpt' / 'weights.json').write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            weights.read_config(self.index)

    def test_resolve_identifies_a_sharded_largo_directory(self):
        config = dict(format=factory.FORMAT, model_id='hyff-efm-charges-w672c1',
                      model_config={factory.CHARGE_BOUND_KEY: 6.824962139129639})
        (self.work / 'ckpt' / 'weights.json').write_text(json.dumps(config))
        for given in (self.work / 'ckpt', self.index, str(self.index)):
            with self.subTest(weights=given):
                selected = registry.resolve(None, given)
                self.assertEqual((selected.name, selected.engine, Path(selected.weights)), ('largo', 'largo', self.index))
                self.assertEqual(selected.sha256, sha(self.index.read_bytes()))
        self.assertEqual(registry.resolve(self.work / 'ckpt').name, 'largo')
        self.assertEqual(registry.resolve('largo', self.work / 'ckpt').name, 'largo')


class LegacyChargeBoundKeyTests(unittest.TestCase):
    def test_earlier_key_spelling_is_read_as_charge_bound(self):
        model_id = 'hyff-efm-charges-w672c1'
        legacy = factory.LEGACY_CHARGE_BOUND_KEYS[0]
        self.assertEqual(factory.validate_model_config(model_id, {legacy: 6.5}), {factory.CHARGE_BOUND_KEY: 6.5})
        self.assertEqual(factory.charge_bound_setting({legacy: 6.5}), 6.5)
        self.assertEqual(factory.charge_bound_setting({factory.CHARGE_BOUND_KEY: 6.5}), 6.5)
        self.assertIsNone(factory.charge_bound_setting({}))
        with self.assertRaisesRegex(ValueError, 'more than once'):
            factory.validate_model_config(model_id, {legacy: 6.5, factory.CHARGE_BOUND_KEY: 6.5})


class ManifestTests(unittest.TestCase):
    """The checkpoint index shipped with the package."""

    def setUp(self):
        self.index = json.loads((PACKAGE / 'index.json').read_text())

    def test_every_model_lists_files_sizes_and_digests(self):
        self.assertEqual(self.index['format'], 'hyff-checkpoints')
        self.assertEqual(sorted(self.index['models']), sorted(registry.MODELS))
        for name, row in self.index['models'].items():
            with self.subTest(model=name):
                paths = [file['path'] for file in row['files']]
                self.assertIn(row['weights'], paths)
                self.assertIn(f'{name}/weights.json', paths)
                entry = next(file for file in row['files'] if file['path'] == row['weights'])
                self.assertEqual(entry['sha256'], row['sha256'])
                for file in row['files']:
                    self.assertRegex(file['sha256'], r'^[0-9a-f]{64}$')
                    self.assertIsInstance(file['size'], int)
                    self.assertLess(file['size'], 2 ** 31)          # release assets must be < 2 GiB
                    self.assertTrue(file['path'].startswith(name + '/'))
                    if file.get('asset'):
                        # The release repository is shared: every asset name starts with hyff-.
                        self.assertEqual(file['asset'], f'hyff-{name}-{Path(file["path"]).name}')
                    else:
                        self.assertTrue(file['path'].endswith('.json'))
                self.assertEqual(row['defaults'], {k: v for k, v in registry.DEFAULT_OPTIONS[name].items()
                                                   if k != 'engine'})

    def test_packaged_architecture_json_matches_the_index(self):
        for name, row in self.index['models'].items():
            architecture = next(file for file in row['files'] if not file.get('asset'))
            data = (PACKAGE / architecture['path']).read_bytes()
            self.assertEqual((len(data), sha(data)), (architecture['size'], architecture['sha256']), name)

    def test_largo_is_sharded_below_two_gib(self):
        row = self.index['models']['largo']
        self.assertTrue(row['weights'].endswith(weights.INDEX_SUFFIX))
        shards = [file for file in row['files'] if re.fullmatch(r'largo/weights-\d{5}-of-\d{5}\.safetensors', file['path'])]
        self.assertGreaterEqual(len(shards), 3)
        self.assertTrue(all(file['size'] <= 1_900_000_000 for file in shards))

    def test_release_location_is_configured_in_one_place(self):
        release = self.index['release']
        self.assertEqual(release['url'], 'https://github.com/{repo}/releases/download/{tag}/{file}')
        self.assertEqual(release['repository'], 'Tencent-Hunyuan/Hyra-results')
        self.assertEqual(release['tag'], 'hyff-v0.2.0')

    def test_packaged_default_url(self):
        # The string only; nothing is contacted.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('HYFF_MIRROR', None)
            url = download.asset_url(self.index['release'], registry.model_files('presto')['files'][1]['asset'])
        self.assertEqual(url, 'https://github.com/Tencent-Hunyuan/Hyra-results/releases/download/hyff-v0.2.0/'
                              'hyff-presto-weights.safetensors')


class MirrorTests(unittest.TestCase):
    RELEASE = dict(repository='Tencent-Hunyuan/Hyra-results', tag='hyff-v0.2.0',
                   url='https://github.com/{repo}/releases/download/{tag}/{file}')

    def test_default_is_the_github_release(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('HYFF_MIRROR', None)
            self.assertEqual(download.asset_url(self.RELEASE, 'hyff-presto-weights.safetensors'),
                             'https://github.com/Tencent-Hunyuan/Hyra-results/releases/download/hyff-v0.2.0/'
                             'hyff-presto-weights.safetensors')

    def test_mirror_replaces_the_base_and_keeps_file_names(self):
        for base in ('https://mirror.example/hyff/hyff-v0.2.0', 'https://mirror.example/hyff/hyff-v0.2.0/'):
            with mock.patch.dict(os.environ, {'HYFF_MIRROR': base}):
                self.assertEqual(download.asset_url(self.RELEASE, 'hyff-largo-weights-00001-of-00003.safetensors'),
                                 'https://mirror.example/hyff/hyff-v0.2.0/hyff-largo-weights-00001-of-00003.safetensors')

    def test_mirror_may_be_a_local_directory(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {'HYFF_MIRROR': directory}):
            url = download.asset_url(self.RELEASE, 'a b.json')
            self.assertEqual(url, Path(directory).resolve().as_uri() + '/a%20b.json')


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Static files with single-range support; can cut a response short to simulate an interruption."""
    cut_after = None          # bytes to send before dropping the connection, once
    requests = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = Path(self.translate_path(self.path))
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        start, status = 0, 200
        match = re.fullmatch(r'bytes=(\d+)-', self.headers.get('Range', ''))
        if match:
            start, status = int(match.group(1)), 206
        type(self).requests.append((self.path, start))
        body = data[start:]
        self.send_response(status)
        self.send_header('Content-Length', str(len(body)))
        if status == 206:
            self.send_header('Content-Range', f'bytes {start}-{len(data) - 1}/{len(data)}')
        self.end_headers()
        cut = type(self).cut_after
        if cut is not None:
            type(self).cut_after = None
            self.wfile.write(body[:cut])
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(body)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='hyff_download_')
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.served = self.work / 'served'
        self.served.mkdir()
        self.payload = os.urandom(3 * download.CHUNK + 12345)
        (self.served / 'm-weights.safetensors').write_bytes(self.payload)
        handler = type('Handler', (_RangeHandler,), dict(requests=[], cut_after=None))
        handler.directory = str(self.served)
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), lambda *a: handler(*a, directory=str(self.served)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.handler = handler
        self.base = f'http://127.0.0.1:{server.server_address[1]}'
        # The local server is reached directly, never through a proxy.
        mirror = mock.patch.dict(os.environ, {'HYFF_MIRROR': self.base, 'no_proxy': '127.0.0.1,localhost',
                                              'NO_PROXY': '127.0.0.1,localhost'})
        mirror.start()
        self.addCleanup(mirror.stop)
        self.release = dict(repository='example/repo', tag='v0', url='https://invalid.example/{repo}/{tag}/{file}')
        self.files = [dict(path='m/weights.safetensors', size=len(self.payload), sha256=sha(self.payload),
                           asset='m-weights.safetensors')]

    def test_download_verifies_and_moves_into_place(self):
        root = self.work / 'cache'
        download.install_files(self.files, self.release, root, self.work / 'unused')
        target = root / 'm' / 'weights.safetensors'
        self.assertEqual(target.read_bytes(), self.payload)
        self.assertFalse(target.with_name(target.name + '.part').exists())
        # Present and verified: no second request.
        count = len(self.handler.requests)
        download.install_files(self.files, self.release, root, self.work / 'unused')
        self.assertEqual(len(self.handler.requests), count)

    def test_interrupted_download_resumes_with_range(self):
        self.handler.cut_after = download.CHUNK + 777
        target = self.work / 'cache' / 'm' / 'weights.safetensors'
        with mock.patch.object(download, 'ATTEMPTS', 1), self.assertRaisesRegex(RuntimeError, 'resumes'):
            download.fetch(f'{self.base}/m-weights.safetensors', target, len(self.payload), sha(self.payload))
        part = target.with_name(target.name + '.part')
        self.assertEqual(part.stat().st_size, download.CHUNK + 777)
        self.assertFalse(target.exists())
        download.fetch(f'{self.base}/m-weights.safetensors', target, len(self.payload), sha(self.payload))
        self.assertEqual(target.read_bytes(), self.payload)
        self.assertEqual(self.handler.requests[-1][1], download.CHUNK + 777)

    def test_corrupted_file_is_rejected(self):
        corrupted = bytearray(self.payload)
        corrupted[1000] ^= 0xFF
        (self.served / 'm-weights.safetensors').write_bytes(bytes(corrupted))
        root = self.work / 'cache'
        with self.assertRaisesRegex(ValueError, 'SHA256.*refused'):
            download.install_files(self.files, self.release, root, self.work / 'unused')
        target = root / 'm' / 'weights.safetensors'
        self.assertFalse(target.exists())
        self.assertTrue(target.with_name(target.name + '.corrupt').is_file())

    def test_missing_asset_reports_http_status(self):
        files = [dict(self.files[0], asset='missing.safetensors')]
        with self.assertRaisesRegex(RuntimeError, 'HTTP 404'):
            download.install_files(files, self.release, self.work / 'cache', self.work / 'unused')

    def test_local_directory_mirror(self):
        with mock.patch.dict(os.environ, {'HYFF_MIRROR': str(self.served)}):
            download.install_files(self.files, self.release, self.work / 'cache', self.work / 'unused')
        self.assertEqual((self.work / 'cache/m/weights.safetensors').read_bytes(), self.payload)

    def test_registry_installs_on_first_use(self):
        """resolve(name) without installed weights downloads into $HYFF_HOME and returns the cached entry."""
        package = self.work / 'package'
        (package / 'm').mkdir(parents=True)
        architecture = json.dumps(dict(format=factory.FORMAT, model_id='hyff-efm-charges-w672c1',
                                       model_config={})).encode()
        (package / 'm' / 'weights.json').write_bytes(architecture)
        index = dict(format='hyff-checkpoints', release=self.release, models=dict(largo=dict(
            weights='largo/weights.safetensors', sha256=sha(self.payload), defaults={},
            files=[dict(path='largo/weights.json', size=len(architecture), sha256=sha(architecture)),
                   dict(self.files[0], path='largo/weights.safetensors')])))
        shutil.move(package / 'm', package / 'largo')
        (package / 'index.json').write_text(json.dumps(index))
        home = self.work / 'home'
        with mock.patch.object(registry, 'PACKAGE_CHECKPOINTS', package), \
                mock.patch.object(registry, 'checkpoint_dirs', return_value=[home / 'checkpoints']), \
                mock.patch.dict(os.environ, {'HYFF_HOME': str(home)}):
            self.assertIsNone(registry.installed('largo'))
            selected = registry.resolve('largo')
            self.assertEqual(Path(selected.weights), home / 'checkpoints/largo/weights.safetensors')
            self.assertEqual(registry.installed('largo'), home / 'checkpoints')
            self.assertEqual((home / 'checkpoints/largo/weights.json').read_bytes(), architecture)


if __name__ == '__main__':
    unittest.main()
