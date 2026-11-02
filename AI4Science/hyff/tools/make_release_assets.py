"""Build the GitHub release assets of the model weights.

Run from a source checkout whose checkpoints/<model>/ directories hold the weights (checkpoints/ is not
part of the repository; the architecture JSON of each model is copied there from src/hyff/checkpoints/
when missing):

    python tools/make_release_assets.py [--repo Tencent-Hunyuan/Hyra-results] [--tag hyff-v0.2.0] [--copy]

1. Writes the checkpoint index that ships with the package, src/hyff/checkpoints/index.json, and the same
   file as checkpoints/index.json: for every model its entry file, the size and sha256 of every file, the release
   asset name of every weight file, and the release location (repository, tag, URL template). The release
   repository and tag are kept from the existing index unless --repo/--tag are given; this is the one place
   where they are set. The architecture JSON of every model is copied to src/hyff/checkpoints/<model>/.
2. Writes dist/release/: one asset per weight file, named hyff-<model>-<file> (one per small model; the
   largo shards plus their index), hyff-manifest.json (a copy of the index) and RELEASE_NOTES.md with the
   model table and the `gh release create` command. Assets are hard links to the checkpoint files (--copy
   copies them). Every asset name starts with ``hyff-`` and the tag with ``hyff-v`` because the release
   repository is shared with other projects.

Rebuild the wheel afterwards so that it carries the new index. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from hyff import __version__  # noqa: E402
from hyff.download import size_text  # noqa: E402
from hyff.registry import DEFAULT_OPTIONS, MODEL_INFO, MODELS  # noqa: E402
from hyff.weights import INDEX_SUFFIX, checkpoint_files, config_path, sha256  # noqa: E402

ASSET_LIMIT = 2 ** 31 - 1        # GitHub release assets must be smaller than 2 GiB
SHARD_LIMIT = 1_900_000_000
URL = 'https://github.com/{repo}/releases/download/{tag}/{file}'
REPOSITORY = 'Tencent-Hunyuan/Hyra-results'
PREFIX = 'hyff-'
# Release-notes table (same numbers as the README model table).
METRICS = {
    'presto': dict(ms_per_step='0.71', ns_per_day='121', force_mae='47.7', energy_mae='5.78'),
    'vivace': dict(ms_per_step='1.27', ns_per_day='68', force_mae='33.5', energy_mae='4.36'),
    'moderato': dict(ms_per_step='4.04', ns_per_day='21', force_mae='16.5', energy_mae='1.36'),
    'andante': dict(ms_per_step='29.5', ns_per_day='2.9', force_mae='7.65', energy_mae='0.620'),
    'largo': dict(ms_per_step='125', ns_per_day='0.69', force_mae='5.79', energy_mae='0.364'),
}
LARGO_STEPS = 720_000


def entry(directory):
    """The weight entry file of one checkpoint directory (sharded index or single file)."""
    indexes = sorted(directory.glob('*' + INDEX_SUFFIX))
    if indexes:
        return indexes[0]
    return directory / 'weights.safetensors'


def build_index(checkpoints, release):
    models = {}
    for name in MODELS:
        weights = entry(checkpoints / name)
        files = []
        for path in checkpoint_files(weights):
            relative = path.relative_to(checkpoints).as_posix()
            row = dict(path=relative, size=path.stat().st_size, sha256=sha256(path))
            if path != config_path(weights):
                row['asset'] = f'{PREFIX}{name}-{path.name}'    # the architecture JSON ships with the package
            if row['size'] >= ASSET_LIMIT or (path.name.endswith('.safetensors') and weights.name.endswith(
                    INDEX_SUFFIX) and row['size'] > SHARD_LIMIT):
                raise SystemExit(f'{relative} is too large for a release asset ({row["size"]} bytes)')
            files.append(row)
        models[name] = dict(weights=weights.relative_to(checkpoints).as_posix(), sha256=sha256(weights),
                            files=files, defaults={k: v for k, v in DEFAULT_OPTIONS[name].items() if k != 'engine'})
    return dict(format='hyff-checkpoints', release=release, models=models)


def place(source, target, copy):
    """Hard-link (or copy) source to target through a temporary name; keep an identical target."""
    if target.exists() and sha256(target) == sha256(source):
        return
    temporary = target.with_name(f'.{target.name}.{os.getpid()}.tmp')
    if copy:
        shutil.copyfile(source, temporary)
    else:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def release_notes(index, assets):
    release = index['release']
    rows = []
    for name in MODELS:
        info = dict(MODEL_INFO[name], **METRICS[name])
        files = [file for file in index['models'][name]['files'] if file.get('asset')]
        size = sum(file['size'] for file in files)
        rows.append(f'| {name} | {info["parameters"]} | {info["active_per_atom"]} | {info["ms_per_step"]} | '
                    f'{info["ns_per_day"]} | {info["force_mae"]} | {info["energy_mae"]} | {size_text(size)} | '
                    + ', '.join(f'`{file["asset"]}`' for file in files) + ' |')
    command = (f'cd dist/release && gh release create {release["tag"]} --repo {release["repository"]} '
               f'--title "HyFF {__version__}" --notes-file RELEASE_NOTES.md \\\n  '
               + ' \\\n  '.join(assets))
    return '\n'.join([
        f'# HyFF {__version__} model weights',
        '',
        'Weights of the five HyFF models (package: `AI4Science/hyff` in this repository). The `hyff` package',
        'downloads them into its cache on first use (`$HYFF_HOME/checkpoints`, default',
        '`~/.cache/hyff/checkpoints`); `hyff download all` fetches all of them ahead of time. Every file is',
        f'checked against the sha256 in `{assets[0]}`, which is the checkpoint index shipped with the package.',
        'To install from another location set `HYFF_MIRROR` to a URL or directory that holds these files under',
        'the same names.',
        '',
        '| Model | Parameters | Active per atom | ms/step | ns/day | Force MAE (meV/Å) | Energy MAE (meV/atom) | Download | Assets |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|',
        *rows,
        '',
        'Speed: 2,586-atom solvated system on one NVIDIA H20, 1 fs NVE with bond constraints, including',
        'neighbour-list rebuilds. Accuracy: full OMol25 validation set (2,762,021 structures).',
        f'largo is distributed as {len([f for f in index["models"]["largo"]["files"] if f.get("asset")]) - 1} '
        'safetensors shards plus their index (Hugging Face sharded layout); these are the EMA weights after '
        f'{LARGO_STEPS:,} optimizer steps.',
        '',
        '## Publishing (maintainers)',
        '',
        'Run in the HyFF source directory that holds the weights, after `python tools/make_release_assets.py`:',
        '',
        '```bash',
        command,
        '```',
        '',
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--repo', help=f'GitHub repository ORGANIZATION/NAME (default: kept from the index, else {REPOSITORY})')
    parser.add_argument('--tag', help='release tag (default: kept from the index)')
    parser.add_argument('--copy', action='store_true', help='copy the weight files instead of hard-linking')
    args = parser.parse_args()
    checkpoints = ROOT / 'checkpoints'
    package = ROOT / 'src/hyff/checkpoints'
    previous = package / 'index.json'
    release = dict(repository=REPOSITORY, tag=f'{PREFIX}v{__version__}', url=URL)
    if previous.is_file():
        release.update(json.loads(previous.read_text()).get('release', {}))
    release.update({key: value for key, value in (('repository', args.repo), ('tag', args.tag)) if value})
    for name in MODELS:
        # Architecture JSONs live in the package; a local weights directory may lack them.
        local, packaged = checkpoints / name / 'weights.json', package / name / 'weights.json'
        if not local.is_file() and packaged.is_file():
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(packaged.read_bytes())
    index = build_index(checkpoints, release)
    text = json.dumps(index, indent=1) + '\n'
    for name, row in index['models'].items():
        architecture = config_path(checkpoints / row['weights'])
        target = package / architecture.relative_to(checkpoints)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or target.read_bytes() != architecture.read_bytes():
            target.write_bytes(architecture.read_bytes())
    previous.write_text(text)
    (checkpoints / 'index.json').write_text(text)   # the same index, next to the weights of a source checkout

    out = ROOT / 'dist/release'
    out.mkdir(parents=True, exist_ok=True)
    manifest = f'{PREFIX}manifest.json'
    assets = [manifest]
    (out / manifest).write_text(text)
    for name, row in index['models'].items():
        for file in row['files']:
            if file.get('asset'):
                place(checkpoints / file['path'], out / file['asset'], args.copy)
                if sha256(out / file['asset']) != file['sha256']:
                    raise SystemExit(f'{file["asset"]} differs from the index')
                assets.append(file['asset'])
    (out / 'RELEASE_NOTES.md').write_text(release_notes(index, assets))
    total = sum((out / asset).stat().st_size for asset in assets)
    print(f'{len(assets)} assets ({size_text(total)}) in {out}; largest '
          f'{max((out / asset).stat().st_size for asset in assets)} bytes')
    print(f'release: {release["repository"]} {release["tag"]}')


if __name__ == '__main__':
    main()
