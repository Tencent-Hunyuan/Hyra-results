"""Download model weights from the release assets into the local cache.

The URL of an asset follows the ``release`` block of the checkpoint index
(``https://github.com/{repo}/releases/download/{tag}/{file}``), or is ``$HYFF_MIRROR/<file>`` when
``HYFF_MIRROR`` is set (an http(s) base URL or a local directory holding the same files). Each file is
written to ``<name>.part`` next to its final place; an interrupted download resumes with an HTTP Range
request, the size and sha256 are checked, and the file is then renamed into place. A file that fails the
check is moved aside to ``<name>.corrupt`` and the download is refused.
"""
from __future__ import annotations

import fcntl
import http.client
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from hyff.weights import sha256

CHUNK = 1 << 20
ATTEMPTS = 3
TIMEOUT = 60


def asset_url(release, asset):
    """URL of one release asset, honouring HYFF_MIRROR."""
    name = urllib.parse.quote(asset)
    mirror = os.environ.get('HYFF_MIRROR')
    if mirror:
        if '://' not in mirror:
            mirror = Path(mirror).expanduser().absolute().as_uri()
        return mirror.rstrip('/') + '/' + name
    return release['url'].format(repo=release['repository'], tag=release['tag'], file=name)


def size_text(size):
    return f'{size / 1e9:.2f} GB' if size >= 1e9 else f'{size / 1e6:.1f} MB'


def _say(text, end='\n'):
    print(text, end=end, file=sys.stderr, flush=True)


class _Progress:
    """A short status line per file, updated in place on a terminal."""

    def __init__(self, label, total, start):
        self.label, self.total, self.tty, self.last = label, total, sys.stderr.isatty(), 0.0
        resume = f', resuming at {size_text(start)}' if start else ''
        _say(f'hyff: downloading {label} ({size_text(total)}{resume})', end='' if self.tty else '\n')

    def update(self, done):
        now = time.monotonic()
        if self.tty and now - self.last > 0.25:
            self.last = now
            _say(f'\rhyff: downloading {self.label} {size_text(done)} / {size_text(self.total)} '
                 f'({100 * done / self.total:.0f}%)', end='')

    def close(self):
        if self.tty:
            _say('')


def _transfer(url, part, size, label):
    have = part.stat().st_size if part.is_file() else 0
    if have > size:
        have = 0
    if have == size:
        return
    request = urllib.request.Request(url, headers={'User-Agent': 'hyff'})
    if have:
        request.add_header('Range', f'bytes={have}-')
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        if have:
            content_range = response.headers.get('Content-Range', '') if getattr(response, 'status', None) == 206 else ''
            if not content_range.startswith(f'bytes {have}-'):
                have = 0  # the server sent the whole file
        progress = _Progress(label, size, have)
        done = have
        with part.open('ab' if have else 'wb') as stream:
            while chunk := response.read(CHUNK):
                done += len(chunk)
                if done > size:
                    raise ValueError(f'{label}: the server sent more than the expected {size} bytes')
                stream.write(chunk)
                progress.update(done)
            stream.flush()
            os.fsync(stream.fileno())
        progress.close()
    if done != size:
        raise ConnectionError(f'incomplete transfer ({done} of {size} bytes)')


def fetch(url, target, size, digest, label=None):
    """Download ``url`` to ``target`` through ``target.part`` (resumed), check size and sha256, rename."""
    target = Path(target)
    label = label or target.name
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + '.part')
    for attempt in range(1, ATTEMPTS + 1):
        try:
            _transfer(url, part, size, label)
            break
        except urllib.error.HTTPError as error:
            if error.code < 500 or attempt == ATTEMPTS:
                raise RuntimeError(f'Could not download {label}: HTTP {error.code}') from None
        except (urllib.error.URLError, http.client.HTTPException, OSError) as error:
            if attempt == ATTEMPTS:
                raise RuntimeError(f'Could not download {label}: {error}. The partial file is kept and the '
                                   'next attempt resumes it.') from None
            _say(f'hyff: {label}: {error}; retrying')
            time.sleep(attempt)
    if part.stat().st_size != size or sha256(part) != digest:
        rejected = target.with_name(target.name + '.corrupt')
        os.replace(part, rejected)
        raise ValueError(f'{label}: size or SHA256 differs from the release manifest; download refused '
                         f'(kept as {rejected})')
    os.replace(part, target)
    _say(f'hyff: {label} verified (sha256)')
    return target


@contextmanager
def locked(directory):
    """Serialize installs of one model directory between processes."""
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / '.lock', 'a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def install_files(files, release, root, package_root):
    """Make every listed file present and verified under ``root``.

    Files with an ``asset`` are downloaded; the others (architecture JSON) are copied from the package.
    """
    for file in files:
        target = root / file['path']
        if target.is_file() and sha256(target) == file['sha256']:
            continue
        if file.get('asset'):
            fetch(asset_url(release, file['asset']), target, file['size'], file['sha256'], file['asset'])
            continue
        data = (package_root / file['path']).read_bytes()
        temporary = target.with_name(target.name + '.part')
        temporary.write_bytes(data)
        if sha256(temporary) != file['sha256']:
            raise ValueError(f'The packaged {file["path"]} differs from the checkpoint index')
        os.replace(temporary, target)
    return root
