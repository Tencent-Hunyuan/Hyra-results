"""Per-run compiler/codegen cache directory scope for multi-GPU sessions."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import os
from pathlib import Path
import threading

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_CACHE_DIRECTORY = ContextVar('hyff_multigpu_cache', default=None)
_CONSTRUCTION_LOCK = threading.RLock()


def validate_cache_directory(directory):
    """Require an existing, writable, absolute directory outside the package."""
    if not isinstance(directory, (str, Path)):
        raise ValueError('MULTIGPU_CACHE_DIRECTORY_REQUIRED')
    path = Path(directory)
    if (not path.is_absolute() or path.absolute() != path.resolve() or
            path.is_relative_to(PACKAGE_ROOT) or not path.is_dir() or
            path.stat().st_uid != os.geteuid() or not os.access(path, os.W_OK | os.X_OK)):
        raise ValueError('MULTIGPU_CACHE_DIRECTORY_INVALID')
    return path


def require_cache_directory(directory):
    path = validate_cache_directory(directory)
    if _CACHE_DIRECTORY.get() != path:
        raise ValueError('MULTIGPU_CACHE_CONTEXT_REQUIRED')
    return path


@contextmanager
def cache_configuration(directory):
    path = validate_cache_directory(directory)
    with _CONSTRUCTION_LOCK:
        token = _CACHE_DIRECTORY.set(path)
        try:
            yield path
        finally:
            _CACHE_DIRECTORY.reset(token)


@contextmanager
def owned_codegen_cache(module):
    """Point a code-generating module at the session cache directory while it builds kernels."""
    path = require_cache_directory(_CACHE_DIRECTORY.get())
    with _CONSTRUCTION_LOCK:
        for value in module._CACHE.values():
            origin = getattr(value, '__file__', None)
            if not isinstance(origin, str) or not Path(origin).resolve().is_relative_to(path):
                raise ValueError('MULTIGPU_EXISTING_CODEGEN_CACHE_MISMATCH')
        previous = module.GEN_DIR
        module.GEN_DIR = path
        try:
            yield path
        finally:
            module.GEN_DIR = previous
