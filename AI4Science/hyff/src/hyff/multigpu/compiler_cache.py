"""Compiler-cache namespaces for an already selected multi-GPU MD rank.

This module imports only the standard library. It must run before model load
or compilation; setting environment variables cannot retarget existing pools.
Caller-supplied paths are used as given; whether they are on a local
filesystem is not checked. No directory is removed or existing file rewritten.
"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import sys
import threading


CACHE_VARIABLES = ('TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR')
OVERRIDE_VARIABLE = 'HYFF_MULTIGPU_CACHE_PER_RANK'
RUN_ID_VARIABLE = 'TORCHELASTIC_RUN_ID'
UNSAFE_PRELOADED = ('torch._inductor.async_compile', 'torch._inductor.codecache')
_CONFIGURED = None
_LOCK = threading.RLock()


class CacheConfigurationError(ValueError):
    """A bounded public code suitable for the launcher's collective stage."""


def _require(value, code):
    if not value:
        raise CacheConfigurationError(code)


def _compiler_modules(module_names):
    prefixes = ('torch._inductor', 'torch._dynamo', 'triton.compiler', 'triton.runtime')
    return sorted(name for name in module_names if type(name) is str and
        (name in ('torch', 'triton') or any(name == prefix or name.startswith(prefix + '.')
                                          for prefix in prefixes)))


def _namespace(environ, uid, pid):
    _require(type(uid) is int and uid >= 0 and type(pid) is int and pid > 0,
             'MD_COMPILER_CACHE_PROCESS_ID_INVALID')
    run_id = environ.get(RUN_ID_VARIABLE)
    _require(RUN_ID_VARIABLE not in environ or type(run_id) is str, 'MD_COMPILER_CACHE_RUN_ID_INVALID')
    if run_id:
        name = 'run-' + hashlib.sha256(run_id.encode('utf-8', 'surrogatepass')).hexdigest()
        source = 'TORCHELASTIC_RUN_ID_SHA256'
    else:
        name, source = 'pid-' + str(pid), 'PROCESS_PID'
    return name, source


def _requested_paths(environ):
    values = {}
    for name in CACHE_VARIABLES:
        value = environ.get(name)
        _require(name not in environ or type(value) is str and bool(value.strip()) and '\0' not in value,
                 'MD_COMPILER_CACHE_VALUE_INVALID')
        values[name] = value
    return values


def _persistent_root(environ, uid, local_rank):
    """Resolve an absolute user cache home without using a launch identifier."""
    cache_home = environ.get('XDG_CACHE_HOME')
    _require(cache_home is None or type(cache_home) is str and '\0' not in cache_home,
             'MD_COMPILER_CACHE_HOME_INVALID')
    if cache_home and Path(cache_home).is_absolute():
        base, source = Path(cache_home), 'XDG_CACHE_HOME'
    else:
        # The XDG specification treats empty or relative XDG_CACHE_HOME as
        # unset. In particular, never create a cache relative to the cwd.
        home = environ.get('HOME')
        _require(type(home) is str and bool(home.strip()) and '\0' not in home
                 and Path(home).is_absolute(), 'MD_COMPILER_CACHE_HOME_INVALID')
        base, source = Path(home) / '.cache', 'HOME'
    root = base / 'hyff' / 'compile-cache' / ('user-' + str(uid)) / ('rank-' + str(local_rank))
    return root, source


def _mkdir(path):
    path.mkdir(parents=True, exist_ok=True)


def configure_compiler_cache(local_rank, *, selected=True, environ=None,
                             module_names=None, mkdir_fn=None, uid=None, pid=None):
    """Configure only the two compiler-cache variables on a selected MD rank.

    Unset variables get persistent per-user, per-local-rank directories under
    XDG_CACHE_HOME, or HOME/.cache when XDG_CACHE_HOME is empty or relative.
    Launch IDs and process IDs do not change these default paths.
    Caller roots receive a rank suffix; literal {local_rank} templates are
    expanded. HYFF_MULTIGPU_CACHE_PER_RANK=1 preserves non-template caller
    paths verbatim. Nothing is inferred from path names.

    A repeated call accepts only the same rank, process/job, policy and current
    effective cache environment. The keyword inputs (environ, module_names,
    mkdir_fn, uid, pid) exist for CPU tests; the launcher passes only the local rank.
    """
    global _CONFIGURED
    if selected is False:
        return dict(status='NOT_SELECTED', selected=False, environment_changed=False,
                    idempotent=False, existing_pools_retargeted=False)
    _require(selected is True, 'MD_COMPILER_CACHE_SELECTION_INVALID')
    _require(type(local_rank) is int and 0 <= local_rank < 8, 'MD_COMPILER_CACHE_LOCAL_RANK_INVALID')
    environ = os.environ if environ is None else environ
    uid = os.getuid() if uid is None else uid
    pid = os.getpid() if pid is None else pid
    mkdir_fn = _mkdir if mkdir_fn is None else mkdir_fn
    with _LOCK:
        override_value = environ.get(OVERRIDE_VARIABLE)
        _require(OVERRIDE_VARIABLE not in environ or override_value == '1', 'MD_COMPILER_CACHE_OVERRIDE_INVALID')
        override = override_value == '1'
        namespace, namespace_source = _namespace(environ, uid, pid)
        context = dict(local_rank=local_rank, uid=uid, pid=pid, namespace=namespace,
                       namespace_source=namespace_source, override_value=override_value)
        requested = _requested_paths(environ)
        imported = _compiler_modules(tuple(sys.modules) if module_names is None else tuple(module_names))
        if _CONFIGURED is not None:
            _require(context == _CONFIGURED['context'] and requested == _CONFIGURED['effective'],
                     'MD_COMPILER_CACHE_ALREADY_CONFIGURED_DIFFERENTLY')
            result = copy.deepcopy(_CONFIGURED['metadata'])
            result.update(status='ALREADY_CONFIGURED', idempotent=True, environment_changed=False,
                          compiler_modules_already_imported=imported)
            for entry in result['variables'].values():
                entry['environment_changed'] = False
            return result

        default_root, default_home_source = (None, None)
        if any(value is None for value in requested.values()):
            default_root, default_home_source = _persistent_root(environ, uid, local_rank)
        variables = {}
        for name, leaf in zip(CACHE_VARIABLES, ('triton', 'inductor')):
            original = requested[name]
            if original is None:
                effective, source = str(default_root / leaf), 'DEFAULT_PERSISTENT'
            elif '{local_rank}' in original:
                effective, source = original.replace('{local_rank}', str(local_rank)), 'CALLER_TEMPLATE'
                _require('{' not in effective and '}' not in effective, 'MD_COMPILER_CACHE_TEMPLATE_INVALID')
            elif override:
                effective, source = original, 'CALLER_PER_RANK'
            else:
                _require('{' not in original and '}' not in original, 'MD_COMPILER_CACHE_TEMPLATE_INVALID')
                effective, source = str(Path(original) / ('hyff-rank-' + str(local_rank))), 'CALLER_PREFIX'
            variables[name] = dict(source=source, requested=original, effective=effective,
                                   environment_changed=original != effective)
        changed = any(value['environment_changed'] for value in variables.values())
        _require(not changed or not any(name == prefix or name.startswith(prefix + '.')
                     for name in imported for prefix in UNSAFE_PRELOADED),
                 'MD_COMPILER_CACHE_PREEXISTING_INDUCTOR_STATE')

        try:
            for path in dict.fromkeys(value['effective'] for value in variables.values()):
                mkdir_fn(Path(path))
        except Exception as error:
            raise CacheConfigurationError('MD_COMPILER_CACHE_DIRECTORY_CREATE_FAILED') from error
        effective = {name: value['effective'] for name, value in variables.items()}
        try:
            if changed:
                environ.update(effective)
        except Exception as error:
            # Restore only the two variables this helper sets. Directories are
            # intentionally left intact; they may already contain user data.
            for name, original in requested.items():
                if original is None:
                    environ.pop(name, None)
                else:
                    environ[name] = original
            raise CacheConfigurationError('MD_COMPILER_CACHE_ENVIRONMENT_UPDATE_FAILED') from error
        metadata = dict(policy_version=2, status='CONFIGURED', selected=True, local_rank=local_rank,
            namespace=namespace, namespace_source=namespace_source, override=override,
            default_scope='PERSISTENT_PER_USER_RANK',
            default_cache_root=str(default_root) if default_root is not None else None,
            default_cache_home_source=default_home_source,
            variables=variables, environment_changed=changed, configuration_environment_changed=changed,
            compiler_modules_already_imported=imported, idempotent=False,
            existing_pools_retargeted=False, directory_policy='CREATE_IF_MISSING_NO_DELETE',
            caller_per_rank_assertion_unverified=override, requested_values_scope='FIRST_CONFIGURATION',
            gpu_performance_verified=False)
        _CONFIGURED = dict(context=context, effective=effective, metadata=copy.deepcopy(metadata))
        return metadata
