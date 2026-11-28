"""Offline tests for hyprover.observability — the CLI entry (main / __main__) and the
loader guard branches. Zero network; uvicorn is faked, no server is bound.

Run with pytest; this module has no standalone test runner.
"""

from __future__ import annotations

import contextlib
import os
import runpy
import sys
import tempfile
import types
import warnings
from pathlib import Path
from unittest.mock import patch
from hyprover.core import KnowledgeStore

import hyprover.observability.app  # ensure the submodule is imported into sys.modules

# The package __init__ re-exports the FastAPI instance as hyprover.observability.app,
# which shadows the submodule on attribute access — so reach the MODULE via sys.modules.
app = sys.modules["hyprover.observability.app"]


@contextlib.contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        yield
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


@contextlib.contextmanager
def _argv(*args):
    old = sys.argv[:]
    sys.argv = ["dashboard", *args]
    try:
        yield
    finally:
        sys.argv = old


def test_project_dir_unset_raises():
    with _env(HYPROVER_DASHBOARD_PROJECT=None, HYPROVER_PROJECT_DIR=None):
        try:
            app._project_dir()
            assert False, "unset project dir should raise"
        except RuntimeError:
            pass


def test_main_project_not_found_fails_fast():
    with tempfile.TemporaryDirectory() as d:
        missing = str(Path(d) / "nope")
        with _env(HYPROVER_DASHBOARD_PROJECT=missing), _argv():
            try:
                app.main()
                assert False, "a missing project dir should SystemExit"
            except SystemExit as e:
                assert "project dir not found" in str(e)


def test_main_happy_path_runs_uvicorn():
    fake = types.ModuleType("uvicorn")
    calls = {}
    fake.run = lambda application, **kw: calls.update(kw, application=application)
    with tempfile.TemporaryDirectory() as d, _env(HYPROVER_DASHBOARD_PROJECT=None), _argv("--project", d, "--host", "127.0.0.1", "--port", "9137"):
        KnowledgeStore.initialize(Path(d), "P")
        sys.modules["uvicorn"] = fake
        try:
            app.main()
        finally:
            sys.modules.pop("uvicorn", None)
    assert calls.get("host") == "127.0.0.1" and calls.get("port") == 9137
    assert calls.get("application") is app.app


def test_module_entrypoint_runs_main():
    orig = app.main
    ran = {"v": False}
    app.main = lambda: ran.__setitem__("v", True)
    try:
        runpy.run_module("hyprover.observability", run_name="__main__")
    finally:
        app.main = orig
    assert ran["v"] is True


def test_app_script_entrypoint_runs_main_without_module_reload():
    fake = types.ModuleType("uvicorn")
    calls = []
    fake.run = lambda application, **kw: calls.append((application, kw))
    original_application = app.app
    with tempfile.TemporaryDirectory() as d, _env(HYPROVER_DASHBOARD_PROJECT=d), \
            _argv("--host", "127.0.0.1", "--port", "9138"), \
            patch.dict(sys.modules, {"uvicorn": fake}), warnings.catch_warnings():
        KnowledgeStore.initialize(Path(d), "P")
        warnings.simplefilter("error", RuntimeWarning)
        executed = runpy.run_path(str(Path(app.__file__).resolve()), run_name="__main__")
    assert calls == [(executed["app"], {"host": "127.0.0.1", "port": 9138, "log_level": "warning"})]
    assert sys.modules["hyprover.observability.app"] is app
    assert app.app is original_application
