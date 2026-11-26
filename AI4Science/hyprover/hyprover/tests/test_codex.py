"""Offline contract tests for Codex configuration, argv and child environments.

Binary lookup uses HYPROVER_CODEX_BIN, the bundled wrapper and PATH, not CODEX_BIN.
Tests preserve model/effort precedence, lazy environment reads, exact provider
arguments, credential placement and PATH behavior. No model or network is used.
Pytest includes fixture-based cases; the standalone runner executes its listed subset.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

from hyprover import codex


@contextlib.contextmanager
def env(**kv):
    """Temporarily set env vars (None deletes), restore after."""
    old = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# All env names the launcher consults — cleared as the baseline so ambient config
# never leaks into a precedence assertion.
_ALL = dict(
    HYPROVER_CODEX_BIN=None, CODEX_BIN=None,
    HYPROVER_MAIN_MODEL=None, HYPROVER_MAIN_EFFORT=None,
    HYPROVER_CODEX_MODEL=None, HYPROVER_CODEX_EFFORT=None,
    HYPROVER_WORKER_MODEL=None,
    HYPROVER_VERIFY_MODEL=None, HYPROVER_VERIFY_EFFORT=None,
)


def test_no_literature_policy_reaches_all_codex_calls():
    with env(HYPROVER_LITERATURE_SEARCH="0"):
        command = codex.exec_cmd("codex", "test-model", "high", "-C", "/test", "prompt")
    assert 'web_search="disabled"' in command
    assert any("Do not search, fetch" in arg for arg in command)
    assert command[-3:] == ["-C", "/test", "prompt"]


# --- resolve_bin precedence ------------------------------------------------- #

def test_resolve_bin_prefers_hyprover_codex_bin_over_alias():
    with env(**{**_ALL, "HYPROVER_CODEX_BIN": "/x/primary", "CODEX_BIN": "/x/alias"}):
        assert codex.resolve_bin() == "/x/primary"


def test_resolve_bin_falls_back_to_wrapper_then_which_then_bare():
    # no env override → the <repo>/bin/codex wrapper wins if it exists, else
    # shutil.which("codex"), else the bare string "codex".
    with env(**_ALL):
        got = codex.resolve_bin()
        import shutil
        wrapper = Path(codex.__file__).resolve().parents[1] / "bin" / "codex"
        if wrapper.exists():
            assert got == str(wrapper)
        else:
            assert got == (shutil.which("codex") or "codex")


def test_resolve_bin_bare_when_nothing_available(monkeypatch=None):
    # simulate: no env, no wrapper, no codex on PATH → bare "codex"
    import shutil
    real_which = shutil.which
    shutil.which = lambda *a, **k: None  # type: ignore[assignment]
    try:
        with env(**_ALL):
            # if the repo happens to ship a bin/codex wrapper, that legitimately
            # wins; only assert the bare-string fallback when no wrapper exists.
            wrapper = Path(codex.__file__).resolve().parents[1] / "bin" / "codex"
            if not wrapper.exists():
                assert codex.resolve_bin() == "codex"
    finally:
        shutil.which = real_which  # type: ignore[assignment]


# --- model / effort precedence ---------------------------------------------- #

def test_model_override_wins_then_neutral_then_default():
    with env(**{**_ALL, "HYPROVER_VERIFY_MODEL": "override-m", "HYPROVER_CODEX_MODEL": "neutral-m"}):
        assert codex.model("HYPROVER_VERIFY_MODEL") == "override-m"
    with env(**{**_ALL, "HYPROVER_CODEX_MODEL": "neutral-m"}):
        assert codex.model("HYPROVER_VERIFY_MODEL") == "neutral-m"
    with env(**_ALL):
        assert codex.model("HYPROVER_VERIFY_MODEL") == codex.DEFAULT_MODEL == "gpt-6-astra"


def test_model_main_wins_over_codex_alias():
    # HYPROVER_MAIN_MODEL is the primary neutral name; HYPROVER_CODEX_MODEL is the
    # back-compat alias used only when HYPROVER_MAIN_MODEL is unset.
    with env(**{**_ALL, "HYPROVER_MAIN_MODEL": "main-m", "HYPROVER_CODEX_MODEL": "legacy-m"}):
        assert codex.model() == "main-m"
    with env(**{**_ALL, "HYPROVER_CODEX_MODEL": "legacy-m"}):
        assert codex.model() == "legacy-m"


def test_effort_override_wins_then_neutral_then_default():
    with env(**{**_ALL, "HYPROVER_VERIFY_EFFORT": "override-e", "HYPROVER_CODEX_EFFORT": "neutral-e"}):
        assert codex.effort("HYPROVER_VERIFY_EFFORT") == "override-e"
    with env(**{**_ALL, "HYPROVER_CODEX_EFFORT": "neutral-e"}):
        assert codex.effort("HYPROVER_VERIFY_EFFORT") == "neutral-e"
    with env(**_ALL):
        assert codex.effort("HYPROVER_VERIFY_EFFORT") == codex.DEFAULT_EFFORT == "max"


def test_first_override_in_order_wins():
    with env(**{**_ALL, "HYPROVER_VERIFY_MODEL": "primary", "HYPROVER_WORKER_MODEL": "other"}):
        # the first listed override name wins
        assert codex.model("HYPROVER_VERIFY_MODEL", "HYPROVER_WORKER_MODEL") == "primary"


# --- subprocess_env --------------------------------------------------------- #

def test_subprocess_env_prepends_dir_for_concrete_path():
    with env(**{**_ALL, "PATH": "/usr/bin:/bin"}):
        out = codex.subprocess_env("/opt/codex/bin/codex")
        assert out["PATH"].split(os.pathsep)[0] == "/opt/codex/bin"
        assert "/usr/bin" in out["PATH"]


def test_subprocess_env_never_injects_cwd_for_bare_codex():
    with env(**{**_ALL, "PATH": "/usr/bin:/bin"}):
        out = codex.subprocess_env("codex")
        # the bare-name fallback has no dir component → PATH is untouched, and the
        # CWD ("" / ".") is NOT injected.
        assert out["PATH"] == "/usr/bin:/bin"
        assert "" not in out["PATH"].split(os.pathsep)
        assert "." not in out["PATH"].split(os.pathsep)


def test_subprocess_env_idempotent_when_dir_already_on_path():
    with env(**{**_ALL, "PATH": "/opt/codex/bin:/usr/bin"}):
        out = codex.subprocess_env("/opt/codex/bin/codex")
        # already present → not duplicated
        assert out["PATH"] == "/opt/codex/bin:/usr/bin"


# --- exec_cmd shape --------------------------------------------------------- #

def test_exec_cmd_shape_quoted_effort_and_verbatim_tail():
    cmd = codex.exec_cmd("/x/codex", "the-model", "xhigh", "-C", "/home", "-")
    assert cmd == [
        "/x/codex", "exec",
        "--model", "the-model",
        "--config", 'model_reasoning_effort="xhigh"',
        "-C", "/home", "-",
    ]


def test_literature_restriction_does_not_override_math_permission():
    import json
    with env(HYPROVER_LITERATURE_SEARCH="0"):
        cmd = codex.exec_cmd("codex", "model", "high", "Trusted project policy allows computation")
        policy = json.loads(next(x.split("=", 1)[1] for x in cmd if x.startswith("developer_instructions=")))
        assert 'web_search="disabled"' in cmd
        assert "forbids external literature retrieval" in policy
        assert "runner/service-supplied project computation policy" in policy
        assert "never lifts this retrieval ban" in policy
        assert "All mathematics must be textual" not in policy


def test_exec_cmd_empty_tail():
    cmd = codex.exec_cmd("codex", "m", "e")
    assert cmd == ["codex", "exec", "--model", "m", "--config", 'model_reasoning_effort="e"']


# --- per-project worker API override (opt-in, fail-closed) ------------------ #

_PW = {
    "HYPROVER_PROJECT_PROJ17_WORKER_API_PROVIDER": None,
    "HYPROVER_PROJECT_PROJ17_WORKER_API_BASE_URL": None,
    "HYPROVER_PROJECT_PROJ17_WORKER_API_VERSION": None,
    "HYPROVER_PROJECT_PROJ17_WORKER_API_KEY": None,
}
_PW_FULL = {
    "HYPROVER_PROJECT_PROJ17_WORKER_API_PROVIDER": "azure",
    "HYPROVER_PROJECT_PROJ17_WORKER_API_BASE_URL": "https://YOUR_RESOURCE.cognitiveservices.azure.com/openai",
    "HYPROVER_PROJECT_PROJ17_WORKER_API_VERSION": "2025-04-01-preview",
    "HYPROVER_PROJECT_PROJ17_WORKER_API_KEY": "secret-key",
}


def test_project_worker_api_unset_is_noop():
    # No HYPROVER_PROJECT_* set → no profile, no extra argv, no injected key. This is
    # the existing-user path; it must be byte-identical to before the feature.
    with env(**_PW):
        assert codex.project_worker_api("Proj17") is None
        assert codex.project_worker_config_args("Proj17") == []
        assert "HYPROVER_PROJECT_WORKER_API_KEY" not in codex.subprocess_env(
            "/x/bin/codex", worker_project="Proj17")


def test_project_worker_api_full_profile_and_injection():
    with env(**_PW_FULL):
        p = codex.project_worker_api("Proj17")
        assert p is not None and p.provider == "azure" and p.api_key == "secret-key"
        args = codex.project_worker_config_args("Proj17")
        assert args[:2] == ["--config", 'model_provider="hyprover_project_worker"']
        assert any("model_providers.hyprover_project_worker=" in a for a in args)
        # the secret enters only via the neutral child env name, never argv
        assert not any("secret-key" in a for a in args)
        env_out = codex.subprocess_env("/x/bin/codex", worker_project="Proj17")
        assert env_out["HYPROVER_PROJECT_WORKER_API_KEY"] == "secret-key"


def test_project_worker_api_partial_fails_closed():
    partial = {**_PW_FULL, "HYPROVER_PROJECT_PROJ17_WORKER_API_KEY": None}
    with env(**partial):
        try:
            codex.project_worker_api("Proj17")
            assert False, "partial profile must raise"
        except ValueError:
            pass


def test_project_worker_api_rejects_non_azure():
    with env(**{**_PW_FULL, "HYPROVER_PROJECT_PROJ17_WORKER_API_PROVIDER": "openai"}):
        try:
            codex.project_worker_api("Proj17")
            assert False, "non-azure provider must raise"
        except ValueError:
            pass


def test_configuration_precedence_is_lazy_and_preserves_whitespace(monkeypatch):
    for resolve, suffix in ((codex.model, "MODEL"), (codex.effort, "EFFORT")):
        for values, expected, queried in [
            ({"FIRST": " "}, " ", ["FIRST"]),
            ({"FIRST": "", "SECOND": "chosen"}, "chosen", ["FIRST", "SECOND"]),
            ({f"HYPROVER_MAIN_{suffix}": "neutral"}, "neutral",
             ["FIRST", "SECOND", f"HYPROVER_MAIN_{suffix}"]),
            ({f"HYPROVER_CODEX_{suffix}": "legacy"}, "legacy",
             ["FIRST", "SECOND", f"HYPROVER_MAIN_{suffix}", f"HYPROVER_CODEX_{suffix}"]),
            ({}, "fallback", ["FIRST", "SECOND", f"HYPROVER_MAIN_{suffix}", f"HYPROVER_CODEX_{suffix}"]),
        ]:
            calls = []
            class Environment(dict):
                def get(self, name, default=None):
                    calls.append(name)
                    return super().get(name, default)
            with monkeypatch.context() as patch:
                patch.setattr(codex.os, "environ", Environment(values))
                assert resolve("FIRST", "SECOND", default="fallback") == expected
            assert calls == queried


def test_provider_command_bytes_and_error_order():
    import pytest

    with env(**{key: "  " + value + "  " for key, value in _PW_FULL.items()}):
        assert codex.project_worker_config_args("Proj17") == [
            "--config", 'model_provider="hyprover_project_worker"', "--config",
            'model_providers.hyprover_project_worker={name="hyprover project worker Azure",'
            'base_url="https://YOUR_RESOURCE.cognitiveservices.azure.com/openai",'
            'wire_api="responses",query_params={api-version="2025-04-01-preview"},'
            'env_http_headers={"api-key"="HYPROVER_PROJECT_WORKER_API_KEY"},supports_websockets=false}',
        ]
    with env(**{**_PW, "HYPROVER_PROJECT_PROJ17_WORKER_API_PROVIDER": "other"}):
        with pytest.raises(ValueError) as caught:
            codex.project_worker_api("Proj17")
        assert str(caught.value) == (
            "incomplete worker API profile for project 'Proj17'; missing "
            "HYPROVER_PROJECT_PROJ17_WORKER_API_BASE_URL, "
            "HYPROVER_PROJECT_PROJ17_WORKER_API_VERSION, HYPROVER_PROJECT_PROJ17_WORKER_API_KEY")
    with env(**{**_PW_FULL, "HYPROVER_PROJECT_PROJ17_WORKER_API_PROVIDER": " AZURE "}):
        assert codex.project_worker_api("Proj17").provider == "AZURE"
    with env(**{key: " \t" for key in _PW}):
        assert codex.project_worker_api("Proj17") is None


def test_binary_resolution_short_circuits_and_preserves_unknown_override(monkeypatch, tmp_path):
    from unittest.mock import Mock

    which = Mock(return_value=None)
    monkeypatch.setattr(codex.shutil, "which", which)
    monkeypatch.setattr(codex, "_REPO_ROOT", tmp_path)
    with env(**{**_ALL, "HYPROVER_CODEX_BIN": "/not-installed/codex"}):
        assert codex.resolve_bin() == "/not-installed/codex"
        which.assert_not_called()
    with env(**{**_ALL, "HYPROVER_CODEX_BIN": "relative/codex"}):
        assert codex.resolve_bin() == "relative/codex"
        which.assert_called_once_with("relative/codex")
    which.reset_mock()
    wrapper = tmp_path / "bin" / "codex"
    wrapper.mkdir(parents=True)
    with env(**_ALL):
        assert codex.resolve_bin() == str(wrapper)
        which.assert_not_called()


def test_subprocess_path_empty_missing_and_existing_positions(monkeypatch):
    for initial, binary, expected in [
        ({}, "codex", {}),
        ({"PATH": ""}, "codex", {"PATH": ""}),
        ({}, "/opt/bin/codex", {"PATH": "/opt/bin"}),
        ({"PATH": ""}, "/opt/bin/codex", {"PATH": "/opt/bin"}),
        ({"PATH": ":/opt/bin:/usr/bin"}, "/opt/bin/codex", {"PATH": ":/opt/bin:/usr/bin"}),
    ]:
        with monkeypatch.context() as patch:
            patch.setattr(codex.os, "environ", initial)
            result = codex.subprocess_env(binary)
            assert result == expected and result is not initial
            assert codex.os.environ == initial


def main() -> None:
    tests = [
        test_resolve_bin_prefers_hyprover_codex_bin_over_alias,
        test_resolve_bin_falls_back_to_wrapper_then_which_then_bare,
        test_resolve_bin_bare_when_nothing_available,
        test_model_override_wins_then_neutral_then_default,
        test_effort_override_wins_then_neutral_then_default,
        test_first_override_in_order_wins,
        test_subprocess_env_prepends_dir_for_concrete_path,
        test_subprocess_env_never_injects_cwd_for_bare_codex,
        test_subprocess_env_idempotent_when_dir_already_on_path,
        test_exec_cmd_shape_quoted_effort_and_verbatim_tail,
        test_exec_cmd_empty_tail,
        test_model_main_wins_over_codex_alias,
        test_project_worker_api_unset_is_noop,
        test_project_worker_api_full_profile_and_injection,
        test_project_worker_api_partial_fails_closed,
        test_project_worker_api_rejects_non_azure,
    ]
    for t in tests:
        t()
        print(f"  [ok] {t.__name__}")
    print("ALL CODEX LAUNCHER TESTS PASSED")


if __name__ == "__main__":
    main()
