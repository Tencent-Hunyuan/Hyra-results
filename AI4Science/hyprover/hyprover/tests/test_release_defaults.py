"""Public-release defaults, user overrides and isolated shell configuration."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from hyprover import codex
from hyprover.execution import layout, loop, scaffold
from hyprover.orchestration import cli

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def clean_defaults(monkeypatch):
    for name in (
        "HYPROVER_MAIN_MODEL", "HYPROVER_CODEX_MODEL", "CODEX_API_MODEL",
        "HYPROVER_WORKER_MODEL", "HYPROVER_MAIN_EFFORT", "HYPROVER_CODEX_EFFORT",
        "HYPROVER_WORKER_EFFORT", "HYPROVER_VERIFY_MODEL", "HYPROVER_VERIFY_EFFORT",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("entry", ["cli", "library"])
def test_new_project_defaults_reach_worker_launch(tmp_path, monkeypatch, clean_defaults, entry):
    monkeypatch.setenv("HYPROVER_AGENTS_ROOT", str(tmp_path / "projects"))
    monkeypatch.setenv("HYPROVER_MCP_TOOL_TIMEOUT", "3600")
    if entry == "cli":
        assert cli.main(["new", "demo"]) == 0
    else:
        scaffold.do_new("demo")
    project = layout.project_dir("demo")
    metadata = json.loads((project / "project.json").read_text())
    expected = ["max", *(f"max{i}" for i in range(2, 9))]
    assert metadata["roles"] == "max:8" and metadata["workers"] == expected
    assert metadata["model"] == "gpt-6-astra"
    assert metadata["allow_math_computation"] is False
    for name in expected:
        worker = layout.WorkerLayout(layout.worker_dir("demo", name))
        role = loop._read_role(worker)
        assert role["MODEL"] == "gpt-6-astra"
        assert role["REASONING_EFFORT"] == "max" and role["ROLE"] == "max"
        assert not worker.pid.exists()
    child = Mock(wait=Mock(return_value=0))
    spawn = Mock(return_value=child)
    monkeypatch.setattr(loop.subprocess, "Popen", spawn)
    assert loop.run_round(worker, role, "Synthetic test", worker.logs / "launch.log", 10) == 0
    argv = spawn.call_args.args[0]
    assert argv[argv.index("--model") + 1] == "gpt-6-astra"
    assert 'model_reasoning_effort="max"' in argv


def test_existing_role_and_explicit_roster_remain_authoritative(tmp_path, monkeypatch, clean_defaults):
    monkeypatch.setenv("HYPROVER_AGENTS_ROOT", str(tmp_path / "projects"))
    scaffold.do_new("demo", roles="high:2", model="custom-model")
    monkeypatch.setenv("HYPROVER_MAIN_MODEL", "new-default")
    monkeypatch.setenv("HYPROVER_MAIN_EFFORT", "max")
    worker = layout.WorkerLayout(layout.worker_dir("demo", "high"))
    assert layout.list_workers("demo") == ["high", "high2"]
    assert loop._read_role(worker)["MODEL"] == "custom-model"
    assert loop._read_role(worker)["REASONING_EFFORT"] == "high"
    monkeypatch.setenv("HYPROVER_WORKER_EFFORT", "max")
    assert loop._read_role(worker)["REASONING_EFFORT"] == "max"
    assert codex.model("HYPROVER_VERIFY_MODEL") == "new-default"


@pytest.fixture
def shell_workspace(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "config").mkdir()
    for name in ("env.sh", "setup-codex.sh"):
        shutil.copyfile(ROOT / "scripts" / name, tmp_path / "scripts" / name)
    environment = {"PATH": os.defpath, "HOME": str(tmp_path), "LANG": "C.UTF-8"}
    return tmp_path, environment


def read_shell_defaults(root, environment):
    result = subprocess.run(
        ["bash", "-c", '. ./scripts/env.sh; printf "%s\\n" "$HYPROVER_MAIN_MODEL" "$HYPROVER_MAIN_EFFORT" "$HYPROVER_WORKER_MODEL"'],
        cwd=root, env=environment, check=True, capture_output=True, text=True,
    )
    return result.stdout.splitlines()


def test_shell_defaults_and_supported_override_chain(shell_workspace):
    root, environment = shell_workspace
    (root / "config" / "unrelated.env").write_text("exit 77\n")
    assert read_shell_defaults(root, environment) == ["gpt-6-astra", "max", "gpt-6-astra"]
    (root / "config" / "codex.env").write_text("CODEX_API_MODEL=backend-model\n")
    assert read_shell_defaults(root, environment) == ["backend-model", "max", "backend-model"]
    (root / "config" / "hyprover.env").write_text("HYPROVER_MAIN_MODEL=override\nHYPROVER_MAIN_EFFORT=high\n")
    assert read_shell_defaults(root, environment) == ["override", "high", "override"]


def test_public_provider_config_uses_environment_reference(shell_workspace):
    root, environment = shell_workspace
    secret = "synthetic-credential-for-tests"
    (root / "config" / "codex.env").write_text(
        "CODEX_API_BASE_URL=https://api.example.com/v1\n"
        f"HYPROVER_CODEX_API_KEY='{secret}'\n"
    )
    subprocess.run(["bash", "scripts/setup-codex.sh", "api"], cwd=root, env=environment,
                   check=True, capture_output=True, text=True)
    config = (root / "runtime" / "codex-home" / "config.toml").read_text()
    assert 'model = "gpt-6-astra"' in config
    assert 'model_reasoning_effort = "max"' in config
    assert 'wire_api = "responses"' in config
    assert 'env_key = "HYPROVER_CODEX_API_KEY"' in config
    assert secret not in config
    assert "developer_instructions" not in config


def test_repository_codex_config_matches_public_defaults():
    config = (ROOT / ".codex" / "config.toml").read_text()
    assert f'model = "{codex.DEFAULT_MODEL}"' in config
    assert f'model_reasoning_effort = "{codex.DEFAULT_EFFORT}"' in config
