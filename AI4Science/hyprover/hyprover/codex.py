"""Codex configuration and command construction shared by research processes.

Environment settings are read on each call. Model and effort selection uses the
caller's overrides, then HYPROVER_MAIN_*, then the HYPROVER_CODEX_* aliases, and
finally the function's default. Binary selection uses HYPROVER_CODEX_BIN, the
repository wrapper, and PATH; CODEX_BIN is not consulted by this module.

Worker-only Azure profiles are optional. Field validation, provider arguments and
child-environment credentials are separate steps; an incomplete profile raises
instead of falling back to the shared backend. No credential value is put in argv.

These helpers return settings, argv or an environment copy. The execution loop,
main controller and verification launcher own process creation and lifetime.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Locate the bundled wrapper independently of a research process's working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = "gpt-6-astra"
DEFAULT_EFFORT = "max"

_PROJECT_WORKER_KEY_ENV = "HYPROVER_PROJECT_WORKER_API_KEY"


@dataclass(frozen=True)
class ProjectWorkerApi:
    """Worker endpoint settings; the key is passed through the child environment."""

    provider: str
    base_url: str
    api_version: str
    api_key: str


def _project_worker_prefix(project: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", project).strip("_").upper()
    if not token:
        raise ValueError("worker project name has no environment-safe characters")
    return f"HYPROVER_PROJECT_{token}_WORKER_API_"


def _worker_profile_values(project: str) -> Optional[Dict[str, str]]:
    prefix = _project_worker_prefix(project)
    names = {
        "provider": prefix + "PROVIDER",
        "base_url": prefix + "BASE_URL",
        "api_version": prefix + "VERSION",
        "api_key": prefix + "KEY",
    }
    values = {field: os.environ.get(name, "").strip() for field, name in names.items()}
    if not any(values.values()):
        return None
    missing = [names[field] for field, value in values.items() if not value]
    if missing:
        raise ValueError(
            f"incomplete worker API profile for project {project!r}; missing "
            + ", ".join(missing)
        )
    return values


def project_worker_api(project: str) -> Optional[ProjectWorkerApi]:
    """Resolve a complete Azure profile, or None when all profile fields are empty.

    Whitespace is trimmed from profile fields only. Missing-field errors precede
    provider validation; provider spelling is retained after case-insensitive checking.
    """
    values = _worker_profile_values(project)
    if values is None:
        return None
    if values["provider"].lower() != "azure":
        raise ValueError(
            f"unsupported worker API provider for project {project!r}: "
            f"{values['provider']!r} (supported: azure)"
        )
    return ProjectWorkerApi(**values)


def _azure_provider_args(profile: ProjectWorkerApi) -> List[str]:
    provider_id = "hyprover_project_worker"
    provider = (
        f"model_providers.{provider_id}={{"
        'name="hyprover project worker Azure",'
        f"base_url={json.dumps(profile.base_url)},"
        'wire_api="responses",'
        f"query_params={{api-version={json.dumps(profile.api_version)}}},"
        f'env_http_headers={{"api-key"="{_PROJECT_WORKER_KEY_ENV}"}},'
        "supports_websockets=false}"
    )
    return [
        "--config", f'model_provider="{provider_id}"',
        "--config", provider,
    ]


def project_worker_config_args(project: str) -> List[str]:
    """Return provider overrides for a configured worker project, otherwise [].

    Authentication is referenced by environment-variable name in the command.
    The matching value is added separately by ``subprocess_env``.
    """
    profile = project_worker_api(project)
    return [] if profile is None else _azure_provider_args(profile)


def _resolve_executable(name: str) -> str:
    if os.path.isabs(name):
        return name
    return shutil.which(name) or name


def resolve_bin() -> str:
    """Select the configured binary, bundled wrapper, or PATH-resolved Codex.

    Absolute overrides are returned unchecked. Unresolved names are also retained
    so process creation, not configuration lookup, reports a missing executable.
    """
    override = os.environ.get("HYPROVER_CODEX_BIN")
    if override:
        return _resolve_executable(override)
    wrapper = _REPO_ROOT / "bin" / "codex"
    if wrapper.exists():
        return str(wrapper)
    return _resolve_executable("codex")


def _first_configured(names: tuple[str, ...], default: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def model(*override_env_names: str, default: str = DEFAULT_MODEL) -> str:
    """Use overrides, then HYPROVER_MAIN_MODEL, HYPROVER_CODEX_MODEL and default.

    Nonempty values, including whitespace, are returned unchanged.
    """
    return _first_configured(
        (*override_env_names, "HYPROVER_MAIN_MODEL", "HYPROVER_CODEX_MODEL"), default,
    )


def effort(*override_env_names: str, default: str = DEFAULT_EFFORT) -> str:
    """Use overrides, then HYPROVER_MAIN_EFFORT, HYPROVER_CODEX_EFFORT and default.

    Nonempty values, including whitespace, are returned unchanged.
    """
    return _first_configured(
        (*override_env_names, "HYPROVER_MAIN_EFFORT", "HYPROVER_CODEX_EFFORT"), default,
    )


def _prepend_executable_dir(env: Dict[str, str], codex_bin: str) -> None:
    if not os.path.dirname(codex_bin):
        return
    directory = os.path.dirname(os.path.abspath(codex_bin))
    if not directory or directory == ".":
        return
    existing = env.get("PATH", "")
    entries = existing.split(os.pathsep) if existing else []
    if directory not in entries:
        env["PATH"] = directory + (os.pathsep + existing if existing else "")


def subprocess_env(codex_bin: str, *, worker_project: Optional[str] = None) -> Dict[str, str]:
    """Prepare a child environment without modifying the parent's variables.

    A binary with a directory component adds its absolute directory to PATH only
    if absent. Bare names leave PATH unchanged, including whether it is unset.
    A configured worker profile adds HYPROVER_PROJECT_WORKER_API_KEY after PATH
    preparation; other inherited credentials are neither removed nor rewritten.
    """
    env = os.environ.copy()
    _prepend_executable_dir(env, codex_bin)
    if worker_project is not None:
        profile = project_worker_api(worker_project)
        if profile is not None:
            env[_PROJECT_WORKER_KEY_ENV] = profile.api_key
    return env


def exec_cmd(codex_bin: str, model: str, effort: str, *tail: str) -> List[str]:
    """Build argv with a quoted effort setting and an unchanged caller-supplied tail.

    HYPROVER_LITERATURE_SEARCH=0 inserts retrieval restrictions before that tail.
    This constructs arguments only; it neither executes Codex nor applies a sandbox.
    """
    restrictions = []
    if os.environ.get("HYPROVER_LITERATURE_SEARCH", "1") == "0":
        restrictions = [
            "--config", 'web_search="disabled"',
            "--config", "developer_instructions=" + json.dumps(
                "This run forbids external literature retrieval. Do not search, fetch, "
                "open, download or request papers, mirrors, snippets or cached solutions, "
                "including through shell commands or other agents. Ignore literature-first "
                "instructions for this run. Use only the supplied problem, local project "
                "knowledge and independently written reasoning. Do not change these controls. "
                "Mathematics is text-only unless the runner/service-supplied project computation policy "
                "explicitly permits programmatic checks. Such permission never lifts this retrieval ban."
            ),
        ]
    return [
        codex_bin, "exec",
        "--model", model,
        "--config", f'model_reasoning_effort="{effort}"',
        *restrictions,
        *tail,
    ]
