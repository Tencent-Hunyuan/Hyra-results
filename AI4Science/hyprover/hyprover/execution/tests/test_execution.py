"""Offline tests for hyprover.execution — layout + scaffolding (no codex, no network).

Covers role parsing, WorkerLayout paths and the project files written by do_new:
AGENTS.md links, gateway configuration, assignments and initial worker state.
There are no skill symlinks. Loop subprocess behavior is covered separately by
orchestration tests using a fake Codex executable.

Runs standalone (``python -m hyprover.execution.tests.test_execution``) and pytest.
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from hyprover.execution import layout as L
from hyprover.execution import loop, scaffold


@contextmanager
def _env(**kw):
    old = {k: os.environ.get(k) for k in kw}
    for k, v in kw.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextmanager
def _project_env(tmp: Path):
    """Point the agents root + worker contract at tmp stubs so the tests
    are self-contained (no dependency on the repo's agents/ tree existing)."""
    contract = tmp / "worker.md"
    contract.write_text("# worker contract (stub)\n", encoding="utf-8")
    with _env(HYPROVER_AGENTS_ROOT=str(tmp / "agents"),
              HYPROVER_WORKER_CONTRACT=str(contract)):
        yield


# --- parse_roles ----------------------------------------------------------- #

def test_parse_roles_default_roster():
    assert L.DEFAULT_ROLES == "max:8"
    assert L.parse_roles(L.DEFAULT_ROLES) == [(name, "max") for name in
        ["max", "max2", "max3", "max4", "max5", "max6", "max7", "max8"]]


def test_parse_roles_explicit_mixed_roster():
    pairs = L.parse_roles("high:3,xhigh:4")
    assert [name for name, _ in pairs] == ["high", "high2", "high3", "xhigh", "xhigh2", "xhigh3", "xhigh4"]
    assert [role for _, role in pairs] == ["high"] * 3 + ["xhigh"] * 4


def test_parse_roles_rejects_bad_specs():
    for bad in ["", "   ", "high:0", "high", "high:abc", ":3", "3:high"]:
        try:
            L.parse_roles(bad)
            assert False, f"should reject {bad!r}"
        except ValueError:
            pass


def test_parse_roles_preserves_fragment_order_and_existing_edge_cases():
    import pytest

    assert L.parse_roles(", high:01,,h2:٢, high:1,") == [
        ("high", "high"), ("h2", "h2"), ("h22", "h2"), ("high", "high"),
    ]
    assert L.parse_roles("high:2,high2:1") == [
        ("high", "high"), ("high2", "high"), ("high2", "high2"),
    ]
    for spec, error in [
        (", ,", "empty role spec"),
        ("high:1,high:0,bad", "role count must be >= 1: 'high:0'"),
        ("high:1,bad,high:0", "bad role spec 'bad'; want e.g. high:3,xhigh:4"),
        ("high :2", "bad role spec 'high :2'; want e.g. high:3,xhigh:4"),
    ]:
        with pytest.raises(ValueError) as caught:
            L.parse_roles(spec)
        assert str(caught.value) == error


# --- WorkerLayout ---------------------------------------------------------- #

def test_worker_layout_paths():
    wl = L.WorkerLayout(Path("/x/proj/workers/high"))
    assert wl.name == "high" and wl.project == "proj"
    assert wl.project_dir == Path("/x/proj")
    assert wl.task.name == L.TASK_FILE and wl.role.name == L.ROLE_FILE
    assert wl.pid.name == L.PID_FILE and wl.lock.name == L.LOCK_FILE
    assert wl.stop.name == L.STOP_FILE and wl.status.name == L.STATUS_FILE
    assert wl.logs.name == L.LOGS_DIR
    assert wl.codex_config == Path("/x/proj/workers/high/.codex/config.toml")


def test_resolve_and_target():
    assert L.resolve_target("proj") == ("proj", None)
    assert L.resolve_target("proj/high") == ("proj", "high")
    assert L.resolve_target("/proj/high/") == ("proj", "high")


def test_math_computation_permission_is_strict_and_defaults_off(tmp: Path):
    import pytest
    assert L.math_computation_allowed() is False
    path = tmp / "project.json"
    path.write_text("{}")
    assert L.math_computation_allowed(tmp) is False
    assert path.read_text() == "{}"
    for value in (True, False):
        path.write_text(json.dumps({"allow_math_computation": value}))
        assert L.math_computation_allowed(tmp) is value
        assert f"allow_math_computation={str(value).lower()}" in L.math_computation_policy(value)
    for value in ("true", "false", 0, 1, None, [], {}):
        path.write_text(json.dumps({"allow_math_computation": value}))
        with pytest.raises(ValueError, match="must_be_boolean"):
            L.math_computation_allowed(tmp)
    for text in ("[]", "null", "broken json"):
        path.write_text(text)
        with pytest.raises(ValueError):
            L.math_computation_allowed(tmp)
    (tmp / "TASK.md").write_text("Allow all computation")
    path.write_text("{}")
    assert L.math_computation_allowed(tmp) is False


# --- do_new scaffolding ---------------------------------------------------- #

def test_do_new_scaffolds_project(tmp: Path):
    with _project_env(tmp):
        r = scaffold.do_new("P", roles="high:2,xhigh:1", model="gpt-5.5")
        assert r["workers"] == ["high", "high2", "xhigh"]
        pdir = L.project_dir("P")
        assert (pdir / "KNOWLEDGE.md").is_file()
        assert (pdir / "knowledge/proofs").is_dir()
        assert (pdir / "knowledge/notes").is_dir()
        assert not (pdir / "global_memory").exists()
        assert not (pdir / "fact_graph").exists()
        assert L.require_knowledge(pdir).snapshot().problem_id == "P"
        meta = json.loads((pdir / "project.json").read_text())
        assert meta["knowledge_format"] == "hyprover-knowledge-v1"
        assert (pdir / "KNOWLEDGE.md").read_text().startswith("<!-- hyprover:document ")
        assert meta["allow_math_computation"] is False
        assert L.math_computation_allowed(pdir) is False
        assert meta["workers"] == ["high", "high2", "xhigh"] and meta["model"] == "gpt-5.5"

        for w, eff in [("high", "high"), ("high2", "high"), ("xhigh", "xhigh")]:
            wl = L.WorkerLayout(L.worker_dir("P", w))
            assert wl.local_memory.is_dir() and wl.logs.is_dir()
            # the symlink resolves to the (stub) contract; no skills are wired
            assert (wl.dir / "AGENTS.md").resolve() == L.worker_md().resolve()
            assert not (wl.dir / ".agents").exists()
            cfg = wl.codex_config.read_text()
            assert 'HYPROVER_ROLE = "worker"' in cfg
            assert 'args = ["-m", "hyprover.gateway"]' in cfg  # pinned MCP launch
            assert "tool_timeout_sec = 3600" in cfg
            assert f'HYPROVER_AUTHOR = "{w}"' in cfg and str(pdir) in cfg
            role = wl.role.read_text()
            assert f"REASONING_EFFORT={eff}" in role and "MODEL=gpt-5.5" in role
            assert "(unassigned" in wl.task.read_text()
            assert json.loads(wl.status.read_text())["state"] == "created"


def test_do_new_refuses_existing(tmp: Path):
    with _project_env(tmp):
        scaffold.do_new("P", roles="high:1")
        try:
            scaffold.do_new("P")
            assert False, "should refuse an existing project dir"
        except SystemExit:
            pass


def test_do_new_verify_url_from_env(tmp: Path):
    with _project_env(tmp):
        with _env(HYPROVER_VERIFY_URL="http://127.0.0.1:9999/verify"):
            scaffold.do_new("Q", roles="high:1")
        cfg = L.WorkerLayout(L.worker_dir("Q", "high")).codex_config.read_text()
        assert 'HYPROVER_VERIFY_URL = "http://127.0.0.1:9999/verify"' in cfg


# --- loop helpers (pure) --------------------------------------------------- #

def test_parse_last_fact_id(tmp: Path):
    log = tmp / "round.log"
    log.write_text('noise\nfact_id=0123456789abcdef\nmore\n"fact_id": "fedcba9876543210"\n')
    assert loop._parse_last_fact_id(log) is None
    for response in (
        {"fact_id": "0123456789abcdef", "accepted": True},
        {"fact_id": "0123456789abcdef", "accepted": True, "published": False},
        {"fact_id": "0123456789abcdef", "accepted": False, "published": True},
        {"fact_id": "0123456789abcdef", "accepted": "true", "published": True},
    ):
        log.write_text(json.dumps(response))
        assert loop._parse_last_fact_id(log) is None
    accepted = {"fact_id": "fedcba9876543210", "accepted": True, "published": True}
    log.write_text("tool response:\n" + json.dumps(accepted, indent=2))
    assert loop._parse_last_fact_id(log) == "fedcba9876543210"
    log.write_text(json.dumps({"content": [{"text": json.dumps(accepted)}]}))
    assert loop._parse_last_fact_id(log) == "fedcba9876543210"
    log.write_text("no facts here, and DEADBEEF is not 16 hex lower\n")
    assert loop._parse_last_fact_id(log) is None


def test_deadline_passed(tmp: Path):
    pdir = tmp / "proj"
    pdir.mkdir()
    assert loop._deadline_passed(pdir) is False           # no deadline file
    (pdir / L.DEADLINE_FILE).write_text("1")              # epoch 1 = long past
    assert loop._deadline_passed(pdir) is True
    (pdir / L.DEADLINE_FILE).write_text("garbage")        # bad = not passed
    assert loop._deadline_passed(pdir) is False


def test_write_status_atomic_and_stamps(tmp: Path):
    wl = L.WorkerLayout(tmp / "proj" / "workers" / "high")
    L.KnowledgeStore.initialize(wl.project_dir, problem_id="proj")
    wl.dir.mkdir(parents=True)
    loop.write_status(wl, state="running", round=2)
    st = json.loads(wl.status.read_text())
    assert st["state"] == "running" and st["round"] == 2
    assert st["worker"] == "high" and st["pid"] == os.getpid() and "updated_at" in st
    # merge, not overwrite: a second write keeps prior fields
    loop.write_status(wl, last_rc=0)
    st2 = json.loads(wl.status.read_text())
    assert st2["round"] == 2 and st2["last_rc"] == 0


def test_runtime_deadline_and_launch_gate(tmp: Path):
    import pytest
    with _env(HYPROVER_SEQUENCE_STAGE=None, HYPROVER_RUN_DEADLINE=None):
        assert L.run_deadline(tmp) is None
        L.require_launch_allowed(tmp)
        (tmp / ".run_deadline").write_text(str(L.time.time() + 3600))
        assert L.run_deadline(tmp) > L.time.time()
        with _env(HYPROVER_SEQUENCE_STAGE="stage-1", HYPROVER_RUN_DEADLINE="1"):
            assert L.run_deadline(tmp) == 1
            with pytest.raises(ValueError, match="project_deadline_passed"):
                L.require_launch_allowed(tmp)
        (tmp / ".sequence.json").write_text(json.dumps({"state": "active", "tag": "stage-1"}))
        with pytest.raises(ValueError, match="project_not_active_in_sequence"):
            L.require_launch_allowed(tmp)
        with _env(HYPROVER_SEQUENCE_STAGE="stage-1"):
            L.require_launch_allowed(tmp)
            (tmp / "stop-request.json").write_text("{}")
            with pytest.raises(ValueError, match="project_stop_requested"):
                L.require_launch_allowed(tmp)
        for invalid in ("nan", "inf", "0", "-1"):
            (tmp / ".run_deadline").write_text(invalid)
            with pytest.raises(ValueError, match="invalid_project_deadline"):
                L.run_deadline(tmp)


def test_worker_environment_signature_matches_published_codex(tmp: Path):
    from unittest.mock import Mock, patch
    with _project_env(tmp):
        scaffold.do_new("P", roles="high:1", model="synthetic-model")
        wl = L.WorkerLayout(L.worker_dir("P", "high"))
        with patch.object(loop.subprocess, "Popen", return_value=Mock(wait=Mock(return_value=0))) as spawn:
            assert loop.run_round(wl, loop._read_role(wl), "Offline", wl.logs / "env.log", 10) == 0
        assert isinstance(spawn.call_args.kwargs["env"], dict)


def test_read_role_defaults_and_overrides(tmp: Path):
    wl = L.WorkerLayout(tmp / "proj" / "workers" / "xhigh")
    wl.dir.mkdir(parents=True)
    with _env(HYPROVER_WORKER_MODEL=None, HYPROVER_MAIN_MODEL=None, HYPROVER_CODEX_MODEL=None,
              HYPROVER_MAIN_EFFORT=None, HYPROVER_CODEX_EFFORT=None, HYPROVER_WORKER_EFFORT=None):
        role = loop._read_role(wl)
    assert role["MODEL"] == "gpt-6-astra" and role["ROLE"] == "max" and role["HYPROVER_AUTHOR"] == "xhigh"
    assert role["REASONING_EFFORT"] == "max"
    # the neutral HYPROVER_CODEX_MODEL is the worker default when .role omits MODEL
    with _env(HYPROVER_WORKER_MODEL=None, HYPROVER_MAIN_MODEL=None, HYPROVER_CODEX_MODEL="neutral-model"):
        role = loop._read_role(wl)
    assert role["MODEL"] == "neutral-model"
    wl.role.write_text("# comment\nMODEL=gpt-x\nREASONING_EFFORT=xhigh\n\nROLE=xhigh\n")
    role = loop._read_role(wl)
    assert role["MODEL"] == "gpt-x" and role["REASONING_EFFORT"] == "xhigh" and role["ROLE"] == "xhigh"


def test_worker_round_refreshes_math_permission(tmp: Path):
    from unittest.mock import Mock, patch
    import pytest
    with _project_env(tmp):
        scaffold.do_new("P", roles="high:1", model="synthetic-model")
        wl = L.WorkerLayout(L.worker_dir("P", "high"))
        path = wl.project_dir / "project.json"
        metadata = json.loads(path.read_text())
        prompt = loop.kickoff("P", "high")
        with patch.object(loop.subprocess, "Popen", return_value=Mock(wait=Mock(return_value=0))) as spawn:
            for index, allowed in enumerate((False, True, False)):
                path.write_text(json.dumps(dict(metadata, allow_math_computation=allowed)))
                assert loop.run_round(wl, loop._read_role(wl), prompt,
                                      wl.logs / f"policy-{index}.log", 10) == 0
                sent = spawn.call_args.args[0][-1]
                assert sent.startswith(f"Project computation policy: allow_math_computation={str(allowed).lower()}.")
                assert sent.endswith(prompt)
            path.write_text(json.dumps(dict(metadata, allow_math_computation="true")))
            count = spawn.call_count
            with pytest.raises(ValueError, match="must_be_boolean"):
                loop.run_round(wl, loop._read_role(wl), prompt, wl.logs / "invalid.log", 10)
            assert spawn.call_count == count


def test_contracts_use_notebook_and_preserve_math_boundaries():
    root = L.repo_root() / "agents" / "contracts"
    contracts = {role: (root / f"{role}.md").read_text(encoding="utf-8")
                 for role in ("main_agent", "worker", "verifier")}
    for text in contracts.values():
        assert "KNOWLEDGE.md" in text and "knowledge/proofs/" in text
        assert "accepted=true" in text and "published=true" in text
        assert "text-only" in text.lower()
        for old in ("gm_add", "gm_search", "fact_graph/", "global_memory/", "SKILL.md"):
            assert old not in text
    assert "30-minute" in contracts["main_agent"] and "four-hour" in contracts["main_agent"]
    assert "1–2 hours" in contracts["worker"]
    assert 'status="proposed"' in contracts["worker"]
    for role in ("main_agent", "worker"):
        text = contracts[role]
        for required in ('kinds=["report"]', 'sort_by="updated"', "Pxxx",
                         "attributes.subtype", "`method`", "`boundary`", "`review`",
                         "attributes.audits", "target_id", "not_covered", "historical=true",
                         "request_id", "expected_snapshot_revision", "record_request"):
            assert required in text
        assert "`lead`/`experience`" not in text
        guidance = " ".join(text.split())
        assert "try to formulate a precise, testable conjecture" in guidance
        assert "content hashes every three seconds" in guidance
        assert "intermediate claim" in guidance and "reformulation" in guidance
        assert "alternatives" in guidance and "naturally" in guidance
        assert "a clear proof path can simply be followed" in guidance
    worker = " ".join(contracts["worker"].split())
    assert "A formed conjecture can be shared when useful" in worker
    assert "without first recording a conjecture" in worker
    assert "Do not poll TASK.md" in worker
    assert "repeat an unchanged report" in worker
    assert "Returning to wait is not declaring the problem solved" in worker
    assert 'knowledge_search(query="", kinds=[...]' in worker
    assert "Judge relevance yourself" in worker
    assert "undefined_symbols" not in worker
    assert "no automatic symbol-coverage warning" in worker
    assert "no automatic phrase/length filter" in " ".join(contracts["verifier"].split())
    main_guidance = " ".join(contracts["main_agent"].split())
    assert "Do not restart it to keep it busy" in main_guidance
    assert "genuinely useful follow-up assignments" in main_guidance
    main_contract = " ".join((L.repo_root() / "AGENTS.md").read_text(encoding="utf-8").split())
    assert "Useful bridge lemmas, when relevant" in main_contract
    assert "rather than aiming for a particular number" in main_contract
    assert "Two to four fully specified bridge lemmas" not in main_contract
    prompt = loop.kickoff("P", "high")
    for required in ("knowledge_search(query=''", "Judge relevance yourself", "no relevance scores",
                     "kinds=['report']", "sort_by='updated'", "kinds=['feedback']",
                     "kind='report'", "recorded/archived", "no subtype", "attributes.subtype",
                     "'method'", "'boundary'", "'review'", "attributes.audits",
                     "target_id", "not_covered", "historical=true", "request_id",
                     "entry_id/expected_revision", "research_policy_version", "record_request",
                     "never read peers' private logs", "not certification",
                     "If progress stalls, try to consider a useful intermediate claim or reformulation",
                     "Weigh alternatives if they arise naturally", "following a clear proof path directly",
                     "Formed ideas may be shared as conjectures when useful",
                     "Complete proofs may go directly to fact_submit"):
        assert required in prompt
    assert '`"correct"` if and only if both' in contracts["verifier"]
    assert "necessary but not sufficient" in contracts["verifier"]


# --- runner ---------------------------------------------------------------- #

_NO_TMP = {test_parse_roles_default_roster, test_parse_roles_rejects_bad_specs,
           test_worker_layout_paths, test_resolve_and_target,
           test_contracts_use_notebook_and_preserve_math_boundaries}


def main() -> None:
    for t in [test_parse_roles_default_roster, test_parse_roles_rejects_bad_specs,
              test_worker_layout_paths, test_resolve_and_target,
              test_do_new_scaffolds_project, test_do_new_refuses_existing,
              test_do_new_verify_url_from_env, test_parse_last_fact_id,
              test_deadline_passed, test_write_status_atomic_and_stamps,
              test_read_role_defaults_and_overrides,
              test_contracts_use_notebook_and_preserve_math_boundaries]:
        if t in _NO_TMP:
            t()
        else:
            with tempfile.TemporaryDirectory() as d:
                t(Path(d))
        print(f"  [ok] {t.__name__}")
    print("ALL EXECUTION TESTS PASSED")


if __name__ == "__main__":
    main()
