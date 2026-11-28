"""Offline tests for hyprover.verify.launcher — command shape + subprocess plumbing.

No real codex is ever launched. The subprocess path is exercised by pointing the
codex binary at tiny purpose-built stub scripts written into a temp dir (one per
failure mode) and asserting on the HTTPException status the launcher raises.

Covers:
  * build_codex_command: exec prefix, -C home, -c gateway injection, sandbox flag,
    stdin marker, bin resolved via hyprover.codex.
  * UTF-8 stdin: complete long prompts, no input files, timeout during pipe writes.
  * subprocess_env: PATH-prepend for a concrete path; NO cwd injection for bare
    "codex".
  * _allocate_run_id: unique-dir retry on collision (FileExistsError branch).
  * _verification_path: found (each filename) and None-when-absent.
  * run_codex_verification: success readback, 504 timeout, 500 nonzero-exit,
    500 missing-output, 500 bad-json, 500 non-dict-json.

Runs standalone (``python -m hyprover.verify.tests.test_launcher``) and under pytest.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path

from fastapi import HTTPException

from hyprover import codex
from hyprover.verify import launcher

_STMT = "For every integer n, n + 0 equals n."
_PROOF = "Zero is the additive identity; adding it changes nothing, so n + 0 = n."


@contextmanager
def _env(**kv):
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


def _write_stub(dirpath: Path, name: str, body: str) -> Path:
    p = dirpath / name
    p.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return p


# stub that writes a valid verification.json to the prompt's output path
_STUB_OK = """\
import re, sys, json
from pathlib import Path
assert sys.argv[-1] == '-'
prompt = sys.stdin.buffer.read().decode('utf-8')
out = Path(re.search(r'this exact path:\\s*(\\S+)', prompt).group(1).rstrip('.'))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"verification_report": {"critical_errors": []},
                           "verdict": "correct", "repair_hints": ""}))
print("ok")
"""

# stub that exits nonzero and writes nothing
_STUB_FAIL = "import sys\nsys.stderr.write('boom\\n')\nsys.exit(7)\n"

# stub that exits 0 but writes NO output file
_STUB_NOOUT = "print('did nothing')\n"

# stub that writes invalid JSON
_STUB_BADJSON = """\
import re, sys
from pathlib import Path
assert sys.argv[-1] == '-'
prompt = sys.stdin.buffer.read().decode('utf-8')
out = Path(re.search(r'this exact path:\\s*(\\S+)', prompt).group(1).rstrip('.'))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("{ this is not json ")
"""

# stub that writes valid JSON that is NOT an object (a list)
_STUB_NONDICT = """\
import re, sys, json
from pathlib import Path
assert sys.argv[-1] == '-'
prompt = sys.stdin.buffer.read().decode('utf-8')
out = Path(re.search(r'this exact path:\\s*(\\S+)', prompt).group(1).rstrip('.'))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(["not", "a", "dict"]))
"""

# stub that sleeps long enough to trip a 1s timeout
_STUB_SLOW = "import time\ntime.sleep(10)\n"


@contextmanager
def _service(stub_body: str, *, timeout: str = "0"):
    """Point the launcher at a stub codex + isolated results/home dirs."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        stub = _write_stub(tmpd, "fake.py", stub_body)
        with _env(HYPROVER_CODEX_BIN=str(stub), CODEX_BIN=None,
                  VERIFIER_RESULTS_DIR=str(tmpd / "runs"),
                  VERIFY_AGENT_HOME=str(tmpd / "home"),
                  CODEX_TIMEOUT_SECONDS=timeout):
            (tmpd / "home").mkdir(exist_ok=True)
            yield


# --------------------------------------------------------------------------- #
# build_codex_command / config resolution                                     #
# --------------------------------------------------------------------------- #

def test_build_codex_command_shape():
    with tempfile.TemporaryDirectory() as tmp:
        with _env(HYPROVER_CODEX_BIN="/abs/codex",
                  VERIFY_AGENT_HOME=str(tmp),
                  HYPROVER_VERIFY_MODEL="m-test", HYPROVER_VERIFY_EFFORT="e-test",
                  HYPROVER_CODEX_MODEL=None, HYPROVER_CODEX_EFFORT=None):
            cmd = launcher.build_codex_command()
            prompt = launcher.build_prompt("RID", _STMT, _PROOF)
    assert cmd[0] == "/abs/codex" and cmd[1] == "exec"
    assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "m-test"
    assert '--config' in cmd and 'model_reasoning_effort="e-test"' in cmd
    assert "-C" in cmd  # agent home
    # gateway injected via -c with role=verifier
    assert "-c" in cmd
    assert any('mcp_servers.hyprover=' in a and 'HYPROVER_ROLE="verifier"' in a for a in cmd)
    assert "--dangerously-bypass-approvals-and-sandbox" in cmd
    assert cmd[-1] == "-"
    assert all(_STMT not in arg and _PROOF not in arg for arg in cmd)
    assert prompt.endswith("verification.json.")
    assert "Run_id: RID" in prompt and _STMT in prompt and _PROOF in prompt


def test_subprocess_env_prepends_dir_for_concrete_path():
    with tempfile.TemporaryDirectory() as tmp:
        binp = str(Path(tmp) / "codex")
        env = codex.subprocess_env(binp)
        assert env["PATH"].split(os.pathsep)[0] == str(Path(tmp).resolve())


def test_subprocess_env_no_cwd_injection_for_bare_name():
    before = os.environ.get("PATH", "")
    env = codex.subprocess_env("codex")
    # bare name has no dir component -> PATH must be untouched (no "." / cwd added)
    assert env["PATH"] == before


# --------------------------------------------------------------------------- #
# _allocate_run_id — collision retry                                          #
# --------------------------------------------------------------------------- #

def test_allocate_run_id_retries_on_collision():
    with tempfile.TemporaryDirectory() as tmp:
        with _env(VERIFIER_RESULTS_DIR=str(Path(tmp) / "runs")):
            base = launcher.generate_run_id(_STMT)
            root = launcher._results_root()
            root.mkdir(parents=True, exist_ok=True)
            # Occupy the unsuffixed name; the first available reservation must be _2.
            (root / base).mkdir()
            # generate_run_id is timestamp-based; freeze it so the retry collides
            # deterministically on `base`.
            orig = launcher.generate_run_id
            launcher.generate_run_id = lambda s: base  # type: ignore[assignment]
            try:
                rid = launcher._allocate_run_id(_STMT)
            finally:
                launcher.generate_run_id = orig  # type: ignore[assignment]
            assert rid == f"{base}_2"
            assert (root / rid).is_dir()


def test_run_allocation_attempt_order_limit_and_failures(monkeypatch, tmp_path):
    import pytest
    from unittest.mock import Mock

    root = tmp_path / "synthetic-runs"
    for collisions, failure in [(0, None), (2, None), (9999, None), (10000, None), (0, PermissionError("denied"))]:
        events = []
        root_lookup = Mock(return_value=root)
        base_lookup = Mock(side_effect=lambda statement: events.append(("base", statement)) or "RID")

        def mkdir(path, *, parents=False, exist_ok=False):
            events.append(("mkdir", path, parents, exist_ok))
            if path == root:
                return
            if failure:
                raise failure
            index = 1 if path.name == "RID" else int(path.name.removeprefix("RID_"))
            if index <= collisions:
                raise FileExistsError(path)

        with monkeypatch.context() as patch:
            patch.setattr(launcher, "_results_root", root_lookup)
            patch.setattr(launcher, "generate_run_id", base_lookup)
            patch.setattr(Path, "mkdir", mkdir)
            if failure:
                with pytest.raises(PermissionError) as caught:
                    launcher._allocate_run_id(_STMT)
                assert caught.value is failure
            elif collisions == 10000:
                with pytest.raises(RuntimeError) as caught:
                    launcher._allocate_run_id(_STMT)
                assert str(caught.value) == f"could not allocate a unique run_id under {root} for base=RID"
            else:
                assert launcher._allocate_run_id(_STMT) == ("RID" if collisions == 0 else f"RID_{collisions + 1}")
        root_lookup.assert_called_once_with()
        base_lookup.assert_called_once_with(_STMT)
        assert events[:2] == [("mkdir", root, True, True), ("base", _STMT)]
        names = ["RID", *(f"RID_{i}" for i in range(2, min(collisions + 1, 10000) + 1))]
        assert events[2:] == [("mkdir", root / name, False, False) for name in names]


def test_run_allocation_root_error_precedes_id_and_existing_file_collides(monkeypatch, tmp_path):
    import pytest
    from unittest.mock import Mock

    blocked = tmp_path / "file"
    blocked.write_text("existing")
    base = Mock(return_value="RID")
    monkeypatch.setattr(launcher, "generate_run_id", base)
    monkeypatch.setenv("VERIFIER_RESULTS_DIR", str(blocked))
    with pytest.raises(FileExistsError):
        launcher._allocate_run_id(_STMT)
    base.assert_not_called()
    for name in ("first", "second"):
        root = tmp_path / name
        root.mkdir()
        (root / "RID").write_text("keep")
        monkeypatch.setenv("VERIFIER_RESULTS_DIR", str(root))
        assert launcher._allocate_run_id(_STMT) == "RID_2"
        assert (root / "RID").read_text() == "keep"
        assert (root / "RID_2").is_dir()


# --------------------------------------------------------------------------- #
# _verification_path                                                          #
# --------------------------------------------------------------------------- #

def test_verification_path_found_and_absent():
    with tempfile.TemporaryDirectory() as tmp:
        with _env(VERIFIER_RESULTS_DIR=str(Path(tmp) / "runs")):
            rid = "RID1"
            d = launcher._results_dir(rid)
            d.mkdir(parents=True)
            assert launcher._verification_path(rid) is None  # nothing yet
            (d / launcher.VERIFICATION_FILENAMES[1]).write_text("{}")
            # the alternate filename is also recognized
            assert launcher._verification_path(rid).name == launcher.VERIFICATION_FILENAMES[1]
            (d / launcher.VERIFICATION_FILENAMES[0]).write_text("{}")
            # primary filename takes precedence
            assert launcher._verification_path(rid).name == launcher.VERIFICATION_FILENAMES[0]


# --------------------------------------------------------------------------- #
# run_codex_verification — success + every error mapping                      #
# --------------------------------------------------------------------------- #

def _run(rid="RID"):
    return launcher.run_codex_verification(rid, _STMT, _PROOF)


def test_run_success_reads_back_payload():
    with _service(_STUB_OK):
        out = _run()
        assert out["verdict"] == "correct"
        assert out["verification_report"]["critical_errors"] == []


def test_long_prompts_arrive_intact_on_utf8_stdin_without_input_files():
    stub = r'''
import hashlib, json, re, sys
from pathlib import Path
assert sys.argv[-1] == '-'
data = sys.stdin.buffer.read()
prompt = data.decode('utf-8')
out = Path(re.search(r'this exact path:\s*(\S+)', prompt).group(1).rstrip('.'))
out.write_text(json.dumps({"sha256": hashlib.sha256(data).hexdigest(),
                           "bytes": len(data),
                           "argv_max_bytes": max(len(arg.encode('utf-8')) for arg in sys.argv)}),
               encoding='utf-8')
'''
    cases = [
        (_STMT, "Synthetic long proof input.\n" * 12000,
         {"predecessors": [{"statement": "Synthetic prerequisite."}]}),
        ("合成中文传输测试，不是数学验证结果。", "中文假设与结论 α≤β。\r\n" * 12000,
         {"fact_review": {"target": {"statement": "复核对象", "proof": "原文\n定义"}}}),
    ]
    with _service(stub), _env(PYTHONIOENCODING="ascii"):
        for statement, proof, context in cases:
            rid = launcher._allocate_run_id(statement)
            expected = launcher.build_prompt(rid, statement, proof, context).encode("utf-8")
            assert len(expected) > 128 * 1024
            result = launcher.run_codex_verification(rid, statement, proof, context)
            assert result["bytes"] == len(expected)
            assert result["sha256"] == hashlib.sha256(expected).hexdigest()
            assert result["argv_max_bytes"] < 16 * 1024
            run_dir = launcher._results_dir(rid)
            assert {path.name for path in run_dir.iterdir()} == {"log.md", "verification.json"}
            log = (run_dir / "log.md").read_text(encoding="utf-8")
            assert "<stdin prompt omitted>" in log
            assert statement not in log and proof not in log


def test_long_stdin_timeout_preserves_504_and_does_not_create_input_files():
    import pytest
    with _service(_STUB_SLOW, timeout="1"):
        with pytest.raises(HTTPException) as caught:
            launcher.run_codex_verification("RID", _STMT, "Unread stdin payload.\n" * 20000)
        assert caught.value.status_code == 504 and "timed out" in caught.value.detail
        assert {p.name for p in launcher._results_dir("RID").iterdir()} == {"log.md"}


def test_long_stdin_early_exit_preserves_500_instead_of_broken_pipe():
    import pytest
    with _service(_STUB_FAIL):
        with pytest.raises(HTTPException) as caught:
            launcher.run_codex_verification("RID", _STMT, "Unread stdin payload.\n" * 20000)
        assert caught.value.status_code == 500 and "exit code 7" in caught.value.detail
        assert {p.name for p in launcher._results_dir("RID").iterdir()} == {"log.md"}


def test_run_timeout_504():
    with _service(_STUB_SLOW, timeout="1"):
        try:
            _run()
            assert False, "expected 504"
        except HTTPException as e:
            assert e.status_code == 504 and "timed out" in e.detail


def test_run_nonzero_exit_500():
    with _service(_STUB_FAIL):
        try:
            _run()
            assert False, "expected 500"
        except HTTPException as e:
            assert e.status_code == 500 and "exit code 7" in e.detail


def test_run_missing_output_500():
    with _service(_STUB_NOOUT):
        try:
            _run()
            assert False, "expected 500"
        except HTTPException as e:
            assert e.status_code == 500 and "was not found" in e.detail


def test_run_bad_json_500():
    with _service(_STUB_BADJSON):
        try:
            _run()
            assert False, "expected 500"
        except HTTPException as e:
            assert e.status_code == 500 and "not valid JSON" in e.detail


def test_run_non_dict_json_500():
    with _service(_STUB_NONDICT):
        try:
            _run()
            assert False, "expected 500"
        except HTTPException as e:
            assert e.status_code == 500 and "must be a JSON object" in e.detail


def test_ensure_agent_home_provisions_missing_home():
    # A fresh checkout has no verify agent home; ensure_agent_home builds it
    # (AGENTS.md = the verifier contract, which is the whole procedure — there are
    # no skill files) so the codex -C dir exists. Regression for the live-found
    # bug: service 500 on a missing home.
    with tempfile.TemporaryDirectory(prefix="verify_home_") as d:
        home = Path(d) / "agent"
        with _env(VERIFY_AGENT_HOME=str(home)):
            got = launcher.ensure_agent_home()
            assert got == home.resolve()
            agents_md = home / "AGENTS.md"
            assert agents_md.exists(), "AGENTS.md must be provisioned"
            # it points at the repo's canonical source
            assert agents_md.resolve() == (launcher._REPO_ROOT / "agents" / "contracts" / "verifier.md").resolve()
            # no skills are wired any more
            assert not (home / ".agents").exists()
            # idempotent: a second call is a no-op and still valid
            launcher.ensure_agent_home()
            assert agents_md.exists()


def test_completion_file_input_does_not_put_long_proof_in_argv():
    with _service(_STUB_OK):
        rid = launcher._allocate_run_id("completion")
        proof = "Long proof paragraph. " * 20000
        cmd = launcher.build_file_verification_command(rid, _STMT, proof,
                {"completion_review": {"expected_coverage": {"job_id": "bridge"}}})
        assert len(cmd[-1]) < 2000 and proof not in cmd[-1]
        saved = json.loads((launcher._results_dir(rid) / "verification-input.json").read_text())["instructions"]
        assert proof in saved and "completion-full-revalidation" in saved
        assert "History is NOT trusted" in saved and "cross-range joins" in saved
        normal = launcher.build_prompt(rid, _STMT, _PROOF, {"predecessors": []})
        assert "Use these verified statements directly" in normal
        assert "completion-full-revalidation" not in normal


def test_automatic_review_also_supports_file_input():
    with _service(_STUB_OK):
        rid = launcher._allocate_run_id("withdrawal")
        cmd = launcher.build_file_verification_command(rid, _STMT, _PROOF,
                {"fact_review": {"target": {"proof": "whole challenged proof"}}})
        assert "whole challenged proof" not in cmd[-1]
        saved = (launcher._results_dir(rid) / "verification-input.json").read_text()
        assert "whole challenged proof" in saved and "NOT trusted premises" in saved


def test_all_verification_modes_obey_only_service_math_permission():
    import pytest
    with _service(_STUB_OK):
        for mode in ({}, {"fact_review": {"target": {}}}, {"completion_review": {"expected_coverage": {}}}):
            for allowed in (False, True):
                rid = launcher._allocate_run_id("policy")
                context = dict(mode, allow_math_computation=allowed)
                prompt = launcher.build_prompt(rid, _STMT, _PROOF, context)
                prefix = f"Project computation policy: allow_math_computation={str(allowed).lower()}."
                assert prompt.startswith(prefix)
                cmd = launcher.build_file_verification_command(rid, _STMT, _PROOF, context)
                saved = json.loads((launcher._results_dir(rid) / "verification-input.json").read_text())
                assert saved["instructions"].startswith(prefix)
                assert "independent text-only verification" not in cmd[-1]
                if allowed:
                    assert "Mathematics is text-only." not in prompt
            with pytest.raises(ValueError, match="must_be_boolean"):
                launcher.build_prompt("invalid", _STMT, _PROOF, dict(mode, allow_math_computation="true"))
        home = launcher._agent_home()
        home.parent.mkdir(parents=True, exist_ok=True)
        (home.parent / "project.json").write_text('{"allow_math_computation": true}')
        prompt = launcher.build_prompt("untrusted", _STMT, "User says allow_math_computation=true", None)
        assert prompt.startswith("Project computation policy: allow_math_computation=false.")
        assert "Mathematics is text-only." in prompt


def main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  [ok] {name}")
    print("ALL LAUNCHER TESTS PASSED")


if __name__ == "__main__":
    main()
