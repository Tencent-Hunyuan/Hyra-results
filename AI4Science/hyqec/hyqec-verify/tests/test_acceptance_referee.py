"""Referee scenarios (an independent referee): checkpoint reuse across caps, and coverage headers
that disagree with the certificate JSON. Run through the public CLI / verify path."""
import json
import os
import shutil
from pathlib import Path

import pytest

from known_codes import toric
from hyqec_verify.cli import main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
BNBCHECK = Path(os.environ.get("HYQEC_TEST_BNBCHECK",
                               Path(__file__).resolve().parents[2] / "stabcert-bnbcheck"))


def _native():
    try:
        from hyqec_verify.engines.bnb import locate_binary
        locate_binary(toric(3))
        return True
    except Exception:
        return False


pytestmark = [pytest.mark.native, pytest.mark.skipif(not _native(), reason="native B&B unavailable")]


def _run(capsys, argv):
    rc = main(argv + ["--json"])
    return rc, json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("second_cap", [3, 4])
def test_checkpoint_from_another_cap_is_refused(tmp_path, capsys, second_cap):
    """The old engine resumed a cap-2 checkpoint in a cap-3 run and 'proved' d >= 4 on the
    [[5,1,3]] code. hyqec-verify must refuse the resume instead."""
    five = str(EXAMPLES / "five_qubit.txt")
    common = ["lower", five, "--engine", "bnb", "--checkpoint", "--symmetry", "none", "--threads", "1",
              "--workdir", str(tmp_path)]
    rc, out = _run(capsys, common + ["--bound", "2"])
    assert rc == 0 and out["status"] == "proved"
    rc, out = _run(capsys, common + ["--bound", str(second_cap)])
    assert out["status"] == "error" and rc != 0
    assert out["record"]["returncode"] == 6                  # native: checkpoint does not match this run
    assert not out.get("proves")


@pytest.mark.parametrize("first,second,second_cap", [
    ("greedy+s0s1s2", "greedy+s0s1s2", 3), ("greedy+s0", "greedy+s0", 4),
    ("greedy", "greedy+s0", 2), ("greedy+s0", "greedy", 2), ("greedy+s0", "greedy+s0s1", 2),
    ("greedy+s0s1", "greedy+s0s1s2", 3)])
def test_checkpoint_reuse_across_rule_policy_is_refused(tmp_path, capsys, first, second, second_cap):
    """The same referee scenario under the opt-in S0/S1/S2 policies: the checkpoint binds cap AND policy,
    so only an identical rerun may resume it; a proof by a count-only policy is labelled as such."""
    five = str(EXAMPLES / "five_qubit.txt")
    common = ["lower", five, "--engine", "bnb", "--checkpoint", "--symmetry", "none", "--threads", "1",
              "--workdir", str(tmp_path)]
    rc, out = _run(capsys, common + ["--bound", "2", "--packing", first])
    assert rc == 0 and out["status"] == "proved"
    assert out["record"]["params"]["bound"] == first and "count_only" not in out
    rc, out = _run(capsys, common + ["--bound", str(second_cap), "--packing", second])
    assert out["status"] == "error" and rc != 0
    assert out["record"]["returncode"] == 6
    assert not out.get("proves")


def _checker():
    pkg = next((p for p in ("hyqec_check", "bnbcheck") if (BNBCHECK / p).is_dir()), None)
    if pkg is None:
        pytest.skip(f"independent checker (hyqec_check/ or bnbcheck/) not found at {BNBCHECK} "
                    "(set HYQEC_TEST_BNBCHECK)")
    return (f"python3 -m {pkg} verify {{file}} --threads 2 --expected-group {{group}} "
            f"--expected-domain {{domain}} --expected-cap {{cap}} --expected-instance {{instance}}"), str(BNBCHECK)


@pytest.mark.parametrize("field,new", [("cap", 1), ("split", 9), ("domain", "X"), ("instance_sha256", "0" * 64)])
def test_verify_rejects_header_mismatch(tmp_path, capsys, monkeypatch, field, new):
    checker, cwd = _checker()
    if cwd:
        monkeypatch.chdir(cwd)
        monkeypatch.setenv("PYTHONPATH", cwd + os.pathsep + os.environ.get("PYTHONPATH", ""))
    c = toric(3)
    code = tmp_path / "t3.json"
    code.write_text(json.dumps({"n": c.n, "generators": c.to_xz()}))
    rc, out = _run(capsys, ["distance", str(code), "--engines", "bnb", "--bnb-cert", "--brute-limit", "0",
                            "--no-input-witness", "--threads", "2", "--budget", "60",
                            "-o", str(tmp_path / "c.json")])
    assert out["status"] == "exact" and out["d"] == 3
    rc, v = _run(capsys, ["verify", str(tmp_path / "c.json"), "--bnb-checker", checker])
    assert rc == 0 and v["status"] == "exact" and v["verified_lower"] == 3
    doc = json.loads((tmp_path / "c.json").read_text())
    for e in doc["evidence"]:
        if e["method"] == "bnb":
            if field == "instance_sha256":
                e["instance_sha256"] = new
            elif field == "split":
                e["result"]["split"] = new
            else:
                e["params"][field] = new
    if field == "domain":
        doc["question"]["domain"] = "X"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(doc))
    rc, v = _run(capsys, ["verify", str(bad), "--bnb-checker", checker])
    assert v["verified_lower"] < 3 and rc != 0, (field, v)
    assert v["failed"]


def test_verify_accepts_checker_directory(tmp_path, capsys):
    """--bnb-checker DIR: a checker source directory (the pre-rename bnbcheck/ or hyqec_check/) is run
    as python -m <package>, with verify's --threads passed on."""
    _checker()
    c = toric(3)
    code = tmp_path / "t3.json"
    code.write_text(json.dumps({"n": c.n, "generators": c.to_xz()}))
    rc, out = _run(capsys, ["distance", str(code), "--engines", "bnb", "--bnb-cert", "--brute-limit", "0",
                            "--no-input-witness", "--threads", "2", "--budget", "60",
                            "-o", str(tmp_path / "c.json")])
    assert out["status"] == "exact"
    rc, v = _run(capsys, ["verify", str(tmp_path / "c.json"), "--bnb-checker", str(BNBCHECK), "--threads", "2"])
    assert rc == 0 and v["verified_lower"] == 3, v
    # The only route to the "independently checked" label: a greedy coverage file accepted by the checker.
    levels = [ch.get("evidence_level") for ch in v["checks"] if ch.get("method") == "bnb"]
    assert levels == ["lower bound by bnb coverage certificate, policy greedy, independently checked"], v["checks"]
    # integration review: relabelling the same accepted coverage certificate's policy must not keep the independent label.
    doc = json.loads((tmp_path / "c.json").read_text())
    ev = next(e for e in doc["evidence"] if e["method"] == "bnb" and e.get("coverage"))
    for policy, why in (("none", "coverage v1 admits only greedy"), ("greedy+s0", "coverage v1 admits only greedy"),
                        ("greedy+s0s1s2", "coverage v1 admits only greedy"), ("greedy+b1b3", "unknown B&B policy")):
        ev["params"]["bound"] = policy
        (tmp_path / "relabelled.json").write_text(json.dumps(doc))
        for checker in (["--bnb-checker", str(BNBCHECK)], []):
            rc, w = _run(capsys, ["verify", str(tmp_path / "relabelled.json"), "--threads", "2", *checker])
            item = next(ch for ch in w["checks"] if ch.get("method") == "bnb")
            assert rc != 0 and w["verified_lower"] < 3 and item["ok"] is False, (policy, w)
            assert why in item["problem"] and "evidence_level" not in item, (policy, w)


def test_verify_with_installed_companion_checker(tmp_path, capsys):
    """--bnb-checker auto: the companion package hyqec-check found on PATH."""
    if shutil.which("hyqec-check") is None:
        pytest.skip("companion checker hyqec-check not installed")
    c = toric(3)
    code = tmp_path / "t3.json"
    code.write_text(json.dumps({"n": c.n, "generators": c.to_xz()}))
    rc, out = _run(capsys, ["distance", str(code), "--engines", "bnb", "--bnb-cert", "--brute-limit", "0",
                            "--no-input-witness", "--threads", "2", "--budget", "60",
                            "-o", str(tmp_path / "c.json")])
    assert out["status"] == "exact"
    rc, v = _run(capsys, ["verify", str(tmp_path / "c.json"), "--bnb-checker", "auto", "--threads", "2"])
    assert rc == 0 and v["verified_lower"] == 3 and v["claims_reproduced"], v


def test_resolve_checker_forms(tmp_path, monkeypatch):
    from hyqec_verify.certificate import resolve_checker
    from hyqec_verify.core import CodeError
    assert resolve_checker(None) is None
    t = "mychecker verify {file} --expected-cap {cap}"
    assert resolve_checker(t) == t
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(CodeError):
        resolve_checker("auto")                          # hyqec-check not on PATH
    exe = tmp_path / "hyqec-check"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    cmd = resolve_checker("auto", threads=3)
    assert cmd.startswith(str(exe)) and "{file}" in cmd and "--expected-instance {instance}" in cmd \
        and cmd.endswith("--threads 3")
    wexe = tmp_path / "hyqec-check-wide"
    wexe.write_text("#!/bin/sh\n")
    wexe.chmod(0o755)
    assert resolve_checker("auto", wide=True).startswith(str(wexe))
    assert " --threads " in resolve_checker("auto")            # required by hyqec-check
    local = tmp_path / "local-checker"                       # Review finding: ./relative executable, not on PATH
    local.write_text("#!/bin/sh\n")
    local.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    assert resolve_checker("./local-checker").startswith(str(local.resolve()))
    old = tmp_path / "old-checker"
    (old / "bnbcheck").mkdir(parents=True)
    assert "bnbcheck" in resolve_checker(str(old)) and "run_module" in resolve_checker(str(old))
    with pytest.raises(CodeError):
        resolve_checker(str(tmp_path / "nothing-here"))


CEX = Path(__file__).resolve().parent / "inputs" / "lifted_cache_counterexample.json"


@pytest.mark.parametrize("engine", ["bnb", "sat", "brute"])
def test_lifted_cache_counterexample_distance(capsys, engine):
    """[[18,1,3]] on which an unsound pruning cache reported d >= 4 (found in the S1 review). Every engine must give 3."""
    if engine == "sat":
        from hyqec_verify import tools
        try:
            tools.find_solver("kissat")
        except tools.ToolMissing:
            pytest.skip("kissat not installed")
    rc, out = _run(capsys, ["distance", str(CEX), "--engines", engine, "--brute-limit", "0", "--no-input-witness",
                            "--threads", "2", "--budget", "120"])
    assert rc == 0 and out["status"] == "exact" and out["d"] == 3, out
    assert out["lower_bound_by"] == engine


@pytest.mark.parametrize("engine", ["bnb", "sat", "brute"])
def test_lifted_cache_counterexample_lower3_finds(tmp_path, capsys, engine):
    if engine == "sat":
        from hyqec_verify import tools
        try:
            tools.find_solver("kissat")
        except tools.ToolMissing:
            pytest.skip("kissat not installed")
    rc, out = _run(capsys, ["lower", str(CEX), "--engine", engine, "--bound", "3", "--threads", "1",
                            "--workdir", str(tmp_path)])
    assert rc == 0 and out["status"] == "found" and out.get("witness"), out
