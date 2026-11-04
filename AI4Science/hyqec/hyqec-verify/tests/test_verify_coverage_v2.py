"""verify consumes coverage-v2 certificates (BNB_CERT_SPEC_v2 2.0-rc1) only through the companion checker
hyqec-check 2.0.0 (hyqec-check-v2): accepted v2 under a parity policy reaches the independently checked
label; any mismatch of instance, cap, domain or rule set, a missing or older checker, a tampered file, or a
verdict about a different question never counts."""
import copy
import gzip
import json
import shutil
import sys
from pathlib import Path

import pytest

from hyqec_verify.certificate import resolve_checker, verify_certificate, write_certificate
from hyqec_verify.cli import main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _exe(name):
    found = shutil.which(name) or Path(sys.executable).with_name(name)
    return str(found) if Path(found).exists() else None


V2 = _exe("hyqec-check-v2")
V1 = _exe("hyqec-check")


def _native():
    try:
        from hyqec_verify.engines.bnb import locate_binary
        from hyqec_verify.io import load
        locate_binary(load(EXAMPLES / "toric_4.json").code, rules=True)
        return True
    except Exception:
        return False


pytestmark = [pytest.mark.native]       # the certificate fixture skips without the native rules build
needs_v2 = pytest.mark.skipif(V2 is None, reason="companion checker hyqec-check-v2 (hyqec-check >= 2.0.0) not installed")


@pytest.fixture(scope="module")
def v2cert(tmp_path_factory):
    """toric 4x4 (d=4), cap 3, policy greedy+s0s1s2: a coverage-v2 certificate from the public CLI."""
    if not _native():
        pytest.skip("native B&B rules build unavailable")
    d = tmp_path_factory.mktemp("v2")
    rc = main(["lower", str(EXAMPLES / "toric_4.json"), "--engine", "bnb", "--bound", "3", "--packing",
               "greedy+s0s1s2", "--threads", "2", "--bnb-cert", "--workdir", str(d / "w"), "-o", str(d / "c.json"),
               "--json"])
    assert rc == 0
    return d / "c.json"


def _copy(cert, tmp_path):
    """The certificate and its artifacts in a fresh directory (edits never touch the fixture)."""
    doc = json.loads(cert.read_text())
    art = cert.parent / f"{cert.stem}.artifacts"
    shutil.copytree(art, tmp_path / art.name)
    p = tmp_path / cert.name
    p.write_text(json.dumps(doc))
    return p, doc


def _bnb(doc):
    return next(e for e in doc["evidence"] if e["method"] == "bnb")


def _verify(path, checker, **kw):
    return verify_certificate(path, bnb_checker=checker, threads=2, **kw)


def _bnb_item(v):
    return next(c for c in v["checks"] if c.get("method") == "bnb")


@needs_v2
def test_v2_accepted_and_labelled(v2cert):
    v = _verify(v2cert, V2)
    it = _bnb_item(v)
    assert v["verified_lower"] == 4 and v["claims_reproduced"] and not v["failed"], v["checks"]
    assert it["coverage_version"] == 2
    import re                                                   # any hyqec-check 2.0.x
    assert re.fullmatch(r"lower bound by bnb coverage certificate v2 \(rules s0s1s2, spec 2\.0-rc1\), "
                        r"policy greedy\+s0s1s2, independently checked by hyqec-check-v2 "
                        r"\(reported version 2\.0\.\d+\)", it["evidence_level"]), it["evidence_level"]


@needs_v2
def test_v2_auto_finds_hyqec_check_v2(v2cert, monkeypatch):
    monkeypatch.setenv("PATH", str(Path(V2).parent))
    v = _verify(v2cert, "auto")
    assert v["verified_lower"] == 4 and "hyqec-check-v2" in _bnb_item(v)["checker"]


def test_v2_without_checker_never_counts(v2cert, monkeypatch):
    v = _verify(v2cert, None)
    assert v["verified_lower"] == 1 and "not checked (no --bnb-checker)" in _bnb_item(v)["evidence_level"]
    monkeypatch.setenv("PATH", "/nonexistent")
    v = _verify(v2cert, "auto")                                  # companion not installed
    assert v["verified_lower"] == 1 and "independent checker unavailable" in _bnb_item(v)["problem"]


@pytest.mark.skipif(V1 is None, reason="hyqec-check (v1 profile) not installed")
def test_v2_with_old_checker_never_counts(v2cert):
    v = _verify(v2cert, V1)                                      # hyqec-check 1.x rejects SBCOV2
    assert v["verified_lower"] == 1 and v["failed"]
    assert "did not accept" in _bnb_item(v)["problem"]


@pytest.mark.parametrize("edit,why", [
    ("instance", "coverage header does not match"),
    ("cap", "coverage header does not match"),
    ("domain", "coverage header does not match"),
    ("rules", "coverage header does not match"),
    ("none", "no v2 rule set"),
])
def test_v2_mismatch_rejected(v2cert, tmp_path, edit, why):
    """Header binding happens before the checker runs; a fake always-accepting checker must not help."""
    p, doc = _copy(v2cert, tmp_path)
    e = _bnb(doc)
    if edit == "instance":
        e["instance_sha256"] = "0" * 64
    elif edit == "cap":
        e["params"]["cap"] = 2
    elif edit == "domain":
        e["params"]["domain"] = doc["question"]["domain"] = "X"
    elif edit == "rules":
        e["params"]["bound"] = "greedy+s0s1"                     # header declares s0s1s2
    else:
        e["params"]["bound"] = "none"
    p.write_text(json.dumps(doc))
    fake = _fake_checker(tmp_path, {})
    try:
        v = _verify(p, fake)
    except Exception as ex:                                      # e.g. X question on a non-CSS code
        assert edit == "domain", ex
        return
    assert v["verified_lower"] < 4 and v["failed"]
    assert any(why in (f.get("problem") or "") for f in v["failed"]), v["failed"]


def _fake_checker(tmp_path, override):
    """A 'checker' that accepts anything, printing a verdict for the real question with fields overridden."""
    script = tmp_path / "fake-checker"
    script.write_text("#!" + sys.executable + "\n"
                      "import json, sys\n"
                      "a = sys.argv\n"
                      "v = lambda f: a[a.index(f) + 1]\n"
                      "r = {'status': 'accepted', 'certificate_verified': True, 'selection': 'all',\n"
                      "     'checker_version': '2.0.0', 'group_sha256': v('--expected-group'),\n"
                      "     'instance_sha256': v('--expected-instance'), 'domain': v('--expected-domain'),\n"
                      "     'cap': int(v('--expected-cap')), 'rules': 's0s1s2', 'rules_version': 1,\n"
                      "     'spec_revision': '2.0-rc1'}\n"
                      f"r.update({override!r})\n"
                      "r = {k: x for k, x in r.items() if x != '__drop__'}\n"
                      "print(json.dumps(r))\n")
    script.chmod(0o755)
    return str(script)


@pytest.mark.parametrize("override", [{"rules": "s0s1"}, {"rules": "__drop__"}, {"spec_revision": "1.1"},
                                      {"spec_revision": "__drop__"}, {"cap": 2}, {"domain": "Z"},
                                      {"instance_sha256": "0" * 64}, {"status": "partial"},
                                      {"selection": "sample"}, {"rules_version": 2},
                                      {"rules_version": True}, {"cap": 3.0}])          # exact JSON types
def test_v2_verdict_must_match_claim(v2cert, tmp_path, override):
    """Exit 0 + certificate_verified is not enough: the verdict must be about this question and rule set."""
    v = _verify(v2cert, _fake_checker(tmp_path, override))
    assert v["verified_lower"] == 1, v["checks"]
    assert "checker verdict does not match" in _bnb_item(v)["problem"]


def test_v2_matching_verdict_counts(v2cert, tmp_path):
    """Control for the test above: the same fake with nothing overridden is accepted."""
    v = _verify(v2cert, _fake_checker(tmp_path, {}))
    assert v["verified_lower"] == 4, v["checks"]


@needs_v2
def test_v2_tampered_file_rejected_by_checker(v2cert, tmp_path):
    """A truncated stream with consistent hashes in the certificate passes header binding; the checker rejects."""
    p, doc = _copy(v2cert, tmp_path)
    e = _bnb(doc)
    f = p.parent / e["artifact_dir"] / e["coverage"]["file"]
    raw = gzip.decompress(f.read_bytes())
    bad = raw[:-16]
    f.write_bytes(gzip.compress(bad, mtime=0))
    import hashlib
    e["coverage"].update(gz_sha256=hashlib.sha256(f.read_bytes()).hexdigest(), gz_bytes=f.stat().st_size,
                         sha256=hashlib.sha256(bad).hexdigest(), bytes=len(bad))
    p.write_text(json.dumps(doc))
    v = _verify(p, V2)
    assert v["verified_lower"] == 1 and "did not accept" in _bnb_item(v)["problem"], v["checks"]


@needs_v2
def test_wrap_coverage_v2(v2cert, tmp_path, capsys):
    """wrap-coverage takes the policy from the v2 rule set; verify then binds it like any v2 evidence."""
    doc = json.loads(v2cert.read_text())
    e = _bnb(doc)
    cov = v2cert.parent / e["artifact_dir"] / e["coverage"]["file"]
    code = tmp_path / "toric_4.json"
    shutil.copy(EXAMPLES / "toric_4.json", code)
    out = tmp_path / "w.json"
    assert main(["wrap-coverage", str(code), str(cov), "-o", str(out), "--json"]) == 0
    capsys.readouterr()
    w = json.loads(out.read_text())
    assert _bnb(w)["params"]["bound"] == "greedy+s0s1s2"
    v = _verify(out, V2)
    assert v["verified_lower"] == 4, v["checks"]


def test_resolve_checker_v2(tmp_path, monkeypatch):
    from hyqec_verify.core import CodeError
    monkeypatch.setenv("PATH", str(tmp_path))
    for name in ("hyqec-check", "hyqec-check-v2"):
        x = tmp_path / name
        x.write_text("#!/bin/sh\n")
        x.chmod(0o755)
    assert resolve_checker("auto", version=2).startswith(str(tmp_path / "hyqec-check-v2"))
    assert resolve_checker("auto", version=1).startswith(str(tmp_path / "hyqec-check "))
    src = tmp_path / "src"
    (src / "hyqec_check_v2").mkdir(parents=True)
    assert "hyqec_check_v2" in resolve_checker(str(src), version=2)
    with pytest.raises(CodeError):
        resolve_checker(str(src), version=1)                     # no v1 package in that directory
    monkeypatch.chdir(tmp_path)
    assert resolve_checker("./hyqec-check-v2", version=2).startswith(str(tmp_path / "hyqec-check-v2"))


@pytest.mark.parametrize("version", ["__drop__", "1.0.0", "1.3.0", "2.0", "2.0.0rc1", {"other": "99"}, 2])
def test_v2_checker_version_required(v2cert, tmp_path, version):
    """The v2 label repeats the checker's reported version, so it must be a version string >= 2.0.0."""
    v = _verify(v2cert, _fake_checker(tmp_path, {"checker_version": version}))
    if version == "2.0":                                         # "2.0" == 2.0.0
        assert v["verified_lower"] == 4
        return
    assert v["verified_lower"] == 1 and "checker_version" in _bnb_item(v)["problem"], v["checks"]


def test_v2_label_names_the_configured_checker(v2cert, tmp_path):
    """A custom checker is labelled by its own name, with the version it reports (metadata, not authentication)."""
    v = _verify(v2cert, _fake_checker(tmp_path, {"checker_version": "7.1.0"}))
    assert v["verified_lower"] == 4
    assert _bnb_item(v)["evidence_level"].endswith("independently checked by fake-checker (reported version 7.1.0)")


def test_blank_or_missing_checker_fails_the_item_only(v2cert, tmp_path):
    v = _verify(v2cert, "   ")                                   # blank: as if no checker was given
    assert v["verified_lower"] == 1 and "not checked (no --bnb-checker)" in _bnb_item(v)["evidence_level"]
    for spec in (str(tmp_path / "no-such-checker"), str(tmp_path / "no-such-checker") + " {file}"):
        v = _verify(v2cert, spec)                                # never an exception, never a bound
        assert v["verified_lower"] == 1 and _bnb_item(v)["ok"] is False, spec
