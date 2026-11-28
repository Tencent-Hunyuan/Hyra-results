"""Offline tests for hyprover.verify — no codex, no API spend.

The full request path (schema validation → subprocess spawn →
verification.json readback → verdict propagation) is exercised
by pointing HYPROVER_CODEX_BIN at ``fake_codex.py`` (a deterministic stub) and calling the
``/verify`` endpoint function directly (avoids an httpx TestClient dependency).

Runs standalone (``python -m hyprover.verify.tests.test_verify``) and under pytest.
"""

from __future__ import annotations

import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path

from hyprover.verify.service import VerifyRequest, verify

FAKE = Path(__file__).resolve().parent / "fake_codex.py"

_GOOD_STATEMENT = "For every integer n, n + 0 equals n."
_GOOD_PROOF = (
    "Zero is the additive identity of the integers, so adding zero to any integer n "
    "leaves the value unchanged. Hence n + 0 = n for every integer n, as required."
)


@contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    try:
        for k, v in kv.items():
            os.environ[k] = v if v is not None else os.environ.get(k, "")
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _ensure_fake_executable():
    FAKE.chmod(FAKE.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _call(statement, proof, tmp):
    with _env(HYPROVER_CODEX_BIN=str(FAKE)), \
            _env(VERIFIER_RESULTS_DIR=str(Path(tmp) / "runs"), VERIFY_AGENT_HOME=str(tmp)):
        return verify(VerifyRequest(statement=statement, proof=proof))


def test_verify_accept_via_fake_codex():
    _ensure_fake_executable()
    with tempfile.TemporaryDirectory() as tmp:
        out = _call(_GOOD_STATEMENT, _GOOD_PROOF, tmp)
        assert out["verdict"] == "correct" and out["verification_report"]["critical_errors"] == []


def test_verify_reject_via_fake_codex():
    _ensure_fake_executable()
    with tempfile.TemporaryDirectory() as tmp:
        out = _call(_GOOD_STATEMENT, _GOOD_PROOF + " [[FAKE:wrong]]", tmp)
        assert out["verdict"] == "wrong" and out["repair_hints"]


def test_short_and_legacy_phrases_reach_verifier_subprocess():
    _ensure_fake_executable()
    for proof in ("QED", "中文证明内容", "The result holds as declared in problem.md."):
        with tempfile.TemporaryDirectory() as tmp:
            out = _call("x=x", proof + " [[FAKE:wrong]]", tmp)
            assert out["verdict"] == "wrong" and out["repair_hints"]
            assert len(list((Path(tmp) / "runs").glob("*/verification.json"))) == 1


def main() -> None:
    test_verify_accept_via_fake_codex()
    print("  [ok] /verify accept via fake_codex -> verdict correct")
    test_verify_reject_via_fake_codex()
    print("  [ok] /verify reject via fake_codex ([[FAKE:wrong]]) -> verdict wrong")
    test_short_and_legacy_phrases_reach_verifier_subprocess()
    print("  [ok] short and legacy phrases reach independent verifier")
    print("ALL VERIFY TESTS PASSED")


if __name__ == "__main__":
    main()
