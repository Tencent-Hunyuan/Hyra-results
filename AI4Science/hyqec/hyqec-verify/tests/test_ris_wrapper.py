"""Python wrapper around the native RIS binary (skipped when it cannot be built)."""
import pytest

from known_codes import known


def _available():
    try:
        from hyqec_verify import native
        native.locate("ris-cpu")
        return True
    except Exception:
        return False


pytestmark = [pytest.mark.native, pytest.mark.skipif(not _available(), reason="native RIS unavailable")]


def test_native_ris_known_codes():
    from hyqec_verify.engines import ris_native
    for c, k, d, dx, dz in known():
        r = ris_native.run(c, seconds=1, seed=1)
        assert r["upper_bound"] == d and r["rejected_by_core"] == 0, c.name
        assert c.check_witness(r["witness_int"]) == d
        for dom, ds in (("X", dx), ("Z", dz)):
            if ds:
                rr = ris_native.run(c, seconds=1, seed=1, domain=dom)
                assert rr["upper_bound"] == ds, (c.name, dom)


def test_upper_auto_easy_code_stays_on_cpu(capsys):
    """surface5-type easy code, no GPU: CPU only, weight 5, runs the requested time."""
    import json
    import time
    from pathlib import Path
    from hyqec_verify.cli import main
    ex = Path(__file__).resolve().parents[1] / "examples" / "xzzx_surface_5.json"
    t0 = time.monotonic()
    assert main(["upper", str(ex), "--seconds", "3", "--threads", "1", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["upper_bound"] == 5 and out["engine"] == "ris-cpu" and not out["notes"]
    assert time.monotonic() - t0 < 3 + 5


def _cuda_device():
    # only when the runner explicitly chose a device (shared machines: never grab GPU 0 by default)
    import os
    import shutil
    import subprocess
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or not shutil.which("nvidia-smi"):
        return False
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_device(), reason="no CUDA device")
def test_upper_auto_hard_code_moves_to_gpu(capsys):
    """452-115-18 (d = 18): with --gpu the stalled CPU phase hands over to the GPU and the search
    keeps going; it must reach weight <= 18 (observed at ~26 s on a shared H20; 40 s allowed)."""
    import json
    from pathlib import Path
    from hyqec_verify.cli import main
    import os
    # HYQEC_TEST_UPSTREAM: a checkout of the qldpc-challenge repository (codes/452-115-18.json)
    code = Path(os.environ.get("HYQEC_TEST_UPSTREAM", "/nonexistent")) / "codes" / "452-115-18.json"
    if not code.exists():
        pytest.skip("set HYQEC_TEST_UPSTREAM to a qldpc-challenge checkout (codes/452-115-18.json)")
    assert main(["upper", str(code), "--seconds", "40", "--threads", "1", "--gpu", "0", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["upper_bound"] <= 18, out
    assert any("GPU" in n for n in out["notes"]) or out["engine"] == "ris-cuda"
