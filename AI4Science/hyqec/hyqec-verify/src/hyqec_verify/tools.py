"""External SAT tools: discovery, the vendored proof checkers, pinned solver builds.

* Proof checkers ``drat-trim`` and ``lrat-check`` are vendored (MIT, see
  ``vendor/drat-trim/``) and compiled with the local C compiler on first use.
* Solvers (Kissat, CaDiCaL) are found via ``HYQEC_KISSAT`` / ``HYQEC_CADICAL``,
  then the hyqec-verify cache (``hyqec-verify tools install``), then ``PATH``.
  Solvers are never trusted: an UNSAT answer counts only with a checked proof,
  a SAT answer only with a model that the core verifies.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from ._env import getenv

__all__ = ["cache_dir", "find_solver", "checker", "lrat_checker", "install_solver", "tool_info", "PINS",
           "ToolMissing"]

VENDOR = Path(__file__).resolve().parent / "vendor" / "drat-trim"

# the solver revisions pinned for reproducible proofs
PINS = {
    "kissat": {"url": "https://github.com/arminbiere/kissat", "rev": "8af8e56f174b778aef3aa45af9f739b2a5f492c2",
               "configure": ["./configure"], "binary": "build/kissat"},
    "cadical": {"url": "https://github.com/arminbiere/cadical", "rev": "c60730422e758ef1cebe7aeddf2dda31c996bf04",
                "configure": ["./configure"], "binary": "build/cadical"},
}


class ToolMissing(RuntimeError):
    pass


def cache_dir() -> Path:
    d = getenv("CACHE")
    return Path(d) if d else Path.home() / ".cache" / "hyqec-verify"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _cc() -> str:
    for c in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if c and shutil.which(c):
            return c
    raise ToolMissing("no C compiler found (set CC) to build the vendored proof checker")


def _build_vendored(name: str, source: str, flags: list[str]) -> Path:
    src = VENDOR / source
    key = hashlib.sha256(src.read_bytes() + " ".join(flags).encode()).hexdigest()[:16]
    out_dir = cache_dir() / "tools" / f"{name}-{key}"
    exe = out_dir / name
    if exe.exists():
        return exe
    if getenv("NO_BUILD"):
        raise ToolMissing(f"{name} not built and HYQEC_NO_BUILD is set")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=out_dir.parent, prefix=f".{name}-"))
    cmd = [_cc(), "-std=c99", "-O2", *flags, str(src), "-o", str(tmp / name)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    (tmp / "build.json").write_text(json.dumps({"cmd": cmd, "returncode": p.returncode,
                                                "stderr": p.stderr[-4000:]}, indent=1))
    if p.returncode != 0:
        raise ToolMissing(f"building {name} failed: {p.stderr[-500:]}")
    try:
        tmp.rename(out_dir)
    except OSError:              # a concurrent build won the race
        shutil.rmtree(tmp, ignore_errors=True)
    return exe


def checker() -> Path:
    """Path of drat-trim (env HYQEC_DRAT_TRIM overrides the vendored build)."""
    env = getenv("DRAT_TRIM")
    if env:
        return Path(env)
    return _build_vendored("drat-trim", "drat-trim.c", [])


def lrat_checker() -> Path:
    env = getenv("LRAT_CHECK")
    if env:
        return Path(env)
    return _build_vendored("lrat-check", "lrat-check.c", ["-DLONGTYPE"])


def _installed(name: str) -> Path | None:
    d = cache_dir() / "tools" / f"{name}-{PINS[name]['rev'][:12]}"
    exe = d / PINS[name]["binary"]
    return exe if exe.exists() else None


def find_solver(name: str = "kissat") -> Path:
    if name not in PINS:
        raise ValueError(f"unknown solver {name!r}")
    env = getenv(name.upper())
    if env:
        return Path(env)
    p = _installed(name)
    if p:
        return p
    w = shutil.which(name)
    if w:
        return Path(w)
    raise ToolMissing(f"{name} not found: run `hyqec-verify tools install {name}` or set HYQEC_{name.upper()}")


def available_solvers() -> list[str]:
    out = []
    for s in PINS:
        try:
            find_solver(s)
            out.append(s)
        except ToolMissing:
            pass
    return out


def install_solver(name: str, from_source: str | None = None, jobs: int = 8, log=print) -> Path:
    """Build a pinned solver revision into the cache (git clone, or a local source tree)."""
    pin = PINS[name]
    dest = cache_dir() / "tools" / f"{name}-{pin['rev'][:12]}"
    exe = dest / pin["binary"]
    if exe.exists():
        return exe
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=dest.parent, prefix=f".{name}-"))
    src = tmp / "src"
    if from_source:
        shutil.copytree(from_source, src, symlinks=True, ignore=shutil.ignore_patterns("build"))
    else:
        log(f"cloning {pin['url']} @ {pin['rev'][:12]}")
        subprocess.run(["git", "clone", "-q", pin["url"], str(src)], check=True)
        subprocess.run(["git", "-C", str(src), "checkout", "-q", pin["rev"]], check=True)
    log(f"building {name}")
    subprocess.run(pin["configure"], cwd=src, check=True, capture_output=True)
    subprocess.run(["make", f"-j{jobs}"], cwd=src, check=True, capture_output=True)
    if not (src / pin["binary"]).exists():
        raise ToolMissing(f"build of {name} produced no {pin['binary']}")
    shutil.rmtree(dest, ignore_errors=True)
    src.rename(dest)
    shutil.rmtree(tmp, ignore_errors=True)
    return exe


def tool_info(path) -> dict:
    path = Path(path)
    info = {"path": str(path), "sha256": sha256_file(path)}
    try:
        p = subprocess.run([str(path), "--version"], capture_output=True, text=True, timeout=10)
        line = (p.stdout or p.stderr).strip().splitlines()
        info["version"] = line[0] if line else None
    except Exception:
        info["version"] = None
    return info
