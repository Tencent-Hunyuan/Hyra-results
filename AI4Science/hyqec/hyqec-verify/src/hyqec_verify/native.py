"""Build optional native engines into a content-addressed, atomic cache.

No compiler runs during installation or import. CXX/NVCC select compilers;
HYQEC_OPENMP=0 disables OpenMP, =1 requires it, and the default probes it.
HYQEC_CXXFLAGS/HYQEC_NVCCFLAGS append explicitly requested build flags.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

from ._env import getenv

TARGETS = ("bnb-cpu", "bnb-cuda", "ris-cpu", "ris-cuda")
_SOURCES = Path(__file__).with_name("native")


class NativeUnavailable(RuntimeError):
    """A native engine could not be located or built."""


def _target(target: str) -> tuple[str, bool]:
    if target not in TARGETS:
        raise ValueError(f"unknown native target {target!r}; expected one of {TARGETS}")
    kind, backend = target.split("-")
    return kind, backend == "cuda"


def _variant(variant) -> str:
    value = str(variant or 1024)
    base = value[:-len("-rules")] if value.endswith("-rules") else value
    if base not in ("1024", "4096", "1024-w64", "1024-w128", "4096-w64", "4096-w128"):
        raise ValueError("variant must be 1024 or 4096, optionally with CPU B&B suffix -w64 or -w128, "
                         "then optionally -rules")
    return value


def _root() -> Path:
    return Path(getenv("CACHE") or (Path.home() / ".cache" / "hyqec-verify")) / "native"


def _source_hash(kind: str) -> str:
    digest = hashlib.sha256()
    paths = sorted(p for d in (kind, "common") for p in (_SOURCES / d).rglob("*") if p.is_file())
    if not paths:
        raise NativeUnavailable(f"sources missing under {_SOURCES / kind}")
    for path in paths:
        relative = path.relative_to(_SOURCES).as_posix().encode()
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little") + relative)
        digest.update(len(data).to_bytes(8, "little") + data)
    return digest.hexdigest()


def _compiler(cuda: bool) -> list[str] | None:
    command = shlex.split(os.environ.get("NVCC" if cuda else "CXX", "nvcc" if cuda else "c++"))
    if not command:
        return None
    path = shutil.which(command[0])
    # Preserve the driver name: resolving clang++ to clang changes link mode.
    return [str(Path(path).absolute()), *command[1:]] if path else None


def _run(command: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kw)


def version(path) -> dict:
    result = _run([str(path), "--version"], timeout=30)
    if result.returncode:
        raise NativeUnavailable(f"{path} --version failed: {result.stdout.strip()}")
    try:
        value = json.loads(result.stdout)
        if value["interface"] != 1 or not isinstance(value["source_sha256"], str):
            raise ValueError("unsupported interface")
        return value
    except (ValueError, KeyError, TypeError) as exc:
        raise NativeUnavailable(f"invalid version response from {path}: {result.stdout!r}") from exc


def _cached(target: str, variant: str, source: str, arch: str | None = None) -> Path | None:
    for entry in sorted(_root().glob(f"{target}-{variant}-*/build.json"), reverse=True):
        try:
            data = json.loads(entry.read_text())
            binary = entry.parent / data["binary"]
            if (data["source_sha256"] == source and (arch is None or data["cuda_arch"] == arch)
                    and binary.is_file()
                    and hashlib.sha256(binary.read_bytes()).hexdigest() == data["binary_sha256"]
                    and version(binary)["source_sha256"] == source):
                return binary
        except (OSError, ValueError, KeyError, NativeUnavailable):
            continue
    return None


def _zlib_error(compiler: list[str], cuda: bool) -> str | None:
    """Compile/link a tiny host-only probe; never runs on import or installation."""
    extra = shlex.split(getenv("NVCCFLAGS" if cuda else "CXXFLAGS", ""))
    with tempfile.TemporaryDirectory(prefix="hyqec-zlib-") as temp:
        probe = Path(temp) / "zlib.cpp"
        probe.write_text('#include <zlib.h>\nint main() { return crc32(0, Z_NULL, 0) != 0; }\n')
        try:
            result = _run([*compiler, *extra, str(probe), "-lz", "-o", str(Path(temp) / "probe")], timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            detail = str(exc)
        else:
            if result.returncode == 0:
                return None
            detail = result.stdout.strip()
    return ("B&B requires zlib development headers and library; compile/link probe failed. "
            "Install zlib1g-dev (Debian/Ubuntu), zlib-devel (Fedora/RHEL), or conda install zlib; "
            "check compiler/include/library flags if already installed.\n" + detail)


def availability(target: str) -> tuple[bool, str | None]:
    """Availability plus an actionable reason when no usable build is present."""
    kind, cuda = _target(target)
    try:
        if _cached(target, "1024", _source_hash(kind)) is not None:
            return True, None
        if getenv("NO_BUILD") == "1":
            return False, "HYQEC_NO_BUILD=1 and no matching cached binary"
        compiler = _compiler(cuda)
        if compiler is None:
            return False, "compiler missing"
        reason = _zlib_error(compiler, cuda) if kind == "bnb" else None
        return reason is None, reason
    except (OSError, NativeUnavailable) as exc:
        return False, str(exc)


def available(target: str) -> bool:
    return availability(target)[0]


def locate(target: str, variant=None) -> Path:
    return build(target, variant=variant)


def build(target: str, *, cuda_arch="native", variant=None, force=False, verbose=False) -> Path:
    kind, cuda = _target(target)
    variant = _variant(variant)
    rules = variant.endswith("-rules")
    size, _, width = variant.removesuffix("-rules").partition("-w")
    if width and (kind != "bnb" or cuda):
        raise ValueError("wide variants are supported only for bnb-cpu")
    if rules and (kind != "bnb" or cuda):
        raise ValueError("the -rules variant (opt-in S0/S1/S2 policies) exists only for bnb-cpu")
    source = _source_hash(kind)
    arch = (cuda_arch or "native") if cuda else None
    if cuda and arch != "native":
        arches = arch.split(",")
        if not all(a.isdigit() and 2 <= len(a) <= 3 for a in arches):
            raise ValueError("cuda_arch must be 'native' or a comma-separated list, e.g. '80,90'")
    compiler = _compiler(cuda)
    if getenv("NO_BUILD") == "1" or compiler is None:
        cached = _cached(target, variant, source, arch)
        if cached:
            return cached
        reason = "HYQEC_NO_BUILD=1" if getenv("NO_BUILD") == "1" else "compiler missing"
        raise NativeUnavailable(f"{target} unavailable: {reason}, no matching cached binary")
    try:
        compiler_version = _run([*compiler, "--version"], timeout=30).stdout.splitlines()[0]
    except (OSError, IndexError, subprocess.TimeoutExpired) as exc:
        raise NativeUnavailable(f"could not query compiler {compiler}") from exc
    flags = ["-O3", "-std=c++17", f"-DNMAX={size}", f"-DMMAX={size}",
             f'-DSTABCERT_SOURCE_SHA256="{source}"']
    if width:
        flags += [f"-DWMAX={width}", "-DCMAX=128"]
    if rules:
        # Opt-in S0/S1/S2 policies (docs/SOUNDNESS.md 5.1); every other build compiles without them.
        flags += ["-DSTABCERT_RULES=1"]
    if cuda:
        flags += ["--expt-relaxed-constexpr", f'-DSTABCERT_CUDA_ARCH="{arch}"']
        if arch == "native":
            flags += ["-arch=native"]
        else:
            flags += [f"-gencode=arch=compute_{a},code=sm_{a}" for a in arches]
            flags += [f"-gencode=arch=compute_{arches[-1]},code=compute_{arches[-1]}"]
    elif kind == "bnb":
        flags += ["-x", "c++"]
    # The B&B work queue uses std::thread independently of OpenMP.
    flags += ["-Xcompiler", "-pthread"] if cuda else ["-pthread"]
    extra = shlex.split(getenv("NVCCFLAGS" if cuda else "CXXFLAGS", ""))
    openmp = getenv("OPENMP", "auto")
    if openmp not in ("auto", "0", "1"):
        raise ValueError("HYQEC_OPENMP must be auto, 0 or 1")
    omp_flags = ["-Xcompiler", "-fopenmp"] if cuda else ["-fopenmp"]
    if openmp != "0":
        with tempfile.TemporaryDirectory(prefix="hyqec-probe-") as temp:
            probe = Path(temp) / ("omp.cu" if cuda else "omp.cpp")
            probe.write_text("#include <omp.h>\nint main() { return omp_get_max_threads() < 1; }\n")
            result = _run([*compiler, *extra, *omp_flags, str(probe), "-o", str(Path(temp) / "probe")])
            if result.returncode == 0:
                flags += omp_flags
            elif openmp == "1":
                raise NativeUnavailable(f"OpenMP requested but compiler probe failed:\n{result.stdout}")
    flags += extra
    libraries = ["-lz"] if kind == "bnb" else []
    key_data = dict(interface=1, target=target, variant=variant, source_sha256=source,
                    compiler=compiler, compiler_version=compiler_version, flags=flags)
    if libraries:
        key_data["libraries"] = libraries
    # -arch=native depends on visible hardware as well as compiler flags.
    if cuda and arch == "native":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        query = ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
        if visible:
            query += [f"--id={visible}"]
        try:
            hardware = _run(query, timeout=30)
            key_data["native_compute_caps"] = hardware.stdout.strip() if hardware.returncode == 0 else "unknown"
        except (OSError, subprocess.TimeoutExpired):
            key_data["native_compute_caps"] = "unknown"
    key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    cache = _root() / f"{target}-{variant}-{key[:16]}"
    binary_name = "stabcert-" + kind + ("-cuda" if cuda else "")
    binary = cache / binary_name
    if binary.exists() and not force:
        cached = _cached(target, variant, source, arch)
        if cached == binary:
            return binary
        try:
            meta = json.loads((cache / "build.json").read_text())
            if (hashlib.sha256(binary.read_bytes()).hexdigest() == meta["binary_sha256"]
                    and version(binary)["source_sha256"] == source):
                return binary
        except (OSError, ValueError, KeyError, NativeUnavailable):
            pass
    if kind == "bnb":
        reason = _zlib_error(compiler, cuda)
        if reason:
            raise NativeUnavailable(reason)
    _root().mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{target}-", dir=_root()))
    source_name = "bnb.cu" if kind == "bnb" else "ris.cu" if cuda else "ris_cpu.cpp"
    command = [*compiler, *flags, str(_SOURCES / kind / source_name), *libraries, "-o", str(temp / binary_name)]
    try:
        if verbose:
            print(shlex.join(command))
        result = _run(command)
        (temp / "build.log").write_text(result.stdout)
        if result.returncode:
            failed = temp.with_name(temp.name + ".failed")
            temp.rename(failed)
            raise NativeUnavailable(f"{target} compile failed; log: {failed / 'build.log'}\n{result.stdout[-4000:]}")
        if version(temp / binary_name)["source_sha256"] != source:
            raise NativeUnavailable("compiled source hash does not match the build inputs")
        metadata = dict(key_data, key=key, command=command, binary=binary_name, cuda_arch=arch,
                        binary_sha256=hashlib.sha256((temp / binary_name).read_bytes()).hexdigest())
        (temp / "build.json").write_text(json.dumps(metadata, indent=2) + "\n")
        # Serialize publication only; concurrent compiles remain independent.
        import fcntl
        with (_root() / ".publish.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if cache.exists():
                if not force and binary.exists():
                    existing = json.loads((cache / "build.json").read_text())
                    if existing.get("binary_sha256") == hashlib.sha256(binary.read_bytes()).hexdigest():
                        return binary
                old = Path(tempfile.mkdtemp(prefix=".old-", dir=_root()))
                old.rmdir()
                cache.rename(old)
                try:
                    temp.rename(cache)
                finally:
                    shutil.rmtree(old)
            else:
                temp.rename(cache)
        return binary
    finally:
        if temp.exists():
            shutil.rmtree(temp)
