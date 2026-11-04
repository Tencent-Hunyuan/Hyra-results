"""Public certificate verifier; native models are a private validated bridge."""

import json
from pathlib import Path
import tempfile
import time

from hyqec_check.algebra import ValidationError, integer
from . import __version__
from .certificate import (DEFAULT_MAX_HEADER, DEFAULT_MAX_PERMUTATION_WORK,
                          prepare_certificate, write_native_model)
from hyqec_check.engines import engine_path
from hyqec_check.transport import run_stream

DEFAULT_ENGINE = engine_path("hyqec-check-v2-native-v2.0.0")


def verify(path, *, threads, mode="all", start=0, stop=0, sample=0, seed=0,
           expected_group=None, expected_instance=None, expected_domain=None, expected_cap=None,
           scratch_dir=None, max_header=DEFAULT_MAX_HEADER, max_proof_bytes=1 << 40,
           max_permutation_work=DEFAULT_MAX_PERMUTATION_WORK, engine=DEFAULT_ENGINE):
    # A v2 entry point also implements the frozen v1 profile.
    from .transport import container_version
    if container_version(Path(path)) == 1:
        from hyqec_check_wide.verify import verify as verify_v1
        return verify_v1(path, threads=threads, mode=mode, start=start, stop=stop,
                         sample=sample, seed=seed, expected_group=expected_group,
                         expected_instance=expected_instance, expected_domain=expected_domain,
                         expected_cap=expected_cap, scratch_dir=scratch_dir, max_header=max_header,
                         max_proof_bytes=max_proof_bytes, max_permutation_work=max_permutation_work)
    started = time.monotonic()
    for name, value in (("threads", threads), ("start", start), ("stop", stop),
                        ("sample", sample), ("seed", seed), ("max_header", max_header),
                        ("max_proof_bytes", max_proof_bytes),
                        ("max_permutation_work", max_permutation_work)):
        integer(value, name)
        if value > (1 << 63) - 1:
            raise ValidationError(f"{name} exceeds supported limit")
    if not 1 <= threads <= 384:
        raise ValidationError("threads must be in 1..384")
    if mode not in ("all", "sample", "range"):
        raise ValidationError("unsupported task selection mode")
    if expected_cap is not None:
        integer(expected_cap, "expected cap")
    with prepare_certificate(Path(path), max_header=max_header,
                             max_permutation_work=max_permutation_work) as (header, instance, stream), \
            tempfile.TemporaryDirectory(prefix="hyqec-check-", dir=scratch_dir) as directory:
        scratch = Path(directory)
        for key, requested in (("group_sha256", expected_group), ("instance_sha256", expected_instance),
                               ("domain", expected_domain), ("cap", expected_cap)):
            if requested is not None and header[key] != requested:
                raise ValidationError(f"validated {key} does not match requested value")
        model = scratch / "validated-model.txt"
        write_native_model(model, header, instance)
        command = [str(engine), str(model), "/dev/stdin", "0", str(scratch / "frontier.bin"),
                   str(threads), mode, str(start), str(stop), str(sample), str(seed), "1"]
        process = run_stream(command, stream, max_proof_bytes)
        try:
            verdict = json.loads(process.stdout)
        except json.JSONDecodeError as error:
            raise ValidationError(f"native checker failed without a verdict (exit {process.returncode})") from error
        if process.returncode and verdict.get("status") != "rejected":
            raise ValidationError(f"native checker failed (exit {process.returncode})")
        verdict.update({"checker_version": __version__, "spec_revision": "2.0-rc1", "limits_profile": "v2-wide", "rules": header["rules"], "rules_version": header["rules_version"], "instance_sha256": instance.instance_sha256, "group_sha256": instance.group_sha256,
                        "n": instance.code.n, "k": instance.code.k, "cap": header["cap"],
                        "domain": instance.domain, "selection": mode, "seed": seed if mode == "sample" else None,
                        "total_wall_seconds": time.monotonic() - started})
        verdict.setdefault("tasks_checked", 0)
        verdict.setdefault("tasks_total", None)
        if "global_trace" in verdict:
            verdict["global_trace_hex"] = f"{int(verdict['global_trace']):016x}"
        if mode == "sample":
            verdict["sample_size"] = sample
        elif mode == "range":
            verdict["task_range"] = [start, stop]
        if verdict.get("certificate_verified"):
            verdict["statement"] = (f"Verified the instance algebra, automorphisms, complete frontier, every task "
                                     f"and footer: no nontrivial {instance.domain} logical Pauli has weight "
                                     f"<= {header['cap']}; the domain distance is at least {header['cap'] + 1} "
                                     "if a logical in that domain exists.")
        elif verdict.get("status") == "partial":
            verdict["statement"] = (f"Verified instance algebra, automorphisms, complete top frontier and "
                                     f"{verdict['tasks_checked']} of {verdict['tasks_total']} tasks. "
                                     "Unchecked task summaries were used only to validate footer consistency. "
                                     "No global distance lower bound is established.")
        else:
            verdict["statement"] = "Certificate rejected; no distance lower bound established."
        return verdict
