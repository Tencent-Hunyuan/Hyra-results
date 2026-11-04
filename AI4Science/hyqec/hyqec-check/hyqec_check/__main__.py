import json
import sys
import time

from .algebra import ValidationError, pauli_string
from .input import read_reference_instance
from .reference import search
from . import __version__
from .verify import verify
from hyqec_check.cli import ArgumentParser, bounded_integer, task_range
from .certificate import DEFAULT_MAX_PERMUTATION_WORK


def main() -> int:
    parser = ArgumentParser(description="Independent stabilizer B&B coverage certificate checker")
    parser.add_argument("--version", action="version", version=f"hyqec-check {__version__} (spec 1.1; original limits)")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-reference-code", help="validate checker-owned fixture format")
    validate.add_argument("instance")
    run = commands.add_parser("reference-search", help="execute local-policy reference B&B, not certificate replay")
    run.add_argument("instance")
    run.add_argument("--cap", required=True, type=bounded_integer())
    run.add_argument("--domain", choices=["P", "X", "Z"], default="P")
    run.add_argument("--threads", type=bounded_integer(), default=1)
    check = commands.add_parser("verify", help="verify a raw or gzip coverage certificate")
    check.add_argument("certificate")
    check.add_argument("--threads", type=bounded_integer(1, 384), required=True)
    selection = check.add_mutually_exclusive_group()
    selection.add_argument("--sample", type=bounded_integer(), help="replay exactly this many seeded random tasks")
    selection.add_argument("--range", type=task_range, metavar="START:STOP", help="replay a half-open task range")
    check.add_argument("--seed", type=bounded_integer(), default=0)
    check.add_argument("--expected-group")
    check.add_argument("--expected-instance")
    check.add_argument("--expected-domain", choices=["P", "X", "Z"])
    check.add_argument("--expected-cap", type=bounded_integer())
    check.add_argument("--scratch-dir")
    check.add_argument("--max-header-bytes", type=bounded_integer(), default=128 * 1024 * 1024)
    check.add_argument("--max-proof-bytes", type=bounded_integer(), default=1 << 40)
    check.add_argument("--max-permutation-work", type=bounded_integer(),
                       default=DEFAULT_MAX_PERMUTATION_WORK,
                       help="maximum permutation_count * n * max(1, m) (default: %(default)s)")
    args = parser.parse_args()
    started = time.monotonic()
    try:
        if args.command == "verify":
            mode, first, last = "all", 0, 0
            if args.sample is not None:
                mode = "sample"
            if args.range is not None:
                first, last = args.range
                mode = "range"
            verdict = verify(args.certificate, threads=args.threads, mode=mode, start=first, stop=last,
                             sample=args.sample or 0, seed=args.seed, expected_group=args.expected_group,
                             expected_instance=args.expected_instance, expected_domain=args.expected_domain,
                             expected_cap=args.expected_cap, scratch_dir=args.scratch_dir,
                             max_header=args.max_header_bytes, max_proof_bytes=args.max_proof_bytes,
                             max_permutation_work=args.max_permutation_work)
            print(json.dumps(verdict, sort_keys=True))
            if verdict["status"] == "rejected":
                return 1
            return 0 if verdict["certificate_verified"] else 2
        instance = read_reference_instance(args.instance)
        if args.command == "reference-search":
            verdict = search(instance, args.cap, args.domain, args.threads)
        else:
            code = instance.code
            verdict = {
                "mode": "algebra-validation", "n": code.n, "k": code.k,
                "rank": len(code.stabilizers), "normalizer_dimension": len(code.normalizer),
                "logical_basis": [pauli_string(v, code.n) for v in code.logicals],
                "permutations_verified": len(instance.permutations), "orbits": instance.orbits,
                "instance_file_sha256": instance.file_sha256,
                "certificate_verified": False,
                "statement": "Code isotropy, dimensions, logical quotient basis, supplied automorphisms "
                             "and orbit partition verified. No distance certificate checked.",
            }
        verdict["status"] = "ok"
        verdict["total_wall_seconds"] = time.monotonic() - started
        print(json.dumps(verdict, sort_keys=True))
        return 0
    except (ValidationError, OSError, ValueError, EOFError) as error:
        print(json.dumps({"status": "rejected", "certificate_verified": False,
                          "checker_version": __version__, "spec_revision": "1.1",
                          "instance_sha256": None, "cap": None, "tasks_checked": 0, "tasks_total": None,
                          "mismatches": [str(error)], "wall_seconds": time.monotonic() - started,
                          "statement": "Certificate rejected; no distance lower bound established."}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
