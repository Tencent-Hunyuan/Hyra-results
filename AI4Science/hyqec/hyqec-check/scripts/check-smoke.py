"""Review supplied artifact bytes/manifests without importing exporter code."""
import gzip
import hashlib
import json
from hyqec_check.engines import environment
from pathlib import Path
import struct
import subprocess
import sys

CHECKER_MODULE = environment("MODULE", "hyqec_check")
CHECKER_MODULE = {"bnbcheck": "hyqec_check", "bnbcheck_wide": "hyqec_check_wide"}.get(CHECKER_MODULE, CHECKER_MODULE)
if CHECKER_MODULE == "hyqec_check":
    from hyqec_check.certificate import MAGIC, parse_instance
elif CHECKER_MODULE == "hyqec_check_wide":
    from hyqec_check_wide.certificate import MAGIC, parse_instance
else:
    raise ValueError("HYQEC_CHECK_MODULE must be hyqec_check or hyqec_check_wide")
from hyqec_check.input import strict_json


def task_path_bytes(payload):
    """Read path bytes from a stream already accepted by the checker."""
    at = 0
    observed = set()

    def integer():
        nonlocal at
        value, shift = 0, 0
        while True:
            byte = payload[at]
            at += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value
            shift += 7

    while at < len(payload):
        tag = payload[at]
        at += 1
        if tag == 0x10:
            integer(); integer()
            rule = payload[at]
            at += 9  # rule byte and fingerprint
            integer(); integer()
            if rule == 2:
                for _ in range(integer()):
                    integer()
            elif rule == 3:
                integer()
        elif tag in (0x11, 0x12, 0x13, 0x1f):
            integer(); integer(); integer()
            at += 8
        elif tag == 0x20:
            integer(); integer()
            length = integer()
            observed.update(payload[at:at + length])
            at += length
            integer()
            at += 8
        elif tag in (0x21, 0x22):
            integer()
        elif tag == 0x7f:
            integer(); integer(); integer(); integer()
            at += 8
        else:
            raise ValueError(f"unexpected accepted record tag {tag}")
    assert at == len(payload)
    return sorted(observed)


source = Path(sys.argv[1])
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
manifest_bytes = (source / "results.json").read_bytes()
manifest = strict_json(manifest_bytes)
reviews = []
pairings = {}
pair_count = 0
runtime_task_differences = []
all_path_bytes = set()
for record in manifest:
    if "file" in record:
        raw_path = source / Path(record["file"]).name
        gzip_path = source / Path(record["gzip"]).name
        stem = raw_path.stem
    else:
        stem = f"{record['case']}-{record['engine']}-s{record['split']}-{record['detail']}"
        raw_path = source / (stem + ".cov")
        gzip_path = source / (stem + ".cov.gz")
    raw = raw_path.read_bytes()
    compressed = gzip_path.read_bytes()
    assert raw[:8] == MAGIC, stem + ": wrong magic"
    length, = struct.unpack("<Q", raw[8:16])
    header = strict_json(raw[16:16 + length])
    payload = raw[16 + length:]
    measured = {"raw_sha256": hashlib.sha256(raw).hexdigest(),
                "gzip_sha256": hashlib.sha256(compressed).hexdigest(),
                "payload_sha256": hashlib.sha256(payload).hexdigest()}
    for key, value in measured.items():
        if key in record:
            assert value == record[key], stem + ": manifest " + key + " mismatch"
    assert gzip.decompress(compressed) == raw, stem + ": gzip/raw bytes differ"
    assert compressed[4:8] == bytes(4) and not (compressed[3] & 8), stem + ": nondeterministic gzip header"
    for key in ("split", "detail"):
        assert header[key] == record[key], stem + ": manifest " + key + " mismatch"
    if "file" in record:
        instance_bytes = (source / (stem + ".inst")).read_bytes()
        instance = parse_instance(instance_bytes.decode("ascii"))
        metadata = strict_json((source / (stem + ".json")).read_bytes())
        assert instance_bytes == header["instance_text"].encode("ascii"), stem + ": instance bytes differ"
        assert instance.instance_sha256 == record["instance_sha256"], stem + ": instance hash mismatch"
        assert instance.group_sha256 == record["group_sha256"] == metadata["group_sha256"], stem + ": group hash mismatch"
        assert header["permutations"] == metadata["permutations"], stem + ": permutation metadata differs"
        assert header["cap"] == record["cap"] and header["domain"] == record["domain"], stem + ": claim differs"
    else:
        normalized = {k: v for k, v in header.items() if k != "provenance"}
        key = (record["case"], record["split"], record["detail"])
        if key in pairings:
            assert pairings[key] == (normalized, payload), stem + ": CPU/CUDA normalized header or payload differs"
            pair_count += 1
        else:
            pairings[key] = (normalized, payload)
    for path in (raw_path, gzip_path):
        command = [sys.executable, "-m", CHECKER_MODULE, "verify", str(path), "--threads", "1",
                   "--expected-cap", str(record["done"]["cap0"]),
                   "--expected-domain", record["done"]["domain"],
                   "--expected-instance", record["done"]["instance_sha256"]]
        if "group_sha256" in record:
            command += ["--expected-group", record["group_sha256"]]
        result = subprocess.run(command, text=True, capture_output=True)
        name = path.name
        (output / (name + ".verdict.json")).write_text(result.stdout)
        (output / (name + ".stderr.txt")).write_text(result.stderr)
        verdict = json.loads(result.stdout)
        review = {"artifact": str(path), "command": command, "returncode": result.returncode,
                  "measured_hashes": measured, "verdict": verdict}
        reviews.append(review)
        (output / "review.json").write_text(json.dumps(reviews, indent=2) + "\n")
        assert result.returncode == 0 and verdict["certificate_verified"], name + ": independent rejection"
        assert verdict["total_nodes"] == record["done"]["nodes"], name + ": run-record node mismatch"
        if verdict["tasks_total"] != record["done"]["tasks_total"]:
            # Spec section 9 permits physical scheduler counters in external
            # run records. They are not the canonical certificate task count.
            assert record["done"].get("aborted_total", 0) > 0, name + ": unexplained run-record task mismatch"
            runtime_task_differences.append({"artifact": name, "canonical_tasks": verdict["tasks_total"],
                                             "runtime_tasks": record["done"]["tasks_total"],
                                             "aborted_total": record["done"]["aborted_total"]})
        assert verdict["global_trace_hex"] == record["done"]["global_trace"], name + ": run-record trace mismatch"
        paths = task_path_bytes(payload)
        if "path_bytes" in record:
            assert paths == record["path_bytes"], name + ": manifest path bytes differ"
        all_path_bytes.update(paths)
        print(name, verdict["status"], verdict["total_nodes"], verdict["global_trace_hex"], flush=True)
summary = {"checker_module": CHECKER_MODULE, "artifacts_checked": len(reviews), "manifest_runs": len(manifest),
           "cpu_cuda_pairs_compared": pair_count,
           "observed_path_bytes": sorted(all_path_bytes),
           "observed_support_positions": sorted({byte >> 1 for byte in all_path_bytes}),
           "runtime_task_count_differences": runtime_task_differences,
           "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "mismatches": []}
(output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary), flush=True)
