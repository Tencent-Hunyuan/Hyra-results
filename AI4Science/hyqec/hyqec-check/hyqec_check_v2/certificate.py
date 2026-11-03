"""V2 identifiers over the unchanged, checker-owned wide instance validator."""
from pathlib import Path

from hyqec_check.algebra import ValidationError
from hyqec_check_wide.certificate import (
    DEFAULT_MAX_HEADER, DEFAULT_MAX_PERMUTATION_WORK,
    validate_header as validate_v1_header, write_native_model as write_v1_model,
)
from .transport import open_certificate

MAGIC = b"SBCOV2\r\n"
RULES = ("greedy", "s0", "s0s1", "s0s1s2")


def validate_header(header, *, max_permutation_work=DEFAULT_MAX_PERMUTATION_WORK):
    if not isinstance(header, dict):
        raise ValidationError("coverage header must be a JSON object")
    for key, value in (("format", "stabcert-bnb-coverage/2"),
                       ("search", "irreducible-pauli-dfs/2"),
                       ("trace", "poly64-state-rules/2")):
        if header.get(key) != value:
            raise ValidationError(f"unsupported {key}")
    if header.get("rules") not in RULES:
        raise ValidationError("unsupported rules")
    if type(header.get("rules_version")) is not int or header["rules_version"] != 1:
        raise ValidationError("unsupported rules_version")
    adapted = dict(header, format="stabcert-bnb-coverage/1",
                   search="irreducible-pauli-dfs/1", trace="poly64-state-rules/1")
    return validate_v1_header(adapted, max_permutation_work=max_permutation_work)


def prepare_certificate(path: Path, *, max_header=DEFAULT_MAX_HEADER,
                        max_permutation_work=DEFAULT_MAX_PERMUTATION_WORK):
    return open_certificate(path, MAGIC, validate_header, max_header, max_permutation_work)


def write_native_model(path, header, instance):
    write_v1_model(path, header, instance)
    # The bridge is private and bounded by validated instance dimensions.
    lines = path.read_text(encoding="ascii").splitlines(keepends=True)
    lines[0] = lines[0].rstrip("\n") + " " + header["rules"] + "\n"
    path.write_text("".join(lines), encoding="ascii")
