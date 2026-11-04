"""Strict coverage-container and independent instance/algebra validation.

Written from the review draft BNB_CERT_SPEC.md, sections 1--3 and 10--11.
Proof records stream through a bounded pipe to the native engine.
"""

from hyqec_check.transport import open_certificate

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

from .algebra import Basis, Code, ValidationError, integer, pauli_string, symplectic
from .input import strict_json

MAGIC = b"SBCOV1\r\n"
HEX256 = re.compile(r"[0-9a-f]{64}\Z")
ASCII_SPACE = " \t\n\r\v\f"
DEFAULT_MAX_HEADER = 128 * 1024 * 1024
DEFAULT_MAX_PERMUTATION_WORK = 10_000_000


@dataclass
class CertificateInstance:
    code: Code
    logicals: tuple[int, ...]
    orbit_indices: tuple[int, ...]
    representatives: tuple[int, ...]
    domain: str
    instance_sha256: str
    group_sha256: str


def group_hash(code: Code) -> str:
    # Separate highest-pivot elimination, as the identity convention requires.
    pivots: dict[int, int] = {}
    for value in code.generators:
        for p in sorted(pivots, reverse=True):
            if value & (1 << p):
                value ^= pivots[p]
        if not value:
            continue
        p = value.bit_length() - 1
        for q, row in pivots.items():
            if row & (1 << p):
                pivots[q] = row ^ value
        pivots[p] = value
    digest = hashlib.sha256(f"stabcert-group/1 n={code.n} rank={len(pivots)}\n".encode("ascii"))
    for p in sorted(pivots):
        digest.update(pivots[p].to_bytes((2 * code.n + 7) // 8, "little"))
    return digest.hexdigest()


def parse_instance(text: str) -> CertificateInstance:
    if not isinstance(text, str) or not text.isascii() or "\0" in text:
        raise ValidationError("instance_text must be NUL-free ASCII")
    # Only the six ASCII whitespace characters delimit tokens.
    tokens = re.split(r"[ \t\n\r\v\f]+", text.strip(ASCII_SPACE))
    position = 0

    def token():
        nonlocal position
        if position == len(tokens):
            raise ValidationError("truncated instance")
        value = tokens[position]
        position += 1
        return value

    def decimal(name, maximum):
        value = token()
        if not value or any(c < "0" or c > "9" for c in value):
            raise ValidationError(f"{name}: expected unsigned decimal token")
        stripped = value.lstrip("0") or "0"
        if len(stripped) > len(str(maximum)) or int(stripped) > maximum:
            raise ValidationError(f"{name} exceeds {maximum}")
        return int(stripped)

    versioned = tokens[0] == "stabcert-bnb-instance"
    if versioned:
        token()
        if decimal("instance version", 1) != 1:
            raise ValidationError("unsupported instance version")
    n = decimal("n", 4096)
    m = decimal("m", 4096)
    nl = decimal("nl", 8192)
    if n == 0:
        raise ValidationError("n must be positive")
    degree = [0] * n

    def row(is_generator):
        length = decimal("row weight", min(n, 32) if is_generator else n)
        previous = -1
        value = 0
        for _ in range(length):
            q = decimal("qubit", n - 1)
            label = decimal("Pauli label", 3)
            if q <= previous or label == 0:
                raise ValidationError("row requires strictly increasing qubits and nonzero labels")
            previous = q
            value |= (label & 1) << q
            value |= ((label >> 1) & 1) << (n + q)
            if is_generator:
                degree[q] += 1
                if degree[q] > 64:
                    raise ValidationError("generator column degree exceeds 64")
        return value

    generators = tuple(row(True) for _ in range(m))
    logicals = tuple(row(False) for _ in range(nl))
    orbit_indices = tuple(decimal("orbit index", n - 1) for _ in range(n))
    no = decimal("number of orbits", n)
    if no == 0 or set(orbit_indices) != set(range(no)):
        raise ValidationError("orbit indices must be contiguous and nonempty")
    representatives = tuple(decimal("orbit representative", n - 1) for _ in range(no))
    if any(orbit_indices[q] != i for i, q in enumerate(representatives)):
        raise ValidationError("representative does not belong to corresponding orbit")
    domain = "P"
    if versioned:
        if token() != "domain":
            raise ValidationError("missing domain keyword")
        domain = token()
        if domain not in ("P", "X", "Z"):
            raise ValidationError("unsupported instance domain")
    if position != len(tokens):
        raise ValidationError("unexpected trailing instance tokens")
    code = Code.from_strings(n, [pauli_string(v, n) for v in generators])
    if code.k <= 0:
        raise ValidationError("coverage version 1 requires k > 0")
    if len(logicals) != 2 * code.k:
        raise ValidationError("supplied logical basis must have exactly 2k rows")
    if any(symplectic(l, g, n) for l in logicals for g in code.stabilizers.rows):
        raise ValidationError("supplied logical row does not commute with S")
    if len(Basis(code.stabilizers.rows + logicals)) != n + code.k:
        raise ValidationError("supplied logical rows do not span N(S)/S")
    return CertificateInstance(code, logicals, orbit_indices, representatives, domain,
                               hashlib.sha256(text.encode("ascii")).hexdigest(), group_hash(code))


def validate_header(header: object, *, max_permutation_work=DEFAULT_MAX_PERMUTATION_WORK) -> CertificateInstance:
    if not isinstance(header, dict):
        raise ValidationError("coverage header must be a JSON object")
    for name, value in {"format": "stabcert-bnb-coverage/1", "search": "irreducible-pauli-dfs/1",
                        "trace": "poly64-state-rules/1", "amax": 96}.items():
        if header.get(name) != value or (name == "amax" and type(header[name]) is not int):
            raise ValidationError(f"unsupported {name}")
    if header.get("detail") not in ("frontier", "full"):
        raise ValidationError("unsupported certificate detail")
    required = {"instance_text", "instance_sha256", "group_sha256", "n", "m", "nl", "k",
                "cap", "domain", "split", "permutations", "provenance"}
    if required - header.keys():
        raise ValidationError(f"missing header fields: {sorted(required - header.keys())}")
    for key in ("instance_sha256", "group_sha256"):
        if not isinstance(header[key], str) or not HEX256.fullmatch(header[key]):
            raise ValidationError(f"invalid {key}")
    provenance = header["provenance"]
    if not isinstance(provenance, dict):
        raise ValidationError("provenance must be an object")
    for key in ("source_sha256", "binary_sha256"):
        if not isinstance(provenance.get(key), str) or not HEX256.fullmatch(provenance[key]):
            raise ValidationError(f"invalid provenance {key}")
    if integer(header["cap"], "cap") > 46 or integer(header["split"], "split", 1) > 47:
        raise ValidationError("cap/split exceeds version-1 limit")
    # Cheap budget check before parsing/algebra or walking any permutations.
    # Header dimensions are untrusted here; exact instance equality is checked
    # below, before permutation validation, so underreporting cannot bypass it.
    n = integer(header["n"], "n", 1)
    m = integer(header["m"], "m")
    if n > 4096 or m > 4096:
        raise ValidationError("header dimensions exceed supported limits")
    permutations = header["permutations"]
    if not isinstance(permutations, list):
        raise ValidationError("permutations must be a list")
    integer(max_permutation_work, "max_permutation_work")
    if len(permutations) * n * max(1, m) > max_permutation_work:
        raise ValidationError("permutation validation exceeds configured work limit")
    instance = parse_instance(header["instance_text"])
    for key, value in {"n": instance.code.n, "m": len(instance.code.generators),
                       "nl": len(instance.logicals), "k": instance.code.k}.items():
        if integer(header[key], key) != value:
            raise ValidationError(f"header {key} disagrees with instance/algebra")
    for key, value in {"instance_sha256": instance.instance_sha256,
                       "group_sha256": instance.group_sha256, "domain": instance.domain}.items():
        if header[key] != value:
            raise ValidationError(f"header {key} mismatch")
    orbits = [[q for q, oi in enumerate(instance.orbit_indices) if oi == i]
              for i in range(len(instance.representatives))]
    instance.code.validate_orbit_order(header["permutations"], orbits, instance.representatives)
    return instance


def prepare_certificate(path: Path, *, max_header=DEFAULT_MAX_HEADER,
                        max_permutation_work=DEFAULT_MAX_PERMUTATION_WORK):
    """Context manager yielding validated header, instance and open records."""
    return open_certificate(path, MAGIC, validate_header, max_header, max_permutation_work)


def write_native_model(path: Path, header: dict, instance: CertificateInstance) -> None:
    code = instance.code
    with path.open("w", encoding="ascii") as output:
        output.write(f"{code.n} {len(code.generators)} {len(code.stabilizers)} {header['cap']} "
                     f"{header['split']} {instance.domain} {header['detail']}\n")
        for row in code.generators:
            support = [(q, ((row >> q) & 1) | (((row >> (code.n + q)) & 1) << 1)) for q in range(code.n)]
            support = [(q, p) for q, p in support if p]
            output.write(str(len(support)) + "".join(f" {q} {p}" for q, p in support) + "\n")
        words = (2 * code.n + 63) // 64
        for pivot, row in sorted(code.stabilizers.pivots.items()):
            output.write(str(pivot) + "".join(f" {(row >> (64 * i)) & ((1 << 64) - 1)}" for i in range(words)) + "\n")
        output.write(" ".join(map(str, instance.orbit_indices)) + "\n")
        output.write(str(len(instance.representatives)) + " " + " ".join(map(str, instance.representatives)) + "\n")
