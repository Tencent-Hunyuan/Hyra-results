"""Reading codes and witnesses.

Supported code inputs (``load``):

* qldpc-challenge JSON: ``code_type: "stabilizer"`` with ``checks.S`` =
  ``[{"X": [...], "Z": [...]}, ...]``, or ``code_type: "CSS"`` with
  ``checks.X`` / ``checks.Z`` = lists of supports. Optional ``k`` and
  ``distance`` (``d``, ``P.witness`` or ``X/Z.witness``) are read as *claims*.
* generator JSON (format id ``stabcert-json``): ``{"n": n, "generators": [{"X": [...], "Z": [...]}, ...]}``
  or ``"generators": ["XZZXI", ...]``; certificates embed this form.
* ``.npz``: ``S`` (m x 2n, rows (x|z)), or ``A`` and ``B`` (S = (A|B)), or ``HX`` and ``HZ``.
* text: one generator per line, dense (``XZZXI``; ``I``, ``_`` or ``.`` for identity)
  or sparse (``X0 Z3 Y7`` / stim-like ``X0*Z3*Y7``); ``#`` starts a comment.
  For sparse text the qubit count is the largest index + 1 unless ``n=`` is given.

Claims are never trusted: ``load`` returns them separately and callers check
them against the computed values.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core import CodeError, StabilizerCode, pauli_from_xz, xz_from_pauli

__all__ = ["LoadedCode", "load", "load_document", "parse_pauli", "parse_witness", "format_pauli"]


@dataclass
class LoadedCode:
    code: StabilizerCode
    source: str | None = None
    fmt: str = ""
    claimed_k: int | None = None
    claimed_rank: int | None = None
    claimed_d: int | None = None
    # claimed witnesses: kind ("P", "X", "Z") -> Pauli int (not yet verified)
    claimed_witnesses: dict = field(default_factory=dict)
    document: Any = None

    def check_claims(self) -> list[str]:
        """Human-readable problems with the claims (empty list = consistent)."""
        out = []
        c = self.code
        if self.claimed_k is not None and self.claimed_k != c.k:
            out.append(f"claimed k={self.claimed_k}, computed k={c.k}")
        if self.claimed_rank is not None and self.claimed_rank != c.rank:
            out.append(f"claimed rank={self.claimed_rank}, computed rank={c.rank}")
        for kind, v in self.claimed_witnesses.items():
            if not c.is_logical(v):
                out.append(f"claimed {kind} witness is not a nontrivial logical")
        return out


_LABEL = {"I": 0, "_": 0, ".": 0, "X": 1, "Z": 2, "Y": 3}
_SPARSE_TOKEN = re.compile(r"^([XYZ])(\d+)$")


def parse_pauli(text: str, n: int | None = None) -> tuple[int, int]:
    """Parse a dense ("XIZY") or sparse ("X0 Z2 Y3", "X0*Z2") Pauli string.

    Returns (pauli_int, n). For a dense string n is its length (must equal n if given);
    for a sparse string n must be given or is taken as the largest index + 1."""
    s = text.strip()
    if s.startswith(("+", "-")):          # stim-style sign; phases are ignored
        s = s[1:].strip()
    if not s:
        raise CodeError("empty Pauli string")
    if all(ch in _LABEL for ch in s):
        if n is not None and len(s) != n:
            raise CodeError(f"dense Pauli string of length {len(s)}, expected {n}")
        nn = len(s)
        x = [q for q, ch in enumerate(s) if _LABEL[ch] & 1]
        z = [q for q, ch in enumerate(s) if _LABEL[ch] & 2]
        return pauli_from_xz(x, z, nn), nn
    toks = [t for t in re.split(r"[\s*]+", s) if t]
    pairs = []
    for t in toks:
        m = _SPARSE_TOKEN.match(t)
        if not m:
            raise CodeError(f"cannot parse Pauli token {t!r}")
        pairs.append((m.group(1), int(m.group(2))))
    qs = [q for _, q in pairs]
    if len(set(qs)) != len(qs):
        raise CodeError(f"repeated qubit in sparse Pauli {text!r}")
    nn = n if n is not None else (max(qs) + 1)
    x = [q for l, q in pairs if l in "XY"]
    z = [q for l, q in pairs if l in "ZY"]
    return pauli_from_xz(x, z, nn), nn


def format_pauli(v: int, n: int) -> str:
    return "".join("IXZY"[((v >> q) & 1) | (((v >> (n + q)) & 1) << 1)] for q in range(n))


def parse_witness(obj: Any, n: int) -> int:
    """A witness as ``{"X": [...], "Z": [...]}``, a Pauli string, or a ``{"witness": ...}`` wrapper."""
    if isinstance(obj, dict) and "witness" in obj and set(obj) - {"X", "Z"}:
        obj = obj["witness"]
    if isinstance(obj, dict):
        if set(obj) - {"X", "Z"}:
            raise CodeError("witness: expected keys X and Z")
        return pauli_from_xz(obj.get("X", []), obj.get("Z", []), n, "witness")
    if isinstance(obj, str):
        return parse_pauli(obj, n)[0]
    raise CodeError("witness: expected {'X': [...], 'Z': [...]} or a Pauli string")


def _int_field(doc: dict, key: str) -> int | None:
    if key not in doc or doc[key] is None:
        return None
    v = doc[key]
    if type(v) is not int:
        raise CodeError(f"field {key!r} must be an int, got {v!r}")
    return v


def _from_challenge(doc: dict) -> LoadedCode:
    n = _int_field(doc, "n")
    if n is None:
        raise CodeError("missing n")
    checks = doc.get("checks")
    if not isinstance(checks, dict):
        raise CodeError("missing checks")
    ctype = doc.get("code_type")
    if ctype == "stabilizer" or (ctype is None and "S" in checks):
        if set(checks) != {"S"}:
            raise CodeError("stabilizer code: checks must contain exactly 'S'")
        code = StabilizerCode.from_xz(checks["S"], n, name=doc.get("name"))
        fmt = "challenge-stabilizer"
    elif ctype in ("CSS", "css") or (ctype is None and {"X", "Z"} <= set(checks)):
        if set(checks) != {"X", "Z"}:
            raise CodeError("CSS code: checks must contain exactly 'X' and 'Z'")
        code = StabilizerCode.css(checks["X"], checks["Z"], n, name=doc.get("name"))
        fmt = "challenge-css"
    else:
        raise CodeError(f"unknown code_type {ctype!r}")
    lc = LoadedCode(code, fmt=fmt, claimed_k=_int_field(doc, "k"), claimed_rank=_int_field(doc, "rank"),
                    document=doc)
    dist = doc.get("distance")
    if isinstance(dist, dict):
        lc.claimed_d = _int_field(dist, "d")
        for kind in ("P", "X", "Z"):
            ent = dist.get(kind)
            if not isinstance(ent, dict) or "witness" not in ent:
                continue
            w = ent["witness"]
            if kind == "P" or isinstance(w, dict) or isinstance(w, str):
                lc.claimed_witnesses[kind] = parse_witness(w, n)
            else:   # CSS side witness: a plain support list of an X-type (X) or Z-type (Z) operator
                lc.claimed_witnesses[kind] = (pauli_from_xz(w, [], n, "X witness") if kind == "X"
                                              else pauli_from_xz([], w, n, "Z witness"))
    return lc


def _from_generator_json(doc: dict) -> LoadedCode:
    gens = doc["generators"]
    if "checks" in doc:
        raise CodeError("give either 'generators' or 'checks', not both")
    if not isinstance(gens, list):
        raise CodeError("generators: expected a list")
    if doc.get("code_type") not in (None, "stabilizer", "CSS", "css"):
        raise CodeError(f"unknown code_type {doc.get('code_type')!r}")
    n = _int_field(doc, "n")
    if gens and all(isinstance(g, str) for g in gens):
        parsed = [parse_pauli(g, n) for g in gens]
        if n is None:
            ns = {nn for _, nn in parsed}
            if len(ns) != 1:
                raise CodeError("generators of different lengths; give n")
            n = ns.pop()
        code = StabilizerCode(n, [v for v, _ in parsed], name=doc.get("name"))
    else:
        if n is None:
            raise CodeError("missing n")
        code = StabilizerCode.from_xz(gens, n, name=doc.get("name"))
    lc = LoadedCode(code, fmt="stabcert-json", claimed_k=_int_field(doc, "k"),
                    claimed_rank=_int_field(doc, "rank"), document=doc)
    if "witness" in doc:
        lc.claimed_witnesses["P"] = parse_witness(doc["witness"], code.n)
    return lc


def load_document(doc: Any) -> LoadedCode:
    if not isinstance(doc, dict):
        raise CodeError("code document must be a JSON object")
    if "generators" in doc:
        return _from_generator_json(doc)
    if "code" in doc and isinstance(doc["code"], dict) and "generators" in doc["code"]:
        return _from_generator_json(doc["code"])          # a certificate
    return _from_challenge(doc)


def _from_npz(path: Path) -> LoadedCode:
    import numpy as np

    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)

        def mat(key):
            a = np.asarray(z[key])
            if a.ndim != 2 or not np.isin(a, (0, 1)).all():
                raise CodeError(f"{key}: expected a 2-D 0/1 matrix")
            return a.astype(np.uint8)

        if "S" in keys:
            S = mat("S")
            if S.shape[1] % 2:
                raise CodeError("S must have 2n columns")
            n = S.shape[1] // 2
            A, B = S[:, :n], S[:, n:]
        elif {"A", "B"} <= keys:
            A, B = mat("A"), mat("B")
            if A.shape != B.shape:
                raise CodeError("A and B must have the same shape")
            n = A.shape[1]
        elif {"HX", "HZ"} <= keys:
            HX, HZ = mat("HX"), mat("HZ")
            if HX.shape[1] != HZ.shape[1]:
                raise CodeError("HX and HZ must have the same number of columns")
            n = HX.shape[1]
            hx = [[int(q) for q in np.flatnonzero(r)] for r in HX]
            hz = [[int(q) for q in np.flatnonzero(r)] for r in HZ]
            return LoadedCode(StabilizerCode.css(hx, hz, n), fmt="npz-css")
        else:
            raise CodeError(f"npz needs S, or A and B, or HX and HZ (found {sorted(keys)})")
    gens = [pauli_from_xz([int(q) for q in np.flatnonzero(a)], [int(q) for q in np.flatnonzero(b)], int(n))
            for a, b in zip(A, B)]
    return LoadedCode(StabilizerCode(int(n), gens), fmt="npz")


def _from_text(text: str, n: int | None = None) -> LoadedCode:
    lines = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"^n\s*=\s*(\d+)$", line)
        if m:
            n = int(m.group(1))
            continue
        lines.append(line)
    if not lines:
        raise CodeError("no generators in text input")
    parsed = [parse_pauli(l, n) for l in lines]
    if n is None:
        ns = {nn for _, nn in parsed}
        n = max(ns)
        dense = [nn for (l, (_, nn)) in zip(lines, parsed) if all(ch in _LABEL for ch in l.lstrip("+-"))]
        if dense and len(set(dense)) != 1:
            raise CodeError("dense generators of different lengths")
        parsed = [parse_pauli(l, n) for l in lines]
    return LoadedCode(StabilizerCode(n, [v for v, _ in parsed]), fmt="text")


def load(path: str | Path, n: int | None = None) -> LoadedCode:
    """Load a code from a file (format chosen by content / suffix)."""
    p = Path(path)
    if p.suffix == ".npz":
        lc = _from_npz(p)
    else:
        text = p.read_text()
        if text.lstrip().startswith("{"):
            try:
                doc = json.loads(text)
            except json.JSONDecodeError as e:
                raise CodeError(f"invalid JSON: {e}") from None
            lc = load_document(doc)
        else:
            lc = _from_text(text, n)
    lc.source = str(p)
    return lc


def witness_dict(v: int, n: int) -> dict:
    return xz_from_pauli(v, n)
