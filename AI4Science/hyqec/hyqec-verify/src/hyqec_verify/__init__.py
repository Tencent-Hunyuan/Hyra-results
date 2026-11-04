"""hyqec-verify: certified distances of quantum stabilizer codes."""
from .core import CodeError, GF2Basis, StabilizerCode, weight
from .io import LoadedCode, load, load_document, parse_pauli, parse_witness, format_pauli

__version__ = "0.3.0"

from .distance import DistanceResult, distance  # noqa: E402
from .certificate import build_certificate, verify_certificate, write_certificate  # noqa: E402

__all__ = [
    "CodeError", "GF2Basis", "StabilizerCode", "weight",
    "LoadedCode", "load", "load_document", "parse_pauli", "parse_witness", "format_pauli",
    "DistanceResult", "distance", "build_certificate", "write_certificate", "verify_certificate",
]
