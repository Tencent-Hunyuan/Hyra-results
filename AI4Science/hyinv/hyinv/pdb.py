"""PDB parsing with numpy and the standard library only: keeps the four backbone atoms N, CA, C, O per residue.

Conventions: only ATOM records of the first model are read (MSE is treated as MET); every residue takes one
slot, and a residue with missing atoms keeps its slot (it is not part of the graph but is still designed);
residue indices within a chain are renumbered 0..n-1 (the PDB numbering is not used and gaps are not filled).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

THREE = dict(ALA="A", CYS="C", ASP="D", GLU="E", PHE="F", GLY="G", HIS="H", ILE="I", LYS="K", LEU="L",
             MET="M", ASN="N", PRO="P", GLN="Q", ARG="R", SER="S", THR="T", VAL="V", TRP="W", TYR="Y", MSE="M")
BACKBONE = ("N", "CA", "C", "O")


@dataclass
class Structure:
    name: str
    coords: np.ndarray        # [L,4,3] float32
    atom_mask: np.ndarray     # [L,4] bool
    chain: np.ndarray         # [L] int64, numbered in order of appearance
    residue_index: np.ndarray  # [L] int64, 0..n-1 within each chain
    native: str               # native sequence (X for unknown residues)
    labels: list              # (chain name, PDB residue number) for each position
    chain_names: list

    def __len__(self):
        return len(self.native)


def read_pdb(path, chains: str | None = None) -> Structure:
    """chains: e.g. "A" or "AB"; all chains are read by default."""
    keep = set(chains) if chains else None
    residues, current = [], None
    for line in Path(path).read_text().splitlines():
        record = line[:6]
        if record.startswith("ENDMDL"):
            break
        if record not in ("ATOM  ", "HETATM") or len(line) < 54:
            continue
        name, resname, chain = line[12:16].strip(), line[17:20].strip(), line[21]
        if (record == "HETATM" and resname != "MSE") or (keep and chain not in keep):
            continue
        key = (chain, int(line[22:26]), line[26], resname)
        if current is None or current["key"] != key:
            current = dict(key=key, altloc=None, atoms={})
            residues.append(current)
        alt = line[16]
        if alt != " ":
            current["altloc"] = current["altloc"] or alt
            if alt != current["altloc"]:
                continue
        if name in BACKBONE and name not in current["atoms"]:
            current["atoms"][name] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    if not residues:
        raise ValueError(f"{path}: no amino-acid residues found (chains={chains!r})")
    n = len(residues)
    coords, mask = np.zeros((n, 4, 3), np.float32), np.zeros((n, 4), bool)
    chain_names = list(dict.fromkeys(r["key"][0] for r in residues))
    chain, index, seen = np.zeros(n, np.int64), np.zeros(n, np.int64), {}
    for i, r in enumerate(residues):
        for a, atom in enumerate(BACKBONE):
            if atom in r["atoms"]:
                coords[i, a], mask[i, a] = r["atoms"][atom], True
        chain[i] = chain_names.index(r["key"][0])
        index[i] = seen[chain[i]] = seen.get(chain[i], -1) + 1
    return Structure(Path(path).stem, coords, mask, chain, index,
                     "".join(THREE.get(r["key"][3], "X") for r in residues),
                     [(r["key"][0], r["key"][1]) for r in residues], chain_names)


def parse_fixed(spec: str | None, st: Structure) -> np.ndarray:
    """Fixed-position spec -> bool [L]. Comma-separated items `12` or `12-20`, prefixed with a chain name for multi-chain input (`A12-20`); numbering follows the PDB."""
    fixed = np.zeros(len(st), bool)
    for token in filter(None, (t.strip() for t in (spec or "").split(","))):
        m = re.fullmatch(r"([A-Za-z])?(-?\d+)(?:-(-?\d+))?", token)
        if m is None:
            raise ValueError(f"cannot parse fixed position {token!r}")
        chain, lo, hi = m.group(1), int(m.group(2)), int(m.group(3) or m.group(2))
        if chain is None and len(st.chain_names) > 1:
            raise ValueError(f"with multiple chains, prefix fixed positions with a chain name, e.g. A{token}")
        hit = np.array([(chain is None or c == chain) and lo <= num <= hi for c, num in st.labels])
        if not hit.any():
            raise ValueError(f"fixed position {token!r} matched no residue")
        fixed |= hit
    if (np.array(list(st.native)) == "X")[fixed].any():
        raise ValueError("fixed positions include non-standard residues and cannot be kept")
    return fixed
