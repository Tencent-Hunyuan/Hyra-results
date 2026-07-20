"""Focused PARP1-inhibitor library generator.

Centered on the olaparib-family phthalazinone scaffold that scores strongly
(a representative strong hit at objective ~-10.24: acetyl-piperazine benzamide
linked via CH2 to an unsubstituted phthalazinone). We enumerate a broad,
drug-like combinatorial grid around that scaffold plus a piperidine/pyrrolidine
variant, seed with a curated set of strong reference molecules as anchors,
canonical-dedupe, and QED-gate (only molecules that could plausibly reach
objective < -10 are worth docking).

Docking is a fast PROXY for binding; strong scores mean "strong on the Vina
docking metric for this fixed PARP1 pocket," nothing more.
"""
from rdkit import Chem
from rdkit.Chem import QED, Descriptors


# --- reference anchor molecules (strong known binders that seed the search) ---
ANCHORS = [
    # phthalazinone-family reference set (acetyl/alkyl-piperazine benzamides)
    "CC(=O)N1CCN(C(=O)c2ccc(Cc3n[nH]c(=O)c4ccccc34)cc2)CC1",
    "CC(=O)N1CCN(C(=O)c2ccc(Cc3n[nH]c(=O)c4cc(F)ccc34)cc2F)CC1",
    "CN(C)C(=O)N1CCN(C(=O)c2ccc(Cc3n[nH]c(=O)c4cc(F)ccc34)cc2F)CC1",
    "CN1CCN(C(=O)c2ccc(Cc3n[nH]c(=O)c4cc(F)ccc34)cc2F)CC1",
    "O=C(CF)N1CCN(C(=O)c2ccc(Cc3n[nH]c(=O)c4cc(F)ccc34)cc2F)CC1",
    "CC(=O)N1CCN(C(=O)c2ccc(Cc3n[nH]c(=O)c4ccccc34)cc2F)CC1",
    "O=C(c1ccc(Cc2n[nH]c(=O)c3ccccc23)cc1)N1CCN(C(=O)C2CC2)CC1",
    "CN1CCN(C(=O)c2ccc(Cc3n[nH]c(=O)c4ccccc34)cc2F)CC1",
    "CC(=O)C1CCN(C(=O)c2ccc(Cc3n[nH]c(=O)c4ccccc34)cc2F)CC1",
    "CCN1CCN(C(=O)c2ccc(Cc3n[nH]c(=O)c4cc(F)ccc34)cc2F)CC1",
    "O=C(c1ccc(Cc2n[nH]c(=O)c3ccccc23)cc1)N1CCN(C(=O)C(F)(F)F)CC1",
    "CCC(=O)N1CCN(C(=O)c2ccc(Cc3n[nH]c(=O)c4cc(F)ccc34)cc2F)CC1",
    "CC(=O)N1CCN(C(=O)c2ccc(Cc3n[nH]c(=O)c4cc(C)ccc34)cc2F)CC1",
    "CCC(=O)N1CCN(C(=O)c2ccc(Cc3n[nH]c(=O)c4ccccc34)cc2)CC1",
    "CC(C)C(=O)N1CCN(C(=O)c2ccc(Cc3n[nH]c(=O)c4cc(F)ccc34)cc2F)CC1",
    "CN1CCN(C(=O)c2ccc(Cc3n[nH]c(=O)c4ccccc34)cc2)CC1",
    "COC(=O)N1CCN(C(=O)c2ccc(Cc3n[nH]c(=O)c4ccccc34)cc2)CC1",
    "CCN1CCN(C(=O)c2ccc(Cc3n[nH]c(=O)c4ccccc34)cc2)CC1",
    "O=C(CF)N1CCN(C(=O)c2ccc(Cc3n[nH]c(=O)c4ccccc34)cc2F)CC1",
    "CCN1CCN(C(=O)c2ccc(Cc3n[nH]c(=O)c4ccccc34)cc2F)CC1",
    "O=C(c1ccc(Cc2n[nH]c(=O)c3ccccc23)cc1F)N1CCC(O)CC1",
    "CC(C)C(=O)N1CCN(C(=O)c2ccc(Cc3n[nH]c(=O)c4ccccc34)cc2)CC1",
    # additional fluoro-substituted phthalazinone anchors
    "O=C(c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1)N1CCN(C(=O)C2CC2)CC1",
    "CN1CCN(C(=O)c2ccc(F)c(Cc3n[nH]c(=O)c4ccccc34)c2)CC1",
    "O=C(c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1)N1CCC(O)CC1",
    "O=C(c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1)N1CCNCC1",
    # marketed PARP1 inhibitors
    "O=C(c1ccc(CC2=NNC(=O)c3ccccc32)cc1F)N1CCN(C(=O)C2CC2)CC1",  # olaparib
    "NC(=O)c1cccc2cn(-c3ccc(C4CCCNC4)cc3)nc12",                  # niraparib
    "NC(=O)c1cccc2[nH]c(C3(C)CCCN3)nc12",                        # veliparib
]


# ------- combinatorial building blocks (phthalazinone benzamide) -------
# N-cap on the distal piperazine nitrogen (solvent-exposed).
CAPS = [
    "",              # free NH piperazine
    "C",             # N-methyl
    "CC",            # N-ethyl
    "CC(=O)",        # acetyl  (best QED among strong binders)
    "CCC(=O)",       # propionyl
    "CC(C)C(=O)",    # isobutyryl
    "C(=O)C1CC1",    # cyclopropanecarbonyl (olaparib cap)
    "COC(=O)",       # methoxycarbonyl
    "CCOC(=O)",      # ethoxycarbonyl
    "CN(C)C(=O)",    # dimethylcarbamoyl
    "CS(=O)(=O)",    # methanesulfonyl
    "FCC(=O)",       # fluoroacetyl
    "OCC(=O)",       # hydroxyacetyl / glycolyl
    "NC(=O)",        # carbamoyl (urea)
]

# 4 middle carbons of the fused benzo ring of the phthalazinone.
BENZO = [
    "cccc",          # unsubstituted
    "c(F)ccc",
    "cc(F)cc",
    "ccc(F)c",
    "cc(Cl)cc",
    "ccc(Cl)c",
    "cc(C)cc",
    "cc(OC)cc",
    "cc(C#N)cc",
    "c(F)cc(F)c",
    "cc(F)c(F)c",
    "cc(F)cc",       # dup guard (dedup handles)
]

# substituent on the central benzamide ring (ortho to the carbonyl, olaparib-like)
XSUB = ["", "F", "Cl", "C"]

LINKERS = ["C", ""]     # CH2 methylene bridge, or direct biaryl


def _piperazine_scaffolds():
    for cap in CAPS:
        for linker in LINKERS:
            for benzo in BENZO:
                for x in XSUB:
                    yield (f"{cap}N1CCN(C(=O)c2ccc({linker}"
                           f"c3n[nH]c(=O)c4{benzo}c34)cc2{x})CC1")


# ---- piperidine / pyrrolidine / morpholine amide variants (raise QED) ----
RINGS = [
    "N1CCCCC1",          # piperidine
    "N1CCC(O)CC1",       # 4-hydroxypiperidine
    "N1CCC(F)CC1",       # 4-fluoropiperidine
    "N1CCC(N)CC1",       # 4-aminopiperidine
    "N1CCC(C(N)=O)CC1",  # piperidine-4-carboxamide
    "N1CCOCC1",          # morpholine
    "N1CCC(O)C1",        # 3-hydroxypyrrolidine
    "N1CCCC1",           # pyrrolidine
    "N1CC(F)C1",         # 3-fluoroazetidine
]


def _amide_scaffolds():
    small_benzo = ["cccc", "c(F)ccc", "cc(F)cc", "ccc(F)c"]
    for ring in RINGS:
        for linker in LINKERS:
            for benzo in small_benzo:
                for x in ["", "F"]:
                    yield (f"O=C(c2ccc({linker}c3n[nH]c(=O)c4{benzo}c34)"
                           f"cc2{x}){ring}")


def build_library(qed_min=0.58, mw_max=500.0):
    """Return (ordered_smiles, meta) - anchors first, then QED-gated grid by QED desc."""
    seen = set()
    anchors_out = []
    for smi in ANCHORS:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        can = Chem.MolToSmiles(m)
        if can in seen:
            continue
        seen.add(can)
        anchors_out.append(can)

    scored = []  # (qed, canonical_smiles)
    for smi in list(_piperazine_scaffolds()) + list(_amide_scaffolds()):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        can = Chem.MolToSmiles(m)
        if can in seen:
            continue
        try:
            q = float(QED.qed(m))
            mw = float(Descriptors.MolWt(m))
        except Exception:
            continue
        if q < qed_min or mw > mw_max:
            continue
        seen.add(can)
        scored.append((q, can))

    scored.sort(key=lambda t: -t[0])  # highest QED first
    ordered = anchors_out + [s for _, s in scored]
    return ordered


if __name__ == "__main__":
    lib = build_library()
    print("library size:", len(lib))
    for s in lib[:20]:
        print(s)
