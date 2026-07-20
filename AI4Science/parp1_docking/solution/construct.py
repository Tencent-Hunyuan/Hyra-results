"""Emit curated PARP1 candidate SMILES for the DOCKSTRING joint objective.

Strategy: objective = docking + 10*(1-QED), scored on top-3 mean (lower better).
Scoring re-docks with its own Vina against the fixed PARP1 pocket, so we
submit a diverse portfolio and let it pick our best 3.

Core idea: keep olaparib's *binding warhead* - 4-(4-fluorobenzyl)phthalazin-
1(2H)-one + meta-benzamide - which anchors the nicotinamide-mimetic H-bonds
(phthalazinone NH/C=O to Gly863/Ser904) and the Tyr stacking that make olaparib
dock ~-12.7 here. Vary ONLY the solvent-exposed amide tail to lift QED from
olaparib's 0.68 to 0.74-0.80 without sacrificing the docking anchor. Add real
clinical inhibitors and a few alternate nicotinamide-mimetic scaffolds as
diversity/anchors. No local docking (the DOCKSTRING receptor pdbqt isn't shipped
here); selection is by chemical knowledge + local QED.
"""
import json
from rdkit import Chem
from rdkit.Chem import QED, Descriptors

CANDIDATES = [
    # --- olaparib itself (reliable strong-docking anchor) ---
    "O=C(c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1)N1CCN(C(=O)C2CC2)CC1",  # olaparib
    # --- olaparib warhead, higher-QED / drug-like tails (near-olaparib size => near-olaparib docking) ---
    "CN1CCN(C(=O)c2ccc(F)c(Cc3n[nH]c(=O)c4ccccc34)c2)CC1",           # N-Me piperazine
    "O=C(c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1)N1CCNCC1",              # piperazine
    "O=C(c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1)N1CCOCC1",             # morpholine amide
    "O=C(c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1)N1CCC(F)CC1",          # 4-F-piperidine
    "O=C(c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1)N1CCCC1",              # pyrrolidine
    "O=C(c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1)N1CCC1",               # azetidine
    "CN(C)C(=O)c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1",                # N,N-dimethyl
    "O=C(c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1)N1CC(F)C1",            # 3-F-azetidine
    "CNC(=O)c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1",                   # N-Me
    "O=C(NC1CC1)c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1",              # cyclopropylamide
    "O=C(c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1)N1CCC(O)CC1",         # 4-OH-piperidine
    "O=C(c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1)N1CCC(O)C1",          # 3-OH-pyrrolidine
    "O=C(c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1)N1CC(O)C1",           # 3-OH-azetidine
    "O=C(NC1COC1)c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1",            # oxetan-3-yl amide
    "O=C(c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1)N1CCN(C)CC1",        # dup guard variant
    # --- real clinical PARP1 inhibitors (reliable strong dockers) ---
    "NC(=O)c1cccc2cn(-c3ccc(C4CCCNC4)cc3)nc12",                    # niraparib
    "NC(=O)c1cccc2[nH]c(nc12)C1(C)CCCN1",                          # veliparib
    "CN1N=C(c2ccc(F)cc2)C2NC(=O)c3ccc(F)cc3C21",                   # talazoparib-like analog
    # --- alternate nicotinamide-mimetic scaffolds (diversity) ---
    "O=c1[nH]nc(Cc2ccc(F)cc2)c2ccccc12",                           # 4-(4-F-benzyl)phthalazinone
    "O=c1[nH]c(CC2CCNCC2)nc2ccccc12",                              # quinazolin-4-one
    "NC(=O)c1cccc2[nH]c(C3CCNCC3)nc12",                            # benzimidazole-4-carboxamide
]

seen, out = set(), []
for smi in CANDIDATES:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        continue
    can = Chem.MolToSmiles(m)
    if can in seen:
        continue
    seen.add(can)
    q = QED.qed(m)
    out.append((smi, q, Descriptors.MolWt(m)))

# keep at most 25 (only the first 25 are scored); order high-QED-first among warhead-preserving
out_sorted = out  # authored order already prioritizes strong+drug-like candidates
mols = [s for s, q, mw in out_sorted][:25]

for s, q, mw in out_sorted[:25]:
    print(f"QED={q:.3f} MW={mw:6.1f}  {s}")
print(f"n_molecules={len(mols)}")

with open("solution.json", "w") as f:
    json.dump({"molecules": mols, "target": "PARP1"}, f, indent=2)
print("wrote solution.json")
