"""PARP1 focused-library virtual screen.

Builds a focused library around proven PARP1 pharmacophores (phthalazinone /
2H-indazole-7-carboxamide / 1H-benzimidazole-4-carboxamide / quinazolinone),
then re-docks each candidate with the standard DOCKSTRING pipeline
(rdkit 3D embed seed 42 -> meeko pdbqt -> AutoDock Vina, exhaustiveness 8,
seed 42, PARP1 pocket box) and ranks by the DOCKSTRING joint objective
    objective = docking_score + 10 * (1 - QED)         (lower is better).

Receptor pdbqt is obtained from the pip 'dockstring' package (the DOCKSTRING
PARP1 target). If it cannot be located, we fall back to
submitting a curated list of potent PARP1 inhibitors + high-QED analogs
(no self-docking) so solution.json is always valid.
"""

import os, sys, glob, json, time, importlib.util
from multiprocessing import Pool, cpu_count

START = time.time()
BUDGET = float(os.environ.get("TIME_BUDGET_SEC", "7200"))
NCPU = max(1, int(os.environ.get("CPUS", str(cpu_count() or 4))))
BOX_CENTER = (26.835, 11.332, 27.744)
BOX_SIZE = (30.0, 30.0, 30.0)
SEED = 42
EXH = 8
HERE = os.path.dirname(os.path.abspath(__file__))
SOL = os.path.join(HERE, "solution.json")

from rdkit import Chem
from rdkit.Chem import QED, Descriptors, AllChem, Lipinski
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


# ----------------------------------------------------------------------------
# Library construction
# ----------------------------------------------------------------------------
CURATED = [
    # marketed / clinical PARP1 inhibitors (approximate canonical SMILES)
    "O=C(C1CC1)N1CCN(C(=O)c2cc(CC3=NNC(=O)c4ccccc43)ccc2F)CC1",  # olaparib
    "NC(=O)c1cccc2cn(-c3ccc(C4CCCNC4)cc3)nc12",                   # niraparib
    "NC(=O)c1cccc2[nH]c(C3(C)CCCN3)nc12",                          # veliparib
    "CNCc1ccc(-c2cc3c([nH]2)c2c(cccc2CCNC3=O)F)cc1",              # rucaparib-like
    "O=C1NN=C(Cc2ccccc2)c2ccccc21",                               # 4-benzylphthalazinone
    "O=c1[nH]c(-c2ccccc2)nc2ccccc12",                            # 2-phenylquinazolinone
    "NC(=O)c1cccc2cn(-c3ccc(C4CCNCC4)cc3)nc12",                   # indazole+piperidin-4-yl
    "NC(=O)c1cccc2[nH]c(C3CCCN3)nc12",                            # benzimidazole+pyrrolidine
]

# Amines forming the aryl-amide tail (phthalazinone family). Ring labels 5/6
# so they never clash with the phthalazinone template (labels 1,2,3).
AMINES = [
    "N5CCN(C(=O)C6CC6)CC5", "N5CCN(C(C)=O)CC5", "N5CCN(C)CC5", "N5CCN(CC)CC5",
    "N5CCN(S(C)(=O)=O)CC5", "N5CCOCC5", "N5CCCCC5", "N5CCC(O)CC5",
    "N5CCC(F)(F)CC5", "N5CCC(N)CC5", "N5CCCC5", "NC5CC5", "NCC5CC5",
    "N(C)C", "NC", "NC5CCOCC5", "N5CCN(CCO)CC5", "N5CCC(C(N)=O)CC5",
    "N5CCN(C(=O)C(C)C)CC5", "N5CCN(CC(F)(F)F)CC5", "N5CCC(C)(O)CC5",
    "N5CCC(F)CC5", "NCCO", "N5CCN(C6CC6)CC5", "N5CCN(c6ncccn6)CC5",
    "N5CCC(N6CCCC6)CC5", "N5CCC(C(F)(F)F)CC5",
]

# 4-(benzyl)-2H-phthalazin-1-one, benzyl aryl bears a meta amide + optional halogen.
BENZYL_TEMPLATES = [
    "Cc2cc(C(=O){A})ccc2F",   # olaparib-like (para-F)
    "Cc2cc(C(=O){A})ccc2",    # meta amide
    "Cc2ccc(C(=O){A})cc2",    # para amide
    "Cc2cc(C(=O){A})cc(F)c2",
    "Cc2cc(C(=O){A})c(F)cc2",
]

# 2-aryl-2H-indazole-7-carboxamide (niraparib family): para/meta substituent.
INDZ_S = [
    "C4CCCNC4", "C4CCNCC4", "C4CCNC4", "C4CCN(C)C4", "C4CCCN(C)C4", "C4CNC4",
    "N4CCNCC4", "N4CCOCC4", "N4CCN(C)CC4", "CN", "CNC", "CN(C)C", "C4CCOCC4",
    "F", "Cl", "C#N", "O", "N4CCCC4", "C4CCN(C)CC4", "C4CCCNC4", "OC",
]
INDZ_TEMPLATES = [
    "NC(=O)c1cccc2cn(-c3ccc({S})cc3)nc12",
    "NC(=O)c1cccc2cn(-c3cccc({S})c3)nc12",
]

# 2-substituted 1H-benzimidazole-4-carboxamide (veliparib family).
BZ_S = [
    "C3(C)CCCN3", "C3CCCN3", "C3CCNC3", "C3CCNCC3", "C3CCCNC3", "C3CCOCC3",
    "C3CCN(C)CC3", "C3CCN(C)C3", "CN", "CNC", "CC3CCCN3", "c3ccncc3",
    "c3ccc(F)cc3", "c3cccnc3", "C3CNC3", "CC(N)C", "C3(N)CCCC3", "CCN",
]
BZ_TMPL = "NC(=O)c1cccc2[nH]c({S})nc12"

QUIN = [
    "O=c1[nH]c(-c2ccc(F)cc2)nc2ccccc12",
    "O=c1[nH]c(-c2ccccc2)nc2ccccc12",
    "O=c1[nH]c(N3CCNCC3)nc2ccccc12",
    "O=c1[nH]c(-c2ccc(CN)cc2)nc2ccccc12",
    "O=c1[nH]c(-c2ccc(N3CCNCC3)cc2)nc2cccc(F)c12",
    "O=c1[nH]c(-c2ccc(F)cc2)nc2cccc(N3CCOCC3)c12",
]

# 2-aryl-1H-benzimidazole-4-carboxamide (aromatic stacking variant of veliparib core).
BZ_ARYL = [
    "c3ccccc3", "c3ccc(F)cc3", "c3ccc(Cl)cc3", "c3ccc(C)cc3", "c3ccc(OC)cc3",
    "c3ccncc3", "c3cccnc3", "c3ccc(CN)cc3", "c3ccc(CO)cc3", "c3ccc(C(N)=O)cc3",
    "c3ccc(N4CCOCC4)cc3", "c3ccc(N4CCNCC4)cc3", "c3ccc(C4CCNCC4)cc3",
    "c3ccc(F)c(F)c3", "c3ccc(C#N)cc3", "c3ccc(CNC)cc3", "c3ccc(CO)c(F)c3",
]

# 4-aryl-2H-phthalazin-1-one: planar biaryl lactam (labels 1,2,3; amine 5/6).
PHTHAL4_ARYL_TEMPLATES = [
    "O=C1NN=C(-c2ccc(C(=O){A})cc2)c3ccccc31",
    "O=C1NN=C(-c2ccc({S})cc2)c3ccccc31",
    "O=C1NN=C(-c2cccc(C(=O){A})c2)c3ccccc31",
]
PHTHAL4_S = [
    "N4CCOCC4", "N4CCNCC4", "N4CCN(C)CC4", "C4CCNCC4", "C4CCNC4", "CN", "CNC",
    "F", "C#N", "O", "N", "N4CCCC4", "OC", "C4CCOCC4", "S(C)(=O)=O",
]


# Fused-benzo ring variants for the phthalazinone (last ring "c3ccccc31").
PHTHAL_BENZO = ["c3ccccc31", "c3ccc(F)cc31", "c3cc(F)ccc31", "c3ccc(Cl)cc31",
                "c3cc(C)ccc31"]
# 5-substituent variants for the indazole/benzimidazole carboxamide core.
INDZ_CORE = ["NC(=O)c1cccc2cn({R})nc12", "NC(=O)c1ccc(F)c2cn({R})nc12",
             "NC(=O)c1cc(F)cc2cn({R})nc12"]
BZ_CORE = ["NC(=O)c1cccc2[nH]c({S})nc12", "NC(=O)c1ccc(F)c2[nH]c({S})nc12"]


def build_raw():
    raw = list(CURATED)
    # Phthalazinone 4-benzyl benzamide, with fused-benzo halogen variants.
    for bt in BENZYL_TEMPLATES:
        for a in AMINES:
            frag = bt.replace("{A}", a)
            for benzo in PHTHAL_BENZO:
                raw.append("O=C1NN=C(C" + frag + ")" + benzo)
    # 2-aryl indazole-7-carboxamide (niraparib family), core-halogen variants.
    aryl = ["-c3ccc({S})cc3", "-c3cccc({S})c3"]
    for core in INDZ_CORE:
        for ar in aryl:
            for s in INDZ_S:
                raw.append(core.replace("{R}", ar.replace("{S}", s)))
    # Benzimidazole-4-carboxamide (veliparib family), alkyl + aryl 2-substituents.
    for core in BZ_CORE:
        for s in list(BZ_S) + list(BZ_ARYL):
            raw.append(core.replace("{S}", s))
    # 4-aryl phthalazinone (planar biaryl lactam).
    for t in PHTHAL4_ARYL_TEMPLATES:
        if "{A}" in t:
            for a in AMINES:
                raw.append(t.replace("{A}", a))
        else:
            for s in PHTHAL4_S:
                raw.append(t.replace("{S}", s))
    raw.extend(QUIN)
    return raw


def build_library():
    """Return (curated_valid, enumerated_ranked_by_qed) canonical SMILES lists."""
    seen = set()
    curated_valid, enum = [], []
    raw = build_raw()
    n_cur = len(CURATED)
    for i, smi in enumerate(raw):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        can = Chem.MolToSmiles(m)
        if can in seen:
            continue
        seen.add(can)
        try:
            qed = float(QED.qed(m))
            mw = float(Descriptors.MolWt(m))
            rot = Lipinski.NumRotatableBonds(m)
        except Exception:
            continue
        is_cur = i < n_cur
        if is_cur:
            curated_valid.append((can, qed))
        else:
            if 240.0 <= mw <= 520.0 and rot <= 9 and qed >= 0.50:
                enum.append((can, qed))
    enum.sort(key=lambda x: -x[1])
    return curated_valid, enum


# ----------------------------------------------------------------------------
# Docking (shared Vina + precomputed maps per worker)
# ----------------------------------------------------------------------------
_V = None


def init_worker(receptor):
    global _V
    from vina import Vina
    _V = Vina(sf_name="vina", cpu=1, seed=SEED, verbosity=0)
    _V.set_receptor(receptor)
    _V.compute_vina_maps(center=list(BOX_CENTER), box_size=list(BOX_SIZE))


def _prep_pdbqt(molH):
    """Return ligand pdbqt string via meeko, handling API variants."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    prep = MoleculePreparation()
    setups = prep.prepare(molH)
    setup = setups[0] if isinstance(setups, (list, tuple)) else setups
    out = PDBQTWriterLegacy.write_string(setup)
    if isinstance(out, tuple):
        pdbqt, ok = out[0], out[1]
        if not ok:
            return None
        return pdbqt
    return out  # older meeko returns just the string


def dock_one(smi):
    """Standard per-molecule docking pipeline; reuse precomputed maps.

    On failure returns {'error': stage:message} so the driver can log it.
    """
    global _V
    stage = "parse"
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return {"error": "parse:None"}
        qed = float(QED.qed(mol))
        stage = "embed"
        molH = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(molH, randomSeed=SEED) != 0:
            if AllChem.EmbedMolecule(molH, randomSeed=SEED, useRandomCoords=True) != 0:
                return {"error": "embed:fail"}
        try:
            AllChem.MMFFOptimizeMolecule(molH)
        except Exception:
            pass
        stage = "meeko"
        pdbqt = _prep_pdbqt(molH)
        if not pdbqt:
            return {"error": "meeko:no_pdbqt"}
        stage = "dock"
        _V.set_ligand_from_string(pdbqt)
        _V.dock(exhaustiveness=EXH, n_poses=1)
        score = float(_V.energies(n_poses=1)[0][0])
    except Exception as e:
        return {"error": f"{stage}:{type(e).__name__}:{e}"[:200]}
    return {"smiles": smi, "docking": score, "qed": qed,
            "objective": score + 10.0 * (1.0 - qed)}


# ----------------------------------------------------------------------------
# Receptor discovery
# ----------------------------------------------------------------------------
def find_receptor():
    cands = []
    spec = importlib.util.find_spec("dockstring")
    if spec is not None and spec.origin:
        pkg = os.path.dirname(spec.origin)
        cands += glob.glob(os.path.join(pkg, "**", "*[Pp][Aa][Rr][Pp]1*.pdbqt"),
                           recursive=True)
    if not cands:
        cands += glob.glob(os.path.join(sys.prefix, "**", "*PARP1*.pdbqt"),
                           recursive=True)
    # prefer files that look like a receptor 'target'
    cands = [c for c in cands if os.path.isfile(c)]
    cands.sort(key=lambda p: (0 if "target" in p.lower() else 1, len(p)))
    return cands[0] if cands else None


# ----------------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------------
def write_solution(smiles_list):
    smiles_list = smiles_list[:25]
    with open(SOL, "w") as f:
        json.dump({"molecules": smiles_list, "target": "PARP1"}, f, indent=2)


def main():
    curated_valid, enum = build_library()
    # Fallback (no-dock) ordering: potent curated first, then best-QED analogs.
    fallback = [s for s, _ in curated_valid] + [s for s, _ in enum]
    write_solution(fallback)
    print(f"[screen] curated={len(curated_valid)} enum={len(enum)} "
          f"fallback_written={min(25,len(fallback))}", flush=True)

    receptor = find_receptor()
    if receptor is None:
        print("[screen] NO receptor pdbqt found -> submitting curated/QED fallback",
              flush=True)
        return
    print(f"[screen] receptor={receptor}", flush=True)

    # candidate list: curated first, then enumerated by QED.
    cand = [s for s, _ in curated_valid] + [s for s, _ in enum]
    # bound the docking set (all are docked; order is irrelevant to selection).
    MAXCAND = 2600
    cand = cand[:MAXCAND]
    # budget-aware cap so a short validate run still finishes docking.
    cap = len(cand) if BUDGET > 1500 else min(len(cand), 22)
    cand = cand[:cap]
    print(f"[screen] docking {len(cand)} candidates on {NCPU} cores "
          f"budget={BUDGET:.0f}s", flush=True)

    deadline = START + BUDGET - 240.0
    results = []
    try:
        pool = Pool(processes=NCPU, initializer=init_worker, initargs=(receptor,))
    except Exception as e:
        print(f"[screen] pool init failed: {e} -> fallback stands", flush=True)
        return

    try:
        done = 0
        errs = {}
        for r in pool.imap_unordered(dock_one, cand, chunksize=1):
            done += 1
            if r is not None and "error" in r:
                e = r["error"]
                errs[e] = errs.get(e, 0) + 1
                if len(errs) <= 8 and errs[e] == 1:
                    print(f"[screen] dock error sample: {e}", flush=True)
            elif r is not None:
                results.append(r)
            if done % 20 == 0 and results:
                results.sort(key=lambda x: x["objective"])
                write_solution([x["smiles"] for x in results])
                b = results[0]
                print(f"[screen] {done}/{len(cand)} best_obj={b['objective']:.2f} "
                      f"dock={b['docking']:.2f} qed={b['qed']:.2f}", flush=True)
            if time.time() > deadline:
                print("[screen] time budget reached, stopping dispatch", flush=True)
                break
    finally:
        try:
            pool.terminate(); pool.join()
        except Exception:
            pass

    if results:
        results.sort(key=lambda x: x["objective"])
        write_solution([x["smiles"] for x in results])
        print("[screen] FINAL top candidates:", flush=True)
        for x in results[:10]:
            print(f"  obj={x['objective']:.3f} dock={x['docking']:.3f} "
                  f"qed={x['qed']:.3f} {x['smiles']}", flush=True)
        top3 = results[:3]
        print(f"[screen] top3_mean_objective={sum(x['objective'] for x in top3)/len(top3):.3f}",
              flush=True)
    else:
        print(f"[screen] no dock results -> fallback stands; error_summary={errs}",
              flush=True)


if __name__ == "__main__":
    main()
