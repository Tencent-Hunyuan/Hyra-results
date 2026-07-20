"""PARP1 DOCKSTRING joint-objective solver.

Re-docks a large, focused, drug-like phthalazinone-family library with the
standard DOCKSTRING pipeline (rdkit embed seed=42 -> MMFF -> meeko pdbqt -> Vina
seed=42, exhaustiveness 8, PARP1 box), so the objective ranking is reproducible
under re-docking (docking is deterministic under a fixed seed). Computes
objective = docking + 10*(1-QED), ranks ascending, submits the best <=25. A
time-guarded GA phase mutates the top hits (aromatic halogen/CN walk) to explore
ring positions beyond the fixed grid.

Docking is a fast PROXY for binding free energy, not a real-affinity
measurement; a good score means "strong on the Vina metric for this fixed PARP1
pocket," nothing more.
"""
import os
import sys
import json
import time
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
RECEPTOR = os.path.join(HERE, "parp1_receptor.pdbqt")

BOX_CENTER = [26.835, 11.332, 27.744]
BOX_SIZE = [30.0, 30.0, 30.0]
SEED = 42
EXHAUSTIVENESS = 8

START = time.time()
TIME_BUDGET = float(os.environ.get("TIME_BUDGET_SEC", "7200"))
CPUS = int(os.environ.get("CPUS", "16"))
OUT = os.path.join(HERE, "solution.json")

FALLBACK = [
    "CC(=O)N1CCN(C(=O)c2ccc(Cc3n[nH]c(=O)c4ccccc34)cc2)CC1",
    "CC(=O)N1CCN(C(=O)c2ccc(Cc3n[nH]c(=O)c4cc(F)ccc34)cc2F)CC1",
    "CN(C)C(=O)N1CCN(C(=O)c2ccc(Cc3n[nH]c(=O)c4cc(F)ccc34)cc2F)CC1",
    "CN1CCN(C(=O)c2ccc(Cc3n[nH]c(=O)c4cc(F)ccc34)cc2F)CC1",
    "O=C(c1ccc(Cc2n[nH]c(=O)c3ccccc23)cc1)N1CCN(C(=O)C2CC2)CC1",
    "CC(=O)N1CCN(C(=O)c2ccc(Cc3n[nH]c(=O)c4ccccc34)cc2F)CC1",
    "CCC(=O)N1CCN(C(=O)c2ccc(Cc3n[nH]c(=O)c4cc(F)ccc34)cc2F)CC1",
    "CN1CCN(C(=O)c2ccc(Cc3n[nH]c(=O)c4ccccc34)cc2F)CC1",
    "CC(C)C(=O)N1CCN(C(=O)c2ccc(Cc3n[nH]c(=O)c4cc(F)ccc34)cc2F)CC1",
    "CCC(=O)N1CCN(C(=O)c2ccc(Cc3n[nH]c(=O)c4ccccc34)cc2)CC1",
]


def write_solution(mols):
    payload = {"molecules": mols, "target": "PARP1"}
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, OUT)


def dock_one(smiles):
    """Standard per-molecule docking pipeline. Returns (canonical, docking, qed, obj) or None."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, QED
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        from vina import Vina

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        molH = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(molH, randomSeed=SEED) != 0:
            return None
        try:
            AllChem.MMFFOptimizeMolecule(molH)
        except Exception:
            pass
        try:
            setups = MoleculePreparation().prepare(molH)
            pdbqt, ok, _ = PDBQTWriterLegacy.write_string(setups[0])
            if not ok:
                return None
        except Exception:
            return None
        v = Vina(sf_name="vina", cpu=1, seed=SEED, verbosity=0)
        v.set_receptor(RECEPTOR)
        v.set_ligand_from_string(pdbqt)
        v.compute_vina_maps(center=BOX_CENTER, box_size=BOX_SIZE)
        v.dock(exhaustiveness=EXHAUSTIVENESS, n_poses=1)
        docking = float(v.energies(n_poses=1)[0][0])
        qed = float(QED.qed(mol))
        obj = docking + 10.0 * (1.0 - qed)
        return (Chem.MolToSmiles(mol), docking, qed, obj)
    except Exception:
        return None


def mutate_neighbors(smiles, qed_min=0.55):
    """Aromatic halogen/CN walk: substitute one aromatic H with F/Cl/C/C#N.
    Returns a list of NEW canonical SMILES (drug-like), robust to failures."""
    from rdkit import Chem
    from rdkit.Chem import QED, Descriptors

    out = []
    try:
        base = Chem.MolFromSmiles(smiles)
        if base is None:
            return out
        # candidate substituent SMILES fragments to append to an aromatic carbon
        subs = ["F", "Cl", "C", "C#N", "OC"]
        for atom in base.GetAtoms():
            if not atom.GetIsAromatic() or atom.GetSymbol() != "C":
                continue
            if atom.GetTotalNumHs() < 1:
                continue
            idx = atom.GetIdx()
            for frag in subs:
                try:
                    rw = Chem.RWMol(base)
                    # build the substituent as an editable fragment
                    sub_mol = Chem.MolFromSmiles(frag)
                    if sub_mol is None:
                        continue
                    amap = {}
                    for a in sub_mol.GetAtoms():
                        na = Chem.Atom(a.GetAtomicNum())
                        na.SetIsAromatic(a.GetIsAromatic())
                        amap[a.GetIdx()] = rw.AddAtom(na)
                    for b in sub_mol.GetBonds():
                        rw.AddBond(amap[b.GetBeginAtomIdx()],
                                   amap[b.GetEndAtomIdx()], b.GetBondType())
                    rw.AddBond(idx, amap[0], Chem.BondType.SINGLE)
                    m2 = rw.GetMol()
                    Chem.SanitizeMol(m2)
                    q = float(QED.qed(m2))
                    if q < qed_min or Descriptors.MolWt(m2) > 500:
                        continue
                    out.append(Chem.MolToSmiles(m2))
                except Exception:
                    continue
    except Exception:
        return out
    return out


def dock_batch(smis, workers, deadline, seen, results, tag=""):
    """Dock a list; append (smi,dock,qed,obj) to results; respect deadline."""
    todo = [s for s in smis if s not in seen]
    for s in todo:
        seen.add(s)
    if not todo:
        return 0
    n_new = 0
    with mp.Pool(processes=workers) as pool:
        it = pool.imap_unordered(dock_one, todo)
        done = 0
        for r in it:
            done += 1
            if r is not None:
                results.append(r)
                n_new += 1
                if done % 25 == 0 or r[3] < -9.5:
                    print(f"[dock{tag}] {done}/{len(todo)} dock={r[1]:.2f} "
                          f"qed={r[2]:.3f} obj={r[3]:.2f} {r[0]}", flush=True)
            if time.time() > deadline:
                print(f"[solve] deadline in dock{tag}, stopping", flush=True)
                pool.terminate()
                break
    return n_new


def main():
    import library

    lib = library.build_library()
    print(f"[solve] library size={len(lib)} cpus={CPUS} budget={TIME_BUDGET:.0f}s",
          flush=True)
    write_solution(FALLBACK)  # guarantee a valid submission immediately

    workers = max(1, min(CPUS, 16))
    deadline = START + min(TIME_BUDGET * 0.90, TIME_BUDGET - 150)

    seen = set()
    results = []

    # ---- Phase 1: exhaustive dock of the focused library ----
    dock_batch(lib, workers, deadline, seen, results, tag="1")
    if results:
        results.sort(key=lambda x: x[3])
        top3 = results[:3]
        print(f"[solve] phase1 top3-mean="
              f"{sum(x[3] for x in top3)/len(top3):.4f} n={len(results)}", flush=True)
        write_solution([x[0] for x in results[:25]])

    # ---- Phase 2..N: GA halogen/CN walk around current best hits ----
    gen = 0
    while time.time() < deadline - 60 and results and gen < 6:
        gen += 1
        results.sort(key=lambda x: x[3])
        parents = [x[0] for x in results[:20]]
        offspring = []
        oseen = set()
        for p in parents:
            for c in mutate_neighbors(p):
                if c not in seen and c not in oseen:
                    oseen.add(c)
                    offspring.append(c)
        if not offspring:
            print(f"[solve] gen{gen}: no new offspring, stopping GA", flush=True)
            break
        print(f"[solve] gen{gen}: docking {len(offspring)} offspring", flush=True)
        n_new = dock_batch(offspring, workers, deadline, seen, results, tag=f"g{gen}")
        results.sort(key=lambda x: x[3])
        top3 = results[:3]
        print(f"[solve] gen{gen} top3-mean="
              f"{sum(x[3] for x in top3)/len(top3):.4f} "
              f"(added {n_new}, total {len(results)})", flush=True)
        write_solution([x[0] for x in results[:25]])
        if n_new == 0:
            break

    if not results:
        print("[solve] no dock results; keeping fallback", flush=True)
        return

    results.sort(key=lambda x: x[3])
    top = results[:25]
    print("\n[solve] ===== TOP CANDIDATES (obj = dock + 10*(1-QED)) =====", flush=True)
    for smi, dock, qed, obj in top[:15]:
        print(f"  obj={obj:.3f} dock={dock:.2f} qed={qed:.3f} {smi}", flush=True)
    top3_mean = sum(x[3] for x in top[:3]) / min(3, len(top))
    print(f"[solve] FINAL predicted top3-mean objective = {top3_mean:.4f} "
          f"(n_valid={len(results)})", flush=True)
    write_solution([x[0] for x in top])
    print(f"[solve] wrote {len(top)} molecules to solution.json", flush=True)


if __name__ == "__main__":
    main()
