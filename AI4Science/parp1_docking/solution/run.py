"""Dock the PARP1 candidate library locally against the real DOCKSTRING PARP1
receptor (shipped inside the `dockstring` pip package) and select the true-best
candidates.

We reproduce the standard DOCKSTRING pipeline: rdkit AddHs -> EmbedMolecule(seed=42)
-> MMFFOptimize -> meeko pdbqt -> Vina(sf=vina, seed=42) -> compute_vina_maps on
the same box -> dock(exhaustiveness=8, n_poses=1). objective = docking + 10*(1-QED).

Parallelised across candidates (one fresh Vina per dock, cpu=1 per worker) so the
per-dock RNG state is fully reproducible. Writes solution.json
incrementally with the current top candidates so we always have a valid output.
"""
import os
import sys
import json
import glob
import time
import signal
from multiprocessing import Pool

from library import valid_library

BOX_CENTER = (26.835, 11.332, 27.744)
BOX_SIZE = (30.0, 30.0, 30.0)
QED_PENALTY_WEIGHT = 10.0
EXHAUSTIVENESS = 8
VINA_SEED = 42

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(HERE, ".run")
os.makedirs(RUN_DIR, exist_ok=True)
SOLUTION_PATH = os.path.join(HERE, "solution.json")

TIME_BUDGET = float(os.environ.get("TIME_BUDGET_SEC", "7200"))
N_CPU = int(os.environ.get("CPUS", "8"))
START = time.time()
# reserve tail time for writing / safety
DEADLINE = START + max(120.0, TIME_BUDGET - 180.0)

# High-confidence fallback set (strongest known molecule first) so a valid
# solution.json exists even if docking never runs.
FALLBACK = [
    "CN1CCN(C(=O)c2ccc(F)c(Cc3n[nH]c(=O)c4ccccc34)c2)CC1",
    "O=C(c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1)N1CCNCC1",
    "O=C(c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1)N1CCOCC1",
    "O=C(c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1)N1CCCC1",
    "O=C(c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1)N1CCC1",
    "CN(C)C(=O)c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1",
    "O=C(c1ccc(CC2=NNC(=O)c3ccccc23)cc1F)N1CCN(C(=O)C2CC2)CC1",
    "NC(=O)c1cccc2cn(-c3ccc(C4CCCNC4)cc3)nc12",
    "NC(=O)c1cccc2[nH]c(nc12)C1(C)CCCN1",
    "O=C1NN=C(Cc2ccc(C(=O)N3CCN(C)CC3)c(F)c2)c2ccccc21",
]


# Confirmed strong candidates from local docking (dock these first as insurance).
PRIORITY = [
    "N#CC1CN(C(=O)c2ccc(F)c(Cc3n[nH]c(=O)c4ccccc34)c2)C1",           # 3-CN-azetidine
    "NC(=O)N1CCN(C(=O)c2ccc(F)c(Cc3n[nH]c(=O)c4ccccc34)c2)CC1",      # carbamoyl-pip
    "O=C(c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1)N1CCC(F)(F)CC1",        # 4,4-diF-pip
    "CN1CCN(C(=O)c2ccc(F)c(Cc3n[nH]c(=O)c4ccccc34)c2)CC1",           # N-Me-pip (strong hit)
    "CN1CCN(C(=O)c2ccc(Cl)c(Cc3n[nH]c(=O)c4ccccc34)c2)CC1",          # Cl variant
    "O=C(c1ccc(F)c(Cc2n[nH]c(=O)c3cccc(F)c23)c1)N1CCNCC1",           # F-phthalazinone
    "CC1CCN(C(=O)c2ccc(F)c(Cc3n[nH]c(=O)c4ccccc34)c2)CC1",           # 4-Me-pip
    "O=C(NC1CCCCC1)c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1",             # cyclohexyl amide
    "CC1(O)CCN(C(=O)c2ccc(F)c(Cc3n[nH]c(=O)c4ccccc34)c2)CC1",        # 4-OH-4-Me-pip
    "O=C(c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1)N1CCC(C#N)CC1",         # 4-CN-pip
]


def write_solution(smiles_list):
    payload = {"molecules": list(smiles_list)[:25], "target": "PARP1"}
    tmp = SOLUTION_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, SOLUTION_PATH)


def find_receptor():
    """Locate the DOCKSTRING PARP1 receptor pdbqt from the installed package."""
    cands = []
    try:
        import dockstring
        base = os.path.dirname(os.path.abspath(dockstring.__file__))
        cands += glob.glob(os.path.join(base, "**", "*.pdbqt"), recursive=True)
    except Exception as e:
        print("dockstring import failed:", e, flush=True)
    # also scan site-packages broadly as a fallback
    for p in list(sys.path):
        if p and os.path.isdir(p):
            cands += glob.glob(os.path.join(p, "**", "*PARP1*.pdbqt"), recursive=True)
    # prefer files that mention PARP1
    parp = [c for c in cands if "parp1" in os.path.basename(c).lower()]
    chosen = parp[0] if parp else (cands[0] if cands else None)
    if chosen:
        print("receptor:", chosen, flush=True)
    return chosen


RECEPTOR = None  # set in worker init


def _init_worker(receptor_path):
    global RECEPTOR
    RECEPTOR = receptor_path
    # keep each worker single-threaded
    os.environ["OMP_NUM_THREADS"] = "1"


def dock_one(args):
    """Standard per-molecule dock. Returns (smi, qed, docking, obj) or None."""
    smi, qed = args
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        from vina import Vina

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        molH = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(molH, randomSeed=VINA_SEED) != 0:
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
        v = Vina(sf_name="vina", cpu=1, seed=VINA_SEED, verbosity=0)
        v.set_receptor(str(RECEPTOR))
        v.set_ligand_from_string(pdbqt)
        v.compute_vina_maps(center=list(BOX_CENTER), box_size=list(BOX_SIZE))
        v.dock(exhaustiveness=EXHAUSTIVENESS, n_poses=1)
        docking = float(v.energies(n_poses=1)[0][0])
    except Exception:
        return None
    obj = docking + QED_PENALTY_WEIGHT * (1.0 - qed)
    return (smi, float(qed), docking, obj)


def main():
    # 1) always leave a valid fallback first
    write_solution(FALLBACK)

    # 2) install-time deps already done by solve.sh; locate receptor
    receptor = find_receptor()
    if receptor is None:
        print("NO RECEPTOR FOUND -> submitting curated fallback set", flush=True)
        write_solution(FALLBACK)
        return

    # 3) build library; dock confirmed strong candidates FIRST (insurance against
    #    truncation), then the rest ordered by QED desc.
    lib = valid_library()
    qed_of = {smi: q for smi, q, mw in lib}
    # canonicalize priority set and keep only those present in the library
    from rdkit import Chem
    prio_canon = []
    for s in (PRIORITY + FALLBACK):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        can = Chem.MolToSmiles(m)
        if can in qed_of and can not in prio_canon:
            prio_canon.append(can)
    prio_set = set(prio_canon)
    rest = sorted([(smi, q) for smi, q, mw in lib if smi not in prio_set],
                  key=lambda x: -x[1])
    tasks = [(s, qed_of[s]) for s in prio_canon] + rest
    print(f"library size = {len(lib)}  (priority={len(prio_canon)})", flush=True)

    results = []
    n_workers = max(1, min(N_CPU, len(tasks)))
    print(f"docking with {n_workers} workers, deadline in "
          f"{DEADLINE - time.time():.0f}s", flush=True)

    pool = Pool(processes=n_workers, initializer=_init_worker, initargs=(receptor,))
    done = 0
    try:
        for r in pool.imap_unordered(dock_one, tasks, chunksize=1):
            done += 1
            if r is not None:
                results.append(r)
                # incremental save of current best-by-objective
                results.sort(key=lambda x: x[3])
                write_solution([x[0] for x in results[:15]])
            if done % 10 == 0 or r is None:
                el = time.time() - START
                best = results[0][3] if results else float("nan")
                print(f"[{el:6.0f}s] docked {done}/{len(tasks)} "
                      f"valid={len(results)} best_obj={best:.3f}", flush=True)
            if time.time() > DEADLINE:
                print("deadline reached, stopping dock loop", flush=True)
                pool.terminate()
                break
    finally:
        try:
            pool.close(); pool.join()
        except Exception:
            pass

    if not results:
        print("no dock succeeded -> fallback", flush=True)
        write_solution(FALLBACK)
        return

    results.sort(key=lambda x: x[3])
    print("=== TOP CANDIDATES (local dock) ===", flush=True)
    for smi, q, dock, obj in results[:20]:
        print(f"obj={obj:8.3f} dock={dock:8.3f} qed={q:.3f}  {smi}", flush=True)
    top3 = results[:3]
    top3_mean = sum(x[3] for x in top3) / len(top3)
    print(f"local top3_mean_objective = {top3_mean:.4f}", flush=True)

    # submit top 12 by local objective (scoring re-docks; extra margin vs any noise)
    write_solution([x[0] for x in results[:12]])
    print("wrote final solution.json", flush=True)


if __name__ == "__main__":
    main()
