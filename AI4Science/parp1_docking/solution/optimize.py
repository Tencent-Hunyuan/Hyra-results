"""Dock the candidate library against the fixed DOCKSTRING PARP1 pocket with the
standard Vina pipeline (rdkit 3D embed -> meeko pdbqt -> Vina,
seed 42, exhaustiveness 8), then submit the lowest-objective candidates.

Local docking is deterministic under the fixed seed (e.g. the N-methyl-
piperazine olaparib analog gives dock -12.224 / obj -9.775), so this is an
exact oracle for the objective
    objective = docking_score + 10 * (1 - QED)   (lower is better).
Scoring uses the mean of the top-3 (smallest) objectives among the <=25 we
submit, so we submit our 25 lowest-objective candidates.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

from rdkit import Chem
from rdkit.Chem import AllChem, QED
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from library import build_library

T0 = time.time()
BUDGET = float(os.environ.get("TIME_BUDGET_SEC", "7200"))
NCPU = int(os.environ.get("CPUS", "16") or "16")
BOX_C = (26.835, 11.332, 27.744)
BOX_S = (30.0, 30.0, 30.0)
SEED = 42
EXHAUST = 8
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "solution.json")

# --- Locate the DOCKSTRING PARP1 receptor bundled in the dockstring package ---
def find_receptor():
    try:
        import dockstring
        d = os.path.dirname(dockstring.__file__)
        p = os.path.join(d, "resources", "targets", "PARP1_target.pdbqt")
        if os.path.isfile(p):
            return p
    except Exception:
        pass
    return None

REC = find_receptor()

# --- Proven fallback set (guarantees a strong valid solution even if docking dies) ---
FALLBACK = [
    "CN1CCN(C(=O)c2ccc(F)c(Cc3n[nH]c(=O)c4ccccc34)c2)CC1",        # N-Me-piperazine (obj ~-9.77)
    "O=C(c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1)N1CCOCC1",           # morpholine
    "O=C(c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1)N1CCNCC1",           # piperazine
    "O=C(c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1)N1CCCC1",            # pyrrolidine
    "O=C(c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1)N1CCC(F)CC1",        # 4-F-piperidine
    "O=C(c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1)N1CCN(C(=O)C2CC2)CC1",  # olaparib
    "O=C(c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1)N1CCC(O)CC1",        # 4-OH-piperidine
    "CN(C)C(=O)c1ccc(F)c(Cc2n[nH]c(=O)c3ccccc23)c1",              # dimethylamide
    "NC(=O)c1cccc2cn(-c3ccc(C4CCCNC4)cc3)nc12",                   # niraparib
    "CN1N=C(c2ccc(F)cc2)C2NC(=O)c3ccc(F)cc3C21",                  # talazoparib-like
]


def write_solution(smiles_list):
    with open(OUT, "w") as f:
        json.dump({"molecules": smiles_list[:25], "target": "PARP1"}, f, indent=2)


# always leave a valid solution on disk immediately
write_solution(FALLBACK)

# ------------------------------- docking worker -------------------------------
_V = None
_ERR_PRINTED = False


def _make_vina():
    """Create a Vina object with maps ready for the fixed PARP1 pocket."""
    from vina import Vina
    v = Vina(sf_name="vina", cpu=1, seed=SEED, verbosity=0)
    v.set_receptor(REC)
    v.compute_vina_maps(center=list(BOX_C), box_size=list(BOX_S))
    return v


def _init():
    global _V
    try:
        _V = _make_vina()
    except Exception:
        import traceback
        sys.stderr.write("WORKER_INIT_FAIL:\n" + traceback.format_exc() + "\n")
        sys.stderr.flush()
        _V = None


def _prep_pdbqt(smi):
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None
    qed = float(QED.qed(mol))
    molH = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(molH, randomSeed=SEED) != 0:
        return None, qed
    try:
        AllChem.MMFFOptimizeMolecule(molH)
    except Exception:
        pass
    setups = MoleculePreparation().prepare(molH)
    pdbqt, ok, _ = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        return None, qed
    return pdbqt, qed


def _dock(smi):
    global _V, _ERR_PRINTED
    try:
        if _V is None:
            _V = _make_vina()  # lazy fallback if initializer failed silently
        pdbqt, qed = _prep_pdbqt(smi)
        if pdbqt is None:
            return None
        _V.set_ligand_from_string(pdbqt)
        _V.dock(exhaustiveness=EXHAUST, n_poses=1)
        ds = float(_V.energies(n_poses=1)[0][0])
        return (smi, ds, qed, ds + 10.0 * (1.0 - qed))
    except Exception:
        if not _ERR_PRINTED:
            import traceback
            sys.stderr.write("WORKER_DOCK_FAIL(%s):\n%s\n" % (smi, traceback.format_exc()))
            sys.stderr.flush()
            _ERR_PRINTED = True
        return None


def _probe():
    """Synchronous single dock in the main process to confirm the pipeline works
    on this host and surface any error verbatim in the logs."""
    try:
        v = _make_vina()
        pdbqt, qed = _prep_pdbqt(FALLBACK[0])
        if pdbqt is None:
            print("PROBE: prep failed", flush=True)
            return False
        v.set_ligand_from_string(pdbqt)
        v.dock(exhaustiveness=EXHAUST, n_poses=1)
        ds = float(v.energies(n_poses=1)[0][0])
        print("PROBE OK: dock=%.3f qed=%.3f obj=%.3f (%.0fs)"
              % (ds, qed, ds + 10 * (1 - qed), time.time() - T0), flush=True)
        return True
    except Exception:
        import traceback
        print("PROBE FAIL:\n" + traceback.format_exc(), flush=True)
        return False


def main():
    if REC is None:
        print("RECEPTOR NOT FOUND -> submitting fallback set", flush=True)
        return

    lib = build_library()
    # Only shrink for an extremely short debug budget; otherwise dock everything
    # (full library docks in ~5 min; the time guard + incremental writes below
    # protect against any overrun).
    if BUDGET < 400:
        lib = lib[:60]
    try:
        print("RECEPTOR=%s size=%d" % (REC, os.path.getsize(REC)), flush=True)
    except Exception:
        print("RECEPTOR=%s" % REC, flush=True)
    print("LIBRARY_SIZE=%d  NCPU=%d  BUDGET=%.0fs" % (len(lib), NCPU, BUDGET), flush=True)

    _probe()  # confirm pipeline on this host; error (if any) is printed verbatim

    results = []
    deadline = T0 + 0.90 * BUDGET

    def flush_best():
        results.sort(key=lambda r: r[3])
        write_solution([r[0] for r in results[:25]])

    n = 0
    with Pool(processes=max(1, NCPU), initializer=_init) as pool:
        for r in pool.imap_unordered(_dock, lib, chunksize=1):
            n += 1
            if r is not None:
                results.append(r)
            if n % 20 == 0 and results:
                flush_best()
                best = min(results, key=lambda x: x[3])
                print("  progress %d/%d  best_obj=%.3f dock=%.3f qed=%.3f  (%.0fs)"
                      % (n, len(lib), best[3], best[1], best[2], time.time() - T0), flush=True)
            if time.time() > deadline:
                print("TIME GUARD hit at %d docked" % n, flush=True)
                pool.terminate()
                break

    if not results:
        print("NO DOCKING RESULTS -> keeping fallback", flush=True)
        return

    results.sort(key=lambda r: r[3])
    flush_best()

    print("=== TOP CANDIDATES (obj, dock, qed, smiles) ===", flush=True)
    for smi, ds, qed, obj in results[:15]:
        print("  obj=%.3f dock=%.3f qed=%.3f  %s" % (obj, ds, qed, smi), flush=True)
    top3 = results[:3]
    top3_mean = sum(r[3] for r in top3) / len(top3)
    print("SUBMITTED=%d  n_valid=%d  TOP3_MEAN_OBJ=%.4f"
          % (min(25, len(results)), len(results), top3_mean), flush=True)


if __name__ == "__main__":
    main()
    # Final safety: ensure solution.json is non-empty & valid
    try:
        with open(OUT) as f:
            d = json.load(f)
        assert isinstance(d.get("molecules"), list) and d["molecules"]
    except Exception:
        write_solution(FALLBACK)
    print("DONE (%.0fs)" % (time.time() - T0), flush=True)
