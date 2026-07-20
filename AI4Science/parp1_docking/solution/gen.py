"""PARP1 DOCKSTRING joint-objective optimizer.

Strategy: enumerate a focused library of PARP1 pharmacophores (phthalazinone /
olaparib family, benzimidazole-4-carboxamide / veliparib family, quinazolinone,
indazole-carboxamide / niraparib family) decorated with drug-like tails, plus
known marketed inhibitors. Each candidate is docked against the DOCKSTRING
PARP1 receptor + pocket box with the standard pipeline (rdkit embed
seed 42 -> meeko -> Vina exhaustiveness 8), and ranked by the exact joint
objective  docking + 10*(1-QED)  (lower is better). Stage 1 docks the whole
library with cpu=1 workers in parallel; stage 2 re-docks the best with cpu=CPUS,
then submits the top candidates.
"""
import os, sys, time, json, multiprocessing as mp

T0 = time.time()
BUDGET = float(os.environ.get("TIME_BUDGET_SEC", "7200"))
NCPU = max(1, int(os.environ.get("CPUS", "4")))
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, ".run"); os.makedirs(RUN, exist_ok=True)
SOL = os.path.join(HERE, "solution.json")

BC = (26.835, 11.332, 27.744)
BS = (30.0, 30.0, 30.0)

def log(*a):
    print(*a, flush=True)

# ---- locate the DOCKSTRING PARP1 receptor (shipped in the dockstring pkg) ----
def find_receptor():
    try:
        import dockstring
        d = os.path.dirname(dockstring.__file__)
        p = os.path.join(d, "resources", "targets", "PARP1_target.pdbqt")
        if os.path.exists(p):
            return p
    except Exception as e:
        log("dockstring import failed:", e)
    # fallback: search site-packages
    import glob
    for base in sys.path:
        for p in glob.glob(os.path.join(base, "**", "PARP1_target.pdbqt"), recursive=True):
            return p
    return None

REC = find_receptor()
log("receptor:", REC)

# ---------------------- curated known PARP1 inhibitors ----------------------
KNOWN = [
    # olaparib
    "O=C1NN=C(Cc2ccc(C(=O)N3CCC(C(=O)C4CC4)CC3)cc2F)c2ccccc21",
    # niraparib
    "O=C(N)c1ccc2c(c1)cn(-c1ccc(C3CCCNC3)cc1)n2",
    # veliparib
    "NC(=O)c1cccc2[nH]c(C3(C)CCCN3)nc12",
    # rucaparib
    "CNCc1ccc2c(c1)c1[nH]c(=O)c3cccc4c3c1n2CC4",
    # pamiparib
    "CC1(F)Cc2c(cccc2C(N)=O)... ",  # placeholder replaced below
]
# clean placeholder
KNOWN = [s for s in KNOWN if "..." not in s]
KNOWN += [
    # pamiparib (BGB-290)
    "CC1CCc2c(cccc2C(N)=O)... ".replace("... ", "") if False else "O=C(N)c1cccc2c1CC(C)(c1nc3ccccc3[nH]1)N2",
    # fluzoparib-like phthalazinone
    "O=C1NN=C(Cc2ccc(CN3CCN(C4CCS(=O)(=O)CC4)CC3)cc2F)c2ccccc21",
    # simple 4-benzylphthalazinone
    "O=C1NN=C(Cc2ccccc2)c2ccccc21",
    # 3-aminobenzamide (classic PARP pharmacophore)
    "Nc1cccc(C(N)=O)c1",
    # benzimidazole-4-carboxamide bare
    "NC(=O)c1cccc2[nH]cnc12",
]

# --------------------------- combinatorial library ---------------------------
def build_library():
    smis = set(KNOWN)

    # ---- phthalazinone (olaparib) family: 4-(substituted-benzyl)phthalazin-1(2H)-one ----
    # Central aryl linkers (tail placeholder {T}). CH2-linked benzamide is the
    # proven olaparib motif; enumerate F/Cl/CN substitution + a pyridine central.
    centrals = [
        "Cc3ccc({T})cc3",       # CH2, para tail, no halogen
        "Cc3ccc({T})cc3F",      # olaparib central: ortho-F
        "Cc3ccc({T})c(F)c3",    # F meta to tail
        "Cc3ccc({T})cc3Cl",
        "Cc3ccc({T})cc3C#N",
        "Cc3cc({T})ccc3F",      # meta tail
        "c3ccc({T})cc3F",       # direct (no CH2)
        "Cc3ccc({T})cc3OC",
    ]
    # Acyl / small caps for a piperazine N4CCN(CAP)CC4
    pz_caps = [
        "C(C)=O", "C(=O)CC", "C(=O)C5CC5", "C(=O)C5CCC5", "C=O",
        "C(N)=O", "C(=O)OC", "S(C)(=O)=O", "CC#N", "C(=O)CO",
        "C(=O)C(F)(F)F", "C", "CC", "C5CC5", "C(=O)N(C)C", "C(=O)C(C)C",
    ]
    # 4-substituents for a piperidine N4CCC(SUB)CC4
    pip_subs = [
        "C(=O)C5CC5", "C(=O)C", "C(N)=O", "C#N", "F", "O", "OC", "N",
        "C(=O)OC", "C(=O)N(C)C", "N5CCCC5", "N5CCOCC5", "S(C)(=O)=O",
        "CO", "C(F)(F)F", "C(=O)C5CCC5", "C(=O)N5CCCC5", "OCC",
    ]
    amines = []
    for c in pz_caps:
        amines.append("N4CCN(%s)CC4" % c)
    for s in pip_subs:
        amines.append("N4CCC(%s)CC4" % s)
    amines += [
        "N4CCC(F)(F)CC4", "N4CCCC4", "N4CCOCC4", "N4CC(F)(F)C4",
        "N4CCC5(CC4)OCCO5", "N4CCN(C5CC5)CC4", "N4CCN(Cc5ccncc5)CC4",
        "N4CCC(C5CC5)CC4", "N4CCC(c5ncccn5)CC4", "N4CCC(C(=O)C)(F)CC4",
        "N4CCC(C#N)(F)CC4", "N4CC(C(N)=O)C4", "N4CCC(N5CCC5)CC4",
        "NCC4CC4", "N4CCC(C(=O)N5CCOCC5)CC4",
    ]
    for c in centrals:
        for a in amines:
            benzyl = c.format(T="C(=O)%s" % a)
            smis.add("O=c1[nH]nc(%s)c2ccccc12" % benzyl)

    # a few sulfonamide / reverse tails on the proven no-F and ortho-F centrals
    for c in ["Cc3ccc({T})cc3", "Cc3ccc({T})cc3F"]:
        for t in ["S(=O)(=O)N4CCN(C)CC4", "S(=O)(=O)N4CCOCC4",
                  "C(=O)NCc4ccncc4", "C(=O)N4CCC(c5ccncc5)CC4",
                  "C(=O)N4CCC(n5cccn5)CC4"]:
            smis.add("O=c1[nH]nc(%s)c2ccccc12" % c.format(T=t))

    # ---- benzimidazole-4-carboxamide (veliparib) family: 2-R ----
    bz_r = [
        "C3(C)CCCN3", "C3CCCN3", "C3CCNCC3", "C3CCN(C)CC3",
        "c3ccccc3", "c3ccc(F)cc3", "c3ccccc3F", "Cc3ccccc3",
        "c3ccc(C(=O)N4CCCCC4)cc3",
        "c3ccc(C(=O)N4CCC(C(=O)C5CC5)CC4)cc3",
        "c3ccc(N4CCCC4)cc3", "c3ccc(CN4CCCC4)cc3",
        "c3cccnc3", "C3CCN(Cc4ccccc4)CC3",
        "c3ccc(F)cc3F", "CC3(N)CCCN3",
        "c3ccc(C4CCNCC4)cc3", "c3ccc(N4CCNCC4)cc3",
    ]
    for r in bz_r:
        smis.add("NC(=O)c1cccc2[nH]c(%s)nc12" % r)

    # ---- quinazolin-4(3H)-one family: 2-R ----
    qz_r = [
        "c3ccccc3", "c3ccc(F)cc3", "Cc3ccccc3",
        "c3ccc(C(=O)N4CCC(C(=O)C5CC5)CC4)cc3",
        "CN3CCCC3", "c3cccnc3", "c3ccc(N4CCNCC4)cc3",
        "c3ccc(CN4CCNCC4)cc3",
    ]
    for r in qz_r:
        smis.add("O=c1[nH]c(%s)nc2ccccc12" % r)

    # ---- indazole-carboxamide (niraparib) family ----
    nz_r = [
        "c3ccc(C4CCCNC4)cc3", "c3ccc(C4CCNCC4)cc3",
        "c3ccc(N4CCNCC4)cc3", "c3ccc(F)cc3", "c3ccccc3",
        "c3ccc(C4CCNC4)cc3", "c3ccc(CN4CCCC4)cc3",
    ]
    for r in nz_r:
        smis.add("O=C(N)c1ccc2c(c1)cn(-%s)n2" % r)
        smis.add("NC(=O)c1ccc2cn(-%s)nc2c1" % r)

    # ---- phenanthridinone / isoquinolinone small set ----
    smis.update([
        "O=c1[nH]c2ccccc2c2ccccc12",              # phenanthridinone
        "O=c1[nH]cc2ccccc2c1",                     # isoquinolinone
        "O=C1NC=Cc2ccc(CN3CCN(C)CC3)cc21",
        "O=c1[nH]nc(Cc2ccc(N3CCN(C)CC3)cc2F)c2ccccc12",
        "O=c1[nH]nc(Cc2ccc(N3CCOCC3)cc2F)c2ccccc12",
        "O=c1[nH]nc(Cc2ccc(C3CCN(C)CC3)cc2F)c2ccccc12",
    ])
    return list(smis)

# --------------------------- validity / QED prefilter ---------------------------
def prefilter(raw):
    from rdkit import Chem
    from rdkit.Chem import QED, Descriptors, rdMolDescriptors
    keep = {}
    for s in raw:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        mw = Descriptors.MolWt(m)
        if mw < 230 or mw > 540:
            continue
        if m.GetNumHeavyAtoms() > 44:
            continue
        try:
            q = float(QED.qed(m))
        except Exception:
            continue
        if q < 0.48:
            continue
        rot = rdMolDescriptors.CalcNumRotatableBonds(m)
        if rot > 11:
            continue
        can = Chem.MolToSmiles(m)
        if can not in keep or q > keep[can][1]:
            keep[can] = (can, q)
    return [v[0] for v in keep.values()]

# ------------------------------ docking worker ------------------------------
_V = None
_CPU = 1
def _winit(cpu):
    global _V, _CPU
    _CPU = cpu
    from vina import Vina
    _V = Vina(sf_name="vina", cpu=cpu, seed=42, verbosity=0)
    _V.set_receptor(REC)
    _V.compute_vina_maps(center=list(BC), box_size=list(BS))

def _dock(smi):
    from rdkit import Chem
    from rdkit.Chem import AllChem, QED
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        molH = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(molH, randomSeed=42) != 0:
            return None
        try:
            AllChem.MMFFOptimizeMolecule(molH)
        except Exception:
            pass
        setups = MoleculePreparation().prepare(molH)
        pdbqt, ok, _ = PDBQTWriterLegacy.write_string(setups[0])
        if not ok:
            return None
        _V.set_ligand_from_string(pdbqt)  # maps already computed once per worker in _winit
        _V.dock(exhaustiveness=8, n_poses=1)
        ds = float(_V.energies(n_poses=1)[0][0])
        q = float(QED.qed(mol))
        return (smi, ds, q, ds + 10.0 * (1.0 - q))
    except Exception:
        return None

def dock_many(smis, cpu, timeleft):
    """Dock a list with a pool of workers (cpu per worker). Returns list of tuples."""
    results = []
    n_workers = max(1, NCPU // cpu)
    deadline = time.time() + timeleft
    with mp.Pool(processes=n_workers, initializer=_winit, initargs=(cpu,)) as pool:
        it = pool.imap_unordered(_dock, smis, chunksize=1)
        done = 0
        for r in it:
            done += 1
            if r is not None:
                results.append(r)
            if done % 20 == 0:
                log("  docked %d/%d  best-so-far=%.2f  t=%ds" % (
                    done, len(smis),
                    min([x[3] for x in results], default=0.0),
                    time.time() - T0))
            if time.time() > deadline:
                log("  stage time budget reached at %d/%d" % (done, len(smis)))
                pool.terminate()
                break
    return results

def write_solution(smis):
    smis = list(dict.fromkeys(smis))[:25]
    with open(SOL, "w") as f:
        json.dump({"molecules": smis, "target": "PARP1"}, f, indent=2)
    log("wrote %d molecules to solution.json" % len(smis))

# ------------------------------------ main ------------------------------------
def main():
    # robust fallback first (known strong + drug-like inhibitors)
    fallback = [
        "O=C1NN=C(Cc2ccc(C(=O)N3CCC(C(=O)C4CC4)CC3)cc2F)c2ccccc21",  # olaparib
        "O=C(N)c1ccc2c(c1)cn(-c1ccc(C3CCCNC3)cc1)n2",                # niraparib
        "NC(=O)c1cccc2[nH]c(C3(C)CCCN3)nc12",                        # veliparib
        "O=c1[nH]nc(Cc2ccc(C(=O)N3CCN(C)CC3)cc2F)c2ccccc12",
        "O=c1[nH]nc(Cc2ccc(C(=O)N3CCC(C(N)=O)CC3)cc2F)c2ccccc12",
        "NC(=O)c1cccc2[nH]c(C3CCNCC3)nc12",
        "O=c1[nH]nc(Cc2ccc(C(=O)N3CCOCC3)cc2F)c2ccccc12",
        "O=c1[nH]nc(Cc2ccc(C(=O)N3CCCC3)cc2F)c2ccccc12",
    ]
    write_solution(fallback)

    if REC is None:
        log("NO RECEPTOR -> submitting curated fallback only")
        return

    raw = build_library()
    log("library raw:", len(raw))
    cand = prefilter(raw)
    log("after prefilter:", len(cand))

    # Stage 1: broad dock, cpu=1 workers in parallel
    stage1_budget = 0.72 * BUDGET - (time.time() - T0)
    log("STAGE1 docking %d candidates, budget=%ds, workers=%d" % (len(cand), int(stage1_budget), NCPU))
    res = dock_many(cand, cpu=1, timeleft=max(60, stage1_budget))
    res.sort(key=lambda r: r[3])
    log("STAGE1 done: %d docked. Top:" % len(res))
    for r in res[:12]:
        log("   obj=%.2f dock=%.2f qed=%.3f  %s" % (r[3], r[1], r[2], r[0]))
    if res:
        write_solution([r[0] for r in res[:20]])

    # Stage 2: re-dock top-40 with cpu=NCPU for higher-fidelity multi-core scoring
    top = [r[0] for r in res[:50]]
    if top and (BUDGET - (time.time() - T0)) > 300:
        log("STAGE2 re-dock top %d with cpu=%d (multi-core)" % (len(top), NCPU))
        res2 = dock_many(top, cpu=NCPU, timeleft=max(60, 0.9 * (BUDGET - (time.time() - T0))))
        res2.sort(key=lambda r: r[3])
        log("STAGE2 done. Top:")
        for r in res2[:15]:
            log("   obj=%.2f dock=%.2f qed=%.3f  %s" % (r[3], r[1], r[2], r[0]))
        if res2:
            final = [r[0] for r in res2[:15]]
            write_solution(final)
            top3 = res2[:3]
            log("PREDICTED top-3 mean objective = %.3f" % (sum(x[3] for x in top3) / len(top3)))

if __name__ == "__main__":
    main()
