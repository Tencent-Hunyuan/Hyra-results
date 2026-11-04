# Roadmap (not in this release)

- **Clifford-labelled symmetry roots** (qubit permutation + per-qubit X/Y/Z relabelling): needs a spec
  revision, checker support and native roots. Measured on the released headline codes: no gain
  (452-115-18 x1.0, 488-123-22 x1.0, 336-10-28 x1.0; 21/22 sampled board codes x1.0). Large gains exist
  for other presentations/codes (e.g. Lee et al.'s plain fold o04829: 1356 -> 6 roots, measured during
  development).
- **Presentation-independent symmetry detection** (nauty on a canonical sparse generating set).
- **GPU independent checker** for B&B coverage certificates (CPU replay of 336-10-28's 1.44e14 nodes
  would take weeks).
- **SAT**: connected-support encoding and adaptive cube refinement (deferred: 2/363 ledger entries
  needed refine.py, none headline).
- **Extra B&B bounds** (Lee et al.'s b1/b3): measured 1.40-1.46x fewer nodes but 1.45-1.53x slower.
