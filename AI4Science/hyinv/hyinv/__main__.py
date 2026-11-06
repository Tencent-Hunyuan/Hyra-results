import argparse

import torch

from . import HyInv, design, read_pdb


def main():
    ap = argparse.ArgumentParser(prog="python -m hyinv", description="HyInv inverse folding: design sequences for a given backbone")
    ap.add_argument("command", choices=["design"])
    ap.add_argument("--pdb", required=True, help="input PDB file")
    ap.add_argument("--chain", help="use only these chains, e.g. A or AB (all by default)")
    ap.add_argument("--fixed", help="positions that keep the native residue (PDB numbering), e.g. 12-20,45 or A12B; prefix a chain name for multi-chain input, e.g. A12-20,B7")
    ap.add_argument("--n", type=int, default=8, help="number of designs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draft-temperature", type=float, default=0.6)
    ap.add_argument("--rewrite-temperature", type=float, default=0.34)
    ap.add_argument("--fast", choices=["off", "exact", "bf16"], default="off",
                    help="accelerated inference (CUDA only): exact = FP32-level, almost always identical to the reference; bf16 = faster, numerically different, sequences differ from the reference")
    ap.add_argument("--weights", help="weights directory (defaults to the bundled weights/)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="designs.fasta")
    a = ap.parse_args()
    if a.fast != "off" and not a.device.startswith("cuda"):
        ap.error("--fast exact/bf16 requires CUDA")
    st = read_pdb(a.pdb, a.chain)
    model = HyInv.from_pretrained(a.weights, a.device)
    seqs = design(model, st, a.n, seed=a.seed, fixed=a.fixed, draft_temperature=a.draft_temperature,
                  rewrite_temperature=a.rewrite_temperature, fast=a.fast)
    with open(a.out, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">{st.name}_design{i} seed={a.seed}\n{s}\n")
    print(f"{len(seqs)} sequences ({len(st)} residues) -> {a.out}")


if __name__ == "__main__":
    main()
