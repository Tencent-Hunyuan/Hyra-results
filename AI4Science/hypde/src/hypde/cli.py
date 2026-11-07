"""Command line: `hypde info`, `hypde forecast`, `hypde evaluate`."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="hypde", description="HyPDE: 2D PDE forecasting")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="model size, datasets, weight check")
    p.add_argument("--weights", default=None)
    p.add_argument("--verify", action="store_true", help="check the weights' sha256")

    p = sub.add_parser("forecast", help="forecast a trajectory of a Well HDF5 file")
    p.add_argument("file")
    p.add_argument("--index", type=int, default=0, help="trajectory inside the file")
    p.add_argument("--dataset", default=None, help="default: the file's dataset_name attribute")
    p.add_argument("--t0", type=int, default=17, help="first forecast frame (needs 6 true frames before it)")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("-o", "--out", required=True, help=".npz with pred, ref and the per-frame VRMSE")
    p.add_argument("--weights", default=None)
    p.add_argument("--device", default=None)

    p = sub.add_parser("evaluate", help="score the test panel (one-step, T[1:20], T[21:60])")
    p.add_argument("--data-root", required=True, help="directory holding <dataset>/data/test/*.hdf5")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--max-traj", type=int, default=0, help="first N trajectories per dataset (0 = all)")
    p.add_argument("--no-onestep", action="store_true", help="skip the one-step pass (one forward per frame)")
    p.add_argument("-o", "--out", required=True, help="results JSON")
    p.add_argument("--weights", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--no-tf32", action="store_true", help="strict fp32 in the processor as well")

    a = ap.parse_args(argv)
    import hypde
    kw = {} if a.weights is None else {"weights": a.weights}

    if a.cmd == "info":
        fc = hypde.load(device="cpu", verify=a.verify, **kw)
        m = fc.model
        print(json.dumps({"version": hypde.__version__, "parameters": m.num_params(),
                          "processor_parameters": m.num_params("processor"),
                          "datasets": sorted(hypde.DATASETS), "sha256_verified": bool(a.verify)}, indent=1))
        return 0

    fc = hypde.load(device=a.device, tf32=not getattr(a, "no_tf32", False), **kw)
    if a.cmd == "forecast":
        import numpy as np
        from .evaluate import forecast, vrmse
        traj = hypde.load_trajectory(a.file, a.index, a.dataset)
        pred, ref = forecast(fc, traj, a.t0, a.steps)
        v = vrmse(pred, ref)
        np.savez(a.out, pred=pred.numpy(), ref=ref.numpy(), vrmse=v, fields=np.array(traj.info.fields))
        print(f"{traj.info.name} {a.file}:{a.index}  {len(pred)} frames from t0={a.t0}  "
              f"mean VRMSE {v.mean():.4f}  -> {a.out}")
        return 0

    from .evaluate import evaluate
    res = evaluate(fc, a.data_root, a.datasets, a.max_traj, onestep=not a.no_onestep,
                   log=lambda s: print(s, file=sys.stderr, flush=True))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    for ds, r in res.items():
        print(f"{ds:<40} " + "  ".join(f"{k} {r[k]:.4f}" for k in ("onestep", "t1_20", "t21_60") if k in r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
