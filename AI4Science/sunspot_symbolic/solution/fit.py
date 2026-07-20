#!/usr/bin/env python3
"""Discover an autoregressive sunspot forecast law.

Self-contained: reimplements the free-running multi-step forecast R2 objective
(HORIZON=24, STRIDE=2, clamp/floor matching srfit), fits a linear AR law by
closed-form least squares (teacher-forcing one-step, which proved ~optimal for
the free-run objective), and selects a cycle-aware lag set by greedy forward
selection scored on a held-out chronological tail (a proxy for out-of-sample
forecast skill). Final params are refit on the full train series and written to
solution.json.
"""
import csv, json, sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HORIZON, STRIDE = 24, 2
PRED_LO, PRED_HI, R2_FLOOR = -100.0, 1000.0, -5.0
W_TRAIN, W_VALID, LAM = 0.25, 0.75, 0.005


def load(path):
    ys = []
    with open(path) as fh:
        r = csv.reader(fh); next(r, None)
        for rec in r:
            if len(rec) >= 2:
                ys.append(float(rec[1]))
    return np.asarray(ys, dtype=float)


def fit_ls(lags, ytr, ridge=1.0):
    """Closed-form one-step least squares for a + sum c_k x_k. Intercept unpenalized."""
    K = max(lags); idx = np.arange(K, ytr.size)
    X = np.column_stack([np.ones(idx.size)] + [ytr[idx - k] for k in lags])
    t = ytr[idx]
    A = X.T @ X + ridge * np.eye(X.shape[1]); A[0, 0] -= ridge
    return np.linalg.solve(A, X.T @ t)


def free_run(lags, coef, series, era_lo, era_hi, horizon=HORIZON, stride=STRIDE):
    """Rolling-origin closed-loop forecast R2 over [era_lo, era_hi). Mirrors srfit."""
    series = np.asarray(series, dtype=float); n = series.size; K = max(lags)
    o_first = max(K, era_lo - 1); o_last = min(n - 2, era_hi - 2)
    if o_last < o_first:
        return R2_FLOOR
    origins = np.arange(o_first, o_last + 1, stride); m = origins.size
    if m == 0:
        return R2_FLOOR
    fill = float(series[max(0, era_lo - 1):era_hi].mean()) if era_hi > era_lo else 0.0
    a = coef[0]; cdict = {k: coef[1 + i] for i, k in enumerate(lags)}
    preds = np.empty((m, horizon)); pred_acc = []; true_acc = []
    for h in range(horizon):
        target_pos = origins + (h + 1)
        step = np.full(m, a, dtype=float)
        for k in lags:
            back = h + 1 - k
            col = series[target_pos - k] if back <= 0 else preds[:, back - 1]
            step = step + cdict[k] * col
        step = np.nan_to_num(step, nan=fill, posinf=PRED_HI, neginf=PRED_LO)
        step = np.clip(step, PRED_LO, PRED_HI)
        preds[:, h] = step
        valid = (target_pos >= era_lo) & (target_pos < era_hi) & (target_pos < n)
        if valid.any():
            pred_acc.append(step[valid]); true_acc.append(series[target_pos[valid]])
    if not pred_acc:
        return R2_FLOOR
    yp = np.concatenate(pred_acc); yt = np.concatenate(true_acc)
    ss = float(np.sum((yt - yt.mean()) ** 2))
    if ss <= 0:
        return R2_FLOOR
    return max(R2_FLOOR, 1.0 - float(np.sum((yt - yp) ** 2)) / ss)


def proxy_score(lags, y, cuts=(0.78, 0.82, 0.86)):
    """Robust held-out-tail proxy for the blended objective: fit on early part,
    free-run forecast the held-back tail; average over several cut points."""
    n = y.size
    tails = []
    for frac in cuts:
        cut = int(n * frac)
        coef = fit_ls(lags, y[:cut])
        tails.append(free_run(lags, coef, y, cut, n))
    tail = float(np.mean(tails))
    coef_full = fit_ls(lags, y)
    full = free_run(lags, coef_full, y, max(lags), n)
    pen = LAM * (len(lags) + 1)
    return W_TRAIN * full + W_VALID * tail - pen, full, tail


def build(lags, coef):
    expr = "a+" + "+".join(f"c{k}*x{k}" for k in lags)
    params = {nm: float(v) for nm, v in zip(["a"] + [f"c{k}" for k in lags], coef)}
    return expr, params


def write_solution(lags, y):
    coef = fit_ls(lags, y)
    expr, params = build(lags, coef)
    out = {"expression": expr, "params": params}
    with open(os.path.join(HERE, "solution.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    return expr, params


def main():
    y = load(os.path.join(HERE, "train.csv"))

    # Safe fallback first, so a timeout still leaves a strong solution.json.
    fallback = [1, 3, 108, 120, 240, 264]
    write_solution(fallback, y)
    best_lags = fallback
    best_sc = proxy_score(fallback, y)[0]
    print(f"fallback {fallback} sc={best_sc:.4f}", flush=True)

    # Candidate lag pool: recent months + cycle / multi-cycle anchors (always-true
    # history under the 24-month horizon). Lag indices are not penalized.
    pool = sorted(set(
        list(range(1, 7)) + [12] +
        list(range(90, 165, 6)) +      # ~1 cycle (7.5-13.5 yr)
        list(range(216, 265, 6))       # ~2 cycles (18-22 yr), capped at 264
    ))

    # Greedy forward selection on the robust proxy.
    chosen = []
    cur = -9.0
    while len(chosen) < 12:
        best_add = None; best_v = cur
        for L in pool:
            if L in chosen:
                continue
            sc = proxy_score(sorted(chosen + [L]), y)[0]
            if sc > best_v:
                best_v = sc; best_add = L
        if best_add is None:
            break
        chosen.append(best_add); cur = best_v
        sc, full, tail = proxy_score(sorted(chosen), y)
        print(f"add {best_add:4d} -> {sorted(chosen)} sc={sc:.4f} full={full:.3f} tail={tail:.3f}", flush=True)

    if cur > best_sc:
        best_lags, best_sc = sorted(chosen), cur

    expr, params = write_solution(best_lags, y)
    print(f"FINAL lags={best_lags} sc={best_sc:.4f}", flush=True)
    print("expr:", expr, flush=True)


if __name__ == "__main__":
    main()
