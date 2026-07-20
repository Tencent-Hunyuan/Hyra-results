"""srfit - the shared, deterministic core of the sunspot forecasting task.

ONE module, so that scoring and self-checking can never disagree about what a
law means or what it scores. It is used, unchanged, both to score a submission
(free-running forecast R^2 on the held-out VALID era, blended with in-train
forecast R^2) and for a solver's local self-check readouts.

THE CONTRACT
============
The solver submits a CONCRETE *autoregressive* law: a symbolic expression for the
NEXT monthly sunspot number as a function of RECENT sunspot numbers, plus the
fitted value of every free parameter it contains::

    {"expression": "a + b*x1 + c*x2 + d*x12",
     "params": {"a": 9.0, "b": 1.35, "c": -0.45, "d": 0.12}}

- A NAME of the form ``x`` followed by a positive integer ``k`` (``x1``, ``x2``,
  ``x12``, ``x132`` …) is the sunspot number ``k`` months ago: ``s[t-k]``. These
  are the law's only *variables* - the calendar year ``t`` is deliberately NOT
  available (a closed-form function of the year cannot forecast sunspots: the
  ~11-yr Schwabe cycle drifts in phase, so any fixed f(year) dephases within a
  couple of decades). The law must instead capture the cycle's *dynamics*: how the
  next value follows from recent values. Lag indices are capped at ``MAX_LAG``
  (and must be ≥ 1).
- Every OTHER name is a free parameter and MUST have a value in ``params``.
- The solver FITS the parameters itself (any method) and reports their values.
  The scorer does NOT fit - it substitutes the reported values and runs the
  recurrence forward. That is what makes the law and the fitting the solver's own.
- The expression is whitelist-parsed with stdlib ``ast``: only ``+ - * / ** %``,
  unary ±, ``pi``/``e``, and a fixed set of math functions are allowed. A neural
  net, a weight array, a loop, an import, an attribute/subscript - anything not
  on the whitelist - is REJECTED. This keeps the product a compact closed-form
  recurrence, not a black box.

HOW IT IS SCORED - free-running multi-step forecast
===================================================
Skill is measured by ROLLING-ORIGIN, CLOSED-LOOP forecasting. From each origin
month ``o`` in an era, the law predicts ``o+1``; that prediction is fed back in as
``x1`` to predict ``o+2``; and so on for ``H`` months (``HORIZON``). The law only
ever sees TRUE history at or before the origin - beyond that it must stand on its
own predictions. This is genuine forecasting (a persistence "copy last value" law
scores far below a law that captures the dynamics), and the visible VALID era
forecast tracks the held-out TEST era forecast closely, so the score predicts
generalization (corr(valid,test) ≈ 0.9 across a model population, vs ≈ 0.14 for
an old chronological f(year) split this replaces).

Determinism: pure numpy, no RNG anywhere on the scoring path. Same law + same
data -> identical score on any machine. (An optional convenience fitter lives at
the bottom; it is never used on the scoring path.)
"""

from __future__ import annotations

import ast
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ── regularization on the SIZE of the law (the requested "对参数个数做正则") ─────
# We charge the law's EFFECTIVE degrees of freedom, where a d.o.f. is anything the
# agent fit to the data - whether it reports it as a free parameter OR bakes it in
# as a numeric literal. Both are billed at the SAME rate (LAMBDA_PARAM):
#
#     {"expression": "a*x1", "params": {"a": 0.9}}   and
#     {"expression": "0.9*x1", "params": {}}
#
# are the SAME one-d.o.f. law and cost the same. This closes the penalty arbitrage
# a solver exploited on the old task (it re-emitted fitted coefficients as literals
# to slip from 0.005/coef to 0.001/literal - byte-identical predictions, free
# score). A literal is billed as a fitted d.o.f. UNLESS it is a small structural
# integer (|v|≤MAX_STRUCTURAL_INT and integer-valued): the 2 in 2*pi, a harmonic
# multiplier, etc. - things you write to BUILD the form, not values you tuned. Note
# a LAG INDEX lives inside a variable NAME (``x132``), not as a literal, so choosing
# which past months to use is not itself penalized - only the coefficient on each
# lag is a parameter, and that is exactly the d.o.f. we want to regularize. Kept
# deliberately SMALL relative to the R² terms so goodness-of-fit dominates.
LAMBDA_PARAM = float(os.environ.get("SR_LAMBDA_PARAM", "0.005"))
LAMBDA_LITERAL = float(os.environ.get("SR_LAMBDA_LITERAL", "0.001"))
# A numeric literal that is an integer with |value| ≤ this is "structural" (part
# of the form, billed at the cheap LAMBDA_LITERAL rate). Anything else - a float,
# or a large integer - is treated as a fitted constant and billed at LAMBDA_PARAM.
MAX_STRUCTURAL_INT = float(os.environ.get("SR_MAX_STRUCTURAL_INT", "16"))

# ── live-score blend: in-train forecast + held-out forecast ─────────────────────
# Both terms are FREE-RUNNING forecast R² (see free_run_forecast). The TRAIN term
# rolls forecasts forward WITHIN the early era the law was fit on (the agent can
# reproduce it exactly from train.csv); the VALID term forecasts the held-out
# FUTURE era. The split is chronological (train = past, valid/test = future), so
# this is genuine "predict later years from earlier years". Because the dynamics
# are stationary across the record, the held-out VALID forecast tracks the hidden
# TEST forecast closely - the agent-visible score predicts generalization. The
# valid term dominates (forecasting the future IS the task); the train term anchors
# the law and rewards a recurrence that is also self-consistent on its own era.
W_TRAIN = float(os.environ.get("SR_W_TRAIN", "0.25"))
W_VALID = float(os.environ.get("SR_W_VALID", "0.75"))

# ── forecast geometry ───────────────────────────────────────────────────────────
# HORIZON: how many months each rolling forecast runs closed-loop before resetting
# to a new origin. 24 (two years) is long enough that a law must capture real
# dynamics (persistence decays badly by ~12-24mo) yet short enough that skill is
# clearly positive for a good law. STRIDE: spacing between successive origins (a
# new forecast launched every STRIDE months across the era). Both env-overridable.
HORIZON = int(os.environ.get("SR_HORIZON", "24"))
STRIDE = int(os.environ.get("SR_STRIDE", "2"))

# Largest lag (months of history) a law may reference. 264 = 22 years ≈ two solar
# cycles, plenty for any sensible recurrence; beyond it a "lag" is just a long
# pseudo-period fishing for phase alignment. ≥1 is required (x0 = "the value we are
# predicting" would be cheating).
MAX_LAG = int(os.environ.get("SR_MAX_LAG", "264"))

# Structural caps: beyond these it is not a compact symbolic expression.
MAX_NODES = int(os.environ.get("SR_MAX_NODES", "200"))
MAX_PARAMS = int(os.environ.get("SR_MAX_PARAMS", "30"))

# Non-finite predictions on the scoring split floor R^2 here (keeps it finite).
R2_FLOOR = float(os.environ.get("SR_R2_FLOOR", "-5.0"))

# Closed-loop forecasts can diverge if a recurrence is unstable; rather than let a
# single runaway origin floor the whole R² to R2_FLOOR (which would erase the
# signal that ranks decent laws), each forecast step is clamped to a generous
# PHYSICAL range before being fed back and scored. The true monthly sunspot number
# sits in ~0–400; these bounds are far outside it, so a sane law is never touched -
# only a diverging one is bounded into a (bad but finite) R². NaN/inf are mapped to
# the era mean / the bounds first. No gaming benefit (the clamp can only HURT a law
# whose honest predictions would have been better), just numerical robustness.
PRED_LO = float(os.environ.get("SR_PRED_LO", "-100.0"))
PRED_HI = float(os.environ.get("SR_PRED_HI", "1000.0"))

# A variable name is a lag reference iff it matches  x<positive-integer>.
_LAG_RE = re.compile(r"^x([1-9][0-9]*)$")


# ── where the held-out splits live (valid.csv / test.csv) ───────────────────────
# The held-out validation/test files must be readable when scoring a law, but must
# NOT sit inside the working directory a solver can browse - otherwise a solver
# could read valid.csv and fit its params straight to the answer. So the holdout
# directory is kept OUTSIDE the working directory and located at run time in this
# order:
#
#   1. $SR_HOLDOUT_DIR             - explicit override (any caller, any layout)
#   2. <work_dir-sibling>/_holdout - a staged sibling next to the working dir
#                                     (NOT under it, so a solver never sees it)
#   3. <work_dir>/holdout          - the in-repo source layout: works when scoring
#                                     runs from a normal checkout where the holdout
#                                     has not been split out.
#
# The first of these that contains valid.csv wins; if none do, the caller raises
# a clear "stage the holdout" error (never a silent zero).
HOLDOUT_DIRNAME = "holdout"        # in-repo source: <work_dir>/holdout/
STAGED_SUFFIX = "_holdout"          # staged sibling: <work_dir> + "_holdout"


def resolve_holdout_dir(task_dir: Path, *, require: str | None = None) -> Path:
    """Return the directory holding the held-out splits (valid/test).

    ``task_dir`` is the base directory the splits are resolved against (the
    dataset directory in a normal checkout). See the notes above for the
    resolution order. ``require`` (e.g. ``"valid.csv"``) names a file that must
    exist in the chosen dir; the first candidate that contains it wins. Returns
    the best candidate even on miss so the caller can raise an error naming a
    concrete path.
    """
    task_dir = Path(task_dir).resolve()
    candidates: list[Path] = []
    env = os.environ.get("SR_HOLDOUT_DIR")
    if env:
        candidates.append(Path(env))
    # Staged sibling next to the working dir (NOT under task_dir).
    candidates.append(task_dir.parent / (task_dir.name + STAGED_SUFFIX))
    # In-repo source layout: <task_dir>/holdout/ (local dev / unsplit checkout).
    candidates.append(task_dir / HOLDOUT_DIRNAME)

    if require:
        for cand in candidates:
            if (cand / require).is_file():
                return cand
    return candidates[0]


SAFE_FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan, "atan2": np.arctan2,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "log10": np.log10, "sqrt": np.sqrt,
    "abs": np.abs, "Abs": np.abs, "sign": np.sign,
    "floor": np.floor, "ceil": np.ceil,
}
SAFE_CONSTS = {"pi": math.pi, "e": math.e}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd,
)


@dataclass
class ParsedExpr:
    """A validated symbolic recurrence and everything derived from it."""
    source: str
    params: list[str]            # free-parameter names, sorted, deterministic
    lags: list[int]              # lag indices k for each x<k> used, sorted ascending
    n_nodes: int
    n_literals: int              # ALL numeric literals (structural + fitted)
    n_fitted_literals: int = 0   # literals billed as fitted d.o.f. (non-structural)
    code: object = field(repr=False, default=None)

    @property
    def max_lag(self) -> int:
        return max(self.lags) if self.lags else 0

    def complexity_penalty(self) -> float:
        # Fitted degrees of freedom - free params AND baked fitted literals - are
        # billed at the same LAMBDA_PARAM rate, so reporting a fitted constant as a
        # literal can no longer dodge the parameter penalty. Only the cheap
        # structural literals (small integers) remain at LAMBDA_LITERAL. Lag indices
        # are NOT charged (they live in the variable name x<k>, not as literals).
        n_structural = self.n_literals - self.n_fitted_literals
        fitted_dof = len(self.params) + self.n_fitted_literals
        return LAMBDA_PARAM * fitted_dof + LAMBDA_LITERAL * n_structural


def parse_expression(expr: str) -> ParsedExpr:
    """Parse + whitelist-validate ``expr``; return a :class:`ParsedExpr`.

    Raises ``ValueError`` on anything that is not a compact closed-form
    recurrence in the lag variables ``x1, x2, …``. Fails CLOSED: an unknown
    construct is rejected, never waved through - this is the line that keeps
    submissions symbolic. A name ``x<k>`` is a lag variable (1 ≤ k ≤ MAX_LAG);
    every other name is a free parameter.
    """
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("expression must be a non-empty string")
    if len(expr) > 5000:
        raise ValueError("expression too long (>5000 chars) - not a compact formula")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"expression is not valid arithmetic syntax: {e}") from e

    params: set[str] = set()
    lags: set[int] = set()
    called: set[str] = set()
    n_nodes = n_literals = n_fitted_literals = 0
    for node in ast.walk(tree):
        n_nodes += 1
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f"disallowed construct {type(node).__name__!r}: only arithmetic "
                f"(+ - * / ** %), unary ±, calls to {sorted(SAFE_FUNCS)}, the lag "
                f"variables x1,x2,…,x{MAX_LAG} (the sunspot number k months ago), "
                f"constants {sorted(SAFE_CONSTS)}, and free parameters are allowed "
                f"- no loops, imports, attributes, subscripts, comparisons, arrays, "
                f"or the calendar year."
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_FUNCS:
                fname = getattr(node.func, "id", type(node.func).__name__)
                raise ValueError(
                    f"call to {fname!r} not allowed; permitted: {sorted(SAFE_FUNCS)}"
                )
            if node.keywords:
                raise ValueError("function calls must not use keyword arguments")
            called.add(node.func.id)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError(f"only real numeric literals allowed, got {node.value!r}")
            n_literals += 1
            # A literal is "fitted" (billed at the param rate) unless it is a small
            # structural integer - the 2 in 2*pi, a harmonic multiplier. A float,
            # or a large integer, is a value tuned to the data, so it costs the same
            # as a free parameter. This is what makes baking a coefficient as a
            # literal no cheaper than naming it as a parameter.
            v = float(node.value)
            is_structural = v == int(v) and abs(v) <= MAX_STRUCTURAL_INT
            if not is_structural:
                n_fitted_literals += 1
        elif isinstance(node, ast.Name):
            if node.id in SAFE_CONSTS or node.id in SAFE_FUNCS:
                continue
            m = _LAG_RE.match(node.id)
            if m:
                k = int(m.group(1))
                if k < 1 or k > MAX_LAG:
                    raise ValueError(
                        f"lag variable {node.id!r} out of range: lag {k} must satisfy "
                        f"1 ≤ k ≤ MAX_LAG={MAX_LAG} (x1 = last month, … x{MAX_LAG} = "
                        f"{MAX_LAG} months ago)."
                    )
                lags.add(k)
            else:
                params.add(node.id)
    params -= called

    if n_nodes > MAX_NODES:
        raise ValueError(f"expression too large: {n_nodes} nodes > MAX_NODES={MAX_NODES}")
    if len(params) > MAX_PARAMS:
        raise ValueError(f"too many parameters: {len(params)} > MAX_PARAMS={MAX_PARAMS}")
    if not lags:
        raise ValueError(
            "expression references no lag variable (x1, x2, …): an autoregressive "
            "forecast law must depend on at least one past sunspot value."
        )

    return ParsedExpr(
        source=expr, params=sorted(params), lags=sorted(lags),
        n_nodes=n_nodes, n_literals=n_literals, n_fitted_literals=n_fitted_literals,
        code=compile(tree, "<sr-expression>", "eval"),
    )


def _check_params(parsed: ParsedExpr, params: dict) -> dict:
    """Validate that every free param has a finite real value; return float view."""
    missing = [p for p in parsed.params if p not in params]
    if missing:
        raise ValueError(
            f"missing value(s) for parameter(s) {missing} - every free name in "
            f"the expression must appear in 'params'"
        )
    out = {}
    for p in parsed.params:
        v = params[p]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"params[{p!r}] must be a finite real number, got {v!r}")
        out[p] = float(v)
    return out


def eval_step(parsed: ParsedExpr, lag_arrays: dict, params: dict) -> np.ndarray:
    """Evaluate the recurrence ONE step for many origins at once.

    ``lag_arrays`` maps each lag index k used by the law to an array of the
    ``s[t-k]`` values (one entry per origin/forecast being advanced). ``params``
    supplies every free parameter. Returns the array of next-step predictions.
    Runs the pre-compiled, whitelisted code with builtins stripped. Raises
    ``ValueError`` on a missing parameter or a missing lag array.
    """
    pvals = _check_params(parsed, params)
    ns: dict = {}
    ns.update(SAFE_CONSTS)
    ns.update(SAFE_FUNCS)
    ns.update(pvals)
    for k in parsed.lags:
        if k not in lag_arrays:
            raise ValueError(f"internal: lag array for x{k} not supplied")
        ns[f"x{k}"] = np.asarray(lag_arrays[k], dtype=float)
    out = eval(parsed.code, {"__builtins__": {}}, ns)  # noqa: S307 - whitelisted AST
    return np.asarray(out, dtype=float)


def _clamp(arr: np.ndarray, fill: float) -> np.ndarray:
    """Map NaN→fill, ±inf→bounds, then clip to [PRED_LO, PRED_HI]. See PRED_* note."""
    arr = np.nan_to_num(arr, nan=fill, posinf=PRED_HI, neginf=PRED_LO)
    return np.clip(arr, PRED_LO, PRED_HI)


def free_run_forecast(
    parsed: ParsedExpr, params: dict, series: np.ndarray,
    era_lo: int, era_hi: int,
    horizon: int = HORIZON, stride: int = STRIDE,
) -> float:
    """Rolling-origin, CLOSED-LOOP multi-step forecast R² over ``[era_lo, era_hi)``.

    ``series`` is the full chronological sunspot vector (train, or train+valid,
    …). For each origin ``o`` (``era_lo-1 ≤ o < era_hi-1``, stepped by ``stride``,
    and ``o ≥ max_lag`` so the deepest lag has true history), the law forecasts
    ``s[o+1], …, s[o+horizon]`` feeding its OWN predictions back in as the lag
    inputs; the lag ``x_k`` at forecast step ``h`` (predicting position ``o+h``,
    1-indexed) uses the true ``series[o+h-k]`` when ``o+h-k ≤ o`` and the law's
    earlier prediction otherwise. Every ``(prediction, truth)`` pair whose target
    ``o+h`` lies in ``[era_lo, era_hi)`` is pooled into a single R². Vectorized
    across all origins. Diverging recurrences are clamped (see ``_clamp``) so one
    runaway origin cannot floor the whole score. Returns ``R2_FLOOR`` if no valid
    pair exists.
    """
    series = np.asarray(series, dtype=float)
    n = series.size
    K = parsed.max_lag
    horizon = int(horizon)
    stride = max(1, int(stride))
    # Origins: last true month is `o`; first forecast target is o+1. Need o>=K so
    # the deepest lag at h=1 (position o+1-K) is real history, and o+1 inside era.
    o_first = max(K, era_lo - 1)
    o_last = min(n - 2, era_hi - 2)   # need at least one target o+1 < era_hi and < n
    if o_last < o_first:
        return R2_FLOOR
    origins = np.arange(o_first, o_last + 1, stride)
    m = origins.size
    if m == 0:
        return R2_FLOOR

    fill = float(series[max(0, era_lo - 1):era_hi].mean()) if era_hi > era_lo else 0.0
    # preds[i, h] = forecast for position origins[i] + (h+1), h = 0..horizon-1
    preds = np.empty((m, horizon), dtype=float)

    pred_acc: list[np.ndarray] = []
    true_acc: list[np.ndarray] = []
    for h in range(horizon):
        target_pos = origins + (h + 1)          # absolute index being predicted
        lag_arrays = {}
        for k in parsed.lags:
            src = target_pos - k                 # absolute index of s[t-k]
            col = np.empty(m, dtype=float)
            back = h + 1 - k                     # >=0 means it's a true-history index
            # source is true history when src <= origin (i.e. h+1-k <= 0)
            true_mask = (h + 1 - k) <= 0
            if true_mask:
                # all origins: src = origin+(h+1-k) <= origin, guaranteed >=0 since o>=K
                col = series[src]
            else:
                # all origins: src is a previously forecast step at column (back-1)
                col = preds[:, back - 1]
            lag_arrays[k] = col
        step = eval_step(parsed, lag_arrays, params)
        step = _clamp(step, fill)
        preds[:, h] = step
        # collect pairs whose target lies in [era_lo, era_hi) and within series
        valid = (target_pos >= era_lo) & (target_pos < era_hi) & (target_pos < n)
        if valid.any():
            pred_acc.append(step[valid])
            true_acc.append(series[target_pos[valid]])

    if not pred_acc:
        return R2_FLOOR
    y_pred = np.concatenate(pred_acc)
    y_true = np.concatenate(true_acc)
    return r2_score(y_true, y_pred)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination; non-finite predictions -> ``R2_FLOOR``."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return R2_FLOOR
    if not np.all(np.isfinite(y_pred)):
        return R2_FLOOR
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0:
        raise ValueError("scoring split has zero variance - cannot compute R^2")
    return max(R2_FLOOR, 1.0 - float(np.sum((y_true - y_pred) ** 2)) / ss_tot)


@dataclass
class Scored:
    parsed: ParsedExpr
    r2: float
    penalty: float
    score: float


def score_law(
    expr: str, params: dict, series: np.ndarray, era_lo: int, era_hi: int,
    horizon: int = HORIZON, stride: int = STRIDE,
) -> Scored:
    """Parse ``expr`` and return free-running forecast R² − penalty on one era.

    Single-era scorer: forecast R² minus penalty on one era, used to score a law
    on the hidden-test era or for a solver's local readouts. The blended
    objective uses :func:`blended_forecast_score` instead. Raises ``ValueError``
    on an invalid/oversized expression or a missing/bad parameter.
    """
    parsed = parse_expression(expr)
    r2 = free_run_forecast(parsed, params, series, era_lo, era_hi, horizon, stride)
    pen = parsed.complexity_penalty()
    return Scored(parsed=parsed, r2=r2, penalty=pen, score=r2 - pen)


@dataclass
class BlendedScored:
    """A law scored on BOTH its own (train) era and the held-out (valid) era."""
    parsed: ParsedExpr
    train_r2: float
    valid_r2: float
    penalty: float
    score: float
    w_train: float
    w_valid: float


def blended_forecast_score(
    expr: str, params: dict,
    series: np.ndarray, train_lo: int, train_hi: int, valid_hi: int,
    horizon: int = HORIZON, stride: int = STRIDE,
) -> BlendedScored:
    """The blended objective: ``W_TRAIN·R²_train_fc + W_VALID·R²_valid_fc − penalty``.

    ``series`` is the contiguous train+valid vector. The TRAIN term free-runs
    forecasts within ``[train_lo, train_hi)`` (the early era the law was fit on);
    the VALID term free-runs forecasts within ``[train_hi, valid_hi)`` (the held-out
    future era) - but with the FULL history up to each origin available, so a valid
    forecast launched near ``train_hi`` legitimately conditions on the late-train
    months. Both use the SAME submitted parameters (no refitting). Raises
    ``ValueError`` on an invalid/oversized expression or a missing/bad parameter,
    so a bad submission raises rather than silently scoring 0.
    """
    parsed = parse_expression(expr)
    train_r2 = free_run_forecast(parsed, params, series, train_lo, train_hi, horizon, stride)
    valid_r2 = free_run_forecast(parsed, params, series, train_hi, valid_hi, horizon, stride)
    pen = parsed.complexity_penalty()
    score = W_TRAIN * train_r2 + W_VALID * valid_r2 - pen
    return BlendedScored(
        parsed=parsed, train_r2=train_r2, valid_r2=valid_r2,
        penalty=pen, score=score, w_train=W_TRAIN, w_valid=W_VALID,
    )


# ── OPTIONAL convenience fitter (for solvers only - scoring never fits) ─────────
# Provided so a solver without scipy can still fit an AR law on the training data;
# a solver is free to ignore this and fit with its own method (e.g.
# scipy.optimize.curve_fit) - only the resulting param VALUES are submitted.
# This fits ONE-STEP-AHEAD (each month from its TRUE recent months) by least
# squares - fast, convex for a linear law, and a good starting point even for a
# nonlinear one. A solver should then check the multi-step forecast skill (what
# is actually scored) and refine if needed.
def _build_onestep_design(parsed: ParsedExpr, series: np.ndarray):
    """Rows (lag-arrays, target) for one-step fitting over the usable span."""
    series = np.asarray(series, dtype=float)
    n = series.size
    K = parsed.max_lag
    idx = np.arange(K, n)
    lag_arrays = {k: series[idx - k] for k in parsed.lags}
    target = series[idx]
    return lag_arrays, target


def fit_on_train(
    expr: str, series: np.ndarray, init: dict | None = None,
    starts: int = 16, iters: int = 120,
) -> dict:
    """Least-squares ONE-STEP fit of the law's free parameters. Returns ``params``.

    Deterministic multistart Levenberg–Marquardt in pure numpy (no scipy). Fits the
    next-month residuals using TRUE recent months (teacher forcing), which is a
    sound, cheap starting point; use it to seed your ``params`` and then verify /
    refine against the multi-step forecast skill that is actually scored.
    """
    parsed = parse_expression(expr)
    names = parsed.params
    p = len(names)
    if p == 0:
        return {}
    lag_arrays, target = _build_onestep_design(parsed, series)
    init = init or {}
    base = np.array([float(init.get(n, 1.0)) for n in names], dtype=float)

    def resid(theta):
        return eval_step(parsed, lag_arrays, dict(zip(names, theta))) - target

    def loss(theta):
        try:
            r = resid(theta)
        except ValueError:
            return math.inf
        return math.inf if not np.all(np.isfinite(r)) else 0.5 * float(r @ r)

    cand_starts = [base, np.full(p, 1.0), np.full(p, 0.1), np.full(p, -0.1),
                   np.full(p, 10.0)]
    rng = np.random.default_rng(0)
    while len(cand_starts) < starts:
        s = base * 10.0 ** rng.uniform(-2, 2, p) * rng.choice([-1.0, 1.0], p)
        cand_starts.append(s + rng.normal(0, 1, p))

    m = target.size
    best_theta, best = base, math.inf
    for s in cand_starts:
        theta = np.array(s, dtype=float)
        cur = loss(theta)
        if not math.isfinite(cur):
            continue
        mu = 1e-3
        for _ in range(iters):
            r0 = resid(theta)
            if not np.all(np.isfinite(r0)):
                break
            J = np.empty((m, p))
            ok = True
            for j in range(p):
                h = 1e-6 * max(1.0, abs(theta[j]))
                tp = theta.copy(); tp[j] += h
                tm = theta.copy(); tm[j] -= h
                fp = eval_step(parsed, lag_arrays, dict(zip(names, tp)))
                fm = eval_step(parsed, lag_arrays, dict(zip(names, tm)))
                if not (np.all(np.isfinite(fp)) and np.all(np.isfinite(fm))):
                    ok = False; break
                J[:, j] = (fp - fm) / (2 * h)
            if not ok:
                break
            JtJ, Jtr = J.T @ J, J.T @ r0
            dg = np.diag(np.maximum(np.diag(JtJ), 1e-12))
            stepped = False
            for _t in range(8):
                try:
                    d = np.linalg.solve(JtJ + mu * dg, -Jtr)
                except np.linalg.LinAlgError:
                    d, *_ = np.linalg.lstsq(JtJ + mu * dg, -Jtr, rcond=None)
                cl = loss(theta + d)
                if math.isfinite(cl) and cl < cur:
                    theta, cur, mu, stepped = theta + d, cl, max(mu / 3, 1e-12), True
                    break
                mu = min(mu * 3, 1e12)
            if not stepped:
                break
        if cur < best:
            best, best_theta = cur, theta
    return {n: float(v) for n, v in zip(names, best_theta)}
