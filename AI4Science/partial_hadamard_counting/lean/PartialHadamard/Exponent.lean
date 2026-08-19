import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.Analysis.Real.Pi.Bounds
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Order.ConditionallyCompleteLattice.Basic
import Mathlib.Order.Filter.AtTopBot.Basic

/-!
# Why the counting exponent is two

The analytic part of the counting argument proves a uniform asymptotic
formula for `N_{n,m}` under the *near-quadratic* hypothesis

`m / (n² log (2πm)) → ∞`.

This file formalizes the elementary bridge from a **power law** to that
hypothesis.  It is the only place where the numerical value of the
exponent enters, so it is exactly the step that decides the constant.

Write `m = 4t` and `t = N^u ω`, so that `ω = t / N^u` measures how far
into the regime `t / n^u → ∞` the pair `(n, t)` sits.  Say that `u`
*bridges* if some fixed multiple of `√ω` is a lower bound for
`m / (N² log (2πm))`, uniformly in `N`; then `ω → ∞` forces the
near-quadratic hypothesis, so the analytic theorem applies throughout the
regime `t / n^u → ∞`.

The results are:

* `bridge_bound`: a quantitative bridge for `u = 2 + 2δ`, `δ > 0`;
* `bridges_of_two_lt`: hence every `u > 2` bridges;
* `forces_of_bridges`: bridging implies the asymptotic implication `Forces`;
* `forces_iff_sequential`: `Forces` is *equivalent* to the sequential premise
  the paper's Corollary 2.3 uses, so it is not an artificial strengthening;
* `regimeRatio_le`: for `1 ≤ u ≤ 2` the ratio is at most `4ω / log N`;
* `not_forces_of_le_two`: no `u ≤ 2` forces, by an explicit divergent family;
* `sInf_admissibleExponents`: the infimum of admissible exponents is `2`;
* `two_notMem_admissibleExponents`: and it is not attained.

`Forces` is the property actually needed: `ω → ∞` drives the regime ratio to
infinity, uniformly in `N`.  `forces_iff_sequential` shows this ε-`W` form is
exactly the statement "along every sequence of pairs with `ω → ∞`, the ratio
tends to infinity", with `N` free to vary along the sequence, which is what
the paper's corollary asserts.  The uniformity is essential rather than
cosmetic: for a *fixed* `N` the ratio grows like `ω / log ω` for **every** `u`,
so a per-`N` reading would make every exponent admissible and leave the constant
with no content.

`Bridges` is a *quantitative* sufficient condition, a priori stronger than
`Forces` (it demands a rate).  On the admissible range the two turn out to be
coextensive (both hold exactly when `u > 2`), but nothing below relies on that,
and `Bridges` is used neither in the negative direction nor in the definition of
`admissibleExponents`.

Nothing here uses the analytic estimates; this is the deterministic
skeleton on which the value `2` rests.
-/

open Real Set Filter

namespace PartialHadamard

/-- The width `m = 4t` attached to `t = N^u ω`. -/
noncomputable def width (N u ω : ℝ) : ℝ := 4 * N ^ u * ω

/-- The quantity `m / (n² log (2πm))` whose divergence drives the analytic
estimate. -/
noncomputable def regimeRatio (N u ω : ℝ) : ℝ :=
  width N u ω / (N ^ 2 * Real.log (2 * π * width N u ω))

/-- `u` *bridges* if a power law with exponent `u` forces the
near-quadratic quantity to grow at least like `√ω`, uniformly in `N`.

A priori stronger than `Forces`, since it supplies a rate; used only as a
convenient sufficient condition. -/
def Bridges (u : ℝ) : Prop :=
  ∃ c > 0, ∀ N ω : ℝ, 2 ≤ N → 1 ≤ ω → c * Real.sqrt ω ≤ regimeRatio N u ω

/-- `u` *forces* the near-quadratic hypothesis if the regime ratio tends to
infinity, uniformly in `N`, as `ω → ∞`.

This is exactly the property the analytic theorem needs: along any sequence
with `t / N^u → ∞`, the quantity `m / (N² log 2πm)` must diverge, so that the
near-quadratic estimate eventually applies.  See `forces_iff_sequential`. -/
def Forces (u : ℝ) : Prop :=
  ∀ C > 0, ∃ W : ℝ, ∀ N ω : ℝ, 2 ≤ N → W ≤ ω → C ≤ regimeRatio N u ω

/-- The sequential form of the same property, matching the paper's
Corollary 2.3: along every sequence of pairs with `ω → ∞`, and with `N`
allowed to vary arbitrarily, the regime ratio tends to infinity. -/
def Sequential (u : ℝ) : Prop :=
  ∀ Nseq ωseq : ℕ → ℝ, (∀ i, 2 ≤ Nseq i) → (∀ i, 1 ≤ ωseq i) →
    Tendsto ωseq atTop atTop →
    Tendsto (fun i ↦ regimeRatio (Nseq i) u (ωseq i)) atTop atTop

/-- Exponents admissible for the problem: `α ≥ 1`, and the power law really
does force the near-quadratic hypothesis. -/
def admissibleExponents : Set ℝ := {u | 1 ≤ u ∧ Forces u}



section Elementary

theorem log_two_pi_width {N u ω : ℝ} (hN : 0 < N) (hω : 0 < ω) :
    Real.log (2 * π * width N u ω)
      = Real.log (8 * π) + u * Real.log N + Real.log ω := by
  have hrw : 2 * π * width N u ω = 8 * π * N ^ u * ω := by
    unfold width; ring
  have h8 : (8 : ℝ) * π ≠ 0 := by positivity
  have hNu : N ^ u ≠ 0 := ne_of_gt (Real.rpow_pos_of_pos hN u)
  rw [hrw, Real.log_mul (mul_ne_zero h8 hNu) (ne_of_gt hω),
    Real.log_mul h8 hNu, Real.log_rpow hN]

/-- `log (8π) ≤ 6`, via `log 8 = 3 log 2`, `log π ≤ π - 1` and `π ≤ 4`.
The true value is `3.2242…`; the slack is deliberate. -/
theorem log_eight_pi_le : Real.log (8 * π) ≤ 6 := by
  have h8 : Real.log 8 = 3 * Real.log 2 := by
    rw [show (8 : ℝ) = 2 ^ (3 : ℕ) by norm_num, Real.log_pow]
    push_cast
    ring
  have h2 : Real.log 2 < 0.6931471808 := Real.log_two_lt_d9
  have hpi : Real.log π ≤ π - 1 := Real.log_le_sub_one_of_pos Real.pi_pos
  have hpi4 : π ≤ 4 := Real.pi_le_four
  rw [Real.log_mul (by norm_num) (ne_of_gt Real.pi_pos), h8]
  linarith

theorem log_eight_pi_pos : 0 < Real.log (8 * π) := by
  apply Real.log_pos
  nlinarith [Real.pi_gt_three]

/-- Splitting `N^(2+2δ)` off its `N²` factor. -/
theorem rpow_two_add {N δ : ℝ} (hN : 0 < N) :
    N ^ (2 + 2 * δ) = N ^ 2 * N ^ (2 * δ) := by
  rw [Real.rpow_add hN]
  congr 1
  rw [show ((2 : ℝ)) = ((2 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]

end Elementary

section Bridge

variable {u : ℝ}

/-- The explicit constant produced by the bridge for `u = 2 + 2δ`. -/
noncomputable def bridgeConst (u δ : ℝ) : ℝ := 2 / (6 + 2 * u / δ)

theorem bridgeConst_pos {δ : ℝ} (hδ : 0 < δ) (hu : 0 < u) :
    0 < bridgeConst u δ := by
  have : 0 < 2 * u / δ := by positivity
  unfold bridgeConst
  positivity

/-- **Quantitative bridge.**  For `u = 2 + 2δ` with `δ > 0`, a power law
with exponent `u` forces `regimeRatio ≥ bridgeConst u δ · √ω`.

This is where the exponent `2` comes from: the numerator carries a factor
`N^(2δ)`, which must dominate the `N^(δ/2)` produced by bounding `log N`.
Any `δ > 0` succeeds, and `δ = 0` leaves nothing to dominate with. -/
theorem bridge_bound {δ : ℝ} (hδ : 0 < δ) (hu : u = 2 + 2 * δ)
    {N ω : ℝ} (hN : 2 ≤ N) (hω : 1 ≤ ω) :
    bridgeConst u δ * Real.sqrt ω ≤ regimeRatio N u ω := by
  have hN0 : (0 : ℝ) < N := by linarith
  have hN1 : (1 : ℝ) ≤ N := by linarith
  have hω0 : (0 : ℝ) < ω := by linarith
  have hu0 : 0 < u := by rw [hu]; linarith
  set A := N ^ (δ / 2) with hAdef
  set P := N ^ (2 * δ) with hPdef
  set W := Real.sqrt ω with hWdef
  set K := 6 + 2 * u / δ with hKdef
  have hK0 : 0 < K := by
    have : 0 < 2 * u / δ := by positivity
    rw [hKdef]; linarith
  have hA1 : 1 ≤ A := Real.one_le_rpow hN1 (by positivity)
  have hP1 : 1 ≤ P := Real.one_le_rpow hN1 (by positivity)
  have hW1 : 1 ≤ W := by
    rw [hWdef]
    simpa using Real.sqrt_le_sqrt hω
  have hWsq : W ^ 2 = ω := Real.sq_sqrt hω0.le
  -- `A ≤ P`, because `δ/2 ≤ 2δ`.
  have hAP : A ≤ P :=
    Real.rpow_le_rpow_of_exponent_le hN1 (by linarith)
  -- `W ≤ ω`, because `ω ≥ 1`.
  have hWω : W ≤ ω := by nlinarith [hWsq, hW1]
  -- Step 1: the logarithm is at most `K (A + W)`.
  have hlogN : Real.log N ≤ 2 / δ * A := by
    have h := Real.log_le_rpow_div hN0.le (show (0 : ℝ) < δ / 2 by positivity)
    calc Real.log N ≤ N ^ (δ / 2) / (δ / 2) := h
      _ = 2 / δ * A := by rw [hAdef]; field_simp
  have hlogω : Real.log ω ≤ 2 * W := by
    have h := Real.log_le_rpow_div hω0.le (show (0 : ℝ) < (1 : ℝ) / 2 by norm_num)
    calc Real.log ω ≤ ω ^ ((1 : ℝ) / 2) / ((1 : ℝ) / 2) := h
      _ = 2 * W := by rw [hWdef, Real.sqrt_eq_rpow]; ring
  have hL : Real.log (2 * π * width N u ω) ≤ K * (A + W) := by
    rw [log_two_pi_width hN0 hω0]
    have hmul : u * Real.log N ≤ 2 * u / δ * A := by
      calc u * Real.log N ≤ u * (2 / δ * A) :=
            mul_le_mul_of_nonneg_left hlogN hu0.le
        _ = 2 * u / δ * A := by ring
    have hKA : 0 ≤ 2 * u / δ := by positivity
    have h6 := log_eight_pi_le
    rw [hKdef]
    nlinarith [hA1, hW1, hKA]
  have hLpos : 0 < Real.log (2 * π * width N u ω) := by
    rw [log_two_pi_width hN0 hω0]
    have h1 : 0 ≤ u * Real.log N :=
      mul_nonneg hu0.le (Real.log_nonneg hN1)
    have h2 : 0 ≤ Real.log ω := Real.log_nonneg hω
    have := log_eight_pi_pos
    linarith
  -- Step 2: cancel `N²` and finish with `WA + ω ≤ 2Pω`.
  have hwidth : width N u ω = N ^ 2 * (4 * P * ω) := by
    rw [width, hu, rpow_two_add hN0, hPdef]; ring
  have hNsq : (0 : ℝ) < N ^ 2 := by positivity
  have hratio : regimeRatio N u ω = 4 * P * ω / Real.log (2 * π * width N u ω) := by
    rw [regimeRatio,
      div_eq_div_iff (ne_of_gt (mul_pos hNsq hLpos)) (ne_of_gt hLpos), hwidth]
    ring
  rw [hratio, bridgeConst]
  rw [div_mul_eq_mul_div, div_le_div_iff₀ (by linarith) hLpos]
  -- `2W · L ≤ 4Pω · K`, using `L ≤ K(A+W)` and `W(A+W) ≤ 2Pω`.
  have hkey : W * (A + W) ≤ 2 * (P * ω) := by
    have h1 : W * A ≤ ω * P := mul_le_mul hWω hAP (by linarith) hω0.le
    nlinarith [hWsq]
  calc 2 * W * Real.log (2 * π * width N u ω)
      ≤ 2 * W * (K * (A + W)) := by
        apply mul_le_mul_of_nonneg_left hL (by linarith)
    _ = K * (2 * (W * (A + W))) := by ring
    _ ≤ K * (2 * (2 * (P * ω))) := by
        apply mul_le_mul_of_nonneg_left _ hK0.le
        linarith
    _ = 4 * P * ω * K := by ring

/-- Every exponent strictly larger than two bridges. -/
theorem bridges_of_two_lt (hu : 2 < u) : Bridges u := by
  refine ⟨bridgeConst u ((u - 2) / 2), bridgeConst_pos (by linarith) (by linarith),
    fun N ω hN hω ↦ bridge_bound (by linarith) (by ring) hN hω⟩

/-- A quantitative bridge gives the asymptotic implication: if the ratio is at
least `c√ω`, then it exceeds any target once `ω ≥ (C/c)²`. -/
theorem forces_of_bridges (hb : Bridges u) : Forces u := by
  obtain ⟨c, hc, hbridge⟩ := hb
  intro C hC
  refine ⟨max 1 ((C / c) ^ 2), fun N ω hN hω ↦ ?_⟩
  have hω1 : (1 : ℝ) ≤ ω := le_trans (le_max_left _ _) hω
  have hsq : (C / c) ^ 2 ≤ ω := le_trans (le_max_right _ _) hω
  have hCc : C / c ≤ Real.sqrt ω := Real.le_sqrt_of_sq_le hsq
  calc C = c * (C / c) := by field_simp
    _ ≤ c * Real.sqrt ω := by exact mul_le_mul_of_nonneg_left hCc hc.le
    _ ≤ regimeRatio N u ω := hbridge N ω hN hω1

/-- Every exponent strictly larger than two forces. -/
theorem forces_of_two_lt (hu : 2 < u) : Forces u :=
  forces_of_bridges (bridges_of_two_lt hu)

/-- **`Forces` is the sequential premise, not a strengthening of it.**

The `ε`-`W` form and the sequential form agree. The forward direction is
routine; the reverse is the informative one: if `Forces` fails, the witnesses
assemble into a sequence with `ω → ∞` along which the ratio stays bounded,
contradicting `Sequential`. Hence the uniformity in `N` built into `Forces`
costs nothing: it is what "uniformly throughout the regime" already means. -/
theorem forces_iff_sequential : Forces u ↔ Sequential u := by
  constructor
  · -- Forward: given `C`, the threshold `W` eventually applies.
    intro hf Nseq ωseq hN hω hdiv
    rw [tendsto_atTop]
    intro C
    obtain ⟨W, hW⟩ := hf (max C 1) (lt_of_lt_of_le one_pos (le_max_right _ _))
    obtain ⟨i₀, hi₀⟩ := (tendsto_atTop.mp hdiv W).exists_forall_of_atTop
    filter_upwards [eventually_ge_atTop i₀] with i hi
    exact le_trans (le_max_left _ _) (hW _ _ (hN i) (hi₀ i hi))
  · -- Reverse: negate `Forces` and build a bad sequence.
    intro hs
    by_contra hf
    rw [Forces] at hf
    push Not at hf
    obtain ⟨C, hC, hbad⟩ := hf
    -- For each `j`, the threshold `j+1` fails: some pair has `ω ≥ j+1` yet
    -- ratio `< C`.  Indexing by `j+1` makes `ω ≥ 1` automatic.
    choose! Nf ωf hNf hωf hlt using fun j : ℕ ↦ hbad ((j : ℝ) + 1)
    have hωge : ∀ j : ℕ, (j : ℝ) + 1 ≤ ωf j := fun j ↦ hωf j
    have hω1 : ∀ j : ℕ, (1 : ℝ) ≤ ωf j := fun j ↦
      le_trans (by linarith [Nat.cast_nonneg (α := ℝ) j]) (hωge j)
    have hdiv : Tendsto ωf atTop atTop := by
      apply tendsto_atTop_mono hωge
      exact tendsto_atTop_add_const_right _ 1 tendsto_natCast_atTop_atTop
    obtain ⟨j, hj⟩ := (tendsto_atTop.mp (hs Nf ωf hNf hω1 hdiv) C).exists
    exact absurd hj (not_le.mpr (hlt j))

/-- **The key upper bound for `1 ≤ u ≤ 2`.**  When the power is at most `2`, the
factor `N^u` cannot beat the `N²` in the denominator, so the ratio is at most
`4ω / log N`, no matter how large `ω` is. -/
theorem regimeRatio_le (h1 : 1 ≤ u) (h2 : u ≤ 2) {N ω : ℝ} (hN : 2 ≤ N)
    (hω : 1 ≤ ω) : regimeRatio N u ω ≤ 4 * ω / Real.log N := by
  have hN0 : (0 : ℝ) < N := by linarith
  have hN1 : (1 : ℝ) < N := by linarith
  have hω0 : (0 : ℝ) < ω := by linarith
  have hlogN : 0 < Real.log N := Real.log_pos hN1
  have hNsq : (0 : ℝ) < N ^ 2 := by positivity
  -- `N^u ≤ N²`.
  have hNu : N ^ u ≤ N ^ 2 := by
    calc N ^ u ≤ N ^ (2 : ℝ) := Real.rpow_le_rpow_of_exponent_le hN1.le h2
      _ = N ^ 2 := by
          rw [show ((2 : ℝ)) = ((2 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  -- `log N ≤ log (2πm)`, using `u ≥ 1` and `ω ≥ 1`.
  have hLge : Real.log N ≤ Real.log (2 * π * width N u ω) := by
    rw [log_two_pi_width hN0 hω0]
    have ha : Real.log N ≤ u * Real.log N := by
      nlinarith [hlogN]
    have hb : 0 ≤ Real.log ω := Real.log_nonneg hω
    have := log_eight_pi_pos
    linarith
  have hLpos : 0 < Real.log (2 * π * width N u ω) := lt_of_lt_of_le hlogN hLge
  rw [regimeRatio, div_le_div_iff₀ (by positivity) hlogN]
  have hwidth : width N u ω = 4 * N ^ u * ω := rfl
  calc width N u ω * Real.log N
      = 4 * ω * Real.log N * N ^ u := by rw [hwidth]; ring
    _ ≤ 4 * ω * Real.log N * N ^ 2 := by
        exact mul_le_mul_of_nonneg_left hNu (by positivity)
    _ = 4 * ω * N ^ 2 * Real.log N := by ring
    _ ≤ 4 * ω * N ^ 2 * Real.log (2 * π * width N u ω) := by
        exact mul_le_mul_of_nonneg_left hLge (by positivity)
    _ = 4 * ω * (N ^ 2 * Real.log (2 * π * width N u ω)) := by ring

/-- **No exponent at most two forces the near-quadratic hypothesis.**

The witness is a genuinely divergent family: taking `ω = j` and
`N = exp (8 j)` gives `t / N^u = ω → ∞`, yet by `regimeRatio_le` the regime
ratio is at most `4ω / log N = 1/2`.  So the hypothesis fails along a sequence
that satisfies the power law, which is exactly what `Forces` forbids.

This is the honest form of the negative direction: fixing `ω` would refute a
*uniform* lower bound while saying nothing about the `ω → ∞` regime. -/
theorem not_forces_of_le_two (h1 : 1 ≤ u) (h2 : u ≤ 2) : ¬ Forces u := by
  intro hforces
  obtain ⟨W, hW⟩ := hforces 1 one_pos
  -- Choose `ω ≥ max 1 W` and then `N` exponentially large in `ω`.
  set ω := max 1 W with hωdef
  have hω1 : (1 : ℝ) ≤ ω := le_max_left _ _
  have hωW : W ≤ ω := le_max_right _ _
  have hω0 : (0 : ℝ) < ω := by linarith
  set N := Real.exp (8 * ω) with hNdef
  have hN2 : (2 : ℝ) ≤ N := by
    rw [hNdef]
    calc (2 : ℝ) ≤ Real.exp 1 := by
          have := Real.exp_one_gt_d9
          linarith
      _ ≤ Real.exp (8 * ω) := Real.exp_le_exp.mpr (by linarith)
  have hlogN : Real.log N = 8 * ω := by rw [hNdef, Real.log_exp]
  -- `regimeRatio ≤ 4ω / log N = 1/2 < 1`.
  have hub := regimeRatio_le h1 h2 hN2 hω1
  have hhalf : 4 * ω / Real.log N = 1 / 2 := by
    rw [hlogN]
    field_simp
    norm_num
  have hlow := hW N ω hN2 hωW
  rw [hhalf] at hub
  linarith

/-- The regime ratio attached to an **integer** pair `(n, t)`, i.e.
`m / (n² log 2πm)` with `m = 4t`.  No `ω` appears: this is the raw quantity the
analytic theorem is about. -/
noncomputable def intRatio (n t : ℕ) : ℝ :=
  (4 * t : ℝ) / ((n : ℝ) ^ 2 * Real.log (2 * π * (4 * t)))

/-- **The negative direction on the original integer domain.**

For `1 ≤ u ≤ 2` there is an explicit family of *integer* pairs `(n_j, t_j)`,
namely `n_j = ⌈e^(16j)⌉` and `t_j = ⌈j · n_j^u⌉`, with

* `n_j ≥ 2` and `t_j ≥ 1`, so the pairs are admissible for the problem;
* `t_j / n_j^u ≥ j`, so the power law with exponent `u` holds and `→ ∞`;
* `intRatio n_j t_j ≤ 1/2`, so the near-quadratic quantity stays bounded.

Hence no `u ≤ 2` forces the near-quadratic hypothesis even when the parameters
are restricted to integers, which is the domain of the counting problem.  This
closes the gap between the real-relaxed `Forces` predicate and the integer
statement. -/
theorem exists_int_witness_of_le_two (h1 : 1 ≤ u) (h2 : u ≤ 2) (j : ℕ) (hj : 1 ≤ j) :
    ∃ n t : ℕ, 2 ≤ n ∧ 1 ≤ t ∧ (j : ℝ) ≤ (t : ℝ) / (n : ℝ) ^ u ∧
      intRatio n t ≤ 1 / 2 := by
  have hj1 : (1 : ℝ) ≤ (j : ℝ) := by exact_mod_cast hj
  -- `n = ⌈e^(16j)⌉`.
  obtain ⟨n, hn⟩ : ∃ n : ℕ, Real.exp (16 * (j : ℝ)) ≤ (n : ℝ) ∧ (n : ℝ) ≤ Real.exp (16 * (j : ℝ)) + 1 :=
    ⟨⌈Real.exp (16 * (j : ℝ))⌉₊, Nat.le_ceil _, by
      have := Nat.ceil_lt_add_one (Real.exp_pos (16 * (j : ℝ))).le
      linarith⟩
  have hexp2 : (2 : ℝ) ≤ Real.exp (16 * (j : ℝ)) := by
    calc (2 : ℝ) ≤ Real.exp 1 := by have := Real.exp_one_gt_d9; linarith
      _ ≤ Real.exp (16 * (j : ℝ)) := Real.exp_le_exp.mpr (by linarith)
  have hN2 : (2 : ℝ) ≤ (n : ℝ) := le_trans hexp2 hn.1
  have hN0 : (0 : ℝ) < (n : ℝ) := by linarith
  have hn2 : 2 ≤ n := by exact_mod_cast hN2
  have hlogN : 16 * (j : ℝ) ≤ Real.log (n : ℝ) := by
    calc 16 * (j : ℝ) = Real.log (Real.exp (16 * (j : ℝ))) := (Real.log_exp _).symm
      _ ≤ Real.log (n : ℝ) := Real.log_le_log (Real.exp_pos _) hn.1
  -- `n^u ≤ n²`.
  have hNu : (n : ℝ) ^ u ≤ (n : ℝ) ^ 2 := by
    calc (n : ℝ) ^ u ≤ (n : ℝ) ^ (2 : ℝ) :=
          Real.rpow_le_rpow_of_exponent_le (by linarith) h2
      _ = (n : ℝ) ^ 2 := by
          rw [show ((2 : ℝ)) = ((2 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  have hNu1 : (1 : ℝ) ≤ (n : ℝ) ^ u :=
    Real.one_le_rpow (by linarith) (by linarith)
  -- `t = ⌈j · n^u⌉`.
  obtain ⟨t, ht⟩ : ∃ t : ℕ, (j : ℝ) * (n : ℝ) ^ u ≤ (t : ℝ) ∧
      (t : ℝ) ≤ (j : ℝ) * (n : ℝ) ^ u + 1 :=
    ⟨⌈(j : ℝ) * (n : ℝ) ^ u⌉₊, Nat.le_ceil _, by
      have := Nat.ceil_lt_add_one (by positivity : (0:ℝ) ≤ (j : ℝ) * (n : ℝ) ^ u)
      linarith⟩
  have htpos : (1 : ℝ) ≤ (t : ℝ) := le_trans (by nlinarith) ht.1
  have ht1 : 1 ≤ t := by exact_mod_cast htpos
  refine ⟨n, t, hn2, ht1, ?_, ?_⟩
  · -- `t / n^u ≥ j`.
    rw [le_div_iff₀ (by positivity : (0:ℝ) < (n : ℝ) ^ u)]
    exact ht.1
  · -- `intRatio ≤ 1/2`, via `4t ≤ 8 j n²` and `log 2π(4t) ≥ log n ≥ 16j`.
    have hlogpos : 0 < Real.log (2 * π * (4 * (t : ℝ))) := by
      apply Real.log_pos
      nlinarith [Real.pi_gt_three, htpos]
    have hlogge : Real.log (n : ℝ) ≤ Real.log (2 * π * (4 * (t : ℝ))) := by
      apply Real.log_le_log hN0
      -- `n ≤ n^u ≤ j n^u ≤ t`, and `2π·4 > 1`.
      have hn_le : (n : ℝ) ≤ (n : ℝ) ^ u := by
        calc (n : ℝ) = (n : ℝ) ^ (1 : ℝ) := (Real.rpow_one _).symm
          _ ≤ (n : ℝ) ^ u := Real.rpow_le_rpow_of_exponent_le (by linarith) h1
      have hjn : (n : ℝ) ^ u ≤ (j : ℝ) * (n : ℝ) ^ u := by nlinarith
      nlinarith [Real.pi_gt_three, ht.1]
    have hnum : 4 * (t : ℝ) ≤ 8 * (j : ℝ) * (n : ℝ) ^ 2 := by
      have h4 : (4 : ℝ) ≤ 4 * (j : ℝ) * (n : ℝ) ^ 2 := by nlinarith
      nlinarith [ht.2, hNu]
    rw [intRatio, div_le_div_iff₀ (by positivity) (by norm_num : (0:ℝ) < 2)]
    calc 4 * (t : ℝ) * 2 ≤ (8 * (j : ℝ) * (n : ℝ) ^ 2) * 2 := by linarith
      _ = (n : ℝ) ^ 2 * (16 * (j : ℝ)) := by ring
      _ ≤ (n : ℝ) ^ 2 * Real.log (2 * π * (4 * (t : ℝ))) :=
          mul_le_mul_of_nonneg_left (le_trans hlogN hlogge) (by positivity)
      _ = 1 * ((n : ℝ) ^ 2 * Real.log (2 * π * (4 * (t : ℝ)))) := by ring

end Bridge

section Infimum

theorem two_lt_of_mem {u : ℝ} (hu : u ∈ admissibleExponents) : 2 < u := by
  by_contra h
  exact not_forces_of_le_two hu.1 (not_lt.mp h) hu.2

theorem mem_admissibleExponents_of_two_lt {u : ℝ} (hu : 2 < u) :
    u ∈ admissibleExponents :=
  ⟨by linarith, forces_of_two_lt hu⟩

theorem admissibleExponents_nonempty : admissibleExponents.Nonempty :=
  ⟨3, mem_admissibleExponents_of_two_lt (by norm_num)⟩

/-- **Two is not attained.** -/
theorem two_notMem_admissibleExponents : 2 ∉ admissibleExponents := by
  intro h
  exact absurd (two_lt_of_mem h) (lt_irrefl 2)

/-- **The counting exponent is two.**  The infimum of the exponents for which
a power law forces the near-quadratic hypothesis is exactly `2`.

Both directions are genuine statements about `Forces`, the asymptotic notion:
every `u > 2` forces (via the quantitative `bridge_bound`), and no `u ≤ 2`
forces (via an explicit family with `ω → ∞` along which the ratio stays
bounded). -/
theorem sInf_admissibleExponents : sInf admissibleExponents = 2 := by
  have hlb : ∀ u ∈ admissibleExponents, (2 : ℝ) ≤ u :=
    fun u hu ↦ (two_lt_of_mem hu).le
  apply le_antisymm
  · apply le_of_forall_gt_imp_ge_of_dense
    intro q hq
    exact csInf_le ⟨2, hlb⟩ (mem_admissibleExponents_of_two_lt hq)
  · exact le_csInf admissibleExponents_nonempty hlb

end Infimum

end PartialHadamard
