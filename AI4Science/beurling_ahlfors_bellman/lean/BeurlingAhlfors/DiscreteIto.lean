import Mathlib.Analysis.Calculus.TaylorIntegral
import Mathlib.Tactic

/-!
# A deterministic multidimensional Itô skeleton

This file formalizes the exact pathwise Taylor/telescoping identity consumed by
any discrete Gaussian approximation of the Bellman argument.  Unlike a theorem
that merely assumes a stochastic Bellman principle, the remainder is explicit
and is identified with the integral third-derivative remainder from mathlib's
higher-dimensional Taylor theorem.

The probabilistic estimates showing that the summed remainder tends to zero,
and the construction of the relevant Gaussian/Brownian process, remain outside
this file.
-/

open scoped BigOperators

namespace BeurlingAhlfors

section DeterministicTaylor

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The exact second-order Taylor remainder for one increment. -/
noncomputable def secondOrderRemainder (u : E → ℝ) (x h : E) : ℝ :=
  u (x + h) - u x - fderiv ℝ u x h
    - (2 : ℝ)⁻¹ * iteratedFDeriv ℝ 2 u x (fun _ ↦ h)

/-- The second-order Taylor remainder equals the integral third-derivative
remainder along the segment from `x` to `x + h`. -/
theorem secondOrderRemainder_eq_integral
    {u : E → ℝ} {x h : E}
    (hu : ∀ t ∈ Set.Icc (0 : ℝ) 1, ContDiffAt ℝ 3 u (x + t • h)) :
    secondOrderRemainder u x h =
      ((2 : ℝ)⁻¹ • ∫ t in (0 : ℝ)..1,
        (1 - t) ^ 2 • iteratedFDeriv ℝ 3 u (x + t • h) (fun _ ↦ h)) := by
  have hTaylor :=
    map_add_eq_sum_add_integral_iteratedFDeriv (n := 2) (f := u) (x := x) (y := h) hu
  unfold secondOrderRemainder
  rw [hTaylor]
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, zero_add, Nat.factorial_zero,
    Nat.factorial_one, Nat.factorial_two, Nat.cast_one, Nat.cast_ofNat, inv_one,
    iteratedFDeriv_zero_apply, iteratedFDeriv_one_apply, one_smul]
  ring

/-- A uniform operator-norm bound on the third derivative gives the expected
cubic bound on the one-step Taylor remainder. -/
theorem norm_secondOrderRemainder_le
    {u : E → ℝ} {x h : E} {C : ℝ}
    (hu : ∀ t ∈ Set.Icc (0 : ℝ) 1, ContDiffAt ℝ 3 u (x + t • h))
    (hC : ∀ t ∈ Set.Icc (0 : ℝ) 1, ‖iteratedFDeriv ℝ 3 u (x + t • h)‖ ≤ C) :
    ‖secondOrderRemainder u x h‖ ≤ (2 : ℝ)⁻¹ * C * ‖h‖ ^ 3 := by
  rw [secondOrderRemainder_eq_integral hu, norm_smul, Real.norm_eq_abs]
  have hIntegral :
      ‖∫ t in (0 : ℝ)..1,
          (1 - t) ^ 2 • iteratedFDeriv ℝ 3 u (x + t • h) (fun _ ↦ h)‖ ≤
        C * ‖h‖ ^ 3 := by
    have hBound : ∀ t ∈ Set.uIoc (0 : ℝ) 1,
        ‖(1 - t) ^ 2 • iteratedFDeriv ℝ 3 u (x + t • h) (fun _ ↦ h)‖ ≤
          C * ‖h‖ ^ 3 := by
      intro t ht
      have htIcc : t ∈ Set.Icc (0 : ℝ) 1 := by
        rw [Set.uIoc_of_le zero_le_one] at ht
        exact ⟨ht.1.le, ht.2⟩
      calc
        ‖(1 - t) ^ 2 • iteratedFDeriv ℝ 3 u (x + t • h) (fun _ ↦ h)‖
            = |1 - t| ^ 2 *
                ‖iteratedFDeriv ℝ 3 u (x + t • h) (fun _ ↦ h)‖ := by
              rw [norm_smul, Real.norm_eq_abs, abs_pow]
        _ ≤ |1 - t| ^ 2 *
              (‖iteratedFDeriv ℝ 3 u (x + t • h)‖ * ‖h‖ ^ 3) := by
            gcongr
            simpa [Fin.prod_const] using
              (ContinuousMultilinearMap.le_opNorm
                (iteratedFDeriv ℝ 3 u (x + t • h)) (fun _ ↦ h))
        _ ≤ 1 * (C * ‖h‖ ^ 3) := by
            have habs : |1 - t| ≤ 1 := by
              rw [abs_of_nonneg (by linarith [htIcc.2])]
              linarith [htIcc.1]
            gcongr
            · simpa using (sq_le_sq₀ (abs_nonneg (1 - t)) zero_le_one).2 habs
            · exact hC t htIcc
        _ = C * ‖h‖ ^ 3 := one_mul _
    simpa using intervalIntegral.norm_integral_le_of_norm_le_const hBound
  have hhalf : |(2 : ℝ)⁻¹| = (2 : ℝ)⁻¹ := by norm_num
  rw [hhalf]
  nlinarith [norm_nonneg (∫ t in (0 : ℝ)..1,
    (1 - t) ^ 2 • iteratedFDeriv ℝ 3 u (x + t • h) (fun _ ↦ h))]

/-- Exact discrete second-order Itô/Taylor identity on an arbitrary finite
path.  No probability assumptions are needed: this is a telescoping identity. -/
theorem discrete_second_order_taylor
    (N : ℕ) (u : E → ℝ) (X : ℕ → E) :
    u (X N) - u (X 0) =
      (∑ k ∈ Finset.range N, fderiv ℝ u (X k) (X (k + 1) - X k))
      + (2 : ℝ)⁻¹ *
          (∑ k ∈ Finset.range N,
            iteratedFDeriv ℝ 2 u (X k) (fun _ ↦ X (k + 1) - X k))
      + ∑ k ∈ Finset.range N,
          secondOrderRemainder u (X k) (X (k + 1) - X k) := by
  have htele :
      u (X N) - u (X 0) =
        ∑ k ∈ Finset.range N, (u (X (k + 1)) - u (X k)) :=
    (Finset.sum_range_sub (fun n ↦ u (X n)) N).symm
  rw [htele]
  have hstep : ∀ k,
      u (X (k + 1)) - u (X k) =
        fderiv ℝ u (X k) (X (k + 1) - X k)
        + (2 : ℝ)⁻¹ * iteratedFDeriv ℝ 2 u (X k) (fun _ ↦ X (k + 1) - X k)
        + secondOrderRemainder u (X k) (X (k + 1) - X k) := by
    intro k
    unfold secondOrderRemainder
    have hX : X k + (X (k + 1) - X k) = X (k + 1) := by
      abel
    rw [hX]
    ring
  rw [Finset.sum_congr rfl (fun k _ ↦ hstep k)]
  simp only [Finset.sum_add_distrib]
  rw [← Finset.mul_sum]

/-- Summing the one-step cubic remainder bound along a finite path. -/
theorem norm_sum_secondOrderRemainder_le
    (N : ℕ) {u : E → ℝ} (X : ℕ → E) {C : ℝ}
    (hu : ∀ k ∈ Finset.range N, ∀ t ∈ Set.Icc (0 : ℝ) 1,
      ContDiffAt ℝ 3 u (X k + t • (X (k + 1) - X k)))
    (hC : ∀ k ∈ Finset.range N, ∀ t ∈ Set.Icc (0 : ℝ) 1,
      ‖iteratedFDeriv ℝ 3 u (X k + t • (X (k + 1) - X k))‖ ≤ C) :
    ‖∑ k ∈ Finset.range N,
        secondOrderRemainder u (X k) (X (k + 1) - X k)‖ ≤
      (2 : ℝ)⁻¹ * C *
        ∑ k ∈ Finset.range N, ‖X (k + 1) - X k‖ ^ 3 := by
  calc
    ‖∑ k ∈ Finset.range N,
        secondOrderRemainder u (X k) (X (k + 1) - X k)‖
        ≤ ∑ k ∈ Finset.range N,
            ‖secondOrderRemainder u (X k) (X (k + 1) - X k)‖ :=
      norm_sum_le _ _
    _ ≤ ∑ k ∈ Finset.range N,
          (2 : ℝ)⁻¹ * C * ‖X (k + 1) - X k‖ ^ 3 := by
        apply Finset.sum_le_sum
        intro k hk
        exact norm_secondOrderRemainder_le (hu k hk) (hC k hk)
    _ = (2 : ℝ)⁻¹ * C *
          ∑ k ∈ Finset.range N, ‖X (k + 1) - X k‖ ^ 3 := by
        rw [Finset.mul_sum]

end DeterministicTaylor

section ComplexColumns

/-- A complex increment represented by its two real coordinates. -/
def complexIncrement (a b : ℝ) : ℂ := a + Complex.I * b

/-- The transformed increment appearing in the Beurling--Ahlfors matrix. -/
def transformedIncrement (a b : ℝ) : ℂ := complexIncrement a b - Complex.I * complexIncrement b (-a)

/-- Coordinate expansion of the basic complex increment. -/
theorem complexIncrement_re_im (a b : ℝ) :
    (complexIncrement a b).re = a ∧ (complexIncrement a b).im = b := by
  simp [complexIncrement]

/-- The two real coordinates of a complex increment have exactly its squared
Euclidean energy. -/
theorem norm_complexIncrement_sq (a b : ℝ) :
    ‖complexIncrement a b‖ ^ 2 = a ^ 2 + b ^ 2 := by
  rw [Complex.sq_norm, Complex.normSq_apply]
  simp [complexIncrement]
  ring

end ComplexColumns

end BeurlingAhlfors
