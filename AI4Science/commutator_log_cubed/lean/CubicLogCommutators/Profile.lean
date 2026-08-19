import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic

/-!
# Quadratic profile and supersolution arithmetic

This file checks the exact second differences and the numerical inequalities
in Lemmas 6.1 and 6.2 at `δ = 1 / (400 n³)`.
-/

set_option autoImplicit false

namespace CubicLogCommutators

noncomputable section

/-- The square of the profile from equations (6.2)--(6.3). -/
def profileSq (N δ i : ℝ) : ℝ :=
  144 * δ ^ 2 * i * (2 * N - i)

/-- The profile itself. -/
def profile (N δ i : ℝ) : ℝ :=
  Real.sqrt (profileSq N δ i)

/-- Equation (6.9): the interior concavity defect is exactly constant. -/
theorem profileSq_interior_defect (N δ i : ℝ) :
    profileSq N δ i -
        (profileSq N δ (i - 1) + profileSq N δ (i + 1)) / 2 =
      144 * δ ^ 2 := by
  unfold profileSq
  ring

/-- Equation (6.10): the zero boundary creates the enlarged endpoint defect. -/
theorem profileSq_endpoint_defect (N δ : ℝ) :
    profileSq N δ N - profileSq N δ (N - 1) / 2 =
      72 * δ ^ 2 * (N ^ 2 + 1) := by
  unfold profileSq
  ring

/-- The endpoint calculation in Lemma 6.2. -/
theorem endpoint_supersolution_arithmetic
    {N δ : ℝ} (hN : 1 ≤ N) (hδ : δ = 1 / (400 * N ^ 3)) :
    4 * (12 * δ * N) * (δ * N + 72 * δ ^ 2 * N ^ 2) ≤
      72 * δ ^ 2 * (N ^ 2 + 1) := by
  have hN0 : 0 < N := lt_of_lt_of_le zero_lt_one hN
  rw [hδ]
  field_simp
  nlinarith [sq_nonneg (N ^ 2 - 1)]

/-- The interior calculation in Lemma 6.2, including the decimal `201.3`. -/
theorem interior_supersolution_arithmetic
    {N δ i : ℝ} (hN : 1 ≤ N) (hi0 : 0 ≤ i) (hiN : i ≤ N)
    (hδ : δ = 1 / (400 * N ^ 3)) :
    4 * (17 * δ * Real.sqrt (N * i)) *
        (2013 / 10 * δ ^ 2 * N * Real.sqrt (N * i)) ≤
      144 * δ ^ 2 := by
  have hN0 : 0 < N := lt_of_lt_of_le zero_lt_one hN
  have hNi : 0 ≤ N * i := mul_nonneg hN0.le hi0
  have hsqrt : (Real.sqrt (N * i)) ^ 2 = N * i := Real.sq_sqrt hNi
  have hNN : N * i ≤ N ^ 2 := by nlinarith
  rw [hδ]
  field_simp
  nlinarith

end

end CubicLogCommutators
