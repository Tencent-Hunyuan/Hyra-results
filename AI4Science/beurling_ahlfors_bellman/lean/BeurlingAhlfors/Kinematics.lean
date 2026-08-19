import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.Normed.Group.Basic
import Mathlib.Tactic

/-!
# Elementary kinematics and the 2 by 2 drift criterion

This file formalizes the finite-dimensional core of Sections 1--2 of the
paper.  The variables are the real coordinates obtained after rotating the
chosen radial direction to the real axis.
-/

namespace BeurlingAhlfors

/-- Squared radial energy of the two input columns. -/
def radialSq (a₁ b₁ : ℝ) : ℝ := a₁ ^ 2 + b₁ ^ 2

/-- Squared tangential energy of the two input columns. -/
def tangentialSq (a₂ b₂ : ℝ) : ℝ := a₂ ^ 2 + b₂ ^ 2

/-- Squared energy of `a - i b` after rotating the chosen direction to the
real axis. -/
def transformedSq (a₁ a₂ b₁ b₂ : ℝ) : ℝ :=
  (a₁ + b₂) ^ 2 + (a₂ - b₁) ^ 2

/-- The algebraic reverse-triangle certificate underlying `|T-P| ≤ W`. -/
theorem kinematic_square_bounds (a₁ a₂ b₁ b₂ : ℝ) :
    (transformedSq a₁ a₂ b₁ b₂ - radialSq a₁ b₁ - tangentialSq a₂ b₂) ^ 2
      ≤ 4 * radialSq a₁ b₁ * tangentialSq a₂ b₂ := by
  have hid :
      transformedSq a₁ a₂ b₁ b₂ - radialSq a₁ b₁ - tangentialSq a₂ b₂ =
        2 * (a₁ * b₂ - a₂ * b₁) := by
    unfold transformedSq radialSq tangentialSq
    ring
  rw [hid]
  have hcs : (a₁ * b₂ - a₂ * b₁) ^ 2 ≤
      radialSq a₁ b₁ * tangentialSq a₂ b₂ := by
    unfold radialSq tangentialSq
    nlinarith [sq_nonneg (a₁ * a₂ + b₁ * b₂)]
  nlinarith

/-- Paper Lemma 2.1 after rotating the chosen radial direction to the
real axis.  The three displayed complex numbers are the vectors `u`, `v`, and
`u + v` from the paper's proof. -/
theorem kinematic_constraint
    {P W T a₁ a₂ b₁ b₂ : ℝ}
    (hP : P = ‖(a₁ : ℂ) - Complex.I * b₁‖)
    (hW : W = ‖(b₂ : ℂ) + Complex.I * a₂‖)
    (hT : T = ‖((a₁ + b₂ : ℝ) : ℂ) + Complex.I * (a₂ - b₁)‖) :
    |T - P| ≤ W := by
  let u : ℂ := (a₁ : ℂ) - Complex.I * b₁
  let v : ℂ := (b₂ : ℂ) + Complex.I * a₂
  have huv : u + v = ((a₁ + b₂ : ℝ) : ℂ) + Complex.I * (a₂ - b₁) := by
    apply Complex.ext
    · simp [u, v]
    · simp [u, v]
      ring
  rw [hP, hW, hT, ← huv]
  simpa [u, v] using abs_norm_sub_norm_le (u + v) u

/-- A symmetric real `2 × 2` quadratic form is nonpositive under the paper's
three scalar drift hypotheses. -/
theorem quadratic_form_nonpos
    {A B C x y : ℝ}
    (hA : A ≤ 0) (hC : C ≤ 0) (hdet : B ^ 2 ≤ A * C) :
    A * x ^ 2 + 2 * B * x * y + C * y ^ 2 ≤ 0 := by
  by_cases hA0 : A = 0
  · have hB : B = 0 := by
      rw [hA0, zero_mul] at hdet
      nlinarith [sq_nonneg B]
    rw [hA0, hB]
    nlinarith [sq_nonneg y]
  · have hAneg : A < 0 := lt_of_le_of_ne hA hA0
    have hsq : 0 ≤ (A * x + B * y) ^ 2 := sq_nonneg _
    nlinarith

/-- Abstract form of Proposition 2.3 after the kinematic estimate has reduced
all angular dependence to a `2 × 2` quadratic form. -/
theorem drift_reduction
    {φrr φrs φss φrOverR φsOverS P T W : ℝ}
    (hW : 0 ≤ W)
    (hφr : φrOverR ≤ 0)
    (hkin : |T - P| ≤ W)
    (hA : φrr + φrOverR ≤ 0)
    (hC : φss + φsOverS + φrOverR ≤ 0)
    (hdet : (|φrs| - φrOverR) ^ 2 ≤
      (φrr + φrOverR) * (φss + φsOverS + φrOverR)) :
    φrr * P ^ 2 + 2 * |φrs| * P * T + (φss + φsOverS) * T ^ 2
        + φrOverR * W ^ 2 ≤ 0 := by
  have hsq : (T - P) ^ 2 ≤ W ^ 2 := by
    have habssq : |T - P| ^ 2 ≤ W ^ 2 :=
      (sq_le_sq₀ (abs_nonneg (T - P)) hW).2 hkin
    simpa [sq_abs] using habssq
  have hreplace : φrOverR * W ^ 2 ≤ φrOverR * (T - P) ^ 2 :=
    mul_le_mul_of_nonpos_left hsq hφr
  have hquad := quadratic_form_nonpos hA hC hdet (x := P) (y := T)
  calc
    φrr * P ^ 2 + 2 * |φrs| * P * T + (φss + φsOverS) * T ^ 2
          + φrOverR * W ^ 2
        ≤ φrr * P ^ 2 + 2 * |φrs| * P * T + (φss + φsOverS) * T ^ 2
          + φrOverR * (T - P) ^ 2 := by linarith
    _ = (φrr + φrOverR) * P ^ 2
          + 2 * (|φrs| - φrOverR) * P * T
          + (φss + φsOverS + φrOverR) * T ^ 2 := by ring
    _ ≤ 0 := hquad

end BeurlingAhlfors
