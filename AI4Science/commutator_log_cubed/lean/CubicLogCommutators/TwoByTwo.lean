import Mathlib.Analysis.CStarAlgebra.Matrix
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic

/-!
# The sharp two-by-two estimate

This file formalizes Lemma 5.2 of the paper for the genuine Euclidean operator
norm on real `2 × 2` matrices. It also proves the defect form used by the
concavity criterion.
-/

open scoped Matrix Matrix.Norms.L2Operator

set_option autoImplicit false

namespace CubicLogCommutators

noncomputable section

/-- The matrix `[[a,b],[c,a]]` from Lemma 5.2. -/
def twoByTwo (a b c : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![a, b; c, a]

private theorem twoByTwo_pointwise_bound
    {a b c : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
    (hbc : b ^ 2 + c ^ 2 ≤ 2 * a ^ 2)
    (x : EuclideanSpace ℝ (Fin 2)) :
    ‖Matrix.toEuclideanCLM (n := Fin 2) (𝕜 := ℝ) (twoByTwo a b c) x‖ ≤
      (a + Real.sqrt ((b ^ 2 + c ^ 2) / 2)) * ‖x‖ := by
  let S : ℝ := b ^ 2 + c ^ 2
  let W : ℝ := (b + c) ^ 2
  let e : ℝ := (b - c) ^ 2
  let r : ℝ := Real.sqrt (S / 2)
  have hS0 : 0 ≤ S := by
    dsimp [S]
    positivity
  have hr0 : 0 ≤ r := by
    dsimp [r]
    positivity
  have hrSq : r ^ 2 = S / 2 := by
    dsimp [r]
    exact Real.sq_sqrt (by positivity)
  have hra : r ≤ a := by
    rw [Real.sqrt_le_iff]
    exact ⟨ha, by dsimp [S]; nlinarith⟩
  have hWe : W + e = 2 * S := by
    dsimp [W, e, S]
    ring
  have hW : W ≤ 4 * a ^ 2 := by
    have haux : W ≤ 2 * S := by
      dsimp [W, S]
      nlinarith [sq_nonneg (b - c)]
    nlinarith
  have hdet :
      0 ≤ (2 * a * r + (b ^ 2 - c ^ 2) / 2) *
          (2 * a * r + (c ^ 2 - b ^ 2) / 2) -
        (a * (b + c)) ^ 2 := by
    have hdiff : (b ^ 2 - c ^ 2) ^ 2 = e * W := by
      dsimp [e, W]
      ring
    have he0 : 0 ≤ e := sq_nonneg _
    have hfirst : e * W / 4 ≤ e * a ^ 2 := by
      have := mul_le_mul_of_nonneg_left hW he0
      nlinarith
    have hrad :
        (b ^ 2 - c ^ 2) ^ 2 / 4 + a ^ 2 * (b + c) ^ 2 ≤
          2 * a ^ 2 * S := by
      rw [hdiff]
      change e * W / 4 + a ^ 2 * W ≤ 2 * a ^ 2 * S
      calc
        _ ≤ e * a ^ 2 + a ^ 2 * W := by linarith
        _ = a ^ 2 * (e + W) := by ring
        _ = 2 * a ^ 2 * S := by rw [add_comm e W, hWe]; ring
    rw [show (a * (b + c)) ^ 2 = a ^ 2 * (b + c) ^ 2 by ring]
    nlinarith
  have hdiag1 : 0 ≤ 2 * a * r + (b ^ 2 - c ^ 2) / 2 := by
    have habs : -(2 * r ^ 2) ≤ b ^ 2 - c ^ 2 := by
      rw [hrSq]
      dsimp [S]
      nlinarith [sq_nonneg b]
    have har : r ^ 2 ≤ a * r := by nlinarith
    nlinarith
  have hdiag2 : 0 ≤ 2 * a * r + (c ^ 2 - b ^ 2) / 2 := by
    have habs : -(2 * r ^ 2) ≤ c ^ 2 - b ^ 2 := by
      rw [hrSq]
      dsimp [S]
      nlinarith [sq_nonneg c]
    have har : r ^ 2 ≤ a * r := by nlinarith
    nlinarith
  let p : ℝ := x.ofLp 0
  let q : ℝ := x.ofLp 1
  have hquad :
      0 ≤ (2 * a * r + (b ^ 2 - c ^ 2) / 2) * p ^ 2 -
          2 * (a * (b + c)) * p * q +
          (2 * a * r + (c ^ 2 - b ^ 2) / 2) * q ^ 2 := by
    let A : ℝ := 2 * a * r + (b ^ 2 - c ^ 2) / 2
    let B : ℝ := a * (b + c)
    let C : ℝ := 2 * a * r + (c ^ 2 - b ^ 2) / 2
    have hA : 0 ≤ A := hdiag1
    have hC : 0 ≤ C := hdiag2
    have hAC : B ^ 2 ≤ A * C := by
      dsimp [A, B, C]
      linarith
    by_cases hA0 : A = 0
    · have hB0 : B = 0 := by
        have : B ^ 2 ≤ 0 := by simpa [hA0] using hAC
        nlinarith [sq_nonneg B]
      dsimp [A, B, C] at *
      simpa [hA0, hB0] using mul_nonneg hC (sq_nonneg q)
    · have hApos : 0 < A := lt_of_le_of_ne hA (Ne.symm hA0)
      have hmul :
          0 ≤ A * (A * p ^ 2 - 2 * B * p * q + C * q ^ 2) := by
        nlinarith [sq_nonneg (A * p - B * q),
          mul_nonneg (sub_nonneg.mpr hAC) (sq_nonneg q)]
      have := (mul_nonneg_iff_of_pos_left hApos).mp hmul
      dsimp [A, B, C] at this ⊢
      exact this
  have hsq :
      ‖Matrix.toEuclideanCLM (n := Fin 2) (𝕜 := ℝ) (twoByTwo a b c) x‖ ^ 2 ≤
        ((a + r) * ‖x‖) ^ 2 := by
    rw [EuclideanSpace.norm_sq_eq]
    simp only [Matrix.ofLp_toEuclideanCLM]
    simp [twoByTwo, Matrix.mulVec, Fin.sum_univ_two]
    have hxSq : ‖x‖ ^ 2 = (x.ofLp 0) ^ 2 + (x.ofLp 1) ^ 2 := by
      rw [EuclideanSpace.norm_sq_eq]
      simp [Fin.sum_univ_two]
    dsimp [p, q] at hquad
    dsimp [r]
    dsimp [S] at hrSq
    nlinarith
  apply (sq_le_sq₀ (norm_nonneg _)
    (mul_nonneg (add_nonneg ha hr0) (norm_nonneg _))).mp
  exact hsq

/-- Lemma 5.2, equation (5.2), for the actual `ℓ²` operator norm. -/
theorem twoByTwo_opNorm_le
    {a b c : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
    (hbc : b ^ 2 + c ^ 2 ≤ 2 * a ^ 2) :
    ‖twoByTwo a b c‖ ≤ a + Real.sqrt ((b ^ 2 + c ^ 2) / 2) := by
  rw [← Matrix.l2_opNorm_toEuclideanCLM]
  apply ContinuousLinearMap.opNorm_le_bound _
    (add_nonneg ha (Real.sqrt_nonneg _))
  intro x
  exact twoByTwo_pointwise_bound ha hb hc hbc x

private theorem sqrt_tangent_defect {a d : ℝ} (ha : 0 < a)
    (hda : d ≤ a ^ 2) :
    Real.sqrt (a ^ 2 - d) ≤ a - d / (2 * a) := by
  rw [Real.sqrt_le_iff]
  constructor
  · have hdiv : d / (2 * a) ≤ a / 2 := by
      rw [div_le_iff₀ (by positivity : 0 < 2 * a)]
      nlinarith
    linarith
  · field_simp
    nlinarith [sq_nonneg d]

/-- Lemma 5.2, equation (5.3), in defect form. -/
theorem twoByTwo_opNorm_defect
    {a b c d : ℝ} (ha : 0 < a) (hb : 0 ≤ b) (hc : 0 ≤ c)
    (hd : d = a ^ 2 - (b ^ 2 + c ^ 2) / 2) (hd0 : 0 ≤ d) :
    (1 / 2 : ℝ) * ‖twoByTwo a b c‖ ≤ a - d / (4 * a) := by
  have hbc : b ^ 2 + c ^ 2 ≤ 2 * a ^ 2 := by nlinarith
  have hbase := twoByTwo_opNorm_le ha.le hb hc hbc
  have hda : d ≤ a ^ 2 := by
    rw [hd]
    have : 0 ≤ (b ^ 2 + c ^ 2) / 2 := by positivity
    linarith
  have hsqrt := sqrt_tangent_defect ha hda
  have hsqrtEq : a ^ 2 - d = (b ^ 2 + c ^ 2) / 2 := by
    rw [hd]
    ring
  have hsqrt' :
      Real.sqrt ((b ^ 2 + c ^ 2) / 2) ≤ a - d / (2 * a) := by
    rw [← hsqrtEq]
    exact hsqrt
  calc
    (1 / 2 : ℝ) * ‖twoByTwo a b c‖ ≤
        (1 / 2) * (a + Real.sqrt ((b ^ 2 + c ^ 2) / 2)) := by gcongr
    _ ≤ (1 / 2) * (a + (a - d / (2 * a))) := by gcongr
    _ = a - d / (4 * a) := by ring

end

end CubicLogCommutators
