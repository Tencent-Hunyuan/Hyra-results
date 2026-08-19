import Mathlib.Analysis.Complex.Exponential
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Algebra of the accumulated main term

The analytic estimate for the `k`-th row contributes a logarithmic factor

`m log 2 + (2k-1) log 2 - (k/2)L - k²/(4m)`.

This file checks, without any asymptotics, that summing those factors for
`k = 1, ..., n-1` produces the exponents displayed in the paper.  In
particular it certifies the denominator `24m`, hence `96t` after `m = 4t`.
-/

open scoped BigOperators

namespace PartialHadamard

section PolynomialSums

/-- Sum of the first `r` odd positive integers. -/
theorem sum_first_odds (r : ℕ) :
    ∑ j ∈ Finset.range r, (2 * j + 1) = r ^ 2 := by
  induction r with
  | zero => simp
  | succ r ih =>
      rw [Finset.sum_range_succ, ih]
      ring

/-- Division-free form of `∑_{k=1}^r k = r(r+1)/2`. -/
theorem two_mul_sum_first (r : ℕ) :
    2 * (∑ j ∈ Finset.range r, (j + 1)) = r * (r + 1) := by
  induction r with
  | zero => simp
  | succ r ih =>
      rw [Finset.sum_range_succ]
      calc
        2 * ((∑ j ∈ Finset.range r, (j + 1)) + (r + 1)) =
            2 * (∑ j ∈ Finset.range r, (j + 1)) + 2 * (r + 1) := by ring
        _ = r * (r + 1) + 2 * (r + 1) := by rw [ih]
        _ = (r + 1) * (r + 1 + 1) := by ring

/-- Division-free form of `∑_{k=1}^r k² = r(r+1)(2r+1)/6`. -/
theorem six_mul_sum_first_squares (r : ℕ) :
    6 * (∑ j ∈ Finset.range r, (j + 1) ^ 2) =
      r * (r + 1) * (2 * r + 1) := by
  induction r with
  | zero => simp
  | succ r ih =>
      rw [Finset.sum_range_succ]
      calc
        6 * ((∑ j ∈ Finset.range r, (j + 1) ^ 2) + (r + 1) ^ 2) =
            6 * (∑ j ∈ Finset.range r, (j + 1) ^ 2) + 6 * (r + 1) ^ 2 := by ring
        _ = r * (r + 1) * (2 * r + 1) + 6 * (r + 1) ^ 2 := by rw [ih]
        _ = (r + 1) * (r + 1 + 1) * (2 * (r + 1) + 1) := by ring

theorem sum_first_cast (r : ℕ) :
    ∑ j ∈ Finset.range r, ((j + 1 : ℕ) : ℝ) =
      (r : ℝ) * (r + 1) / 2 := by
  have h := congrArg (fun z : ℕ ↦ (z : ℝ)) (two_mul_sum_first r)
  push_cast at h ⊢
  apply (eq_div_iff (by norm_num : (2 : ℝ) ≠ 0)).2
  simpa [mul_comm] using h

theorem sum_first_squares_cast (r : ℕ) :
    ∑ j ∈ Finset.range r, (((j + 1 : ℕ) : ℝ) ^ 2) =
      (r : ℝ) * (r + 1) * (2 * r + 1) / 6 := by
  have h := congrArg (fun z : ℕ ↦ (z : ℝ)) (six_mul_sum_first_squares r)
  push_cast at h ⊢
  apply (eq_div_iff (by norm_num : (6 : ℝ) ≠ 0)).2
  simpa [mul_comm] using h

theorem sum_first_odds_cast (r : ℕ) :
    ∑ j ∈ Finset.range r, (2 * ((j : ℕ) : ℝ) + 1) = (r : ℝ) ^ 2 := by
  have h := congrArg (fun z : ℕ ↦ (z : ℝ)) (sum_first_odds r)
  push_cast at h
  simpa [Nat.cast_ofNat] using h

end PolynomialSums

section LogMainTerm

/-- Logarithm of the paper's `k`-th row-extension main factor.  The
parameters `ℓ₂` and `L` are later instantiated as `log 2` and
`log (2πm)`.  Keeping them symbolic makes the exponent calculation more
general and purely algebraic. -/
noncomputable def rowLogFactor (ℓ₂ m L : ℝ) (k : ℕ) : ℝ :=
  m * ℓ₂ + (2 * (k : ℝ) - 1) * ℓ₂
    - ((k : ℝ) / 2) * L - (k : ℝ) ^ 2 / (4 * m)

/-- Accumulated logarithm, including the `2^m` choices for the first row. -/
noncomputable def accumulatedLogMain (ℓ₂ m L : ℝ) (r : ℕ) : ℝ :=
  m * ℓ₂ + ∑ j ∈ Finset.range r, rowLogFactor ℓ₂ m L (j + 1)

/-- Closed form after adjoining `r` rows after the first one. -/
noncomputable def closedLogMain (ℓ₂ m L : ℝ) (r : ℕ) : ℝ :=
  (((r : ℝ) + 1) * m + (r : ℝ) ^ 2) * ℓ₂
    - (((r : ℝ) + 1) * r / 4) * L
    - ((r : ℝ) * (r + 1) * (2 * r + 1)) / (24 * m)

/-- Exact exponent calculation behind `M_n = B_{n,m}` (with `r=n-1`). -/
theorem accumulatedLogMain_eq_closedLogMain
    (ℓ₂ m L : ℝ) (r : ℕ) (hm : m ≠ 0) :
    accumulatedLogMain ℓ₂ m L r = closedLogMain ℓ₂ m L r := by
  unfold accumulatedLogMain closedLogMain rowLogFactor
  simp_rw [div_eq_mul_inv, Finset.sum_sub_distrib, Finset.sum_add_distrib,
    ← Finset.sum_mul]
  rw [sum_first_cast, sum_first_squares_cast]
  have hodd :
      (∑ j ∈ Finset.range r, (2 * ((j + 1 : ℕ) : ℝ) - 1)) = (r : ℝ) ^ 2 := by
    calc
      (∑ j ∈ Finset.range r, (2 * ((j + 1 : ℕ) : ℝ) - 1)) =
          ∑ j ∈ Finset.range r, (2 * (j : ℝ) + 1) := by
            apply Finset.sum_congr rfl
            intro j _
            push_cast
            ring
      _ = (r : ℝ) ^ 2 := sum_first_odds_cast r
  rw [hodd]
  simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  field_simp
  ring

/-- Product formulation of the same calculation.  Defining the positive
factors by exponentiating their logarithms avoids irrelevant side
conditions about real powers. -/
noncomputable def rowMainFactor (ℓ₂ m L : ℝ) (k : ℕ) : ℝ :=
  Real.exp (rowLogFactor ℓ₂ m L k)

noncomputable def accumulatedMain (ℓ₂ m L : ℝ) (r : ℕ) : ℝ :=
  Real.exp (m * ℓ₂) *
    ∏ j ∈ Finset.range r, rowMainFactor ℓ₂ m L (j + 1)

noncomputable def closedMain (ℓ₂ m L : ℝ) (r : ℕ) : ℝ :=
  Real.exp (closedLogMain ℓ₂ m L r)

theorem accumulatedMain_eq_closedMain
    (ℓ₂ m L : ℝ) (r : ℕ) (hm : m ≠ 0) :
    accumulatedMain ℓ₂ m L r = closedMain ℓ₂ m L r := by
  unfold accumulatedMain closedMain rowMainFactor
  rw [← Real.exp_sum, ← Real.exp_add]
  simpa [accumulatedLogMain] using congrArg Real.exp
    (accumulatedLogMain_eq_closedLogMain ℓ₂ m L r hm)

/-- Substituting `m=4t` changes the correction denominator from `24m`
to `96t`; this is the paper's `1/96` coefficient. -/
theorem correction_denominator_four_mul (n t : ℕ) :
    ((n : ℝ) * (n - 1) * (2 * n - 1)) / (24 * (4 * t)) =
      ((n : ℝ) * (n - 1) * (2 * n - 1)) / (96 * t) := by
  ring

end LogMainTerm

end PartialHadamard
