import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Tactic

/-!
# Explicit constants in the fixed-point and final estimates
-/

set_option autoImplicit false

namespace CubicLogCommutators

noncomputable section

/-- The coefficient in Proposition 6.3 is strictly below `0.85`. -/
theorem contraction_coefficient_lt :
    (192 + 104 * Real.sqrt 2) / 400 < (85 : ℝ) / 100 := by
  have hs0 : 0 ≤ Real.sqrt 2 := Real.sqrt_nonneg _
  have hs2 : (Real.sqrt 2) ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  nlinarith

/-- The final base-conversion constant is below `5.3 × 10⁶`. -/
theorem final_constant_lt :
    2402 * (9 / Real.log 2) ^ 3 < 5300000 := by
  have hlog : (69314 : ℝ) / 100000 < Real.log 2 := by
    linarith [Real.log_two_gt_d9]
  have hlog0 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  rw [div_pow]
  calc
    2402 * (9 ^ 3 / Real.log 2 ^ 3) =
        (2402 * 9 ^ 3) / Real.log 2 ^ 3 := by ring
    _ < 5300000 := by
      rw [div_lt_iff₀ (pow_pos hlog0 3)]
      have hpow :
          ((69314 : ℝ) / 100000) ^ 3 < (Real.log 2) ^ 3 := by
        exact pow_lt_pow_left₀ hlog (by norm_num) (by norm_num)
      nlinarith

end

end CubicLogCommutators
