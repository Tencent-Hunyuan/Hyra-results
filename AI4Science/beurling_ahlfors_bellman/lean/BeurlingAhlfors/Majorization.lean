import Mathlib.Analysis.MeanInequalities
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic

/-!
# Exact endpoint majorization

The paper derives its two endpoint norm bounds from weighted AM--GM.  Because
both profiles use rational exponents, we clear roots by writing `r = x^3`,
`s = y^3` at `p = 16/3`, and `r = x^2`, `s = y^2` at `p = 15/2`.
This produces ordinary polynomial inequalities in nonnegative `x,y`.
-/

namespace BeurlingAhlfors

private theorem amgm_16_6 {x y : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) :
    16 * x ^ 6 * y ^ 10 ≤ 6 * x ^ 16 + 10 * y ^ 16 := by
  have h := Real.geom_mean_le_arith_mean2_weighted
    (w₁ := (6 : ℝ) / 16) (w₂ := (10 : ℝ) / 16)
    (p₁ := x ^ 16) (p₂ := y ^ 16)
    (by norm_num) (by norm_num) (by positivity) (by positivity) (by norm_num)
  have hxpow : (x ^ (16 : ℕ)) ^ ((6 : ℝ) / 16) = x ^ (6 : ℕ) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hx]
    norm_num
  have hypow : (y ^ (16 : ℕ)) ^ ((10 : ℝ) / 16) = y ^ (10 : ℕ) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hy]
    norm_num
  rw [hxpow, hypow] at h
  nlinarith

private theorem amgm_16_9 {x y : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) :
    16 * x ^ 9 * y ^ 7 ≤ 9 * x ^ 16 + 7 * y ^ 16 := by
  have h := Real.geom_mean_le_arith_mean2_weighted
    (w₁ := (9 : ℝ) / 16) (w₂ := (7 : ℝ) / 16)
    (p₁ := x ^ 16) (p₂ := y ^ 16)
    (by norm_num) (by norm_num) (by positivity) (by positivity) (by norm_num)
  have hxpow : (x ^ (16 : ℕ)) ^ ((9 : ℝ) / 16) = x ^ (9 : ℕ) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hx]
    norm_num
  have hypow : (y ^ (16 : ℕ)) ^ ((7 : ℝ) / 16) = y ^ (7 : ℕ) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hy]
    norm_num
  rw [hxpow, hypow] at h
  nlinarith

private theorem amgm_16_12 {x y : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) :
    16 * x ^ 12 * y ^ 4 ≤ 12 * x ^ 16 + 4 * y ^ 16 := by
  have h := Real.geom_mean_le_arith_mean2_weighted
    (w₁ := (12 : ℝ) / 16) (w₂ := (4 : ℝ) / 16)
    (p₁ := x ^ 16) (p₂ := y ^ 16)
    (by norm_num) (by norm_num) (by positivity) (by positivity) (by norm_num)
  have hxpow : (x ^ (16 : ℕ)) ^ ((12 : ℝ) / 16) = x ^ (12 : ℕ) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hx]
    norm_num
  have hypow : (y ^ (16 : ℕ)) ^ ((4 : ℝ) / 16) = y ^ (4 : ℕ) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hy]
    norm_num
  rw [hxpow, hypow] at h
  nlinarith

private theorem amgm_16_15 {x y : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) :
    16 * x ^ 15 * y ≤ 15 * x ^ 16 + y ^ 16 := by
  have h := Real.geom_mean_le_arith_mean2_weighted
    (w₁ := (15 : ℝ) / 16) (w₂ := (1 : ℝ) / 16)
    (p₁ := x ^ 16) (p₂ := y ^ 16)
    (by norm_num) (by norm_num) (by positivity) (by positivity) (by norm_num)
  have hxpow : (x ^ (16 : ℕ)) ^ ((15 : ℝ) / 16) = x ^ (15 : ℕ) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hx]
    norm_num
  have hypow : (y ^ (16 : ℕ)) ^ ((1 : ℝ) / 16) = y := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hy]
    norm_num
  rw [hxpow, hypow] at h
  nlinarith

/-- Exact M1 rational certificate for the `p = 16/3` profile. -/
theorem quintic_M1_certificate :
    (6185 : ℕ) * 88 ^ 15 ≤ 19 * 47843904707555223828076719789998 := by
  norm_num

/-- Exact M2 rational certificate for the `p = 16/3` profile. -/
theorem quintic_M2_certificate :
    (57 : ℕ) * 2084821415846460515200768
      ≤ 22830 * 98960 * 47 ^ 10 := by
  norm_num

private theorem scaled_amgm_16_6 {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 < z) :
    x ^ 6 * y ^ 10 ≤ (6 / 16 : ℝ) * (x ^ 16 / z ^ 10)
      + (10 / 16 : ℝ) * z ^ 6 * y ^ 16 := by
  have h := amgm_16_6 (x := x / z) (y := y) (div_nonneg hx hz.le) hy
  have h' : (x / z) ^ 6 * y ^ 10 ≤
      (6 / 16 : ℝ) * (x / z) ^ 16 + (10 / 16 : ℝ) * y ^ 16 := by
    nlinarith
  calc
    x ^ 6 * y ^ 10 = z ^ 6 * ((x / z) ^ 6 * y ^ 10) := by
      field_simp [ne_of_gt hz]
    _ ≤ z ^ 6 * ((6 / 16 : ℝ) * (x / z) ^ 16 + (10 / 16 : ℝ) * y ^ 16) := by
      exact mul_le_mul_of_nonneg_left h' (by positivity)
    _ = (6 / 16 : ℝ) * (x ^ 16 / z ^ 10)
          + (10 / 16 : ℝ) * z ^ 6 * y ^ 16 := by
      field_simp [ne_of_gt hz]

private theorem scaled_amgm_16_9 {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 < z) :
    x ^ 9 * y ^ 7 ≤ (9 / 16 : ℝ) * (x ^ 16 / z ^ 7)
      + (7 / 16 : ℝ) * z ^ 9 * y ^ 16 := by
  have h := amgm_16_9 (x := x / z) (y := y) (div_nonneg hx hz.le) hy
  have h' : (x / z) ^ 9 * y ^ 7 ≤
      (9 / 16 : ℝ) * (x / z) ^ 16 + (7 / 16 : ℝ) * y ^ 16 := by
    nlinarith
  calc
    x ^ 9 * y ^ 7 = z ^ 9 * ((x / z) ^ 9 * y ^ 7) := by
      field_simp [ne_of_gt hz]
    _ ≤ z ^ 9 * ((9 / 16 : ℝ) * (x / z) ^ 16 + (7 / 16 : ℝ) * y ^ 16) := by
      exact mul_le_mul_of_nonneg_left h' (by positivity)
    _ = (9 / 16 : ℝ) * (x ^ 16 / z ^ 7)
          + (7 / 16 : ℝ) * z ^ 9 * y ^ 16 := by
      field_simp [ne_of_gt hz]

private theorem scaled_amgm_16_12 {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 < z) :
    x ^ 12 * y ^ 4 ≤ (12 / 16 : ℝ) * (x ^ 16 / z ^ 4)
      + (4 / 16 : ℝ) * z ^ 12 * y ^ 16 := by
  have h := amgm_16_12 (x := x / z) (y := y) (div_nonneg hx hz.le) hy
  have h' : (x / z) ^ 12 * y ^ 4 ≤
      (12 / 16 : ℝ) * (x / z) ^ 16 + (4 / 16 : ℝ) * y ^ 16 := by
    nlinarith
  calc
    x ^ 12 * y ^ 4 = z ^ 12 * ((x / z) ^ 12 * y ^ 4) := by
      field_simp [ne_of_gt hz]
    _ ≤ z ^ 12 * ((12 / 16 : ℝ) * (x / z) ^ 16 + (4 / 16 : ℝ) * y ^ 16) := by
      exact mul_le_mul_of_nonneg_left h' (by positivity)
    _ = (12 / 16 : ℝ) * (x ^ 16 / z ^ 4)
          + (4 / 16 : ℝ) * z ^ 12 * y ^ 16 := by
      field_simp [ne_of_gt hz]

private theorem scaled_amgm_16_15 {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 < z) :
    x ^ 15 * y ≤ (15 / 16 : ℝ) * (x ^ 16 / z)
      + (1 / 16 : ℝ) * z ^ 15 * y ^ 16 := by
  have h := amgm_16_15 (x := x / z) (y := y) (div_nonneg hx hz.le) hy
  have h' : (x / z) ^ 15 * y ≤
      (15 / 16 : ℝ) * (x / z) ^ 16 + (1 / 16 : ℝ) * y ^ 16 := by
    nlinarith
  calc
    x ^ 15 * y = z ^ 15 * ((x / z) ^ 15 * y) := by
      field_simp [ne_of_gt hz]
    _ ≤ z ^ 15 * ((15 / 16 : ℝ) * (x / z) ^ 16 + (1 / 16 : ℝ) * y ^ 16) := by
      exact mul_le_mul_of_nonneg_left h' (by positivity)
    _ = (15 / 16 : ℝ) * (x ^ 16 / z)
          + (1 / 16 : ℝ) * z ^ 15 * y ^ 16 := by
      field_simp [ne_of_gt hz]

/-- Polynomial form of the paper's majorization inequality for the quintic
profile. -/
theorem quintic_majorization_polynomial {x y : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) :
    (19 / 6185 : ℝ) *
        (729 * y ^ 16 - 20736 * x ^ 6 * y ^ 10 - 61440 * x ^ 9 * y ^ 7
          - 56320 * x ^ 12 * y ^ 4 - 4320 * x ^ 15 * y)
      ≥ y ^ 16 - 22830 * x ^ 16 := by
  let z : ℝ := 47 / 88
  have hz : 0 < z := by norm_num [z]
  have h6 := scaled_amgm_16_6 hx hy hz
  have h9 := scaled_amgm_16_9 hx hy hz
  have h12 := scaled_amgm_16_12 hx hy hz
  have h15 := scaled_amgm_16_15 hx hy hz
  norm_num [z] at h6 h9 h12 h15
  have hsum :=
    add_le_add (mul_le_mul_of_nonneg_left h6 (by norm_num : (0 : ℝ) ≤ 20736))
      (add_le_add (mul_le_mul_of_nonneg_left h9 (by norm_num : (0 : ℝ) ≤ 61440))
        (add_le_add (mul_le_mul_of_nonneg_left h12 (by norm_num : (0 : ℝ) ≤ 56320))
          (mul_le_mul_of_nonneg_left h15 (by norm_num : (0 : ℝ) ≤ 4320))))
  norm_num at hsum
  ring_nf at hsum
  have hX : 0 ≤ x ^ 16 := by positivity
  have hY : 0 ≤ y ^ 16 := by positivity
  have hA : (19 / 6185 : ℝ) * (390904015471211346600144 / 52599132235830049 : ℝ)
      ≤ 22830 := by norm_num
  have hB : 1 ≤ (19 / 6185 : ℝ) *
      (729 - 29650017392813429616537423565865 / 73486926950056298395851554816 : ℝ) := by
    norm_num
  nlinarith

private theorem amgm_15_4 {x y : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) :
    15 * x ^ 4 * y ^ 11 ≤ 4 * x ^ 15 + 11 * y ^ 15 := by
  have h := Real.geom_mean_le_arith_mean2_weighted
    (w₁ := (4 : ℝ) / 15) (w₂ := (11 : ℝ) / 15)
    (p₁ := x ^ 15) (p₂ := y ^ 15)
    (by norm_num) (by norm_num) (by positivity) (by positivity) (by norm_num)
  have hxpow : (x ^ (15 : ℕ)) ^ ((4 : ℝ) / 15) = x ^ (4 : ℕ) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hx]
    norm_num
  have hypow : (y ^ (15 : ℕ)) ^ ((11 : ℝ) / 15) = y ^ (11 : ℕ) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hy]
    norm_num
  rw [hxpow, hypow] at h
  nlinarith

private theorem amgm_15_6 {x y : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) :
    15 * x ^ 6 * y ^ 9 ≤ 6 * x ^ 15 + 9 * y ^ 15 := by
  have h := Real.geom_mean_le_arith_mean2_weighted
    (w₁ := (6 : ℝ) / 15) (w₂ := (9 : ℝ) / 15)
    (p₁ := x ^ 15) (p₂ := y ^ 15)
    (by norm_num) (by norm_num) (by positivity) (by positivity) (by norm_num)
  have hxpow : (x ^ (15 : ℕ)) ^ ((6 : ℝ) / 15) = x ^ (6 : ℕ) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hx]
    norm_num
  have hypow : (y ^ (15 : ℕ)) ^ ((9 : ℝ) / 15) = y ^ (9 : ℕ) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hy]
    norm_num
  rw [hxpow, hypow] at h
  nlinarith

private theorem amgm_15_8 {x y : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) :
    15 * x ^ 8 * y ^ 7 ≤ 8 * x ^ 15 + 7 * y ^ 15 := by
  have h := Real.geom_mean_le_arith_mean2_weighted
    (w₁ := (8 : ℝ) / 15) (w₂ := (7 : ℝ) / 15)
    (p₁ := x ^ 15) (p₂ := y ^ 15)
    (by norm_num) (by norm_num) (by positivity) (by positivity) (by norm_num)
  have hxpow : (x ^ (15 : ℕ)) ^ ((8 : ℝ) / 15) = x ^ (8 : ℕ) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hx]
    norm_num
  have hypow : (y ^ (15 : ℕ)) ^ ((7 : ℝ) / 15) = y ^ (7 : ℕ) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hy]
    norm_num
  rw [hxpow, hypow] at h
  nlinarith

private theorem amgm_15_10 {x y : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) :
    15 * x ^ 10 * y ^ 5 ≤ 10 * x ^ 15 + 5 * y ^ 15 := by
  have h := Real.geom_mean_le_arith_mean2_weighted
    (w₁ := (10 : ℝ) / 15) (w₂ := (5 : ℝ) / 15)
    (p₁ := x ^ 15) (p₂ := y ^ 15)
    (by norm_num) (by norm_num) (by positivity) (by positivity) (by norm_num)
  have hxpow : (x ^ (15 : ℕ)) ^ ((10 : ℝ) / 15) = x ^ (10 : ℕ) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hx]
    norm_num
  have hypow : (y ^ (15 : ℕ)) ^ ((5 : ℝ) / 15) = y ^ (5 : ℕ) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hy]
    norm_num
  rw [hxpow, hypow] at h
  nlinarith

private theorem amgm_15_12 {x y : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) :
    15 * x ^ 12 * y ^ 3 ≤ 12 * x ^ 15 + 3 * y ^ 15 := by
  have h := Real.geom_mean_le_arith_mean2_weighted
    (w₁ := (12 : ℝ) / 15) (w₂ := (3 : ℝ) / 15)
    (p₁ := x ^ 15) (p₂ := y ^ 15)
    (by norm_num) (by norm_num) (by positivity) (by positivity) (by norm_num)
  have hxpow : (x ^ (15 : ℕ)) ^ ((12 : ℝ) / 15) = x ^ (12 : ℕ) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hx]
    norm_num
  have hypow : (y ^ (15 : ℕ)) ^ ((3 : ℝ) / 15) = y ^ (3 : ℕ) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hy]
    norm_num
  rw [hxpow, hypow] at h
  nlinarith

/-- Exact M1 rational certificate for the `p = 15/2` profile. -/
theorem sextic_M1_certificate :
    (454 : ℕ) * 178813934326171875 ≤ 105 * 773178782568192676 := by
  norm_num

/-- Exact M2 rational certificate for the `p = 15/2` profile. -/
theorem sextic_M2_certificate :
    (7 : ℕ) * 6369209766331093750
      ≤ 22865000 * 227 * 8 ^ 11 := by
  norm_num

private theorem scaled_amgm_15_4 {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 < z) :
    x ^ 4 * y ^ 11 ≤ (4 / 15 : ℝ) * (x ^ 15 / z ^ 11)
      + (11 / 15 : ℝ) * z ^ 4 * y ^ 15 := by
  have h := amgm_15_4 (x := x / z) (y := y) (div_nonneg hx hz.le) hy
  have h' : (x / z) ^ 4 * y ^ 11 ≤
      (4 / 15 : ℝ) * (x / z) ^ 15 + (11 / 15 : ℝ) * y ^ 15 := by nlinarith
  calc
    x ^ 4 * y ^ 11 = z ^ 4 * ((x / z) ^ 4 * y ^ 11) := by field_simp [ne_of_gt hz]
    _ ≤ z ^ 4 * ((4 / 15 : ℝ) * (x / z) ^ 15 + (11 / 15 : ℝ) * y ^ 15) := by
      exact mul_le_mul_of_nonneg_left h' (by positivity)
    _ = (4 / 15 : ℝ) * (x ^ 15 / z ^ 11) + (11 / 15 : ℝ) * z ^ 4 * y ^ 15 := by
      field_simp [ne_of_gt hz]

private theorem scaled_amgm_15_6 {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 < z) :
    x ^ 6 * y ^ 9 ≤ (6 / 15 : ℝ) * (x ^ 15 / z ^ 9)
      + (9 / 15 : ℝ) * z ^ 6 * y ^ 15 := by
  have h := amgm_15_6 (x := x / z) (y := y) (div_nonneg hx hz.le) hy
  have h' : (x / z) ^ 6 * y ^ 9 ≤
      (6 / 15 : ℝ) * (x / z) ^ 15 + (9 / 15 : ℝ) * y ^ 15 := by nlinarith
  calc
    x ^ 6 * y ^ 9 = z ^ 6 * ((x / z) ^ 6 * y ^ 9) := by field_simp [ne_of_gt hz]
    _ ≤ z ^ 6 * ((6 / 15 : ℝ) * (x / z) ^ 15 + (9 / 15 : ℝ) * y ^ 15) := by
      exact mul_le_mul_of_nonneg_left h' (by positivity)
    _ = (6 / 15 : ℝ) * (x ^ 15 / z ^ 9) + (9 / 15 : ℝ) * z ^ 6 * y ^ 15 := by
      field_simp [ne_of_gt hz]

private theorem scaled_amgm_15_8 {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 < z) :
    x ^ 8 * y ^ 7 ≤ (8 / 15 : ℝ) * (x ^ 15 / z ^ 7)
      + (7 / 15 : ℝ) * z ^ 8 * y ^ 15 := by
  have h := amgm_15_8 (x := x / z) (y := y) (div_nonneg hx hz.le) hy
  have h' : (x / z) ^ 8 * y ^ 7 ≤
      (8 / 15 : ℝ) * (x / z) ^ 15 + (7 / 15 : ℝ) * y ^ 15 := by nlinarith
  calc
    x ^ 8 * y ^ 7 = z ^ 8 * ((x / z) ^ 8 * y ^ 7) := by field_simp [ne_of_gt hz]
    _ ≤ z ^ 8 * ((8 / 15 : ℝ) * (x / z) ^ 15 + (7 / 15 : ℝ) * y ^ 15) := by
      exact mul_le_mul_of_nonneg_left h' (by positivity)
    _ = (8 / 15 : ℝ) * (x ^ 15 / z ^ 7) + (7 / 15 : ℝ) * z ^ 8 * y ^ 15 := by
      field_simp [ne_of_gt hz]

private theorem scaled_amgm_15_10 {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 < z) :
    x ^ 10 * y ^ 5 ≤ (10 / 15 : ℝ) * (x ^ 15 / z ^ 5)
      + (5 / 15 : ℝ) * z ^ 10 * y ^ 15 := by
  have h := amgm_15_10 (x := x / z) (y := y) (div_nonneg hx hz.le) hy
  have h' : (x / z) ^ 10 * y ^ 5 ≤
      (10 / 15 : ℝ) * (x / z) ^ 15 + (5 / 15 : ℝ) * y ^ 15 := by nlinarith
  calc
    x ^ 10 * y ^ 5 = z ^ 10 * ((x / z) ^ 10 * y ^ 5) := by field_simp [ne_of_gt hz]
    _ ≤ z ^ 10 * ((10 / 15 : ℝ) * (x / z) ^ 15 + (5 / 15 : ℝ) * y ^ 15) := by
      exact mul_le_mul_of_nonneg_left h' (by positivity)
    _ = (10 / 15 : ℝ) * (x ^ 15 / z ^ 5) + (5 / 15 : ℝ) * z ^ 10 * y ^ 15 := by
      field_simp [ne_of_gt hz]

private theorem scaled_amgm_15_12 {x y z : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 < z) :
    x ^ 12 * y ^ 3 ≤ (12 / 15 : ℝ) * (x ^ 15 / z ^ 3)
      + (3 / 15 : ℝ) * z ^ 12 * y ^ 15 := by
  have h := amgm_15_12 (x := x / z) (y := y) (div_nonneg hx hz.le) hy
  have h' : (x / z) ^ 12 * y ^ 3 ≤
      (12 / 15 : ℝ) * (x / z) ^ 15 + (3 / 15 : ℝ) * y ^ 15 := by nlinarith
  calc
    x ^ 12 * y ^ 3 = z ^ 12 * ((x / z) ^ 12 * y ^ 3) := by field_simp [ne_of_gt hz]
    _ ≤ z ^ 12 * ((12 / 15 : ℝ) * (x / z) ^ 15 + (3 / 15 : ℝ) * y ^ 15) := by
      exact mul_le_mul_of_nonneg_left h' (by positivity)
    _ = (12 / 15 : ℝ) * (x ^ 15 / z ^ 3) + (3 / 15 : ℝ) * z ^ 12 * y ^ 15 := by
      field_simp [ne_of_gt hz]

/-- Polynomial form of the paper's majorization inequality for the sextic
profile. -/
theorem sextic_majorization_polynomial {x y : ℝ} (hx : 0 ≤ x) (hy : 0 ≤ y) :
    (105 / 454 : ℝ) *
        (12 * y ^ 15 - 675 * x ^ 4 * y ^ 11 - 3300 * x ^ 6 * y ^ 9
          - 6600 * x ^ 8 * y ^ 7 - 5423 * x ^ 10 * y ^ 5
          - 2640 * x ^ 12 * y ^ 3)
      ≥ y ^ 15 - 22865000 * x ^ 15 := by
  let z : ℝ := 8 / 25
  have hz : 0 < z := by norm_num [z]
  have h4 := scaled_amgm_15_4 hx hy hz
  have h6 := scaled_amgm_15_6 hx hy hz
  have h8 := scaled_amgm_15_8 hx hy hz
  have h10 := scaled_amgm_15_10 hx hy hz
  have h12 := scaled_amgm_15_12 hx hy hz
  norm_num [z] at h4 h6 h8 h10 h12
  have hsum :=
    add_le_add (mul_le_mul_of_nonneg_left h4 (by norm_num : (0 : ℝ) ≤ 675))
      (add_le_add (mul_le_mul_of_nonneg_left h6 (by norm_num : (0 : ℝ) ≤ 3300))
        (add_le_add (mul_le_mul_of_nonneg_left h8 (by norm_num : (0 : ℝ) ≤ 6600))
          (add_le_add (mul_le_mul_of_nonneg_left h10 (by norm_num : (0 : ℝ) ≤ 5423))
            (mul_le_mul_of_nonneg_left h12 (by norm_num : (0 : ℝ) ≤ 2640)))))
  norm_num at hsum
  ring_nf at hsum
  have hX : 0 ≤ x ^ 15 := by positivity
  have hY : 0 ≤ y ^ 15 := by positivity
  have hA : (105 / 454 : ℝ) * (636920976633109375 / 6442450944 : ℝ)
      ≤ 22865000 := by norm_num
  have hB : 1 ≤ (105 / 454 : ℝ) *
      (12 - 1372588429345869824 / 178813934326171875 : ℝ) := by norm_num
  nlinarith


/-- A small helper for clearing rational roots in the endpoint profiles. -/
private theorem root_pow_eq_rpow {x : ℝ} (hx : 0 ≤ x) (d n : ℕ) (hd : d ≠ 0) :
    (x ^ ((d : ℝ)⁻¹)) ^ n = x ^ ((n : ℝ) / d) := by
  rw [← Real.rpow_natCast, ← Real.rpow_mul hx]
  congr 1
  field_simp

/-- The actual majorization inequality for the quintic Bellman function on the
closed quadrant.  This is (A4) at `p = 16/3`, with `Λ^p = 22830`. -/
theorem quintic_majorization {r s : ℝ} (hr : 0 ≤ r) (hs : 0 ≤ s) :
    (19 / 6185 : ℝ) *
        (729 * s ^ (16 / 3 : ℝ) - 20736 * r ^ 2 * s ^ (10 / 3 : ℝ)
          - 61440 * r ^ 3 * s ^ (7 / 3 : ℝ)
          - 56320 * r ^ 4 * s ^ (4 / 3 : ℝ)
          - 4320 * r ^ 5 * s ^ (1 / 3 : ℝ))
      ≥ s ^ (16 / 3 : ℝ) - 22830 * r ^ (16 / 3 : ℝ) := by
  let x : ℝ := r ^ ((3 : ℝ)⁻¹)
  let y : ℝ := s ^ ((3 : ℝ)⁻¹)
  have hx : 0 ≤ x := Real.rpow_nonneg hr _
  have hy : 0 ≤ y := Real.rpow_nonneg hs _
  have h := quintic_majorization_polynomial hx hy
  have hx3 : x ^ 3 = r := Real.rpow_inv_natCast_pow hr (by norm_num)
  have hy3 : y ^ 3 = s := Real.rpow_inv_natCast_pow hs (by norm_num)
  have hx16 : x ^ 16 = r ^ (16 / 3 : ℝ) := root_pow_eq_rpow hr 3 16 (by norm_num)
  have hy16 : y ^ 16 = s ^ (16 / 3 : ℝ) := root_pow_eq_rpow hs 3 16 (by norm_num)
  have hy10 : y ^ 10 = s ^ (10 / 3 : ℝ) := root_pow_eq_rpow hs 3 10 (by norm_num)
  have hy7 : y ^ 7 = s ^ (7 / 3 : ℝ) := root_pow_eq_rpow hs 3 7 (by norm_num)
  have hy4 : y ^ 4 = s ^ (4 / 3 : ℝ) := root_pow_eq_rpow hs 3 4 (by norm_num)
  have hy1 : y = s ^ (1 / 3 : ℝ) := by
    dsimp [y]
    congr 1
    norm_num
  rw [show x ^ 6 = r ^ 2 by rw [← hx3]; ring,
    show x ^ 9 = r ^ 3 by rw [← hx3]; ring,
    show x ^ 12 = r ^ 4 by rw [← hx3]; ring,
    show x ^ 15 = r ^ 5 by rw [← hx3]; ring,
    hx16, hy16, hy10, hy7, hy4, hy1] at h
  exact h

/-- The actual majorization inequality for the sextic Bellman function on the
closed quadrant.  This is (A4) at `p = 15/2`, with `Λ^p = 22865000`. -/
theorem sextic_majorization {r s : ℝ} (hr : 0 ≤ r) (hs : 0 ≤ s) :
    (105 / 454 : ℝ) *
        (12 * s ^ (15 / 2 : ℝ) - 675 * r ^ 2 * s ^ (11 / 2 : ℝ)
          - 3300 * r ^ 3 * s ^ (9 / 2 : ℝ)
          - 6600 * r ^ 4 * s ^ (7 / 2 : ℝ)
          - 5423 * r ^ 5 * s ^ (5 / 2 : ℝ)
          - 2640 * r ^ 6 * s ^ (3 / 2 : ℝ))
      ≥ s ^ (15 / 2 : ℝ) - 22865000 * r ^ (15 / 2 : ℝ) := by
  let x : ℝ := r ^ ((2 : ℝ)⁻¹)
  let y : ℝ := s ^ ((2 : ℝ)⁻¹)
  have hx : 0 ≤ x := Real.rpow_nonneg hr _
  have hy : 0 ≤ y := Real.rpow_nonneg hs _
  have h := sextic_majorization_polynomial hx hy
  have hx2 : x ^ 2 = r := Real.rpow_inv_natCast_pow hr (by norm_num)
  have hy2 : y ^ 2 = s := Real.rpow_inv_natCast_pow hs (by norm_num)
  have hx15 : x ^ 15 = r ^ (15 / 2 : ℝ) := root_pow_eq_rpow hr 2 15 (by norm_num)
  have hy15 : y ^ 15 = s ^ (15 / 2 : ℝ) := root_pow_eq_rpow hs 2 15 (by norm_num)
  have hy11 : y ^ 11 = s ^ (11 / 2 : ℝ) := root_pow_eq_rpow hs 2 11 (by norm_num)
  have hy9 : y ^ 9 = s ^ (9 / 2 : ℝ) := root_pow_eq_rpow hs 2 9 (by norm_num)
  have hy7 : y ^ 7 = s ^ (7 / 2 : ℝ) := root_pow_eq_rpow hs 2 7 (by norm_num)
  have hy5 : y ^ 5 = s ^ (5 / 2 : ℝ) := root_pow_eq_rpow hs 2 5 (by norm_num)
  have hy3 : y ^ 3 = s ^ (3 / 2 : ℝ) := root_pow_eq_rpow hs 2 3 (by norm_num)
  rw [show x ^ 4 = r ^ 2 by rw [← hx2]; ring,
    show x ^ 6 = r ^ 3 by rw [← hx2]; ring,
    show x ^ 8 = r ^ 4 by rw [← hx2]; ring,
    show x ^ 10 = r ^ 5 by rw [← hx2]; ring,
    show x ^ 12 = r ^ 6 by rw [← hx2]; ring,
    hx15, hy15, hy11, hy9, hy7, hy5, hy3] at h
  exact h

end BeurlingAhlfors
