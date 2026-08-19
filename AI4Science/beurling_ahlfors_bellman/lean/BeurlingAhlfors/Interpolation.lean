import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Analysis.SpecialFunctions.Artanh
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Tactic

open Real Finset

namespace BeurlingAhlfors

lemma artanh_bounds {x : ℝ} (hx0 : 0 ≤ x) (hx1 : x < 1) (n : ℕ) :
    (∑ i ∈ range n, x ^ (2 * i + 1) / (2 * i + 1)) ≤ Real.artanh x ∧
    Real.artanh x ≤ (∑ i ∈ range n, x ^ (2 * i + 1) / (2 * i + 1))
      + x ^ (2 * n + 1) / (1 - x ^ 2) := by
  have hxmem : x ∈ Set.Icc (-1 : ℝ) 1 := by constructor <;> linarith
  rw [Real.artanh_eq_half_log hxmem]
  exact ⟨Real.sum_range_le_log_div hx0 hx1 n, Real.log_div_le_sum_range_add hx0 hx1 n⟩

lemma log_seven_identity :
    Real.log 7 = 3 * Real.log 2 - 2 * Real.artanh (1 / 15) := by
  rw [Real.artanh_eq_half_log (by norm_num : (1 / 15 : ℝ) ∈ Set.Icc (-1) 1)]
  rw [show ((1 + (1 / 15 : ℝ)) / (1 - 1 / 15)) = 8 / 7 by norm_num]
  rw [Real.log_div (by norm_num : (8 : ℝ) ≠ 0) (by norm_num : (7 : ℝ) ≠ 0)]
  rw [show (8 : ℝ) = 2 ^ (3 : ℕ) by norm_num, Real.log_pow]
  ring

lemma log_22830_identity :
    Real.log 22830 = 9 * Real.log 2 + 7 * Real.log 3 - 2 * Real.log 7
      - 2 * Real.artanh (179 / 373069) := by
  rw [Real.artanh_eq_half_log (by norm_num : (179 / 373069 : ℝ) ∈ Set.Icc (-1) 1)]
  rw [show ((1 + (179 / 373069 : ℝ)) / (1 - 179 / 373069)) = 186624 / 186445 by norm_num]
  rw [Real.log_div (by norm_num : (186624 : ℝ) ≠ 0) (by norm_num : (186445 : ℝ) ≠ 0)]
  have h22830 : (22830 : ℝ) = 2 * 3 * 5 * 761 := by norm_num
  have h186624 : (186624 : ℝ) = 2 ^ (8 : ℕ) * 3 ^ (6 : ℕ) := by norm_num
  have h186445 : (186445 : ℝ) = 5 * 7 ^ (2 : ℕ) * 761 := by norm_num
  rw [h22830, h186624, h186445]
  repeat' rw [Real.log_mul (by positivity) (by positivity)]
  repeat' rw [Real.log_pow]
  ring

lemma log_22865000_identity :
    Real.log 22865000 = 5 * Real.log 2 + 5 * Real.log 3 + 5 * Real.log 5
      - 2 * Real.artanh (287 / 9433) := by
  rw [Real.artanh_eq_half_log (by norm_num : (287 / 9433 : ℝ) ∈ Set.Icc (-1) 1)]
  rw [show ((1 + (287 / 9433 : ℝ)) / (1 - 287 / 9433)) = 4860 / 4573 by norm_num]
  rw [Real.log_div (by norm_num : (4860 : ℝ) ≠ 0) (by norm_num : (4573 : ℝ) ≠ 0)]
  have hA : (22865000 : ℝ) = 2 ^ (3 : ℕ) * 5 ^ (4 : ℕ) * 4573 := by norm_num
  have hB : (4860 : ℝ) = 2 ^ (2 : ℕ) * 3 ^ (5 : ℕ) * 5 := by norm_num
  rw [hA, hB]
  repeat' rw [Real.log_mul (by positivity) (by positivity)]
  repeat' rw [Real.log_pow]
  ring

lemma log_u_identity :
    Real.log (1523958 / 1000000) = Real.log 3 - Real.log 2
      + 2 * Real.artanh (3993 / 503993) := by
  rw [Real.artanh_eq_half_log (by norm_num : (3993 / 503993 : ℝ) ∈ Set.Icc (-1) 1)]
  rw [show ((1 + (3993 / 503993 : ℝ)) / (1 - 3993 / 503993)) = 253993 / 250000 by norm_num]
  rw [Real.log_div (by norm_num : (253993 : ℝ) ≠ 0) (by norm_num : (250000 : ℝ) ≠ 0)]
  rw [Real.log_div (by norm_num : (1523958 : ℝ) ≠ 0) (by norm_num : (1000000 : ℝ) ≠ 0)]
  have h1 : (1523958 : ℝ) = 6 * 253993 := by norm_num
  have h2 : (1000000 : ℝ) = 4 * 250000 := by norm_num
  rw [h1, h2]
  repeat' rw [Real.log_mul (by positivity) (by positivity)]
  rw [show (6 : ℝ) = 2 * 3 by norm_num, Real.log_mul (by norm_num) (by norm_num)]
  rw [show (4 : ℝ) = 2 ^ (2 : ℕ) by norm_num, Real.log_pow]
  ring

end BeurlingAhlfors

namespace BeurlingAhlfors

lemma artanh_179_bounds :
    (0.0004798040 : ℝ) < Real.artanh (179 / 373069) ∧
    Real.artanh (179 / 373069) < (0.0004798041 : ℝ) := by
  have h := artanh_bounds (x := (179 : ℝ) / 373069) (by norm_num) (by norm_num) 2
  norm_num [Finset.sum_range_succ] at h ⊢
  constructor <;> linarith

lemma artanh_287_bounds :
    (0.0304344966 : ℝ) < Real.artanh (287 / 9433) ∧
    Real.artanh (287 / 9433) < (0.0304344967 : ℝ) := by
  have h := artanh_bounds (x := (287 : ℝ) / 9433) (by norm_num) (by norm_num) 4
  norm_num [Finset.sum_range_succ] at h ⊢
  constructor <;> linarith

lemma artanh_3993_bounds :
    (0.0079228948 : ℝ) < Real.artanh (3993 / 503993) ∧
    Real.artanh (3993 / 503993) < (0.0079228949 : ℝ) := by
  have h := artanh_bounds (x := (3993 : ℝ) / 503993) (by norm_num) (by norm_num) 3
  norm_num [Finset.sum_range_succ] at h ⊢
  constructor <;> linarith


end BeurlingAhlfors

namespace BeurlingAhlfors

lemma log_two_identity : Real.log 2 = 2 * Real.artanh (1 / 3) := by
  rw [Real.artanh_eq_half_log (by norm_num : (1 / 3 : ℝ) ∈ Set.Icc (-1) 1)]
  ring_nf

lemma log_two_bounds_tight :
    (0.6931471804 : ℝ) < Real.log 2 ∧ Real.log 2 < (0.6931471806 : ℝ) := by
  have h := artanh_bounds (x := (1 : ℝ) / 3) (by norm_num) (by norm_num) 40
  rw [log_two_identity]
  norm_num [Finset.sum_range_succ] at h ⊢
  constructor <;> linarith

end BeurlingAhlfors

namespace BeurlingAhlfors

lemma log_three_identity : Real.log 3 = Real.log 2 + 2 * Real.artanh (1 / 5) := by
  rw [Real.artanh_eq_half_log (by norm_num : (1 / 5 : ℝ) ∈ Set.Icc (-1) 1)]
  rw [show ((1 + (1 / 5 : ℝ)) / (1 - 1 / 5)) = 3 / 2 by norm_num]
  rw [Real.log_div (by norm_num : (3 : ℝ) ≠ 0) (by norm_num : (2 : ℝ) ≠ 0)]
  ring

lemma log_five_identity : Real.log 5 = 2 * Real.log 2 + 2 * Real.artanh (1 / 9) := by
  rw [Real.artanh_eq_half_log (by norm_num : (1 / 9 : ℝ) ∈ Set.Icc (-1) 1)]
  rw [show ((1 + (1 / 9 : ℝ)) / (1 - 1 / 9)) = 5 / 4 by norm_num]
  rw [Real.log_div (by norm_num : (5 : ℝ) ≠ 0) (by norm_num : (4 : ℝ) ≠ 0)]
  rw [show (4 : ℝ) = 2 ^ (2 : ℕ) by norm_num, Real.log_pow]
  ring

lemma log_three_bounds_tight :
    (1.0986122884 : ℝ) < Real.log 3 ∧ Real.log 3 < (1.0986122888 : ℝ) := by
  have h := artanh_bounds (x := (1 : ℝ) / 5) (by norm_num) (by norm_num) 25
  have h2 := log_two_bounds_tight
  rw [log_three_identity]
  norm_num [Finset.sum_range_succ] at h ⊢
  constructor <;> linarith

lemma log_five_bounds_tight :
    (1.6094379120 : ℝ) < Real.log 5 ∧ Real.log 5 < (1.6094379126 : ℝ) := by
  have h := artanh_bounds (x := (1 : ℝ) / 9) (by norm_num) (by norm_num) 20
  have h2 := log_two_bounds_tight
  rw [log_five_identity]
  norm_num [Finset.sum_range_succ] at h ⊢
  constructor <;> linarith

lemma log_seven_bounds_tight :
    (1.9459101484 : ℝ) < Real.log 7 ∧ Real.log 7 < (1.9459101492 : ℝ) := by
  have h := artanh_bounds (x := (1 : ℝ) / 15) (by norm_num) (by norm_num) 18
  have h2 := log_two_bounds_tight
  rw [log_seven_identity]
  norm_num [Finset.sum_range_succ] at h ⊢
  constructor <;> linarith

lemma endpoint_log_bounds :
    (10.0358307358 : ℝ) < Real.log 22830 ∧ Real.log 22830 < (10.0358307422 : ℝ) ∧
    (16.9451179106 : ℝ) < Real.log 22865000 ∧ Real.log 22865000 < (16.9451179168 : ℝ) ∧
    (0.4213108974 : ℝ) < Real.log (1523958 / 1000000) ∧
      Real.log (1523958 / 1000000) < (0.4213108982 : ℝ) := by
  have h2 := log_two_bounds_tight
  have h3 := log_three_bounds_tight
  have h5 := log_five_bounds_tight
  have h7 := log_seven_bounds_tight
  have h179 := artanh_179_bounds
  have h287 := artanh_287_bounds
  have h3993 := artanh_3993_bounds
  rw [log_22830_identity, log_22865000_identity, log_u_identity]
  constructor
  · linarith
  constructor
  · linarith
  constructor
  · linarith
  constructor
  · linarith
  constructor <;> linarith

end BeurlingAhlfors

namespace BeurlingAhlfors

lemma interpolation_range_I_numeric :
    let c : ℝ := (3 / 10) * Real.log 22830
    let g : ℝ := Real.sqrt (c * (c - 2))
    let tstar : ℝ := g / c
    let delta : ℝ := (tstar - 3 / 5) / (1 - (3 / 5) * tstar)
    c > 2 ∧ 0 < tstar ∧ tstar < 5 / 8 ∧
      g < (1.7444519015 : ℝ) ∧
      (-(0.0315657689 : ℝ) < delta ∧ delta < -(0.0315657667 : ℝ)) := by
  dsimp
  have hlog := endpoint_log_bounds
  have hcLo : (3.0107492207 : ℝ) < (3 / 10 : ℝ) * Real.log 22830 := by linarith
  have hcHi : (3 / 10 : ℝ) * Real.log 22830 < (3.0107492227 : ℝ) := by linarith
  set c : ℝ := (3 / 10) * Real.log 22830 with hc
  set g : ℝ := Real.sqrt (c * (c - 2)) with hg
  have hc2 : 2 < c := by linarith
  have hprod : 0 ≤ c * (c - 2) := mul_nonneg (by linarith) (by linarith)
  have hg0 : 0 ≤ g := by simp [g]
  have hgLo : (1.7444518991 : ℝ) < g := by
    rw [hg, Real.lt_sqrt (by norm_num)]
    dsimp [c]
    nlinarith
  have hgHi : g < (1.7444519015 : ℝ) := by
    rw [hg, Real.sqrt_lt' (by norm_num)]
    dsimp [c]
    nlinarith
  have hc0 : 0 < c := by linarith
  have htLo : (0.5794079048 : ℝ) < g / c := by
    rw [lt_div_iff₀ hc0]
    nlinarith
  have htHi : g / c < (0.5794079061 : ℝ) := by
    rw [div_lt_iff₀ hc0]
    nlinarith
  set ts : ℝ := g / c with hts
  have hden : 0 < 1 - (3 / 5 : ℝ) * ts := by nlinarith
  have hdLo : (-(0.0315657689 : ℝ)) < (ts - 3 / 5) / (1 - (3 / 5) * ts) := by
    rw [lt_div_iff₀ hden]
    nlinarith
  have hdHi : (ts - 3 / 5) / (1 - (3 / 5) * ts) < -(0.0315657667 : ℝ) := by
    rw [div_lt_iff₀ hden]
    nlinarith
  exact ⟨hc2, by linarith, by linarith, hgHi, hdLo, hdHi⟩

end BeurlingAhlfors

namespace BeurlingAhlfors

lemma range_I_envelope_certificate_simple :
    let u : ℝ := 1523958 / 1000000
    let c : ℝ := (3 / 10) * Real.log 22830
    let g : ℝ := Real.sqrt (c * (c - 2))
    let tstar : ℝ := g / c
    g - 2 * Real.artanh tstar < Real.log u := by
  dsimp
  have hrange := interpolation_range_I_numeric
  dsimp at hrange
  rcases hrange with ⟨hc2, ht0, htHi, hgHi, hdLo, hdHi⟩
  set c : ℝ := (3 / 10) * Real.log 22830
  set g : ℝ := Real.sqrt (c * (c - 2))
  set ts : ℝ := g / c
  have htLow : (0.5794079048 : ℝ) < ts := by
    have hlog := endpoint_log_bounds
    have hcLo : (3.0107492207 : ℝ) < c := by dsimp [c]; linarith
    have hcHi : c < (3.0107492227 : ℝ) := by dsimp [c]; linarith
    have hc0 : 0 < c := by linarith
    have hgLo : (1.7444518991 : ℝ) < g := by
      rw [show g = Real.sqrt (c * (c - 2)) by rfl, Real.lt_sqrt (by norm_num)]
      nlinarith
    rw [show ts = g / c by rfl, lt_div_iff₀ hc0]
    nlinarith
  have hmono : Real.artanh (0.5794079048 : ℝ) < Real.artanh ts :=
    Real.artanh_lt_artanh (by norm_num) (by linarith) htLow
  have hseries := artanh_bounds (x := (0.5794079048 : ℝ)) (by norm_num) (by norm_num) 15
  have hu := endpoint_log_bounds
  norm_num [Finset.sum_range_succ] at hseries
  linarith

end BeurlingAhlfors

namespace BeurlingAhlfors

lemma interpolation_range_II_numeric :
    let L1 : ℝ := (3 / 16) * Real.log 22830
    let L2 : ℝ := (2 / 15) * Real.log 22865000
    let c : ℝ := (120 / 13) * (L2 - L1)
    let K : ℝ := L1 - (5 / 8) * c
    let g : ℝ := Real.sqrt (c * (c - 2))
    let tstar : ℝ := g / c
    c > 2 ∧ -(0.2969209258 : ℝ) < K ∧ K < -(0.2969209117 : ℝ) ∧
      (0.6528762097 : ℝ) < tstar ∧ tstar < (0.6528762201 : ℝ) ∧
      g < (2.2758107359 : ℝ) := by
  dsimp
  have hlog := endpoint_log_bounds
  have hL1Lo : (1.8817182629 : ℝ) < (3 / 16 : ℝ) * Real.log 22830 := by linarith
  have hL1Hi : (3 / 16 : ℝ) * Real.log 22830 < (1.8817182642 : ℝ) := by linarith
  have hL2Lo : (2.2593490547 : ℝ) < (2 / 15 : ℝ) * Real.log 22865000 := by linarith
  have hL2Hi : (2 / 15 : ℝ) * Real.log 22865000 < (2.2593490556 : ℝ) := by linarith
  set L1 : ℝ := (3 / 16) * Real.log 22830
  set L2 : ℝ := (2 / 15) * Real.log 22865000
  set c : ℝ := (120 / 13) * (L2 - L1)
  set K : ℝ := L1 - (5 / 8) * c
  set g : ℝ := Real.sqrt (c * (c - 2))
  have hcLo : (3.4858226815 : ℝ) < c := by dsimp [c]; linarith
  have hcHi : c < (3.4858227019 : ℝ) := by dsimp [c]; linarith
  have hKLo : (-(0.2969209258 : ℝ)) < K := by dsimp [K]; linarith
  have hKHi : K < -(0.2969209117 : ℝ) := by dsimp [K]; linarith
  have hc2 : 2 < c := by linarith
  have hgLo : (2.2758107135 : ℝ) < g := by
    rw [show g = Real.sqrt (c * (c - 2)) by rfl, Real.lt_sqrt (by norm_num)]
    nlinarith
  have hgHi : g < (2.2758107359 : ℝ) := by
    rw [show g = Real.sqrt (c * (c - 2)) by rfl, Real.sqrt_lt' (by norm_num)]
    nlinarith
  have hc0 : 0 < c := by linarith
  have htLo : (0.6528762097 : ℝ) < g / c := by
    rw [lt_div_iff₀ hc0]
    nlinarith
  have htHi : g / c < (0.6528762201 : ℝ) := by
    rw [div_lt_iff₀ hc0]
    nlinarith
  exact ⟨hc2, hKLo, hKHi, htLo, htHi, hgHi⟩

lemma range_II_envelope_certificate_simple :
    let u : ℝ := 1523958 / 1000000
    let L1 : ℝ := (3 / 16) * Real.log 22830
    let L2 : ℝ := (2 / 15) * Real.log 22865000
    let c : ℝ := (120 / 13) * (L2 - L1)
    let K : ℝ := L1 - (5 / 8) * c
    let g : ℝ := Real.sqrt (c * (c - 2))
    let tstar : ℝ := g / c
    K + g - 2 * Real.artanh tstar < Real.log u := by
  dsimp
  have hrange := interpolation_range_II_numeric
  dsimp at hrange
  rcases hrange with ⟨hc2, hKLo, hKHi, htLo, htHi, hgHi⟩
  set L1 : ℝ := (3 / 16) * Real.log 22830
  set L2 : ℝ := (2 / 15) * Real.log 22865000
  set c : ℝ := (120 / 13) * (L2 - L1)
  set K : ℝ := L1 - (5 / 8) * c
  set g : ℝ := Real.sqrt (c * (c - 2))
  set ts : ℝ := g / c
  have hmono : Real.artanh (0.6528762097 : ℝ) < Real.artanh ts :=
    Real.artanh_lt_artanh (by norm_num) (by linarith) htLo
  have hseries := artanh_bounds (x := (0.6528762097 : ℝ)) (by norm_num) (by norm_num) 12
  have hu := endpoint_log_bounds
  norm_num [Finset.sum_range_succ] at hseries
  linarith

/-- The third interpolation range follows from the global
`√(2 p (p-1))` estimate once `p ≥ 15/2`.  This theorem checks the final
constant comparison exactly. -/
theorem range_III_constant_certificate :
    (30 : ℝ) / 13 < (1523958 / 1000000 : ℝ) ^ 2 := by
  norm_num

end BeurlingAhlfors
