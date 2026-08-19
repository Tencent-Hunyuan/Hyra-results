import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Analysis.SpecialFunctions.Artanh
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Tactic

/-!
# Calculus behind the interpolation envelopes

This file proves the one-variable maximization used after Riesz--Thorin.  The
exact endpoint and decimal certificates are in `Interpolation.lean`.
-/

open Set

namespace BeurlingAhlfors

/-- The logarithmic interpolation envelope after the change of variables
`t = 1 - 2/p`. -/
noncomputable def interpolationEnvelope (c K t : ℝ) : ℝ :=
  K + c * t - 2 * Real.artanh t

/-- Derivative of the interpolation envelope on `(-1,1)`. -/
theorem hasDerivAt_interpolationEnvelope {c K t : ℝ} (ht : t ∈ Set.Ioo (-1 : ℝ) 1) :
    HasDerivAt (interpolationEnvelope c K) (c - 2 / (1 - t ^ 2)) t := by
  have h := Real.hasDerivAt_half_log_one_add_div_one_sub_sub_sum_range 0 ht.1 ht.2
  simp only [Finset.sum_range_zero, sub_zero, pow_zero] at h
  have ha : Real.artanh =ᶠ[nhds t]
      (fun x => 1 / 2 * Real.log ((1 + x) / (1 - x))) := by
    filter_upwards [Ioo_mem_nhds ht.1 ht.2] with x hx
    exact Real.artanh_eq_half_log ⟨hx.1.le, hx.2.le⟩
  have hArt : HasDerivAt Real.artanh (1 / (1 - t ^ 2)) t :=
    h.congr_of_eventuallyEq ha
  have hlin0 := ((hasDerivAt_id t).const_mul c).const_add K
  have heqc : c * (1 : ℝ) = c := by ring
  rw [heqc] at hlin0
  have hlin : HasDerivAt (fun x : ℝ => K + c * x) c t := by
    simpa only [id_eq] using hlin0
  have htwo : HasDerivAt (fun x : ℝ => 2 * Real.artanh x)
      (2 / (1 - t ^ 2)) t := by
    have h' := hArt.const_mul 2
    have heq : 2 * (1 / (1 - t ^ 2)) = 2 / (1 - t ^ 2) := by ring
    rw [heq] at h'
    exact h'
  have hsub := hlin.sub htwo
  exact hsub.congr_of_eventuallyEq (Filter.Eventually.of_forall fun x => by
    rfl)

/-- The unique critical point of `c t - 2 artanh t` for `c > 2`. -/
noncomputable def criticalPoint (c : ℝ) : ℝ := Real.sqrt (1 - 2 / c)

/-- The square-root form of the critical-point identity. -/
theorem criticalPoint_sq {c : ℝ} (hc : 2 < c) :
    criticalPoint c ^ 2 = 1 - 2 / c := by
  unfold criticalPoint
  apply Real.sq_sqrt
  have hc0 : 0 < c := by linarith
  rw [sub_nonneg, div_le_one₀ hc0]
  linarith

/-- The critical point lies in `(0,1)`. -/
theorem criticalPoint_mem_Ioo {c : ℝ} (hc : 2 < c) :
    criticalPoint c ∈ Set.Ioo 0 1 := by
  have hc0 : 0 < c := by linarith
  have hs : 0 < 1 - 2 / c := by
    rw [sub_pos, div_lt_one hc0]
    linarith
  constructor
  · exact Real.sqrt_pos.2 hs
  · unfold criticalPoint
    rw [Real.sqrt_lt' zero_lt_one]
    nlinarith [div_pos (show (0 : ℝ) < 2 by norm_num) hc0]

/-- The derivative is positive before the critical point. -/
theorem interpolationEnvelope_deriv_pos {c t : ℝ} (hc : 2 < c)
    (ht0 : 0 ≤ t) (ht : t < criticalPoint c) :
    0 < c - 2 / (1 - t ^ 2) := by
  have hcp0 := (criticalPoint_mem_Ioo hc).1
  have htsq : t ^ 2 < criticalPoint c ^ 2 := by nlinarith
  rw [criticalPoint_sq hc] at htsq
  have hc0 : 0 < c := by linarith
  have hfrac : 0 < 2 / c := div_pos (by norm_num) hc0
  have hden : 0 < 1 - t ^ 2 := by nlinarith
  rw [sub_pos, div_lt_iff₀ hden]
  have hdiv : 2 / c < 1 - t ^ 2 := by nlinarith
  have hmul : 2 < (1 - t ^ 2) * c := (div_lt_iff₀ hc0).1 hdiv
  simpa [mul_comm] using hmul

/-- The derivative is negative after the critical point. -/
theorem interpolationEnvelope_deriv_neg {c t : ℝ} (hc : 2 < c)
    (ht : criticalPoint c < t) (ht1 : t < 1) :
    c - 2 / (1 - t ^ 2) < 0 := by
  have hcp0 := (criticalPoint_mem_Ioo hc).1
  have ht0 : 0 < t := hcp0.trans ht
  have htsq : criticalPoint c ^ 2 < t ^ 2 := by nlinarith
  rw [criticalPoint_sq hc] at htsq
  have hden : 0 < 1 - t ^ 2 := by nlinarith
  rw [sub_neg, lt_div_iff₀ hden]
  have hc0 : 0 < c := by linarith
  have hdiv : 1 - t ^ 2 < 2 / c := by nlinarith
  have hmul : (1 - t ^ 2) * c < 2 := (lt_div_iff₀ hc0).1 hdiv
  simpa [mul_comm] using hmul

/-- On an interval containing the critical point, the interpolation envelope
attains its maximum there. -/
theorem interpolationEnvelope_max_at_criticalPoint
    {c K a b : ℝ} (hc : 2 < c)
    (ha : 0 ≤ a) (hab : a ≤ criticalPoint c)
    (hcb : criticalPoint c ≤ b) (hb : b < 1) :
    ∀ t ∈ Set.Icc a b,
      interpolationEnvelope c K t ≤ interpolationEnvelope c K (criticalPoint c) := by
  have hcp0 := (criticalPoint_mem_Ioo hc).1
  have hcp1 := (criticalPoint_mem_Ioo hc).2
  intro t ht
  rcases le_total t (criticalPoint c) with hleft | hright
  · by_cases hEq : t = criticalPoint c
    · simp [hEq]
    · have hlt : t < criticalPoint c := lt_of_le_of_ne hleft hEq
      have hmono : StrictMonoOn (interpolationEnvelope c K)
          (Set.Icc a (criticalPoint c)) := by
        apply strictMonoOn_of_deriv_pos (convex_Icc _ _)
        · intro x hx
          exact (hasDerivAt_interpolationEnvelope
            ⟨by linarith [ha, hx.1], by linarith [hcp1, hx.2]⟩).continuousAt.continuousWithinAt
        · intro x hx
          rw [interior_Icc] at hx
          rw [(hasDerivAt_interpolationEnvelope
            ⟨by linarith [ha, hx.1], by linarith [hcp1, hx.2]⟩).deriv]
          exact interpolationEnvelope_deriv_pos hc (by linarith [ha, hx.1]) hx.2
      exact (hmono ⟨ht.1, hleft⟩ ⟨hab, le_rfl⟩ hlt).le
  · by_cases hEq : t = criticalPoint c
    · simp [hEq]
    · have hgt : criticalPoint c < t := lt_of_le_of_ne hright (Ne.symm hEq)
      have hanti : StrictAntiOn (interpolationEnvelope c K)
          (Set.Icc (criticalPoint c) b) := by
        apply strictAntiOn_of_deriv_neg (convex_Icc _ _)
        · intro x hx
          exact (hasDerivAt_interpolationEnvelope
            ⟨by linarith [hcp0, hx.1], by linarith [hb, hx.2]⟩).continuousAt.continuousWithinAt
        · intro x hx
          rw [interior_Icc] at hx
          rw [(hasDerivAt_interpolationEnvelope
            ⟨by linarith [hcp0, hx.1], by linarith [hb, hx.2]⟩).deriv]
          exact interpolationEnvelope_deriv_neg hc hx.1 (by linarith [hb, hx.2])
      exact (hanti ⟨le_rfl, hcb⟩ ⟨hright, ht.2⟩ hgt).le

end BeurlingAhlfors
