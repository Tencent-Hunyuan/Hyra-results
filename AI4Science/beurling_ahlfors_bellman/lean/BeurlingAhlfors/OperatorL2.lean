import Mathlib.Analysis.Fourier.LpSpace
import Mathlib.MeasureTheory.Function.Holder
import Mathlib.Tactic

/-!
# The Beurling--Ahlfors Fourier multiplier on `L²`

This file adds the actual operator to the formal development.  We use the
paper's multiplier `conj ξ / ξ`, represented on `ℝ²` through the standard
identification with `ℂ`.  Its modulus is one away from the origin and zero at
the origin, so multiplication by the symbol is a contraction on Fourier-side
`L²`; conjugating by Plancherel gives the Beurling--Ahlfors `L²` operator.
-/

open MeasureTheory FourierTransform
open scoped ENNReal NNReal

namespace BeurlingAhlfors

abbrev Plane := EuclideanSpace ℝ (Fin 2)

/-- The complex coordinate attached to a frequency in `ℝ²`. -/
def frequencyComplex (ξ : Plane) : ℂ := ξ 0 + Complex.I * ξ 1

/-- The complex coordinate vanishes exactly at the zero frequency. -/
theorem frequencyComplex_eq_zero_iff (ξ : Plane) :
    frequencyComplex ξ = 0 ↔ ξ = 0 := by
  constructor
  · intro h
    ext i
    fin_cases i
    · have hre := congrArg Complex.re h
      simpa [frequencyComplex] using hre
    · have him := congrArg Complex.im h
      simpa [frequencyComplex] using him
  · rintro rfl
    simp [frequencyComplex]

/-- The Beurling--Ahlfors multiplier.  The value at zero is irrelevant on
Lebesgue `L²`; choosing zero makes the definition total and measurable. -/
noncomputable def multiplier (ξ : Plane) : ℂ :=
  if frequencyComplex ξ = 0 then 0
  else starRingEnd ℂ (frequencyComplex ξ) / frequencyComplex ξ

private theorem continuous_frequencyComplex : Continuous frequencyComplex := by
  unfold frequencyComplex
  fun_prop

/-- The multiplier is measurable. -/
theorem measurable_multiplier : Measurable multiplier := by
  unfold multiplier
  apply Measurable.ite
  · exact measurableSet_eq_fun continuous_frequencyComplex.measurable measurable_const
  · exact measurable_const
  · exact (Complex.continuous_conj.measurable.comp continuous_frequencyComplex.measurable).div
      continuous_frequencyComplex.measurable

/-- The multiplier has pointwise modulus at most one. -/
theorem norm_multiplier_le_one (ξ : Plane) : ‖multiplier ξ‖ ≤ 1 := by
  unfold multiplier
  split_ifs with h
  · simp
  · rw [Complex.norm_div, Complex.norm_conj, div_self]
    exact norm_ne_zero_iff.mpr h

/-- Away from the null singleton at the origin, the multiplier is unimodular. -/
theorem norm_multiplier_eq_one_ae :
    ∀ᵐ ξ : Plane ∂volume, ‖multiplier ξ‖ = 1 := by
  filter_upwards [MeasureTheory.Measure.ae_ne (volume : Measure Plane) (0 : Plane)] with ξ hξ
  unfold multiplier
  rw [if_neg]
  · rw [Complex.norm_div, Complex.norm_conj, div_self]
    exact norm_ne_zero_iff.mpr fun hzero ↦ hξ ((frequencyComplex_eq_zero_iff ξ).mp hzero)
  · exact fun hzero ↦ hξ ((frequencyComplex_eq_zero_iff ξ).mp hzero)

/-- The multiplier as an `L∞` class. -/
noncomputable def multiplierLp : Lp ℂ ∞ (volume : Measure Plane) :=
  (memLp_top_of_bound measurable_multiplier.aestronglyMeasurable 1
    (ae_of_all _ norm_multiplier_le_one)).toLp multiplier

@[simp]
theorem multiplierLp_coe : (multiplierLp : Plane → ℂ) =ᵐ[volume] multiplier := by
  unfold multiplierLp
  exact MemLp.coeFn_toLp _

/-- Multiplication by the multiplier on Fourier-side `L²`. -/
noncomputable def multiplierL2 :
    Lp ℂ 2 (volume : Measure Plane) →L[ℂ] Lp ℂ 2 (volume : Measure Plane) :=
  ((ContinuousLinearMap.lsmul ℂ ℂ).holderL volume ∞ 2 2) multiplierLp

/-- The `L∞` norm of the multiplier is at most one. -/
theorem norm_multiplierLp_le_one : ‖multiplierLp‖ ≤ 1 := by
  rw [Lp.norm_def, eLpNorm_congr_ae multiplierLp_coe, eLpNorm_exponent_top]
  have h :=
    eLpNormEssSup_le_of_ae_bound (μ := (volume : Measure Plane))
      (ae_of_all _ norm_multiplier_le_one)
  have h' : eLpNormEssSup multiplier (volume : Measure Plane) ≤ (1 : ℝ≥0∞) := by
    simpa using h
  simpa using ENNReal.toReal_mono (by norm_num : (1 : ℝ≥0∞) ≠ ∞) h'

/-- Fourier-side multiplication is contractive. -/
theorem norm_multiplierL2_apply_le (f : Lp ℂ 2 (volume : Measure Plane)) :
    ‖multiplierL2 f‖ ≤ ‖f‖ := by
  calc
    ‖multiplierL2 f‖
        ≤ ‖(ContinuousLinearMap.lsmul ℂ ℂ : ℂ →L[ℂ] ℂ →L[ℂ] ℂ)‖
            * ‖multiplierLp‖ * ‖f‖ := by
          simpa [multiplierL2] using
            ContinuousLinearMap.norm_holder_apply_apply_le
              (ContinuousLinearMap.lsmul ℂ ℂ : ℂ →L[ℂ] ℂ →L[ℂ] ℂ)
              multiplierLp f
    _ = ‖multiplierLp‖ * ‖f‖ := by simp
    _ ≤ 1 * ‖f‖ := by
      exact mul_le_mul_of_nonneg_right norm_multiplierLp_le_one (norm_nonneg _)
    _ = ‖f‖ := one_mul _

/-- Fourier-side multiplication by the symbol preserves the `L²` norm exactly. -/
theorem norm_multiplierL2_apply_eq (f : Lp ℂ 2 (volume : Measure Plane)) :
    ‖multiplierL2 f‖ = ‖f‖ := by
  rw [Lp.norm_def, Lp.norm_def]
  congr 1
  apply eLpNorm_congr_norm_ae
  have hholder := ContinuousLinearMap.coeFn_holder (r := 2)
    (ContinuousLinearMap.lsmul ℂ ℂ : ℂ →L[ℂ] ℂ →L[ℂ] ℂ) multiplierLp f
  filter_upwards [hholder, multiplierLp_coe, norm_multiplier_eq_one_ae] with ξ hout hm hnorm
  change ‖(ContinuousLinearMap.holder 2
      (ContinuousLinearMap.lsmul ℂ ℂ : ℂ →L[ℂ] ℂ →L[ℂ] ℂ) multiplierLp f : Plane → ℂ) ξ‖ =
    ‖(f : Plane → ℂ) ξ‖
  rw [hout, hm]
  change ‖multiplier ξ * (f : Plane → ℂ) ξ‖ = ‖(f : Plane → ℂ) ξ‖
  rw [norm_mul, hnorm, one_mul]

/-- The Beurling--Ahlfors transform on `L²`, defined by Fourier multiplier. -/
noncomputable def beurlingAhlforsL2 :
    Lp ℂ 2 (volume : Measure Plane) →L[ℂ] Lp ℂ 2 (volume : Measure Plane) :=
  (Lp.fourierTransformₗᵢ Plane ℂ).symm.toContinuousLinearEquiv.toContinuousLinearMap.comp
    (multiplierL2.comp
      (Lp.fourierTransformₗᵢ Plane ℂ).toContinuousLinearEquiv.toContinuousLinearMap)

/-- Plancherel and the unimodular multiplier give the `L²` contraction. -/
theorem norm_beurlingAhlforsL2_apply_le (f : Lp ℂ 2 (volume : Measure Plane)) :
    ‖beurlingAhlforsL2 f‖ ≤ ‖f‖ := by
  change ‖(Lp.fourierTransformₗᵢ Plane ℂ).symm
      (multiplierL2 ((Lp.fourierTransformₗᵢ Plane ℂ) f))‖ ≤ ‖f‖
  rw [(Lp.fourierTransformₗᵢ Plane ℂ).symm.norm_map]
  exact (norm_multiplierL2_apply_le _).trans_eq
    ((Lp.fourierTransformₗᵢ Plane ℂ).norm_map f)

/-- The Fourier multiplier is unimodular almost everywhere, hence the
Beurling--Ahlfors transform is an exact `L²` isometry. -/
theorem norm_beurlingAhlforsL2_apply_eq (f : Lp ℂ 2 (volume : Measure Plane)) :
    ‖beurlingAhlforsL2 f‖ = ‖f‖ := by
  change ‖(Lp.fourierTransformₗᵢ Plane ℂ).symm
      (multiplierL2 ((Lp.fourierTransformₗᵢ Plane ℂ) f))‖ = ‖f‖
  rw [(Lp.fourierTransformₗᵢ Plane ℂ).symm.norm_map, norm_multiplierL2_apply_eq,
    (Lp.fourierTransformₗᵢ Plane ℂ).norm_map]

/-- The transform bundled as a complex-linear isometry on `L²`. -/
noncomputable def beurlingAhlforsL2Isometry :
    Lp ℂ 2 (volume : Measure Plane) →ₗᵢ[ℂ] Lp ℂ 2 (volume : Measure Plane) where
  toLinearMap := beurlingAhlforsL2.toLinearMap
  norm_map' := norm_beurlingAhlforsL2_apply_eq

/-- The `L²` operator norm is exactly one, not merely at most one. -/
theorem norm_beurlingAhlforsL2_eq_one : ‖beurlingAhlforsL2‖ = 1 := by
  let f : Lp ℂ 2 (volume : Measure Plane) :=
    indicatorConstLp 2 (measurableSet_closedBall : MeasurableSet
      (Metric.closedBall (0 : Plane) 1)) measure_closedBall_lt_top.ne (1 : ℂ)
  have hfNorm : ‖f‖ ≠ 0 := by
    rw [norm_indicatorConstLp (by norm_num : (2 : ℝ≥0∞) ≠ 0)
      (by norm_num : (2 : ℝ≥0∞) ≠ ∞)]
    have hmeasure :
        (volume : Measure Plane).real (Metric.closedBall (0 : Plane) 1) ≠ 0 := by
      intro hzero
      have hzero' : (volume : Measure Plane) (Metric.closedBall (0 : Plane) 1) = 0 :=
        (measureReal_eq_zero_iff measure_closedBall_lt_top.ne).mp hzero
      exact (Metric.measure_closedBall_pos (volume : Measure Plane) 0 zero_lt_one).ne' hzero'
    simp only [norm_one, one_mul]
    rw [Real.rpow_ne_zero (measureReal_nonneg : 0 ≤
      (volume : Measure Plane).real (Metric.closedBall (0 : Plane) 1)) (by norm_num)]
    exact hmeasure
  haveI : NontrivialTopology (Lp ℂ 2 (volume : Measure Plane)) :=
    NontrivialTopology.of_exists_norm_ne_zero ⟨f, hfNorm⟩
  have hmap :
      beurlingAhlforsL2Isometry.toContinuousLinearMap = beurlingAhlforsL2 := by
    ext g
    rfl
  rw [← hmap]
  exact LinearIsometry.norm_toContinuousLinearMap beurlingAhlforsL2Isometry

/-- The paper's normalized coefficient at `p = 2` is exactly one. -/
theorem beurlingAhlforsL2_bound (f : Lp ℂ 2 (volume : Measure Plane)) :
    ‖beurlingAhlforsL2 f‖ ≤ (1523958 / 1000000 : ℝ) * (2 - 1) * ‖f‖ := by
  have h := norm_beurlingAhlforsL2_apply_le f
  nlinarith [norm_nonneg f]

end BeurlingAhlfors
