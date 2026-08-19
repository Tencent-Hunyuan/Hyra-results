import Mathlib.Analysis.Complex.Exponential
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Deterministic accumulation of relative row-extension errors

This file formalizes the finite-product argument used in the last stage of
the partial-Hadamard counting proof.  It is deliberately independent of the
Fourier-analytic estimates which produce the individual errors.

There are no axioms and no placeholders in this file.
-/

open scoped BigOperators

namespace PartialHadamard

section FiniteProducts

variable {ι : Type*}

/-- The elementary inequality
`1 - ∑ i in s, e i ≤ ∏ i in s, (1 - e i)` for errors in `[0,1]`.

Keeping this lemma separate makes the bootstrap invariant in the paper
explicit instead of hiding it in the phrase "the product stays above one
half". -/
theorem one_sub_sum_le_prod_one_sub (s : Finset ι) (e : ι → ℝ)
    (he0 : ∀ i ∈ s, 0 ≤ e i) (he1 : ∀ i ∈ s, e i ≤ 1) :
    1 - ∑ i ∈ s, e i ≤ ∏ i ∈ s, (1 - e i) := by
  classical
  induction s using Finset.induction_on with
  | empty => simp
  | @insert a s ha ih =>
      have ha0 : 0 ≤ e a := he0 a (by simp)
      have ha1 : e a ≤ 1 := he1 a (by simp)
      have hs0 : 0 ≤ ∑ i ∈ s, e i :=
        Finset.sum_nonneg fun i hi ↦ he0 i (by simp [hi])
      have hi := ih
        (fun i hi ↦ he0 i (by simp [hi]))
        (fun i hi ↦ he1 i (by simp [hi]))
      calc
        1 - ∑ i ∈ insert a s, e i
            = (1 - ∑ i ∈ s, e i) * (1 - e a)
                - (∑ i ∈ s, e i) * e a := by
                  rw [Finset.sum_insert ha]
                  ring
        _ ≤ (1 - ∑ i ∈ s, e i) * (1 - e a) := by
              nlinarith
        _ ≤ (∏ i ∈ s, (1 - e i)) * (1 - e a) := by
              exact mul_le_mul_of_nonneg_right hi (sub_nonneg.mpr ha1)
        _ = ∏ i ∈ insert a s, (1 - e i) := by
              rw [Finset.prod_insert ha]
              ring

/-- A product of relative perturbations is at least `1 -` the sum of the
absolute error bounds. -/
theorem one_sub_sum_le_prod_one_add (s : Finset ι) (ξ e : ι → ℝ)
    (he0 : ∀ i ∈ s, 0 ≤ e i) (he1 : ∀ i ∈ s, e i ≤ 1)
    (hξ : ∀ i ∈ s, |ξ i| ≤ e i) :
    1 - ∑ i ∈ s, e i ≤ ∏ i ∈ s, (1 + ξ i) := by
  classical
  calc
    1 - ∑ i ∈ s, e i ≤ ∏ i ∈ s, (1 - e i) :=
      one_sub_sum_le_prod_one_sub s e he0 he1
    _ ≤ ∏ i ∈ s, (1 + ξ i) := by
      apply Finset.prod_le_prod
      · intro i hi
        exact sub_nonneg.mpr (he1 i hi)
      · intro i hi
        have hneg : -e i ≤ ξ i := by
          have := (neg_le_of_abs_le (hξ i hi))
          exact this
        linarith

/-- The exact finite-product perturbation bound

`|∏(1+ξᵢ)-1| ≤ ∏(1+|ξᵢ|)-1`.

This version has no smallness assumptions. -/
theorem abs_prod_one_add_sub_one_le_prod (s : Finset ι) (ξ : ι → ℝ) :
    |(∏ i ∈ s, (1 + ξ i)) - 1| ≤ (∏ i ∈ s, (1 + |ξ i|)) - 1 := by
  classical
  induction s using Finset.induction_on with
  | empty => simp
  | @insert a s ha ih =>
      let P : ℝ := ∏ i ∈ s, (1 + ξ i)
      let Q : ℝ := ∏ i ∈ s, (1 + |ξ i|)
      have hQ : 1 ≤ Q := by
        dsimp [Q]
        exact Finset.one_le_prod (fun i _ ↦ by linarith [abs_nonneg (ξ i)])
      have hfac : |1 + ξ a| ≤ 1 + |ξ a| := by
        simpa [abs_one] using abs_add_le (1 : ℝ) (ξ a)
      have hstep :
          |(P - 1) * (1 + ξ a) + ξ a|
            ≤ (Q - 1) * (1 + |ξ a|) + |ξ a| := by
        calc
          |(P - 1) * (1 + ξ a) + ξ a|
              ≤ |P - 1| * |1 + ξ a| + |ξ a| := by
                  simpa [abs_mul] using
                    abs_add_le ((P - 1) * (1 + ξ a)) (ξ a)
          _ ≤ (Q - 1) * (1 + |ξ a|) + |ξ a| := by
                exact add_le_add_left
                  (mul_le_mul (by simpa [P, Q] using ih) hfac
                    (abs_nonneg _) (sub_nonneg.mpr hQ)) _
      rw [Finset.prod_insert ha, Finset.prod_insert ha]
      change |(1 + ξ a) * P - 1| ≤ (1 + |ξ a|) * Q - 1
      calc
        |(1 + ξ a) * P - 1|
            = |(P - 1) * (1 + ξ a) + ξ a| := by ring_nf
        _ ≤ (Q - 1) * (1 + |ξ a|) + |ξ a| := hstep
        _ = (1 + |ξ a|) * Q - 1 := by ring

/-- Exponential form of the preceding finite-product bound. -/
theorem abs_prod_one_add_sub_one_le_exp (s : Finset ι) (ξ : ι → ℝ) :
    |(∏ i ∈ s, (1 + ξ i)) - 1|
      ≤ Real.exp (∑ i ∈ s, |ξ i|) - 1 := by
  classical
  calc
    |(∏ i ∈ s, (1 + ξ i)) - 1|
        ≤ (∏ i ∈ s, (1 + |ξ i|)) - 1 :=
          abs_prod_one_add_sub_one_le_prod s ξ
    _ ≤ Real.exp (∑ i ∈ s, |ξ i|) - 1 :=
      sub_le_sub_right
        (Real.prod_one_add_le_exp_sum s (fun i ↦ abs_nonneg (ξ i))) 1

/-- The paper's two conclusions from individual bounds `|ξᵢ| ≤ eᵢ` and
total error at most `1/2`: the multiplicative correction remains at least
`1/2`, and its distance from `1` is at most `exp (∑eᵢ)-1`. -/
theorem product_relative_error_control (s : Finset ι) (ξ e : ι → ℝ)
    (he0 : ∀ i ∈ s, 0 ≤ e i) (hξ : ∀ i ∈ s, |ξ i| ≤ e i)
    (hsum : ∑ i ∈ s, e i ≤ (1 / 2 : ℝ)) :
    (1 / 2 : ℝ) ≤ ∏ i ∈ s, (1 + ξ i) ∧
      |(∏ i ∈ s, (1 + ξ i)) - 1|
        ≤ Real.exp (∑ i ∈ s, e i) - 1 := by
  classical
  have he1 : ∀ i ∈ s, e i ≤ 1 := by
    intro i hi
    have hle : e i ≤ ∑ j ∈ s, e j :=
      Finset.single_le_sum (fun j hj ↦ he0 j hj) hi
    linarith
  constructor
  · calc
      (1 / 2 : ℝ) ≤ 1 - ∑ i ∈ s, e i := by linarith
      _ ≤ ∏ i ∈ s, (1 + ξ i) :=
        one_sub_sum_le_prod_one_add s ξ e he0 he1 hξ
  · calc
      |(∏ i ∈ s, (1 + ξ i)) - 1|
          ≤ Real.exp (∑ i ∈ s, |ξ i|) - 1 :=
            abs_prod_one_add_sub_one_le_exp s ξ
      _ ≤ Real.exp (∑ i ∈ s, e i) - 1 := by
            exact sub_le_sub_right
              (Real.exp_le_exp.mpr
                (Finset.sum_le_sum fun i hi ↦ hξ i hi)) 1

end FiniteProducts

section Recurrences

/-- Unroll a multiplicative recurrence exactly.  This is the algebraic
content behind writing the row-by-row count as a main term times a product
of relative corrections. -/
theorem unroll_multiplicative_recurrence
    (a v ξ : ℕ → ℝ) (n : ℕ)
    (hstep : ∀ k < n, a (k + 1) = a k * v k * (1 + ξ k)) :
    a n = a 0 * (∏ k ∈ Finset.range n, v k) *
      ∏ k ∈ Finset.range n, (1 + ξ k) := by
  induction n with
  | zero => simp
  | succ n ih =>
      have ih' := ih (fun k hk ↦ hstep k (Nat.lt_succ_of_lt hk))
      rw [hstep n (Nat.lt_succ_self n), ih', Finset.prod_range_succ,
        Finset.prod_range_succ]
      ring

/-- An unconditional row-step estimate implies the advertised global
relative error.  The analytic part of the paper is isolated in `hstep` and
`hξ`; this theorem proves all remaining deterministic algebra. -/
theorem global_relative_error_of_step_estimates
    (a v ξ e : ℕ → ℝ) (n : ℕ)
    (ha0 : a 0 ≠ 0) (hv : ∀ k < n, v k ≠ 0)
    (hstep : ∀ k < n, a (k + 1) = a k * v k * (1 + ξ k))
    (hξ : ∀ k < n, |ξ k| ≤ e k) :
    |a n / (a 0 * ∏ k ∈ Finset.range n, v k) - 1|
      ≤ Real.exp (∑ k ∈ Finset.range n, e k) - 1 := by
  classical
  have hprod : (∏ k ∈ Finset.range n, v k) ≠ 0 := by
    exact Finset.prod_ne_zero_iff.mpr fun k hk ↦ hv k (Finset.mem_range.mp hk)
  rw [unroll_multiplicative_recurrence a v ξ n hstep]
  rw [mul_div_cancel_left₀ _ (mul_ne_zero ha0 hprod)]
  simpa using abs_prod_one_add_sub_one_le_exp (Finset.range n) ξ |>.trans
    (sub_le_sub_right
      (Real.exp_le_exp.mpr
        (Finset.sum_le_sum fun k hk ↦ hξ k (Finset.mem_range.mp hk))) 1)

/-- Bootstrap form of the row-extension argument.

The one-step estimate is assumed only when the lower bound already proved at
the previous stages is available.  The conclusion both maintains that lower
bound and represents the final answer by a relative correction.  Writing the
relative error as a factor instead of a quotient makes the statement valid
even when the main term vanishes. -/
theorem bootstrap_relative_error_control
    (a v ξ e : ℕ → ℝ) (n : ℕ)
    (ha0 : 0 ≤ a 0) (hv0 : ∀ k < n, 0 ≤ v k)
    (he0 : ∀ k < n, 0 ≤ e k)
    (hsum : ∑ k ∈ Finset.range n, e k ≤ (1 / 2 : ℝ))
    (hstep : ∀ k < n,
      (a 0 * ∏ j ∈ Finset.range k, v j) / 2 ≤ a k →
        a (k + 1) = a k * v k * (1 + ξ k) ∧ |ξ k| ≤ e k) :
    (a 0 * ∏ k ∈ Finset.range n, v k) / 2 ≤ a n ∧
      ∃ δ : ℝ,
        a n = (a 0 * ∏ k ∈ Finset.range n, v k) * (1 + δ) ∧
          |δ| ≤ Real.exp (∑ k ∈ Finset.range n, e k) - 1 := by
  classical
  let M : ℕ → ℝ := fun k ↦ a 0 * ∏ j ∈ Finset.range k, v j
  let P : ℕ → ℝ := fun k ↦ ∏ j ∈ Finset.range k, (1 + ξ j)
  have hprefix : ∀ k, k ≤ n → M k / 2 ≤ a k ∧ a k = M k * P k := by
    intro k
    induction k using Nat.strong_induction_on with
    | h k ih =>
        intro hkn
        cases k with
        | zero =>
            constructor
            · simp only [M, Finset.range_zero, Finset.prod_empty, mul_one]
              linarith
            · simp [M, P]
        | succ k =>
            have hklt : k < n := Nat.lt_of_succ_le hkn
            have hkle : k ≤ n := Nat.le_trans (Nat.le_succ k) hkn
            obtain ⟨hklower, hkeq⟩ := ih k (Nat.lt_succ_self k) hkle
            obtain ⟨hrec, hξk⟩ := hstep k hklt (by simpa [M] using hklower)
            have he0' : ∀ j ∈ Finset.range (k + 1), 0 ≤ e j := by
              intro j hj
              exact he0 j (lt_of_lt_of_le (Finset.mem_range.mp hj) hkn)
            have hξ' : ∀ j ∈ Finset.range (k + 1), |ξ j| ≤ e j := by
              intro j hj
              have hjk : j < k + 1 := Finset.mem_range.mp hj
              have hjn : j < n := lt_of_lt_of_le hjk hkn
              have hjlower := (ih j hjk (Nat.le_of_lt hjn)).1
              exact (hstep j hjn (by simpa [M] using hjlower)).2
            have hsum' :
                ∑ j ∈ Finset.range (k + 1), e j ≤ (1 / 2 : ℝ) := by
              exact (Finset.sum_le_sum_of_subset_of_nonneg
                (Finset.range_mono hkn)
                (fun j hj _ ↦ he0 j (Finset.mem_range.mp hj))).trans hsum
            have hcontrol := product_relative_error_control
              (Finset.range (k + 1)) ξ e he0' hξ' hsum'
            have hcurrent : a (k + 1) = M (k + 1) * P (k + 1) := by
              rw [hrec, hkeq]
              simp only [M, P, Finset.prod_range_succ]
              ring
            have hM0 : 0 ≤ M (k + 1) := by
              dsimp [M]
              exact mul_nonneg ha0 (Finset.prod_nonneg fun j hj ↦
                hv0 j (lt_of_lt_of_le (Finset.mem_range.mp hj) hkn))
            constructor
            · calc
                M (k + 1) / 2 = M (k + 1) * (1 / 2 : ℝ) := by ring
                _ ≤ M (k + 1) * P (k + 1) :=
                  mul_le_mul_of_nonneg_left (by simpa [P] using hcontrol.1) hM0
                _ = a (k + 1) := hcurrent.symm
            · exact hcurrent
  have hn := hprefix n (Nat.le_refl n)
  have he0' : ∀ k ∈ Finset.range n, 0 ≤ e k :=
    fun k hk ↦ he0 k (Finset.mem_range.mp hk)
  have hξ' : ∀ k ∈ Finset.range n, |ξ k| ≤ e k := by
    intro k hk
    have hkn := Finset.mem_range.mp hk
    have hklower := (hprefix k (Nat.le_of_lt hkn)).1
    exact (hstep k hkn (by simpa [M] using hklower)).2
  have hcontrol := product_relative_error_control
    (Finset.range n) ξ e he0' hξ' hsum
  constructor
  · simpa [M] using hn.1
  · refine ⟨P n - 1, ?_, ?_⟩
    · calc
        a n = M n * P n := hn.2
        _ = M n * (1 + (P n - 1)) := by ring
        _ = (a 0 * ∏ k ∈ Finset.range n, v k) *
              (1 + (P n - 1)) := by rfl
    · simpa [P] using hcontrol.2

/-- Quotient form of `bootstrap_relative_error_control`, for a nonzero main
term. -/
theorem bootstrap_relative_error_control_div
    (a v ξ e : ℕ → ℝ) (n : ℕ)
    (ha0 : 0 ≤ a 0) (hv0 : ∀ k < n, 0 ≤ v k)
    (he0 : ∀ k < n, 0 ≤ e k)
    (hsum : ∑ k ∈ Finset.range n, e k ≤ (1 / 2 : ℝ))
    (hstep : ∀ k < n,
      (a 0 * ∏ j ∈ Finset.range k, v j) / 2 ≤ a k →
        a (k + 1) = a k * v k * (1 + ξ k) ∧ |ξ k| ≤ e k)
    (hmain : a 0 * ∏ k ∈ Finset.range n, v k ≠ 0) :
    (a 0 * ∏ k ∈ Finset.range n, v k) / 2 ≤ a n ∧
      |a n / (a 0 * ∏ k ∈ Finset.range n, v k) - 1|
        ≤ Real.exp (∑ k ∈ Finset.range n, e k) - 1 := by
  obtain ⟨hlower, δ, hδeq, hδ⟩ :=
    bootstrap_relative_error_control a v ξ e n ha0 hv0 he0 hsum hstep
  refine ⟨hlower, ?_⟩
  rw [hδeq, mul_div_cancel_left₀ _ hmain]
  simpa using hδ

end Recurrences

end PartialHadamard
