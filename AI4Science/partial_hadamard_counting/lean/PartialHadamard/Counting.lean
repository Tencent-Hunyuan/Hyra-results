import Mathlib.Analysis.SpecialFunctions.Pow.Real
import PartialHadamard.Basic
import PartialHadamard.ErrorAccumulation
import PartialHadamard.MainTerm

/-!
# From row estimates to the counting formula

`Basic.lean` proves the exact row recursion, `MainTerm.lean` evaluates the
product of the paper's row factors in closed form, and
`ErrorAccumulation.lean` controls the accumulated relative error.  This
file joins the three into the shape of the paper's explicit theorem:

`|N_{n,m} / B_{n,m} - 1| ≤ exp (∑ e_k) - 1`.

The analytic per-row estimate is **not** formalized.  It enters as the
hypothesis `hstep`, exactly as the paper's Proposition 7.2 supplies it.
So the theorem below certifies the deterministic half of the argument:
the induction, the product algebra, and the error bookkeeping, and makes
the analytic input an explicit, auditable assumption.
-/

open scoped BigOperators

namespace PartialHadamard

/-- The paper's row factor `V_k = 2^m 2^(2k-1) (2πm)^(-k/2) exp(-k²/(4m))`,
written through its logarithm so that positivity is automatic. -/
noncomputable def rowFactor (m L : ℝ) (k : ℕ) : ℝ :=
  rowMainFactor (Real.log 2) m L k

/-- The paper's main term `B_{n,m}`, as the closed form of the accumulated
product after adjoining `n-1` rows to the first one. -/
noncomputable def mainTerm (m L : ℝ) (r : ℕ) : ℝ :=
  closedMain (Real.log 2) m L r

theorem rowFactor_pos (m L : ℝ) (k : ℕ) : 0 < rowFactor m L k :=
  Real.exp_pos _

theorem mainTerm_pos (m L : ℝ) (r : ℕ) : 0 < mainTerm m L r :=
  Real.exp_pos _

/-- The first row contributes `2^m`, which is the base value of the
recursion. -/
theorem two_pow_eq_exp (m : ℕ) :
    ((2 : ℝ) ^ m) = Real.exp ((m : ℝ) * Real.log 2) := by
  rw [← Real.rpow_natCast (2 : ℝ) m,
    Real.rpow_def_of_pos (by norm_num : (0 : ℝ) < 2), mul_comm]

/-- The accumulated product of row factors, started from the `2^m` choices
for the first row, is exactly the closed-form main term. -/
theorem two_pow_mul_prod_rowFactor (m : ℕ) (L : ℝ) (r : ℕ) (hm : (m : ℝ) ≠ 0) :
    ((2 : ℝ) ^ m) * ∏ j ∈ Finset.range r, rowFactor (m : ℝ) L (j + 1)
      = mainTerm (m : ℝ) L r := by
  rw [two_pow_eq_exp]
  exact accumulatedMain_eq_closedMain (Real.log 2) (m : ℝ) L r hm

/-- **Assembled counting estimate.**

Given the analytic row estimate `hstep` (available whenever the count so
far is at least half its predicted value, which is the paper's bootstrap
hypothesis), the exact combinatorics of `Basic.lean` and the product
algebra of `MainTerm.lean` yield the paper's global relative error bound.

Indices are shifted so that `k` counts rows adjoined after the first:
`N (k+1) m` is the count of `(k+1)`-row matrices. -/
theorem relative_error_of_row_estimates
    (n m : ℕ) (hn : 1 ≤ n) (hm : (m : ℝ) ≠ 0) (L : ℝ) (ξ e : ℕ → ℝ)
    (he0 : ∀ k < n - 1, 0 ≤ e k)
    (hsum : ∑ k ∈ Finset.range (n - 1), e k ≤ (1 / 2 : ℝ))
    (hstep : ∀ k < n - 1,
      mainTerm (m : ℝ) L k / 2 ≤ (N (k + 1) m : ℝ) →
        (N (k + 2) m : ℝ) = (N (k + 1) m : ℝ) * rowFactor (m : ℝ) L (k + 1)
            * (1 + ξ k) ∧ |ξ k| ≤ e k) :
    |(N n m : ℝ) / mainTerm (m : ℝ) L (n - 1) - 1|
      ≤ Real.exp (∑ k ∈ Finset.range (n - 1), e k) - 1 := by
  classical
  set a : ℕ → ℝ := fun k ↦ (N (k + 1) m : ℝ) with ha
  set v : ℕ → ℝ := fun k ↦ rowFactor (m : ℝ) L (k + 1) with hv
  -- `a 0 * ∏_{j<k} v j` is the closed-form main term at stage `k`.
  have hmain : ∀ k, a 0 * ∏ j ∈ Finset.range k, v j = mainTerm (m : ℝ) L k := by
    intro k
    have h0 : a 0 = ((2 : ℝ) ^ m) := by simp [ha, N_one]
    rw [h0, hv]
    exact two_pow_mul_prod_rowFactor m L k hm
  have hbootstrap := bootstrap_relative_error_control_div a v ξ e (n - 1)
    (by simp [ha]) (fun k _ ↦ (rowFactor_pos _ _ _).le) he0 hsum
    (by
      intro k hk hlower
      rw [hmain k] at hlower
      exact hstep k hk hlower)
    (by rw [hmain (n - 1)]; exact ne_of_gt (mainTerm_pos _ _ _))
  have hlast : a (n - 1) = (N n m : ℝ) := by
    have : n - 1 + 1 = n := by omega
    rw [ha]
    simp only [this]
  rw [hmain (n - 1), hlast] at hbootstrap
  exact hbootstrap.2

end PartialHadamard
