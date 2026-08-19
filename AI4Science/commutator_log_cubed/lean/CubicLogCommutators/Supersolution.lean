import CubicLogCommutators.TwoByTwo
import Mathlib.Tactic

/-!
# The monotone supersolution argument

This file formalizes Corollary 5.4 and the induction at the heart of
Proposition 5.3, independently of the operator-valued Neumann-series wrapper.
-/

set_option autoImplicit false

namespace CubicLogCommutators

noncomputable section

open scoped Matrix.Norms.L2Operator

/-- The scalar majorant map from equation (5.1). -/
def localMajorant (center left right : ℝ) : ℝ :=
  (1 / 2 : ℝ) * ‖twoByTwo center left right‖

/-- Corollary 5.4: a discrete concavity defect absorbs a source term. -/
theorem concavityCriterion
    {center left right source defect : ℝ}
    (hcenter : 0 < center) (hleft : 0 ≤ left) (hright : 0 ≤ right)
    (hdef : defect = center ^ 2 - (left ^ 2 + right ^ 2) / 2)
    (hdef0 : 0 ≤ defect) (hsource : 4 * center * source ≤ defect) :
    source + localMajorant center left right ≤ center := by
  have hnorm := twoByTwo_opNorm_defect hcenter hleft hright hdef hdef0
  have hsource' : source ≤ defect / (4 * center) := by
    rw [le_div_iff₀ (by positivity : 0 < 4 * center)]
    nlinarith
  dsimp [localMajorant]
  linarith

/-- The abstract induction behind the partial-Neumann-sum bound in Proposition 5.3. -/
theorem supersolutionInduction
    {ι : Type*} (A : (ι → ℝ) → ι → ℝ)
    (hmono : ∀ {p q : ι → ℝ}, (∀ i, p i ≤ q i) → ∀ i, A p i ≤ A q i)
    (source profile : ι → ℝ)
    (hsource : ∀ i, source i ≤ profile i)
    (hsuper : ∀ i, source i + A profile i ≤ profile i)
    (approximant : ℕ → ι → ℝ)
    (hzero : ∀ i, approximant 0 i ≤ source i)
    (hsucc : ∀ k i, approximant (k + 1) i ≤
      source i + A (approximant k) i) :
    ∀ k i, approximant k i ≤ profile i := by
  intro k
  induction k with
  | zero =>
      intro i
      exact (hzero i).trans (hsource i)
  | succ k ih =>
      intro i
      calc
        approximant (k + 1) i ≤ source i + A (approximant k) i := hsucc k i
        _ ≤ source i + A profile i := by
          gcongr
          exact hmono ih i
        _ ≤ profile i := hsuper i

end

end CubicLogCommutators
