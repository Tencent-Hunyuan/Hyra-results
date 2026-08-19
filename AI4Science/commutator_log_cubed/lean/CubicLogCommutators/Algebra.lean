import Mathlib.Tactic.NoncommRing

/-!
# Algebraic identity behind the right inverse

The paper's formula `T L = I - E` follows coordinatewise from this doubled
identity after substituting `v*`, `u*` for `vStar`, `uStar` and dividing by
two.
-/

set_option autoImplicit false

namespace CubicLogCommutators

/-- The additive commutator convention used in the paper. -/
def commutator {A : Type*} [Mul A] [Sub A] (a b : A) : A :=
  a * b - b * a

/-- The coordinatewise cancellation behind Lemma 4.1 (`T L = I - E`),
written without fractions. -/
theorem rightInverse_doubled_identity
    {A : Type*} [Ring A]
    (u v uStar vStar xPrev x xNext : A)
    (huu : uStar * u = 1) (hvv : vStar * v = 1)
    (huv : uStar * v = 0) (hvu : vStar * u = 0) :
    commutator v (-(x * vStar + xNext * uStar)) +
        commutator u (-(xPrev * vStar + x * uStar)) =
      2 * x -
        (v * x * vStar + v * xNext * uStar +
          u * xPrev * vStar + u * x * uStar) := by
  simp only [commutator, mul_add, mul_neg, add_mul, neg_mul, sub_eq_add_neg]
  simp only [mul_assoc]
  rw [huu, hvv, huv, hvu]
  simp
  noncomm_ring

end CubicLogCommutators
