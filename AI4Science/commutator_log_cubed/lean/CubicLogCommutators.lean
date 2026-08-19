import CubicLogCommutators.Algebra
import CubicLogCommutators.Constants
import CubicLogCommutators.Profile
import CubicLogCommutators.Supersolution
import CubicLogCommutators.TwoByTwo

/-!
Root import for the machine-checked core of the cubic-logarithmic commutator
argument. The formalization covers the sharp scalar `2 × 2` operator-norm
estimate, its concavity-defect consequence, the supersolution induction, the
quadratic profile arithmetic, and the final explicit constants.

It does not formalize the surrounding `B(H)` right-inverse construction or the
Banach fixed-point existence argument; see `../formalization-map.md`.
-/
