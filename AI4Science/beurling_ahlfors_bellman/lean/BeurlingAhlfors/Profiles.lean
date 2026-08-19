import Mathlib.Tactic

/-!
# Exact Bellman-profile certificates

The paper's two endpoint Bellman functions reduce their drift conditions to
explicit polynomial identities and nonnegativity statements.  This file checks
those certificates over the real numbers.
-/

open Polynomial

namespace BeurlingAhlfors

section GeneralProfile

variable {p a₀ b₂ b₃ b₄ b₅ b₆ t : ℝ}

/-- Profile with the five monomials needed for both endpoint certificates.
Unused coefficients may be set to zero. -/
def profile (a₀ b₂ b₃ b₄ b₅ b₆ t : ℝ) : ℝ :=
  a₀ - b₂ * t ^ 2 - b₃ * t ^ 3 - b₄ * t ^ 4 - b₅ * t ^ 5 - b₆ * t ^ 6

def profileDeriv (b₂ b₃ b₄ b₅ b₆ t : ℝ) : ℝ :=
  -(2 * b₂ * t + 3 * b₃ * t ^ 2 + 4 * b₄ * t ^ 3
    + 5 * b₅ * t ^ 4 + 6 * b₆ * t ^ 5)

def profileSecond (b₂ b₃ b₄ b₅ b₆ t : ℝ) : ℝ :=
  -(2 * b₂ + 6 * b₃ * t + 12 * b₄ * t ^ 2
    + 20 * b₅ * t ^ 3 + 30 * b₆ * t ^ 4)

def driftP1 (_p _a₀ b₂ b₃ b₄ b₅ b₆ t : ℝ) : ℝ :=
  t * profileSecond b₂ b₃ b₄ b₅ b₆ t + profileDeriv b₂ b₃ b₄ b₅ b₆ t

def driftP2 (p a₀ b₂ b₃ b₄ b₅ b₆ t : ℝ) : ℝ :=
  p ^ 2 * t * profile a₀ b₂ b₃ b₄ b₅ b₆ t
    - (2 * p - 1) * t ^ 2 * profileDeriv b₂ b₃ b₄ b₅ b₆ t
    + t ^ 3 * profileSecond b₂ b₃ b₄ b₅ b₆ t
    + profileDeriv b₂ b₃ b₄ b₅ b₆ t

def driftP3 (p _a₀ b₂ b₃ b₄ b₅ b₆ t : ℝ) : ℝ :=
  t ^ 2 * profileSecond b₂ b₃ b₄ b₅ b₆ t
    - (p - 1) * t * profileDeriv b₂ b₃ b₄ b₅ b₆ t
    - profileDeriv b₂ b₃ b₄ b₅ b₆ t

def driftQ (p _a₀ b₂ b₃ b₄ b₅ b₆ t : ℝ) : ℝ :=
  t ^ 2 * profileSecond b₂ b₃ b₄ b₅ b₆ t
    - (p - 1) * t * profileDeriv b₂ b₃ b₄ b₅ b₆ t

def driftDet (p a₀ b₂ b₃ b₄ b₅ b₆ t : ℝ) : ℝ :=
  driftP1 p a₀ b₂ b₃ b₄ b₅ b₆ t * driftP2 p a₀ b₂ b₃ b₄ b₅ b₆ t
    - driftP3 p a₀ b₂ b₃ b₄ b₅ b₆ t ^ 2

end GeneralProfile

section Quintic

/-- The paper's quintic profile at `p = 16/3`. -/
def quinticProfile (t : ℝ) : ℝ :=
  profile 729 20736 61440 56320 4320 0 t

def quinticPi1 (t : ℝ) : ℝ :=
  13500 * t ^ 3 + 112640 * t ^ 2 + 69120 * t + 10368

def quinticPi2 (t : ℝ) : ℝ :=
  540 * t ^ 5 + 112640 * t ^ 4 + 400620 * t ^ 3
    + 512640 * t ^ 2 + 207360 * t + 23328

def quinticPi3 (t : ℝ) : ℝ :=
  2700 * t ^ 4 + 120740 * t ^ 3 + 245760 * t ^ 2
    + 120960 * t + 15552

def quinticQPoly (t : ℝ) : ℝ :=
  1080 * t ^ 3 + 45056 * t ^ 2 + 64512 * t + 20736

def quinticN (t : ℝ) : ℝ :=
  464734800 * t ^ 4 + 1114106400 * t ^ 3 - 80778240 * t ^ 2
    - 144771840 * t + 24753600

def quinticM (t : ℝ) : ℝ :=
  6454 * t ^ 4 + 15473 * t ^ 3 - 1122 * t ^ 2 - 2011 * t + 343

def quinticQ1 (t : ℝ) : ℝ := 5680 * t ^ 2 - 2782 * t + 343

def quinticQ2 (t : ℝ) : ℝ := 15429 * t ^ 2 - 6802 * t + 771

theorem quintic_drift_identities (t : ℝ) :
    driftP1 (16 / 3) 729 20736 61440 56320 4320 0 t = -8 * t * quinticPi1 t ∧
    driftP2 (16 / 3) 729 20736 61440 56320 4320 0 t = -(8 / 9) * t * quinticPi2 t ∧
    driftP3 (16 / 3) 729 20736 61440 56320 4320 0 t = (8 / 3) * t * quinticPi3 t ∧
    driftQ (16 / 3) 729 20736 61440 56320 4320 0 t = (20 / 3) * t ^ 2 * quinticQPoly t := by
  unfold driftP1 driftP2 driftP3 driftQ profile profileDeriv profileSecond
    quinticPi1 quinticPi2 quinticPi3 quinticQPoly
  constructor
  · ring
  constructor
  · ring
  constructor <;> ring

theorem quintic_det_identity (t : ℝ) :
    driftDet (16 / 3) 729 20736 61440 56320 4320 0 t
      = (128 / 9) * t ^ 5 * quinticN t := by
  unfold driftDet driftP1 driftP2 driftP3 profile profileDeriv profileSecond quinticN
  ring

theorem quintic_N_split (t : ℝ) :
    quinticN t = 72000 * quinticM t
      + (46800 * t ^ 4 + 50400 * t ^ 3 + 5760 * t ^ 2 + 20160 * t + 57600) := by
  unfold quinticN quinticM
  ring

theorem quintic_M_split (t : ℝ) :
    quinticM t = 6454 * t ^ 4 + 44 * t ^ 3 + quinticQ1 t + t * quinticQ2 t := by
  unfold quinticM quinticQ1 quinticQ2
  ring

theorem quintic_Q1_nonneg (t : ℝ) : 0 ≤ quinticQ1 t := by
  unfold quinticQ1
  nlinarith [sq_nonneg (5680 * t - 1391)]

theorem quintic_Q2_nonneg (t : ℝ) : 0 ≤ quinticQ2 t := by
  unfold quinticQ2
  nlinarith [sq_nonneg (15429 * t - 3401)]

theorem quintic_N_nonneg {t : ℝ} (ht : 0 ≤ t) : 0 ≤ quinticN t := by
  have hq1 := quintic_Q1_nonneg t
  have hq2 := quintic_Q2_nonneg t
  rw [quintic_N_split, quintic_M_split]
  positivity

theorem quintic_drift_certificate {t : ℝ} (ht : 0 ≤ t) :
    profileDeriv 20736 61440 56320 4320 0 t ≤ 0 ∧
    driftP1 (16 / 3) 729 20736 61440 56320 4320 0 t ≤ 0 ∧
    driftP2 (16 / 3) 729 20736 61440 56320 4320 0 t ≤ 0 ∧
    0 ≤ driftQ (16 / 3) 729 20736 61440 56320 4320 0 t ∧
    0 ≤ driftDet (16 / 3) 729 20736 61440 56320 4320 0 t := by
  have hderiv : profileDeriv 20736 61440 56320 4320 0 t ≤ 0 := by
    unfold profileDeriv
    have hinner :
        2 * 20736 * t + 3 * 61440 * t ^ 2 + 4 * 56320 * t ^ 3
          + 5 * 4320 * t ^ 4 + 6 * 0 * t ^ 5 ≥ 0 := by
      positivity
    linarith
  refine ⟨hderiv, ?_⟩
  rcases quintic_drift_identities t with ⟨h1, h2, h3, hq⟩
  rw [h1, h2, hq, quintic_det_identity]
  have hn := quintic_N_nonneg ht
  unfold quinticPi1 quinticPi2 quinticQPoly
  constructor
  · have hpi1 : 0 ≤ 13500 * t ^ 3 + 112640 * t ^ 2 + 69120 * t + 10368 := by
      positivity
    nlinarith
  constructor
  · have hpi2 : 0 ≤ 540 * t ^ 5 + 112640 * t ^ 4 + 400620 * t ^ 3
        + 512640 * t ^ 2 + 207360 * t + 23328 := by
      positivity
    nlinarith
  constructor
  · positivity
  · positivity

end Quintic

section Sextic

/-- The paper's sextic profile at `p = 15/2`. -/
def sexticProfile (t : ℝ) : ℝ :=
  profile 12 675 3300 6600 5423 2640 t

def sexticTheta1 (t : ℝ) : ℝ :=
  19008 * t ^ 4 + 27115 * t ^ 3 + 21120 * t ^ 2 + 5940 * t + 540

def sexticTheta2 (t : ℝ) : ℝ :=
  4752 * t ^ 6 + 27115 * t ^ 5 + 77352 * t ^ 4 + 75152 * t ^ 3
    + 37455 * t ^ 2 + 7920 * t + 540

def sexticTheta3 (t : ℝ) : ℝ :=
  9504 * t ^ 5 + 33451 * t ^ 4 + 47806 * t ^ 3 + 28380 * t ^ 2
    + 6930 * t + 540

def sexticQPoly (t : ℝ) : ℝ :=
  864 * t ^ 4 + 2465 * t ^ 3 + 3360 * t ^ 2 + 1620 * t + 270

def sexticPositive (t : ℝ) : ℝ :=
  8415792 * t ^ 5 + 278226432 * t ^ 4 + 389020324 * t ^ 3
    + 231176704 * t ^ 2 + 40095165 * t + 95040

theorem sextic_drift_identities (t : ℝ) :
    driftP1 (15 / 2) 12 675 3300 6600 5423 2640 t = -5 * t * sexticTheta1 t ∧
    driftP2 (15 / 2) 12 675 3300 6600 5423 2640 t = -(5 / 4) * t * sexticTheta2 t ∧
    driftP3 (15 / 2) 12 675 3300 6600 5423 2640 t = (5 / 2) * t * sexticTheta3 t ∧
    driftQ (15 / 2) 12 675 3300 6600 5423 2640 t = (55 / 2) * t ^ 2 * sexticQPoly t := by
  unfold driftP1 driftP2 driftP3 driftQ profile profileDeriv profileSecond
    sexticTheta1 sexticTheta2 sexticTheta3 sexticQPoly
  constructor
  · ring
  constructor
  · ring
  constructor <;> ring

theorem sextic_det_identity (t : ℝ) :
    driftDet (15 / 2) 12 675 3300 6600 5423 2640 t
      = (25 / 4) * t ^ 6 * sexticPositive t := by
  unfold driftDet driftP1 driftP2 driftP3 profile profileDeriv profileSecond sexticPositive
  ring

theorem sextic_drift_certificate {t : ℝ} (ht : 0 ≤ t) :
    profileDeriv 675 3300 6600 5423 2640 t ≤ 0 ∧
    driftP1 (15 / 2) 12 675 3300 6600 5423 2640 t ≤ 0 ∧
    driftP2 (15 / 2) 12 675 3300 6600 5423 2640 t ≤ 0 ∧
    0 ≤ driftQ (15 / 2) 12 675 3300 6600 5423 2640 t ∧
    0 ≤ driftDet (15 / 2) 12 675 3300 6600 5423 2640 t := by
  have hderiv : profileDeriv 675 3300 6600 5423 2640 t ≤ 0 := by
    unfold profileDeriv
    have hinner :
        2 * 675 * t + 3 * 3300 * t ^ 2 + 4 * 6600 * t ^ 3
          + 5 * 5423 * t ^ 4 + 6 * 2640 * t ^ 5 ≥ 0 := by
      positivity
    linarith
  refine ⟨hderiv, ?_⟩
  rcases sextic_drift_identities t with ⟨h1, h2, h3, hq⟩
  rw [h1, h2, hq, sextic_det_identity]
  unfold sexticTheta1 sexticTheta2 sexticQPoly sexticPositive
  constructor
  · have htheta1 : 0 ≤ 19008 * t ^ 4 + 27115 * t ^ 3 + 21120 * t ^ 2
        + 5940 * t + 540 := by
      positivity
    nlinarith
  constructor
  · have htheta2 : 0 ≤ 4752 * t ^ 6 + 27115 * t ^ 5 + 77352 * t ^ 4
        + 75152 * t ^ 3 + 37455 * t ^ 2 + 7920 * t + 540 := by
      positivity
    nlinarith
  constructor
  · positivity
  · positivity

end Sextic

end BeurlingAhlfors
