import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Fintype.Powerset

/-!
# Exact combinatorics of partial Hadamard matrices

This file contains the finite, exact part of the row-extension argument.  A
sign is encoded by a Boolean (`false` is `+1`, `true` is `-1`), so the ambient
type of sign matrices is finite by construction.

There are no analytic estimates, axioms, or placeholders in this file.
-/

open scoped BigOperators

namespace PartialHadamard

/-- A sign, represented as an integer.  We use `false = +1` and `true = -1`. -/
def bitSign : Bool → ℤ
  | false => 1
  | true => -1

@[simp] theorem bitSign_false : bitSign false = 1 := rfl
@[simp] theorem bitSign_true : bitSign true = -1 := rfl

@[simp] theorem bitSign_mul_self (b : Bool) : bitSign b * bitSign b = 1 := by
  cases b <;> rfl

/-- A sign row of width `m`. -/
abbrev SignRow (m : ℕ) := Fin m → Bool

/-- An `n × m` Boolean encoding of a sign matrix. -/
abbrev SignMatrix (n m : ℕ) := Fin n → Fin m → Bool

/-- Integer inner product of two sign rows. -/
def rowDot {m : ℕ} (x y : SignRow m) : ℤ :=
  ∑ c, bitSign (x c) * bitSign (y c)

theorem rowDot_comm {m : ℕ} (x y : SignRow m) : rowDot x y = rowDot y x := by
  unfold rowDot
  apply Finset.sum_congr rfl
  intro c _
  exact mul_comm _ _

@[simp] theorem rowDot_self {m : ℕ} (x : SignRow m) : rowDot x x = m := by
  simp [rowDot, bitSign_mul_self]

/-- The rows of `H` are pairwise orthogonal. -/
def IsPartialHadamard {n m : ℕ} (H : SignMatrix n m) : Prop :=
  ∀ {i j : Fin n}, i ≠ j → rowDot (H i) (H j) = 0

/-- The finite type of `n × m` partial Hadamard sign matrices. -/
abbrev PHM (n m : ℕ) := {H : SignMatrix n m // IsPartialHadamard H}

noncomputable instance instFintypePHM (n m : ℕ) : Fintype (PHM n m) :=
  Fintype.ofFinite _

/-- The exact number of `n × m` partial Hadamard matrices. -/
noncomputable def N (n m : ℕ) : ℕ :=
  Fintype.card (PHM n m)

/-- A row is admissible after `H` if it is orthogonal to every row of `H`. -/
def IsExtension {k m : ℕ} (H : SignMatrix k m) (x : SignRow m) : Prop :=
  ∀ i, rowDot (H i) x = 0

/-- The finite type of rows that may be appended to `H`. -/
abbrev ExtensionRows {k m : ℕ} (H : PHM k m) :=
  {x : SignRow m // IsExtension H.1 x}

noncomputable instance instFintypeExtensionRows {k m : ℕ} (H : PHM k m) :
    Fintype (ExtensionRows H) :=
  Fintype.ofFinite _

/-- The exact number of admissible next rows. -/
noncomputable def X {k m : ℕ} (H : PHM k m) : ℕ :=
  Fintype.card (ExtensionRows H)

/-- Delete the last row. -/
def restrictRows {k m : ℕ} (H : SignMatrix (k + 1) m) : SignMatrix k m :=
  fun i ↦ H i.castSucc

/-- The last row. -/
def lastRow {k m : ℕ} (H : SignMatrix (k + 1) m) : SignRow m :=
  H (Fin.last k)

/-- Append a row at the end. -/
def appendRow {k m : ℕ} (H : SignMatrix k m) (x : SignRow m) :
    SignMatrix (k + 1) m :=
  fun i ↦ Fin.lastCases x H i

@[simp] theorem restrictRows_appendRow {k m : ℕ} (H : SignMatrix k m) (x : SignRow m) :
    restrictRows (appendRow H x) = H := by
  funext i c
  simp [restrictRows, appendRow]

@[simp] theorem lastRow_appendRow {k m : ℕ} (H : SignMatrix k m) (x : SignRow m) :
    lastRow (appendRow H x) = x := by
  funext c
  simp [lastRow, appendRow]

@[simp] theorem appendRow_restrictRows_lastRow {k m : ℕ} (H : SignMatrix (k + 1) m) :
    appendRow (restrictRows H) (lastRow H) = H := by
  funext i c
  cases i using Fin.lastCases <;> simp [appendRow, restrictRows, lastRow]

/-- Appending a row preserves the partial-Hadamard property exactly when the
old rows are pairwise orthogonal and the new row is admissible. -/
theorem isPartialHadamard_appendRow_iff {k m : ℕ} (H : SignMatrix k m) (x : SignRow m) :
    IsPartialHadamard (appendRow H x) ↔ IsPartialHadamard H ∧ IsExtension H x := by
  constructor
  · intro h
    constructor
    · intro i j hij
      simpa [appendRow] using
        h (i := i.castSucc) (j := j.castSucc) (fun h' ↦ hij (Fin.castSucc_inj.mp h'))
    · intro i
      simpa [appendRow] using
        h (i := i.castSucc) (j := Fin.last k) (Fin.ne_of_lt i.castSucc_lt_last)
  · rintro ⟨hH, hx⟩ i j hij
    by_cases hi : i = Fin.last k
    · subst i
      by_cases hj : j = Fin.last k
      · subst j
        exact (hij rfl).elim
      · obtain ⟨j₀, rfl⟩ := Fin.exists_castSucc_eq.mpr hj
        simpa [appendRow, rowDot_comm] using hx j₀
    · obtain ⟨i₀, rfl⟩ := Fin.exists_castSucc_eq.mpr hi
      by_cases hj : j = Fin.last k
      · subst j
        simpa [appendRow] using hx i₀
      · obtain ⟨j₀, rfl⟩ := Fin.exists_castSucc_eq.mpr hj
        simpa [appendRow] using
          hH (fun h ↦ hij (congrArg Fin.castSucc h))

/-- Splitting off the last row is an equivalence with a partial Hadamard
matrix together with one of its admissible extension rows. -/
noncomputable def splitLastEquiv (k m : ℕ) :
    PHM (k + 1) m ≃ Σ H : PHM k m, ExtensionRows H where
  toFun A := by
    have hA : IsPartialHadamard
        (appendRow (restrictRows A.1) (lastRow A.1)) := by
      rw [appendRow_restrictRows_lastRow]
      exact A.2
    have hs : IsPartialHadamard (restrictRows A.1) ∧
        IsExtension (restrictRows A.1) (lastRow A.1) :=
      (isPartialHadamard_appendRow_iff (restrictRows A.1) (lastRow A.1)).mp
        hA
    exact ⟨⟨restrictRows A.1, hs.1⟩, ⟨lastRow A.1, hs.2⟩⟩
  invFun p :=
    ⟨appendRow p.1.1 p.2.1,
      (isPartialHadamard_appendRow_iff p.1.1 p.2.1).2 ⟨p.1.2, p.2.2⟩⟩
  left_inv A := by
    apply Subtype.ext
    simp
  right_inv p := by
    rcases p with ⟨⟨H, hH⟩, ⟨x, hx⟩⟩
    apply Sigma.ext
    · apply Subtype.ext
      exact restrictRows_appendRow H x
    · simp only [lastRow_appendRow]
      apply (Subtype.heq_iff_coe_eq (fun y ↦ by simp)).2
      rfl

/-- Exact row recursion, before any Fourier analysis. -/
theorem row_recursion (k m : ℕ) :
    N (k + 1) m = ∑ H : PHM k m, X H := by
  classical
  calc
    N (k + 1) m = Fintype.card (Σ H : PHM k m, ExtensionRows H) :=
      Fintype.card_congr (splitLastEquiv k m)
    _ = ∑ H : PHM k m, X H := by simp [X]

/-- A one-row sign matrix is just an arbitrary sign row. -/
noncomputable def oneRowEquiv (m : ℕ) : PHM 1 m ≃ SignRow m where
  toFun H := H.1 0
  invFun x :=
    ⟨fun _ ↦ x, by
      intro i j hij
      exact (hij (Subsingleton.elim i j)).elim⟩
  left_inv H := by
    apply Subtype.ext
    funext i c
    exact congrArg (fun j ↦ H.1 j c) (Subsingleton.elim 0 i)
  right_inv x := rfl

/-- Exact base count: `N₁,m = 2^m`. -/
@[simp] theorem N_one (m : ℕ) : N 1 m = 2 ^ m := by
  classical
  simpa [N, SignRow] using Fintype.card_congr (oneRowEquiv m)

/-- Every extension count is bounded by the number of all sign rows. -/
theorem X_le_two_pow {k m : ℕ} (H : PHM k m) : X H ≤ 2 ^ m := by
  classical
  simpa [X, ExtensionRows, SignRow] using
    Fintype.card_subtype_le (IsExtension H.1)

/-- Trivial ambient-space bound `Nₙ,ₘ ≤ 2^(nm)`. -/
theorem N_le_two_pow (n m : ℕ) : N n m ≤ 2 ^ (n * m) := by
  classical
  have h := Fintype.card_subtype_le (@IsPartialHadamard n m)
  calc
    N n m ≤ Fintype.card (SignMatrix n m) := by simpa [N] using h
    _ = (2 ^ m) ^ n := by simp [SignMatrix]
    _ = 2 ^ (m * n) := by rw [← pow_mul]
    _ = 2 ^ (n * m) := by rw [Nat.mul_comm]

/-- The integer matrix represented by a Boolean sign matrix. -/
def toIntMatrix {n m : ℕ} (H : SignMatrix n m) : Fin n → Fin m → ℤ :=
  fun i c ↦ bitSign (H i c)

/-- Every sign column has squared Euclidean norm `n`. -/
@[simp] theorem column_sq_norm {n m : ℕ} (H : SignMatrix n m) (c : Fin m) :
    ∑ i, (toIntMatrix H i c) ^ 2 = n := by
  simp [toIntMatrix, pow_two, bitSign_mul_self]

/-- The row Gram matrix, defined entrywise without importing matrix algebra. -/
def gram {n m : ℕ} (H : SignMatrix n m) : Fin n → Fin n → ℤ :=
  fun i j ↦ ∑ c, toIntMatrix H i c * toIntMatrix H j c

/-- Entrywise form of the Gram identity. -/
theorem gram_apply {n m : ℕ} (H : PHM n m) (i j : Fin n) :
    gram H.1 i j = if i = j then (m : ℤ) else 0 := by
  change rowDot (H.1 i) (H.1 j) = if i = j then (m : ℤ) else 0
  by_cases hij : i = j
  · subst j
    simp
  · rw [H.2 hij]
    simp [hij]

/-- Exact Gram identity `H Hᵀ = m I`. -/
theorem gram_identity {n m : ℕ} (H : PHM n m) :
    gram H.1 = fun i j ↦ if i = j then (m : ℤ) else 0 := by
  funext i j
  exact gram_apply H i j

section TwoRows

/-- Coordinatewise multiplication of sign rows, encoded as Boolean XOR. -/
def rowMul {m : ℕ} (x y : SignRow m) : SignRow m :=
  fun c ↦ Bool.xor (x c) (y c)

@[simp] theorem bitSign_xor (a b : Bool) :
    bitSign (Bool.xor a b) = bitSign a * bitSign b := by
  cases a <;> cases b <;> rfl

@[simp] theorem rowMul_left_cancel {m : ℕ} (x y : SignRow m) :
    rowMul x (rowMul x y) = y := by
  funext c
  simp [rowMul]

/-- Positions carrying the sign `-1`. -/
def rowSupport {m : ℕ} (x : SignRow m) : Finset (Fin m) :=
  Finset.univ.filter fun c ↦ x c = true

/-- The row whose negative positions are exactly `s`. -/
def rowOfSupport {m : ℕ} (s : Finset (Fin m)) : SignRow m :=
  fun c ↦ decide (c ∈ s)

@[simp] theorem rowOfSupport_rowSupport {m : ℕ} (x : SignRow m) :
    rowOfSupport (rowSupport x) = x := by
  funext c
  cases h : x c <;> simp [rowOfSupport, rowSupport, h]

@[simp] theorem rowSupport_rowOfSupport {m : ℕ} (s : Finset (Fin m)) :
    rowSupport (rowOfSupport s) = s := by
  ext c
  simp [rowSupport, rowOfSupport]

/-- Sum of the signs in a row. -/
def rowSum {m : ℕ} (x : SignRow m) : ℤ :=
  ∑ c, bitSign (x c)

theorem bitSign_eq_one_sub_two_indicator (b : Bool) :
    bitSign b = 1 - 2 * (if b = true then 1 else 0) := by
  cases b <;> rfl

theorem rowSum_eq_card_support {m : ℕ} (x : SignRow m) :
    rowSum x = (m : ℤ) - 2 * (rowSupport x).card := by
  unfold rowSum
  simp_rw [bitSign_eq_one_sub_two_indicator]
  rw [Finset.sum_sub_distrib, ← Finset.mul_sum]
  simp [rowSupport]

theorem rowDot_eq_rowSum_mul {m : ℕ} (x y : SignRow m) :
    rowDot x y = rowSum (rowMul x y) := by
  unfold rowDot rowSum rowMul
  simp only [bitSign_xor]

theorem rowSum_eq_zero_iff_support_card (q : ℕ) (x : SignRow (2 * q)) :
    rowSum x = 0 ↔ (rowSupport x).card = q := by
  rw [rowSum_eq_card_support]
  constructor
  · intro h
    have hmul : (2 : ℤ) * (q : ℤ) = 2 * ((rowSupport x).card : ℤ) := by
      simpa [Nat.cast_mul] using sub_eq_zero.mp h
    have hq : (q : ℤ) = ((rowSupport x).card : ℤ) :=
      Int.eq_of_mul_eq_mul_left (show (2 : ℤ) ≠ 0 by decide) hmul
    exact (Int.ofNat_inj.mp hq).symm
  · intro h
    rw [h]
    simp [Nat.cast_mul]

theorem rowDot_mul_self_left {m : ℕ} (x y : SignRow m) :
    rowDot x (rowMul x y) = rowSum y := by
  unfold rowDot rowSum rowMul
  apply Finset.sum_congr rfl
  intro c _
  rw [bitSign_xor, ← mul_assoc, bitSign_mul_self, one_mul]

/-- For one existing row of even width, extension rows are equivalent to
choosing the `q` negative positions of the normalized next row. -/
noncomputable def extensionSupportEquiv (q : ℕ) (H : PHM 1 (2 * q)) :
    ExtensionRows H ≃ {s : Finset (Fin (2 * q)) // s.card = q} where
  toFun x :=
    ⟨rowSupport (rowMul (H.1 0) x.1), by
      apply (rowSum_eq_zero_iff_support_card q _).1
      rw [← rowDot_eq_rowSum_mul]
      exact x.2 0⟩
  invFun s :=
    ⟨rowMul (H.1 0) (rowOfSupport s.1), by
      intro i
      have hi : i = 0 := Subsingleton.elim i 0
      subst i
      rw [rowDot_mul_self_left]
      apply (rowSum_eq_zero_iff_support_card q _).2
      simpa using s.2⟩
  left_inv x := by
    apply Subtype.ext
    simp
  right_inv s := by
    apply Subtype.ext
    simp

/-- Every one-row partial Hadamard matrix of width `2q` has exactly
`choose (2q) q` admissible next rows. -/
theorem X_one_even (q : ℕ) (H : PHM 1 (2 * q)) :
    X H = Nat.choose (2 * q) q := by
  classical
  calc
    X H = Fintype.card {s : Finset (Fin (2 * q)) // s.card = q} :=
      Fintype.card_congr (extensionSupportEquiv q H)
    _ = Nat.choose (2 * q) q := by simp

/-- Exact two-row count in every even width. -/
theorem N_two_even (q : ℕ) :
    N 2 (2 * q) = 2 ^ (2 * q) * Nat.choose (2 * q) q := by
  classical
  calc
    N 2 (2 * q) = ∑ H : PHM 1 (2 * q), X H := by
      simpa using row_recursion 1 (2 * q)
    _ = Fintype.card (PHM 1 (2 * q)) * Nat.choose (2 * q) q := by
      simp [X_one_even]
    _ = 2 ^ (2 * q) * Nat.choose (2 * q) q := by
      rw [← N, N_one]

end TwoRows

end PartialHadamard
