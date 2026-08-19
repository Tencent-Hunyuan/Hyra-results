# Hyra Results

Companion artifacts for **Hyra**: **H**un**y**uan **R**esearch **A**gent.

This repository collects concrete solutions that Hyra produced across a range of
open problems in science, mathematics, engineering, and creative design,
released alongside the Hyra launch post. Each folder holds the final solution
artifact and, where relevant, the self-contained scripts that reproduce it.

> 📝 Launch post: **[hy.tencent.com/research/hyra](https://hy.tencent.com/research/hyra)**

## News

- **2026-08-19**: 🧾 **Three new results, each with a Lean 4 formalization of its
  finite and numerical core.**
  - **Beurling–Ahlfors transform.** Iwaniec's 1982 conjecture predicts the sharp
    `Lᵖ` operator norm `p*−1`; the best proven uniform coefficient drops from
    `1.575` (Bañuelos and Janakiraman, 2008) to **1.523958**. Paper:
    **[`AI4Science/beurling_ahlfors_bellman/`](AI4Science/beurling_ahlfors_bellman/)**.
  - **Partial Hadamard matrices.** Their asymptotic count is now known
    throughout every power-law range above `n²`, not just the cubic range
    `m ≫ n³`, lowering the critical exponent from **3**
    ([arXiv:2603.30013](https://arxiv.org/abs/2603.30013)) to **2**. Paper:
    **[`AI4Science/partial_hadamard_counting/`](AI4Science/partial_hadamard_counting/)**.
  - **Commutators close to the identity.** The identity is never a commutator,
    but comes within `ε` at a price: the least `‖D‖·‖X‖` needed falls from Tao's
    `O(log⁵(1/ε))`, since refined to `O(log⁴)`, to **O(log³(1/ε))**, against
    Popa's `log(1/ε)` lower bound. Paper:
    **[`AI4Science/commutator_log_cubed/`](AI4Science/commutator_log_cubed/)**.

- **2026-08-17**: 📐 **Blaschke–Lebesgue in three dimensions.** The least volume
  of a convex body of constant width `w` is conjectured to be Meissner's
  `≈ 0.419860 w³`; the best certified lower bound rises from
  `4π/33 · w³ ≈ 0.380799 w³` (Nishioka,
  [arXiv:2606.01754](https://arxiv.org/abs/2606.01754)) to **> 0.411040 w³**,
  closing **77.4%** of the remaining gap. Paper:
  **[`AI4Science/3d_blaschke_lebesgue/`](AI4Science/3d_blaschke_lebesgue/)**.

- **2026-07-29**: 🎉 **The sum-vs-difference problem is settled.** For finite
  `A ⊆ ℤ`, the optimal exponent relating `|A+A|` to `|A−A|` has supremum
  **exactly 2**, approached arbitrarily closely but never attained: a complete,
  machine-checked resolution going beyond the `sums_diffs` record below. Proof:
  **[sum-diff-proof](https://github.com/linhaowei1/sum-diff-proof)**.

## Results

Each row compares the best prior published result (**Prev best**, from the cited
system or leaderboard) against **Hyra**. Arrows mark the better direction
(↓ lower is better, ↑ higher is better); the Hyra-winning value is in bold.

| Track | Task | Metric | Prev best | Hyra |
|---|---|---|---|---|
| **AI4AI** | [`nanochat_autoresearch`](AI4AI/nanochat_autoresearch/) | val BPB ↓ | 0.9109 <sup>e</sup> | **0.9015** |
| | [`nanogpt_speedrun`](AI4AI/nanogpt_speedrun/) | wall-clock ↓ | 77.5 s <sup>e</sup> | **76.4 s** (mean val loss: **3.280**) |
| | [`sol_execbench`](AI4AI/sol_execbench/) | score ↑ | 0.754 <sup>e</sup> | **0.771** |
| **AI4Science** | [`autocorrelation_first`](AI4Science/autocorrelation_first/) | C₁ ↓ | 1.502870 <sup>a</sup> | **1.502850** |
| | [`autocorrelation_second`](AI4Science/autocorrelation_second/) | R ↑ | 0.962694 <sup>b</sup> | **0.962901** |
| | [`erdos_min_overlap`](AI4Science/erdos_min_overlap/) | C₅ ↓ | 0.380868 <sup>b</sup> | **0.380859** |
| | [`sums_diffs`](AI4Science/sums_diffs/) | C(A) ↑ | 1.14489 <sup>b</sup> | **1.21079** |
| | [`packing_records`](AI4Science/packing_records/) | records broken | n/a <sup>f</sup> | **100** |
| | [`smallest_adder`](AI4Science/smallest_adder/) | params ↓ | 36 <sup>c</sup> | **15** |
| | [`parp1_docking`](AI4Science/parp1_docking/) | objective ↓ | −9.77 <sup>d</sup> | **−10.60** |
| | [`qubit_routing`](AI4Science/qubit_routing/) | CNOTs added ↓ | 269,037 <sup>b</sup> | **258,369** |
| | [`sunspot_symbolic`](AI4Science/sunspot_symbolic/) | forecast R² ↑ | 0.47 <sup>g</sup> | **0.78** |
| | [`3d_blaschke_lebesgue`](AI4Science/3d_blaschke_lebesgue/) | Vol/w³ bound ↑ | 0.380799 <sup>h</sup> | **0.411040** |
| | [`beurling_ahlfors_bellman`](AI4Science/beurling_ahlfors_bellman/) | C_BA ↓ | 1.575 <sup>i</sup> | **1.523958** |
| | [`partial_hadamard_counting`](AI4Science/partial_hadamard_counting/) | regime exponent ↓ | 3 <sup>j</sup> | **2** |
| | [`commutator_log_cubed`](AI4Science/commutator_log_cubed/) | log exponent ↓ | 4 <sup>k</sup> | **3** |

**Prev-best sources.** 

- <sup>a</sup> TTT-Discover: *Learning to Discover at Test
Time* ([arXiv:2601.16175](https://arxiv.org/abs/2601.16175)). 
- <sup>b</sup>
SimpleTES: *Evaluation-driven Scaling for Scientific Discovery*
([arXiv:2604.19341](https://arxiv.org/abs/2604.19341)). 
- <sup>c</sup> AdderBoard
trained-weights leaderboard ([github.com/anadim/AdderBoard](https://github.com/anadim/AdderBoard)).
- <sup>d</sup> Olaparib, an approved PARP1 inhibitor (drug baseline).
- <sup>e</sup> Recursive: *First Steps Toward Automated AI Research*
([github.com/recursive-org/first-steps-toward-automated-ai-research](https://github.com/recursive-org/first-steps-toward-automated-ai-research));
AI4AI baselines: nanoGPT-speedrun, nanochat, and SOL-ExecBench.
- <sup>f</sup> Erich Friedman's *Packing Center*
([erich-friedman.github.io/packing](https://erich-friedman.github.io/packing/)).
Hyra's record-improving packings are credited there as "Found by Haowei Lin":
**100** across 28 shape-in-shape families, each beating the previously listed
best.
- <sup>g</sup> Baseline: a "copy last frame" (persistence) forecast. Hyra's score is
the forecast R² on a fully-held-out, half-century-long segment of the record.
- <sup>h</sup> Nishioka: *An improved lower bound for the three-dimensional
Blaschke–Lebesgue problem from spectral and dual perspectives*
([arXiv:2606.01754](https://arxiv.org/abs/2606.01754)), which gives `4π/33 ≈
0.380799109526`. Hyra's certified bound is `(130838246407123/10¹⁵)·π >
0.411040473721188`.
- <sup>i</sup> R. Bañuelos and P. Janakiraman, *Lᵖ-bounds for the
Beurling–Ahlfors transform* (2008), which gives the uniform coefficient
`1.575`. Iwaniec's conjectured sharp value is `1`.
- <sup>j</sup> D. Davis: *Counting Partial Hadamard Matrices in the Cubic
Regime* ([arXiv:2603.30013](https://arxiv.org/abs/2603.30013)).
- <sup>k</sup> B. Bilich: *An O(log⁴(1/ε)) refinement of Tao's construction of
commutators close to the identity*
([github.com/bilichboris/TaoCommutators](https://github.com/bilichboris/TaoCommutators));
the published exponent is Tao's `5`, since refined to `4`.

Metrics are as defined by each benchmark; comparisons are against the cited
published results.

**Metric notes.**
- **C₁**: first autoconvolution/autocorrelation constant, `max(f∗f)/(∫f)²` (minimize).
- **R**: second autocorrelation ratio, `‖f∗f‖₂² / (‖f∗f‖₁·‖f∗f‖∞)` (maximize).
- **C₅**: Erdős minimum-overlap constant (minimize).
- **C(A)**: sum-vs-difference exponent, `log(|A+A|/|A|) / log(|A−A|/|A|)` (maximize).
- **params**: unique trainable parameters of a transformer that adds two
  10-digit integers at ≥ 0.99 accuracy (minimize).
- **objective**: PARP1 docking objective, `Vina score + 10·(1 − QED)` (minimize).
- **CNOTs added**: extra CNOTs inserted by SWAP routing (minimize; 1 SWAP = 3 CNOTs).
- **forecast R²**: rolling-origin, free-running (24-month) forecast R² for monthly
  sunspot numbers (maximize).
- **Vol/w³ bound**: largest proven universal lower bound on `Vol(K)/w³` over all
  convex bodies `K ⊂ ℝ³` of constant width `w` (maximize; the conjectured
  optimum is `≈ 0.419860`).
- **C_BA**: smallest proven uniform coefficient in
  `‖B‖_{Lᵖ→Lᵖ} ≤ C_BA·(p*−1)` for the Beurling–Ahlfors transform, over all
  `1 < p < ∞` (minimize; the conjectured optimum is `1`).
- **regime exponent**: infimum of the power-law exponents `u` for which
  `m/nᵘ → ∞` is known to force the asymptotic formula for `N(n,m)` (minimize;
  the endpoint value itself is not claimed).
- **log exponent**: smallest proven `k` in `m(ε) = O(logᵏ(1/ε))` for
  commutators within `ε` of the identity (minimize; the matching lower bound
  is `k = 1`).

> **Note.** The results above were current as of **2026-07-10**, except
> `3d_blaschke_lebesgue`, added **2026-08-17**, and the three entries added
> **2026-08-19**. Several of these
> problems live on public, continuously-updated leaderboards; later entries there
> may *warm-start from Hyra's published solutions* to reach still-better numbers.

### AI4Fun

Creative and game-playing demos (no leaderboard comparison):

- [`AI4Fun/reversi/`](AI4Fun/reversi/): a **AlphaZero-style
  Reversi (Othello) bot** for the [Botzone](https://www.botzone.org.cn/) 8×8
  arena (C++ pattern/n-tuple net + PUCT-MCTS + exact endgame solver).
- [`AI4Fun/music/`](AI4Fun/music/): a five-part **arrangement of 望春風**
  (*Bāng-chhun-hong*, 1933, 鄧雨賢).
- [`AI4Fun/3d_penguin/`](AI4Fun/3d_penguin/): a procedural **3-D QQ-penguin**
  built by a single [Blender](https://docs.blender.org/api/current/) `bpy` script.
- [`AI4Fun/3d_hunyuan/`](AI4Fun/3d_hunyuan/): a procedural **3-D Tencent Hunyuan
  (混元) logo orb**.

## Citation

If you use these results, please cite:

```bibtex
@misc{hyra2026,
  title        = {Hyra: Hunyuan Research Agent},
  author       = {{Hyra Team}},
  year         = {2026},
  howpublished = {\url{https://hy.tencent.com/research/hyra}},
}
```

## License

This repository is licensed under the Apache License, Version 2.0; see
[`LICENSE`](LICENSE).
