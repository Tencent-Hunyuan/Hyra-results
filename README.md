# Hyra Results

Companion artifacts for **Hyra**: **H**un**y**uan **R**esearch **A**gent.

This repository collects concrete solutions that Hyra produced across a range of
open problems in science, mathematics, engineering, and creative design,
released alongside the Hyra launch post. Each folder holds the final solution
artifact and, where relevant, the self-contained scripts that reproduce it.

> 📝 Launch post: **[hy.tencent.com/research/hyra](https://hy.tencent.com/research/hyra)**
>
> 🚀 Hyra 2.0 blog: **[hy.tencent.ai/research/hyra2](https://hy.tencent.ai/research/hyra2)**

## News

- **2026-11-04**: 🌊 **HyPDE, a 2D foundation model for PDEs, released with weights and code.**
  **HyPDE**, a 323M-parameter fused-history transformer trained from scratch on
  thirteen 2D physics datasets, forecasts simulated fields from their last six
  frames. Zero-shot on the test split of thirteen datasets of The Well and
  FlowBench, against Walrus (1.29B parameters, released weights, no
  fine-tuning on either side), it has the lower median VRMSE on **9 of 13**
  datasets in one-step forecasting, **10 of 13** over rollout frames 1-20 and
  **8 of 13** over frames 21-60. Weights, the `hypde` package and the full
  table: **[`AI4Science/hypde/`](AI4Science/hypde/)**.

- **2026-11-03**: 🔐 **Certified distances of non-CSS quantum codes.** The
  qldpc-challenge board labels all 184 of its general (non-CSS) stabilizer codes
  as upper bounds only. **hyqec**, a prover plus an independently written
  certificate checker, proves the exact Pauli-weight distance of **165** of
  them: 122 with SAT proofs checked by `drat-trim`, 42 with re-executable
  branch-and-bound records, and the board's `[[452,115,18]]` (**d = 18**) with
  a coverage certificate of 124.6 billion search nodes that the independent
  checker re-executed in full. Code:
  **[`AI4Science/hyqec/`](AI4Science/hyqec/)**.

- **2026-11-02**: 🚀 **Hyra 2.0 blog released.**
  Read **[Hyra 2.0: The Next Step for Auto Research](https://hy.tencent.ai/research/hyra2)**
  for an overview of Hyra's progress in developing reusable scientific tools,
  models, and workflows.

- **2026-11-02**: 🧫 **HyCell, a single-cell embedding model, released with weights and code.**
  **HyCell**, a 16.1M-parameter model, maps the raw counts of a human cell to
  a 512-dimensional embedding and decodes embeddings back to gene expression.
  On seven atlases held out from pretraining, its frozen embeddings reach an
  overall score of **0.735**, level with Stack (**0.736**, 218M parameters).
  Weights, the `hycell` package and the technical report:
  **[`AI4Science/hycell/`](AI4Science/hycell/)**.
  
- **2026-11-02**: ⚛️ **HyFF force fields released with weights and code.** Five
  machine-learning force fields for molecular simulation, from 2.3M to 1.2B
  parameters, covering 83 elements with total charge and spin and trained on
  OMol25. On the full OMol25 validation set, **HyFF Andante** reaches a force
  error comparable to UMA-S-1.2 (7.65 vs 7.55 meV/Å) at **29×** its MD speed,
  and **HyFF Largo** has the lowest energy error of the evaluated models
  (0.364 meV/atom; DPA4-Pro: 0.380) at **9.7×** DPA4-Pro's speed (MD of a
  2,586-atom system on one H20). The `hyff` package runs NVE, NVT and NPT
  molecular dynamics and batched prediction:
  **[`AI4Science/hyff/`](AI4Science/hyff/)**.

- **2026-11-02**: 🧬 **Two protein models released with weights and code.**
  **HyInv**, a 1.96M-parameter inverse-folding model, and **HyPLM-Flash**, a
  650M-parameter protein language model with a fast INT8 inference engine, are
  published under `AI4Science/` together with their weights, inference code and
  technical reports: **[`AI4Science/hyinv/`](AI4Science/hyinv/)**,
  **[`AI4Science/hyplm_flash/`](AI4Science/hyplm_flash/)**.

- **2026-11-02**: 📐 **Two results on Petty's and Yau's conjectures.**
  - **Petty's equilateral-set conjecture in dimension five.** The highest
    settled dimension rises from **4** (Makeev, 2005) to **5**: every
    five-dimensional real normed space contains **six equidistant points**.
    Paper: **[`AI4Science/petty_dimension_five/`](AI4Science/petty_dimension_five/)**.
  - **Yau's circle-action conjecture (unrestricted form).** A closed
    **seven-manifold** with strictly positive sectional curvature admits
    **no effective smooth circle action**, giving a counterexample. The
    manifold is not simply connected; the simply connected version remains
    open. Paper:
    **[`AI4Science/yau_circle_action_conjecture/`](AI4Science/yau_circle_action_conjecture/)**.

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

Each row compares the best prior published result or conjecture status
(**Prev best**, from the cited sources) against **Hyra**. Arrows mark the better
direction (↓ lower is better, ↑ higher is better); the Hyra result is in bold.

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
| | [`petty_dimension_five`](AI4Science/petty_dimension_five/) | dimension settled ↑ | 4 <sup>l</sup> | **5** |
| | [`yau_circle_action_conjecture`](AI4Science/yau_circle_action_conjecture/) | conjecture status | open <sup>m</sup> | **counterexample (dim 7)** |
| | [`hyqec`](AI4Science/hyqec/) | board non-CSS codes with exact d ↑ | 0 / 184 <sup>n</sup> | **165 / 184** (122 with `drat-trim`-checked SAT proofs, 1 independently re-checked certificate) |

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
- <sup>l</sup> V. V. Makeev: *Equilateral simplices in normed 4-space*,
Zap. Nauchn. Sem. POMI 329 (2005), 88–91; English translation in
J. Math. Sci. 140(4) (2007), 548–550, establishing the four-dimensional case
of Petty's equilateral-set conjecture.
- <sup>m</sup> S.-T. Yau: *Open problems in geometry*, Proc. Sympos. Pure
Math. 54, Part 1 (1993), 1–28; see also *Structure of manifolds with positive
curvature based on geometric analysis*, Notices of the ICCM 1(2) (2013),
24–27, for the smooth circle-action conjecture.
- <sup>n</sup> The qldpc-challenge board
([github.com/unitaryfoundation/qldpc-challenge](https://github.com/unitaryfoundation/qldpc-challenge))
labels every one of its general stabilizer (non-CSS) entries as an upper bound
(`d≤`) only, as its certifier does not minimise Pauli weight (`TRACKS.md`,
2026-10-27). Codes are from the 2026-10-16 snapshot. Exact distances of other
non-CSS codes have been computed elsewhere (e.g. Lee et al.,
[arXiv:2609.30069](https://arxiv.org/abs/2609.30069)); the count is about this
board. Hyra's 165 are made up as follows. 122 have SAT proofs checked by
`drat-trim`; the CNF encoding is hyqec's own. 1 (`[[452,115,18]]`) has a
branch-and-bound coverage certificate re-executed by the independent checker.
42 have branch-and-bound search records that only the same prover can re-run.

Metrics are as defined by each benchmark; comparisons are against the cited
published results and conjecture statements.

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
- **dimension settled**: highest dimension `n` for which every real
  `n`-dimensional normed space is known to contain `n + 1` equidistant points,
  as conjectured by Petty (maximize).
- **conjecture status**: status of the unrestricted claim that every closed
  manifold admitting strictly positive sectional curvature admits an effective
  smooth circle action. The counterexample has `π₁ ≅ (ℤ/6)²`; the simply
  connected version remains open.
- **board non-CSS codes with exact d**: number of the board's 184 general
  stabilizer codes whose Pauli-weight distance is proven exactly, not only
  bounded (maximize).

> **Note.** The results above were current as of **2026-07-10**, except
> `3d_blaschke_lebesgue`, added **2026-08-17**, the three entries added
> **2026-08-19**, `petty_dimension_five` and `yau_circle_action_conjecture`,
> added **2026-11-02**, and `hyqec`, added **2026-11-03**. Several of these
> problems live on public, continuously-updated leaderboards; later entries there
> may *warm-start from Hyra's published solutions* to reach still-better numbers.

### Model releases

Trained models, each with its weights, inference code and technical report:

- [`AI4Science/hyinv/`](AI4Science/hyinv/): **HyInv**, a 1.96-million-parameter
  message-passing graph network for protein inverse folding (backbone →
  sequence), trained with structure-reward reinforcement learning. The 7.8 MB
  weights ship in the folder; see its README for the CLI and Python API.
- [`AI4Science/hyplm_flash/`](AI4Science/hyplm_flash/): **HyPLM-Flash**, a
  650-million-parameter protein language model with an INT8 GPU inference engine
  (about 130,000 residues per second on one H20). The 2.6 GB weights are a
  GitHub Release on this repository; `weights/download.sh` fetches, merges and
  verifies them.
- [`AI4Science/hycell/`](AI4Science/hycell/): **HyCell**, a 16.1-million-parameter
  embedding model for single-cell transcriptomes, pretrained without cell-type
  labels on 79.5 million human cells of the CZ CELLxGENE Census. It maps raw
  counts to 512-dimensional embeddings and decodes them back to expression, and
  can be fine-tuned into a cell-type annotator. The 64 MB weights ship in the
  folder as part of the `hycell` package.
- [`AI4Science/hyff/`](AI4Science/hyff/): **HyFF**, five machine-learning force
  fields for molecular simulation (Presto, Vivace, Moderato, Andante and Largo;
  2.3 million to 1.2 billion parameters, 83 elements), with the `hyff` package
  for single-structure inference, NVE/NVT/NPT molecular dynamics and batched
  prediction. The weights (9 MB to 4.8 GB per model) are a GitHub Release on
  this repository; the package downloads and verifies them on first use.

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
