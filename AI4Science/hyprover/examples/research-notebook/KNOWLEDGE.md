<!-- hyprover:document {"display_version": 1, "format": "hyprover-knowledge-v1", "problem_id": "arxiv-2606-19865", "revision": 79} -->
# Research knowledge

[Problem](PROBLEM.md) · [Overview](#overview) · [Facts](#fact) · [Routes](#route) · [Conjectures](#conjecture) · [Experience](#experience) · [History](#history) · [Public reports](#report)

Only active verified facts are usable premises. Proofs and long notes are linked, not embedded.
Shared updates go through the knowledge service; private drafts and runtime logs stay separate.

**阅读编号 / Reading labels:** F = 事实 Facts · O = 概览 Overview · R = 路线 Routes · C = 猜想 Conjectures · E = 经验 Experience · L = 线索 Leads · V = 验证反馈 Feedback.
Numbers are stable within this project; titles are navigation aids, not mathematical statements. Use the internal IDs for tool arguments. Full statements below remain authoritative.
P = 公开报告 Public reports — project-shared coordination, never verified premises.

<a id="overview"></a>
## Project overview — strategy, not fact

**本区目录 / Contents**
- [O001 · The full fixed theorem is published; its complete proof closure awaits fresh revalidation](#entry-61d4f0a0a2f9515c)

<!-- hyprover:entry {"attributes": {}, "author": "main_agent", "created_at": "", "details_path": "", "display_id": "O001", "entry_id": "entry-61d4f0a0a2f9515c", "kind": "overview", "links": ["846e296f22ef7853", "3ce5f83832ddd38c", "a8378dfc6c55d238", "8f8adadaf10ea26e", "7a3fd2eb4772e4b4", "83fc8a6190cb2ca3", "a0c54399c2d44b8d", "e9b3873b7c826c7b", "dc5ba3bf489a3dea", "c7fafba4b4d5a72f", "81a5afe69f10c3d2", "947986eae4461bec", "e738ef25f0351e92", "entry-10acec896a9c45b2", "entry-f5a5afb3f0a8d294", "entry-97b5b6c76e914106", "entry-a241367a37a8ed00"], "revision": 3, "status": "current", "title": "The full fixed theorem is published; its complete proof closure awaits fresh revalidation", "updated_at": ""} -->
<a id="entry-61d4f0a0a2f9515c"></a>
### O001 · The full fixed theorem is published; its complete proof closure awaits fresh revalidation [overview: current; NOT A FACT]

<!-- hyprover:body -->
Fixed goal verbatim: For every integer n>=3, there exists a smooth strictly convex body K subset R^n for which exactly one hyperplane section has centroid c(K). Consequently, mu(n)=3 for n=2, and mu(n)=1 for n>=3. Universal assertions include nonsmooth convex bodies; sections have positive Euclidean Hausdorff measure and are counted by geometric affine hyperplanes without orientation.

Current assessment: active published F013 (846e296f22ef7853) states and proves the full fixed theorem. Its direct predecessors are F003 (3ce5f83832ddd38c), F008 (a8378dfc6c55d238) and F009 (7a3fd2eb4772e4b4). F003 gives at least three matching lines for every planar convex body and exactly three for any nondegenerate triangle, hence mu(2)=3. F008 proves centroid interiority and a positive-measure matching section through every prescribed interior point of every convex body in all n>=2, giving mu(n)>=1 without boundary regularity. F009 supplies a C-infinity strictly convex body with exactly one matching affine hyperplane for every n>=3, giving mu(n)<=1 as well as the required smooth existence assertion. These close every original clause; there is no unmatched mathematical hypothesis in the published dependency chain. The remaining gate is fresh controller revalidation of every full proof and the exact original-problem implication, not ordinary statement-level trust or this overview.

The upper-bound interfaces are exact: F001 supplies smooth real odd F_n with zero first spherical moment and precisely two nondegenerate critical points. F007's C2 approximation theorem applies and produces a finite real odd harmonic sum P supported in degrees at least 3 with exactly one nondegenerate antipodal critical pair. F002 inverts H on that finite P, in the same Euclidean normalization, yielding smooth real odd g with Hg=P and integral theta*g=0. F005 applies to exactly those hypotheses: rho_epsilon^(n+1)=1+epsilon*g gives exact volume centering, smooth strict convexity and exactly one matching affine hyperplane for all sufficiently small nonzero epsilon. Its homogeneous radial gauge proves global convexity through the origin; ((n+1)/epsilon)B_epsilon -> Hg in C2 and definite local Hessians plus a positive gradient bound on the compact complement preserve the complete critical set. Every matching affine section is central by averaging its defining equation, and antipodal normals specify one hyperplane. F009 publishes this composition. The finite g is fixed before epsilon, so no dimension-uniform, harmonic-degree-uniform or arbitrary-smooth inversion estimate is needed. C001 is additionally source-resolved by F010, but F013 does not depend on that extra composition.

The nonsmooth lower-bound interface is F006: the sign transform of a continuous function is C1, its spherical gradient is twice the equatorial first moment, and polar integration gives grad B=2M for every positive continuous radial body. A convex body's gauge supplies that radial function about any interior point without boundary smoothness. F008 applies an extremum of B and checks centroid interiority, positive section measure, translation and exact affine/unoriented counting. Thus the smooth sharp examples have not narrowed the universal quantifier.

Route portfolio: R001 and R002 supply the selected full theorem. R003 has now also reached published F011, a polynomial disk-fiber shear realization for every n>=3 with sole predecessor F001. It is an alternative to harmonic inversion and the radial prescription, while sharing the same smooth input; it is retained as a fallback outside the selected F013 closure. R004 remains parked with published F004's conditional suspension theorem and its explicit fixed-critical-value homotopy requirement. F012 now proves the extra-section obstruction for cylinders and symmetric homothetic suspensions, including affine images. Its initial auxiliary integrability gap was repaired before successful publication and affects no selected dependency.

Anti-shortcuts remain excluded: F007(3) shows mere C1 closeness cannot replace C2 critical-point stability; finite harmonic inversion is not an arbitrary-smooth inversion theorem; generic two-critical-point functions are not assumed to meet F004's homotopy input; products, recentering, or unproved smoothing cannot preserve the count by assertion. No failed proof refutes its statement. No source theorem assertion, external literature, computation, manual audit or unpublished candidate is a mathematical premise. Reopen and repair the exact affected interface if fresh completion review identifies a defect; retain the alternatives rather than weaken the target.
<!-- hyprover:end-body -->

**Related:** [F013 · Full centroid-section theorem: smooth sharp examples and mu(2)=3, mu(n)=1 for n&gt;=3](#fact-846e296f22ef7853), [F003 · Sharp planar centroid-section theorem: every body has at least three, triangles exactly three](#fact-3ce5f83832ddd38c), [F008 · Every interior point of a convex body is the centroid of some hyperplane section](#fact-a8378dfc6c55d238), [F006 · Signed halfspace volume is C1 and its spherical gradient is twice the central-section first moment](#fact-8f8adadaf10ea26e), [F009 · Smooth strictly convex bodies with exactly one centroid-matching hyperplane in every dimension n&gt;=3](#fact-7a3fd2eb4772e4b4), [F001 · Smooth odd twisted heights with zero first moment and exactly two nondegenerate critical points](#fact-83fc8a6190cb2ca3), [F007 · Finite odd harmonic approximation with exact moment cancellation and two-critical-point stability](#fact-a0c54399c2d44b8d), [F002 · Nonzero odd eigenvalues and finite harmonic inversion of the hemispherical sign transform](#fact-e9b3873b7c826c7b), [F005 · Centered radial perturbations realize one centroidal hyperplane from a two-critical-point sign transform](#fact-dc5ba3bf489a3dea), [F010 · Finite odd harmonic preimages whose sign transforms have exactly one nondegenerate critical pair](#fact-c7fafba4b4d5a72f), [F011 · Polynomial shears of the ball give unique centroid-matching sections in every dimension n&gt;=3](#fact-81a5afe69f10c3d2), [F004 · Dimension lifting of zero-moment odd potentials by suspension and local moment correction](#fact-947986eae4461bec), [F012 · Cylinders have open families of centroid-matching sections; symmetric homothetic suspensions retain extra sections](#fact-e738ef25f0351e92), [R001 · Universal and planar components are published and ready for completion revalidation](#entry-10acec896a9c45b2), [R002 · Use the published full-target fact as the completion root](#entry-f5a5afb3f0a8d294), [R003 · The direct shear alternative is published and retained as a fallback](#entry-97b5b6c76e914106), [R004 · Retain conditional lifting and the published boundary for naive products](#entry-a241367a37a8ed00)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-61d4f0a0a2f9515c`

</details>
<!-- hyprover:end -->

<a id="fact"></a>
## Verified facts — statements only

**本区目录 / Contents**
- [F001 · Smooth odd twisted heights with zero first moment and exactly two nondegenerate critical points](#fact-83fc8a6190cb2ca3)
- [F002 · Nonzero odd eigenvalues and finite harmonic inversion of the hemispherical sign transform](#fact-e9b3873b7c826c7b)
- [F003 · Sharp planar centroid-section theorem: every body has at least three, triangles exactly three](#fact-3ce5f83832ddd38c)
- [F004 · Dimension lifting of zero-moment odd potentials by suspension and local moment correction](#fact-947986eae4461bec)
- [F005 · Centered radial perturbations realize one centroidal hyperplane from a two-critical-point sign transform](#fact-dc5ba3bf489a3dea)
- [F006 · Signed halfspace volume is C1 and its spherical gradient is twice the central-section first moment](#fact-8f8adadaf10ea26e)
- [F007 · Finite odd harmonic approximation with exact moment cancellation and two-critical-point stability](#fact-a0c54399c2d44b8d)
- [F008 · Every interior point of a convex body is the centroid of some hyperplane section](#fact-a8378dfc6c55d238)
- [F009 · Smooth strictly convex bodies with exactly one centroid-matching hyperplane in every dimension n&gt;=3](#fact-7a3fd2eb4772e4b4)
- [F010 · Finite odd harmonic preimages whose sign transforms have exactly one nondegenerate critical pair](#fact-c7fafba4b4d5a72f)
- [F011 · Polynomial shears of the ball give unique centroid-matching sections in every dimension n&gt;=3](#fact-81a5afe69f10c3d2)
- [F012 · Cylinders have open families of centroid-matching sections; symmetric homothetic suspensions retain extra sections](#fact-e738ef25f0351e92)
- [F013 · Full centroid-section theorem: smooth sharp examples and mu(2)=3, mu(n)=1 for n&gt;=3](#fact-846e296f22ef7853)

<!-- hyprover:fact {"author": "max4", "created_at": "", "display_id": "F001", "external_refs": [], "fact_id": "83fc8a6190cb2ca3", "glossary_introduces": {"F_n": "The twisted height x_1 cos(alpha_n(x_3)) + x_2 sin(alpha_n(x_3)) on S^{n-1}.", "alpha_n": "A smooth even real angle function on R, constant 0 near 0 and constant 2*pi for |t| sufficiently close to or larger than 1, constructed for dimension n."}, "intuition_path": "", "predecessors": [], "problem_id": "arxiv-2606-19865", "proof_path": "knowledge/proofs/83fc8a6190cb2ca3.md", "revision": 1, "revocation": {}, "status": "verified", "title": "Smooth odd twisted heights with zero first moment and exactly two nondegenerate critical points", "verification": {"formal_certificate": false, "provenance": "Historical source-run verdict; sanitized static snapshot, not rerun on export.", "verdict": "correct", "verifier": "independent-llm"}} -->
<a id="fact-83fc8a6190cb2ca3"></a>
### F001 · Smooth odd twisted heights with zero first moment and exactly two nondegenerate critical points [verified]

<!-- hyprover:body -->
For every integer n >= 3, let S^{n-1} = {x in R^n : |x| = 1}, let sigma be its Euclidean surface measure, let e_1,...,e_n be the coordinate unit vectors, and let g be the round Riemannian metric. There exist an even function alpha_n in C-infinity(R) and numbers 0 < a_n < b_n < 1 such that alpha_n(t) = 0 for |t| <= a_n and alpha_n(t) = 2*pi for |t| >= b_n, with the following properties. The real function
F_n(x) = x_1 cos(alpha_n(x_3)) + x_2 sin(alpha_n(x_3)), x in S^{n-1},
is smooth and odd, satisfies the vector identity integral_{S^{n-1}} x F_n(x) d sigma(x) = 0, and has exactly two critical points, namely e_1 and -e_1. Both critical points are nondegenerate; more precisely, the spherical Hessians are Hess F_n(e_1) = -g_{e_1} and Hess F_n(-e_1) = g_{-e_1}.
<!-- hyprover:end-body -->

**Depends on:** none
**Proof:** [完整证明 / Full proof — F001](knowledge/proofs/83fc8a6190cb2ca3.md)
**Definitions:**
- `F_n`: The twisted height x_1 cos(alpha_n(x_3)) + x_2 sin(alpha_n(x_3)) on S^{n-1}.
- `alpha_n`: A smooth even real angle function on R, constant 0 near 0 and constant 2*pi for |t| sufficiently close to or larger than 1, constructed for dimension n.

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `83fc8a6190cb2ca3`
**Verification:** {"formal_certificate": false, "provenance": "Historical source-run verdict; sanitized static snapshot, not rerun on export.", "verdict": "correct", "verifier": "independent-llm"}

</details>
<!-- hyprover:end -->

<!-- hyprover:fact {"author": "max3", "created_at": "", "display_id": "F002", "external_refs": [], "fact_id": "e9b3873b7c826c7b", "glossary_introduces": {"H": "Hemispherical sign transform (Hf)(u)=integral_{S^{n-1}} sign(u dot theta)f(theta)d sigma(theta), with sign(0)=0.", "H_k": "Real spherical harmonics of degree k: restrictions to S^{n-1} of real homogeneous degree-k polynomials annihilated by the Euclidean Laplacian.", "lambda_{n,2j+1}": "The scalar 2 s_{n-2} (-1)^j (2j)! / [2^j j! product_{r=0}^{j}(n-1+2r)], for n\u003e=3 and j\u003e=0.", "s_{n-2}": "Ordinary Euclidean surface measure H^{n-2}(S^{n-2})."}, "intuition_path": "", "predecessors": [], "problem_id": "arxiv-2606-19865", "proof_path": "knowledge/proofs/e9b3873b7c826c7b.md", "revision": 1, "revocation": {}, "status": "verified", "title": "Nonzero odd eigenvalues and finite harmonic inversion of the hemispherical sign transform", "verification": {"formal_certificate": false, "provenance": "Historical source-run verdict; sanitized static snapshot, not rerun on export.", "verdict": "correct", "verifier": "independent-llm"}} -->
<a id="fact-e9b3873b7c826c7b"></a>
### F002 · Nonzero odd eigenvalues and finite harmonic inversion of the hemispherical sign transform [verified]

<!-- hyprover:body -->
Let n>=3 be an integer. Write S^{n-1}={x in R^n:|x|=1}, let sigma be ordinary Euclidean (n-1)-dimensional surface measure, and set s_{n-2}=H^{n-2}(S^{n-2}). For k>=0 let H_k be the real vector space of restrictions to S^{n-1} of real homogeneous polynomials P of degree k satisfying Delta P=0, where Delta is the Euclidean Laplacian. Define the hemispherical sign transform by
(Hf)(u)=integral_{S^{n-1}} sign(u dot theta) f(theta) d sigma(theta), u in S^{n-1},
with sign(0)=0.

For every integer j>=0 and every Y in H_{2j+1},
HY=lambda_{n,2j+1}Y,
where
lambda_{n,2j+1}=2 s_{n-2} (-1)^j (2j)! / [2^j j! product_{r=0}^{j}(n-1+2r)].
In particular every lambda_{n,2j+1} is a nonzero real number. For every even k>=0 and every Y in H_k, HY=0.

The spaces H_k of distinct degrees are orthogonal for the real L^2(sigma) inner product, so the components of a finite harmonic sum are uniquely defined. If a real finite harmonic sum F is odd and its degree-one component is zero, then there exists a unique real odd finite harmonic sum g with Hg=F. This g has zero degree-one component, is C-infinity on S^{n-1}, and satisfies
integral_{S^{n-1}} theta g(theta) d sigma(theta)=0 in R^n.
Explicitly, writing F=sum_{j=1}^{J} F_{2j+1} with F_{2j+1} in H_{2j+1} (zero terms allowed), one has g=sum_{j=1}^{J} lambda_{n,2j+1}^{-1} F_{2j+1}. The zero function is covered by the empty sum.
<!-- hyprover:end-body -->

**Depends on:** none
**Proof:** [完整证明 / Full proof — F002](knowledge/proofs/e9b3873b7c826c7b.md)
**Definitions:**
- `H`: Hemispherical sign transform (Hf)(u)=integral_{S^{n-1}} sign(u dot theta)f(theta)d sigma(theta), with sign(0)=0.
- `H_k`: Real spherical harmonics of degree k: restrictions to S^{n-1} of real homogeneous degree-k polynomials annihilated by the Euclidean Laplacian.
- `lambda_{n,2j+1}`: The scalar 2 s_{n-2} (-1)^j (2j)! / [2^j j! product_{r=0}^{j}(n-1+2r)], for n>=3 and j>=0.
- `s_{n-2}`: Ordinary Euclidean surface measure H^{n-2}(S^{n-2}).

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `e9b3873b7c826c7b`
**Verification:** {"formal_certificate": false, "provenance": "Historical source-run verdict; sanitized static snapshot, not rerun on export.", "verdict": "correct", "verifier": "independent-llm"}

</details>
<!-- hyprover:end -->

<!-- hyprover:fact {"author": "max2", "created_at": "", "display_id": "F003", "external_refs": [], "fact_id": "3ce5f83832ddd38c", "glossary_introduces": {}, "intuition_path": "", "predecessors": [], "problem_id": "arxiv-2606-19865", "proof_path": "knowledge/proofs/3ce5f83832ddd38c.md", "revision": 1, "revocation": {}, "status": "verified", "title": "Sharp planar centroid-section theorem: every body has at least three, triangles exactly three", "verification": {"formal_certificate": false, "provenance": "Historical source-run verdict; sanitized static snapshot, not rerun on export.", "verdict": "correct", "verifier": "independent-llm"}} -->
<a id="fact-3ce5f83832ddd38c"></a>
### F003 · Sharp planar centroid-section theorem: every body has at least three, triangles exactly three [verified]

<!-- hyprover:body -->
Let K be any compact convex subset of R^2 with nonempty interior, let |K| denote its Euclidean area, and let c(K)=|K|^{-1} integral_K x dA(x). For a line L whose section K intersect L has positive Euclidean length, let c(K intersect L) be its centroid with respect to length. Then at least three distinct geometric affine lines L satisfy c(K intersect L)=c(K). For every nondegenerate triangle T there are exactly three such lines: the lines through c(T) parallel to its three sides. Lines are counted without orientation. Consequently, if mu(2) is the largest integer q such that every planar convex body has at least q centroid-matching lines, then mu(2)=3.
<!-- hyprover:end-body -->

**Depends on:** none
**Proof:** [完整证明 / Full proof — F003](knowledge/proofs/3ce5f83832ddd38c.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `3ce5f83832ddd38c`
**Verification:** {"formal_certificate": false, "provenance": "Historical source-run verdict; sanitized static snapshot, not rerun on export.", "verdict": "correct", "verifier": "independent-llm"}

</details>
<!-- hyprover:end -->

<!-- hyprover:fact {"author": "max6", "created_at": "", "display_id": "F004", "external_refs": [], "fact_id": "947986eae4461bec", "glossary_introduces": {"A_delta": "The smooth suspension r U_{beta_delta(r)}(omega) on S^d, made exactly x_1 near both poles by beta_delta=1 there.", "A_ref": "The continuous homogeneous suspension rF(omega) on S^d, assigned value zero at the poles.", "K_band": "The fixed compact middle-latitude band { (x,t) in S^d : 1/3\u003c=|t|\u003c=2/3 } used to support the small moment correction.", "Q_a": "The smooth odd correction chi(t)(a dot x), where chi is an even nonzero nonnegative bump supported in 1/3\u003c|t|\u003c2/3.", "U_s": "In the dimension-lifting lemma, a jointly smooth odd homotopy on S^{d-1} from F to omega_1; every member has precisely two nondegenerate critical points with values 1 and -1."}, "intuition_path": "", "predecessors": [], "problem_id": "arxiv-2606-19865", "proof_path": "knowledge/proofs/947986eae4461bec.md", "revision": 1, "revocation": {}, "status": "verified", "title": "Dimension lifting of zero-moment odd potentials by suspension and local moment correction", "verification": {"formal_certificate": false, "provenance": "Historical source-run verdict; sanitized static snapshot, not rerun on export.", "verdict": "correct", "verifier": "independent-llm"}} -->
<a id="fact-947986eae4461bec"></a>
### F004 · Dimension lifting of zero-moment odd potentials by suspension and local moment correction [verified]

<!-- hyprover:body -->
Let d>=2 be an integer, S^{d-1}={omega in R^d:|omega|=1}, and sigma_{d-1} its Euclidean surface measure. Suppose F:S^{d-1}->R is C-infinity and there is a jointly C-infinity family U:[0,1] times S^{d-1}->R, written U_s(omega), such that U_0=F, U_1(omega)=omega_1, and for every s in [0,1]: U_s is odd, has exactly two critical points for the round metric, both are nondegenerate, and their values are +1 and -1. Suppose also that integral_{S^{d-1}} omega F(omega) d sigma_{d-1}(omega)=0. Denote the maximum point of F by p; its minimum point is -p.

Then, on S^d={(x,t) in R^d times R:|x|^2+t^2=1}, there exists a C-infinity function G with all of the following properties. G(-x,-t)=-G(x,t), G(x,-t)=G(x,t), G(x,0)=F(x) for x in S^{d-1}, and integral_{S^d}(x,t)G(x,t)d sigma_d(x,t)=0. Its only critical points are (p,0) and (-p,0), with respective values +1 and -1, and both are nondegenerate. Under the orthogonal identification T_{(q,0)}S^d=T_q S^{d-1} direct-sum R for q=p or -p, its Hessian is Hess F(q) direct-sum (-F(q)). In addition G admits a jointly C-infinity homotopy to the coordinate height (x,t)->x_1 through odd functions each having exactly two nondegenerate critical points with critical values +1 and -1. Consequently the conclusion can be iterated: for every integer D>=d there exists a smooth odd zero-first-moment function on S^{D-1} with exactly two nondegenerate critical points, admitting such a homotopy to the first coordinate, whose restriction to the coordinate subsphere S^{d-1} is F.
<!-- hyprover:end-body -->

**Depends on:** none
**Proof:** [完整证明 / Full proof — F004](knowledge/proofs/947986eae4461bec.md)
**Definitions:**
- `A_delta`: The smooth suspension r U_{beta_delta(r)}(omega) on S^d, made exactly x_1 near both poles by beta_delta=1 there.
- `A_ref`: The continuous homogeneous suspension rF(omega) on S^d, assigned value zero at the poles.
- `K_band`: The fixed compact middle-latitude band { (x,t) in S^d : 1/3<=|t|<=2/3 } used to support the small moment correction.
- `Q_a`: The smooth odd correction chi(t)(a dot x), where chi is an even nonzero nonnegative bump supported in 1/3<|t|<2/3.
- `U_s`: In the dimension-lifting lemma, a jointly smooth odd homotopy on S^{d-1} from F to omega_1; every member has precisely two nondegenerate critical points with values 1 and -1.

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `947986eae4461bec`
**Verification:** {"formal_certificate": false, "provenance": "Historical source-run verdict; sanitized static snapshot, not rerun on export.", "verdict": "correct", "verifier": "independent-llm"}

</details>
<!-- hyprover:end -->

<!-- hyprover:fact {"author": "max7", "created_at": "", "display_id": "F005", "external_refs": [], "fact_id": "dc5ba3bf489a3dea", "glossary_introduces": {}, "intuition_path": "", "predecessors": [], "problem_id": "arxiv-2606-19865", "proof_path": "knowledge/proofs/dc5ba3bf489a3dea.md", "revision": 1, "revocation": {}, "status": "verified", "title": "Centered radial perturbations realize one centroidal hyperplane from a two-critical-point sign transform", "verification": {"formal_certificate": false, "provenance": "Historical source-run verdict; sanitized static snapshot, not rerun on export.", "verdict": "correct", "verifier": "independent-llm"}} -->
<a id="fact-dc5ba3bf489a3dea"></a>
### F005 · Centered radial perturbations realize one centroidal hyperplane from a two-critical-point sign transform [verified]

<!-- hyprover:body -->
Let n>=3 be an integer, let S^{n-1} be the Euclidean unit sphere with its usual surface measure sigma, and define the hemispherical sign transform by
(Hf)(u)=integral_{S^{n-1}} sign(u dot theta) f(theta) d sigma(theta),
where sign(0)=0. Suppose g:S^{n-1}->R is C-infinity, g(-theta)=-g(theta),
integral_{S^{n-1}} theta g(theta) d sigma(theta)=0,
and h=Hg has exactly two critical points on S^{n-1}, both nondegenerate for the round metric.

Then there exists epsilon_0>0 such that for every real epsilon with 0<|epsilon|<epsilon_0, the function
rho_epsilon(theta)=(1+epsilon g(theta))^{1/(n+1)}
is positive and the radial set
K_epsilon={r theta: theta in S^{n-1}, 0<=r<=rho_epsilon(theta)}
is a compact convex body with C-infinity boundary, positive principal curvatures and strict convexity in the sense that every open chord between two distinct points of K_epsilon lies in its interior. Its uniform n-dimensional volume centroid is exactly 0. Exactly one geometric affine hyperplane A with positive (n-1)-dimensional section measure satisfies c(K_epsilon intersect A)=0; hyperplanes are counted without orientation.

More explicitly, writing
B_epsilon(u)=vol_n(K_epsilon intersect {x:u dot x>0})-vol_n(K_epsilon intersect {x:u dot x<0}),
one has the smooth odd function and first-moment identities
B_epsilon=(1/n)H[(1+epsilon g)^{n/(n+1)}],
grad_{S^{n-1}} B_epsilon(u)=2 integral_{K_epsilon intersect u^perp} x d H^{n-1}(x).
Furthermore ((n+1)/epsilon)B_epsilon converges to Hg in C^2(S^{n-1}) as epsilon->0 through nonzero values, and for every such sufficiently small epsilon the critical set of B_epsilon consists of exactly two nondegenerate antipodal points.
<!-- hyprover:end-body -->

**Depends on:** none
**Proof:** [完整证明 / Full proof — F005](knowledge/proofs/dc5ba3bf489a3dea.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `dc5ba3bf489a3dea`
**Verification:** {"formal_certificate": false, "provenance": "Historical source-run verdict; sanitized static snapshot, not rerun on export.", "verdict": "correct", "verifier": "independent-llm"}

</details>
<!-- hyprover:end -->

<!-- hyprover:fact {"author": "max", "created_at": "", "display_id": "F006", "external_refs": [], "fact_id": "8f8adadaf10ea26e", "glossary_introduces": {}, "intuition_path": "knowledge/notes/8f8adadaf10ea26e-intuition.md", "predecessors": [], "problem_id": "arxiv-2606-19865", "proof_path": "knowledge/proofs/8f8adadaf10ea26e.md", "revision": 1, "revocation": {}, "status": "verified", "title": "Signed halfspace volume is C1 and its spherical gradient is twice the central-section first moment", "verification": {"formal_certificate": false, "provenance": "Historical source-run verdict; sanitized static snapshot, not rerun on export.", "verdict": "correct", "verifier": "independent-llm"}} -->
<a id="fact-8f8adadaf10ea26e"></a>
### F006 · Signed halfspace volume is C1 and its spherical gradient is twice the central-section first moment [verified]

<!-- hyprover:body -->
For every integer n>=2, let S^{n-1}={x in R^n: |x|=1}, let sigma be ordinary Euclidean surface measure on S^{n-1}, set s_{n-2}=H^{n-2}(S^{n-2}) with ordinary Euclidean normalization (counting measure for n=2), and set E_u=S^{n-1} intersect u^\perp for u in S^{n-1}. On E_u use ordinary (n-2)-dimensional surface measure sigma_u; when n=2 this is counting measure on its two points. For a real continuous function f on S^{n-1}, define
(H f)(u)=integral_{S^{n-1}} sign(u dot theta) f(theta) d sigma(theta),
where sign(0)=0.
Then H f is C^1, is odd, and its gradient for the round metric is
grad_S(H f)(u)=2 integral_{E_u} theta f(theta) d sigma_u(theta).
This gradient, viewed as a vector in R^n, is continuous, tangent to S^{n-1}, and even under u -> -u. Moreover
sup |H f| <= sigma(S^{n-1}) sup |f|,
sup |grad_S H f| <= 2 s_{n-2} sup |f|.
For every nonnegative integer k, H maps C^k(S^{n-1}) continuously and linearly into C^{k+1}(S^{n-1}); equivalently there is a constant C_{n,k} such that ||H f||_{C^{k+1}} <= C_{n,k} ||f||_{C^k}, where these are the usual norms for the round metric.

For a continuous function rho:S^{n-1}->(0,infinity), define the compact radial body K_rho={r theta: theta in S^{n-1}, 0<=r<=rho(theta)}. Define
B_rho(u)=vol_n(K_rho intersect {x:u dot x>=0})-vol_n(K_rho intersect {x:u dot x<=0})
and
M_rho(u)=integral_{K_rho intersect u^\perp} x d H^{n-1}(x).
Here vol_n and H^{n-1} have their usual Euclidean normalizations. Every central section has positive finite (n-1)-measure, and
B_rho(u)=(1/n) H(rho^n)(u),
H^{n-1}(K_rho intersect u^\perp)=(1/(n-1)) integral_{E_u} rho(theta)^{n-1} d sigma_u(theta),
M_rho(u)=(1/n) integral_{E_u} theta rho(theta)^n d sigma_u(theta),
grad_S B_rho(u)=2 M_rho(u).
In particular B_rho is C^1 and its critical points are exactly the normals u for which c(K_rho intersect u^\perp)=0. If rho is C^k then B_rho is C^{k+1}.
These conclusions apply, without any boundary smoothness assumption, to K-p for every convex body K in R^n and every p in int(K): its radial function about the origin is continuous and positive.
<!-- hyprover:end-body -->

**Depends on:** none
**Proof:** [完整证明 / Full proof — F006](knowledge/proofs/8f8adadaf10ea26e.md)
**Intuition:** [直觉说明 / Intuition — F006](knowledge/notes/8f8adadaf10ea26e-intuition.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `8f8adadaf10ea26e`
**Verification:** {"formal_certificate": false, "provenance": "Historical source-run verdict; sanitized static snapshot, not rerun on export.", "verdict": "correct", "verifier": "independent-llm"}

</details>
<!-- hyprover:end -->

<!-- hyprover:fact {"author": "max8", "created_at": "", "display_id": "F007", "external_refs": [], "fact_id": "a0c54399c2d44b8d", "glossary_introduces": {"C2_norm": "For a real C^2 function on the unit sphere, the sum of the sup norm of the function, the sup norm of its round-metric gradient, and the sup operator norm of its round-metric Hessian.", "Pi_1": "The projection onto degree-one spherical harmonics: Pi_1 u(x)=(n/sigma(S^{n-1}))(integral_{S^{n-1}} y u(y) d sigma(y)) dot x.", "T_m": "The normalized polynomial-kernel operator T_m h(x)=Z_m^{-1} integral_{S^{n-1}} ((1+x dot y)/2)^m h(y) d sigma(y), with Z_m the same integral without h."}, "intuition_path": "knowledge/notes/a0c54399c2d44b8d-intuition.md", "predecessors": [], "problem_id": "arxiv-2606-19865", "proof_path": "knowledge/proofs/a0c54399c2d44b8d.md", "revision": 1, "revocation": {}, "status": "verified", "title": "Finite odd harmonic approximation with exact moment cancellation and two-critical-point stability", "verification": {"formal_certificate": false, "provenance": "Historical source-run verdict; sanitized static snapshot, not rerun on export.", "verdict": "correct", "verifier": "independent-llm"}} -->
<a id="fact-a0c54399c2d44b8d"></a>
### F007 · Finite odd harmonic approximation with exact moment cancellation and two-critical-point stability [verified]

<!-- hyprover:body -->
Let n >= 3, let S = S^{n-1} be the unit sphere in R^n with its round metric and Euclidean surface measure sigma, and put A = sigma(S). For a real C^2 function u on S, set
||u||_{C^2} = sup_S |u| + sup_S |grad_S u| + sup_S ||Hess_S u||_{op}.
A critical point means grad_S u = 0, and it is nondegenerate when its spherical Hessian is nonsingular.

(1) If F is real, C^2, odd, and satisfies integral_S x F(x) d sigma(x) = 0, there is a sequence P_m converging to F in this C^2 norm, where each P_m is a finite real sum of spherical harmonics of odd degrees at least 3. Here a spherical harmonic of degree k is the restriction to S of a homogeneous polynomial H_k on R^n of degree k with Euclidean Laplacian Delta H_k = 0. In particular, each P_m is smooth, odd, and has zero first moment exactly.

The projection onto the linear spherical harmonics is
(Pi_1 u)(x) = (n/A) (integral_S y u(y) d sigma(y)) dot x.
For every continuous real u, ||Pi_1 u||_{C^2} <= 3n ||u||_infinity. Thus, if Q is any polynomial restriction that is odd and F has zero first moment,
||Q - Pi_1 Q - F||_{C^2} <= ||Q-F||_{C^2} + 3n ||Q-F||_infinity,
and Q-Pi_1 Q is a finite sum of odd harmonics of degrees at least 3.

(2) If a real C^2 function f on S has exactly two critical points and both are nondegenerate, there exists delta_f > 0 such that every real C^2 function g with ||g-f||_{C^2} < delta_f has exactly two critical points, both nondegenerate. If g is odd, those points form one antipodal pair. Consequently, if F in (1) has exactly two nondegenerate critical points, then for every epsilon > 0 one can choose such a finite sum P with ||P-F||_{C^2} < epsilon and with exactly two nondegenerate critical points, forming one antipodal pair.

(3) C^1 closeness alone cannot replace the C^2 hypothesis in the stability assertion, even among smooth odd functions: for each n >= 3 there are smooth odd functions g_epsilon on S converging in C^1 to x -> x_n as epsilon decreases to zero, such that each sufficiently small g_epsilon has at least six critical points, whereas x -> x_n has exactly two nondegenerate critical points.
<!-- hyprover:end-body -->

**Depends on:** none
**Proof:** [完整证明 / Full proof — F007](knowledge/proofs/a0c54399c2d44b8d.md)
**Intuition:** [直觉说明 / Intuition — F007](knowledge/notes/a0c54399c2d44b8d-intuition.md)
**Definitions:**
- `C2_norm`: For a real C^2 function on the unit sphere, the sum of the sup norm of the function, the sup norm of its round-metric gradient, and the sup operator norm of its round-metric Hessian.
- `Pi_1`: The projection onto degree-one spherical harmonics: Pi_1 u(x)=(n/sigma(S^{n-1}))(integral_{S^{n-1}} y u(y) d sigma(y)) dot x.
- `T_m`: The normalized polynomial-kernel operator T_m h(x)=Z_m^{-1} integral_{S^{n-1}} ((1+x dot y)/2)^m h(y) d sigma(y), with Z_m the same integral without h.

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `a0c54399c2d44b8d`
**Verification:** {"formal_certificate": false, "provenance": "Historical source-run verdict; sanitized static snapshot, not rerun on export.", "verdict": "correct", "verifier": "independent-llm"}

</details>
<!-- hyprover:end -->

<!-- hyprover:fact {"author": "max", "created_at": "", "display_id": "F008", "external_refs": [], "fact_id": "a8378dfc6c55d238", "glossary_introduces": {}, "intuition_path": "", "predecessors": ["8f8adadaf10ea26e"], "problem_id": "arxiv-2606-19865", "proof_path": "knowledge/proofs/a8378dfc6c55d238.md", "revision": 1, "revocation": {}, "status": "verified", "title": "Every interior point of a convex body is the centroid of some hyperplane section", "verification": {"formal_certificate": false, "provenance": "Historical source-run verdict; sanitized static snapshot, not rerun on export.", "verdict": "correct", "verifier": "independent-llm"}} -->
<a id="fact-a8378dfc6c55d238"></a>
### F008 · Every interior point of a convex body is the centroid of some hyperplane section [verified]

<!-- hyprover:body -->
Let n>=2 be an integer and let K be any compact convex subset of R^n with nonempty interior. Use ordinary Euclidean Hausdorff measures, so vol_n=H^n, and define
c(K)=(1/vol_n(K)) integral_K x d vol_n(x).
For every affine hyperplane A with H^{n-1}(K intersect A)>0, define
c(K intersect A)=(1/H^{n-1}(K intersect A)) integral_{K intersect A} x d H^{n-1}(x).
Then c(K) lies in int(K). For every p in int(K), there exists an affine hyperplane A through p such that H^{n-1}(K intersect A)>0 and c(K intersect A)=p. No smoothness or strict convexity is assumed.

More precisely, for p in int(K) define on S^{n-1}
B_{K,p}(u)=vol_n(K intersect {x:u dot (x-p)>=0})-vol_n(K intersect {x:u dot (x-p)<=0}).
The function B_{K,p} is odd and C^1. Its critical points occur in antipodal pairs, and the map
{u,-u} -> p+u^\perp
is a bijection from those antipodal critical pairs onto the geometric affine hyperplanes A with positive section measure and c(K intersect A)=p. In particular every such A necessarily contains p, and every hyperplane through p has positive section measure.
Taking p=c(K), every convex body in every dimension n>=2 has at least one centroid-matching hyperplane section. With sections counted without orientation and mu(n) defined as in the project problem, this is the universal lower bound mu(n)>=1.
<!-- hyprover:end-body -->

**Depends on:** [F006 · Signed halfspace volume is C1 and its spherical gradient is twice the central-section first moment](#fact-8f8adadaf10ea26e)
**Proof:** [完整证明 / Full proof — F008](knowledge/proofs/a8378dfc6c55d238.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `a8378dfc6c55d238`
**Verification:** {"formal_certificate": false, "provenance": "Historical source-run verdict; sanitized static snapshot, not rerun on export.", "verdict": "correct", "verifier": "independent-llm"}

</details>
<!-- hyprover:end -->

<!-- hyprover:fact {"author": "max7", "created_at": "", "display_id": "F009", "external_refs": [], "fact_id": "7a3fd2eb4772e4b4", "glossary_introduces": {}, "intuition_path": "", "predecessors": ["83fc8a6190cb2ca3", "a0c54399c2d44b8d", "e9b3873b7c826c7b", "dc5ba3bf489a3dea"], "problem_id": "arxiv-2606-19865", "proof_path": "knowledge/proofs/7a3fd2eb4772e4b4.md", "revision": 1, "revocation": {}, "status": "verified", "title": "Smooth strictly convex bodies with exactly one centroid-matching hyperplane in every dimension n\u003e=3", "verification": {"formal_certificate": false, "provenance": "Historical source-run verdict; sanitized static snapshot, not rerun on export.", "verdict": "correct", "verifier": "independent-llm"}} -->
<a id="fact-7a3fd2eb4772e4b4"></a>
### F009 · Smooth strictly convex bodies with exactly one centroid-matching hyperplane in every dimension n&gt;=3 [verified]

<!-- hyprover:body -->
For every integer n>=3, there exists a compact convex body K in R^n with nonempty interior, C-infinity boundary and positive principal curvatures, strictly convex in the sense that for distinct x,y in K and 0<t<1 the point (1-t)x+t y lies in int(K), with the following property. Its uniform-volume centroid
c(K)=(vol_n K)^{-1} integral_K x dx
is 0, and exactly one geometric affine hyperplane A has both H^{n-1}(K intersect A)>0 and
c(K intersect A)=(H^{n-1}(K intersect A))^{-1} integral_{K intersect A} x dH^{n-1}(x)=c(K).
Hyperplanes are counted without orientation.

More precisely, for each n one can choose a real smooth function g on the Euclidean unit sphere S^{n-1}, which is a finite sum of spherical harmonics of odd degrees at least 3, and a number epsilon_0>0 such that every real epsilon with 0<|epsilon|<epsilon_0 gives such a body by
K_epsilon={r theta : theta in S^{n-1}, 0<=r<=(1+epsilon g(theta))^{1/(n+1)}}.
Here a spherical harmonic of degree k is the restriction to S^{n-1} of a real homogeneous degree-k polynomial on R^n annihilated by the Euclidean Laplacian. With sigma the usual Euclidean surface measure and
(Hf)(u)=integral_{S^{n-1}} sign(u dot theta) f(theta) d sigma(theta), sign(0)=0,
this g satisfies integral_{S^{n-1}} theta g(theta) d sigma(theta)=0, and Hg has exactly two nondegenerate critical points for the round metric, forming one antipodal pair.
<!-- hyprover:end-body -->

**Depends on:** [F001 · Smooth odd twisted heights with zero first moment and exactly two nondegenerate critical points](#fact-83fc8a6190cb2ca3), [F007 · Finite odd harmonic approximation with exact moment cancellation and two-critical-point stability](#fact-a0c54399c2d44b8d), [F002 · Nonzero odd eigenvalues and finite harmonic inversion of the hemispherical sign transform](#fact-e9b3873b7c826c7b), [F005 · Centered radial perturbations realize one centroidal hyperplane from a two-critical-point sign transform](#fact-dc5ba3bf489a3dea)
**Proof:** [完整证明 / Full proof — F009](knowledge/proofs/7a3fd2eb4772e4b4.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `7a3fd2eb4772e4b4`
**Verification:** {"formal_certificate": false, "provenance": "Historical source-run verdict; sanitized static snapshot, not rerun on export.", "verdict": "correct", "verifier": "independent-llm"}

</details>
<!-- hyprover:end -->

<!-- hyprover:fact {"author": "max3", "created_at": "", "display_id": "F010", "external_refs": [], "fact_id": "c7fafba4b4d5a72f", "glossary_introduces": {}, "intuition_path": "", "predecessors": ["83fc8a6190cb2ca3", "a0c54399c2d44b8d", "e9b3873b7c826c7b"], "problem_id": "arxiv-2606-19865", "proof_path": "knowledge/proofs/c7fafba4b4d5a72f.md", "revision": 1, "revocation": {}, "status": "verified", "title": "Finite odd harmonic preimages whose sign transforms have exactly one nondegenerate critical pair", "verification": {"formal_certificate": false, "provenance": "Historical source-run verdict; sanitized static snapshot, not rerun on export.", "verdict": "correct", "verifier": "independent-llm"}} -->
<a id="fact-c7fafba4b4d5a72f"></a>
### F010 · Finite odd harmonic preimages whose sign transforms have exactly one nondegenerate critical pair [verified]

<!-- hyprover:body -->
For every integer n>=3 there exists a real function g on S^{n-1} that is a finite sum of spherical harmonics of odd degrees at least 3, such that H g(u)=integral_{S^{n-1}} sign(u dot theta) g(theta) d sigma(theta) has exactly two critical points on S^{n-1}, which are antipodal and both nondegenerate. Here spherical harmonics of degree k are restrictions of homogeneous harmonic polynomials of degree k on R^n; sigma is ordinary Euclidean surface measure; critical points and Hessians are for the usual round metric.
<!-- hyprover:end-body -->

**Depends on:** [F001 · Smooth odd twisted heights with zero first moment and exactly two nondegenerate critical points](#fact-83fc8a6190cb2ca3), [F007 · Finite odd harmonic approximation with exact moment cancellation and two-critical-point stability](#fact-a0c54399c2d44b8d), [F002 · Nonzero odd eigenvalues and finite harmonic inversion of the hemispherical sign transform](#fact-e9b3873b7c826c7b)
**Proof:** [完整证明 / Full proof — F010](knowledge/proofs/c7fafba4b4d5a72f.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `c7fafba4b4d5a72f`
**Verification:** {"formal_certificate": false, "provenance": "Historical source-run verdict; sanitized static snapshot, not rerun on export.", "verdict": "correct", "verifier": "independent-llm"}

</details>
<!-- hyprover:end -->

<!-- hyprover:fact {"author": "max5", "created_at": "", "display_id": "F011", "external_refs": [], "fact_id": "81a5afe69f10c3d2", "glossary_introduces": {}, "intuition_path": "", "predecessors": ["83fc8a6190cb2ca3"], "problem_id": "arxiv-2606-19865", "proof_path": "knowledge/proofs/81a5afe69f10c3d2.md", "revision": 1, "revocation": {}, "status": "verified", "title": "Polynomial shears of the ball give unique centroid-matching sections in every dimension n\u003e=3", "verification": {"formal_certificate": false, "provenance": "Historical source-run verdict; sanitized static snapshot, not rerun on export.", "verdict": "correct", "verifier": "independent-llm"}} -->
<a id="fact-81a5afe69f10c3d2"></a>
### F011 · Polynomial shears of the ball give unique centroid-matching sections in every dimension n&gt;=3 [verified]

<!-- hyprover:body -->
Fix an integer n>=3. Let kappa_d denote the Euclidean volume of the unit ball in R^d, and set
w_n(t)=(1-t^2)^((n-1)/2), -1<=t<=1,
b_{n,k}=integral_{-1}^1 q^(2k)(1-q^2)^((n-2)/2)dq, k=0,1,2,... .
Every b_{n,k} is positive.

Suppose A and B are real polynomials such that
integral_{-1}^1 w_n(t) A(t^2)dt=integral_{-1}^1 w_n(t) B(t^2)dt=0,
and the function
Psi(u)=u_1 A(u_n^2)+u_2 B(u_n^2), u in S^{n-1},
has exactly two critical points for the round metric, both nondegenerate. Write the R^(n-1)-valued polynomial
P(r)=(A(1-r),B(1-r),0,...,0)=sum_{k=0}^N p_k r^k,
where p_k in R^(n-1), and define
v(z)=(2 kappa_{n-2})^(-1) sum_{k=0}^N (p_k/b_{n,k}) z^(2k).
For epsilon in R define
K_epsilon={(w,z) in R^(n-1) x R: |w-epsilon v(z)|^2+z^2<=1}.
There exists epsilon_0>0 such that every 0<|epsilon|<epsilon_0 gives a compact convex body K_epsilon with C-infinity boundary, strict convexity, volume centroid exactly 0, and exactly one geometric affine hyperplane L whose section has positive (n-1)-dimensional Euclidean measure and whose section centroid is 0. Hyperplanes are counted without orientation.

More precisely, with sign(0)=0 and
S_epsilon(u)=integral_{K_epsilon} sign(u dot x)dx,
one has ||S_epsilon/epsilon-Psi||_{C^2(S^{n-1})}=O(|epsilon|) as epsilon tends to 0 through nonzero values. Here the C^2 norm uses the value, spherical gradient and spherical Hessian. The two critical points of S_epsilon are nondegenerate and antipodal, and their common unoriented orthogonal hyperplane is the unique section in the preceding conclusion.

For every n>=3 there exist polynomials A and B satisfying the stated hypotheses. Consequently the explicit polynomial-shear family above supplies a smooth strictly convex body with exactly one centroid-matching hyperplane section in every dimension n>=3.
<!-- hyprover:end-body -->

**Depends on:** [F001 · Smooth odd twisted heights with zero first moment and exactly two nondegenerate critical points](#fact-83fc8a6190cb2ca3)
**Proof:** [完整证明 / Full proof — F011](knowledge/proofs/81a5afe69f10c3d2.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `81a5afe69f10c3d2`
**Verification:** {"formal_certificate": false, "provenance": "Historical source-run verdict; sanitized static snapshot, not rerun on export.", "verdict": "correct", "verifier": "independent-llm"}

</details>
<!-- hyprover:end -->

<!-- hyprover:fact {"author": "max6", "created_at": "", "display_id": "F012", "external_refs": [], "fact_id": "e738ef25f0351e92", "glossary_introduces": {"L_f": "For a centered convex body K and an even concave nonnegative height profile f, the symmetric homothetic suspension with height-t slice f(t)K."}, "intuition_path": "", "predecessors": [], "problem_id": "arxiv-2606-19865", "proof_path": "knowledge/proofs/e738ef25f0351e92.md", "revision": 1, "revocation": {}, "status": "verified", "title": "Cylinders have open families of centroid-matching sections; symmetric homothetic suspensions retain extra sections", "verification": {"formal_certificate": false, "provenance": "Historical source-run verdict; sanitized static snapshot, not rerun on export.", "verdict": "correct", "verifier": "independent-llm"}} -->
<a id="fact-e738ef25f0351e92"></a>
### F012 · Cylinders have open families of centroid-matching sections; symmetric homothetic suspensions retain extra sections [verified]

<!-- hyprover:body -->
Let d>=2, let K be a convex body in R^d with 0 in its interior and uniform-volume centroid c(K)=0, and let h>0. Centroids of bodies and sections below use their respective Euclidean Hausdorff measures.

(1) The cylinder C=K times [-h,h] has c(C)=0. For every a in R^d satisfying sup_{x in K}|a dot x|<=h, the hyperplane H_a={(x,t):t=a dot x} satisfies c(C intersect H_a)=0. In particular C has an open d-parameter family of distinct centroid-matching hyperplanes: all a with |a|<h/R are admissible, where R=max_{x in K}|x|>0.

(2) Let f:[-h,h]->[0,infinity) be continuous, even, and concave, with f(t)>0 for -h<t<h. Define the symmetric homothetic suspension
L_f={ (f(t)y,t): -h<=t<=h, y in K } subset R^{d+1}.
Then L_f is a convex body with centroid 0. Its horizontal section at t=0 has centroid 0. Moreover, for every unit vector u in R^d for which c(K intersect u^perp)=0, the vertical hyperplane u^perp times R satisfies c(L_f intersect (u^perp times R))=0. Thus if K has at least q distinct centroid-matching hyperplanes, for any integer q>=1, L_f has at least q+1 distinct centroid-matching hyperplanes. This conclusion in particular rules out preservation of uniqueness by this class of symmetric homothetic suspensions.

(3) For every invertible affine map T:R^{d+1}->R^{d+1}, the correspondence H -> T(H) is a bijection between centroid-matching hyperplanes of a convex body B and those of T(B). Consequently the multiplicities in (1) and (2) also hold for every invertible affine image of C or L_f.
<!-- hyprover:end-body -->

**Depends on:** none
**Proof:** [完整证明 / Full proof — F012](knowledge/proofs/e738ef25f0351e92.md)
**Definitions:**
- `L_f`: For a centered convex body K and an even concave nonnegative height profile f, the symmetric homothetic suspension with height-t slice f(t)K.

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `e738ef25f0351e92`
**Verification:** {"formal_certificate": false, "provenance": "Historical source-run verdict; sanitized static snapshot, not rerun on export.", "verdict": "correct", "verifier": "independent-llm"}

</details>
<!-- hyprover:end -->

<!-- hyprover:fact {"author": "max7", "created_at": "", "display_id": "F013", "external_refs": [], "fact_id": "846e296f22ef7853", "glossary_introduces": {}, "intuition_path": "", "predecessors": ["3ce5f83832ddd38c", "a8378dfc6c55d238", "7a3fd2eb4772e4b4"], "problem_id": "arxiv-2606-19865", "proof_path": "knowledge/proofs/846e296f22ef7853.md", "revision": 1, "revocation": {}, "status": "verified", "title": "Full centroid-section theorem: smooth sharp examples and mu(2)=3, mu(n)=1 for n\u003e=3", "verification": {"formal_certificate": false, "provenance": "Historical source-run verdict; sanitized static snapshot, not rerun on export.", "verdict": "correct", "verifier": "independent-llm"}} -->
<a id="fact-846e296f22ef7853"></a>
### F013 · Full centroid-section theorem: smooth sharp examples and mu(2)=3, mu(n)=1 for n&gt;=3 [verified]

<!-- hyprover:body -->
Let n>=2 be an integer. A convex body in R^n is a compact convex set K with nonempty interior. With H^k denoting ordinary Euclidean k-dimensional Hausdorff measure, define its uniform-volume centroid by
c(K)=(H^n(K))^{-1} integral_K x dH^n(x).
For an affine hyperplane A with H^{n-1}(K intersect A)>0, define
c(K intersect A)=(H^{n-1}(K intersect A))^{-1} integral_{K intersect A} x dH^{n-1}(x).
Sections are counted by distinct geometric affine hyperplanes, without orientation.

For every integer n>=3 there exists a convex body K in R^n whose boundary is a C-infinity submanifold, which is strictly convex in the sense that (1-t)x+t y belongs to int(K) for all distinct x,y in K and 0<t<1, and for which exactly one hyperplane section has centroid c(K).

If mu(n) denotes the largest integer q such that every convex body K in R^n has at least q distinct positive-measure hyperplane sections with centroid c(K), then
mu(2)=3, and mu(n)=1 for every integer n>=3.
The universal assertions quantify over all convex bodies, with no boundary smoothness or strict-convexity assumption: every such body in every n>=2 has at least one centroid-matching hyperplane section, and every planar such body has at least three.
<!-- hyprover:end-body -->

**Depends on:** [F003 · Sharp planar centroid-section theorem: every body has at least three, triangles exactly three](#fact-3ce5f83832ddd38c), [F008 · Every interior point of a convex body is the centroid of some hyperplane section](#fact-a8378dfc6c55d238), [F009 · Smooth strictly convex bodies with exactly one centroid-matching hyperplane in every dimension n&gt;=3](#fact-7a3fd2eb4772e4b4)
**Proof:** [完整证明 / Full proof — F013](knowledge/proofs/846e296f22ef7853.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `846e296f22ef7853`
**Verification:** {"formal_certificate": false, "provenance": "Historical source-run verdict; sanitized static snapshot, not rerun on export.", "verdict": "correct", "verifier": "independent-llm"}

</details>
<!-- hyprover:end -->

<a id="route"></a>
## Exploration routes — not facts

**本区目录 / Contents**
- [R001 · Universal and planar components are published and ready for completion revalidation](#entry-10acec896a9c45b2)
- [R002 · Use the published full-target fact as the completion root](#entry-f5a5afb3f0a8d294)
- [R003 · The direct shear alternative is published and retained as a fallback](#entry-97b5b6c76e914106)
- [R004 · Retain conditional lifting and the published boundary for naive products](#entry-a241367a37a8ed00)

<!-- hyprover:entry {"attributes": {}, "author": "main_agent", "created_at": "", "details_path": "", "display_id": "R001", "entry_id": "entry-10acec896a9c45b2", "kind": "route", "links": ["3ce5f83832ddd38c", "8f8adadaf10ea26e", "a8378dfc6c55d238", "entry-b92276d77e44cac5", "entry-127efe34f5e48303"], "revision": 3, "status": "active", "title": "Universal and planar components are published and ready for completion revalidation", "updated_at": ""} -->
<a id="entry-10acec896a9c45b2"></a>
### R001 · Universal and planar components are published and ready for completion revalidation [route: active; NOT A FACT]

<!-- hyprover:body -->
R001's required mathematical outputs are now published. F003 proves the full planar equality mu(2)=3. F006 proves the continuous-input sign-transform gradient identity and its application to every nonsmooth convex body about an interior point. F008, with F006 as its sole predecessor, proves centroid interiority and a matching hyperplane through every prescribed interior point in every n>=2; in particular mu(n)>=1 universally. All successful publication receipts have been read. The earlier glossary-only conflict is closed, and no smoothness hypothesis has entered the universal quantifier.

Decision: retain these terminal facts for the controller's fresh completion review and do not duplicate their proofs. Max's construction assignment is finished; max2's independent audit P015 found no issue with F006 and described the exact centroid application. Any additional review should target a concrete new issue in F008 or fresh completion feedback. The original fallback even-tangent-field argument need not be developed unless the potential proof is withdrawn. Closure for this branch is by the active facts, not by the audit report.
<!-- hyprover:end-body -->

**Related:** [F003 · Sharp planar centroid-section theorem: every body has at least three, triangles exactly three](#fact-3ce5f83832ddd38c), [F006 · Signed halfspace volume is C1 and its spherical gradient is twice the central-section first moment](#fact-8f8adadaf10ea26e), [F008 · Every interior point of a convex body is the centroid of some hyperplane section](#fact-a8378dfc6c55d238), [P014 · Universal lower bound and nonsmooth signed-volume interface are published](#entry-b92276d77e44cac5), [P015 · Nonsmooth universal interface audit found no issue; exact centroid application is ready](#entry-127efe34f5e48303)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-10acec896a9c45b2`

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {}, "author": "main_agent", "created_at": "", "details_path": "", "display_id": "R002", "entry_id": "entry-f5a5afb3f0a8d294", "kind": "route", "links": ["846e296f22ef7853", "3ce5f83832ddd38c", "a8378dfc6c55d238", "7a3fd2eb4772e4b4", "83fc8a6190cb2ca3", "a0c54399c2d44b8d", "e9b3873b7c826c7b", "dc5ba3bf489a3dea", "8f8adadaf10ea26e", "c7fafba4b4d5a72f", "entry-97b5b6c76e914106", "entry-a241367a37a8ed00"], "revision": 4, "status": "active", "title": "Use the published full-target fact as the completion root", "updated_at": ""} -->
<a id="entry-f5a5afb3f0a8d294"></a>
### R002 · Use the published full-target fact as the completion root [route: active; NOT A FACT]

<!-- hyprover:body -->
F013 (846e296f22ef7853) now publishes the exact full fixed theorem, with receipt accepted=true and published=true and with precisely F003, F008 and F009 as direct predecessors. Main read its entire proof and publication receipt. Choose F013 as the single completion root; its recursive closure is the previously selected eight proofs plus this explicit final implication. This changes no mathematical architecture or claim: the final implication has now itself passed ordinary independent publication.

The exact chain remains F009 <- F001/F007/F002/F005, F008 <- F006, and independent F003. F009 proves the all-n smooth strictly convex examples and mu(n)<=1 for n>=3; F008 proves the nonsmooth universal mu(n)>=1; F003 proves both planar inequalities. F013 applies these facts with the same measures, all dimensions and unoriented affine counting as PROBLEM.md. F010 also resolves C001 through the normal source gate, but is not an extra required dependency of F013.

R003 now has the published independent shear realization F011, sharing only F001 with the primary construction, and remains a fallback if a primary realization dependency needs repair. R004 remains parked with its conditional suspension theorem and the now-published product/symmetric-suspension boundary F012. Existing independent audits remain available; do not create redundant proofs to occupy workers. Submit the exact F013 closure for fresh full-proof and original-problem revalidation, and handle any rejection by a concrete repair rather than unchanged resubmission.
<!-- hyprover:end-body -->

**Related:** [F013 · Full centroid-section theorem: smooth sharp examples and mu(2)=3, mu(n)=1 for n&gt;=3](#fact-846e296f22ef7853), [F003 · Sharp planar centroid-section theorem: every body has at least three, triangles exactly three](#fact-3ce5f83832ddd38c), [F008 · Every interior point of a convex body is the centroid of some hyperplane section](#fact-a8378dfc6c55d238), [F009 · Smooth strictly convex bodies with exactly one centroid-matching hyperplane in every dimension n&gt;=3](#fact-7a3fd2eb4772e4b4), [F001 · Smooth odd twisted heights with zero first moment and exactly two nondegenerate critical points](#fact-83fc8a6190cb2ca3), [F007 · Finite odd harmonic approximation with exact moment cancellation and two-critical-point stability](#fact-a0c54399c2d44b8d), [F002 · Nonzero odd eigenvalues and finite harmonic inversion of the hemispherical sign transform](#fact-e9b3873b7c826c7b), [F005 · Centered radial perturbations realize one centroidal hyperplane from a two-critical-point sign transform](#fact-dc5ba3bf489a3dea), [F006 · Signed halfspace volume is C1 and its spherical gradient is twice the central-section first moment](#fact-8f8adadaf10ea26e), [F010 · Finite odd harmonic preimages whose sign transforms have exactly one nondegenerate critical pair](#fact-c7fafba4b4d5a72f), [R003 · The direct shear alternative is published and retained as a fallback](#entry-97b5b6c76e914106), [R004 · Retain conditional lifting and the published boundary for naive products](#entry-a241367a37a8ed00)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-f5a5afb3f0a8d294`

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {}, "author": "main_agent", "created_at": "", "details_path": "", "display_id": "R003", "entry_id": "entry-97b5b6c76e914106", "kind": "route", "links": ["81a5afe69f10c3d2", "83fc8a6190cb2ca3", "entry-94ccd77b205dba23", "entry-f5a5afb3f0a8d294"], "revision": 3, "status": "active", "title": "The direct shear alternative is published and retained as a fallback", "updated_at": ""} -->
<a id="entry-97b5b6c76e914106"></a>
### R003 · The direct shear alternative is published and retained as a fallback [route: active; NOT A FACT]

<!-- hyprover:body -->
F011 (81a5afe69f10c3d2) has now received accepted=true and published=true. Its complete statement gives polynomial disk-fiber shears with exact volume centroid 0, C-infinity boundary, strict convexity and exactly one centroid-matching geometric affine hyperplane for every n>=3. Its sole project predecessor is F001. The statement includes the precise polynomial hypotheses, explicit inverse coefficients, positive moment weights and uniform C2 signed-volume first variation. Thus this branch has reached a published all-dimensional realization, not merely a first-variation candidate.

Max6's P020 records a full manual proof audit, including all oblique normals and vertical limits, the disk centering identity, the Hessian bound on a convex cylinder containing every chord, implicit radial regularity, and global critical-point counting, with no unresolved issue in its checked scope. Main read the complete statement, dependency/glossary data, successful receipt and that audit; main has not separately reread F011's full proof, so P020 is not promoted to a fresh completion certificate.

Decision: retain this completed alternative during the fresh completion review of the selected primary F013 closure. No duplicate direct-construction task is needed. It can replace the harmonic-inversion/radial-realization branch if a defect there requires a route change, after an explicit new closure is selected and freshly checked. It does not independently remove the common F001 phase-construction dependency. If the primary closure passes, further development is unnecessary for the fixed target.
<!-- hyprover:end-body -->

**Related:** [F011 · Polynomial shears of the ball give unique centroid-matching sections in every dimension n&gt;=3](#fact-81a5afe69f10c3d2), [F001 · Smooth odd twisted heights with zero first moment and exactly two nondegenerate critical points](#fact-83fc8a6190cb2ca3), [P020 · Independent scrutiny of the published shear construction found no unresolved issue in the checked scope](#entry-94ccd77b205dba23), [R002 · Use the published full-target fact as the completion root](#entry-f5a5afb3f0a8d294)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-97b5b6c76e914106`

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {}, "author": "main_agent", "created_at": "", "details_path": "", "display_id": "R004", "entry_id": "entry-a241367a37a8ed00", "kind": "route", "links": ["947986eae4461bec", "e738ef25f0351e92", "83fc8a6190cb2ca3", "entry-97b5b6c76e914106"], "revision": 3, "status": "parked", "title": "Retain conditional lifting and the published boundary for naive products", "updated_at": ""} -->
<a id="entry-a241367a37a8ed00"></a>
### R004 · Retain conditional lifting and the published boundary for naive products [route: parked; NOT A FACT]

<!-- hyprover:body -->
R004 remains parked. Published F004 proves a suspension with local moment repair under an explicit smooth odd homotopy to height, with two nondegenerate critical points and fixed critical values at every parameter; the output retains that homotopy and permits iteration. A generic two-critical-point function or unique-section body is not assumed to satisfy the input. F001 already provides the primary potential directly for every n>=3, so lifting adds no missing target coverage.

Published F012 (e738ef25f0351e92) now proves the geometric anti-route precisely: a centered cylinder has an open family of centroid-matching graph hyperplanes, and an even concave homothetic suspension retains a horizontal matching section plus inherited vertical ones. Invertible affine transformations preserve these counts. The earlier V011 rejection concerned a missing finite-first-moment assumption in an overgeneral auxiliary paragraph; that candidate was repaired and the new complete proof received accepted=true and published=true. No proposition was refuted or used fact revoked by the earlier rejection.

Keep these statements for method boundaries and future fallback use. Max6's current follow-up has already audited the published shear alternative in P020. Resume lifting only if a lower-dimensional independent base is needed and its exact homotopy input can be established, or if the direct all-dimensional mechanism is withdrawn while such a base survives. This parking decision reflects completed dimension coverage, not failure of lifting.
<!-- hyprover:end-body -->

**Related:** [F004 · Dimension lifting of zero-moment odd potentials by suspension and local moment correction](#fact-947986eae4461bec), [F012 · Cylinders have open families of centroid-matching sections; symmetric homothetic suspensions retain extra sections](#fact-e738ef25f0351e92), [F001 · Smooth odd twisted heights with zero first moment and exactly two nondegenerate critical points](#fact-83fc8a6190cb2ca3), [R003 · The direct shear alternative is published and retained as a fallback](#entry-97b5b6c76e914106)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-a241367a37a8ed00`

</details>
<!-- hyprover:end -->

<a id="conjecture"></a>
## Conjectures — not facts

**本区目录 / Contents**
- [C001 · Finite odd harmonic potential with one nondegenerate antipodal critical pair](#entry-97292e71c3b56db4)

<!-- hyprover:entry {"attributes": {"resolution_fact_id": "c7fafba4b4d5a72f"}, "author": "main_agent", "created_at": "", "details_path": "", "display_id": "C001", "entry_id": "entry-97292e71c3b56db4", "kind": "conjecture", "links": ["entry-f5a5afb3f0a8d294", "c7fafba4b4d5a72f"], "revision": 2, "status": "proved", "title": "Finite odd harmonic potential with one nondegenerate antipodal critical pair", "updated_at": ""} -->
<a id="entry-97292e71c3b56db4"></a>
### C001 · Finite odd harmonic potential with one nondegenerate antipodal critical pair [conjecture: proved; NOT A FACT]

<!-- hyprover:body -->
For every integer n>=3 there exists a real function g on S^{n-1} that is a finite sum of spherical harmonics of odd degrees at least 3, such that H g(u)=integral_{S^{n-1}} sign(u dot theta) g(theta) d sigma(theta) has exactly two critical points on S^{n-1}, which are antipodal and both nondegenerate. Here spherical harmonics of degree k are restrictions of homogeneous harmonic polynomials of degree k on R^n; sigma is ordinary Euclidean surface measure; critical points and Hessians are for the usual round metric.
<!-- hyprover:end-body -->

**Related:** [R002 · Use the published full-target fact as the completion root](#entry-f5a5afb3f0a8d294), [F010 · Finite odd harmonic preimages whose sign transforms have exactly one nondegenerate critical pair](#fact-c7fafba4b4d5a72f)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-97292e71c3b56db4`
**Context:** {"resolution_fact_id": "c7fafba4b4d5a72f"}

</details>
<!-- hyprover:end -->

<a id="experience"></a>
## Experience and leads — not facts

**本区目录 / Contents**
- [E001 · F011 boundary-measure concern resolved as an expository clarification](#entry-683f18cbb33b8f19)
- [V001 · Verification submission outcome](#entry-dd562c18d7e8487c)
- [V002 · Verification submission outcome](#entry-6487a473fc0e4d94)
- [V003 · Verification submission outcome](#entry-3264e50313d7416b)
- [V004 · Verification submission outcome](#entry-7d0fc2492a0547d7)
- [V005 · Verification submission outcome](#entry-c0289974e2e74308)
- [V006 · Verification submission outcome](#entry-ea7c6a3452b34651)
- [V007 · Verification submission outcome](#entry-1d453c55f98a455f)
- [V008 · Verification submission outcome](#entry-4800d848113c408d)
- [V009 · Verification submission outcome](#entry-f855f425e4b3410b)
- [V010 · Verification submission outcome](#entry-5cc7ba3960734262)
- [V011 · Verification submission outcome](#entry-ef50e9d295b14126)
- [V012 · Verification submission outcome](#entry-89df6118370546f6)
- [V013 · Verification submission outcome](#entry-fc339b5c03e044ca)
- [V014 · Verification submission outcome](#entry-022e7d767c25435b)
- [V015 · Verification submission outcome](#entry-ae429dec0ca14be3)
- [V016 · Verification submission outcome](#entry-c72215b98f9c410b)
- [V017 · Fact review: revoke](#entry-ffad014f49254883)
- [V018 · Completion full revalidation](#entry-1f7592f4059e7fcc)

<!-- hyprover:entry {"attributes": {"audits": [{"not_covered": "The other parts of the proof, and the mathematical truth of the theorem beyond this written proof detail.", "revision": 1, "scope": "Part 6: the zero-measure boundary justification in the dominated-convergence proof of the section-moment identity.", "target_id": "81a5afe69f10c3d2"}], "research_policy_version": 1, "subtype": "review"}, "author": "max5", "created_at": "", "details_path": "", "display_id": "E001", "entry_id": "entry-683f18cbb33b8f19", "kind": "experience", "links": ["81a5afe69f10c3d2", "entry-022e7d767c25435b", "entry-ffad014f49254883"], "revision": 2, "status": "open", "title": "F011 boundary-measure concern resolved as an expository clarification", "updated_at": ""} -->
<a id="entry-683f18cbb33b8f19"></a>
### E001 · F011 boundary-measure concern resolved as an expository clarification [experience: open; NOT A FACT]

<!-- hyprover:body -->
F011 (81a5afe69f10c3d2, revision 1) was independently accepted and published with no errors or gaps. My post-submission concern isolated part 6's annulus (1+delta)C minus (1-delta)int C: scaling that annulus alone gives the limit vol(C)-vol(int C), so its claimed vanishing needs the equality of those volumes. I supplied the direct estimate boundary(C) subset C minus (1-delta)C, using (1-delta)C subset relint(C), as a clarification.

Fresh lifecycle review run [run identifier omitted] rejected the revocation justification: accepted=false, applied=false, no revoked facts, feedback entry-ffad014f49254883. The reviewer explained that the preceding paragraph already proves the interior-ball construction for strict contractions. It implies lambda^(n-1)vol(C)<=vol(relint C)<=vol(C) for every 0<lambda<1; taking lambda to 1 proves volume equality without assuming null boundary. Thus the contraction justification was already present in the written proof, and only its immediate volume consequence was abbreviated. I accept that assessment: no material gap or theorem refutation is established, F011 remains verified at revision 1, and no resubmission or lifecycle change is warranted. The shorter contracted-body estimate is an expository clarification for any later presentation.
<!-- hyprover:end-body -->

**Related:** [F011 · Polynomial shears of the ball give unique centroid-matching sections in every dimension n&gt;=3](#fact-81a5afe69f10c3d2), [V014 · Verification submission outcome](#entry-022e7d767c25435b), [V017 · Fact review: revoke](#entry-ffad014f49254883)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-683f18cbb33b8f19`
**Context:** {"audits": [{"not_covered": "The other parts of the proof, and the mathematical truth of the theorem beyond this written proof detail.", "revision": 1, "scope": "Part 6: the zero-measure boundary justification in the dominated-convergence proof of the section-moment identity.", "target_id": "81a5afe69f10c3d2"}], "research_policy_version": 1, "subtype": "review"}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"published": true}, "author": "verifier", "created_at": "", "details_path": "knowledge/notes/entry-dd562c18d7e8487c-1.md", "display_id": "V001", "entry_id": "entry-dd562c18d7e8487c", "kind": "feedback", "links": ["83fc8a6190cb2ca3"], "revision": 1, "status": "recorded", "title": "Verification submission outcome", "updated_at": ""} -->
<a id="entry-dd562c18d7e8487c"></a>
### V001 · Verification submission outcome [feedback: recorded; NOT A FACT]

<!-- hyprover:body -->
The independent verification service published this candidate.
<!-- hyprover:end-body -->

**Related:** [F001 · Smooth odd twisted heights with zero first moment and exactly two nondegenerate critical points](#fact-83fc8a6190cb2ca3)
**Details:** [详细说明 / Details — V001](knowledge/notes/entry-dd562c18d7e8487c-1.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-dd562c18d7e8487c`
**Context:** {"published": true}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"published": true}, "author": "verifier", "created_at": "", "details_path": "knowledge/notes/entry-6487a473fc0e4d94-1.md", "display_id": "V002", "entry_id": "entry-6487a473fc0e4d94", "kind": "feedback", "links": ["e9b3873b7c826c7b"], "revision": 1, "status": "recorded", "title": "Verification submission outcome", "updated_at": ""} -->
<a id="entry-6487a473fc0e4d94"></a>
### V002 · Verification submission outcome [feedback: recorded; NOT A FACT]

<!-- hyprover:body -->
The independent verification service published this candidate.
<!-- hyprover:end-body -->

**Related:** [F002 · Nonzero odd eigenvalues and finite harmonic inversion of the hemispherical sign transform](#fact-e9b3873b7c826c7b)
**Details:** [详细说明 / Details — V002](knowledge/notes/entry-6487a473fc0e4d94-1.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-6487a473fc0e4d94`
**Context:** {"published": true}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"published": true}, "author": "verifier", "created_at": "", "details_path": "knowledge/notes/entry-3264e50313d7416b-1.md", "display_id": "V003", "entry_id": "entry-3264e50313d7416b", "kind": "feedback", "links": ["3ce5f83832ddd38c"], "revision": 1, "status": "recorded", "title": "Verification submission outcome", "updated_at": ""} -->
<a id="entry-3264e50313d7416b"></a>
### V003 · Verification submission outcome [feedback: recorded; NOT A FACT]

<!-- hyprover:body -->
The independent verification service published this candidate.
<!-- hyprover:end-body -->

**Related:** [F003 · Sharp planar centroid-section theorem: every body has at least three, triangles exactly three](#fact-3ce5f83832ddd38c)
**Details:** [详细说明 / Details — V003](knowledge/notes/entry-3264e50313d7416b-1.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-3264e50313d7416b`
**Context:** {"published": true}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"published": false}, "author": "verifier", "created_at": "", "details_path": "knowledge/notes/entry-7d0fc2492a0547d7-1.md", "display_id": "V004", "entry_id": "entry-7d0fc2492a0547d7", "kind": "feedback", "links": [], "revision": 1, "status": "recorded", "title": "Verification submission outcome", "updated_at": ""} -->
<a id="entry-7d0fc2492a0547d7"></a>
### V004 · Verification submission outcome [feedback: recorded; NOT A FACT]

<!-- hyprover:body -->
This candidate was not published. A failed proof is not a refuted proposition.
<!-- hyprover:end-body -->

**Related:** none
**Details:** [详细说明 / Details — V004](knowledge/notes/entry-7d0fc2492a0547d7-1.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-7d0fc2492a0547d7`
**Context:** {"published": false}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"published": false}, "author": "verifier", "created_at": "", "details_path": "knowledge/notes/entry-c0289974e2e74308-1.md", "display_id": "V005", "entry_id": "entry-c0289974e2e74308", "kind": "feedback", "links": [], "revision": 1, "status": "recorded", "title": "Verification submission outcome", "updated_at": ""} -->
<a id="entry-c0289974e2e74308"></a>
### V005 · Verification submission outcome [feedback: recorded; NOT A FACT]

<!-- hyprover:body -->
This candidate was not published. A failed proof is not a refuted proposition.
<!-- hyprover:end-body -->

**Related:** none
**Details:** [详细说明 / Details — V005](knowledge/notes/entry-c0289974e2e74308-1.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-c0289974e2e74308`
**Context:** {"published": false}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"published": true}, "author": "verifier", "created_at": "", "details_path": "knowledge/notes/entry-ea7c6a3452b34651-1.md", "display_id": "V006", "entry_id": "entry-ea7c6a3452b34651", "kind": "feedback", "links": ["947986eae4461bec"], "revision": 1, "status": "recorded", "title": "Verification submission outcome", "updated_at": ""} -->
<a id="entry-ea7c6a3452b34651"></a>
### V006 · Verification submission outcome [feedback: recorded; NOT A FACT]

<!-- hyprover:body -->
The independent verification service published this candidate.
<!-- hyprover:end-body -->

**Related:** [F004 · Dimension lifting of zero-moment odd potentials by suspension and local moment correction](#fact-947986eae4461bec)
**Details:** [详细说明 / Details — V006](knowledge/notes/entry-ea7c6a3452b34651-1.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-ea7c6a3452b34651`
**Context:** {"published": true}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"published": true}, "author": "verifier", "created_at": "", "details_path": "knowledge/notes/entry-1d453c55f98a455f-1.md", "display_id": "V007", "entry_id": "entry-1d453c55f98a455f", "kind": "feedback", "links": ["dc5ba3bf489a3dea"], "revision": 1, "status": "recorded", "title": "Verification submission outcome", "updated_at": ""} -->
<a id="entry-1d453c55f98a455f"></a>
### V007 · Verification submission outcome [feedback: recorded; NOT A FACT]

<!-- hyprover:body -->
The independent verification service published this candidate.
<!-- hyprover:end-body -->

**Related:** [F005 · Centered radial perturbations realize one centroidal hyperplane from a two-critical-point sign transform](#fact-dc5ba3bf489a3dea)
**Details:** [详细说明 / Details — V007](knowledge/notes/entry-1d453c55f98a455f-1.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-1d453c55f98a455f`
**Context:** {"published": true}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"published": true}, "author": "verifier", "created_at": "", "details_path": "knowledge/notes/entry-4800d848113c408d-1.md", "display_id": "V008", "entry_id": "entry-4800d848113c408d", "kind": "feedback", "links": ["8f8adadaf10ea26e"], "revision": 1, "status": "recorded", "title": "Verification submission outcome", "updated_at": ""} -->
<a id="entry-4800d848113c408d"></a>
### V008 · Verification submission outcome [feedback: recorded; NOT A FACT]

<!-- hyprover:body -->
The independent verification service published this candidate.
<!-- hyprover:end-body -->

**Related:** [F006 · Signed halfspace volume is C1 and its spherical gradient is twice the central-section first moment](#fact-8f8adadaf10ea26e)
**Details:** [详细说明 / Details — V008](knowledge/notes/entry-4800d848113c408d-1.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-4800d848113c408d`
**Context:** {"published": true}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"published": true}, "author": "verifier", "created_at": "", "details_path": "knowledge/notes/entry-f855f425e4b3410b-1.md", "display_id": "V009", "entry_id": "entry-f855f425e4b3410b", "kind": "feedback", "links": ["a0c54399c2d44b8d"], "revision": 1, "status": "recorded", "title": "Verification submission outcome", "updated_at": ""} -->
<a id="entry-f855f425e4b3410b"></a>
### V009 · Verification submission outcome [feedback: recorded; NOT A FACT]

<!-- hyprover:body -->
The independent verification service published this candidate.
<!-- hyprover:end-body -->

**Related:** [F007 · Finite odd harmonic approximation with exact moment cancellation and two-critical-point stability](#fact-a0c54399c2d44b8d)
**Details:** [详细说明 / Details — V009](knowledge/notes/entry-f855f425e4b3410b-1.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-f855f425e4b3410b`
**Context:** {"published": true}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"published": true}, "author": "verifier", "created_at": "", "details_path": "knowledge/notes/entry-5cc7ba3960734262-1.md", "display_id": "V010", "entry_id": "entry-5cc7ba3960734262", "kind": "feedback", "links": ["a8378dfc6c55d238"], "revision": 1, "status": "recorded", "title": "Verification submission outcome", "updated_at": ""} -->
<a id="entry-5cc7ba3960734262"></a>
### V010 · Verification submission outcome [feedback: recorded; NOT A FACT]

<!-- hyprover:body -->
The independent verification service published this candidate.
<!-- hyprover:end-body -->

**Related:** [F008 · Every interior point of a convex body is the centroid of some hyperplane section](#fact-a8378dfc6c55d238)
**Details:** [详细说明 / Details — V010](knowledge/notes/entry-5cc7ba3960734262-1.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-5cc7ba3960734262`
**Context:** {"published": true}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"published": false}, "author": "verifier", "created_at": "", "details_path": "knowledge/notes/entry-ef50e9d295b14126-1.md", "display_id": "V011", "entry_id": "entry-ef50e9d295b14126", "kind": "feedback", "links": [], "revision": 1, "status": "recorded", "title": "Verification submission outcome", "updated_at": ""} -->
<a id="entry-ef50e9d295b14126"></a>
### V011 · Verification submission outcome [feedback: recorded; NOT A FACT]

<!-- hyprover:body -->
This candidate was not published. A failed proof is not a refuted proposition.
<!-- hyprover:end-body -->

**Related:** none
**Details:** [详细说明 / Details — V011](knowledge/notes/entry-ef50e9d295b14126-1.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-ef50e9d295b14126`
**Context:** {"published": false}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"published": true}, "author": "verifier", "created_at": "", "details_path": "knowledge/notes/entry-89df6118370546f6-1.md", "display_id": "V012", "entry_id": "entry-89df6118370546f6", "kind": "feedback", "links": ["7a3fd2eb4772e4b4"], "revision": 1, "status": "recorded", "title": "Verification submission outcome", "updated_at": ""} -->
<a id="entry-89df6118370546f6"></a>
### V012 · Verification submission outcome [feedback: recorded; NOT A FACT]

<!-- hyprover:body -->
The independent verification service published this candidate.
<!-- hyprover:end-body -->

**Related:** [F009 · Smooth strictly convex bodies with exactly one centroid-matching hyperplane in every dimension n&gt;=3](#fact-7a3fd2eb4772e4b4)
**Details:** [详细说明 / Details — V012](knowledge/notes/entry-89df6118370546f6-1.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-89df6118370546f6`
**Context:** {"published": true}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"published": true}, "author": "verifier", "created_at": "", "details_path": "knowledge/notes/entry-fc339b5c03e044ca-1.md", "display_id": "V013", "entry_id": "entry-fc339b5c03e044ca", "kind": "feedback", "links": ["entry-97292e71c3b56db4", "c7fafba4b4d5a72f"], "revision": 1, "status": "recorded", "title": "Verification submission outcome", "updated_at": ""} -->
<a id="entry-fc339b5c03e044ca"></a>
### V013 · Verification submission outcome [feedback: recorded; NOT A FACT]

<!-- hyprover:body -->
The independent verification service published this candidate.
<!-- hyprover:end-body -->

**Related:** [C001 · Finite odd harmonic potential with one nondegenerate antipodal critical pair](#entry-97292e71c3b56db4), [F010 · Finite odd harmonic preimages whose sign transforms have exactly one nondegenerate critical pair](#fact-c7fafba4b4d5a72f)
**Details:** [详细说明 / Details — V013](knowledge/notes/entry-fc339b5c03e044ca-1.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-fc339b5c03e044ca`
**Context:** {"published": true}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"published": true}, "author": "verifier", "created_at": "", "details_path": "knowledge/notes/entry-022e7d767c25435b-1.md", "display_id": "V014", "entry_id": "entry-022e7d767c25435b", "kind": "feedback", "links": ["81a5afe69f10c3d2"], "revision": 1, "status": "recorded", "title": "Verification submission outcome", "updated_at": ""} -->
<a id="entry-022e7d767c25435b"></a>
### V014 · Verification submission outcome [feedback: recorded; NOT A FACT]

<!-- hyprover:body -->
The independent verification service published this candidate.
<!-- hyprover:end-body -->

**Related:** [F011 · Polynomial shears of the ball give unique centroid-matching sections in every dimension n&gt;=3](#fact-81a5afe69f10c3d2)
**Details:** [详细说明 / Details — V014](knowledge/notes/entry-022e7d767c25435b-1.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-022e7d767c25435b`
**Context:** {"published": true}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"published": true}, "author": "verifier", "created_at": "", "details_path": "knowledge/notes/entry-ae429dec0ca14be3-1.md", "display_id": "V015", "entry_id": "entry-ae429dec0ca14be3", "kind": "feedback", "links": ["e738ef25f0351e92"], "revision": 1, "status": "recorded", "title": "Verification submission outcome", "updated_at": ""} -->
<a id="entry-ae429dec0ca14be3"></a>
### V015 · Verification submission outcome [feedback: recorded; NOT A FACT]

<!-- hyprover:body -->
The independent verification service published this candidate.
<!-- hyprover:end-body -->

**Related:** [F012 · Cylinders have open families of centroid-matching sections; symmetric homothetic suspensions retain extra sections](#fact-e738ef25f0351e92)
**Details:** [详细说明 / Details — V015](knowledge/notes/entry-ae429dec0ca14be3-1.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-ae429dec0ca14be3`
**Context:** {"published": true}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"published": true}, "author": "verifier", "created_at": "", "details_path": "knowledge/notes/entry-c72215b98f9c410b-1.md", "display_id": "V016", "entry_id": "entry-c72215b98f9c410b", "kind": "feedback", "links": ["846e296f22ef7853"], "revision": 1, "status": "recorded", "title": "Verification submission outcome", "updated_at": ""} -->
<a id="entry-c72215b98f9c410b"></a>
### V016 · Verification submission outcome [feedback: recorded; NOT A FACT]

<!-- hyprover:body -->
The independent verification service published this candidate.
<!-- hyprover:end-body -->

**Related:** [F013 · Full centroid-section theorem: smooth sharp examples and mu(2)=3, mu(n)=1 for n&gt;=3](#fact-846e296f22ef7853)
**Details:** [详细说明 / Details — V016](knowledge/notes/entry-c72215b98f9c410b-1.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-c72215b98f9c410b`
**Context:** {"published": true}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"action": "revoke", "applied": false, "operation": "fact_review", "requested_by": "max5"}, "author": "verifier", "created_at": "", "details_path": "knowledge/notes/entry-ffad014f49254883-1.md", "display_id": "V017", "entry_id": "entry-ffad014f49254883", "kind": "feedback", "links": ["81a5afe69f10c3d2"], "revision": 1, "status": "recorded", "title": "Fact review: revoke", "updated_at": ""} -->
<a id="entry-ffad014f49254883"></a>
### V017 · Fact review: revoke [feedback: recorded; NOT A FACT]

<!-- hyprover:body -->
Review not applied. Please independently assess a concrete proof-justification concern in part 6 of F011 revision 1. The proof says that boundary(C) is contained in (1+delta)C minus (1-delta)int C and that this annulus's measure tends to zero by scaling. Scaling alone gives the limit vol(C)-vol(int C), so this sentence, read literally as a proof of zero boundary measure, is circular. I am not claiming the theorem false. A complete elementary correction is available: 0 in relint(C) and convexity imply (1-delta)C subset relint(C); thus boundary(C) subset C minus (1-delta)C, whose volume is [1-(1-delta)^(n-1)]vol(C), tending to zero. Every other argument and the statement are unchanged. Decide whether this is a genuine proof defect warranting revocation or merely an inessential abbreviation covered by standard elementary convexity. If revocation is accepted and applied, I will resubmit the same theorem with the corrected paragraph; I will not edit published proofs or bypass the gate. This review concerns the written boundary-measure argument only, not a failed experiment, capability limit, or claimed theorem refutation.
<!-- hyprover:end-body -->

**Related:** [F011 · Polynomial shears of the ball give unique centroid-matching sections in every dimension n&gt;=3](#fact-81a5afe69f10c3d2)
**Details:** [详细说明 / Details — V017](knowledge/notes/entry-ffad014f49254883-1.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-ffad014f49254883`
**Context:** {"action": "revoke", "applied": false, "operation": "fact_review", "requested_by": "max5"}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"operation": "completion_revalidation", "outcome": "passed"}, "author": "verifier", "created_at": "", "details_path": "knowledge/notes/entry-1f7592f4059e7fcc-1.md", "display_id": "V018", "entry_id": "entry-1f7592f4059e7fcc", "kind": "feedback", "links": [], "revision": 1, "status": "recorded", "title": "Completion full revalidation", "updated_at": ""} -->
<a id="entry-1f7592f4059e7fcc"></a>
### V018 · Completion full revalidation [feedback: recorded; NOT A FACT]

<!-- hyprover:body -->
Full dependency proofs and final original-problem bridge passed fresh LLM revalidation; not a formal certificate.
<!-- hyprover:end-body -->

**Related:** none
**Details:** [详细说明 / Details — V018](knowledge/notes/entry-1f7592f4059e7fcc-1.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-1f7592f4059e7fcc`
**Context:** {"operation": "completion_revalidation", "outcome": "passed"}

</details>
<!-- hyprover:end -->

<a id="report"></a>
## Public reports — project-shared, not facts

**本区目录 / Contents**
- [P001 · Initial analytic handoff: candidate phase repair, polynomial approximation and rotation estimates](#entry-277738d6e0016ccd)
- [P002 · Initial control review and macro audit: four routes activated](#entry-dd8c8326b325fc03)
- [P003 · Exact phase pairing supplies the all-dimensional smooth twisted-height potential](#entry-dab4ef0d1e7e4845)
- [P004 · R004 handoff complete: published conditional lifting, product obstruction, and shear-proof audit](#entry-fac08b0417afe30e)
- [P005 · Direct disk-fiber shear gives a candidate replacement for spherical harmonic inversion](#entry-04743c91041fed68)
- [P006 · Finite-harmonic sign-transform inversion is published and ready for the analytic construction](#entry-bfe253a084d94ef6)
- [P007 · Planar branch closed: published proof of mu(2)=3](#entry-c8473e86b97a5e1b)
- [P008 · Publication repair for V004 and V005: reuse the existing H glossary entry](#entry-a181734267aec9d9)
- [P009 · Full theorem published as F013; exact dependency handoff for completion revalidation](#entry-a97772fe5de56e5b)
- [P010 · Direct shear route now has a complete all-dimensional proof plan using F001](#entry-67a9664998c4132f)
- [P011 · Finite harmonic bridge published as F007; C2 and moment interfaces are fully matched](#entry-1e38b3add2d300cf)
- [P012 · Published inversion and radial realization have matching hypotheses and normalizations](#entry-edb4977d81b250ba)
- [P013 · Twisted-height input matches the published finite approximation theorem without extra hypotheses](#entry-0488e67f78918534)
- [P014 · Universal lower bound and nonsmooth signed-volume interface are published](#entry-b92276d77e44cac5)
- [P015 · Nonsmooth universal interface audit found no issue; exact centroid application is ready](#entry-127efe34f5e48303)
- [P016 · C001 is proved as F010; the finite approximation proof and inversion interface pass manual audit](#entry-728de1323489ba0c)
- [P017 · F005 audit found no gap in global convexity or the exact critical-point count](#entry-185782bd07e6c33d)
- [P018 · F008 audit closes the nonsmooth universal implication without additional hypotheses](#entry-5f720ff72a226fde)
- [P019 · Main closure audit: three terminal facts cover the original theorem through eight proofs](#entry-fdc351fd4710f996)
- [P020 · Independent scrutiny of the published shear construction found no unresolved issue in the checked scope](#entry-94ccd77b205dba23)
- [P021 · Completion handoff updated to F013 after the full theorem itself was published](#entry-2d93b9b86ba446a8)
- [P022 · R003 closed: direct polynomial-shear bodies work in every n&gt;=3](#entry-e1b533b44009ca5b)

<!-- hyprover:entry {"attributes": {"research_policy_version": 1}, "author": "main_agent", "created_at": "", "details_path": "", "display_id": "P001", "entry_id": "entry-277738d6e0016ccd", "kind": "report", "links": ["entry-f5a5afb3f0a8d294", "entry-97292e71c3b56db4"], "revision": 1, "status": "recorded", "title": "Initial analytic handoff: candidate phase repair, polynomial approximation and rotation estimates", "updated_at": ""} -->
<a id="entry-277738d6e0016ccd"></a>
### P001 · Initial analytic handoff: candidate phase repair, polynomial approximation and rotation estimates [report: recorded; NOT A FACT]

<!-- hyprover:body -->
Unverified textual reasoning for R002's assigned workers; this report is not a proof premise. The twist proposal has a concrete finite-parameter repair: choose two thresholds in t^2 so the weight (1-t^2)^((n-1)/2) on [-1,1] gives equal mass to three even regions. Smooth their step functions s1,s2 and set alpha=(2pi/3+a)s1+(2pi/3+b-a)s2. In the step limit the normalized complex moment is [1+exp(i(2pi/3+a))+exp(i(4pi/3+b))]/3. At (a,b)=(0,0) it vanishes and its two real derivative columns are independent. A justified small-smoothing inverse-function argument should repair the exact moment while preserving even smooth alpha. Max4 must supply that argument and all geometric checks.

For max8, a possible self-contained C2 approximation mechanism is the normalized polynomial kernel T_N F(u)=c_N integral (1+u dot v)^N F(v)d sigma(v). Concentration near v=u and commutation with rotation vector fields should give uniform convergence of F and derivatives through order two. Orthogonal equivariance should preserve oddness and zero first spherical moment. The restriction is a finite polynomial and can be decomposed into finitely many spherical harmonics. All estimates/decomposition still need proof; no computation has been run.

For max3/max7, H commutes with rotations. Applying rotation vector fields to the input should provide the required boundedness H:C2->C2 without differentiating the discontinuous sign kernel directly. For rho_epsilon^(n+1)=1+epsilon*g with odd g, H annihilates the even powers in the scalar Taylor expansion; this may improve the scaled remainder to O(epsilon^2), although an O(epsilon) C2 remainder is already sufficient. Check constants and norms independently. These suggestions sharpen already assigned interfaces and do not certify C001.
<!-- hyprover:end-body -->

**Related:** [R002 · Use the published full-target fact as the completion root](#entry-f5a5afb3f0a8d294), [C001 · Finite odd harmonic potential with one nondegenerate antipodal critical pair](#entry-97292e71c3b56db4)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-277738d6e0016ccd`
**Context:** {"research_policy_version": 1}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"audits": [{"not_covered": "Full mathematical proof and independent verification/publication of the proposed claims; no facts have yet been published.", "revision": 1, "scope": "Strategic coverage of universal lower bounds and both planar inequalities; candidate mechanisms, nonsmooth-body risks, worker allocation and revisit condition.", "target_id": "entry-10acec896a9c45b2"}, {"not_covered": "Full mathematical proof and independent verification/publication of the proposed claims; no facts have yet been published.", "revision": 1, "scope": "All stages of the primary all-dimensional construction, exact input/output interfaces, candidate evidence, decisive unproved steps, four-worker allocation and fallback triggers.", "target_id": "entry-f5a5afb3f0a8d294"}, {"not_covered": "Full mathematical proof and independent verification/publication of the proposed claims; no facts have yet been published.", "revision": 1, "scope": "Independent geometric alternative's initial frontier, absent base example, smoothing/centering/uniqueness obstacles, allocation and first-return decision criteria.", "target_id": "entry-97b5b6c76e914106"}, {"not_covered": "Full mathematical proof and independent verification/publication of the proposed claims; no facts have yet been published.", "revision": 1, "scope": "Dimension-extension alternative's conditional input, all-oblique-section obstacle, direct-all-n competitor, allocation and parking condition.", "target_id": "entry-a241367a37a8ed00"}, {"not_covered": "Full mathematical proof and independent verification/publication of the proposed claims; no facts have yet been published.", "revision": 1, "scope": "Alignment with every clause of the original problem and separation of speculative architecture from verified knowledge.", "target_id": "entry-61d4f0a0a2f9515c"}, {"not_covered": "Full mathematical proof and independent verification/publication of the proposed claims; no facts have yet been published.", "revision": 1, "scope": "Usefulness and precise role of the finite odd harmonic critical-pair bridge in the proposed dependency closure.", "target_id": "entry-97292e71c3b56db4"}], "research_policy_version": 1}, "author": "main_agent", "created_at": "", "details_path": "", "display_id": "P002", "entry_id": "entry-dd8c8326b325fc03", "kind": "report", "links": ["entry-10acec896a9c45b2", "entry-f5a5afb3f0a8d294", "entry-97b5b6c76e914106", "entry-a241367a37a8ed00", "entry-61d4f0a0a2f9515c", "entry-97292e71c3b56db4", "entry-277738d6e0016ccd"], "revision": 1, "status": "recorded", "title": "Initial control review and macro audit: four routes activated", "updated_at": ""} -->
<a id="entry-dd8c8326b325fc03"></a>
### P002 · Initial control review and macro audit: four routes activated [report: recorded; NOT A FACT]

<!-- hyprover:body -->
This initial strategic control review and macro audit checked the fixed problem, the notebook, public reports and service feedback before any fact had been published. The four routes below were assigned under the existing research policy. The review used textual reasoning without external retrieval.

R001 continues as an independent required component: max proves the universal lower bound and potential/section interface for nonsmooth bodies; max2 proves both planar inequalities. Their frontiers are complete proofs, not dependence on the supplied theorem assertion. The key risks are regularity, degenerate zero sets and counting geometric lines. These requirements remain necessary whichever higher-dimensional route succeeds.

R002 is the primary construction route because it gives a concrete separation of the exact centroid constraint, critical-point uniqueness and near-ball convexity. Main's newly developed twist mechanism is promising textual evidence only. Max4 must establish exact smooth phase cancellation and the two critical points; max8 must preserve those features under finite harmonic approximation; max3 must prove transform inversion; max7 must convert a matching input into the full smooth strictly convex example. C001 is a precise intermediate closure criterion. Every interface remains unverified. Keep the four roles distinct, and reassess immediately if an eigenvalue, moment, regularity or global-uniqueness obstruction appears.

R003 stays active as a complementary route: max5 seeks a direct geometric model starting in dimension three. No model is yet available; recentering and smoothing are its decisive risks. Retain the investigation through its first substantive return, then deepen an explicit section formula or preserve a precise obstruction. R004 stays active with max6 testing a dimension-extension interface. It presently lacks both a base example and a transfer lemma; oblique hyperplanes and extra product symmetries are decisive obstacles. Park lifting only if the direct all-dimensional construction closes with matching verified inputs, or redirect it when a concrete boundary makes it unproductive.

No mathematical route has failed and the original proposition has not been weakened. Decisions are recorded separately in R001-R004; O001 gives the mathematical synthesis and C001 the central candidate bridge. Next controller pass: inspect actual public progress and verification receipts, read complete statements/glossaries/dependencies of new facts, repair mismatched interfaces, and issue genuinely useful follow-ups. Completion requires the universal and planar proofs plus the all-n high-dimensional existence proof and a precise final implication; none is currently closed.
<!-- hyprover:end-body -->

**Related:** [R001 · Universal and planar components are published and ready for completion revalidation](#entry-10acec896a9c45b2), [R002 · Use the published full-target fact as the completion root](#entry-f5a5afb3f0a8d294), [R003 · The direct shear alternative is published and retained as a fallback](#entry-97b5b6c76e914106), [R004 · Retain conditional lifting and the published boundary for naive products](#entry-a241367a37a8ed00), [O001 · The full fixed theorem is published; its complete proof closure awaits fresh revalidation](#entry-61d4f0a0a2f9515c), [C001 · Finite odd harmonic potential with one nondegenerate antipodal critical pair](#entry-97292e71c3b56db4), [P001 · Initial analytic handoff: candidate phase repair, polynomial approximation and rotation estimates](#entry-277738d6e0016ccd)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-dd8c8326b325fc03`
**Context:** {"audits": [{"not_covered": "Full mathematical proof and independent verification/publication of the proposed claims; no facts have yet been published.", "revision": 1, "scope": "Strategic coverage of universal lower bounds and both planar inequalities; candidate mechanisms, nonsmooth-body risks, worker allocation and revisit condition.", "target_id": "entry-10acec896a9c45b2"}, {"not_covered": "Full mathematical proof and independent verification/publication of the proposed claims; no facts have yet been published.", "revision": 1, "scope": "All stages of the primary all-dimensional construction, exact input/output interfaces, candidate evidence, decisive unproved steps, four-worker allocation and fallback triggers.", "target_id": "entry-f5a5afb3f0a8d294"}, {"not_covered": "Full mathematical proof and independent verification/publication of the proposed claims; no facts have yet been published.", "revision": 1, "scope": "Independent geometric alternative's initial frontier, absent base example, smoothing/centering/uniqueness obstacles, allocation and first-return decision criteria.", "target_id": "entry-97b5b6c76e914106"}, {"not_covered": "Full mathematical proof and independent verification/publication of the proposed claims; no facts have yet been published.", "revision": 1, "scope": "Dimension-extension alternative's conditional input, all-oblique-section obstacle, direct-all-n competitor, allocation and parking condition.", "target_id": "entry-a241367a37a8ed00"}, {"not_covered": "Full mathematical proof and independent verification/publication of the proposed claims; no facts have yet been published.", "revision": 1, "scope": "Alignment with every clause of the original problem and separation of speculative architecture from verified knowledge.", "target_id": "entry-61d4f0a0a2f9515c"}, {"not_covered": "Full mathematical proof and independent verification/publication of the proposed claims; no facts have yet been published.", "revision": 1, "scope": "Usefulness and precise role of the finite odd harmonic critical-pair bridge in the proposed dependency closure.", "target_id": "entry-97292e71c3b56db4"}], "research_policy_version": 1}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"research_policy_version": 1}, "author": "max4", "created_at": "", "details_path": "", "display_id": "P003", "entry_id": "entry-dab4ef0d1e7e4845", "kind": "report", "links": ["83fc8a6190cb2ca3", "entry-f5a5afb3f0a8d294", "entry-97292e71c3b56db4"], "revision": 1, "status": "recorded", "title": "Exact phase pairing supplies the all-dimensional smooth twisted-height potential", "updated_at": ""} -->
<a id="entry-dab4ef0d1e7e4845"></a>
### P003 · Exact phase pairing supplies the all-dimensional smooth twisted-height potential [report: recorded; NOT A FACT]

<!-- hyprover:body -->
Assignment R002's smooth-potential construction is complete and independently published as fact 83fc8a6190cb2ca3 (accepted=true, published=true; no predecessor facts). For every n>=3 it gives F_n(x)=x_1 cos(alpha_n(x_3))+x_2 sin(alpha_n(x_3)), with alpha_n smooth and even, zero near 0 and 2*pi near |t|=1, satisfying integral_{S^{n-1}} x F_n(x) d sigma=0. Its only critical points are +/-e_1, and the spherical Hessians there are exactly -g and +g.

The exact cancellation avoids an approximation/implicit-function step. Put w(t)=(1-t^2)^((n-1)/2), A=integral_0^1 w, and B(t)=A^{-1} integral_0^t w for 0<=t<=1. Choose a smooth real phase beta on [0,1] with beta(s+1/2)=beta(s)+pi and beta constant near 0,1/2,1; construct it by duplicating a smooth 0-to-pi transition on the two half intervals. Then alpha_n(t)=beta(B(|t|)), extended constantly outside [-1,1], is smooth because of its endpoint plateaus, and integral_{-1}^1 w(t) exp(i alpha_n(t))dt=2A integral_0^1 exp(i beta(s))ds=0. The angle-dependent planar rotation is an explicit odd sphere diffeomorphism, so F_n is a coordinate height composed with that diffeomorphism. Sphere slicing gives the required moment weight.

Usable handoff to max8: the published F_n is smooth, odd, has no degree-one component, and has a globally unique antipodal critical pair with definite Hessians. It is exactly the ordinary coordinate height on neighborhoods of both critical points. Thus the assigned approximation and stability interface now has its required input in every dimension n>=3. Transform inversion and transfer to convex bodies remain separate project obligations; this fact alone does not close the full theorem.
<!-- hyprover:end-body -->

**Related:** [F001 · Smooth odd twisted heights with zero first moment and exactly two nondegenerate critical points](#fact-83fc8a6190cb2ca3), [R002 · Use the published full-target fact as the completion root](#entry-f5a5afb3f0a8d294), [C001 · Finite odd harmonic potential with one nondegenerate antipodal critical pair](#entry-97292e71c3b56db4)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-dab4ef0d1e7e4845`
**Context:** {"research_policy_version": 1}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"research_policy_version": 1}, "author": "max6", "created_at": "", "details_path": "", "display_id": "P004", "entry_id": "entry-fac08b0417afe30e", "kind": "report", "links": ["entry-a241367a37a8ed00", "947986eae4461bec", "e738ef25f0351e92", "83fc8a6190cb2ca3", "entry-97b5b6c76e914106", "81a5afe69f10c3d2", "entry-94ccd77b205dba23"], "revision": 3, "status": "recorded", "title": "R004 handoff complete: published conditional lifting, product obstruction, and shear-proof audit", "updated_at": ""} -->
<a id="entry-fac08b0417afe30e"></a>
### P004 · R004 handoff complete: published conditional lifting, product obstruction, and shear-proof audit [report: recorded; NOT A FACT]

<!-- hyprover:body -->
The independent service accepted and published max6's lifting theorem as F004, fact 947986eae4461bec (accepted=true, published=true, verdict=correct; no predecessors or external references). Its exact input is a smooth odd zero-first-moment F on S^{d-1}, together with a jointly smooth odd homotopy U_s from F to omega_1 such that every U_s has exactly two nondegenerate critical points with values +1 and -1. Its output is a smooth odd zero-moment G on S^d, even in the added coordinate, equal to F on the equator, with only the inherited critical pair and explicit Hessian Hess F direct-sum (-F). G retains the same homotopy property, so the theorem can be iterated through every higher dimension.

The construction is the distinct suspension r U_{beta(r)}(omega), made linear near both poles, followed by an exact moment correction chi(t) a dot x in a fixed middle-latitude band where the gradient is bounded away from zero. Constancy of the angular critical values makes their parameter derivatives vanish, excluding all off-equator critical points even during the cap transition. The moment correction vanishes near the two critical points and is small in C1 only on a region where no critical point exists; it does not invoke a general C1 Morse-stability claim.

The input hypothesis is stronger than merely having two critical points. A generic unique-section body or generic two-critical-point odd function has not been proved to meet it. For applying the theorem to F001, an explicit adapter is U_s=x1*cos((1-s)*alpha_d(x3))+x2*sin((1-s)*alpha_d(x3)): it is height composed with the odd diffeomorphism rotating the first coordinate plane by the displayed angle, whose inverse uses the opposite angle since x3 is unchanged. This checks fixed values and nondegeneracy directly in any downstream use; F001 supplies exact zero first moment at s=0. Nevertheless F001 already covers every dimension, so no such extra composition is needed for R002.

The separate geometric boundary is now independently accepted and published as F012, fact e738ef25f0351e92, with no predecessors or external references. Centered cylinders have an open family of matching graph sections. Even concave homothetic suspensions have a matching horizontal section and inherit a vertical one from every matching base section, giving at least q+1 sections when the base has q. Invertible affine changes preserve these counts. The first verification requested boundedness in the auxiliary centroid-covariance assertion; that assumption was added, covering all compact applications, and the repaired proof passed. The target proposition was never refuted by that repair.

R004 revision 2 has parked further lifting because F001 already removes the dimension-coverage obstacle. Its redirected scrutiny of R003 is complete: max6 read the full published F011 shear proof and recorded the version-bound no-issue audit P020 (entry-94ccd77b205dba23), including all oblique and vertical normals, uniform C2 control, exact centering and global strict convexity. The reusable lifting interface and its explicit extra hypothesis remain available if a later lower-dimensional base makes them useful. No further lifting or duplicate construction is being pursued without a new substantive assignment.
<!-- hyprover:end-body -->

**Related:** [R004 · Retain conditional lifting and the published boundary for naive products](#entry-a241367a37a8ed00), [F004 · Dimension lifting of zero-moment odd potentials by suspension and local moment correction](#fact-947986eae4461bec), [F012 · Cylinders have open families of centroid-matching sections; symmetric homothetic suspensions retain extra sections](#fact-e738ef25f0351e92), [F001 · Smooth odd twisted heights with zero first moment and exactly two nondegenerate critical points](#fact-83fc8a6190cb2ca3), [R003 · The direct shear alternative is published and retained as a fallback](#entry-97b5b6c76e914106), [F011 · Polynomial shears of the ball give unique centroid-matching sections in every dimension n&gt;=3](#fact-81a5afe69f10c3d2), [P020 · Independent scrutiny of the published shear construction found no unresolved issue in the checked scope](#entry-94ccd77b205dba23)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-fac08b0417afe30e`
**Context:** {"research_policy_version": 1}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"research_policy_version": 1}, "author": "max5", "created_at": "", "details_path": "knowledge/notes/entry-04743c91041fed68-1.md", "display_id": "P005", "entry_id": "entry-04743c91041fed68", "kind": "report", "links": ["entry-97b5b6c76e914106", "entry-a241367a37a8ed00", "entry-f5a5afb3f0a8d294"], "revision": 1, "status": "recorded", "title": "Direct disk-fiber shear gives a candidate replacement for spherical harmonic inversion", "updated_at": ""} -->
<a id="entry-04743c91041fed68"></a>
### P005 · Direct disk-fiber shear gives a candidate replacement for spherical harmonic inversion [report: recorded; NOT A FACT]

<!-- hyprover:body -->
For R003, I am developing the explicit all-dimensional family K_epsilon={(w,z) in R^(n-1) x R: |w-epsilon*v(z)|^2+z^2<=1}, with an even polynomial v supported in the first two horizontal coordinates. This is a nonlinear shear of the ball, with exact volume centroid epsilon times the weighted mean of v under (1-z^2)^((n-1)/2) dz; the last centroid coordinate is zero. Small epsilon should preserve smooth strict convexity. A textual derivation gives first variation of the signed halfspace-volume potential at u=(a,t), s=|a|: D(u)=2*kappa_(n-2)*a dot integral_{-1}^1 v(sq)*(1-q^2)^((n-2)/2)dq. Thus v(z)=sum c_k z^(2k) gives D(u)=a dot sum [2*kappa_(n-2)*b_k*c_k]*(1-t^2)^k, where b_k=integral q^(2k)(1-q^2)^((n-2)/2)dq>0. This is an elementary invertible coefficient map onto potentials u1*A(t^2)+u2*B(t^2). The no-degree-one condition on that potential appears exactly equivalent to the weighted mean-zero condition on v. I will submit the complete conditional shear-realization lemma after checking the centering identity, uniform C2 remainder, and global critical-point stability. This is not yet a verified fact.
<!-- hyprover:end-body -->

**Related:** [R003 · The direct shear alternative is published and retained as a fallback](#entry-97b5b6c76e914106), [R004 · Retain conditional lifting and the published boundary for naive products](#entry-a241367a37a8ed00), [R002 · Use the published full-target fact as the completion root](#entry-f5a5afb3f0a8d294)
**Details:** [详细说明 / Details — P005](knowledge/notes/entry-04743c91041fed68-1.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-04743c91041fed68`
**Context:** {"research_policy_version": 1}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"research_policy_version": 1}, "author": "max3", "created_at": "", "details_path": "", "display_id": "P006", "entry_id": "entry-bfe253a084d94ef6", "kind": "report", "links": ["e9b3873b7c826c7b", "83fc8a6190cb2ca3", "entry-f5a5afb3f0a8d294", "entry-97292e71c3b56db4"], "revision": 1, "status": "recorded", "title": "Finite-harmonic sign-transform inversion is published and ready for the analytic construction", "updated_at": ""} -->
<a id="entry-bfe253a084d94ef6"></a>
### P006 · Finite-harmonic sign-transform inversion is published and ready for the analytic construction [report: recorded; NOT A FACT]

<!-- hyprover:body -->
Max3's assigned interface is complete: F002, fact e9b3873b7c826c7b, was independently accepted and published with no predecessors and no external references. For every n>=3 and j>=0, the hemispherical sign transform H acts on degree 2j+1 by the nonzero real scalar
lambda_{n,2j+1}=2|S^{n-2}|(-1)^j(2j)!/[2^j j! product_{r=0}^j(n-1+2r)].
Even-degree harmonics are annihilated.

Consequently every real odd finite harmonic sum F with zero degree-one component has the unique real odd finite harmonic preimage g=sum lambda_{n,2j+1}^{-1}F_{2j+1}, also with zero degree one. The published statement includes smoothness and the exact vector identity integral theta*g(theta)d sigma(theta)=0. The full proof uses polynomial coefficient recurrence and two applications of the divergence theorem; it does not invoke Funk-Hecke or a special-function assertion.

Handoff to max8/max7: once the finite approximation F is available, F002 supplies g with exactly Hg=F and exact first-moment cancellation, in every required dimension. Max4's published F001 (83fc8a6190cb2ca3) supplies the smooth two-critical-point input, but it must still pass through the finite approximation interface before F002 is applied; F002 alone does not invert arbitrary smooth data. The finite approximation and radial-body realization are already assigned separately, so no duplicate branch is being started. The full theorem is not claimed complete. Max3 is ready for a focused follow-up or a dependency check.
<!-- hyprover:end-body -->

**Related:** [F002 · Nonzero odd eigenvalues and finite harmonic inversion of the hemispherical sign transform](#fact-e9b3873b7c826c7b), [F001 · Smooth odd twisted heights with zero first moment and exactly two nondegenerate critical points](#fact-83fc8a6190cb2ca3), [R002 · Use the published full-target fact as the completion root](#entry-f5a5afb3f0a8d294), [C001 · Finite odd harmonic potential with one nondegenerate antipodal critical pair](#entry-97292e71c3b56db4)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-bfe253a084d94ef6`
**Context:** {"research_policy_version": 1}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"research_policy_version": 1}, "author": "max2", "created_at": "", "details_path": "", "display_id": "P007", "entry_id": "entry-c8473e86b97a5e1b", "kind": "report", "links": ["3ce5f83832ddd38c", "entry-10acec896a9c45b2"], "revision": 1, "status": "recorded", "title": "Planar branch closed: published proof of mu(2)=3", "updated_at": ""} -->
<a id="entry-c8473e86b97a5e1b"></a>
### P007 · Planar branch closed: published proof of mu(2)=3 [report: recorded; NOT A FACT]

<!-- hyprover:body -->
Worker max2 completed the full planar assignment in R001. Fact 3ce5f83832ddd38c (Sharp planar centroid-section theorem: every body has at least three, triangles exactly three) received accepted=true, published=true, verdict=correct, with no predecessors and no external references. Its statement covers every compact planar convex body with nonempty interior, without boundary regularity, and proves that every nondegenerate triangle has exactly three centroid-matching lines, through the area centroid and parallel to the sides. Therefore mu(2)=3. An explicit sharp example is Delta={(s,t): s>=0,t>=0,s+t<=1}, with centroid (1/3,1/3) and precisely the three matching lines s=1/3, t=1/3, and s+t=2/3.

The self-contained proof establishes centroid interiority and radial continuity. For f(theta)=rho(theta)^3-rho(theta+pi)^3, centering implies zero first sine/cosine moments. If f had at most two zero classes modulo pi, its signs would switch at exactly one of them, so multiplication by a suitable first harmonic would be nonnegative and positive on an interval, contradicting moment vanishing. This explicitly handles tangential zeros; zero intervals and identically zero functions already give infinitely many lines. Triangle sharpness follows by classifying opposite boundary points via barycentric coordinates: the coordinates must be a permutation of (0,1/3,2/3).

The planar assignment has no remaining gap. The higher-dimensional universal bound and smooth strictly convex unique-section construction are separate obligations still governed by the other assignments. The planar terminal fact is available for the main agent's eventual fresh completion revalidation. All reasoning and verification were text-only; no external literature was used.
<!-- hyprover:end-body -->

**Related:** [F003 · Sharp planar centroid-section theorem: every body has at least three, triangles exactly three](#fact-3ce5f83832ddd38c), [R001 · Universal and planar components are published and ready for completion revalidation](#entry-10acec896a9c45b2)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-c8473e86b97a5e1b`
**Context:** {"research_policy_version": 1}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"research_policy_version": 1}, "author": "max3", "created_at": "", "details_path": "", "display_id": "P008", "entry_id": "entry-a181734267aec9d9", "kind": "report", "links": ["e9b3873b7c826c7b", "entry-7d0fc2492a0547d7", "entry-c0289974e2e74308", "entry-f5a5afb3f0a8d294"], "revision": 1, "status": "recorded", "title": "Publication repair for V004 and V005: reuse the existing H glossary entry", "updated_at": ""} -->
<a id="entry-a181734267aec9d9"></a>
### P008 · Publication repair for V004 and V005: reuse the existing H glossary entry [report: recorded; NOT A FACT]

<!-- hyprover:body -->
Max3 resumed after the new service feedback and read its complete public attachments. V004 (entry-7d0fc2492a0547d7, run [run identifier omitted]) and V005 (entry-c0289974e2e74308, run [run identifier omitted]) both report verdict=correct, empty critical_errors/gaps, but accepted=false and published=false at stage=publication with ValueError: glossary_conflict: H. These are publication failures, not mathematical rejections, and neither candidate is yet an available premise.

The canonical H glossary entry already published with F002 (e9b3873b7c826c7b) is exactly: "Hemispherical sign transform (Hf)(u)=integral_{S^{n-1}} sign(u dot theta)f(theta)d sigma(theta), with sign(0)=0." Suggested repair for the submitting workers: retain the complete definitions in each statement and proof, but omit H from glossary_introduces because it already exists; introduce only genuinely new symbols, then resubmit and inspect the new publication receipt. If retaining H in that metadata is necessary, its existing definition must be reused verbatim. No proof change or new mathematical dependency on F002 is needed solely to reuse a glossary symbol; declare F002 as predecessor only if its mathematical conclusion is used.

F002 remains the completed assigned inversion interface, with no predecessors and no unresolved verification issue. Max8 still owns finite harmonic approximation, max7 radial realization, and max the universal section interface; no duplicate proof work has been started by max3.
<!-- hyprover:end-body -->

**Related:** [F002 · Nonzero odd eigenvalues and finite harmonic inversion of the hemispherical sign transform](#fact-e9b3873b7c826c7b), [V004 · Verification submission outcome](#entry-7d0fc2492a0547d7), [V005 · Verification submission outcome](#entry-c0289974e2e74308), [R002 · Use the published full-target fact as the completion root](#entry-f5a5afb3f0a8d294)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-a181734267aec9d9`
**Context:** {"research_policy_version": 1}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"research_policy_version": 1}, "author": "max7", "created_at": "", "details_path": "", "display_id": "P009", "entry_id": "entry-a97772fe5de56e5b", "kind": "report", "links": ["846e296f22ef7853", "7a3fd2eb4772e4b4", "3ce5f83832ddd38c", "a8378dfc6c55d238", "8f8adadaf10ea26e", "83fc8a6190cb2ca3", "a0c54399c2d44b8d", "e9b3873b7c826c7b", "dc5ba3bf489a3dea", "entry-c72215b98f9c410b", "entry-89df6118370546f6", "entry-f5a5afb3f0a8d294", "entry-10acec896a9c45b2"], "revision": 5, "status": "recorded", "title": "Full theorem published as F013; exact dependency handoff for completion revalidation", "updated_at": ""} -->
<a id="entry-a97772fe5de56e5b"></a>
### P009 · Full theorem published as F013; exact dependency handoff for completion revalidation [report: recorded; NOT A FACT]

<!-- hyprover:body -->
Max7's assigned radial bridge and all-dimensional assembly are complete: F005 is the conditional realization, and F009 (7a3fd2eb4772e4b4) is the unconditional smooth strictly convex existence theorem for every n>=3. F009 received accepted=true, published=true, verdict=correct with no errors or gaps (run [run identifier omitted]).

The final implication is now also published as F013, fact 846e296f22ef7853, revision 1, proof knowledge/proofs/846e296f22ef7853.md. Its complete receipt has accepted=true, published=true, verdict=correct, no errors or gaps, run [run identifier omitted], feedback entry-c72215b98f9c410b. Its statement retains the full fixed target: for every n>=3 a C-infinity strictly convex body has exactly one centroid-matching geometric affine hyperplane; mu(2)=3 and mu(n)=1 for all n>=3; universal quantifiers include all nonsmooth convex bodies and positive-measure sections are counted without orientation.

The exact terminal dependency graph is F013 (846e296f22ef7853) <- F003 (3ce5f83832ddd38c), F008 (a8378dfc6c55d238), F009 (7a3fd2eb4772e4b4). F008 <- F006 (8f8adadaf10ea26e). F009 <- F001 (83fc8a6190cb2ca3), F007 (a0c54399c2d44b8d), F002 (e9b3873b7c826c7b), F005 (dc5ba3bf489a3dea). F001, F002, F003, F005, F006 and F007 have no project predecessors. All these facts were read at revision 1 and the service rechecked dependencies at publication. They use no external references.

The F013 proof applies F008 at p=c(K) for the universal lower bound, uses F009's examples to exclude any universal q>=2 in n>=3, and uses F003's lower bound and exact triangle count to establish both planar inequalities. Its ordinary verification treats the three direct predecessor statements as trusted premises. It is therefore a published full-target implication, not the separate fresh completion revalidation required by RUN_POLICY.md. Main should use the graph above for that remaining run-level check and retain project completion authority. I have no unsubmitted mathematical step left in the assigned construction or terminal implication; private recall and both new receipts are saved.
<!-- hyprover:end-body -->

**Related:** [F013 · Full centroid-section theorem: smooth sharp examples and mu(2)=3, mu(n)=1 for n&gt;=3](#fact-846e296f22ef7853), [F009 · Smooth strictly convex bodies with exactly one centroid-matching hyperplane in every dimension n&gt;=3](#fact-7a3fd2eb4772e4b4), [F003 · Sharp planar centroid-section theorem: every body has at least three, triangles exactly three](#fact-3ce5f83832ddd38c), [F008 · Every interior point of a convex body is the centroid of some hyperplane section](#fact-a8378dfc6c55d238), [F006 · Signed halfspace volume is C1 and its spherical gradient is twice the central-section first moment](#fact-8f8adadaf10ea26e), [F001 · Smooth odd twisted heights with zero first moment and exactly two nondegenerate critical points](#fact-83fc8a6190cb2ca3), [F007 · Finite odd harmonic approximation with exact moment cancellation and two-critical-point stability](#fact-a0c54399c2d44b8d), [F002 · Nonzero odd eigenvalues and finite harmonic inversion of the hemispherical sign transform](#fact-e9b3873b7c826c7b), [F005 · Centered radial perturbations realize one centroidal hyperplane from a two-critical-point sign transform](#fact-dc5ba3bf489a3dea), [V016 · Verification submission outcome](#entry-c72215b98f9c410b), [V012 · Verification submission outcome](#entry-89df6118370546f6), [R002 · Use the published full-target fact as the completion root](#entry-f5a5afb3f0a8d294), [R001 · Universal and planar components are published and ready for completion revalidation](#entry-10acec896a9c45b2)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-a97772fe5de56e5b`
**Context:** {"research_policy_version": 1}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"research_policy_version": 1}, "author": "max5", "created_at": "", "details_path": "knowledge/notes/entry-67a9664998c4132f-1.md", "display_id": "P010", "entry_id": "entry-67a9664998c4132f", "kind": "report", "links": ["entry-04743c91041fed68", "83fc8a6190cb2ca3", "entry-97b5b6c76e914106", "entry-a241367a37a8ed00"], "revision": 1, "status": "recorded", "title": "Direct shear route now has a complete all-dimensional proof plan using F001", "updated_at": ""} -->
<a id="entry-67a9664998c4132f"></a>
### P010 · Direct shear route now has a complete all-dimensional proof plan using F001 [report: recorded; NOT A FACT]

<!-- hyprover:body -->
The new published F001 (83fc8a6190cb2ca3) supplies the only non-elementary input needed by the direct shear family reported in P005. Its coefficients cos(alpha_n(t)) and sin(alpha_n(t)) are even and constant near t=0, so they are smooth functions of t^2. Univariate polynomial approximation in C2, followed by subtraction of each weighted mean, retains the special potential form u1*A(u_n^2)+u2*B(u_n^2) and exact moment cancellation. The two nondegenerate critical points persist under this C2 approximation. The shear coefficient map then realizes this polynomial potential, with exact body centering and small-parameter convexity. I am preparing one complete submission proving this direct conditional realization and the resulting all-n existence, with F001 as the sole project predecessor. Max8 need not modify its separate finite-harmonic assignment for this branch.
<!-- hyprover:end-body -->

**Related:** [P005 · Direct disk-fiber shear gives a candidate replacement for spherical harmonic inversion](#entry-04743c91041fed68), [F001 · Smooth odd twisted heights with zero first moment and exactly two nondegenerate critical points](#fact-83fc8a6190cb2ca3), [R003 · The direct shear alternative is published and retained as a fallback](#entry-97b5b6c76e914106), [R004 · Retain conditional lifting and the published boundary for naive products](#entry-a241367a37a8ed00)
**Details:** [详细说明 / Details — P010](knowledge/notes/entry-67a9664998c4132f-1.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-67a9664998c4132f`
**Context:** {"research_policy_version": 1}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"research_policy_version": 1}, "author": "max8", "created_at": "", "details_path": "", "display_id": "P011", "entry_id": "entry-1e38b3add2d300cf", "kind": "report", "links": ["a0c54399c2d44b8d", "83fc8a6190cb2ca3", "e9b3873b7c826c7b", "entry-f5a5afb3f0a8d294", "entry-97292e71c3b56db4", "entry-f855f425e4b3410b"], "revision": 2, "status": "recorded", "title": "Finite harmonic bridge published as F007; C2 and moment interfaces are fully matched", "updated_at": ""} -->
<a id="entry-1e38b3add2d300cf"></a>
### P011 · Finite harmonic bridge published as F007; C2 and moment interfaces are fully matched [report: recorded; NOT A FACT]

<!-- hyprover:body -->
Max8's assigned R002 reduction is complete. F007, fact a0c54399c2d44b8d, received accepted=true, published=true, verdict=correct, with no critical errors or gaps, no predecessors, and no external references (run [run identifier omitted]). It proves that every real C2 odd F on S^{n-1}, n>=3, with integral x*F=0 is arbitrarily C2-approximable by real finite sums of odd spherical harmonics of degrees at least 3. If F has exactly two nondegenerate critical points, sufficiently close approximants retain exactly one nondegenerate antipodal critical pair. The hypothesis is only C2, so smooth inputs are covered.

The norm is sup|u|+sup|grad_S u|+sup||Hess_S u||op. The proof is self-contained: normalized polynomial kernels ((1+x dot y)/2)^m concentrate uniformly; rotation derivatives give intrinsic C2 convergence; an explicit Laplacian recurrence decomposes each finite polynomial into homogeneous harmonics; and subtracting the linear projection preserves convergence with ||Pi_1 u||C2<=3n||u||infinity. Exact moment cancellation is retained.

For the stability handoff, choose disjoint coordinate balls of radius r around the maximum/minimum, with original coordinate Hessians bounded by -2 lambda I and +2 lambda I, a constant C controlling coordinate Hessians by the intrinsic C2 norm, and gamma>0 equal to the minimum gradient norm on the compact complement. The proof's sufficient perturbation threshold is min(lambda/C,lambda*r^2/4,gamma/2). It establishes interior existence by a boundary-value gap, local uniqueness and nondegeneracy by definite Hessians, and global exclusion by gamma. F007 also includes a verified C1 counterexample: smooth odd functions converge to height x_n in C1 yet have at least six critical points.

Published F001 (83fc8a6190cb2ca3) matches all input hypotheses, and the output of F007 matches the finite, odd, degree-one-free input of F002 (e9b3873b7c826c7b). Thus these handoff conditions have no remaining regularity or moment mismatch. Max7 has claimed responsibility for the all-dimensional body/existence assembly; max8 is not duplicating it. The full project theorem and its final dependency revalidation are not declared complete by this report. All work was text-only and used no external literature.
<!-- hyprover:end-body -->

**Related:** [F007 · Finite odd harmonic approximation with exact moment cancellation and two-critical-point stability](#fact-a0c54399c2d44b8d), [F001 · Smooth odd twisted heights with zero first moment and exactly two nondegenerate critical points](#fact-83fc8a6190cb2ca3), [F002 · Nonzero odd eigenvalues and finite harmonic inversion of the hemispherical sign transform](#fact-e9b3873b7c826c7b), [R002 · Use the published full-target fact as the completion root](#entry-f5a5afb3f0a8d294), [C001 · Finite odd harmonic potential with one nondegenerate antipodal critical pair](#entry-97292e71c3b56db4), [V009 · Verification submission outcome](#entry-f855f425e4b3410b)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-1e38b3add2d300cf`
**Context:** {"research_policy_version": 1}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"audits": [{"not_covered": "Fresh re-verification of the eigenvalue proof or inversion on arbitrary smooth data.", "revision": 1, "scope": "Complete published statement and glossary: domain n\u003e=3, real finite odd inverse, degree-one exclusion, exact moment cancellation, and Euclidean-measure H normalization matched against F005.", "target_id": "e9b3873b7c826c7b"}, {"not_covered": "Fresh independent completion revalidation, separate re-verification of all convexity and critical-point stability details, and construction of the finite harmonic potential.", "revision": 1, "scope": "Complete published statement and successful publication receipt; F002-to-F005 input compatibility; proof steps 1, 3 and 5 for the transform normalization, C2 scaling and conversion from all affine sections to one antipodal normal pair.", "target_id": "dc5ba3bf489a3dea"}], "research_policy_version": 1}, "author": "max3", "created_at": "", "details_path": "", "display_id": "P012", "entry_id": "entry-edb4977d81b250ba", "kind": "report", "links": ["e9b3873b7c826c7b", "dc5ba3bf489a3dea", "entry-f5a5afb3f0a8d294", "entry-1e38b3add2d300cf"], "revision": 1, "status": "recorded", "title": "Published inversion and radial realization have matching hypotheses and normalizations", "updated_at": ""} -->
<a id="entry-edb4977d81b250ba"></a>
### P012 · Published inversion and radial realization have matching hypotheses and normalizations [report: recorded; NOT A FACT]

<!-- hyprover:body -->
Max3 checked the exact interface between F002 (e9b3873b7c826c7b, revision 1) and newly published F005 (dc5ba3bf489a3dea, revision 1). V007 records accepted=true and published=true for F005. Both facts use ordinary Euclidean sphere surface measure and exactly Hf(u)=integral sign(u dot theta)f(theta)d sigma(theta), with sign(0)=0; there is no missing probability-measure or factor-two conversion.

Given a real odd finite harmonic potential F with zero degree one and exactly two nondegenerate critical points, F002 supplies a real smooth odd finite harmonic g with Hg=F and the exact vector identity integral theta*g=0. These are precisely all the g hypotheses in F005, in the same range n>=3. The radial formula rho_epsilon^(n+1)=1+epsilon*g enforces exact centering, while F005 uses B_epsilon=(1/n)H[rho_epsilon^n] and ((n+1)/epsilon)B_epsilon -> Hg in C2. Its section identity grad B_epsilon=2 integral_section x has the same normalization. Large inverse eigenvalues in F002 do not require a new uniform estimate: the statements only need one fixed finite g, and F005 permits its small-epsilon threshold to depend on g.

This is a no-issue statement/interface and normalization audit, not a new verified premise or fresh completion revalidation. I also read the corresponding normalization, Taylor-limit and affine-section arguments in the F005 proof. The outstanding finite F input remains max8's approximation assignment as described in P011; max7 already owns the all-dimensional existence assembly. F001 is smooth and cannot be fed directly into the finite-sum conclusion of F002 without that bridge. No duplicate construction or arbitrary-smooth inversion project has been started.
<!-- hyprover:end-body -->

**Related:** [F002 · Nonzero odd eigenvalues and finite harmonic inversion of the hemispherical sign transform](#fact-e9b3873b7c826c7b), [F005 · Centered radial perturbations realize one centroidal hyperplane from a two-critical-point sign transform](#fact-dc5ba3bf489a3dea), [R002 · Use the published full-target fact as the completion root](#entry-f5a5afb3f0a8d294), [P011 · Finite harmonic bridge published as F007; C2 and moment interfaces are fully matched](#entry-1e38b3add2d300cf)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-edb4977d81b250ba`
**Context:** {"audits": [{"not_covered": "Fresh re-verification of the eigenvalue proof or inversion on arbitrary smooth data.", "revision": 1, "scope": "Complete published statement and glossary: domain n\u003e=3, real finite odd inverse, degree-one exclusion, exact moment cancellation, and Euclidean-measure H normalization matched against F005.", "target_id": "e9b3873b7c826c7b"}, {"not_covered": "Fresh independent completion revalidation, separate re-verification of all convexity and critical-point stability details, and construction of the finite harmonic potential.", "revision": 1, "scope": "Complete published statement and successful publication receipt; F002-to-F005 input compatibility; proof steps 1, 3 and 5 for the transform normalization, C2 scaling and conversion from all affine sections to one antipodal normal pair.", "target_id": "dc5ba3bf489a3dea"}], "research_policy_version": 1}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"audits": [{"not_covered": "Fresh independent service revalidation or completion certification.", "revision": 1, "scope": "Complete published statement and proof: smooth phase pairing, endpoint smoothness, odd diffeomorphism, exact first moment, unique nondegenerate critical pair; all F007(1)-(2) input hypotheses.", "target_id": "83fc8a6190cb2ca3"}, {"not_covered": "Part (3)'s C1 counterexample; later transform inversion and convex-body proofs; fresh independent completion revalidation.", "revision": 1, "scope": "Complete published statement and successful receipt; proof parts 1-4 for intrinsic C2 polynomial approximation, finite harmonic decomposition, exact degree-one removal, and local/global critical-point stability; exact match with F001.", "target_id": "a0c54399c2d44b8d"}], "research_policy_version": 1}, "author": "max4", "created_at": "", "details_path": "", "display_id": "P013", "entry_id": "entry-0488e67f78918534", "kind": "report", "links": ["83fc8a6190cb2ca3", "a0c54399c2d44b8d", "entry-f855f425e4b3410b", "entry-f5a5afb3f0a8d294"], "revision": 1, "status": "recorded", "title": "Twisted-height input matches the published finite approximation theorem without extra hypotheses", "updated_at": ""} -->
<a id="entry-0488e67f78918534"></a>
### P013 · Twisted-height input matches the published finite approximation theorem without extra hypotheses [report: recorded; NOT A FACT]

<!-- hyprover:body -->
Max4 checked the newly published F007 (a0c54399c2d44b8d, revision 1) against the assigned construction F001 (83fc8a6190cb2ca3, revision 1), including the complete F001 proof and the approximation, projection and stability arguments in F007. Both facts were still active at notebook snapshot 39. F007's receipt explicitly has accepted=true, published=true, fact_id=a0c54399c2d44b8d and no reported mathematical gaps. This is a no-issue manual interface audit, not a new fact or fresh independent completion revalidation.

For each fixed integer n>=3, F001 supplies a real smooth, hence C2, odd function on exactly the round sphere and with exactly the Euclidean surface measure used by F007. Its vector first moment is zero, and its only critical points are +/-e1 with Hessians -g and +g. Thus every hypothesis of F007(1)-(2) is supplied by the published F001 statement. The endpoint plateaus in F001's phase construction also justify smoothness at x3=0 and the two x3 poles; no singular behavior enters the approximation input.

The F007 proof controls function, gradient and Hessian errors through the commuting rotation fields, so the first-moment correction remains small in the required C2 topology: for the odd polynomial Q, subtract Pi_1 Q and use ||Pi_1(Q-F_n)||_C2<=3n||Q-F_n||_infinity. The finite harmonic decomposition and the divergence identity remove precisely degree one, leaving real odd degrees at least three with exact zero moment. Its stability proof separately supplies an interior extremum and local uniqueness from definite coordinate Hessians in each critical neighborhood, and excludes further critical points by the positive gradient minimum on their compact complement. There is no unproved C1-to-C2 stability step.

Consequently the already published F007 can be applied to each F_n with approximation error below its positive stability tolerance. The resulting finite potential need only have one nondegenerate antipodal critical pair; its points need not remain exactly +/-e1, and the later inversion and radial realization statements do not require that location. No uniform approximation degree or tolerance across all n is required by the fixed theorem. This completes the checked smooth-to-finite input interface for max7's existing assembly assignment. P003 remains the original construction handoff; max4 has not duplicated the all-dimensional body assembly.
<!-- hyprover:end-body -->

**Related:** [F001 · Smooth odd twisted heights with zero first moment and exactly two nondegenerate critical points](#fact-83fc8a6190cb2ca3), [F007 · Finite odd harmonic approximation with exact moment cancellation and two-critical-point stability](#fact-a0c54399c2d44b8d), [V009 · Verification submission outcome](#entry-f855f425e4b3410b), [R002 · Use the published full-target fact as the completion root](#entry-f5a5afb3f0a8d294)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-0488e67f78918534`
**Context:** {"audits": [{"not_covered": "Fresh independent service revalidation or completion certification.", "revision": 1, "scope": "Complete published statement and proof: smooth phase pairing, endpoint smoothness, odd diffeomorphism, exact first moment, unique nondegenerate critical pair; all F007(1)-(2) input hypotheses.", "target_id": "83fc8a6190cb2ca3"}, {"not_covered": "Part (3)'s C1 counterexample; later transform inversion and convex-body proofs; fresh independent completion revalidation.", "revision": 1, "scope": "Complete published statement and successful receipt; proof parts 1-4 for intrinsic C2 polynomial approximation, finite harmonic decomposition, exact degree-one removal, and local/global critical-point stability; exact match with F001.", "target_id": "a0c54399c2d44b8d"}], "research_policy_version": 1}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"research_policy_version": 1}, "author": "max", "created_at": "", "details_path": "", "display_id": "P014", "entry_id": "entry-b92276d77e44cac5", "kind": "report", "links": ["8f8adadaf10ea26e", "a8378dfc6c55d238", "3ce5f83832ddd38c", "entry-10acec896a9c45b2", "entry-f5a5afb3f0a8d294"], "revision": 1, "status": "recorded", "title": "Universal lower bound and nonsmooth signed-volume interface are published", "updated_at": ""} -->
<a id="entry-b92276d77e44cac5"></a>
### P014 · Universal lower bound and nonsmooth signed-volume interface are published [report: recorded; NOT A FACT]

<!-- hyprover:body -->
Max's R001 assignment is complete. F006 (8f8adadaf10ea26e) and F008 (a8378dfc6c55d238) both received accepted=true, published=true, verdict=correct, with no mathematical errors or gaps. F006 has no predecessors; F008 depends only on F006. Neither proof uses external references or mathematical computation.

F006 gives the exact common normalization for every n>=2: Hf(u)=integral sign(u dot theta)f(theta)d sigma, grad_S Hf(u)=2 integral_{theta perp u} theta*f(theta)d sigma_{n-2}; B_rho=(1/n)H(rho^n); M_rho=(1/n) integral_{theta perp u} theta*rho(theta)^n d sigma_{n-2}; therefore grad_S B_rho=2M_rho. In dimension two, the equatorial measure is counting measure on the two points. The proof uses an exact moving-strip boundary and a uniform continuity remainder, so f continuous is enough for Hf to be C1. In fact H:C^k->C^{k+1} is bounded for every integer k>=0. A positive continuous radial function suffices for all section identities and positive section measure; every translated convex body K-p with p interior has such a radial function. There is no boundary smoothness assumption for the identity. If a later argument needs B_rho in C2, rho in C1 is sufficient, and a C1 remainder in the input is controlled in C2 after applying H.

F008 proves the stronger universal statement: for every convex body K in R^n, n>=2, and every p in int(K), some positive-measure hyperplane section has centroid p. A maximum of B_{K,p} gives existence by F006. The fact also proves c(K) is interior using a uniform gap from supporting hyperplanes, proves every matching affine hyperplane contains p, and gives a bijection between antipodal critical pairs of B_{K,p} and the unoriented matching hyperplanes. Thus the required nonsmooth universal lower bound mu(n)>=1 is closed in every dimension.

For final assembly, F008 supplies the universal lower bound and F003 (3ce5f83832ddd38c) supplies the sharp planar value. Max7's existing all-dimensional construction assembly and main's fresh completion revalidation are separate obligations; this handoff does not claim project-wide completion. The remaining construction and audit work is already allocated in the public reports, so I am not duplicating it. The initial glossary-only publication conflict for F006 was repaired by omitting the already existing H glossary entry; the published receipt, rather than the earlier correct-but-unpublished verdict, is the usable result.
<!-- hyprover:end-body -->

**Related:** [F006 · Signed halfspace volume is C1 and its spherical gradient is twice the central-section first moment](#fact-8f8adadaf10ea26e), [F008 · Every interior point of a convex body is the centroid of some hyperplane section](#fact-a8378dfc6c55d238), [F003 · Sharp planar centroid-section theorem: every body has at least three, triangles exactly three](#fact-3ce5f83832ddd38c), [R001 · Universal and planar components are published and ready for completion revalidation](#entry-10acec896a9c45b2), [R002 · Use the published full-target fact as the completion root](#entry-f5a5afb3f0a8d294)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-b92276d77e44cac5`
**Context:** {"research_policy_version": 1}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"audits": [{"not_covered": "Fresh independent service revalidation or completion certification; a separately published universal existence theorem; higher-dimensional sharpness construction.", "revision": 1, "scope": "Complete statement, proof and publication receipt: continuous-input sign-transform differentiability and Ck gain, n=2 measure conventions, gauge radial continuity for nonsmooth convex bodies, polar factors, positive section measure, and exact interface for the application at the volume centroid.", "target_id": "8f8adadaf10ea26e"}], "research_policy_version": 1}, "author": "max2", "created_at": "", "details_path": "knowledge/notes/entry-127efe34f5e48303-1.md", "display_id": "P015", "entry_id": "entry-127efe34f5e48303", "kind": "report", "links": ["8f8adadaf10ea26e", "entry-4800d848113c408d", "entry-10acec896a9c45b2", "3ce5f83832ddd38c"], "revision": 1, "status": "recorded", "title": "Nonsmooth universal interface audit found no issue; exact centroid application is ready", "updated_at": ""} -->
<a id="entry-127efe34f5e48303"></a>
### P015 · Nonsmooth universal interface audit found no issue; exact centroid application is ready [report: recorded; NOT A FACT]

<!-- hyprover:body -->
Max2 completed the independent audit assigned by R001 revision 2. I read F006 (8f8adadaf10ea26e, revision 1) in full, including its complete proof and V008 publication receipt. The receipt has accepted=true, published=true, fact_id=8f8adadaf10ea26e, verdict=correct, and empty errors/gaps. No concrete mathematical defect was found. This is a manual audit and handoff, not a new fact or fresh completion revalidation.

The crucial differentiability argument only assumes continuous f. In the chart w(v)=(u+v)/sqrt(1+|v|^2), the moving sign boundary on each equatorial fiber is exactly b(v dot eta)=-(v dot eta)/sqrt(1+(v dot eta)^2). Its displacement gives the linear term 2 integral_E (v dot eta) f(eta), with the uniform remainder bounded by s_(n-2)||f||infinity |v|^3 + 2s_(n-2)|v| omega(|v|). This proves Frechet differentiability; rotating the fixed equator proves continuity of the resulting gradient. The n=2 counting-measure convention and Jacobian exponent -1/2 are consistent. No unproved differentiation of a merely continuous radial function is used.

For nonsmooth convex K and an interior point p, the gauge proof supplies a positive continuous radial function for K-p. The two polar calculations have matching exponent n and factor 1/n, giving grad B=2M exactly. The positive radial minimum supplies positive finite section measure, so M=0 is equivalent to zero section centroid. Smoothness of the boundary is unnecessary. The proof's higher Ck-to-C(k+1) claim is also justified by local smooth rotations and differentiation over a fixed compact equator.

The remaining universal implication is short and valid in every n>=2: c(K) is interior, as the explicit nearest-point/support argument in the details shows; apply F006 to p=c(K), choose a maximum of its C1 potential on the compact sphere, and use grad B=0. Translate the section back. Every affine section whose centroid is p necessarily contains p, since averaging u dot x=a over a section in {u dot x=a} gives u dot p=a. Opposite normals specify the same geometric hyperplane; if B is identically zero all central sections work. At the checked snapshot 43, the separate universal existence theorem was not yet among the active fact statements. Max retains its submission ownership under R001; this audit neither duplicates that candidate nor treats an unsubmitted implication as a new premise. F003 remains the already published planar terminal result.
<!-- hyprover:end-body -->

**Related:** [F006 · Signed halfspace volume is C1 and its spherical gradient is twice the central-section first moment](#fact-8f8adadaf10ea26e), [V008 · Verification submission outcome](#entry-4800d848113c408d), [R001 · Universal and planar components are published and ready for completion revalidation](#entry-10acec896a9c45b2), [F003 · Sharp planar centroid-section theorem: every body has at least three, triangles exactly three](#fact-3ce5f83832ddd38c)
**Details:** [详细说明 / Details — P015](knowledge/notes/entry-127efe34f5e48303-1.md)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-127efe34f5e48303`
**Context:** {"audits": [{"not_covered": "Fresh independent service revalidation or completion certification; a separately published universal existence theorem; higher-dimensional sharpness construction.", "revision": 1, "scope": "Complete statement, proof and publication receipt: continuous-input sign-transform differentiability and Ck gain, n=2 measure conventions, gauge radial continuity for nonsmooth convex bodies, polar factors, positive section measure, and exact interface for the application at the volume centroid.", "target_id": "8f8adadaf10ea26e"}], "research_policy_version": 1}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"audits": [{"not_covered": "none", "revision": 1, "scope": "Complete published statement and proof, parts 1-5: polynomial-kernel C2 approximation; rotation-frame estimates; harmonic decomposition; exact degree-one projection and moment cancellation; local and global critical-point stability; the C1 counterexample.", "target_id": "a0c54399c2d44b8d"}, {"not_covered": "Independent recheck of the phase construction and moment-cancellation proof.", "revision": 1, "scope": "Complete published statement and its input compatibility with F007: every n\u003e=3, real smooth odd potential, exact zero first moment, and exactly two nondegenerate critical points.", "target_id": "83fc8a6190cb2ca3"}, {"not_covered": "Fresh re-verification of the eigenvalue proof or inversion on arbitrary smooth data.", "revision": 1, "scope": "Complete published statement and its compatibility with the finite output of F007: same Euclidean sphere measure, odd harmonics of degrees at least 3, real nonzero eigenvalues, finite smooth inverse, and exact zero first moment.", "target_id": "e9b3873b7c826c7b"}], "research_policy_version": 1}, "author": "max3", "created_at": "", "details_path": "", "display_id": "P016", "entry_id": "entry-728de1323489ba0c", "kind": "report", "links": ["c7fafba4b4d5a72f", "entry-97292e71c3b56db4", "83fc8a6190cb2ca3", "a0c54399c2d44b8d", "e9b3873b7c826c7b", "dc5ba3bf489a3dea", "entry-f5a5afb3f0a8d294"], "revision": 1, "status": "recorded", "title": "C001 is proved as F010; the finite approximation proof and inversion interface pass manual audit", "updated_at": ""} -->
<a id="entry-728de1323489ba0c"></a>
### P016 · C001 is proved as F010; the finite approximation proof and inversion interface pass manual audit [report: recorded; NOT A FACT]

<!-- hyprover:body -->
The exact statement of C001 was independently accepted and published as F010, fact c7fafba4b4d5a72f (revision 1), with accepted=true, published=true, verdict=correct, no critical errors or gaps, run [run identifier omitted]. The service resolved C001 (entry-97292e71c3b56db4) to status proved at revision 2. Its declared predecessors are F001 (83fc8a6190cb2ca3), F007 (a0c54399c2d44b8d), and F002 (e9b3873b7c826c7b). This closes the precise reusable analytic bridge for every integer n>=3: a real finite sum g of odd harmonics of degrees at least 3 whose hemispherical sign transform has exactly one antipodal pair of nondegenerate critical points.

I independently read and checked the complete F007 proof at revision 1. The normalized polynomial kernels concentrate uniformly; equivariance transfers that convergence to first and second rotation derivatives; the explicit rotation-frame identities control the intrinsic C2 norm. The homogeneous Laplacian recurrence has positive denominators in the required range. The divergence identity removes exactly degree one, and the bounded projection preserves convergence and exact first-moment cancellation. The local definite-Hessian and boundary-gap estimates give existence and uniqueness near each extremum, while the compact-complement gradient bound excludes every other critical point. The separate C1 counterexample is smooth and odd and supplies the claimed three northern points and their antipodes. No mathematical defect was found. This is a manual no-issue audit, not a replacement for fresh service completion revalidation.

For the exact composition, F001 supplies a real smooth odd F_n with zero first moment and two nondegenerate critical points. Choose one finite approximation P from F007 inside its C2 stability threshold; P has only odd degrees at least 3. Apply F002 term by term: g=sum lambda_(n,2j+1)^(-1) P_(2j+1), so Hg=P exactly. All measure normalizations and dimension ranges agree. The chosen g is smooth, real, odd and has integral theta*g(theta)d sigma(theta)=0 by F002, matching F005's input. Neither an arbitrary-smooth inversion theorem nor an estimate uniform in the harmonic degree is required, because one finite P and g are fixed before the radial perturbation parameter is chosen.

Max7 retains the terminal convex-body construction and main retains the final dependency-closure/completion decision. F010 is available as an explicit source-resolved bridge, and this report supplies the focused approximation audit requested by R002 revision 2. No external retrieval or mathematical computation was used.
<!-- hyprover:end-body -->

**Related:** [F010 · Finite odd harmonic preimages whose sign transforms have exactly one nondegenerate critical pair](#fact-c7fafba4b4d5a72f), [C001 · Finite odd harmonic potential with one nondegenerate antipodal critical pair](#entry-97292e71c3b56db4), [F001 · Smooth odd twisted heights with zero first moment and exactly two nondegenerate critical points](#fact-83fc8a6190cb2ca3), [F007 · Finite odd harmonic approximation with exact moment cancellation and two-critical-point stability](#fact-a0c54399c2d44b8d), [F002 · Nonzero odd eigenvalues and finite harmonic inversion of the hemispherical sign transform](#fact-e9b3873b7c826c7b), [F005 · Centered radial perturbations realize one centroidal hyperplane from a two-critical-point sign transform](#fact-dc5ba3bf489a3dea), [R002 · Use the published full-target fact as the completion root](#entry-f5a5afb3f0a8d294)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-728de1323489ba0c`
**Context:** {"audits": [{"not_covered": "none", "revision": 1, "scope": "Complete published statement and proof, parts 1-5: polynomial-kernel C2 approximation; rotation-frame estimates; harmonic decomposition; exact degree-one projection and moment cancellation; local and global critical-point stability; the C1 counterexample.", "target_id": "a0c54399c2d44b8d"}, {"not_covered": "Independent recheck of the phase construction and moment-cancellation proof.", "revision": 1, "scope": "Complete published statement and its input compatibility with F007: every n\u003e=3, real smooth odd potential, exact zero first moment, and exactly two nondegenerate critical points.", "target_id": "83fc8a6190cb2ca3"}, {"not_covered": "Fresh re-verification of the eigenvalue proof or inversion on arbitrary smooth data.", "revision": 1, "scope": "Complete published statement and its compatibility with the finite output of F007: same Euclidean sphere measure, odd harmonics of degrees at least 3, real nonzero eigenvalues, finite smooth inverse, and exact zero first moment.", "target_id": "e9b3873b7c826c7b"}], "research_policy_version": 1}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"audits": [{"not_covered": "Fresh independent service revalidation or final completion certification; separate re-verification of transform inversion, finite harmonic approximation, or the terminal all-dimensional assembly; duplicate detailed normalization/affine-section audit already recorded in P012.", "revision": 1, "scope": "Complete statement, proof and publication receipt read; detailed check of Step 2's C1 extension through the origin, global strong-convexity inequality, exact every-open-chord strict convexity, smooth boundary and positive curvature; Step 4's local existence, uniqueness and nondegeneracy plus compact-complement exclusion; Step 3's two-sided epsilon convergence and the absence of any dimension- or input-uniformity requirement.", "target_id": "dc5ba3bf489a3dea"}], "research_policy_version": 1}, "author": "max4", "created_at": "", "details_path": "", "display_id": "P017", "entry_id": "entry-185782bd07e6c33d", "kind": "report", "links": ["dc5ba3bf489a3dea", "entry-1d453c55f98a455f", "entry-edb4977d81b250ba", "entry-f5a5afb3f0a8d294"], "revision": 1, "status": "recorded", "title": "F005 audit found no gap in global convexity or the exact critical-point count", "updated_at": ""} -->
<a id="entry-185782bd07e6c33d"></a>
### P017 · F005 audit found no gap in global convexity or the exact critical-point count [report: recorded; NOT A FACT]

<!-- hyprover:body -->
Max4 completed the focused manual audit assigned by R002 revision 2. I read the complete statement and proof of F005 (dc5ba3bf489a3dea, revision 1), its V007 receipt, and P012's earlier normalization/interface audit. V007 explicitly records accepted=true, published=true and the same fact_id, with no errors or gaps. F005 remained active at revision 1 in snapshot 55. No concrete mathematical defect was found. This report supplements P012 and completes the assigned counterpressure; it is neither a new fact nor fresh independent completion revalidation.

Proof Step 2 handles the origin correctly. For q_epsilon(x)=|x|^2*rho_epsilon(x/|x|)^(-2), positive degree-two homogeneity carries D^2 q_epsilon>=I from the unit sphere to every nonzero point. The bounds q_epsilon=O(|x|^2) and grad q_epsilon=O(|x|) give differentiability at 0 with derivative 0 and continuity of that derivative. Thus Q=q_epsilon-|x|^2/2 is C1 globally. Along any affine line through 0, the derivative of Q is nondecreasing on each side and has the same limit at the crossing, so it is nondecreasing on the whole line. Lines avoiding 0 are directly covered by the nonnegative second derivative. Consequently Q is convex and q_epsilon((1-t)x+ty)<=(1-t)q_epsilon(x)+t*q_epsilon(y)-t*(1-t)*|x-y|^2/2. For distinct x,y in K_epsilon this is strictly below 1. The equality int(K_epsilon)={q_epsilon<1}, justified by continuity and outward radial dilations at q_epsilon=1, gives precisely the problem's every-open-chord definition, including chords through 0. Smoothness at 0 is never required: the boundary stays away from 0, Euler's identity makes its defining gradient nonzero, and its second fundamental form is D^2 q_epsilon restricted to tangent vectors divided by |grad q_epsilon|, hence positive.

Proof Step 4 gives all three local conclusions independently. Exactly two nondegenerate critical points of h=Hg must be its maximum and minimum, with definite Hessians. Fix disjoint coordinate balls and constants r,lambda before choosing epsilon. Coordinate C2 convergence gives definite perturbed Hessians and gradient error <lambda*r; equation (5) then leaves boundary radial derivative at most -lambda*r^2 at the maximum ball and at least lambda*r^2 at the minimum ball. Moving inward excludes boundary extrema, so an interior extremum exists in each closed ball. Integrating the definite coordinate Hessian along the segment between two proposed critical points proves uniqueness. Definiteness at each interior critical point gives nondegeneracy also for the round Hessian, since the coordinate and covariant Hessians agree there by pullback.

The complement of the two open coordinate images is compact and contains no critical point of h. Its positive gradient minimum m, together with C1 error <m/2, excludes every additional critical point, including the chart boundaries. Oddness pairs the two surviving points antipodally. Step 3's C2 error is bounded by C*|epsilon| on a symmetric parameter interval, and multiplication by (n+1)/epsilon preserves critical points and nondegeneracy for either sign. Negative epsilon only reverses the extrema of B_epsilon relative to its normalized function. A single positive threshold is obtained by intersecting the finitely many small-parameter requirements.

All constants may depend on the fixed n, g, h=Hg and chosen charts. F005 quantifies over n and g before asserting existence of epsilon_0, so neither dimension-uniform bounds nor estimates uniform over harmonic degrees or all smooth inputs are needed. The complete proof was read, with this audit concentrated on Step 2, Step 4, and the two-sided limit/quantifier interface in Step 3. The transform inversion, finite approximation and terminal all-dimensional assembly are outside this audit's scope and retain their existing ownership. No repair or lifecycle review is warranted by these checks. All reasoning was text-only and used no external retrieval.
<!-- hyprover:end-body -->

**Related:** [F005 · Centered radial perturbations realize one centroidal hyperplane from a two-critical-point sign transform](#fact-dc5ba3bf489a3dea), [V007 · Verification submission outcome](#entry-1d453c55f98a455f), [P012 · Published inversion and radial realization have matching hypotheses and normalizations](#entry-edb4977d81b250ba), [R002 · Use the published full-target fact as the completion root](#entry-f5a5afb3f0a8d294)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-185782bd07e6c33d`
**Context:** {"audits": [{"not_covered": "Fresh independent service revalidation or final completion certification; separate re-verification of transform inversion, finite harmonic approximation, or the terminal all-dimensional assembly; duplicate detailed normalization/affine-section audit already recorded in P012.", "revision": 1, "scope": "Complete statement, proof and publication receipt read; detailed check of Step 2's C1 extension through the origin, global strong-convexity inequality, exact every-open-chord strict convexity, smooth boundary and positive curvature; Step 4's local existence, uniqueness and nondegeneracy plus compact-complement exclusion; Step 3's two-sided epsilon convergence and the absence of any dimension- or input-uniformity requirement.", "target_id": "dc5ba3bf489a3dea"}], "research_policy_version": 1}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"audits": [{"not_covered": "Fresh independent completion revalidation; a new audit baseline for F006 beyond P015; the planar sharpness proof and higher-dimensional construction proofs.", "revision": 1, "scope": "Complete statement, proof and successful publication receipt; independent uniform-support proof of centroid interiority; exact F006 hypotheses for arbitrary nonsmooth convex bodies in every n\u003e=2; positive finite section measure; extrema; translation; all-affine-section correspondence and unoriented counting; terminal mu(n)\u003e=1 implication.", "target_id": "a8378dfc6c55d238"}], "research_policy_version": 1}, "author": "max2", "created_at": "", "details_path": "", "display_id": "P018", "entry_id": "entry-5f720ff72a226fde", "kind": "report", "links": ["a8378dfc6c55d238", "8f8adadaf10ea26e", "entry-5cc7ba3960734262", "entry-4800d848113c408d", "entry-127efe34f5e48303", "entry-10acec896a9c45b2", "3ce5f83832ddd38c", "entry-a97772fe5de56e5b"], "revision": 1, "status": "recorded", "title": "F008 audit closes the nonsmooth universal implication without additional hypotheses", "updated_at": ""} -->
<a id="entry-5f720ff72a226fde"></a>
### P018 · F008 audit closes the nonsmooth universal implication without additional hypotheses [report: recorded; NOT A FACT]

<!-- hyprover:body -->
Max2 completed the follow-up audit of F008 (a8378dfc6c55d238, revision 1), the now-published terminal universal theorem under R001 revision 2. I read its complete statement, proof, predecessor F006 (8f8adadaf10ea26e, revision 1), canonical transform glossary, and the full V010 receipt. V010 has accepted=true, published=true, fact_id=a8378dfc6c55d238, verdict=correct, and empty errors/gaps; V008 likewise confirms F006's publication. No concrete mathematical defect was found. This is a version-bound manual audit, not a new fact or fresh independent completion revalidation.

The centroid-interiority argument is self-contained and does not assume that the centroid already lies in K. From a closed ball B(a,r) contained in K, put Q=B(a,r/2), V=vol_n(K), and delta=r*vol_n(Q)/(2V)>0. For every unit u the support function obeys h_K(u)-u dot c(K)>=delta. The proof separately derives the intersection-of-support-halfspaces representation by the nearest-point variational inequality. Consequently B(c(K),delta) is contained in K. Compactness and positive volume justify every centroid integral; the missing-integrability issue identified in unrelated V011 does not occur here.

For every p in int(K), F006 supplies a positive continuous radial representation of K-p and the C1 odd potential B_{K,p}, with exactly grad_S B_{K,p}(u)=2 integral_{(K-p) intersect u-perp} y dH^{n-1}(y). Its maximum on the compact sphere is critical, and every central section contains an (n-1)-ball, so its positive finite measure turns zero first moment into centroid p after translation. These steps require neither boundary smoothness nor strict convexity and apply in n=2 as well as every n>=3. The n=2 equatorial counting measure and the matching polar factors were already checked in P015; F008 introduces no measure conversion.

The final correspondence is exact: averaging the affine equation u dot x=b over a positive-measure matching section yields b=u dot p. Thus every matching affine hyperplane contains p, its centroid condition is equivalent to the critical-point equation, and the only unit normals for one geometric hyperplane are u and -u. Constant potentials and nonisolated critical sets cause no exception; the theorem requires existence, not nondegeneracy.

Therefore the application p=c(K) and the nonsmooth universal lower bound mu(n)>=1 are now contained in the active published F008 itself. The separate terminal-submission gap noted at snapshot 43 in P015 is closed; no further universal composition or duplicate fact is needed from max or max2. F003 remains the separate sharp planar theorem. P009 revision 4 allocates the full-target implication to max7, while main retains fresh dependency revalidation and completion control.
<!-- hyprover:end-body -->

**Related:** [F008 · Every interior point of a convex body is the centroid of some hyperplane section](#fact-a8378dfc6c55d238), [F006 · Signed halfspace volume is C1 and its spherical gradient is twice the central-section first moment](#fact-8f8adadaf10ea26e), [V010 · Verification submission outcome](#entry-5cc7ba3960734262), [V008 · Verification submission outcome](#entry-4800d848113c408d), [P015 · Nonsmooth universal interface audit found no issue; exact centroid application is ready](#entry-127efe34f5e48303), [R001 · Universal and planar components are published and ready for completion revalidation](#entry-10acec896a9c45b2), [F003 · Sharp planar centroid-section theorem: every body has at least three, triangles exactly three](#fact-3ce5f83832ddd38c), [P009 · Full theorem published as F013; exact dependency handoff for completion revalidation](#entry-a97772fe5de56e5b)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-5f720ff72a226fde`
**Context:** {"audits": [{"not_covered": "Fresh independent completion revalidation; a new audit baseline for F006 beyond P015; the planar sharpness proof and higher-dimensional construction proofs.", "revision": 1, "scope": "Complete statement, proof and successful publication receipt; independent uniform-support proof of centroid interiority; exact F006 hypotheses for arbitrary nonsmooth convex bodies in every n\u003e=2; positive finite section measure; extrema; translation; all-affine-section correspondence and unoriented counting; terminal mu(n)\u003e=1 implication.", "target_id": "a8378dfc6c55d238"}], "research_policy_version": 1}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"audits": [{"not_covered": "Fresh independent controller revalidation of the frozen completion closure and original-problem bridge; formal proof certification.", "revision": 1, "scope": "Complete statement and full proof: exact paired-phase cancellation; smooth extension across 0 and the poles; odd sphere diffeomorphism; unique critical pair and Hessians; Euclidean slice weight and every first moment.", "target_id": "83fc8a6190cb2ca3"}, {"not_covered": "Fresh independent controller revalidation of the frozen completion closure and original-problem bridge; formal proof certification.", "revision": 1, "scope": "Complete statement and full proof: homogeneous coefficient recurrence; equatorial average and half-ball flux constants; all odd-degree eigenvalues nonzero for every n\u003e=3; finite inverse, degree-one exclusion and exact first moment.", "target_id": "e9b3873b7c826c7b"}, {"not_covered": "Fresh independent controller revalidation of the frozen completion closure and original-problem bridge; formal proof certification.", "revision": 1, "scope": "Complete statement and full proof: centroid interiority and radial continuity for nonsmooth planar bodies; first-harmonic moment obstruction including sign-preserving zeros; exact triangle boundary-pair classification; both inequalities for mu(2).", "target_id": "3ce5f83832ddd38c"}, {"not_covered": "Fresh independent controller revalidation of the frozen completion closure and original-problem bridge; formal proof certification.", "revision": 1, "scope": "Complete statement and full proof: smooth transform and derivative normalization; strong convexity of the radial gauge through the origin; smooth boundary and curvature; exact centering; C2 remainder; local existence/uniqueness and global exclusion; all affine section counting.", "target_id": "dc5ba3bf489a3dea"}, {"not_covered": "Fresh independent controller revalidation of the frozen completion closure and original-problem bridge; formal proof certification.", "revision": 1, "scope": "Complete statement and full proof: continuous-input differentiability with uniform strip remainder; local-frame continuity and Ck gain; n=2 counting measure; polar identities; gauge continuity for arbitrary translated convex bodies.", "target_id": "8f8adadaf10ea26e"}, {"not_covered": "Fresh independent controller revalidation of the frozen completion closure and original-problem bridge; formal proof certification.", "revision": 1, "scope": "Complete statement and full proof including part (3): normalized kernel concentration and two rotation derivatives; intrinsic Hessian control; finite harmonic decomposition; exact linear projection removal; two-critical-point C2 stability; smooth odd C1 counterexample.", "target_id": "a0c54399c2d44b8d"}, {"not_covered": "Fresh independent controller revalidation of the frozen completion closure and original-problem bridge; formal proof certification.", "revision": 1, "scope": "Complete statement and full proof: uniform support gap giving centroid interiority; exact application of F006 for every n\u003e=2 and every interior point; positive section measure; extremum argument; affine and antipodal bijection; universal mu(n)\u003e=1.", "target_id": "a8378dfc6c55d238"}, {"not_covered": "Fresh independent controller revalidation of the frozen completion closure and original-problem bridge; formal proof certification.", "revision": 1, "scope": "Complete statement and full proof: exact F001-to-F007-to-F002-to-F005 hypothesis matching for arbitrary fixed n\u003e=3; legitimate simultaneous approximation/stability conclusion; all-n smooth strictly convex existence and full unoriented affine-section count.", "target_id": "7a3fd2eb4772e4b4"}], "research_policy_version": 1}, "author": "main_agent", "created_at": "", "details_path": "", "display_id": "P019", "entry_id": "entry-fdc351fd4710f996", "kind": "report", "links": ["83fc8a6190cb2ca3", "e9b3873b7c826c7b", "3ce5f83832ddd38c", "dc5ba3bf489a3dea", "8f8adadaf10ea26e", "a0c54399c2d44b8d", "a8378dfc6c55d238", "7a3fd2eb4772e4b4", "entry-61d4f0a0a2f9515c", "entry-10acec896a9c45b2", "entry-f5a5afb3f0a8d294", "entry-97b5b6c76e914106", "entry-a241367a37a8ed00", "entry-ef50e9d295b14126", "entry-89df6118370546f6"], "revision": 1, "status": "recorded", "title": "Main closure audit: three terminal facts cover the original theorem through eight proofs", "updated_at": ""} -->
<a id="entry-fdc351fd4710f996"></a>
### P019 · Main closure audit: three terminal facts cover the original theorem through eight proofs [report: recorded; NOT A FACT]

<!-- hyprover:body -->
This closure audit read the fixed problem and policy, validated the notebook, browsed the public reports and service feedback, recovered all four routes, and checked successful publication receipts. The audit used textual reasoning without external retrieval.

Main read the complete statements, dependencies, glossaries and full proofs of the eight facts in the selected terminal closure: F009 and its predecessors F001/F007/F002/F005; F008 and its predecessor F006; and F003. The proof-specific audit scopes are recorded below. No concrete defect or unmatched hypothesis was found in that closure. This manual read and synthesis is not a new verified fact, a formal certificate, or the controller's mandatory fresh completion revalidation. Published receipts for all eight have accepted=true, published=true and the matching fact_id.

Exact final bridge: for every n>=3, F009 supplies a C-infinity strictly convex body with exactly one centroid-matching geometric affine hyperplane. Thus that body proves mu(n)<=1 and directly supplies the requested smooth existence assertion. F008 applies to every convex body, including nonsmooth ones, proves its volume centroid interior, and gives at least one positive-measure section centered there; hence mu(n)>=1. Therefore mu(n)=1 for each n>=3. F003 gives at least three matching lines for every planar convex body and exactly three for every nondegenerate triangle, so mu(2)=3. All statements count opposite normals as the same geometric hyperplane. These are every clause of PROBLEM.md.

Strategic decisions are in R001 revision 3, R002 revision 3, R003 revision 2 and R004 revision 2; O001 revision 2 now records the closed mathematical interfaces. Finished component workers were assigned genuinely different follow-ups: max2 nonsmooth universality, max3 finite approximation and the intermediate source interface, max4 global radial convexity/critical stability, and max6 the direct shear alternative. The max/max7/max8 core submissions reached publication during this pass; max5's direct shear remains a complementary construction. R004 is parked because F001 already supplies every required dimension, not because its published conditional theorem failed.

V011 rejects an auxiliary product/suspension candidate only because its opening affine-centroid identity claims too much for arbitrary finite-measure sets without a finite first moment. The stated geometric applications are compact; max6 should restrict that auxiliary paragraph to bounded measurable sets (or assume an integrable first absolute moment) before any resubmission, then continue its shear scrutiny. The unpublished candidate is not in the selected dependency closure, no used fact has been revoked, and the rejection does not refute its geometric conclusions.

Completion handoff uses target IDs 7a3fd2eb4772e4b4, a8378dfc6c55d238 and 3ce5f83832ddd38c. The controller must freshly revalidate every recursive proof in full and this final original-problem bridge; any rejection requires reading the detailed public feedback and repairing the concrete issue rather than resubmitting unchanged evidence.
<!-- hyprover:end-body -->

**Related:** [F001 · Smooth odd twisted heights with zero first moment and exactly two nondegenerate critical points](#fact-83fc8a6190cb2ca3), [F002 · Nonzero odd eigenvalues and finite harmonic inversion of the hemispherical sign transform](#fact-e9b3873b7c826c7b), [F003 · Sharp planar centroid-section theorem: every body has at least three, triangles exactly three](#fact-3ce5f83832ddd38c), [F005 · Centered radial perturbations realize one centroidal hyperplane from a two-critical-point sign transform](#fact-dc5ba3bf489a3dea), [F006 · Signed halfspace volume is C1 and its spherical gradient is twice the central-section first moment](#fact-8f8adadaf10ea26e), [F007 · Finite odd harmonic approximation with exact moment cancellation and two-critical-point stability](#fact-a0c54399c2d44b8d), [F008 · Every interior point of a convex body is the centroid of some hyperplane section](#fact-a8378dfc6c55d238), [F009 · Smooth strictly convex bodies with exactly one centroid-matching hyperplane in every dimension n&gt;=3](#fact-7a3fd2eb4772e4b4), [O001 · The full fixed theorem is published; its complete proof closure awaits fresh revalidation](#entry-61d4f0a0a2f9515c), [R001 · Universal and planar components are published and ready for completion revalidation](#entry-10acec896a9c45b2), [R002 · Use the published full-target fact as the completion root](#entry-f5a5afb3f0a8d294), [R003 · The direct shear alternative is published and retained as a fallback](#entry-97b5b6c76e914106), [R004 · Retain conditional lifting and the published boundary for naive products](#entry-a241367a37a8ed00), [V011 · Verification submission outcome](#entry-ef50e9d295b14126), [V012 · Verification submission outcome](#entry-89df6118370546f6)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-fdc351fd4710f996`
**Context:** {"audits": [{"not_covered": "Fresh independent controller revalidation of the frozen completion closure and original-problem bridge; formal proof certification.", "revision": 1, "scope": "Complete statement and full proof: exact paired-phase cancellation; smooth extension across 0 and the poles; odd sphere diffeomorphism; unique critical pair and Hessians; Euclidean slice weight and every first moment.", "target_id": "83fc8a6190cb2ca3"}, {"not_covered": "Fresh independent controller revalidation of the frozen completion closure and original-problem bridge; formal proof certification.", "revision": 1, "scope": "Complete statement and full proof: homogeneous coefficient recurrence; equatorial average and half-ball flux constants; all odd-degree eigenvalues nonzero for every n\u003e=3; finite inverse, degree-one exclusion and exact first moment.", "target_id": "e9b3873b7c826c7b"}, {"not_covered": "Fresh independent controller revalidation of the frozen completion closure and original-problem bridge; formal proof certification.", "revision": 1, "scope": "Complete statement and full proof: centroid interiority and radial continuity for nonsmooth planar bodies; first-harmonic moment obstruction including sign-preserving zeros; exact triangle boundary-pair classification; both inequalities for mu(2).", "target_id": "3ce5f83832ddd38c"}, {"not_covered": "Fresh independent controller revalidation of the frozen completion closure and original-problem bridge; formal proof certification.", "revision": 1, "scope": "Complete statement and full proof: smooth transform and derivative normalization; strong convexity of the radial gauge through the origin; smooth boundary and curvature; exact centering; C2 remainder; local existence/uniqueness and global exclusion; all affine section counting.", "target_id": "dc5ba3bf489a3dea"}, {"not_covered": "Fresh independent controller revalidation of the frozen completion closure and original-problem bridge; formal proof certification.", "revision": 1, "scope": "Complete statement and full proof: continuous-input differentiability with uniform strip remainder; local-frame continuity and Ck gain; n=2 counting measure; polar identities; gauge continuity for arbitrary translated convex bodies.", "target_id": "8f8adadaf10ea26e"}, {"not_covered": "Fresh independent controller revalidation of the frozen completion closure and original-problem bridge; formal proof certification.", "revision": 1, "scope": "Complete statement and full proof including part (3): normalized kernel concentration and two rotation derivatives; intrinsic Hessian control; finite harmonic decomposition; exact linear projection removal; two-critical-point C2 stability; smooth odd C1 counterexample.", "target_id": "a0c54399c2d44b8d"}, {"not_covered": "Fresh independent controller revalidation of the frozen completion closure and original-problem bridge; formal proof certification.", "revision": 1, "scope": "Complete statement and full proof: uniform support gap giving centroid interiority; exact application of F006 for every n\u003e=2 and every interior point; positive section measure; extremum argument; affine and antipodal bijection; universal mu(n)\u003e=1.", "target_id": "a8378dfc6c55d238"}, {"not_covered": "Fresh independent controller revalidation of the frozen completion closure and original-problem bridge; formal proof certification.", "revision": 1, "scope": "Complete statement and full proof: exact F001-to-F007-to-F002-to-F005 hypothesis matching for arbitrary fixed n\u003e=3; legitimate simultaneous approximation/stability conclusion; all-n smooth strictly convex existence and full unoriented affine-section count.", "target_id": "7a3fd2eb4772e4b4"}], "research_policy_version": 1}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"audits": [{"not_covered": "Fresh independent completion revalidation of the project and re-proving F001's phase-cancellation construction.", "revision": 1, "scope": "Complete statement and seven-part proof: coefficient/centering identity, global strict convexity, implicit radial regularity and uniform C2 control including vertical normals, first variation for all normals, global critical-point stability, section identity, and polynomial input construction.", "target_id": "81a5afe69f10c3d2"}, {"not_covered": "Independent audit of F001's proof of the angle construction.", "revision": 1, "scope": "Full published statement matched to F011's polynomial-input step: dimension range n\u003e=3, even angle and endpoint plateaus, exact zero first moment, and exactly two nondegenerate critical points preserved by the coordinate permutation.", "target_id": "83fc8a6190cb2ca3"}], "research_policy_version": 1}, "author": "max6", "created_at": "", "details_path": "", "display_id": "P020", "entry_id": "entry-94ccd77b205dba23", "kind": "report", "links": ["81a5afe69f10c3d2", "83fc8a6190cb2ca3", "entry-97b5b6c76e914106", "entry-04743c91041fed68", "entry-67a9664998c4132f"], "revision": 1, "status": "recorded", "title": "Independent scrutiny of the published shear construction found no unresolved issue in the checked scope", "updated_at": ""} -->
<a id="entry-94ccd77b205dba23"></a>
### P020 · Independent scrutiny of the published shear construction found no unresolved issue in the checked scope [report: recorded; NOT A FACT]

<!-- hyprover:body -->
Following R003/R004's reassignment, max6 read the complete statement, dependency list and seven-part proof of F011 (81a5afe69f10c3d2, revision 1), together with the matching F001 statement, P005/P010 and their public notes, and the successful V014 publication feedback. This is a manual audit report, not a new proof premise or fresh completion revalidation. No unresolved mathematical issue was found in the checked scope.

All oblique normals and the vertical limits are covered. For u=(a,t), s=|a|>0, slicing the original ball gives D(u)=(2*kappa_{n-2}/s)*a dot integral_{-s}^{s}v(z)(1-z^2/s^2)^((n-2)/2)dz. Substitution z=s*q gives exactly F011's polynomial coefficient map. At s=0, the symmetric z marginal makes the signed volume zero. More importantly, uniform C2 control is proved independently of the apparent 1/s expression: the jointly smooth implicit radial function and local rotations fixing the hemisphere yield a jointly smooth signed-volume function on all normals, with a finite-cover bound for the Taylor remainder. Thus pointwise extension at vertical normals is not being mistaken for a C2 estimate.

Exact centering follows from the disk identity b_k*d_k=b_0*c_k and the volume-preserving shear Jacobian. The Hessian identity D2 Phi[(h,s)] = 2|h-epsilon*v'(z)*s|^2 + 2(1-epsilon*(w-epsilon*v(z)) dot v''(z))*s^2 is bounded positively on a convex cylinder containing the whole body. This verifies global strict convexity along every chord, including the polar caps; positive curvature at boundary points alone is not the argument. The origin is interior for small epsilon, and centering is exact with no translation.

I also checked the global contraction proof for critical-point stability, the coefficient approximation with weighted mean removal, the coordinate permutation from F001, and the conversion from two antipodal critical normals to one geometric affine hyperplane. For the null-boundary step in part 6, an explicit expansion of the convexity/scaling argument is: (1-delta)C lies in the relative interior of C, hence boundary(C) is contained in C minus (1-delta)C and has measure at most [1-(1-delta)^(n-1)]*measure(C), tending to zero. This supplies a direct noncircular measure bound for that step. The published construction therefore provides the intended alternative to the radial/harmonic realization, while retaining its declared dependence on F001.
<!-- hyprover:end-body -->

**Related:** [F011 · Polynomial shears of the ball give unique centroid-matching sections in every dimension n&gt;=3](#fact-81a5afe69f10c3d2), [F001 · Smooth odd twisted heights with zero first moment and exactly two nondegenerate critical points](#fact-83fc8a6190cb2ca3), [R003 · The direct shear alternative is published and retained as a fallback](#entry-97b5b6c76e914106), [P005 · Direct disk-fiber shear gives a candidate replacement for spherical harmonic inversion](#entry-04743c91041fed68), [P010 · Direct shear route now has a complete all-dimensional proof plan using F001](#entry-67a9664998c4132f)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-94ccd77b205dba23`
**Context:** {"audits": [{"not_covered": "Fresh independent completion revalidation of the project and re-proving F001's phase-cancellation construction.", "revision": 1, "scope": "Complete statement and seven-part proof: coefficient/centering identity, global strict convexity, implicit radial regularity and uniform C2 control including vertical normals, first variation for all normals, global critical-point stability, section identity, and polynomial input construction.", "target_id": "81a5afe69f10c3d2"}, {"not_covered": "Independent audit of F001's proof of the angle construction.", "revision": 1, "scope": "Full published statement matched to F011's polynomial-input step: dimension range n\u003e=3, even angle and endpoint plateaus, exact zero first moment, and exactly two nondegenerate critical points preserved by the coordinate permutation.", "target_id": "83fc8a6190cb2ca3"}], "research_policy_version": 1}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"audits": [{"not_covered": "Fresh controller revalidation of all nine proofs in the frozen recursive closure and the original-problem bridge; formal proof certification.", "revision": 1, "scope": "Complete full-target statement, full proof and accepted/published receipt; exact application of F003/F008/F009, dimension quantifiers, nonsmooth universality, existence of the defining largest integers, measure conventions and unoriented affine counting matched against PROBLEM.md.", "target_id": "846e296f22ef7853"}], "research_policy_version": 1}, "author": "main_agent", "created_at": "", "details_path": "", "display_id": "P021", "entry_id": "entry-2d93b9b86ba446a8", "kind": "report", "links": ["846e296f22ef7853", "entry-fdc351fd4710f996", "entry-c72215b98f9c410b", "entry-f5a5afb3f0a8d294", "c7fafba4b4d5a72f", "81a5afe69f10c3d2", "e738ef25f0351e92"], "revision": 1, "status": "recorded", "title": "Completion handoff updated to F013 after the full theorem itself was published", "updated_at": ""} -->
<a id="entry-2d93b9b86ba446a8"></a>
### P021 · Completion handoff updated to F013 after the full theorem itself was published [report: recorded; NOT A FACT]

<!-- hyprover:body -->
After P019's eight-proof audit, F013 (846e296f22ef7853) was published with accepted=true, published=true and a matching fact_id. Main read its complete statement and proof and the complete V016 receipt, finding no additional gap. Select F013 as the single completion target. Its three predecessors are exactly P019's earlier terminal targets F003/F008/F009, so the recursive closure consists of those eight unchanged revision-1 proofs plus F013's final implication. P019's audit baselines remain untouched; this entry adds the new proof's baseline.

The final bridge is unchanged and is now itself published: F008 gives mu(n)>=1 for every n>=2 and every convex body, including nonsmooth ones; F009 supplies exactly-one C-infinity strictly convex examples for every n>=3 and hence mu(n)<=1; F003 supplies at least three planar lines universally and exactly three for triangles, hence mu(2)=3. F013 explicitly preserves all definitions and quantifiers of the fixed problem. Request fresh completion revalidation of F013 and every recursive dependency in full. All workers remain alive under the existing controller; no shutdown or outward publication is requested.

Late returns were also read: C001 is source-resolved by published F010; the alternative direct shear theorem F011 is published and has a no-issue manual audit P020; and the auxiliary product/suspension result is repaired and published as F012. Their complete statements, dependencies and successful publication receipts were read, but they are not added to the selected F013 proof dependency graph. R003/R004 and O001 were refreshed accordingly. No unresolved service rejection affects the selected closure.
<!-- hyprover:end-body -->

**Related:** [F013 · Full centroid-section theorem: smooth sharp examples and mu(2)=3, mu(n)=1 for n&gt;=3](#fact-846e296f22ef7853), [P019 · Main closure audit: three terminal facts cover the original theorem through eight proofs](#entry-fdc351fd4710f996), [V016 · Verification submission outcome](#entry-c72215b98f9c410b), [R002 · Use the published full-target fact as the completion root](#entry-f5a5afb3f0a8d294), [F010 · Finite odd harmonic preimages whose sign transforms have exactly one nondegenerate critical pair](#fact-c7fafba4b4d5a72f), [F011 · Polynomial shears of the ball give unique centroid-matching sections in every dimension n&gt;=3](#fact-81a5afe69f10c3d2), [F012 · Cylinders have open families of centroid-matching sections; symmetric homothetic suspensions retain extra sections](#fact-e738ef25f0351e92)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-2d93b9b86ba446a8`
**Context:** {"audits": [{"not_covered": "Fresh controller revalidation of all nine proofs in the frozen recursive closure and the original-problem bridge; formal proof certification.", "revision": 1, "scope": "Complete full-target statement, full proof and accepted/published receipt; exact application of F003/F008/F009, dimension quantifiers, nonsmooth universality, existence of the defining largest integers, measure conventions and unoriented affine counting matched against PROBLEM.md.", "target_id": "846e296f22ef7853"}], "research_policy_version": 1}

</details>
<!-- hyprover:end -->

<!-- hyprover:entry {"attributes": {"research_policy_version": 1}, "author": "max5", "created_at": "", "details_path": "", "display_id": "P022", "entry_id": "entry-e1b533b44009ca5b", "kind": "report", "links": ["81a5afe69f10c3d2", "83fc8a6190cb2ca3", "entry-97b5b6c76e914106", "entry-a241367a37a8ed00", "entry-022e7d767c25435b", "entry-683f18cbb33b8f19", "entry-ffad014f49254883"], "revision": 1, "status": "recorded", "title": "R003 closed: direct polynomial-shear bodies work in every n\u003e=3", "updated_at": ""} -->
<a id="entry-e1b533b44009ca5b"></a>
### P022 · R003 closed: direct polynomial-shear bodies work in every n&gt;=3 [report: recorded; NOT A FACT]

<!-- hyprover:body -->
The independent verifier accepted and published F011, fact 81a5afe69f10c3d2, revision 1, with accepted=true, published=true, verdict=correct and no mathematical errors or gaps (run [run identifier omitted], feedback entry-022e7d767c25435b). This completes max5's geometric construction assignment in every n>=3, beyond the initial three-dimensional target. Its only project predecessor is F001, 83fc8a6190cb2ca3.

The explicit family is K_epsilon={(w,z) in R^(n-1) x R: |w-epsilon*v(z)|^2+z^2<=1}, for an even polynomial curve v with only its first two coordinates nonzero. The proof chooses polynomial coefficients A,B with exact weighted mean zero and a two-critical-point potential Psi(u)=u1*A(u_n^2)+u2*B(u_n^2). The coefficient inverse v(z)=(2*kappa_(n-2))^(-1) sum_k p_k*z^(2k)/b_(n,k), where P(r)=(A(1-r),B(1-r),0,...)=sum_k p_k*r^k and b_(n,k)=integral q^(2k)(1-q^2)^((n-2)/2)dq>0, gives exact body centering. A Hessian estimate proves smooth strict convexity for every sufficiently small nonzero epsilon. Uniform C2 convergence S_epsilon/epsilon -> Psi, with S_epsilon(u)=integral_K sign(u dot x)dx, proves exactly two antipodal nondegenerate critical normals. The exact derivative dS(u)[h]=2 integral_(K intersect u-perp) h dot x then gives exactly one geometric centroid-matching hyperplane, covering all orientations and all affine sections.

No input remains unmatched: F001's even, plateaued twist coefficients become smooth functions of t^2; a self-contained univariate C2 polynomial approximation and weighted constant correction supply A,B while preserving the two critical points. The direct fiber construction requires neither a separate dimension-lifting theorem nor spherical-harmonic inversion nor an unproved smoothing/recentering step. The full theorem's planar and universal components, its final implication, and fresh completion revalidation remain under their existing owners; F011 is an alternative high-dimensional construction, not a replacement claim for those tasks.

A focused fresh lifecycle review of my elementary boundary-measure concern rejected revocation (accepted=false, applied=false, no revoked facts; feedback entry-ffad014f49254883). The preceding interior-ball argument already proves the strict-contraction inclusion and hence the required volume equality. E001 records the resolved expository clarification. F011 remains active at revision 1 with no unresolved construction issue. No external retrieval or mathematical computation was used. I will return to host waiting after persisting this handoff.
<!-- hyprover:end-body -->

**Related:** [F011 · Polynomial shears of the ball give unique centroid-matching sections in every dimension n&gt;=3](#fact-81a5afe69f10c3d2), [F001 · Smooth odd twisted heights with zero first moment and exactly two nondegenerate critical points](#fact-83fc8a6190cb2ca3), [R003 · The direct shear alternative is published and retained as a fallback](#entry-97b5b6c76e914106), [R004 · Retain conditional lifting and the published boundary for naive products](#entry-a241367a37a8ed00), [V014 · Verification submission outcome](#entry-022e7d767c25435b), [E001 · F011 boundary-measure concern resolved as an expository clarification](#entry-683f18cbb33b8f19), [V017 · Fact review: revoke](#entry-ffad014f49254883)

<details>
<summary>记录信息 / Record metadata</summary>

Internal ID: `entry-e1b533b44009ca5b`
**Context:** {"research_policy_version": 1}

</details>
<!-- hyprover:end -->

<a id="history"></a>
## Revocations and history — not usable as premises
