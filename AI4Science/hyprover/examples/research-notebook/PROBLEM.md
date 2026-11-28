# Grünbaum–Loewner centroid-section theorem

## Source and scope

The fixed target is Theorem 1.1 of [arXiv:2606.19865v2](https://arxiv.org/abs/2606.19865v2), dated 31 July 2026. This file supplies only the theorem statement and necessary definitions, not the article's proof. It is a known theorem used as a research task, not a claim of an original discovery.

This is a sanitized, static archive. Research sessions were configured without external literature retrieval. The setup extraction had returned more article content than requested before a fresh-context handoff; the archive does not certify strict blindness. See the example README for export and verification limitations.

## Definitions

Let \(n\ge2\) be an integer. Work in Euclidean space \(\mathbb R^n\). A convex body is a compact convex set \(K\subset\mathbb R^n\) with nonempty interior. Write \(\mathcal H^k\) for k-dimensional Hausdorff measure with the usual Euclidean normalization. Its uniform-volume centroid is

\[
c(K)=\frac{1}{\mathcal H^n(K)}\int_K x\,d\mathcal H^n(x).
\]

A hyperplane section is \(K\cap H\), where H is an affine hyperplane and \(\mathcal H^{n-1}(K\cap H)>0\). Its centroid is

\[
c(K\cap H)=\frac{1}{\mathcal H^{n-1}(K\cap H)}
\int_{K\cap H}x\,d\mathcal H^{n-1}(x).
\]

Count distinct geometric affine hyperplanes without orientation: opposite normals describe the same section.

K is smooth if \(\partial K\) is a \(C^\infty\) submanifold of \(\mathbb R^n\). It is strictly convex if, for all distinct \(x,y\in K\) and \(\lambda\in(0,1)\),

\[
\lambda x+(1-\lambda)y\in\operatorname{int}K.
\]

Let \(\mu(n)\) be the largest integer q such that every convex body in \(\mathbb R^n\) has at least q hyperplane sections whose centroids equal the body's centroid. The universal quantifier includes nonsmooth bodies.

## Complete target

For every integer \(n\ge3\), there exists a smooth strictly convex body \(K\subset\mathbb R^n\) for which exactly one hyperplane section has centroid \(c(K)\). Consequently,

\[
\mu(n)=
\begin{cases}
3,&n=2,\\
1,&n\ge3.
\end{cases}
\]

A complete argument must cover the smooth strictly convex existence assertion in every \(n\ge3\), the universal lower bound without boundary regularity, and both planar inequalities. Selected dimensions, approximate centering, or extra hypotheses do not suffice.

## Reading

Read [KNOWLEDGE.md](KNOWLEDGE.md) and follow F013's predecessor links for the full proof chain. Historical LLM verification is not a formal certificate; non-factual entries are never additional premises.
