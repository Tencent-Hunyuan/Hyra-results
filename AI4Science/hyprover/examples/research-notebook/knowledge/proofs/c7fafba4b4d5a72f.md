# Verified proof

[Back to knowledge](../../KNOWLEDGE.md#fact-c7fafba4b4d5a72f)

<!-- hyprover:body -->
Fix an arbitrary integer n>=3. Let S=S^{n-1} be the Euclidean unit sphere in R^n, equipped with its round metric and ordinary Euclidean surface measure sigma. A spherical harmonic of degree k means the restriction of a real homogeneous polynomial of degree k annihilated by the Euclidean Laplacian. All three predecessor facts use these same definitions and measure normalization.

1. Obtain a smooth centered potential with one critical pair.

By fact 83fc8a6190cb2ca3, there is a real C-infinity function F_n on S which is odd, satisfies
integral_S x F_n(x) d sigma(x)=0,
and has exactly two critical points e_1 and -e_1. Their spherical Hessians are respectively minus and plus the round metric, so both are nondegenerate. Here e_1 is the first coordinate unit vector. In particular F_n is C^2 and meets every hypothesis of parts (1) and (2) of fact a0c54399c2d44b8d.

2. Obtain a finite harmonic potential with the same critical-point count.

Use the norm
||u||_{C^2}=sup_S |u|+sup_S |grad_S u|+sup_S ||Hess_S u||_{op}.
By part (1) of fact a0c54399c2d44b8d, there is a sequence P_m converging to F_n in this norm, where every P_m is a real finite sum of spherical harmonics of odd degrees at least 3. By part (2) of that fact, there is a number delta>0 such that any real C^2 function within delta of F_n in this norm has exactly two critical points, both nondegenerate. Choose one index m for which ||P_m-F_n||_{C^2}<delta, and put P=P_m. Then P has exactly two nondegenerate critical points. Because P is odd, these points form an antipodal pair, also as stated in part (2) of the cited fact.

Consequently, for some finite integer J>=1 we may write
P=sum_{j=1}^{J} P_{2j+1},
where P_{2j+1} is a real spherical harmonic of degree 2j+1; some terms may be zero. There is no degree-one term. P cannot be identically zero, since the zero function is critical at every point of S, whereas P has exactly two critical points.

3. Invert only this finite sum.

Define the transform on continuous real functions by
(Hf)(u)=integral_S sign(u dot theta) f(theta) d sigma(theta),
with sign(0)=0. The equator has sigma-measure zero, so this convention agrees with the statement regardless of the value assigned to sign at zero. By fact e9b3873b7c826c7b, for every j>=0 its eigenvalue on degree 2j+1 is the nonzero real number
lambda_{n,2j+1}=2 s_{n-2}(-1)^j(2j)!/[2^j j! product_{r=0}^{j}(n-1+2r)],
where s_{n-2} is the ordinary Euclidean surface measure of S^{n-2}.

Set
g=sum_{j=1}^{J} lambda_{n,2j+1}^{-1} P_{2j+1}.
All coefficients are real and finite. Thus g is a real finite sum of spherical harmonics of odd degrees at least 3, as required. Each term is a polynomial restriction, so g is smooth and odd. The same predecessor also gives integral_S theta g(theta) d sigma(theta)=0, although this additional property is not needed to state the present conclusion.

Linearity of the integral over this finite sum and the eigenvalue identity give, for every u in S,
(Hg)(u)=sum_{j=1}^{J} lambda_{n,2j+1}^{-1}(H P_{2j+1})(u)
       =sum_{j=1}^{J} P_{2j+1}(u)
       =P(u).
In particular Hg is precisely the smooth function P, so its critical points and spherical Hessians coincide with those of P. It therefore has exactly two critical points, which are antipodal and both nondegenerate.

The argument applies to each integer n>=3, without requiring a bound uniform in n or in the finite harmonic degree. This proves the entire statement using exactly the three declared published predecessors.
<!-- hyprover:end-body -->
