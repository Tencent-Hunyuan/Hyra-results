# Verified proof

[Back to knowledge](../../KNOWLEDGE.md#fact-dc5ba3bf489a3dea)

<!-- hyprover:body -->
All surface and Hausdorff measures below have the usual Euclidean normalization. Norms C^k on the sphere can be defined using the round metric; equivalently one may use a fixed finite smooth atlas with relatively compact coordinate patches. No published project fact or external mathematical source is used.

1. Regularity and a derivative formula for H.

For every smooth f on the sphere the function Hf is smooth. To see this without differentiating the discontinuous sign function, fix u_0 on the sphere. On a neighborhood U of u_0 choose a smooth family of rotations R(u) with R(u)u_0=u. Such a family is obtained by applying Gram--Schmidt to a fixed orthonormal basis of u_0^perp projected onto u^perp, adjoining u as the last column, and choosing the orientation consistently; the projected vectors are independent when u is sufficiently close to u_0. Change variables theta=R(u)eta to obtain
(Hf)(u)=integral_{S^{n-1}} sign(u_0 dot eta) f(R(u)eta) d sigma(eta).
The domain and sign factor are now fixed. Derivatives of any specified finite order in local coordinates act only on f(R(u)eta), whose derivatives are bounded on compact coordinate patches times the sphere. Differentiation under the integral is therefore valid. The chain rule, a finite cover, and compactness give, for each fixed integer k>=0, a constant C_{n,k}<infinity (for the fixed norms) such that
||Hf||_{C^k} <= C_{n,k} ||f||_{C^k}.
This argument also applies to C^k input by the same dominated differentiation in each chart. In particular H maps smooth functions to smooth functions and is continuous in C^2. Directly from its definition Hf(-u)=-Hf(u), and H1=0.

For a smooth f, u in S^{n-1}, and v in T_u S^{n-1}=u^perp, set A=v u^T-u v^T. Thus A is skew-symmetric and Au=v. The curve u(t)=exp(tA)u has u'(0)=v. The preceding change of variables gives
d(Hf)_u[v]=integral_{S^{n-1}} sign(u dot eta) grad_S f(eta) dot A eta d sigma(eta).
Let E={eta in S^{n-1}:u dot eta=0}, Omega_+={u dot eta>0}, and Omega_-={u dot eta<0}. The tangent vector field X(eta)=A eta has spherical divergence zero. Indeed, in any orthonormal tangent frame e_1,...,e_{n-1}, its divergence is sum_i e_i dot A e_i=0 by skew-symmetry. The spherical divergence theorem for fX on the two hemispheres applies, since their boundaries are the smooth equator E. Their outward unit conormals on E are -u and +u, respectively. On E, X(eta)=-u(v dot eta); hence
integral_{Omega_+} grad_S f dot X d sigma = integral_E f(eta)(v dot eta) d sigma_E(eta),
integral_{Omega_-} grad_S f dot X d sigma = -integral_E f(eta)(v dot eta) d sigma_E(eta).
Subtracting proves
d(Hf)_u[v]=2 integral_E (v dot eta) f(eta) d sigma_E(eta).                 (1)
Here sigma_E is (n-2)-dimensional Euclidean surface measure on the unit sphere in u^perp.

2. Smoothness, strict convexity, positive curvature, and exact centering.

Choose an initial delta>0 such that 1+epsilon g(theta)>=1/2 for all theta whenever |epsilon|<=delta. This is possible because g is bounded. The resulting rho_epsilon is smooth, positive, and converges to 1 in C^k for every fixed k as epsilon->0; this follows directly by differentiating the smooth function (epsilon,theta) |-> (1+epsilon g(theta))^{1/(n+1)} on the compact parameter interval and sphere.

For x!=0 define
q_epsilon(x)=|x|^2 rho_epsilon(x/|x|)^{-2},
and set q_epsilon(0)=0. It is smooth off 0 and positively homogeneous of degree two. Its Hessian on |x|=1 converges uniformly to the Hessian 2I of q_0(x)=|x|^2: the expression for q_epsilon is smooth jointly in epsilon and x on an annulus containing the sphere, so its second x derivatives converge uniformly there. Shrink delta so that
D^2 q_epsilon(theta) >= I
as quadratic forms for every theta in S^{n-1} and |epsilon|<=delta. Homogeneity gives D^2 q_epsilon(r theta)=D^2 q_epsilon(theta) for every r>0, so this lower bound holds at every nonzero x.

The functions q_epsilon satisfy q_epsilon(x)=O(|x|^2) and grad q_epsilon(x)=O(|x|) as x->0. The latter follows from degree-one homogeneity of their gradients and boundedness on the unit sphere. Consequently q_epsilon is C^1 on R^n with derivative zero at 0. The function Q_epsilon(x)=q_epsilon(x)-|x|^2/2 is convex: on any affine line it is C^1 and its second derivative is nonnegative wherever the line is not at 0. A nonconstant affine line meets 0 at most once, and continuity of the first derivative across that point shows that the first derivative is nondecreasing on the entire line. This is exactly the one-dimensional criterion for convexity. It follows that for x,y in R^n and 0<t<1,
q_epsilon((1-t)x+ty)
<= (1-t)q_epsilon(x)+t q_epsilon(y)-(t(1-t)/2)|x-y|^2.                 (2)

The radial set K_epsilon is exactly {q_epsilon<=1}. Since rho_epsilon has positive minimum and finite maximum, this set is compact and contains a Euclidean ball about 0. Its interior is {q_epsilon<1}: that sublevel is open, while a point with q_epsilon=1 has arbitrarily close radial dilates with q_epsilon>1. Formula (2) proves convexity and, for distinct x,y in K_epsilon, puts the open segment strictly in {q_epsilon<1}. Thus it gives the required strict convexity globally, including chords through 0.

The boundary {q_epsilon=1} is smooth, since Euler's identity grad q_epsilon(x) dot x=2q_epsilon(x)=2 holds there, so the gradient is nonzero. It is also the embedded radial image theta |-> rho_epsilon(theta)theta. For its outward unit normal nu=grad q_epsilon/|grad q_epsilon|, the second fundamental form on a nonzero tangent vector w is
w dot D_w nu = D^2 q_epsilon[w,w]/|grad q_epsilon| >0.
Hence all principal curvatures (with this positive convention) are positive.

Polar integration in R^n gives the vector first moment
integral_{K_epsilon} x dx
= (1/(n+1)) integral_{S^{n-1}} theta rho_epsilon(theta)^{n+1} d sigma(theta)
= (1/(n+1)) ( integral theta d sigma + epsilon integral theta g(theta) d sigma )
=0.
The first integral vanishes by theta |-> -theta, and the second is the assumed vector identity. The volume is positive, so the centroid is exactly 0. No translation is used.

3. The signed halfspace volume and its C^2 limit.

A second polar integration gives
B_epsilon(u)=(1/n) integral sign(u dot theta) rho_epsilon(theta)^n d sigma(theta)
=(1/n)H[(1+epsilon g)^{a}](u),     a=n/(n+1).                         (3)
The central hyperplane has n-dimensional volume zero, so open versus closed halfspaces does not affect this identity. Step 1 makes B_epsilon smooth and odd.

Taylor's formula with integral remainder, applied pointwise, is
(1+epsilon g)^a=1+a epsilon g+epsilon^2 r_epsilon,
r_epsilon(theta)=a(a-1)g(theta)^2 integral_0^1 (1-s)(1+s epsilon g(theta))^{a-2} ds.
For |epsilon|<=delta, all denominators stay bounded away from zero. The displayed expression and its first two spherical derivatives are continuous on the compact set of parameters epsilon, s, theta, and bounded there. Thus sup_{|epsilon|<=delta}||r_epsilon||_{C^2}<infinity. Since H1=0, equation (3) becomes
B_epsilon=(epsilon/(n+1))Hg+(epsilon^2/n)H r_epsilon.
For epsilon!=0 define F_epsilon=((n+1)/epsilon)B_epsilon. The C^2 bound for H proves
||F_epsilon-h||_{C^2} <= C |epsilon| ->0.                              (4)
Every F_epsilon is odd. Multiplication by the nonzero scalar (n+1)/epsilon changes neither critical points nor nondegeneracy.

4. Global stability of precisely two critical points.

The given function h is smooth by Step 1 and is nonconstant, since a constant function would have every point critical. On the compact sphere it has a maximum and a minimum, which are distinct critical points. They must be its entire critical set; denote them p_+ and p_-. Since h is odd, the antipodal map sends a critical point to a critical point and interchanges its maximum and minimum; hence p_-=-p_+. The Hessian at the maximum is negative semidefinite and at the minimum positive semidefinite. Nondegeneracy therefore makes them negative definite and positive definite, respectively.

Choose smooth charts phi_+, phi_- from neighborhoods of 0 in R^{n-1} to neighborhoods of p_+, p_-, with phi_+(0)=p_+, phi_-(0)=p_-. Their images can be made disjoint. At a critical point, the coordinate Hessian is the pullback of the round Hessian, so its definiteness is unchanged. By continuity, shrink to a common radius r>0, with the charts defined on a neighborhood of the closed Euclidean ball Bbar_r, and choose lambda>0 such that, for h_+=h composed with phi_+ and h_-=h composed with phi_-,
D^2 h_+(z) <= -2 lambda I,       D^2 h_-(z) >= 2 lambda I
throughout Bbar_r. Both coordinate gradients at 0 vanish. Integrating their Hessians along the segment from 0 to z gives, for |z|=r,
z dot grad h_+(z) <= -2 lambda r^2,
z dot grad h_-(z) >=  2 lambda r^2.                                   (5)

C^2 convergence (4) implies convergence of the coordinate gradients and Hessians on these closed balls. For all sufficiently small nonzero epsilon, writing F_{epsilon,+}=F_epsilon composed with phi_+ and F_{epsilon,-}=F_epsilon composed with phi_-, we therefore have
D^2 F_{epsilon,+} <= -lambda I,   D^2 F_{epsilon,-} >= lambda I,
and
||grad F_{epsilon,+}-grad h_+||_infinity < lambda r,
||grad F_{epsilon,-}-grad h_-||_infinity < lambda r.
Together with (5), these imply strictly negative and strictly positive radial derivatives, respectively, on |z|=r.

The maximum of F_{epsilon,+} on Bbar_r is attained in its interior: at every boundary point z, moving a short distance in the inward direction -z increases the value because -z dot grad F_{epsilon,+}(z)>0. Thus an interior critical point exists. It is unique, since for any distinct z,w in B_r the Hessian bound gives
(grad F_{epsilon,+}(z)-grad F_{epsilon,+}(w)) dot (z-w)
= integral_0^1 (z-w)^T D^2 F_{epsilon,+}(w+s(z-w))(z-w) ds
<= -lambda |z-w|^2<0,
which forbids both gradients being zero. The same argument, with minimum and the opposite inequality, gives exactly one interior critical point of F_{epsilon,-}. Their definite coordinate Hessians make them nondegenerate.

The remaining set
C=S^{n-1} \ (phi_+(B_r) union phi_-(B_r))
is compact and contains neither critical point of h. Hence |grad_S h| has a positive minimum m on C. By C^1 convergence, for all sufficiently small nonzero epsilon,
sup_C |grad_S F_epsilon-grad_S h|<m/2.
No critical point of F_epsilon can lie on C. This proves there are exactly the two already found, with no unexamined part of the sphere. Since F_epsilon is odd, critical points occur in antipodal pairs: differentiating F_epsilon(-u)=-F_epsilon(u) shows that if its differential vanishes at u, it vanishes at -u. The antipodal map has no fixed point on the sphere, so our two points form one antipodal pair. Equation (4) and the argument work for both signs of epsilon. Take epsilon_0 small enough to meet all bounds in Steps 2--4.

5. Relating critical points to all affine hyperplane sections.

For u on the sphere the section K_epsilon intersect u^perp has nonempty relative interior, because K_epsilon contains a ball about 0. Its positive finite (n-1)-dimensional measure makes its centroid well-defined. Polar integration within the (n-1)-dimensional Euclidean space u^perp gives
M_epsilon(u):=integral_{K_epsilon intersect u^perp} x d H^{n-1}(x)
=(1/n) integral_{S^{n-1} intersect u^perp} theta rho_epsilon(theta)^n d sigma_E(theta).
The vector M_epsilon(u) belongs to u^perp. Apply (1) with f=rho_epsilon^n and use (3): for every v in u^perp,
d(B_epsilon)_u[v]=(2/n) integral_E (v dot theta)rho_epsilon(theta)^n d sigma_E(theta)
=2 v dot M_epsilon(u).
The defining property of the round gradient now proves grad_S B_epsilon(u)=2M_epsilon(u). Thus a central hyperplane section has centroid 0 if and only if its normal is a critical point of B_epsilon. Step 4 gives exactly two such unit normals, and they are antipodal, so they specify exactly one geometric central hyperplane.

Finally, if any affine hyperplane A={x:u dot x=b}, |u|=1, has a section of positive (n-1)-measure, its centroid c_A satisfies u dot c_A=b, by integrating the defining affine equation over that section and dividing by its measure. If c_A=0 this forces b=0. Consequently every affine section with the required centroid is central and was included in the preceding count. Since c(K_epsilon)=0, this completes all conclusions.
<!-- hyprover:end-body -->
