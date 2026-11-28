# Verified proof

[Back to knowledge](../../KNOWLEDGE.md#fact-8f8adadaf10ea26e)

<!-- hyprover:body -->
We first prove the transform formula for arbitrary continuous f. Fix u in S^{n-1}. Apart from the two poles, write a point of S^{n-1} as
theta(t,eta)=sqrt(1-t^2) eta+t u,
where -1<t<1 and eta in E_u. A tangent vector in an E_u direction is multiplied by sqrt(1-t^2). The t derivative has length (1-t^2)^(-1/2) and is orthogonal to those directions. There are n-2 equatorial tangent directions. Thus the surface Jacobian is (1-t^2)^((n-3)/2), giving
integral_{S^{n-1}} q(theta) d sigma
=integral_{E_u} integral_{-1}^1 q(theta(t,eta))(1-t^2)^((n-3)/2) dt d sigma_u(eta).
For n=2, E_u has two points, each parametrizing one open semicircle; its t derivative gives the same formula with exponent -1/2. The omitted poles have zero surface measure in every n>=2.

Use the local chart v -> w(v)=(u+v)/sqrt(1+|v|^2), for v in u^\perp near zero. Put a=v dot eta. Since normalization is positive,
sign(w(v) dot theta(t,eta))=sign(t+a sqrt(1-t^2)).
The function t/sqrt(1-t^2) is strictly increasing on (-1,1), with range R. The expression in the last sign therefore has a single zero at
b(a)=-a/sqrt(1+a^2),
and its sign is sign(t-b(a)). Set
F(eta,t)=f(theta(t,eta))(1-t^2)^((n-3)/2).
The exact difference of the sign integrals is consequently
H f(w(v))-H f(u)
=-2 integral_{E_u} integral_0^{b(v dot eta)} F(eta,t) dt d sigma_u(eta),
where integrals with a negative upper limit are oriented integrals. This identity follows separately on each eta fiber by observing that the sign changes precisely on the interval with endpoints 0 and b.

On E_u times [-1/2,1/2], F is continuous, and F(eta,0)=f(eta). Compactness gives a function omega(delta) tending to zero as delta tends to zero such that
|F(eta,t)-f(eta)|<=omega(delta) whenever |t|<=delta<=1/2.
Also |b(a)|<=|a| and |b(a)+a|<=|a|^3/2. For |v|<=1/2, substituting F(eta,t)=f(eta)+(F(eta,t)-f(eta)) in the exact difference identity therefore gives
H f(w(v))-H f(u)
=2 integral_{E_u} (v dot eta) f(eta) d sigma_u(eta)+R(v),
with
|R(v)|<=sigma_u(E_u) ||f||_infinity |v|^3
 +2 sigma_u(E_u) |v| omega(|v|).
In particular R(v)=o(|v|). The chart has derivative equal to the identity on u^\perp at v=0. This proves differentiability at u, with tangent gradient
G(u)=2 integral_{E_u} eta f(eta) d sigma_u(eta).
Every vector integrated is perpendicular to u, so G(u) is tangent.

We verify continuity and the higher regularity without differentiating f when it is merely continuous. Fix u_0. A smooth orthogonal matrix R(u), for u in a neighborhood of u_0, can be chosen with R(u_0)=I and R(u)u_0=u. One explicit local construction is to fix an orthonormal basis e_1,...,e_{n-1} of u_0^\perp, project each e_j to u^\perp, and apply Gram-Schmidt in that order. The denominators are nonzero near u_0 because they are nonzero at u_0. The resulting vectors and u are a smooth orthonormal frame, defining R(u). Orthogonal change of variables on E_{u_0} yields
G(u)=2 R(u) integral_{E_{u_0}} eta f(R(u)eta) d sigma_{u_0}(eta).
Uniform continuity of f on the sphere proves continuity of this expression. Thus H f is C^1, not just pointwise differentiable. If f is C^k, differentiation up to order k under this fixed compact-domain integral is justified by the chain rule and continuity and boundedness of those derivatives. It shows that G is C^k. In local coordinates every first partial derivative of H f is the inner product of G with the corresponding coordinate tangent vector; these partial derivatives are C^k, so H f is C^{k+1}. On a smaller closed neighborhood the derivatives of R and of the coordinate maps through every fixed order are bounded. The displayed formula and the chain rule bound the derivatives of G through order k by a constant times ||f||_{C^k}. A finite collection of such neighborhoods covers the sphere, yielding the stated C^{k+1} norm bound. The order-zero bound for H f and the explicit gradient bound follow directly by taking absolute values in their integrals. Linearity follows from the integral. Finally H f(-u)=-H f(u) follows from sign(-s)=-sign(s), and G(-u)=G(u) follows from E_{-u}=E_u with identical measure.

Now let rho be positive and continuous. Its minimum m is positive and its maximum M is finite. K_rho is the image of the compact set [0,1] times S^{n-1} under (s,theta)->s rho(theta)theta. It contains the closed ball of radius m and is contained in the ball of radius M. The n-dimensional measure of a hyperplane is zero. Thus B_rho is the integral of sign(u dot x) over K_rho. Polar integration in R^n gives
B_rho(u)=integral_{S^{n-1}} sign(u dot theta) integral_0^{rho(theta)} r^{n-1} dr d sigma(theta)
=(1/n)H(rho^n)(u).
In the (n-1)-dimensional Euclidean space u^\perp the radial function of the section is rho restricted to E_u. Its polar volume element is r^{n-2} dr d sigma_u. Therefore
H^{n-1}(K_rho intersect u^\perp)
=integral_{E_u} integral_0^{rho(theta)} r^{n-2} dr d sigma_u
=(1/(n-1))integral_{E_u} rho^{n-1} d sigma_u,
and, component by component,
M_rho(u)=integral_{E_u} theta integral_0^{rho(theta)} r^{n-1} dr d sigma_u
=(1/n)integral_{E_u} theta rho^n d sigma_u.
These formulas are also valid in the one-dimensional section space when n=2, with its two polar rays counted once each. The area formula is positive and finite because m<=rho<=M and the equatorial sphere has positive finite measure. Applying the transform gradient formula to rho^n gives grad_S B_rho=2 M_rho. Since the section area is positive, its centroid is zero exactly when M_rho=0, which is exactly the critical-point condition. The claimed regularity follows because rho^n is C^k whenever rho is C^k.

It remains to justify the nonsmooth convex-body application. Translate p to zero and write L=K-p. There are numbers 0<r<R with r times the closed unit ball contained in L and L contained in R times that ball. Define the gauge
gamma(x)=inf{t>0:x in tL}, gamma(0)=0.
It is positively homogeneous. The two ball inclusions give |x|/R<=gamma(x)<=|x|/r. Convexity proves subadditivity: for any s>gamma(x), t>gamma(y), the radial structure of L gives x in sL and y in tL, and convexity gives (x+y)/(s+t) in L. Letting s and t decrease to gamma(x) and gamma(y) gives gamma(x+y)<=gamma(x)+gamma(y). Consequently
gamma(x)-gamma(y)<=gamma(x-y)<=|x-y|/r,
and interchanging x and y gives |gamma(x)-gamma(y)|<=|x-y|/r.
Here the radial structure used above is simply that 0 in L and convexity imply lambda L subset L for 0<=lambda<=1; together with the infimum definition this ensures x in sL for every s>gamma(x). Closedness of L gives gamma(x)<=1 exactly when x in L. For theta on the unit sphere the ray intersects L in the interval 0<=s<=1/gamma(theta). Hence its radial function is rho(theta)=1/gamma(theta), which is positive, finite, and continuous because gamma is Lipschitz and gamma(theta)>=1/R. This realizes L as K_rho and establishes every claimed application.
<!-- hyprover:end-body -->
