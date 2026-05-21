# LMS Gauge Invariants and Exact Finite-N Reduction

This note gives a focused mathematical account of the exact finite-$N$ reduction used in the LMSSPP package. The central point is that the reduction does not require a hydrodynamic limit, a Poisson kernel closure, or a moment-matching approximation. It is an exact statement about a finite weighted cloud on the sphere and its orbit under the real Mobius group.

The key construction is the exact Busemann centering map. Given a cloud $x=(x_1,\dots,x_N) \in (S^{d-1})^N$, it chooses a canonical boost parameter $z(x) \in B^d$ such that the boosted cloud has zero weighted barycenter. This turns the usual LMS reduced variables into a canonical moving-frame decomposition:

$$
\text{full cloud } x(t)
\quad \longleftrightarrow \quad
\text{boost coordinate } w(t),\ \text{rotation } \zeta(t),\ \text{frozen shape constants } p^0.
$$

The result is a finite-$N$ Watanabe-Strogatz-style reconstruction theorem: one stores $N$ constants once, evolves only $w(t) \in B^d$ and $\zeta(t) \in SO(d)$, and reconstructs the full $N$-point cloud by one Mobius transformation at each time.

---

## 1. Conventions

The implementation uses row vectors. Points lie on $S^{d-1} \subset \mathbb R^d$, weights satisfy

$$
a_i \ge 0, \qquad \sum_{i=1}^N a_i=1,
$$

and $A \in \mathfrak{so}(d)$ acts on row vectors by right multiplication with $A^\top$.

The real Mobius boost $M_w$ is implemented as `mobius_sphere`. On the sphere,

$$
M_w(x)=\frac{(1-|w|^2)(x-w)}{|x-w|^2}-w,
\qquad x \in S^{d-1},\quad w \in B^d.
$$

The skew operator used throughout the code is

$$
\alpha(u,v)=v u^\top-u v^\top.
$$

Thus $\alpha(u,v)$ is a $d \times d$ skew-symmetric matrix. This is the matrix returned by `alpha_operator(u, v)`.

The full finite-$N$ LMS/Kuramoto-on-sphere dynamics with linear order parameter is

$$
\dot x_i=x_i A^\top+Z-\langle Z,x_i\rangle x_i,
\qquad
Z=K\sum_{j=1}^N a_j x_j.
$$

Equivalently, the coupling term can be written pairwise as

$$
\dot x_i=x_i A^\top+K\sum_{j=1}^N a_j\bigl(x_j-\langle x_j,x_i\rangle x_i\bigr).
$$

These two forms are identical for the linear mean field.

---

## 2. The LMS orbit ansatz

Fix a base cloud

$$
p=(p_1,\dots,p_N) \in (S^{d-1})^N.
$$

The finite LMS orbit through $p$ is parametrized by a boost $w \in B^d$ and a rotation $\zeta \in SO(d)$:

$$
x_i=M_w(p_i)\,\zeta^\top.
$$

Define the body-frame field

$$
F_p(w)=\sum_{i=1}^N a_i M_w(p_i),
$$

and include coupling by

$$
Z_{\mathrm{body}}(w)=K F_p(w).
$$

The lab-frame order parameter is

$$
Z=Z_{\mathrm{body}}\zeta^\top.
$$

The lab-frame conformal center is

$$
z=-w\zeta^\top.
$$

With these conventions, the closed finite-dimensional reduced equations are

$$
\boxed{
\dot w=-\frac12(1-|w|^2)Z_{\mathrm{body}}(w)
}
$$

and

$$
\boxed{
\dot\zeta=A\zeta-\zeta\,\alpha\bigl(w,Z_{\mathrm{body}}(w)\bigr).
}
$$

This is the row-vector form of the LMS reduction implemented by `integrate_lms_reduced_euler`. The code also computes the same rotation equation as

$$
\dot\zeta=\bigl(A-\alpha(w\zeta^\top,Z)\bigr)\zeta,
$$

which is equivalent because

$$
\alpha(w\zeta^\top,Z_{\mathrm{body}}\zeta^\top)
=\zeta\,\alpha(w,Z_{\mathrm{body}})\,\zeta^\top.
$$

The important point is that $F_p(w)$ depends on the stored base cloud $p$ and the low-dimensional coordinate $w$, but it does not require evolving the individual $x_i$.

---

## 3. The canonical centered slice

The reduction above depends on a choice of base cloud $p$. Exact Busemann centering chooses a canonical representative of the Mobius orbit.

Define the centered slice

$$
\mathcal S_0=
\left\{p \in (S^{d-1})^N:\sum_{i=1}^N a_i p_i=0\right\}.
$$

Rotations preserve this slice because

$$
\sum_i a_i(p_i\zeta^\top)=\left(\sum_i a_i p_i\right)\zeta^\top.
$$

Boosts do not preserve it. Therefore $\mathcal S_0$ fixes the boost freedom and leaves exactly the rotational freedom. It is a boost gauge, not a complete gauge.

Given an observed cloud $x=(x_i)$, define the canonical deboost center $z(x) \in B^d$ by

$$
\boxed{
\sum_{i=1}^N a_i M_{z(x)}(x_i)=0.
}
$$

Then the canonical centered representative is

$$
\boxed{
p_i^\star(x)=M_{z(x)}(x_i).
}
$$

By construction,

$$
\sum_i a_i p_i^\star(x)=0.
$$

In the widget's reduced coordinate convention, when $\zeta=I$,

$$
\boxed{w=-z.}
$$

So the canonical-gauge module uses $z$ for the deboost parameter, while the LMS reduced state uses $w=-z$ at identity rotation.

Under the usual generic and non-majority hypotheses, for example $a_i>0$, $\sum_i a_i=1$, and $\max_i a_i<1/2$, the Busemann potential below has a unique interior critical point. Numerically, failure of the hypotheses appears as poor centering residuals, unstable centers near $\partial B^d$, or non-unique effective gauges.

---

## 4. The Busemann potential and exact inversion

The canonical center can be found as a critical point of a finite-$N$ Busemann potential. For a fixed observed cloud $x$, define

$$
\Phi_x(z)=\sum_{i=1}^N a_i\log\frac{1-|z|^2}{|z-x_i|^2}.
$$

The Euclidean gradient identity is

$$
\boxed{
\nabla_{\mathrm{euc}}\Phi_x(z)=\frac{2}{1-|z|^2}\sum_i a_i M_z(x_i).
}
$$

Since the Poincare ball metric is conformal to the Euclidean metric with factor $2/(1-|z|^2)$, the hyperbolic gradient is

$$
\boxed{
\nabla_{\mathrm{hyp}}\Phi_x(z)=\frac12(1-|z|^2)\sum_i a_i M_z(x_i).
}
$$

Therefore the exact centering equation

$$
\sum_i a_iM_z(x_i)=0
$$

is equivalent to

$$
\nabla_{\mathrm{euc}}\Phi_x(z)=0
\qquad\text{and}\qquad
\nabla_{\mathrm{hyp}}\Phi_x(z)=0.
$$

This is the exact finite-$N$ inverse problem. It uses the full cloud, not only the centroid. It is therefore different from the Poisson-manifold shrink approximation, which estimates a radius from a continuum closure relation.

The package implementation in `lmsspp.core.canonical_gauge` solves this problem by:

- forming a local finite-$N$ initializer from the barycenter and inertia tensor,
- maximizing $\Phi_x$ in rapidity coordinates so the iterate stays inside $B^d$,
- polishing the residual equation $\sum_i a_iM_z(x_i)=0$ by a small Newton solve,
- returning $z$, $w=-z$, the centered cloud $P=M_z(x)$, and centering diagnostics.

---

## 5. Exact finite-$N$ reconstruction from an arbitrary initial cloud

Given an initial cloud $x^0=(x_i^0)$, compute its canonical center

$$
\sum_i a_iM_{z_0}(x_i^0)=0.
$$

Then define the frozen constants

$$
\boxed{
p_i^0=M_{z_0}(x_i^0).
}
$$

These constants satisfy

$$
\sum_i a_i p_i^0=0.
$$

The reduced initial state is

$$
\boxed{
w_0=-z_0,
\qquad
\zeta_0=I.
}
$$

Because $M_{-z_0}$ is the inverse boost of $M_{z_0}$, the initial cloud is reconstructed exactly by

$$
x_i^0=M_{w_0}(p_i^0).
$$

Now evolve only $(w(t),\zeta(t))$ by the reduced equations in Section 2, using the frozen constants $p^0$ to compute

$$
F_{p^0}(w)=\sum_i a_iM_w(p_i^0).
$$

At any time $t$, reconstruct the full cloud by

$$
\boxed{
x_i(t)=M_{w(t)}(p_i^0)\,\zeta(t)^\top.
}
$$

This is the exact finite-$N$ reduction in operational form. The $N$-dependence is stored in the constants $p_i^0$. The evolved variables have dimension

$$
d+\frac{d(d-1)}2,
$$

independent of $N$.

This is the finite-dimensional analogue of Watanabe-Strogatz reconstruction: the individual particles are not independently integrated; they are reconstructed from a finite list of constants and low-dimensional collective coordinates.

---

## 6. The moving canonical frame

The canonical centered representative can also be computed at each time from the evolving cloud:

$$
p_i^\star(t)=M_{z(t)}(x_i(t)),
\qquad
\sum_i a_i p_i^\star(t)=0.
$$

For a trajectory on one LMS Mobius orbit, the canonical center is

$$
z(t)=-w(t)\zeta(t)^\top.
$$

Using the factorization

$$
x_i(t)=M_{w(t)}(p_i^0)\zeta(t)^\top
=M_{-z(t)}\bigl(p_i^0\zeta(t)^\top\bigr),
$$

we obtain

$$
\boxed{
p_i^\star(t)=p_i^0\zeta(t)^\top.
}
$$

Thus, after exact Busemann centering, the cloud moves only by rotation. The boost component has been removed by the canonical gauge.

In row-vector convention this means

$$
\boxed{
\dot p_i^\star=p_i^\star\Omega^\top
}
$$

for a skew-symmetric matrix $\Omega(t)$. Equivalently, the covariant derivative

$$
D_t p_i^\star=\dot p_i^\star-p_i^\star\Omega^\top
$$

vanishes.

This is the most compact finite-$N$ conservation law: the canonical centered shape is covariantly constant.

---

## 7. Shape invariants

Since $p_i^\star(t)=p_i^0\zeta(t)^\top$, all rotational invariants of the centered cloud are conserved.

The most direct invariant is the Gram matrix

$$
G_{ij}=\langle p_i^\star,p_j^\star\rangle.
$$

Along an exact LMS orbit,

$$
\boxed{
G_{ij}(t)=G_{ij}(0).
}
$$

The weighted inertia tensor is

$$
T(t)=\sum_i a_i p_i^\star(t)^\top p_i^\star(t),
$$

where the expression is the matrix outer product. It transforms by conjugation:

$$
T(t)=\zeta(t)T(0)\zeta(t)^\top.
$$

Therefore

$$
\boxed{
\operatorname{spec} T(t)=\operatorname{spec} T(0).
}
$$

Higher weighted moment tensors

$$
T_m(t)=\sum_i a_i\, p_i^\star(t)^{\otimes m}
$$

also rotate covariantly, so their rotational contractions and spectra give further invariants.

For $d=2$, the same statement contains the classical Watanabe-Strogatz constants: after centering, all phases rotate together, so cross-ratios are conserved. For $d>2$, Gram and moment invariants are the natural higher-dimensional replacement.

---

## 8. The connection term

The centered cloud rotates with a connection determined by the LMS variables. In row-vector convention the predicted moving-frame connection is

$$
\boxed{
\Omega=A+\alpha(z,Z),
}
$$

where

$$
\alpha(z,Z)=Zz^\top-zZ^\top.
$$

Thus

$$
\boxed{
\dot p_i^\star=p_i^\star\bigl(A+\alpha(z,Z)\bigr)^\top.
}
$$

The same connection appears in the $z$-coordinate equation. Under the row-vector convention used in the code,

$$
\boxed{
\dot z=zA^\top+\frac12(1+|z|^2)Z-\langle Z,z\rangle z.
}
$$

The term $\alpha(z,Z)$ is not an extra force. It is the rotational connection induced by changing the boost center. Geometrically, it is present because infinitesimal boosts do not commute. In column-vector notation, if $B_U$ and $B_V$ are infinitesimal boost generators, then their commutator is rotational:

$$
[B_U,B_V](y)=\alpha(U,V)y.
$$

This is the finite-dimensional analogue of Thomas precession: a changing boost frame carries a curvature term in the rotational direction.

---

## 9. Estimating the connection from data

The moving-frame equations give useful diagnostics for simulations and empirical trajectories.

Given a canonical centered trajectory $P(t)=(p_i^\star(t))$, define

$$
T=\sum_i a_i p_i^{\star\top}p_i^\star,
$$

and

$$
C=\sum_i a_i \dot p_i^{\star\top}p_i^\star.
$$

If $\dot p_i^\star=p_i^\star\Omega^\top$, then, in matrix form,

$$
C=\Omega T.
$$

When $T$ is invertible,

$$
\Omega=CT^{-1}.
$$

In rank-deficient or noisy cases, use the pseudoinverse:

$$
Q=CT^+.
$$

Then decompose

$$
\Omega_{\mathrm{data}}=\frac12(Q-Q^\top),
\qquad
S_{\mathrm{strain}}=\frac12(Q+Q^\top).
$$

For exact LMS motion, $S_{\mathrm{strain}}$ should vanish up to numerical error. The skew part should match

$$
\Omega_{\mathrm{pred}}=A+\alpha(z,Z).
$$

The package diagnostics compute:

- centering error $\left|\sum_i a_i p_i^\star\right|$,
- Gram drift $\max_{i,j}|G_{ij}(t)-G_{ij}(0)|$,
- inertia spectrum drift,
- connection mismatch $\|\Omega_{\mathrm{data}}-\Omega_{\mathrm{pred}}\|$,
- covariant derivative error $\left(\sum_i a_i|\dot p_i^\star-p_i^\star\Omega_{\mathrm{pred}}^\top|^2\right)^{1/2}$,
- strain error $\|S_{\mathrm{strain}}\|$,
- optional $z$-equation error.

These are implemented in `moving_frame_diagnostics`.

---

## 10. Local source structure at the centered representative

Let $P=(p_i)$ be centered:

$$
\sum_i a_i p_i=0.
$$

Define

$$
F_P(u)=\sum_i a_iM_u(p_i),
\qquad
T=\sum_i a_i p_i^\top p_i.
$$

For small $u$, in row-vector convention,

$$
F_P(u)=-2u(I-T)+O(|u|^2).
$$

The reduced LMS boost equation near $u=0$ is

$$
\dot u=-\frac12(1-|u|^2)F_P(u),
$$

so

$$
\boxed{
\dot u=u(I-T)+O(|u|^2).
}
$$

Thus the local source exponents at the centered representative are the eigenvalues of

$$
\boxed{I-T.}
$$

This gives a direct finite-$N$ stability diagnostic from the canonical shape constants.

---

## 11. Euler-Sundman form

For a fixed centered cloud $P$, define the Busemann potential

$$
\Phi_P(w)=\sum_i a_i\log\frac{1-|w|^2}{|w-p_i|^2}.
$$

As above,

$$
\nabla_{\mathrm{euc}}\Phi_P(w)=\frac{2}{1-|w|^2}F_P(w).
$$

The physical-time reduced equation is

$$
\dot w=-\frac12(1-|w|^2)F_P(w).
$$

Introduce Euler-Sundman time $\tau$ by

$$
\frac{dt}{d\tau}=\frac{4}{(1-|w|^2)^2}.
$$

Then

$$
\boxed{
\frac{dw}{d\tau}=-\nabla_{\mathrm{euc}}\Phi_P(w).
}
$$

The conservative-lift diagnostic used in the code is

$$
\boxed{
H_{\mathrm{ES}}(w,w_\tau)=\frac12|w_\tau|^2-\frac12|\nabla_{\mathrm{euc}}\Phi_P(w)|^2.
}
$$

On the LMS gradient branch,

$$
H_{\mathrm{ES}}=0.
$$

This form is mainly useful for diagnostics and theoretical interpretation. The widget and core integrator usually evolve the physical-time equation directly.

---

## 12. What is exact, and what is not being assumed

The exact reduction described here is finite-$N$. It assumes only that the points remain on one LMS Mobius orbit generated by the identical-$A$ sphere dynamics with linear order parameter. It does not assume:

- a hydrodynamic limit,
- a continuum Poisson kernel,
- the Poisson shrink law $|Z|/K=f_d(|z|)|z|$,
- or that the centroid alone determines the reduced center.

The Poisson closure is still useful as a special continuum reference family. It is not needed for exact finite-$N$ reconstruction.

The exact finite-$N$ algorithm is:

1. Given $x^0$, solve $\sum_i a_iM_{z_0}(x_i^0)=0$.
2. Store $p_i^0=M_{z_0}(x_i^0)$.
3. Set $w_0=-z_0$ and $\zeta_0=I$.
4. Evolve

$$
\dot w=-\frac12(1-|w|^2)K\sum_i a_iM_w(p_i^0),
$$

$$
\dot\zeta=A\zeta-\zeta\alpha\left(w,K\sum_i a_iM_w(p_i^0)\right).
$$

5. Reconstruct

$$
x_i(t)=M_{w(t)}(p_i^0)\zeta(t)^\top.
$$

The canonical moving-frame theorem is:

$$
\boxed{
M_{z(t)}(x_i(t))=p_i^0\zeta(t)^\top.
}
$$

Consequently, the centered shape is conserved up to rotation, the Gram matrix is conserved, the inertia spectrum is conserved, and the rotation of the centered frame is governed by

$$
\boxed{
\Omega=A+\alpha(z,Z).
}
$$

This is the mathematical structure implemented by `lmsspp.core.canonical_gauge`.
