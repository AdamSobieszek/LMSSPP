# LMS Exact Busemann Inversion Note

This note reformulates a finite-$N$ construction that is *implicit* in the LMS geometry but not stated explicitly in the paper:

1. an **exact inverse problem**: given a weighted cloud $x_1,\dots,x_N \in S^2$, recover a canonical reduced LMS center;
2. a **state-construction problem**: given a desired radius/direction for $w$, build a reduced LMS state on the same exact finite-$N$ Möbius orbit.

The first problem is solved by an exact finite-$N$ **Busemann inversion**.  
The second is solved by an exact **orbit projection**.


---

## 0. The LMS ingredients we use

The starting point is the reduced LMS ansatz.

> “Now suppose all the terms $A_i$ in (1) are equal. Fix a base point $p=(p_1,\dots,p_N)\in X$. Then if the points $p_i$ are in sufficiently general position, every element in the $G$-orbit of $p$ can be expressed uniquely as $gp$ for some $g\in G$, with parameters $w,z$ and $\zeta$.”

> “The reduced equations are$$ \dot w=-\frac12(1-|w|^2)\,\zeta^{-1}Z,$$ $$ \dot\zeta=(A-\alpha(\zeta w,Z))\zeta,$$ with $A$ and $Z$ evaluated at $\zeta M_w(p)$.”

For linear order parameter $Z=\sum_i a_i x_i$, LMS then prove:

> “The advantage of the $\dot w$ equation (5) is that for an order parameter function of the form$$ Z=\sum_{i=1}^N a_i x_i,$$ $\zeta$ drops out of the $\dot w$ equation and we get the reduced equation$$ \dot w=-\frac12(1-|w|^2)Z(M_w(p)).$$”

And they identify the hyperbolic potential

> “Next, we show that the hyperbolic potential for $V$, up to an additive constant, is given by$$ \Phi(w)=\sum_{i=1}^N a_i \log \frac{1-|w|^2}{|w-p_i|^2}.$$”

with hyperbolic gradient identity

> “Hence we see that$$ \nabla_{\mathrm{hyp}}\Phi(w)=\frac12(1-|w|^2)Z(M_w(p))=-V(w).$$”

The present note uses exactly these LMS formulas, but applies them to the **observed cloud itself** as boundary data for an inverse problem.

---

## 1. Reduced coordinates and the gauge issue

For a base cloud $p_1,\dots,p_N \in S^{d-1}$ and reduced state $(w,\zeta)$, LMS write
$$
x_i=\zeta M_w(p_i).
$$

In the widget initialization we fix
$$
\zeta_0=I,
$$
so initially
$$
x_i(0)=M_{w_0}(p_i).
$$

Since LMS also have
$$
z=-\zeta w,
$$
we get at initialization
$$
z_0=-w_0.
$$

### Proposition 1 (Gauge dependence of $w$)
The reduced center $w$ is **not** an intrinsic observable of a cloud by itself. It depends on the chosen base representative $p$ on the Möbius orbit.

#### Proof
If one chooses $p_i=x_i(0)$, then the initial group element is the identity, hence
$$
w_0=0,\qquad z_0=0.
$$
But if one chooses a different base representative on the same orbit, the corresponding reduced parameters change. LMS explicitly prove that if $p'=M(p)$, then the corresponding reduced parameter transforms as
$$
w'=M(w).
$$
So $w$ is an orbit coordinate, not a raw statistic of the boundary cloud. $\square$

---

## 2. The finite-$N$ inverse problem

Let
$$
x_1,\dots,x_N\in S^{d-1},\qquad a_i>0,\qquad \sum_{i=1}^N a_i=1.
$$

Define the empirical measure
$$
\mu_N=\sum_{i=1}^N a_i \delta_{x_i},
$$
and its first moment
$$
m[\mu_N]=\sum_{i=1}^N a_i x_i.
$$

The old widget inversion used the scalar surrogate
$$
|m[\mu_N]| \approx f_d(r)\,r,\qquad r=|w|,
$$
motivated by the continuum Poisson orbit. That is useful, but it is not an exact finite-$N$ inversion rule for a general cloud.

The exact finite-$N$ inverse problem is instead:

> Given an observed weighted cloud $x_1,\dots,x_N$, choose a canonical reduced center and a canonical deboosted representative on its Möbius orbit.

---

## 3. Canonical finite-$N$ gauge condition

We seek $w_* \in B^d$ and base points $p_i \in S^{d-1}$ such that
$$
x_i=M_{w_*}(p_i),
$$
and the deboosted base cloud has zero weighted barycenter:
$$
\sum_{i=1}^N a_i p_i=0.
$$

Since $M_w^{-1}=M_{-w}$, this is equivalent to
$$
p_i=M_{-w_*}(x_i),
$$
and therefore to the nonlinear equation
$$
\sum_{i=1}^N a_i M_{-w_*}(x_i)=0.
$$

### Definition 2 (Exact finite-$N$ inversion condition)
A reduced center $w_*$ is an **exact finite-$N$ inverse center** of the weighted cloud $(x_i,a_i)$ if
$$
\sum_{i=1}^N a_i M_{-w_*}(x_i)=0.
$$

This says: deboost the observed cloud by the unknown reduced center, and require the resulting base representative to have zero weighted barycenter.

---

## 4. Sign-clean variable $v=-w$

It is convenient to remove the sign convention and define
$$
v:=-w.
$$

Then the exact inversion equation becomes
$$
R_x(v):=\sum_{i=1}^N a_i M_v(x_i)=0.
$$

Once $v_*$ is found, we return to the widget convention via
$$
w_*=-v_*.
$$

Because $\zeta_0=I$ in the widget, we also have
$$
z_*=-w_*=v_*.
$$

### Corollary 3 (Sign interpretation)
In the initialization gauge $\zeta_0=I$:

- $v_*$ points in the physical dipole direction,
- $w_*=-v_*$ points in the opposite reduced-coordinate direction,
- $z_*=v_*$ and $|z_*|=|w_*|$.

---

## 5. Exact finite-$N$ Busemann cloud potential

LMS define the hyperbolic potential
$$
\Phi(w)=\sum_{i=1}^N a_i \log \frac{1-|w|^2}{|w-p_i|^2}.
$$

For inversion, we use the observed cloud itself as the boundary anchor set and define
$$
\Phi_x(v):=\sum_{i=1}^N a_i \log \frac{1-|v|^2}{|v-x_i|^2}.
$$

If
$$
\mathcal B_\xi(v):=\log \frac{|v-\xi|^2}{1-|v|^2}
$$
denotes the Busemann function based at $\xi\in S^{d-1}$, then
$$
\Phi_x(v)=-\sum_{i=1}^N a_i\,\mathcal B_{x_i}(v).
$$

So the exact cloud potential is the negative weighted sum of Busemann functions attached to the observed cloud.

---

## 6. Critical points solve the inverse problem

### Proposition 4 (LMS gradient identity for the observed cloud)
For the cloud potential $\Phi_x$, one has
$$
\nabla_{\mathrm{hyp}}\Phi_x(v)
=
\frac12(1-|v|^2)\sum_{i=1}^N a_i M_v(x_i)
=
\frac12(1-|v|^2)R_x(v).
$$

#### Proof
This is exactly the LMS computation, with the observed cloud $x_i$ substituted for the fixed anchors $p_i$. LMS prove
$$
\nabla_{\mathrm{hyp}}\Phi(w)=\frac12(1-|w|^2)\sum_i a_i M_w(p_i),
$$
so replacing $(w,p_i)$ by $(v,x_i)$ yields the stated formula. $\square$

### Corollary 5 (Exact inversion as a critical-point equation)
A point $v_*\in B^d$ solves the exact inversion equation
$$
R_x(v_*)=0
$$
if and only if it is a critical point of $\Phi_x$:
$$
\nabla_{\mathrm{hyp}}\Phi_x(v_*)=0.
$$

Hence the finite-$N$ inverse problem is equivalent to finding a critical point of the cloud’s weighted Busemann potential.

---

## 7. Existence and uniqueness under LMS assumptions

The exact inversion is not automatically unique for arbitrary weighted clouds. The right hypotheses are the same ones LMS use for their finite-$N$ gradient analysis.

### Assumptions
We assume

1. $a_i>0$ for all $i$,
2. $\sum_i a_i=1$,
3. $\max_i a_i<\tfrac12$,
4. the points $x_i$ are distinct and in generic position.

These are the LMS “positive weights, no majority cluster” hypotheses.

### Theorem 6 (Exact finite-$N$ Busemann inversion)
Under the assumptions above, the cloud potential
$$
\Phi_x(v)=\sum_{i=1}^N a_i \log \frac{1-|v|^2}{|v-x_i|^2}
$$
has a unique critical point $v_*\in B^d$. Equivalently, the residual equation
$$
R_x(v)=\sum_{i=1}^N a_i M_v(x_i)=0
$$
has a unique solution in $B^d$.

That unique point is the unique maximizer of $\Phi_x$ and defines the canonical exact inverse center
$$
w_*=-v_*,
\qquad
p_i=M_{v_*}(x_i)=M_{-w_*}(x_i),
$$
with
$$
\sum_{i=1}^N a_i p_i=0.
$$

#### Proof
Apply the LMS hyperbolic gradient theory to the cloud $x_i$ regarded as the anchor set. LMS prove, for the same weight assumptions, that the corresponding reduced flow is a hyperbolic gradient system, that the potential tends to $-\infty$ at the boundary, and that the interior fixed point is unique. By Proposition 4, critical points of $\Phi_x$ are exactly zeros of $R_x$. Since the unique interior critical point is the only one and the boundary value tends to $-\infty$, it is the unique maximizer. The formula for $p_i$ follows from $M_w^{-1}=M_{-w}$. $\square$

### Interpretation
The exact inverse center is the unique Möbius deboost that sends the observed weighted cloud to the zero-barycenter gauge.

---

## 8. Möbius equivariance of the exact center

The exact center is canonical because it is equivariant under ambient hyperbolic isometries.

### Theorem 7 (Möbius equivariance)
Let $g\in \mathrm{Isom}^+(B^d)$ and let
$$
gx := (g x_1,\dots,g x_N).
$$
If $v_*(x)$ denotes the unique exact Busemann center of $x$, then
$$
v_*(gx)=g(v_*(x)).
$$

#### Proof
The Busemann cocycle implies that under an isometry $g$, the cloud potential transforms by an additive constant:
$$
\Phi_{gx}(gv)=\Phi_x(v)+C_{g,x},
$$
where $C_{g,x}$ is independent of $v$. Therefore critical points are carried to critical points under $g$. By uniqueness, the critical point of $\Phi_{gx}$ must be $g(v_*(x))$. $\square$

### Remark
This is the formal sense in which the inversion is canonical: it is not tied to a Euclidean coordinate chart, but is intrinsic to the Möbius geometry of the weighted cloud.

---

## 9. Local finite-$N$ initializer

The exact inversion is nonlinear, but it has a clean finite-$N$ small-center approximation.

Define
$$
\mu := \sum_{i=1}^N a_i x_i,
\qquad
C := \sum_{i=1}^N a_i x_i x_i^\top.
$$

For small $v$, the Möbius map expands as
$$
M_v(x_i) \approx x_i - 2v + 2\langle v,x_i\rangle x_i.
$$

Summing with weights gives
$$
R_x(v)
\approx
\mu - 2v + 2Cv
=
\mu - 2(I-C)v.
$$

### Proposition 8 (Finite-$N$ local inversion formula)
If $I-C$ is invertible, then near the incoherent regime the exact inverse center satisfies
$$
v_* \approx \frac12 (I-C)^{-1}\mu.
$$

#### Proof
Set the linearized residual to zero:
$$
\mu - 2(I-C)v \approx 0.
$$
Solving for $v$ gives the formula. $\square$

### Remark
This is a genuine finite-$N$ initializer. Unlike the Poisson shrink law, it depends on both:

- the first moment $\mu$,
- the second moment matrix $C$.

In implementation one should regularize:
$$
v^{(0)}=\frac12 (I-C+\lambda I)^{-1}\mu,
\qquad \lambda>0 \text{ small}.
$$

---

## 10. Exact solver in rapidity coordinates

Since the exact center must lie in the open ball $B^d$, it is convenient to optimize in unconstrained rapidity coordinates $y\in \mathbb R^d$:
$$
v(y)=
\begin{cases}
\dfrac{\tanh |y|}{|y|}\,y,& y\neq 0,\\
0,& y=0.
\end{cases}
$$

Then $v(y)\in B^d$ automatically, and one may optimize
$$
\widetilde\Phi_x(y):=\Phi_x(v(y))
$$
over all $y\in \mathbb R^d$.

### Algorithm 9 (Exact Busemann inversion)
Given weighted cloud $(x_i,a_i)$:

1. compute the regularized local initializer
   $$
   v^{(0)}=\frac12 (I-C+\lambda I)^{-1}\mu;
   $$
2. convert $v^{(0)}$ to rapidity coordinates $y^{(0)}$;
3. maximize $\widetilde\Phi_x(y)=\Phi_x(v(y))$ with LBFGS or another smooth optimizer;
4. stop when the exact residual
   $$
   |R_x(v)|=\left|\sum_{i=1}^N a_i M_v(x_i)\right|
   $$
   is below tolerance;
5. output
   $$
   v_*=v(y_*),\qquad w_*=-v_*,\qquad p_i=M_{v_*}(x_i).
   $$

This is exact up to numerical optimization tolerance.

---

## 11. Exact recovery of the reduced LMS state

### Corollary 10 (Canonical deboosted representative)
Let $v_*$ be the exact Busemann center. Then
$$
w_*=-v_*,
\qquad
p_i=M_{v_*}(x_i)=M_{-w_*}(x_i)
$$
satisfy
$$
x_i=M_{w_*}(p_i),
\qquad
\sum_{i=1}^N a_i p_i=0.
$$

So the observed cloud decomposes into:

- an exact finite-$N$ reduced center $w_*$,
- a canonically deboosted base representative $p$.

This is the precise finite-$N$ inverse-construction layer missing from the original LMS exposition.

---

## 12. Radius control is a second problem

The widget also has a user-facing target radius
$$
|w|=r_{\mathrm{target}}
$$
and target dipole axis $u\in S^2$.

For non-Poisson initializers, the cloud generator creates a boundary cloud $x_i$ with some intended structure, but after exact inversion one generally obtains
$$
|w_{\mathrm{exact}}|\neq r_{\mathrm{target}}.
$$

This is expected. The exact Busemann center depends on the full cloud, whereas the UI radius is an external constraint.

So we must distinguish two problems:

1. **exact inversion** from the cloud;
2. **exact placement** of the resulting deboosted base cloud at the desired UI radius.

The second must be solved by moving along the same Möbius orbit, not by altering the inverse problem.

---

## 13. Exact orbit projection to a prescribed radius

Suppose exact inversion has produced
$$
w_{\mathrm{exact}},
\qquad
p_i=M_{-w_{\mathrm{exact}}}(x_i),
\qquad
\sum_{i=1}^N a_i p_i=0.
$$

Now choose the user-prescribed target reduced center
$$
w_{\mathrm{target}}=-r_{\mathrm{target}}\,u.
$$

Define the corrected cloud by exact orbit placement:
$$
x_i^{\mathrm{target}}:=M_{w_{\mathrm{target}}}(p_i).
$$

This does not modify the exact deboosted base cloud. It only selects a different point on the same Möbius orbit.

### Theorem 11 (Exact orbit projection)
Let
$$
v_{\mathrm{target}}:=-w_{\mathrm{target}}.
$$
Then
$$
\sum_{i=1}^N a_i M_{v_{\mathrm{target}}}(x_i^{\mathrm{target}})=0.
$$

Hence $v_{\mathrm{target}}$ is an exact critical point of the Busemann inverse problem for the corrected cloud.

If, moreover, the exact inverse center is unique for the corrected cloud, then re-inversion returns exactly $v_{\mathrm{target}}$, equivalently $w_{\mathrm{target}}$.

#### Proof
Using $M_{-w}=M_w^{-1}$,
$$
M_{v_{\mathrm{target}}}(x_i^{\mathrm{target}})
=
M_{-w_{\mathrm{target}}}(M_{w_{\mathrm{target}}}(p_i))
=
p_i.
$$
Therefore
$$
\sum_{i=1}^N a_i M_{v_{\mathrm{target}}}(x_i^{\mathrm{target}})
=
\sum_{i=1}^N a_i p_i
=
0.
$$
So $v_{\mathrm{target}}$ is a critical point of the corrected cloud’s inversion problem. Under uniqueness, it is the unique one, hence re-inversion returns it. $\square$

### Remark
This is why the radius correction is exact: it is not a heuristic deformation of the inverse solve, but an exact move on the same Möbius orbit.

---

## 14. Final two-stage algorithm

### Algorithm 12 (Exact inversion + exact orbit projection)

#### Stage A: Exact finite-$N$ inversion
Given weighted cloud $x_i$:

1. compute the local initializer
   $$
   v^{(0)}=\frac12 (I-C+\lambda I)^{-1}\mu;
   $$
2. solve the exact Busemann inverse problem
   $$
   \sum_i a_i M_v(x_i)=0
   $$
   by maximizing
   $$
   \Phi_x(v)=\sum_i a_i \log \frac{1-|v|^2}{|v-x_i|^2};
   $$
3. obtain
   $$
   v_*,\qquad w_*=-v_*,\qquad p_i=M_{v_*}(x_i).
   $$

This produces the canonical deboosted base cloud.

#### Stage B: Exact radius-controlled orbit placement
If the widget prescribes target axis $u$ and target radius $r_{\mathrm{target}}$, set
$$
w_{\mathrm{target}}=-r_{\mathrm{target}}u,
$$
and place the base cloud on the same exact orbit by
$$
x_i^{\mathrm{target}}=M_{w_{\mathrm{target}}}(p_i).
$$

The reduced state passed to the LMS integrator is then
$$
(w_0,p)=(w_{\mathrm{target}},p).
$$

Under uniqueness, exact re-inversion of $x_i^{\mathrm{target}}$ returns $w_{\mathrm{target}}$.

---

## 15. Relation to the old Poisson-shrink inversion

The old method uses only the weighted centroid
$$
m=\sum_{i=1}^N a_i x_i,
$$
chooses direction
$$
w \parallel -m,
$$
and estimates radius by solving
$$
|m| \approx f_d(r)\,r.
$$

### Comparison

**Poisson shrink**
- first-moment surrogate,
- continuum closure inspired,
- cheap,
- not exact for arbitrary finite clouds.

**Exact Busemann**
- full-cloud inversion,
- finite-$N$ Möbius-geometric,
- uses the LMS potential itself,
- exact up to numerical tolerance.

Both are useful:
- `poisson_shrink` for speed and backward compatibility,
- `busemann_exact` for exact finite-$N$ reconstruction.

---

## 16. Mathematical status

The LMS paper gives:

- the Möbius reduced dynamics,
- the LMS hyperbolic potential,
- the hyperbolic gradient-flow interpretation,
- and the finite-$N$ uniqueness theory for the linear-order-parameter gradient system.

What it does **not** state explicitly is the following inverse-construction layer:

1. use the observed cloud itself as boundary anchor data for an LMS/Busemann potential;
2. solve the resulting finite-$N$ inverse problem
   $$
   \sum_i a_i M_{-w}(x_i)=0;
   $$
3. use the deboosted base representative thus obtained as the exact orbit representative for further radius-controlled placement.

So the present note should be read as a finite-$N$ inverse-construction theorem built directly on top of LMS geometry.

---

## 17. Final statement

### Theorem 13 (Exact finite-$N$ Busemann inversion with exact radius control)
Under the LMS positivity and non-majority assumptions, every generic weighted cloud $x_1,\dots,x_N\in S^{d-1}$ admits a unique exact Busemann inverse center $w_*\in B^d$ characterized by
$$
\sum_{i=1}^N a_i M_{-w_*}(x_i)=0.
$$

The associated deboosted representative
$$
p_i=M_{-w_*}(x_i)
$$
is the unique zero-barycenter representative on the same Möbius orbit:
$$
\sum_{i=1}^N a_i p_i=0.
$$

If one further prescribes a target radius/direction through
$$
w_{\mathrm{target}}=-r_{\mathrm{target}}u,
$$
then the corrected cloud
$$
x_i^{\mathrm{target}}=M_{w_{\mathrm{target}}}(p_i)
$$
lies on the same exact Möbius orbit and has $w_{\mathrm{target}}$ as an exact inverse center. Under uniqueness, exact re-inversion returns precisely $w_{\mathrm{target}}$.

### One-line summary
The complete construction is
$$
\boxed{
\text{exact finite-}N\text{ inversion} \;+\; \text{exact orbit projection}.
}
$$
