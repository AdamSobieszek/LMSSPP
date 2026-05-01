Below is a computational write-up you can adapt directly into implementation notes. I will use a sign convention that avoids ambiguity.

Let

[
c(x)\in B^d
]

denote the **canonical deboost parameter**, defined by

[
p_i^\star=M_{c(x)}(x_i),
\qquad
\sum_i a_i p_i^\star=0.
]

In your earlier notation, if you define (w_\star) by

[
p_i^\star=M_{-w_\star}(x_i),
]

then

[
\boxed{c=-w_\star.}
]

Everything below is written in the (c)-convention because it makes the LMS (z)-coordinate identification direct:

[
\boxed{z=c.}
]

---

# 1. Core numerical objects

Given a weighted cloud

[
x=(x_1,\dots,x_N)\in(S^{d-1})^N,
\qquad
a_i>0,\qquad
\sum_i a_i=1,
]

define the LMS boost

[
M_c(y)
======

\frac{(1-|c|^2)y-(1-2\langle c,y\rangle+|y|^2)c}
{1-2\langle c,y\rangle+|c|^2|y|^2}.
]

For (y\in S^{d-1}), this simplifies to

[
M_c(y)
======

\frac{(1-|c|^2)(y-c)}{|y-c|^2}-c.
]

The basic residual is

[
\boxed{
B_x(c):=\sum_{i=1}^N a_i M_c(x_i).
}
]

The canonical gauge is the unique solution

[
\boxed{
B_x(c_\star)=0.
}
]

The normalized cloud is

[
\boxed{
p_i^\star=M_{c_\star}(x_i).
}
]

The first diagnostic should always be

[
\boxed{
\left|\sum_i a_i p_i^\star\right|\approx 0.
}
]

---

# 2. Busemann potential for root-finding

The corresponding potential is

[
\Phi_x(c)
=========

\sum_{i=1}^N a_i
\log\frac{1-|c|^2}{|c-x_i|^2}.
]

It satisfies

[
\boxed{
\nabla_{\mathrm{hyp}}\Phi_x(c)
==============================

\frac12(1-|c|^2)B_x(c).
}
]

Equivalently, using the Euclidean gradient,

[
\boxed{
\nabla_{\mathrm{euc}}\Phi_x(c)
==============================

\frac{2}{1-|c|^2}B_x(c).
}
]

So you can find (c_\star) in either of two equivalent ways:

[
B_x(c)=0,
]

or

[
\nabla_{\mathrm{euc}}\Phi_x(c)=0.
]

Because (\Phi_x) is hyperbolically concave under the usual non-majority assumptions, this critical point is unique. Numerically, that makes the problem well-conditioned unless the cloud is close to a degenerate/majority configuration.

---

# 3. Recommended helper functions

A minimal helper layer should include:

```python
boost(c, y)
```

Returns (M_c(y)).

```python
boost_cloud(c, X)
```

Returns (M_c(x_i)) for all cloud points.

```python
residual(c, X, a)
```

Returns

[
B_x(c)=\sum_i a_i M_c(x_i).
]

```python
busemann_potential(c, X, a)
```

Returns

[
\Phi_x(c)=\sum_i a_i\log\frac{1-|c|^2}{|c-x_i|^2}.
]

```python
canonical_center(X, a)
```

Solves

[
B_x(c)=0
]

and returns (c_\star).

```python
canonical_cloud(X, a)
```

Returns

[
c_\star,\qquad p_i^\star=M_{c_\star}(x_i).
]

This is the central gauge-fixing routine.

---

# 4. Canonical moving-frame quantities

Once you have

[
p_i^\star=M_{c_\star}(x_i),
]

define the centered barycenter

[
\boxed{
\bar p^\star:=\sum_i a_i p_i^\star.
}
]

This should be numerically zero.

Next define the Gram matrix

[
\boxed{
G_{ij}:=\langle p_i^\star,p_j^\star\rangle.
}
]

This is the most direct finite-(N) shape invariant.

For a trajectory (x(t)) on a single LMS/Möbius orbit, the canonical centered clouds satisfy

[
p_i^\star(t)=\zeta(t)p_i^\star(0)
]

for some rotation (\zeta(t)\in SO(d)). Therefore

[
\boxed{
G_{ij}(t)=G_{ij}(0).
}
]

Computational test:

```python
G = canonical_gram(P)
```

where

[
P=(p_i^\star).
]

Then track

[
\boxed{
\Delta_G(t)
===========

\max_{i,j}|G_{ij}(t)-G_{ij}(0)|.
}
]

This should remain near numerical precision for an exact LMS simulation.

---

# 5. Weighted inertia tensor

Define

[
\boxed{
T:=\sum_i a_i p_i^\star\otimes p_i^\star.
}
]

This is the weighted covariance/inertia tensor of the centered representative.

Since

[
p_i^\star(t)=\zeta(t)p_i^\star(0),
]

we have

[
T(t)=\zeta(t)T(0)\zeta(t)^{-1}.
]

Therefore the eigenvalues of (T) are conserved:

[
\boxed{
\operatorname{spec}T(t)=\operatorname{spec}T(0).
}
]

Computational helpers:

```python
inertia_tensor(P, a)
```

returns

[
T=\sum_i a_i p_i p_i^\top.
]

```python
inertia_spectrum(P, a)
```

returns eigenvalues of (T).

Diagnostics:

[
\Delta_{\mathrm{spec}}(t)
=========================

\left|
\operatorname{sort}(\lambda(T(t)))
----------------------------------

\operatorname{sort}(\lambda(T(0)))
\right|.
]

In (d=2), (\operatorname{tr}T=1), so there is essentially one anisotropy invariant.

---

# 6. Moving-frame connection from data

For a canonically centered trajectory

[
p_i^\star(t)=M_{c(t)}(x_i(t)),
]

the moving-frame theorem says

[
\boxed{
\dot p_i^\star=\Omega p_i^\star,
\qquad
\Omega\in\mathfrak{so}(d).
}
]

This is a very strong testable prediction.

Given numerical derivatives (\dot p_i^\star), define

[
T=\sum_i a_i p_i^\star\otimes p_i^\star,
]

and

[
C=\sum_i a_i \dot p_i^\star\otimes p_i^\star.
]

If

[
\dot p_i^\star=\Omega p_i^\star,
]

then

[
C=\Omega T.
]

So if (T) is invertible,

[
\boxed{
\Omega_{\mathrm{data}}=C T^{-1}.
}
]

With noisy data or rank issues, use a pseudoinverse and skew projection:

[
\boxed{
\Omega_{\mathrm{data}}
======================

\operatorname{skew}\left(C T^+\right),
}
]

where

[
\operatorname{skew}(Q)=\frac12(Q-Q^\top).
]

The symmetric part

[
\boxed{
S_{\mathrm{strain}}
===================

\operatorname{sym}\left(C T^+\right)
}
]

should vanish for exact Möbius-orbit motion. Thus

[
\boxed{
|S_{\mathrm{strain}}|
}
]

is a useful numerical diagnostic for departure from exact LMS/Möbius dynamics.

Recommended helpers:

```python
estimate_connection(P, Pdot, a)
```

returns

[
\Omega_{\mathrm{data}}.
]

```python
strain_error(P, Pdot, a)
```

returns

[
\left|\operatorname{sym}(C T^+)\right|.
]

---

# 7. Predicted LMS connection

For the LMS dynamics

[
\dot x_i=A x_i+Z-\langle Z,x_i\rangle x_i,
]

with

[
Z=\sum_i a_i x_i,
]

the canonical boost coordinate is

[
z=c_\star.
]

The LMS lifted frame equation gives

[
\boxed{
\Omega_{\mathrm{pred}}=A+\alpha(z,Z).
}
]

Here

[
\alpha(z,Z)y
============

\langle z,y\rangle Z-\langle Z,y\rangle z.
]

As a matrix,

[
\boxed{
\alpha(z,Z)=Zz^\top-zZ^\top.
}
]

So computationally:

```python
alpha_matrix(z, Z):
    return np.outer(Z, z) - np.outer(z, Z)
```

Then

```python
Omega_pred = A + alpha_matrix(z, Z)
```

The key connection test is

[
\boxed{
\Omega_{\mathrm{data}}\approx \Omega_{\mathrm{pred}}.
}
]

Even more directly, test the covariant derivative:

[
\boxed{
D_t p_i^\star
=============

\dot p_i^\star-\Omega_{\mathrm{pred}}p_i^\star
\approx 0.
}
]

Diagnostic:

[
\boxed{
E_{\mathrm{cov}}(t)
===================

\left(
\sum_i a_i
\left|
\dot p_i^\star-\Omega_{\mathrm{pred}}p_i^\star
\right|^2
\right)^{1/2}.
}
]

This is probably the most important test of the gauge-theoretic interpretation.

---

# 8. Testing the (z)-equation directly

The canonical boost coordinate

[
z(t)=c_\star(x(t))
]

should satisfy

[
\boxed{
\dot z
======

Az+\frac12(1+|z|^2)Z-\langle Z,z\rangle z.
}
]

So from your simulation:

1. compute (z(t)) by canonical centering;
2. finite-difference (\dot z(t));
3. compute

[
\dot z_{\mathrm{pred}}
======================

Az+\frac12(1+|z|^2)Z-\langle Z,z\rangle z;
]

4. compare.

Diagnostic:

[
\boxed{
E_z(t)
======

|\dot z_{\mathrm{num}}-\dot z_{\mathrm{pred}}|.
}
]

This test checks that the canonical inverse center evolves exactly as the LMS (z)-coordinate.

---

# 9. Canonical cloud conservation tests

For every time step:

1. Compute canonical center (z(t)=c_\star(x(t))).
2. Compute canonical cloud

[
p_i^\star(t)=M_{z(t)}(x_i(t)).
]

Then compute:

### Centering error

[
\boxed{
E_{\mathrm{center}}(t)
======================

\left|
\sum_i a_i p_i^\star(t)
\right|.
}
]

### Gram conservation

[
\boxed{
E_{\mathrm{Gram}}(t)
====================

|G(t)-G(0)|_{\max}.
}
]

### Inertia spectrum conservation

[
\boxed{
E_T(t)
======

|\lambda(T(t))-\lambda(T(0))|.
}
]

### Covariant-frame error

[
\boxed{
E_{\mathrm{cov}}(t)
===================

\left(
\sum_i a_i
|\dot p_i^\star-\Omega_{\mathrm{pred}}p_i^\star|^2
\right)^{1/2}.
}
]

### Connection mismatch

[
\boxed{
E_\Omega(t)
===========

|\Omega_{\mathrm{data}}-\Omega_{\mathrm{pred}}|_F.
}
]

These should all stay small for a correct implementation and exact LMS dynamics.

---

# 10. Suggested code architecture

A clean module structure could look like this (Please first locate all existing helper functions to deduce which modules are not needed, then rebuild the code architecture)

## `mobius.py`

Core geometry.

```python
boost(c, y)
boost_cloud(c, X)
residual(c, X, a)
busemann_potential(c, X, a)
```

## `canonical_gauge.py`

Gauge fixing.

```python
canonical_center(X, a)
canonical_cloud(X, a)
canonical_state(X, a)
```

where

```python
canonical_state
```

returns a dictionary like:

```python
{
    "z": c_star,
    "w": -c_star,
    "P": P_star,
    "center_error": norm(sum_i a_i P_i)
}
```

## `invariants.py`

Shape invariants.

```python
gram_matrix(P)
inertia_tensor(P, a)
inertia_spectrum(P, a)
higher_moment_tensor(P, a, order)
```

## `connection.py`

Moving-frame quantities.

```python
alpha_matrix(z, Z)
predicted_connection(z, Z, A)
estimate_connection(P, Pdot, a)
covariant_derivative_error(P, Pdot, Omega, a)
strain_error(P, Pdot, a)
```

## `diagnostics.py`

Full trajectory tests.

```python
trajectory_canonical_data(X_series, a)
gram_conservation_error(G_series)
inertia_spectrum_error(T_series)
z_equation_error(z_series, X_series, A_series, a, dt)
connection_error(P_series, z_series, X_series, A_series, a, dt)
```

---

# 11. Practical root-finding notes

The canonical center satisfies

[
B_x(c)=0,
\qquad |c|<1.
]

Good strategies:

### Option A: unconstrained ball parametrization

Use

[
c=\tanh r,u,
]

or more generally

[
c=\frac{y}{\sqrt{1+|y|^2}},
\qquad y\in\mathbb R^d.
]

Then solve in (y) unconstrained.

### Option B: projected Newton or trust-region solve

Use residual

[
B_x(c)
]

and enforce (|c|<1).

### Option C: maximize (\Phi_x)

Because (\Phi_x) is concave, maximize

[
\Phi_x(c)
]

inside the ball. This is often numerically more stable than solving the residual directly.

The Euclidean gradient is

[
\nabla_{\mathrm{euc}}\Phi_x(c)
==============================

\frac{2}{1-|c|^2}B_x(c).
]

So you can provide the optimizer with an exact gradient once `residual` is implemented.

---

# 12. Important distinction: moving center vs frozen flow

There are two different computational experiments.

## A. Moving-orbit experiment

You simulate

[
\dot x_i=A x_i+Z-\langle Z,x_i\rangle x_i.
]

At each time, compute (z(t)) and (p^\star(t)). Then test:

[
p_i^\star(t)=\zeta(t)p_i^\star(0),
]

[
G_{ij}(t)=G_{ij}(0),
]

[
\dot p_i^\star=(A+\alpha(z,Z))p_i^\star.
]

This is the gauge-theoretic test.

## B. Frozen-flow experiment

Fix a centered cloud (p) with

[
\sum_i a_i p_i=0.
]

Then integrate the autonomous reduced flow

[
\dot u
======

-\frac12(1-|u|^2)\sum_i a_i M_u(p_i).
]

This tests the Busemann gradient and Euler–Sundman conservative lift.

Do not confuse these. Along the moving physical orbit, (z(t)) is a moving canonical center. Along the frozen flow, (u(t)) is a reduced LMS coordinate relative to a fixed centered representative.

---

# 13. Frozen-flow BAV diagnostics

For a fixed anchor cloud (p), define

[
F_p(u)=\sum_i a_iM_u(p_i),
]

[
\Phi_p(u)=\sum_i a_i\log\frac{1-|u|^2}{|u-p_i|^2}.
]

Then

[
\dot u
======

-\frac12(1-|u|^2)F_p(u).
]

The Euler–Sundman time satisfies

[
dt=\frac{4}{(1-|u|^2)^2},d\tau.
]

In (\tau)-time,

[
u_\tau=-\nabla_{\mathrm{euc}}\Phi_p(u).
]

The conservative Hamiltonian is

[
\boxed{
H_{\mathrm{ES}}
===============

## \frac12|u_\tau|^2

\frac12|\nabla_{\mathrm{euc}}\Phi_p(u)|^2.
}
]

On the gradient branch,

[
\boxed{
H_{\mathrm{ES}}=0.
}
]

Numerical helper:

```python
es_energy(u, u_tau, P, a)
```

computes

[
H_{\mathrm{ES}}.
]

This is the right place to test the BAV/Euler–Sundman conservative lift.

---

# 14. Local source test from inertia tensor

For a centered cloud (p),

[
\sum_i a_i p_i=0,
]

define

[
T=\sum_i a_i p_i\otimes p_i.
]

Near (u=0),

[
F_p(u)
======

-2(I-T)u+O(|u|^2).
]

Therefore the frozen LMS flow satisfies

[
\boxed{
\dot u=(I-T)u+O(|u|^2).
}
]

So the eigenvalues of

[
I-T
]

are the local source exponents.

Helper:

```python
source_matrix(P, a):
    T = inertia_tensor(P, a)
    return I - T
```

Then compare numerically against the Jacobian of the frozen reduced flow at (u=0).

Expected:

[
\boxed{
D\dot u|_{0}=I-T.
}
]

This is a clean way to test the local theory.

---

# 15. A compact testing checklist

For each trajectory (x(t)):

### Gauge fixing

[
\left|\sum_i a_i M_{z(t)}(x_i(t))\right|\ll1.
]

### (z)-equation

[
\dot z
\approx
Az+\frac12(1+|z|^2)Z-\langle Z,z\rangle z.
]

### Centered shape conservation

[
\langle p_i^\star(t),p_j^\star(t)\rangle
\approx
\langle p_i^\star(0),p_j^\star(0)\rangle.
]

### Inertia spectrum conservation

[
\operatorname{spec}T(t)
\approx
\operatorname{spec}T(0).
]

### Predicted connection

[
\Omega_{\mathrm{pred}}
======================

A+\alpha(z,Z).
]

### Covariant constancy

[
\dot p_i^\star
\approx
\Omega_{\mathrm{pred}}p_i^\star.
]

### Data-estimated connection

[
\Omega_{\mathrm{data}}
======================

\operatorname{skew}\left(
\left(\sum_i a_i\dot p_i^\star\otimes p_i^\star\right)
\left(\sum_i a_i p_i^\star\otimes p_i^\star\right)^+
\right).
]

Compare

[
\Omega_{\mathrm{data}}\approx\Omega_{\mathrm{pred}}.
]

### Strain error

[
\operatorname{sym}\left(
\left(\sum_i a_i\dot p_i^\star\otimes p_i^\star\right)
T^+
\right)
\approx 0.
]

This last quantity is especially useful for debugging or for testing nonideal simulations.

---

# 16. The computational interpretation

The formal gauge theory translates into this pipeline:

[
x(t)
\longrightarrow
z(t)=c_\star(x(t))
\longrightarrow
p^\star(t)=M_{z(t)}x(t)
\longrightarrow
\text{shape invariants and connection}.
]

The canonical gauge removes the boost component. What remains is pure rotation:

[
p^\star(t)=\zeta(t)p^\star(0).
]

Therefore the cloud’s intrinsic shape is constant, and all nontrivial motion has been decomposed into:

[
\boxed{
\text{boost motion }z(t)
}
]

and

[
\boxed{
\text{rotational connection }\Omega(t)=A+\alpha(z,Z).
}
]

This is the computational version of the central theorem. The key numerical prediction is not merely that some scalar is conserved; it is that, after canonical centering, the entire cloud should move by rotation only.
