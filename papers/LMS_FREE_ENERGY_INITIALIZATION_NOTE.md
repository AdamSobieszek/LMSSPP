# LMS Initialization by Entropy Shells, Energy Extremization, and Poisson Reference States

This note gives a self-contained formulation of an initialization problem for the LMS alignment model on $S^2$ in which:

- the user-selected scalar control is an entropy level, not a reduced radius,
- the three distinguished initial states at that entropy are:
  - a Poisson-manifold state,
  - a minimum-energy state,
  - a maximum-energy state,
- and the corresponding variational relation is written in a mathematically honest measure-theoretic form.

The purpose of this note is to separate clearly:

- the actual LMS dynamics,
- the Poisson manifold coming from LMS Möbius geometry,
- and the free-energy relation used only to define initial conditions.

---

## 1. Ambient Measure Space and Basic Observables

Let
$$
\Omega := S^2 \subset \mathbb R^3,
$$
and let $\mathcal P(\Omega)$ denote the Borel probability measures on $S^2$.

For any $\mu \in \mathcal P(\Omega)$, define its first moment
$$
m[\mu] := \int_{S^2} x \, d\mu(x) \in \mathbb R^3,
$$
and its order parameter
$$
Z[\mu] := K\,m[\mu],
$$
where $K > 0$ is the LMS coupling strength.

The squared dipole magnitude is
$$
|m[\mu]|^2 = \left| \int_{S^2} x \, d\mu(x) \right|^2.
$$

For the finite-$N$ empirical measure
$$
\mu_N := \frac{1}{N}\sum_{i=1}^N \delta_{x_i},
$$
the corresponding observables are
$$
m[\mu_N] = \frac{1}{N}\sum_{i=1}^N x_i,
\qquad
Z[\mu_N] = \frac{K}{N}\sum_{i=1}^N x_i.
$$

---

## 2. The LMS Poisson Manifold

Let $\sigma$ be the uniform probability measure on $S^2$. The distinguished LMS continuum family is the Möbius orbit of $\sigma$:
$$
\mathcal M_P := \{ \mu_z := (M_{-z})_\# \sigma : z \in B^3 \}.
$$

Relative to $\sigma$, each $\mu_z$ has density
$$
d\mu_z(x) = P_{\mathrm{hyp}}(z,x)\,d\sigma(x),
$$
where the hyperbolic Poisson kernel on $B^3 \times S^2$ is
$$
P_{\mathrm{hyp}}(z,x) = \left( \frac{1-|z|^2}{|z-x|^2} \right)^2.
$$

For this family, the order parameter closes:
$$
Z[\mu_z] = K\,f_3(|z|)\,z.
$$
Equivalently,
$$
m[\mu_z] = f_3(|z|)\,z.
$$

Hence the Poisson family is a one-parameter family up to rotation:

- the direction of $z$ determines the dipole axis,
- the radius $r := |z|$ determines the dipole magnitude,
- and therefore also determines all one-parameter observables restricted to $\mathcal M_P$.

---

## 3. Alignment Energy

For the linear mean-field LMS model, the natural continuum alignment energy is
$$
\mathcal E[\mu]
:=
-\frac{K}{2}|m[\mu]|^2
=
-\frac{K}{2}\left|\int_{S^2} x\,d\mu(x)\right|^2.
$$

Equivalently,
$$
\mathcal E[\mu]
=
-\frac{K}{2}
\iint_{S^2 \times S^2} x \cdot y \, d\mu(x)\,d\mu(y).
$$

For the empirical measure $\mu_N$ this becomes
$$
\mathcal E_N(x_1,\dots,x_N)
:=
-\frac{K}{2}\left|\frac{1}{N}\sum_{i=1}^N x_i\right|^2.
$$

Interpretation:

- minimizing $\mathcal E$ means increasing alignment,
- maximizing $\mathcal E$ means decreasing alignment,
- because $\mathcal E$ is most negative when the dipole magnitude $|m[\mu]|$ is large.

Restricted to the Poisson manifold,
$$
\mathcal E[\mu_z]
=
-\frac{K}{2} f_3(|z|)^2 |z|^2.
$$

Thus, on $\mathcal M_P$, the energy is determined entirely by $|z|$.

---

## 4. Entropy: Continuum and Particle Versions

### 4.1. Continuum configurational entropy

If $\mu$ is absolutely continuous with respect to $\sigma$, say
$$
d\mu = \rho \, d\sigma,
$$
then its configurational entropy relative to the uniform measure is
$$
\mathcal S[\mu \mid \sigma]
:=
-\int_{S^2} \rho(x)\log \rho(x)\,d\sigma(x).
$$

This is the natural continuum entropy on the sphere.

However, the empirical measure $\mu_N$ is singular with respect to $\sigma$, so $\mathcal S[\mu_N \mid \sigma]$ is not directly usable for particle optimization.

### 4.2. Regularized entropy valid for all measures

To obtain a quantity defined both for smooth measures and empirical measures, choose a normalized spherical smoothing kernel $K_\kappa(x,y)$ satisfying
$$
K_\kappa(x,y) \ge 0,
\qquad
\int_{S^2} K_\kappa(x,y)\,d\sigma(y) = 1
\quad \text{for all } x \in S^2.
$$

For $\mu \in \mathcal P(S^2)$ define the smoothed density
$$
\rho_{\mu,\kappa}(x)
:=
\int_{S^2} K_\kappa(x,y)\,d\mu(y).
$$

Then define the regularized entropy
$$
\mathcal S_\kappa[\mu]
:=
-\int_{S^2} \log \rho_{\mu,\kappa}(x)\,d\mu(x).
$$

This is well-defined for every probability measure $\mu$ on $S^2$.

For an empirical measure $\mu_N = \frac{1}{N}\sum_i \delta_{x_i}$,
$$
\rho_{\mu_N,\kappa}(x)
=
\frac{1}{N}\sum_{j=1}^N K_\kappa(x,x_j),
$$
and therefore
$$
\mathcal S_{\kappa,N}(x_1,\dots,x_N)
=
-\frac{1}{N}\sum_{i=1}^N
\log\left(
\frac{1}{N}\sum_{j=1}^N K_\kappa(x_i,x_j)
\right).
$$

This is the mathematically honest particle-level entropy surrogate corresponding to the kernel-based entropy currently used in the widget.

---

## 5. What the Current Radius-Based Initialization Really Does

The existing widget family fixes a reduced-radius-derived shell and then extremizes a kernel entropy proxy on that shell.

More precisely, for a chosen target axis $u \in S^2$ and reduced radius $r \in [0,1)$, the current construction first converts $r$ into the Poisson-manifold dipole magnitude
$$
q(r) := f_3(r)\,r,
$$
and then approximately solves
$$
\text{maximize or minimize } \mathcal S_{\kappa,N}(x_1,\dots,x_N)
$$
subject to
$$
\left|\frac{1}{N}\sum_{i=1}^N x_i\right| \approx q(r),
\qquad
\frac{\sum_i x_i}{\left|\sum_i x_i\right|} \approx u.
$$

So the present construction is:

- fixed dipole shell,
- variable entropy.

This is not yet the entropy-controlled formulation.

---

## 6. The Proposed Entropy-Shell Initialization Problem

The new proposal is to reverse the logic.

Instead of fixing the reduced radius $r$ or the dipole shell $q(r)$, we choose an entropy level
$$
s_{\mathrm{target}}
$$
and define three distinguished states at that entropy.

Fix a target axis $u \in S^2$. Then define:

### 6.1. Poisson state at fixed entropy

Find the Poisson-manifold state $\mu^P_{s,u}$ such that
$$
\mu^P_{s,u} = \mu_{r u}
$$
for the unique or selected radius $r$ solving
$$
\mathcal S_\kappa[\mu_{r u}] = s_{\mathrm{target}}.
$$

Since $\mu_{r u}$ depends only on $r$ and $u$, this reduces to a one-dimensional root-finding problem in $r$.

This state is the distinguished LMS geometric reference at the chosen entropy.

### 6.2. Minimum-energy state at fixed entropy

Define the minimum-energy state by
$$
\mu^-_{s,u}
\in
\operatorname*{arg\,min}
\left\{
\mathcal E[\mu]
:
\mu \in \mathcal P(S^2),
\ \mathcal S_\kappa[\mu] = s_{\mathrm{target}},
\ m[\mu]\cdot u \ge 0
\right\}.
$$

This is the most aligned state available on that entropy shell.

### 6.3. Maximum-energy state at fixed entropy

Define the maximum-energy state by
$$
\mu^+_{s,u}
\in
\operatorname*{arg\,max}
\left\{
\mathcal E[\mu]
:
\mu \in \mathcal P(S^2),
\ \mathcal S_\kappa[\mu] = s_{\mathrm{target}},
\ m[\mu]\cdot u \ge 0
\right\}.
$$

This is the least aligned, most cancellation-dominated state available on that entropy shell.

The axis condition $m[\mu]\cdot u \ge 0$ fixes the sign convention and prevents the optimizer from flipping to the opposite pole.

---

## 7. Particle Approximation of the Entropy-Shell Problem

In a finite-$N$ numerical implementation, one replaces the measure problem by:

### Poisson state

Sample
$$
\mu^P_{s,u,N} \approx \mu^P_{s,u}
$$
using points drawn from the Poisson state $\mu_{r u}$ with $r$ chosen from
$$
\mathcal S_{\kappa,N}^{\mathrm{Poi}}(r,u) \approx s_{\mathrm{target}}.
$$

### Minimum-energy state

Solve approximately
$$
(x_1^-,\dots,x_N^-)
\in
\operatorname*{arg\,min}
\left\{
\mathcal E_N(x_1,\dots,x_N)
:
\mathcal S_{\kappa,N}(x_1,\dots,x_N)=s_{\mathrm{target}},
\ m_N \cdot u \ge 0,
\ x_i \in S^2
\right\},
$$
where
$$
m_N := \frac{1}{N}\sum_{i=1}^N x_i.
$$

### Maximum-energy state

Solve approximately
$$
(x_1^+,\dots,x_N^+)
\in
\operatorname*{arg\,max}
\left\{
\mathcal E_N(x_1,\dots,x_N)
:
\mathcal S_{\kappa,N}(x_1,\dots,x_N)=s_{\mathrm{target}},
\ m_N \cdot u \ge 0,
\ x_i \in S^2
\right\}.
$$

In practice these constrained problems can be solved by projected optimization with penalty terms or augmented Lagrangian methods.

---

## 8. Free-Energy Relation

The entropy-shell optimization problems can be written using a Lagrange multiplier.

For a stationary point of energy at fixed entropy, the first-variation relation is
$$
\delta \mathcal E[\mu] = \beta \, \delta \mathcal S_\kappa[\mu]
$$
for some scalar multiplier $\beta$.

Equivalently, stationary points are critical points of the free-energy functional
$$
\mathcal F_\beta[\mu]
:=
\mathcal E[\mu] - \beta \,\mathcal S_\kappa[\mu].
$$

Thus:

- minimizing energy at fixed entropy corresponds to minimizing $\mathcal F_\beta$ for an appropriate $\beta$,
- maximizing energy at fixed entropy corresponds to maximizing $\mathcal E$ on the same entropy shell, or equivalently minimizing $-\mathcal E - \beta \mathcal S_\kappa$.

This is the exact sense in which a free-energy relation can be introduced for initialization.

---

## 9. What This Free-Energy Relation Does Not Mean

The free-energy relation above is a variational principle for selecting initial conditions. It is not, by itself, the evolution law of the LMS system.

The actual LMS reduced dynamics is governed by the Möbius/hyperbolic equations, and in the linear mean-field setting the reduced variable $w$ descends the LMS hyperbolic potential
$$
\Phi(w)
=
\sum_i a_i \log \frac{1-|w|^2}{|w-p_i|^2}.
$$

Therefore:

- the Poisson manifold is special because of LMS group geometry,
- the free-energy functional is useful for initialization,
- but the LMS ODE is not literally a thermodynamic gradient flow of $\mathcal E - \beta \mathcal S_\kappa$.

This distinction should remain explicit in both the code and the user-facing interpretation.

---

## 10. Why the Poisson Manifold Is a Reference Family, Not the Entropy Extremizer

The Poisson family $\mathcal M_P$ arises from the Möbius action on the uniform measure:
$$
\mu_z = (M_{-z})_\# \sigma.
$$

Its importance is geometric and dynamical:

- it is the distinguished LMS orbit of the uniform state,
- it is the continuum family on which the order parameter closes,
- and it is the family naturally parameterized by the reduced variable $z \in B^3$.

By contrast, if one solves a generic entropy-constrained energy optimization problem on $S^2$, the resulting optimizer need not lie on $\mathcal M_P$.

Hence:

- Poisson states should be treated as distinguished LMS reference states,
- not as the generic maximizers or minimizers of entropy-constrained alignment energy.

This is why the proposed three-state initialization is meaningful:

- one state lies on the LMS Poisson manifold,
- two comparison states are off-manifold energy extrema at the same entropy.

---

## 11. Two Honest Uses of the Poisson Manifold

### 11.1. Poisson as a starting point

Use $\mu^P_{s,u}$ as the reference state on the entropy shell, then initialize the numerical optimizer from a particle sample of that state and move off-manifold while preserving entropy.

This gives:

- a canonical LMS starting point,
- a fair comparison between Poisson and off-manifold states,
- and a natural way to define the minimum-energy and maximum-energy branches.

### 11.2. Poisson as an objective or regularizer

Alternatively, one may include a penalty that measures distance to the Poisson family, for example
$$
\mathcal J_{\beta,\lambda}[\mu]
:=
\mathcal E[\mu] - \beta \mathcal S_\kappa[\mu]
+ \lambda \, \mathrm{dist}(\mu,\mathcal M_P)^2.
$$

This would favor states that are both entropy-energy optimal and close to the LMS manifold.

That is a different construction from the three-state entropy-shell problem above, but it may be useful if the goal is to stay geometrically close to LMS-reduced states.

---

## 12. Practical Algorithm for an Entropy Slider

A mathematically honest widget redesign can proceed as follows.

### Step 1. Replace the radius slider by an entropy slider

The primary scalar control becomes
$$
s_{\mathrm{target}}.
$$

This should be documented as:

- continuum entropy if one works at measure level,
- regularized kernel entropy in the actual particle implementation.

### Step 2. Construct the Poisson reference state

For a chosen axis $u$, solve numerically for $r$ in
$$
\mathcal S_\kappa[\mu_{r u}] = s_{\mathrm{target}}.
$$

Then sample points from $\mu_{r u}$.

### Step 3. Construct the two energy-extremal comparison states

Starting from either the Poisson sample or from randomized perturbations of it, solve:
$$
\min \mathcal E_N
\quad \text{subject to} \quad
\mathcal S_{\kappa,N} = s_{\mathrm{target}},
\quad x_i \in S^2,
\quad m_N \cdot u \ge 0,
$$
and
$$
\max \mathcal E_N
\quad \text{subject to} \quad
\mathcal S_{\kappa,N} = s_{\mathrm{target}},
\quad x_i \in S^2,
\quad m_N \cdot u \ge 0.
$$

### Step 4. Display the three states as:

- `Minimum Energy`,
- `Poisson`,
- `Maximum Energy`.

These labels are more honest than `low entropy` and `high entropy` once entropy is the constrained variable.

---

## 13. Interpretation: Initial Low Entropy and the Arrow of Time

The relevant analogue of the low-entropy-past interpretation is not a literal reversal of the LMS time variable.

Instead, the causal asymmetry enters through the initialization rule:

- one chooses a special entropy shell,
- one chooses a special branch on that shell,
- and then one evolves those initial states by the deterministic LMS dynamics.

So the asymmetry comes from the selected initial condition class, not from a hidden change of sign in the ODE.

This is the correct interpretation to keep explicit in the widget.

---

## 14. Recommended Mathematical Summary

The proposed initialization scheme should be stated as follows:

> For each target axis $u \in S^2$ and entropy level $s_{\mathrm{target}}$, define three distinguished probability measures on $S^2$:
> 
> - the Poisson reference state $\mu^P_{s,u} \in \mathcal M_P$ with $\mathcal S_\kappa[\mu^P_{s,u}] = s_{\mathrm{target}}$,
> - the minimum-energy state $\mu^-_{s,u}$ solving $\min \mathcal E[\mu]$ at fixed $\mathcal S_\kappa[\mu] = s_{\mathrm{target}}$,
> - the maximum-energy state $\mu^+_{s,u}$ solving $\max \mathcal E[\mu]$ at fixed $\mathcal S_\kappa[\mu] = s_{\mathrm{target}}$,
> 
> subject in each case to the axis gauge $m[\mu]\cdot u \ge 0$.

This produces a geometrically distinguished LMS state together with two off-manifold comparison states on the same entropy shell.

---

## 15. Implementation Convention Recommended for the Widget

For code and UI purposes, the clean convention is:

- theory layer:
  - write $\mathcal S[\mu \mid \sigma]$ for the continuum configurational entropy,
- implementation layer:
  - write $\mathcal S_\kappa[\mu]$ and $\mathcal S_{\kappa,N}$ for the regularized entropy actually used in numerics,
- UI layer:
  - label the slider simply as `Entropy`,
  - but document in the note or tooltip that this means the kernel-regularized entropy used to approximate continuum configurational entropy.

This keeps the mathematical language strict while preserving a usable numerical implementation.
