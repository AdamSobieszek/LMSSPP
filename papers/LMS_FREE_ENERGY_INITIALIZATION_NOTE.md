# LMS Free-Energy Initialization Note (Mathematically Corrected)

This note states, in measure-theoretic form, the initialization problems used by the LMS entropy-shell widgets, and separates:

- exact LMS continuum geometry,
- finite-$N$ numerical objectives,
- and UI coordinates used in practice.

The goal is mathematical correctness first, then implementation mapping.

---

## 1. State Space and Basic Observables

Let $\Omega=S^2\subset\mathbb R^3$, and $\mathcal P(\Omega)$ be Borel probability measures on $S^2$.

For $\mu\in\mathcal P(\Omega)$ define first moment and order parameter:

$$
m[\mu]:=\int_{S^2}x\,d\mu(x),\qquad Z[\mu]:=K\,m[\mu],\quad K>0.
$$

The alignment energy functional used by the widget is

$$
\mathcal E[\mu]:=-\frac K2\,|m[\mu]|^2
=-\frac K2\left|\int_{S^2}x\,d\mu(x)\right|^2.
$$

For an empirical measure $\mu_N=\frac1N\sum_{i=1}^N\delta_{x_i}$:

$$
m_N=\frac1N\sum_{i=1}^Nx_i,
\qquad
\mathcal E_N(x_1,\dots,x_N)=-\frac K2\,|m_N|^2.
$$

So minimizing $\mathcal E$ means maximizing $|m|$ (strong alignment), and maximizing $\mathcal E$ means minimizing $|m|$ (cancellation).

---

## 2. Exact LMS Poisson-Manifold Geometry

Let $\sigma$ be uniform measure on $S^2$. The LMS Poisson manifold is

$$
\mathcal M_P:=\{\mu_z=(M_{-z})_\#\sigma:\ z\in B^3\}.
$$

Its density relative to $\sigma$ is

$$
\rho_z(x)=P_{\mathrm{hyp}}(z,x)=\left(\frac{1-|z|^2}{|z-x|^2}\right)^2.
$$

On $\mathcal M_P$, closure gives

$$
m[\mu_z]=f_3(|z|)z,
\qquad
q(r):=|m[\mu_z]|=f_3(r)r,\ r=|z|.
$$

Hence along Poisson states

$$
\mathcal E[\mu_z]= -\frac K2\,q(r)^2.
$$

This part is continuum LMS structure, not a numerical approximation.

---

## 3. Entropy Objects (What Is Exact vs What Is Used)

### 3.1 Continuum configurational entropy (reference object)

If $d\mu=\rho\,d\sigma$ with $\rho\in L^1(\sigma)$, define

$$
\mathcal S_{\mathrm{cont}}[\mu]:=-\int_{S^2}\rho\log\rho\,d\sigma.
$$

For empirical $\mu_N$, this is not directly usable (singular measure).

### 3.2 Particle kernel entropy used by optimizer

The implementation uses kernel

$$
K_\kappa(x,y)=\exp\!\big(\kappa(x\cdot y-1)\big),\quad \kappa>0.
$$

For points $x_1,\dots,x_N\in S^2$:

$$
H^{\mathrm{raw}}_{\kappa,N}
:=-\frac1N\sum_{i=1}^N
\log\!\left(\frac1N\sum_{j=1}^N e^{\kappa(x_i\cdot x_j-1)}\right).
$$

It is normalized by the finite-sample uniform baseline used in code:

$$
\rho_{\mathrm{unif}}(m,\kappa)
=\frac1m+\left(1-\frac1m\right)\frac{1-e^{-2\kappa}}{2\kappa},
\qquad
H_{\mathrm{unif}}(m,\kappa)=-\log\rho_{\mathrm{unif}}(m,\kappa).
$$

Then

$$
S_{\kappa,N}:=\operatorname{clip}\!\left(\frac{H^{\mathrm{raw}}_{\kappa,N}}{H_{\mathrm{unif}}(m,\kappa)},\ 0,\ 1.2\right).
$$

This $S_{\kappa,N}$ is the entropy shell constraint actually optimized in the entropy-shell branch.

### 3.3 Continuum-Poisson slider coordinate used in UI

In `continuum_poisson` mode, the slider does **not** use raw $\mathcal S_{\mathrm{cont}}$ directly. It uses a normalized monotone transform along Poisson states:

$$
\mathrm{KL}(r):=\int_{S^2}\rho_r\log\rho_r\,d\sigma,
\qquad
S_{\mathrm{cont}}^{\mathrm{ui}}(r):=1-\frac{\mathrm{KL}(r)}{\mathrm{KL}(r_{\max})}.
$$

This is a coordinate on Poisson manifold, clipped/monotone-corrected numerically.

---

## 4. Shell Coordinate Maps Used by the Widgets

The entropy shell backend precomputes tables in $r\in[0,r_{\max}]$:

- $S_{\kappa}^{\mathrm{Poi}}(r)$ (kernel entropy coordinate on Poisson manifold),
- $S_{\mathrm{cont}}^{\mathrm{ui}}(r)$ (continuum-poisson UI coordinate),
- $q(r)=f_3(r)r$.

Monotonicity is enforced numerically by monotone-decreasing projection.

Given current mode:

- `constant_entropy` + `kernel`:
  - slider gives $s$,
  - invert $s=S_{\kappa}^{\mathrm{Poi}}(r)$,
  - internal target is $S_{\kappa,N}\approx s$.
- `constant_entropy` + `continuum_poisson`:
  - slider gives $s$ in continuum-poisson coordinate,
  - invert $s=S_{\mathrm{cont}}^{\mathrm{ui}}(r)$,
  - convert to internal kernel target $s_\kappa=S_{\kappa}^{\mathrm{Poi}}(r)$,
  - optimize with $S_{\kappa,N}\approx s_\kappa$.
- `constant_energy`:
  - slider gives $q_0$ (not $\mathcal E$ directly),
  - radius is $r=q^{-1}(q_0)$,
  - implied energy target is $\mathcal E_0=-\frac K2 q_0^2$.

---

## 5. Variational Initialization Problems (Ideal Measure Form)

Fix axis $u\in S^2$ and gauge constraint $m[\mu]\cdot u\ge0$.

### 5.1 Constant-entropy family

For target entropy $s$:

$$
\mu_s^P:=\mu_{ru}\in\mathcal M_P,
\quad r\ \text{such that}\ S_{\kappa}^{\mathrm{Poi}}(r)=s
\ \text{(or calibrated from }S_{\mathrm{cont}}^{\mathrm{ui}}\text{)}.
$$

Then define off-manifold extremizers:

$$
\mu_s^-\in\arg\min\{\mathcal E[\mu]:\mu\in\mathcal P(S^2),\ S_\kappa[\mu]=s,\ m[\mu]\cdot u\ge0\},
$$

$$
\mu_s^+\in\arg\max\{\mathcal E[\mu]:\mu\in\mathcal P(S^2),\ S_\kappa[\mu]=s,\ m[\mu]\cdot u\ge0\}.
$$

### 5.2 Constant-energy family

For target dipole shell $q_0$ (equivalently energy shell $\mathcal E_0=-\frac K2q_0^2$):

$$
\mu_{q_0}^P:=\mu_{ru},\quad q(r)=q_0,
$$

and entropy extremizers on that shell:

$$
\arg\min/\arg\max\{S_\kappa[\mu]:\ |m[\mu]|=q_0,\ m[\mu]\cdot u\ge0\}.
$$

---

## 6. Finite-$N$ Algorithms Actually Implemented

### 6.1 Poisson branch

- Sample base points $p_i\sim\sigma$.
- Set $w=-ru$ and $x_i=M_w(p_i)$.
- In constant-entropy mode, refine $r$ by bisection so monitored $S_{\kappa,N}$ matches target more closely.

### 6.2 Constant-entropy, `min_energy`/`max_energy`

Optimization variable: $x_i\in S^2$ with per-step renormalization and move cap.

Penalty objective:

$$
L_\pm
=\lambda_S\big(S_{\kappa,N}(x)-s_\star\big)^2
+\lambda_E\,\Phi_\pm(m_N)
+\lambda_U\,D_u(m_N),
$$

with

$$
\Phi_-(m_N)=-|m_N|^2\quad(\text{min-energy branch}),
\qquad
\Phi_+(m_N)=+|m_N|^2\quad(\text{max-energy branch}),
$$

and axis-gauge penalty

$$
D_u(m)=\begin{cases}
\max(0,0.95-\cos\angle(m,u))^2,& |m|\ge0.05,\\
0,& |m|<0.05.
\end{cases}
$$

Important implementation detail (post-fix): entropy shell is not enforced on one fixed subset.

- gradient entropy uses full cloud for moderate $N$, otherwise randomized batches,
- feasibility/selection uses robust monitor entropy (full for moderate $N$, averaged randomized estimates for larger $N$),
- candidate acceptance requires shell tolerance and axis gauge,
- max-energy branch uses deterministic multi-restart from tangent perturbations.

### 6.3 Constant-energy branch

Current implementation reuses legacy shell-by-$q$ initializers for entropy extremes. It is a separate path from entropy-shell energy optimization.

---

## 7. Free-Energy Relation: Correct Mathematical Statement

For constrained extrema at fixed entropy, one introduces a Lagrangian (measure-level formalism):

$$
\mathcal L[\mu]=\mathcal E[\mu]+\beta\big(S_\kappa[\mu]-s_\star\big)+\gamma\,G_u[\mu],
$$

where $G_u$ encodes axis gauge/sign convention constraints.

At stationary points (formally):

$$
\delta\mathcal E[\mu]+\beta\,\delta S_\kappa[\mu]+\gamma\,\delta G_u[\mu]=0.
$$

Equivalent free-energy-style form is branch-dependent. For example, minimizing $\mathcal E$ at fixed entropy is equivalent to minimizing

$$
\mathcal F_\beta[\mu]=\mathcal E[\mu]-\beta S_\kappa[\mu]
$$

for an appropriate multiplier $\beta$, with constraints included.

This is an initialization variational principle, not the LMS evolution equation.

---

## 8. What Must Not Be Confused

1. LMS dynamics is governed by reduced Möbius equations (or full point dynamics), not by a thermodynamic gradient flow of $\mathcal E-\beta S_\kappa$.
2. Poisson manifold is geometrically distinguished in LMS; generic entropy/energy constrained optimizers need not lie on it.
3. `continuum_poisson` slider uses a calibrated Poisson-manifold coordinate, then optimization still uses kernel entropy internally.
4. In UI, `energy0` slider controls $q_0=|m|$ shell; energy is derived as $-\frac K2q_0^2$.

---

## 9. Recommended Notation for Documentation and Code

- Theory layer:
  - $\mathcal S_{\mathrm{cont}}$ for continuum configurational entropy,
  - $\mathcal E$ for alignment energy.
- Implementation layer:
  - $S_{\kappa,N}$ for particle kernel entropy proxy,
  - $S_{\mathrm{cont}}^{\mathrm{ui}}$ for continuum-poisson slider coordinate,
  - $q=|m|=|Z|/K$ for dipole shell coordinate.
- Widget labels:
  - `Minimum Energy / Poisson / Maximum Energy` in constant-entropy mode,
  - `Minimum Entropy / Poisson / Maximum Entropy` in constant-energy mode.

This naming keeps equations, numerics, and UI semantics consistent.

---

## 10. One-Line Mathematical Summary

Given axis $u\in S^2$, the widget compares one LMS manifold reference state and two off-manifold constrained extremizers, where the active constraint is either fixed entropy shell or fixed dipole/energy shell, and all finite-$N$ optimization is done with explicit kernel-entropy penalties plus sphere/gauge constraints.
