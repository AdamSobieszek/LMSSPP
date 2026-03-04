# LMS on the Sphere: Why Low Entropy Can Coexist with Low `|w|`

This note is self-contained and tied to:

- paper source: `/Users/adamsobieszek/PycharmProjects/ManipyTraversal/notebooks/kuramoto/lms_paper.tex`
- implementation: `/Users/adamsobieszek/PycharmProjects/ManipyTraversal/notebooks/kuramoto/LMS.py`
- widget initialization logic: `/Users/adamsobieszek/PycharmProjects/ManipyTraversal/notebooks/kuramoto/lms_ball3d_widget.py`

It has three goals:

1. Re-derive the core LMS equations directly from the paper.
2. Show exactly how low-entropy states with low `|w|` are created in code.
3. State the main theoretical interpretation: two distinct entropy-dissipation mechanisms exist in the hyperbolic flow.

---

## 1. Core equations from the paper (general dimension `d`)

### 1.1 Kuramoto-on-sphere dynamics

From `lms_paper.tex` (Eq. `\label{governingeqn}`), the model is

$$
\dot x_i = A_i x_i + Z - \langle Z, x_i\rangle x_i,\quad x_i\in S^{d-1}\subset \mathbb{R}^d.
$$

For identical intrinsic rotation (`A_i = A`), the dynamics lies on a Mobius-group orbit.

### 1.2 Real Mobius boost on the ball/sphere

From `lms_paper.tex` (Sec. Hyperbolic geometry, Eqs. around lines with `M_w`):

$$
M_w(x)=\frac{(1-\|w\|^2)(x-\|x\|^2 w)}{1-2\langle w,x\rangle + \|w\|^2\|x\|^2}-w,
$$

and for `|x|=1`,

$$
M_w(x)=\frac{(1-\|w\|^2)(x-w)}{\|x-w\|^2}-w.
$$

Any orientation-preserving isometry can be written as

$$
g(x)=\zeta M_w(x)=M_{-z}(\xi x),\quad z=-\zeta w,\quad |z|=|w|.
$$

So `|z|` and `|w|` are always equal (same orbit coordinate radius, different frame).

### 1.3 Reduced equations in `(w, zeta)`

From `lms_paper.tex` (Eq. `\label{wdot}` and system `\label{wzeta}`):

$$
\dot w = -\frac12 (1-\|w\|^2)\,\zeta^{-1}Z,\qquad
\dot \zeta = (A-\alpha(\zeta w,Z))\zeta.
$$

For linear order parameter

$$
Z=\sum_i a_i x_i,
$$

the `w` equation decouples from `zeta`:

$$
\dot w=-\frac12 (1-\|w\|^2)\,Z(M_w(p)).
$$

### 1.4 Hyperbolic metric and gradient form

From `lms_paper.tex` (Sec. Existence of Hyperbolic Gradient):

Hyperbolic metric on `B^d`:

$$
ds=\frac{2|dw|}{1-\|w\|^2},\quad \phi(w)=\frac{2}{1-\|w\|^2}.
$$

Hence

$$
\nabla_{\mathrm{hyp}}\Phi = \phi^{-2}\nabla_{\mathrm{euc}}\Phi
=\frac14(1-\|w\|^2)^2\nabla_{\mathrm{euc}}\Phi.
$$

Potential:

$$
\Phi(w)=\sum_i a_i \log\frac{1-\|w\|^2}{\|w-p_i\|^2}
=\frac{1}{d-1}\sum_i a_i \log P_{\mathrm{hyp}}(w,p_i),
$$

with hyperbolic Poisson kernel

$$
P_{\mathrm{hyp}}(z,x)=\left(\frac{1-\|z\|^2}{\|z-x\|^2}\right)^{d-1}.
$$

Paper derivation gives

$$
\nabla_{\mathrm{euc}}\Phi(w)=\frac{2}{1-\|w\|^2}\,Z(M_w(p)),
$$
so
$$
\nabla_{\mathrm{hyp}}\Phi(w)=\frac12(1-\|w\|^2)\,Z(M_w(p)),
$$
and therefore
$$
\dot w = -\nabla_{\mathrm{hyp}}\Phi(w).
$$

---

## 2. Why `d=2` is special and `d>2` gives `|Z| != K|w|`

### 2.1 Continuum `Z(z)` relation from the paper

For uniform base measure, `lms_paper.tex` gives

$$
Z(z)=K\int_{S^{d-1}} M_{-z}(x)\,d\sigma(x).
$$

For `d=2`, residue/Cauchy calculation yields exactly

$$
Z(z)=Kz.
$$

For general `d`, paper gives

$$
Z(z)=K\,f_d(|z|)\,z,\qquad
f_d(r)=\frac{F(1,1-d/2;1+d/2;r^2)}{F(1,1-d/2;1+d/2;1)}.
$$

So

$$
\frac{|Z|}{K}=f_d(|z|)\,|z|=f_d(|w|)\,|w|.
$$

Because `f_2(r)=1`, we get `|Z|/K = |w|` only in `d=2`.
For `d>2`, `f_d` is nontrivial, so `|Z|/K` and `|w|` generally differ.

### 2.2 Interpretation in terms of Poisson-kernel exponent

In `d=2`, hyperbolic and Euclidean Poisson kernels coincide on the disk in this context, and the first moment map is linear (`Z \propto z`).
In higher `d`, exponent `d-1` in

$$
P_{\mathrm{hyp}}(z,x)=\left(\frac{1-\|z\|^2}{\|z-x\|^2}\right)^{d-1}
$$

changes the first-moment transfer law through the hypergeometric factor `f_d`.

That is the exact source of the extra radial nonlinearity.

---

## 3. What the implementation actually does

### 3.1 LMS equations and autograd potential path

In `/Users/adamsobieszek/PycharmProjects/ManipyTraversal/notebooks/kuramoto/LMS.py`:

- `mobius_sphere` implements
  $$
  M_w(x)=\frac{(1-|w|^2)(x-w)}{|x-w|^2}-w.
  $$
- `hyperbolic_potential_lms` implements
  $$
  \Phi(w)=\sum_i a_i\log\frac{1-|w|^2}{|w-p_i|^2}.
  $$
- `hyperbolic_grad_autograd` and `lms_vector_field_autograd` compute
  $$
  \dot w=-\nabla_{\mathrm{hyp}}\Phi(w).
  $$
- `lms_reduced_rhs` (explicit mode) uses
  $$
  \dot w=-\frac12(1-|w|^2)Z_{\mathrm{body}},
  $$
  matching paper Eq. `\label{wdot}`.

So explicit and autograd backends are two implementations of the same reduced LMS flow.

### 3.2 How low-entropy initial states are constructed in the widget

In `/Users/adamsobieszek/PycharmProjects/ManipyTraversal/notebooks/kuramoto/lms_ball3d_widget.py`:

- Initial reduced state:
  $$
  w_0 = r_0\,c,\quad c=\text{unit}(w\_\text{az},w\_\text{el}).
  $$
- If `low_entropy=False`: base points are uniform random on sphere.
- If `low_entropy=True`: base points are clustered as
  $$
  p_i = \frac{c + 0.08\,\xi_i}{\|c + 0.08\,\xi_i\|},\quad \xi_i\sim\mathcal N(0,I).
  $$

Crucial point: in current code, `w0` and base-point entropy are controlled independently.
So low entropy with low `|w|` is intentionally possible.

However, for a *fixed physical state* represented by

$$
x_i=\zeta M_w(p_i),
$$

`w` and `p_i` are not conceptually independent coordinates of that same state: changing base points changes the coordinate value of `w` by Mobius covariance (`w' = M(w)` in the paper).
So “independence” above is only about how the simulator *chooses initial conditions*, not a structural statement about reduced coordinates on one orbit.

---

## 4. Corrected near-`|w|=0` analysis: what controls `|w|` and what does not

Let weighted moments of base points be

$$
m:=\sum_i a_i p_i,\qquad
C:=\sum_i a_i\,p_i p_i^\top,\qquad \sum_i a_i=1.
$$

From the paper expansion

$$
M_w(p_i)=p_i-2w+2\langle w,p_i\rangle p_i+O(|w|^2),
$$

we get

$$
Z(w)=\sum_i a_i M_w(p_i)=m-2(I-C)w+O(|w|^2).
$$

So near the origin:

$$
\dot w=-\frac12(1-|w|^2)Z(w)
=-\frac12 m + (I-C)w + O(|w|^2).
$$

And for the potential:

$$
\Phi(w)
=2\,m^\top w + 2\,w^\top(C-I)w + O(|w|^3),
$$
$$
\nabla\Phi(w)=2m+4(C-I)w+O(|w|^2).
$$

### Consequence

The leading forcing of `w` is the dipole moment `m` (first moment), not “entropy” directly.
High concentration can still produce small `|m|` (for example, symmetric multi-cluster/antipodal patterns), so low entropy does not imply large `|w|`.

This corrects the naive reading that concentration alone explains `|w|`.
The relevant split is between modes that contribute to `m` and modes that mostly live in higher moments (`C`, higher harmonics).

---

## 5. `_\parallel` / `_\perp` decomposition relative to `\Delta^{d-1}`

The synchronized manifold is

$$
\Delta^{d-1}=\{(u,\dots,u):u\in S^{d-1}\}.
$$

Choose a direction `u` (naturally `u = m/|m|` if `m \neq 0`) and decompose

$$
p_i=\alpha_i u+\eta_i,\qquad
\alpha_i=\langle p_i,u\rangle,\quad \eta_i\perp u.
$$

Define:

$$
m_\parallel := u^\top m=\sum_i a_i\alpha_i,\qquad
m_\perp := \Pi_{u^\perp}m=\sum_i a_i\eta_i.
$$

For second moments:

$$
\lambda_\parallel := u^\top C u=\sum_i a_i\alpha_i^2,\qquad
C_\perp := \Pi_{u^\perp} C \Pi_{u^\perp}.
$$

A variance split (about weighted mean) is:

$$
\sigma_\parallel^2 := \sum_i a_i(\alpha_i-m_\parallel)^2
=\lambda_\parallel-m_\parallel^2,
$$
$$
\sigma_\perp^2 := \sum_i a_i\|\eta_i-m_\perp\|^2
=\mathrm{tr}(C_\perp)-\|m_\perp\|^2.
$$

Now decompose `w = w_\parallel u + w_\perp`. Projecting the near-zero expansion:

$$
Z_\parallel(w)=m_\parallel -2(1-\lambda_\parallel)w_\parallel + O(|w|^2),
$$
$$
Z_\perp(w)=m_\perp -2(I-C_\perp)w_\perp + O(|w|^2)
$$
(plus optional mixed terms if `u` is not an eigenvector of `C`).

If `u = m/|m|`, then `m_\perp=0` and:

- `w_\parallel` has constant forcing `-\frac12 m_\parallel`;
- `w_\perp` has no constant forcing; it is shaped by anisotropic stiffness through `C_\perp`.

So near `w=0`, the component that most directly contributes to `|w|` growth is the `\parallel` dipole part (`m_\parallel`), while many entropy/anisotropy features can sit in `\perp`/higher-moment structure without strongly forcing `|w|`.

### 5.1 Maximizing/minimizing what contributes to `|w|` near zero

From

$$
\dot w(0)=-\frac12 m,
$$

the quantity to maximize/minimize for immediate `|w|` growth is `|m|` (or `|m_\parallel|` in a chosen frame).

For fixed projection magnitudes `|\alpha_i|`, the weighted sum

$$
m_\parallel=\sum_i a_i\alpha_i
$$

is:

- maximized in magnitude when signs align (same hemisphere along `u`);
- minimized (possibly to `0`) by sign cancellation (balanced opposite hemispheres).

Hence one can have concentrated states with:

1. **large `|m_\parallel|`** (single-cluster type) -> strong direct forcing of `|w|`;
2. **small `|m_\parallel|`** (balanced antipodal/multicluster type) -> weak direct forcing of `|w|`.

Both can have low geometric entropy, but only the first strongly drives `|w|` at leading order.

What can remain large while not directly forcing `|w|` near `0`:

- `\sigma_\perp^2` and anisotropic parts of `C_\perp`;
- higher-order angular structure beyond the dipole.

These appear in linear-response/stiffness terms and higher-order terms, not in the constant forcing term.

---

## 6. Main theoretical contribution: two distinct entropy-dissipation mechanisms

Given the LMS hyperbolic gradient flow and the fact that `|w|->1` implies collapse to near-synchrony (`H->0`), there are two separate mechanisms:

## Mechanism A: `\parallel` dipole-alignment dissipation (direct `|w|` channel)

- Entropy decreases through the part of the state that creates nonzero first moment `m_\parallel`.
- This directly forces `w` already at first order near `w=0`.
- This is the channel most tightly tied to growth of `|w|`.

## Mechanism B: `\perp`/higher-moment anisotropy dissipation (weak direct `|w|` forcing near 0)

- Entropy can be low (or decrease) due to anisotropy concentrated in second/higher moments with small dipole.
- Near `w=0`, these modes mainly modify linear response via `C` (and higher derivatives), not the constant forcing of `w`.
- Thus low entropy can coexist with low `|w|` when dipole cancellation is strong.

Hence low entropy and large `|w|` are not equivalent statements.
They coincide in one regime (strong dipole-alignment channel), but separate in another (anisotropy-dominated, dipole-cancelled channel).

---

## 7. Specialization to `d=3` (`S^2 / B^3`)

For `d=3`,

$$
f_3(r)=\frac{F(1,-1/2;5/2;r^2)}{F(1,-1/2;5/2;1)},
$$

so continuum law is

$$
Z(z)=K f_3(|z|) z,\qquad \frac{|Z|}{K}=f_3(|w|)\,|w|.
$$

Therefore in `S^2/B^3` the radial relation between order-parameter magnitude and `|w|` is nonlinear, unlike `d=2`.

In particular, near `r=0`,

$$
\frac{|Z|}{K}\approx f_3(0)\,|w|=\frac{4}{3}|w|,
$$

so even the small-radius slope differs from the `d=2` identity.

In the widget:

- `|w|` is directly the reduced coordinate norm.
- `|Z|/K` is a distinct observable that generally differs from `|w|` in `d=3`.
- A low-entropy toggle can make entropy low at frame 0 even for very small `r0=|w0|`.

This is expected from the equations above and reflects the two mechanisms, not a bug.

---

## 8. Practical diagnostic to separate mechanisms in experiments

To distinguish Mechanism A vs B numerically, track:

1. `H_base = H({p_i})` once at recompute.
2. `H_t = H({M_{w_t}(p_i)})` over time.
3. `m_\parallel(t), \|m_\perp(t)\|` from the pushed-forward points.
4. `\sigma_\parallel^2(t), \sigma_\perp^2(t)`.
5. `|w_t|` and `|Z_t|/K`.

Interpretation:

- low `H_base`, low `|w_0|`, small `m_\parallel` but structured `\sigma_\perp` -> anisotropy-dominated regime.
- large `m_\parallel` with rapid `|w|` growth -> dipole-alignment regime.
- falling `H_t` together with `|w_t|\to 1` -> late-time hyperbolic concentration regime.
