# Poisson-to-CS Bridge in Euclidean Coordinates

This note isolates the part of the LMS / hyperbolic-Poisson story that survives after passing from the sphere to Euclidean coordinates, and then states the exact radial transport that matches the weakly singular Cucker-Smale-type kernel
\[
W_\alpha(x)=\frac{|x|^{2-\alpha}}{(2-\alpha)(1-\alpha)},\qquad \alpha\in(0,1).
\]

To avoid clashing with the sphere paper's use of `d`, I will use:

- `n` for the Euclidean chart dimension.
- `S^n = \partial B^{n+1}` for the boundary sphere.
- in the CS application, set `n=d`, so the physical space is `\mathbb R^d`.

## Setup

Let `\sigma_n` be the uniform probability measure on `S^n`, and for `a\in B^{n+1}` let
\[
\rho_a := (M_{-a})_\# \sigma_n,
\]
where `M_{-a}` is the hyperbolic Möbius boost from the LMS paper. Its density with respect to `\sigma_n` is the hyperbolic Poisson kernel
\[
P_n(a,\xi)=\left(\frac{1-|a|^2}{|\xi-a|^2}\right)^n,\qquad \xi\in S^n.
\]

After rotation we may assume
\[
a = r e_{n+1},\qquad r\in[0,1).
\]
Let `w_\infty=-e_{n+1}` be the antipode to the mode axis and let
\[
\Pi_{w_\infty}:S^n\setminus\{w_\infty\}\to \mathbb R^n
\]
be stereographic projection from `w_\infty`, with inverse
\[
\xi(u)=\left(\frac{2u}{1+|u|^2},\,\frac{1-|u|^2}{1+|u|^2}\right),\qquad u\in\mathbb R^n.
\]

Define the Euclidean scale parameter
\[
s=\frac{1-r}{1+r}.
\]

## Proposition

**Proposition (Euclidean image of the Möbius-Poisson orbit and exact CS matching).**
Let `n\ge 1`. Under the stereographic chart `\Pi_{w_\infty}` centered on the antipode to the Poisson mode axis, the following hold.

1. **Exact Euclideanized Poisson family.**
   The pushforward `(\Pi_{w_\infty})_\#\rho_a` has density
   \[
   p_{n,s}(u)=c_n\left(\frac{s}{s^2+|u|^2}\right)^n,\qquad
   c_n=\frac{\Gamma(n)}{\pi^{n/2}\Gamma(n/2)}.
   \]
   More generally, after an Euclidean translation by `\mu\in\mathbb R^n`,
   \[
   p_{n,s,\mu}(u)=c_n\left(\frac{s}{s^2+|u-\mu|^2}\right)^n.
   \]
   Thus the Möbius orbit of the uniform boundary measure becomes a location-scale Poisson-kernel / spherical-Cauchy / multivariate-`t` family on `\mathbb R^n`.

2. **Score and Hessian of the Euclideanized Poisson family.**
   Writing
   \[
   V_{n,s,\mu}(u):=-\log p_{n,s,\mu}(u)
   = n\log\!\big(s^2+|u-\mu|^2\big)+\text{const},
   \]
   one has
   \[
   \nabla_u V_{n,s,\mu}(u)
   = \frac{2n}{s^2+|u-\mu|^2}(u-\mu),
   \]
   \[
   \nabla_u^2 V_{n,s,\mu}(u)
   = \frac{2n}{s^2+|u-\mu|^2}
   \left(
   I_n - 2\frac{(u-\mu)\otimes(u-\mu)}{s^2+|u-\mu|^2}
   \right).
   \]

3. **Two exact asymptotic regimes.**
   Let `r_u:=|u-\mu|`.

   - Near the mode, if `r_u \ll s`,
     \[
     V_{n,s,\mu}(u)
     = \text{const} + \frac{n}{s^2}r_u^2 - \frac{n}{2s^4}r_u^4 + O(r_u^6/s^6),
     \]
     so the score is linear to leading order:
     \[
     \nabla_u V_{n,s,\mu}(u)
     = \frac{2n}{s^2}(u-\mu)+O(r_u^2/s^4).
     \]
     Hence the Poisson chart is locally quadratic, i.e. it behaves like the nonsingular `\alpha=0` case near the mode.

   - In the far field, if `r_u \gg s`,
     \[
     V_{n,s,\mu}(u)=\text{const}+2n\log r_u + O(s^2/r_u^2),
     \]
     \[
     \nabla_u V_{n,s,\mu}(u)
     = \frac{2n}{r_u^2}(u-\mu) + O(s^2/r_u^3),
     \]
     \[
     \nabla_u^2 V_{n,s,\mu}(u)
     = \frac{2n}{r_u^2}
     \left(I_n-2\hat r_u\otimes\hat r_u\right)+O(s^2/r_u^4),
     \qquad \hat r_u=\frac{u-\mu}{|u-\mu|}.
     \]
     Hence the far field is logarithmic rather than power-law, formally resembling the `\alpha=2` radial Hessian law, not any fixed `0<\alpha<1`.

4. **High-dimensional shell asymptotics.**
   If `U\sim p_{n,s,\mu}` and `R=|U-\mu|`, then
   \[
   \frac{R^2}{s^2+R^2}\sim \mathrm{Beta}\!\left(\frac n2,\frac n2\right).
   \]
   Consequently,
   \[
   \frac{R}{s}\to 1 \quad \text{in probability as } n\to\infty,
   \]
   and more precisely
   \[
   \sqrt{n}\left(\frac{R}{s}-1\right)\Rightarrow \mathcal N(0,1).
   \]
   So in high dimension the Euclideanized Poisson family concentrates on a thin shell of radius `s`.

5. **Local Gaussian expansion at the mode.**
   If one zooms into the mode on the `n^{-1/2}` scale,
   \[
   u=\mu+\frac{s}{\sqrt{2n}}y,
   \]
   then
   \[
   -\log p_{n,s,\mu}(u)
   = \text{const}_n + \frac{|y|^2}{2} - \frac{|y|^4}{8n}
   + O\!\left(\frac{|y|^6}{n^2}\right).
   \]
   Hence the Poisson chart has a local Gaussian profile near the mode, but this is not a global measure-level Gaussianization because the mass still lives on the shell `R\approx s`.

6. **Exact radial transport to the weakly singular CS Gibbs family.**
   Define
   \[
   q_{\alpha,\beta,\mu}(x)=Z_{n,\alpha,\beta}^{-1}
   \exp\!\big(-\beta W_\alpha(x-\mu)\big),
   \qquad
   W_\alpha(z)=\frac{|z|^{2-\alpha}}{(2-\alpha)(1-\alpha)},
   \]
   with `\alpha\in(0,1)` and `\beta>0`. Then
   \[
   -\nabla_x \log q_{\alpha,\beta,\mu}(x)
   = \beta \nabla W_\alpha(x-\mu)
   = \frac{\beta}{1-\alpha}|x-\mu|^{-\alpha}(x-\mu),
   \]
   \[
   -\nabla_x^2 \log q_{\alpha,\beta,\mu}(x)
   = \beta D^2W_\alpha(x-\mu)
   = \frac{\beta}{1-\alpha}|x-\mu|^{-\alpha}
   \left(I_n-\alpha \hat x\otimes \hat x\right),
   \]
   for `x\neq \mu`, `\hat x=(x-\mu)/|x-\mu|`.

   If `F_{\mathrm{PK},n,s}` is the radial CDF of `p_{n,s,\mu}` and `F_{\alpha,\beta,n}` is the radial CDF of `q_{\alpha,\beta,\mu}`, then the unique monotone radial transport sending `p_{n,s,\mu}` to `q_{\alpha,\beta,\mu}` is
   \[
   T_{\alpha,\beta}(u)
   = \mu + \psi_{\alpha,\beta}(|u-\mu|)\frac{u-\mu}{|u-\mu|},
   \qquad
   \psi_{\alpha,\beta}
   = F_{\alpha,\beta,n}^{-1}\circ F_{\mathrm{PK},n,s}.
   \]

7. **Fibered phase-space version.**
   In phase space `\mathbb R^n_x\times \mathbb R^n_\omega`, if a probability measure `\boldsymbol{\mu}` has the fibered form
   \[
   \boldsymbol{\mu}(dx,d\omega)=p_{n,s(\omega),m(\omega)}(x)\,\nu(d\omega),
   \]
   then the exact fiberwise radial transport to the CS Gibbs family is
   \[
   \mathcal T_{\alpha,\beta}(x,\omega)
   =
   \left(
   m(\omega)+\psi_{\alpha,\beta,\omega}(|x-m(\omega)|)
   \frac{x-m(\omega)}{|x-m(\omega)|},
   \ \omega
   \right),
   \]
   where
   \[
   \psi_{\alpha,\beta,\omega}
   =
   F_{\alpha,\beta,n}^{-1}\circ F_{\mathrm{PK},n,s(\omega)}.
   \]
   Thus the `\omega`-marginal is preserved exactly, and only the `x`-fibers are remapped.

In particular, Möbius + Cayley/stereographic coordinates alone do **not** produce the weakly singular CS kernel with fixed `0<\alpha<1`; they produce the Euclideanized Poisson family. The fixed-`\alpha` CS kernel appears exactly only after the additional radial transport in item 6, applied fiberwise in item 7.

## Proof Sketch

### 1. Stereographic pushforward of the Poisson kernel

With `a=re_{n+1}` and `\xi=\xi(u)`, one computes
\[
|\xi(u)-re_{n+1}|^2
=
\frac{(1-r)^2+(1+r)^2|u|^2}{1+|u|^2}.
\]
The pullback of normalized spherical measure is
\[
d\sigma_n(\xi(u))
= C_n\left(\frac{2}{1+|u|^2}\right)^n du,
\]
for the appropriate normalization constant `C_n`. Multiplying by the Poisson kernel
\[
P_n(re_{n+1},\xi(u))
=
\left(
\frac{1-r^2}{|\xi(u)-re_{n+1}|^2}
\right)^n
\]
and simplifying with `s=(1-r)/(1+r)` gives
\[
(\Pi_{w_\infty})_\#\rho_a(du)
= c_n\left(\frac{s}{s^2+|u|^2}\right)^n du.
\]
The translated version is immediate.

### 2. Score and Hessian

Take minus the logarithm of `p_{n,s,\mu}`:
\[
V_{n,s,\mu}(u)=n\log(s^2+|u-\mu|^2)+\text{const}.
\]
Differentiating once gives the score, and differentiating again gives the Hessian formula in item 2.

### 3. Local and far-field regimes

For `r_u\ll s`, expand
\[
\log(s^2+r_u^2)=\log s^2 + \log\!\left(1+\frac{r_u^2}{s^2}\right)
\]
in powers of `r_u^2/s^2`. For `r_u\gg s`, expand instead
\[
\log(s^2+r_u^2)=2\log r_u + \log\!\left(1+\frac{s^2}{r_u^2}\right).
\]
Differentiation yields the score and Hessian asymptotics.

### 4. Radial law and shell asymptotics

The radial density of `R=|U-\mu|` is
\[
f_{R}(r)
=
\frac{2\Gamma(n)}{\Gamma(n/2)^2}
\frac{s^n r^{n-1}}{(s^2+r^2)^n},
\qquad r>0.
\]
Set
\[
B=\frac{R^2}{s^2+R^2}.
\]
Then a direct change of variables gives
\[
B\sim \mathrm{Beta}\!\left(\frac n2,\frac n2\right).
\]
The law of large numbers for this beta family gives `B\to 1/2`, hence `R/s\to 1`. The CLT for the beta family and the delta method for
\[
g(b)=\sqrt{\frac{b}{1-b}},\qquad g'(1/2)=2,
\]
yield
\[
\sqrt n\left(\frac{R}{s}-1\right)\Rightarrow \mathcal N(0,1).
\]

### 5. Local Gaussian zoom

Substitute `u=\mu+\frac{s}{\sqrt{2n}}y` into `V_{n,s,\mu}`:
\[
V_{n,s,\mu}(u)=\text{const}_n + n\log\!\left(1+\frac{|y|^2}{2n}\right).
\]
Now expand the logarithm:
\[
n\log\!\left(1+\frac{|y|^2}{2n}\right)
=
\frac{|y|^2}{2}
-\frac{|y|^4}{8n}
+O\!\left(\frac{|y|^6}{n^2}\right).
\]
This is the precise local Gaussian asymptotic around the mode.

### 6. Exact matching to the CS kernel

The Gibbs family `q_{\alpha,\beta,\mu}` is radial around `\mu`, and by direct differentiation its score and Hessian are exactly `\beta\nabla W_\alpha` and `\beta D^2W_\alpha`.

Since both the source `p_{n,s,\mu}` and target `q_{\alpha,\beta,\mu}` are radial about the same center, the unique monotone radial transport is obtained by matching radial quantiles:
\[
\psi_{\alpha,\beta}=F_{\alpha,\beta,n}^{-1}\circ F_{\mathrm{PK},n,s}.
\]
Applying this radius map along rays from `\mu` gives the transport `T_{\alpha,\beta}`.

### 7. Fibered phase-space transport

If the phase-space measure is fibered over `\omega`,
\[
\boldsymbol{\mu}(dx,d\omega)=\mu_\omega(dx)\,\nu(d\omega),
\]
then applying the radial transport in each `x`-fiber while leaving `\omega` unchanged preserves the `\omega`-marginal exactly. This gives the formula for `\mathcal T_{\alpha,\beta}` in item 7.

## Explicit Radial CDFs

The radial CDF of the Euclideanized Poisson family is
\[
F_{\mathrm{PK},n,s}(r)
=
I_{\frac{r^2}{s^2+r^2}}
\left(\frac n2,\frac n2\right),
\]
where `I_z(a,b)` is the regularized incomplete beta function.

For the CS Gibbs family, write
\[
m=2-\alpha,\qquad
\lambda=\frac{\beta}{(2-\alpha)(1-\alpha)}.
\]
Then the radial density is
\[
f_{\alpha,\beta,n}(r)
=
\frac{m\,\lambda^{n/m}}{\Gamma(n/m)}
r^{n-1}e^{-\lambda r^m},
\qquad r>0,
\]
and the radial CDF is
\[
F_{\alpha,\beta,n}(r)
=
\frac{\gamma\!\left(\frac n m,\lambda r^m\right)}
{\Gamma\!\left(\frac n m\right)},
\]
where `\gamma` is the lower incomplete gamma function.

Therefore the exact radius map is
\[
\psi_{\alpha,\beta}(r)
=
F_{\alpha,\beta,n}^{-1}
\!\left(
I_{\frac{r^2}{s^2+r^2}}
\left(\frac n2,\frac n2\right)
\right).
\]

## Consequence for the CS Comparison

Set `n=d` for the CS model in `\mathbb R^d`. Then:

- the Möbius-Poisson orbit becomes a Poisson-kernel / multivariate-`t` family on `\mathbb R^d`,
- its score is locally linear near the mode and logarithmic in the far field,
- its high-dimensional mass lies on a thin shell `|x-\mu|\approx s`,
- it does **not** realize the fixed weakly singular Hessian kernel
  \[
  D^2W_\alpha(x)=\frac{1}{1-\alpha}|x|^{-\alpha}
  \left(I_d-\alpha \hat x\otimes\hat x\right),
  \qquad \alpha\in(0,1),
  \]
  by stereographic / Cayley coordinates alone,
- but it can be matched exactly to that kernel by the fiberwise radial quantile transport above.

So the mathematically clean bridge is
\[
\text{Möbius-Poisson family on } S^d
\;\longrightarrow\;
\text{Poisson-kernel family on } \mathbb R^d
\;\xrightarrow{\text{radial quantile transport}}\;
\exp\!\big(-\beta W_\alpha\big)\text{-family on } \mathbb R^d,
\]
with the last step performed fiberwise in `x` when the phase space is `\mathbb R^d_x\times\mathbb R^d_\omega`.

## References

- LMS paper source:
  `/Users/adamsobieszek/pitch/pitch-website/public/notebooks/kuramoto/LMSSPP/notebooks/docs/lms_paper.tex`
- The Euclideanized family above is the Poisson-kernel / spherical-Cauchy family obtained from harmonic measure under stereographic projection.
