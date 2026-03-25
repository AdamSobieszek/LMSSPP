
**1. What the “Poisson restriction” is**
Let $\mathcal P(S^{d-1})$ be all probability measures on the sphere.  
Define
$$
\mathcal M_P := \{\rho_z : z\in B^d\},\qquad
d\rho_z(x)=P_{\mathrm{hyp}}(z,x)\,d\sigma(x),
$$
with $\sigma$ uniform and
$$
P_{\mathrm{hyp}}(z,x)=\left(\frac{1-|z|^2}{|z-x|^2}\right)^{d-1}.
$$
$\mathcal M_P$ is a $d$-dimensional nonlinear submanifold of $\mathcal P(S^{d-1})$ (not a linear subspace).

**2. Orbit characterization (your first heuristic)**
Yes, exactly:
$$
\mathcal M_P = G\cdot \sigma
$$
where $G=\mathrm{Mob}^+(B^d)$ acts by pushforward on measures.  
Reason:
- every $g\in G$ decomposes as $g=\zeta M_w$, with $\zeta\in SO(d)$, $w\in B^d$,
- $\zeta_*\sigma=\sigma$, so $g_*\sigma=(M_w)_*\sigma=(M_{-z})_*\sigma$,
- $(M_{-z})_*\sigma$ has density $P_{\mathrm{hyp}}(z,\cdot)$ w.r.t. $\sigma$.
So “Poisson kernels are exactly the orbit of uniform measure” is correct.

**3. Closure relation on that manifold**
On $\mathcal M_P$,
$$
Z(z)=K\int x\,d\rho_z(x)=K\,f_d(|z|)\,z,
$$
with
$$
f_d(r)=\frac{F(1,1-d/2;1+d/2;r^2)}{F(1,1-d/2;1+d/2;1)}.
$$
Outside $\mathcal M_P$, this formula is not exact in general.

**4. Dirichlet principle statement (your item 1)**
Correct:
- For boundary trace $f$, the hyperbolic harmonic extension $u$ minimizes $E_{\mathrm{hyp}}(u)$ among all extensions with same trace.
- The Poisson kernel is the representing kernel of that minimizer:
$$
u(z)=\int_{S^{d-1}} f(x)\,P_{\mathrm{hyp}}(z,x)\,d\sigma(x).
$$

**5. Fisher compatibility statement (your item 2)**
Also correct, with precise wording:
- On the statistical manifold $z\mapsto \rho_z$, define Fisher–Rao metric
$$
g^{\mathrm F}_{ij}(z)=\int \partial_i\log p_z(x)\,\partial_j\log p_z(x)\,p_z(x)\,d\sigma(x),\quad p_z=P_{\mathrm{hyp}}(z,\cdot).
$$
- Then
$$
g^{\mathrm F} = c_d\, g_{\mathrm hyp}
$$
for some $c_d>0$ depending only on dimension.
So the Poisson family is information-geometrically hyperbolic (up to scale).

**6. Are these the only distributions with that Fisher/hyperbolic property?**
- As a global claim over all families in $\mathcal P(S^{d-1})$: **not strictly unique without extra assumptions**.
- Under natural symmetry assumptions (Möbius-equivariant $d$-parameter family through $\sigma$, full isotropy at $\sigma$): this Poisson orbit is the canonical/essentially unique one (up to reparameterization and scale).

**7. Finite $N$ vs infinite $N$**
- Exact $\rho_z\in\mathcal M_P$ is measure-level (continuum).
- At finite $N$, you only get an empirical approximation:
$$
\mu_N=\frac1N\sum_{i=1}^N \delta_{x_i}\approx \rho_z.
$$
So “Poisson init” in code means sampling from that family, not exact equality.

Great i have also this list of explanations from an old chat can you read and inspect relevant code fragments to make sure you understand our project?:

- You are continuing development of a Kuramoto-on-sphere LMS codebase in `/notebooks/kuramoto/LMS.py` and `notebooks/kuramoto/lms_ball3d_widget.py`; assume the main interactive target is the 3D Plotly widget for $S^2/B^3$, with a backward two-sheet subclass.

- Mathematical state space and variables: $x_i(t)\in S^{d-1}\subset\mathbb R^d$, reduced coordinates $(w,\zeta)\in B^d\times SO(d)$, lab-frame center $z=-\zeta w$, order parameter $Z\in\mathbb R^d$, and body-frame order parameter $Z_{\text{body}}=\zeta^{-1}Z$.

- Real Möbius boost on $B^d$:  
  $$
  M_w(x)=\frac{(1-|w|^2)(x-|x|^2w)}{1-2\langle w,x\rangle+|w|^2|x|^2}-w,
  $$
  with sphere restriction ($|x|=1$):  
  $$
  M_w(x)=\frac{(1-|w|^2)(x-w)}{|x-w|^2}-w.
  $$
  Computationally this is implemented in `mobius_ball` / `mobius_sphere` in `LMS.py`.

- Reduced observables are computed from base points $p_i$: $x_i^{\text{body}}=M_w(p_i)$, $x_i^{\text{lab}}=\zeta x_i^{\text{body}}$,  
  $$
  Z_{\text{body}}=K\sum_i a_i x_i^{\text{body}},\quad Z=Z_{\text{body}}\zeta^\top,\quad z=-w\zeta^\top
  $$
  (row-vector convention in code).

- Reduced dynamics used in the integrator:  
  $$
  \dot w=-\frac12(1-|w|^2)\,Z_{\text{body}}\quad\text{(explicit mode)},
  $$
  $$
  \dot\zeta=(A-\alpha(\zeta w,Z))\zeta,\quad \alpha(u,v)y=\langle u,y\rangle v-\langle v,y\rangle u.
  $$

- Autograd backend for $\dot w$: define hyperbolic potential  
  $$
  \Phi(w)=\sum_i a_i\log\frac{1-|w|^2}{|w-p_i|^2},
  $$
  compute Euclidean gradient $\nabla_{\text{euc}}\Phi$ by PyTorch autograd, convert to hyperbolic gradient  
  $$
  \nabla_{\text{hyp}}\Phi=\frac{(1-|w|^2)^2}{4}\nabla_{\text{euc}}\Phi,
  $$
  then $\dot w=-\nabla_{\text{hyp}}\Phi$.

- Time integration is Euler in `integrate_lms_reduced_euler`, with optional `project_to_so` re-orthogonalization of $\zeta$, clamping $w$ to the open ball, and optional storage modes (`store_points` / trajectory mode memory-vs-fps).

- For visualization, $d=3$ is used in the widget, but the math/utilities are dimension-general in `LMS.py`.

- Important closure equation from LMS continuum Poisson manifold:  
  $$
  Z(z)=K\,f_d(|z|)\,z,\quad
  f_d(r)=\frac{F(1,1-d/2;1+d/2;r^2)}{F(1,1-d/2;1+d/2;1)}.
  $$
  In code this appears as `_shrink_fd` and uses a series for $F(1,b;c;u)$ plus gamma-ratio at $u=1$.

- Distinguish clearly in diagnostics: top plot `|Z|/K` is empirical finite-$N$ from simulated points; thermodynamic “conformal” line uses $f_d(|z|)|z|$, i.e. closure prediction; they differ off Poisson manifold / at finite $N$.

- Current thermodynamic metrics include empirical barycenter norm $||\mu||$, conformal proxy $f_d(|z|)|z|$, variance to empirical center, variance to conformal-rescaled center, and entropy proxy from anisotropy decomposition (parallel/perpendicular variance wrt axis from $z$ or fallback).

- Initial-state control is now 3-mode cyclic (`high`, `low`, `poisson`) through one button:
  - `high`: optimize boundary cloud to match target $r_0$ while maximizing entropy-like spread term.
  - `low`: optimize boundary cloud to match $r_0$ while minimizing entropy-like spread term.
  - `poisson`: finite-$N$ Poisson-manifold construction: sample $p_i\sim \sigma$ uniformly on $S^{d-1}$, set $w_0=r_0\hat u$, and use $x_i(0)=M_{w_0}(p_i)$ implicitly through reduced state; this is the finite-$N$ approximation to $(M_{-z})_*\sigma$.

- High/low initialization details: start from random uniform cloud, warm-start dipole toward target direction, optimize points on sphere with Adam under radius/direction loss plus entropy term, refine radius lock, then estimate $w_0$ from first moment by numerically inverting $q(r)=f_d(r)r$, recover base points by inverse Möbius pushforward.

- Entropy-direction toggle changes effective sign of conformal attraction by using `coupling = conformal_sign * K` with `conformal_sign = +1` (dissipate) or `-1` (increase), while `dt` remains independently user-controlled.

- Two-sheet backward subclass `LMSBall3DBackwardTwoSheetWidget` adds:
  - optional forced backward simulation (`dt -> -|dt|`),
  - outer-sheet map
    $$
    \bar x=\frac{x/|x|-x}{|x/|x|-x|^2}
    $$
    with near-zero fallback direction and radius cap for rendering,
  - extra traces for $\bar w,\bar z,\overline{Z/K}$ markers/paths/vectors.

- Row-2 plot in two-sheet subclass was changed from xyz-vs-frame to planar projection diagnostics: define
  $$
  e_\infty=\frac{w(T)}{|w(T)|},\quad e_\perp=\text{GS}\!\left(\frac{w(0)}{|w(0)|}\perp e_\infty\right),
  $$
  then plot for each of $w,z,Z/K$:
  $$
  (\langle \cdot,e_\infty\rangle,\ \langle \cdot,e_\perp\rangle)
  $$
  as parametric curves with current-frame markers.

- UI/interaction requirements already implemented: forward/reverse play, drag-pause camera behavior, layout toggle, frame toggle (lab/co-rotating), speed scaling, recompute, and robust path decimation during play.

- Critical stability fix already applied: when updating Plotly `FigureWidget` traces, assign Python lists (`.tolist()`) rather than raw numpy arrays to avoid ipywidgets/plotly `ValueError: truth value of an array is ambiguous` in `_remove_overlapping_props`.

- Conceptual interpretation required in future reasoning: “Poisson base cloud” means empirical approximation to the Möbius orbit of uniform measure $G\cdot \sigma$, not a Poisson point process; exact closure is continuum-level, finite $N$ only approximates it.

- When evaluating discrepancies, separate three causes: finite-$N$ sampling error, off-manifold initialization/dynamics, and frame/proxy mismatch (empirical $||Z||/K$ vs closure $f_d(|z|)|z|$).

- Maintain row-vector algebra conventions used in code ($x @ \zeta^\top$, etc.); if proposing formulas in column-vector form, explicitly convert to avoid sign/transposition mistakes.

- Preserve existing user-facing controls and defaults unless asked; prefer subclass-specific modifications over global behavior changes.

- If adding new diagnostics, prioritize closure error and manifold distance measures, e.g. $\delta_{\text{closure}}(t)=\big||Z|/K-f_d(|z|)|z|\big|$, and visualize against init-state mode.

- Keep mathematical statements strict: $\mathcal M_P=\{(M_{-z})_*\sigma:z\in B^d\}\subset\mathcal P(S^{d-1})$ is a finite-dimensional manifold (not linear subspace), Dirichlet principle gives minimal hyperbolic energy extension, and Fisher metric on Poisson family is hyperbolic up to scale.

- Treat this as an advanced research-grade code+math session: any change should explicitly map equation $\to$ variable $\to$ tensor operation $\to$ plotted observable.



# Reasons why each element was implemented this way

- The reduced $(w,\zeta)$ formulation was implemented instead of full $N(d-1)$-dimensional state integration because LMS reduction is exact for identical rotation term $A$ and linear mean-field coupling, yielding dramatic computational savings while preserving the Möbius-orbit dynamics; this keeps parameter scans and interactive recomputes feasible.

- The code keeps both `explicit` and `autograd` backends for $\dot w$ because they serve different verification roles: `explicit` is direct from reduced equations, while `autograd` enforces the hyperbolic-gradient construction from $\Phi$, giving a structural check that numerics match the intended geometric variational flow.

- `z`, `w`, and `Z` are all displayed separately because conflating them hides frame effects and closure gaps; in $d>2$, $z$ is not itself the centroid order parameter, so plotting all three avoids importing $d=2$ intuition incorrectly.

- The shrink factor $f_d$ is implemented via hypergeometric ratio rather than ad hoc fit/interpolation because the LMS continuum formula is explicit, low-dimensional, cheap to evaluate, and provides a principled “closure baseline” for diagnostics.

- A preallocated trajectory container (`integrate_lms_reduced_euler` with optional `store_points`) is used instead of list/append stacks because it reduces memory churn, enables memory/fps modes, and supports deterministic shape contracts needed by interactive widgets.

- Memory mode reconstructs points from reduced state on demand because storing $[T,N,3]$ clouds dominates RAM at large $N,T$; the tradeoff is extra per-frame compute, but this is predictable and bounded.

- Rotation re-projection (`project_to_so`) is included because explicit Euler on matrix ODE drifts off $SO(3)$; projecting avoids long-run geometric inconsistency that would contaminate frame transforms and displayed observables.

- `dt` is left user-controlled (including sign) while entropy direction is a separate conformal-force sign control because these are conceptually different operations: one changes time parametrization/orientation, the other changes flow direction relative to potential geometry.

- The 3-state init mode (`high`, `low`, `poisson`) exists because a binary high/low entropy control could not express manifold-consistent initializations; `poisson` explicitly targets the LMS closure manifold approximation at finite $N$, while high/low preserve exploratory off-manifold behavior.

- High/low initializations are optimization-based rather than closed-form because simultaneously matching target $r_0$, direction bias, and anisotropy objectives under sphere constraints is multi-objective and nonlinear; optimization gives practical control that simple templates do not.

- Poisson initialization is implemented as finite-$N$ sampling from uniform base points plus prescribed reduced center $w_0$ because exact Poisson measures are continuum objects; this is the least assumption-heavy finite approximation consistent with group-orbit construction.

- The two-sheet map for backward widget was added because near-origin dynamics in $B^3$ are visually compressed; inversion-like outer-sheet embedding separates trajectories that otherwise overlap visually, improving qualitative analysis of backward-time structure.

- Row-2 projection-plane diagnostics in two-sheet mode use basis from $w_0,w_\infty$ with Gram-Schmidt because xyz components are coordinate-artifacts; the chosen basis is dynamically meaningful (asymptotic axis + orthogonal transient axis).

- Plotly updates now coerce arrays to Python lists because `FigureWidget` trait syncing is brittle with numpy arrays in delta-merging paths; list coercion is a pragmatic reliability fix, not a mathematical requirement.

- Camera drag handling pauses playback because continuous server-driven frame pushes fight manual camera edits in notebook comm channels; pausing on edit prioritizes user control and restores smooth interaction.

- Separate empirical and conformal diagnostics are retained (e.g., `|Z|/K` vs $f_d(|z|)|z|$) because forcing them to coincide in UI would hide closure error, which is scientifically relevant (finite-$N$, off-manifold, non-Poisson effects).

- Entropy proxy uses parallel/perpendicular variance decomposition rather than only histogram entropy because it directly relates to components that do/do not contribute to dipole alignment and is cheaper/more stable frame-to-frame for online plotting.

- Backward widget subclasses the forward widget instead of duplicating code because most mechanics (controls, rendering, caching, event flow) are shared; this reduces regression surface when core behavior changes.

- Static export + backend recompute architecture exists because pure static HTML cannot recompute trajectories; dual-mode keeps embeddability while preserving scientific interactivity where a Python backend is available.

- Domain-specific labels and notation were gradually tightened because misuse of “phase” and $d=2$-centric wording caused conceptual confusion in $d=3$; explicit geometric naming helps maintain mathematical correctness in interpretation.

**Potential improvements (currently no fully satisfactory reason to keep as-is)**

- Replace Euler with adaptive integrators (or at least RK variants for $(w,\zeta)$) to reduce discretization bias near boundary and in high-curvature regimes.

- Use manifold-aware Lie-group integration for $\zeta$ (exponential map / Cayley updates) instead of project-after-step, which is numerically serviceable but not structure-preserving in a principled sense.

- Improve finite-$N$ Poisson initialization by directly sampling from $P_{\mathrm{hyp}}(z,\cdot)$ (importance or rejection on sphere) rather than relying mainly on reduced-state construction plus uniform base sampling.

- Make closure-error diagnostics first-class (`delta_closure`) and expose them in plots/stats to quantify when conformal proxy is trustworthy.

- Formalize manifold-distance metrics from empirical measure to Poisson family (e.g., moment mismatch or likelihood under Poisson kernel) rather than proxy-only entropy heuristics.

- Revisit high/low init optimization objective weights (currently tuned heuristically); derive scale-normalized objectives tied to theoretical invariants.

- Unify sign conventions and row/column vector notation across docs/code to remove residual cognitive overhead and risk of subtle transposition/sign bugs.

- Add rigorous regression tests for: invariance properties, frame transforms, closure consistency in Poisson mode, and long-run drift behavior under different backends/integrators.

- Reduce widget recompute latency further with partial recomputation and incremental cache invalidation instead of full rebuild on most control changes.

- Move heavy per-frame math client-side for playback-only mode (where feasible) to cut notebook comm overhead and improve UI smoothness at high frame rates.

- Explicitly separate exploratory “visual hacks” (outer sheet, capped radii) from physically meaningful observables in UI state to prevent interpretation drift.

- Validate entropy proxy choices against alternative information-theoretic estimators on $S^2$ to ensure conclusions are not estimator-dependent.