export type TerritoryId = "lms" | "inverse" | "construction" | "invariants" | "continuum" | "bridges";

export type NodeStatus =
  | "LMS theorem"
  | "proved extension"
  | "algorithm"
  | "diagnostic"
  | "bridge";

export type Marker = "visualize" | "play";

export type MapNode = {
  id: string;
  title: string;
  territory: TerritoryId;
  level: number;
  row: number;
  col: number;
  status: NodeStatus;
  markers?: Marker[];
  teaser: string;
  detail: string[];
  formulas: string[];
  sources: string[];
};

export type EdgeKind = "foundation" | "exact" | "algorithm" | "invariant" | "bridge";

export type MapEdge = {
  source: string;
  target: string;
  kind: EdgeKind;
  label: string;
  levelRelation: "same" | "between";
  lane?: number;
  arcSide?: "up" | "down";
};

export type Territory = {
  id: TerritoryId;
  title: string;
  color: string;
  description: string;
};

export const territories: Territory[] = [
  {
    id: "lms",
    title: "LMS foundations",
    color: "#f6b73c",
    description: "The original hyperbolic reduction and gradient engine.",
  },
  {
    id: "inverse",
    title: "Exact inverse",
    color: "#00c4bb",
    description: "Canonical Busemann centering and boost-gauge fixing.",
  },
  {
    id: "construction",
    title: "State construction",
    color: "#f58b43",
    description: "Algorithms that turn a cloud into reduced LMS state data.",
  },
  {
    id: "invariants",
    title: "Moving frames",
    color: "#b46cff",
    description: "Gauge-fixed shape invariants and connection diagnostics.",
  },
  {
    id: "continuum",
    title: "Continuum layer",
    color: "#7bd88f",
    description: "OA, Poisson kernels, and the measure/PDE interpretation.",
  },
  {
    id: "bridges",
    title: "Passages",
    color: "#6cb6ff",
    description: "Research bridges to barycenters, WS constants, and OT.",
  },
];

export const nodes: MapNode[] = [
  {
    id: "lms-root",
    title: "LMS foundations",
    territory: "lms",
    level: 0,
    row: 0,
    col: 0,
    status: "LMS theorem",
    markers: ["visualize"],
    teaser: "The base geometry behind every later construction.",
    detail: [
      String.raw`LMS supplies the geometric engine: identical-$A$ Kuramoto-on-sphere dynamics are induced by the Mobius isometry group of the Poincare ball $\Ball^d$.`,
      "Everything else on this map either chooses a canonical representative on an LMS orbit or studies what remains after that gauge choice.",
    ],
    formulas: [
      String.raw`$$x_i=M_w(p_i)\zeta^T$$`,
      String.raw`$$\dim G=\frac{d(d+1)}{2}$$`,
    ],
    sources: ["LMS.tex, Preliminaries and Reduced Equations"],
  },
  {
    id: "sphere-flow",
    title: "Sphere dynamics",
    territory: "lms",
    level: 1,
    row: 0,
    col: 1,
    status: "LMS theorem",
    teaser: "The common vector field keeps every particle on the sphere.",
    detail: [
      String.raw`The Kuramoto-on-sphere equation has a common skew term $A$ and a common mean field $Z$. Its conformal form is exactly what lets the Mobius group generate the flow.`,
    ],
    formulas: [
      String.raw`$$\dot{x}_i=x_iA^T+Z-\inner{Z}{x_i}x_i$$`,
    ],
    sources: ["LMS.tex, Preliminaries"],
  },
  {
    id: "mobius-boosts",
    title: "Mobius boosts",
    territory: "lms",
    level: 1,
    row: 0,
    col: 2,
    status: "LMS theorem",
    markers: ["visualize"],
    teaser: "The boost maps are hyperbolic isometries with simple inverses.",
    detail: [
      "Boosts move points in the ball and act on the boundary sphere. They provide the finite-dimensional coordinates used in both LMS reduction and exact Busemann inversion.",
    ],
    formulas: [
      String.raw`$$M_w(x)=\frac{(1-\norm{w}^2)(x-w)}{\norm{x-w}^2}-w$$`,
      String.raw`$$M_w^{-1}=M_{-w}$$`,
    ],
    sources: ["LMS.tex, Hyperbolic geometry"],
  },
  {
    id: "group-orbit",
    title: "Finite Mobius orbit",
    territory: "lms",
    level: 1,
    row: 0,
    col: 3,
    status: "LMS theorem",
    teaser: "The N-body trajectory remains on one group orbit.",
    detail: [
      String.raw`For generic base data $p$, the full cloud can be represented by a boost $w$ and a rotation $\zeta$. The $N$ particles are not independent degrees of freedom once the orbit is fixed.`,
    ],
    formulas: [
      String.raw`$$x_i(t)=M_{w(t)}(p_i)\zeta(t)^T$$`,
    ],
    sources: ["LMS.tex, Reduced equations"],
  },
  {
    id: "reduced-odes",
    title: "Reduced ODEs",
    territory: "lms",
    level: 2,
    row: 0,
    col: 4,
    status: "LMS theorem",
    markers: ["play"],
    teaser: "The particle system closes on boost and rotation coordinates.",
    detail: [
      String.raw`The LMS equations evolve only $w$ and $\zeta$. For linear mean-field coupling, the boost equation becomes the central low-dimensional object.`,
    ],
    formulas: [
      String.raw`$$\dot{w}=-\frac12(1-\norm{w}^2)\zeta^{-1}Z$$`,
      String.raw`$$\dot{\zeta}=(A-\alpha(\zeta w,Z))\zeta$$`,
    ],
    sources: ["LMS.tex, Reduced equations"],
  },
  {
    id: "linear-decoupling",
    title: "Linear mean-field decoupling",
    territory: "lms",
    level: 2,
    row: 0,
    col: 5,
    status: "LMS theorem",
    teaser: String.raw`For $Z=\sum_i a_i x_i$, the boost dynamics drop the rotation.`,
    detail: [
      String.raw`When $Z=\sum_i a_i x_i$, the linear order parameter respects rotations, so $\zeta$ disappears from the $w$ equation. This is the exact finite-$N$ analog of the low-dimensional collective coordinate.`,
    ],
    formulas: [
      String.raw`$$\dot{w}=-\frac12(1-\norm{w}^2)Z(M_w(p))$$`,
    ],
    sources: ["LMS.tex, Comparison of Z versus W coordinates"],
  },
  {
    id: "lms-potential",
    title: "Busemann potential",
    territory: "lms",
    level: 3,
    row: 0,
    col: 6,
    status: "LMS theorem",
    markers: ["visualize", "play"],
    teaser: "The reduced vector field is a hyperbolic gradient flow.",
    detail: [
      "The potential is the negative weighted Busemann sum attached to the base anchors. This identity is the bridge from LMS dynamics to exact inversion.",
    ],
    formulas: [
      String.raw`$$\Phi_p(w)=\sum_i a_i\log\frac{1-\norm{w}^2}{\norm{w-p_i}^2}$$`,
      String.raw`$$\grad_{\hyp}\Phi_p(w)=\frac12(1-\norm{w}^2)\sum_i a_iM_w(p_i)$$`,
    ],
    sources: ["LMS.tex, Existence of Hyperbolic Gradient"],
  },
  {
    id: "inverse-root",
    title: "Exact inverse",
    territory: "inverse",
    level: 0,
    row: 1,
    col: 0,
    status: "proved extension",
    markers: ["visualize"],
    teaser: "Use the observed cloud itself as Busemann boundary data.",
    detail: [
      "The inverse layer asks for the canonical reduced center of an observed finite cloud. It fixes the boost gauge by selecting the unique deboosted representative with zero barycenter.",
    ],
    formulas: [
      String.raw`$$\sum_i a_iM_v(x_i)=0$$`,
    ],
    sources: ["LMS_EXACT_BUSEMANN_INVERSION_NOTE.md"],
  },
  {
    id: "gauge-dependence",
    title: "w is a coordinate",
    territory: "inverse",
    level: 1,
    row: 1,
    col: 1,
    status: "proved extension",
    teaser: "The reduced center depends on the chosen base representative.",
    detail: [
      String.raw`A raw cloud does not have an intrinsic LMS $w$-coordinate. If the base representative changes by a Mobius map, the reduced coordinate changes with it.`,
    ],
    formulas: [
      String.raw`$$p'=M(p)\quad\Longrightarrow\quad w'=M(w)$$`,
    ],
    sources: ["Exact inversion note, Proposition 1"],
  },
  {
    id: "centered-slice",
    title: "Centered slice",
    territory: "inverse",
    level: 1,
    row: 1,
    col: 2,
    status: "proved extension",
    teaser: "Zero weighted barycenter fixes boost freedom.",
    detail: [
      String.raw`The slice removes the boost gauge and leaves only rotations. It is the finite-$N$ analog of choosing a canonical frame on a conformal orbit.`,
    ],
    formulas: [
      String.raw`$$S_0=\left\{p:\sum_i a_ip_i=0\right\}$$`,
    ],
    sources: ["Gauge invariants draft, Section 3"],
  },
  {
    id: "empirical-cloud",
    title: "Observed empirical measure",
    territory: "inverse",
    level: 1,
    row: 1,
    col: 3,
    status: "proved extension",
    teaser: "The whole cloud is used, not only its centroid.",
    detail: [
      String.raw`Exact inversion treats the finite weighted cloud as boundary data. This keeps the finite-$N$ problem distinct from a Poisson-shrink moment closure.`,
    ],
    formulas: [
      String.raw`$$\mu_N=\sum_i a_i\delta_{x_i}$$`,
    ],
    sources: ["Exact inversion note, Section 2"],
  },
  {
    id: "residual-equation",
    title: "Exact residual",
    territory: "inverse",
    level: 2,
    row: 1,
    col: 4,
    status: "proved extension",
    markers: ["play"],
    teaser: "The inverse center zeros the deboosted barycenter.",
    detail: [
      "Solving the residual equation means finding the boost that sends the observed cloud to the centered slice.",
    ],
    formulas: [
      String.raw`$$R_x(v)=\sum_i a_iM_v(x_i)=0$$`,
    ],
    sources: ["Exact inversion note, Definition 2"],
  },
  {
    id: "cloud-potential",
    title: "Cloud Busemann objective",
    territory: "inverse",
    level: 2,
    row: 1,
    col: 5,
    status: "proved extension",
    teaser: "The LMS potential is reused with observed anchors.",
    detail: [
      String.raw`Substituting $x_i$ for $p_i$ in the LMS gradient identity makes finite-$N$ inversion a critical-point problem for the observed cloud's Busemann potential.`,
    ],
    formulas: [
      String.raw`$$\Phi_x(v)=\sum_i a_i\log\frac{1-\norm{v}^2}{\norm{v-x_i}^2}$$`,
    ],
    sources: ["Exact inversion note, Sections 5-6"],
  },
  {
    id: "unique-center",
    title: "Unique maximizer",
    territory: "inverse",
    level: 3,
    row: 1,
    col: 6,
    status: "proved extension",
    markers: ["visualize"],
    teaser: "Under LMS assumptions, the inverse center is unique.",
    detail: [
      "Positive weights, no majority mass, and generic anchors give a unique interior critical point. That point is the canonical exact inverse center.",
    ],
    formulas: [
      String.raw`$$\max_i a_i<\frac12\quad\Longrightarrow\quad \exists!\,v_*\in\Ball^d$$`,
    ],
    sources: ["Exact inversion note, Theorem 6"],
  },
  {
    id: "mobius-equivariance",
    title: "Mobius equivariance",
    territory: "inverse",
    level: 3,
    row: 1,
    col: 7,
    status: "proved extension",
    teaser: "The center transforms naturally under ambient isometries.",
    detail: [
      "Equivariance is what makes the construction canonical rather than a coordinate artifact.",
    ],
    formulas: [
      String.raw`$$v_*(gx)=g(v_*(x))$$`,
    ],
    sources: ["Exact inversion note, Theorem 7"],
  },
  {
    id: "construction-root",
    title: "State construction",
    territory: "construction",
    level: 0,
    row: 2,
    col: 0,
    status: "algorithm",
    teaser: "Turn a cloud into exact reduced LMS initial data.",
    detail: [
      String.raw`The construction layer is operational: solve the inverse problem, store the deboosted base cloud, optionally place it at a prescribed radius, and evolve the reduced ODE for $(w,\zeta)$.`,
    ],
    formulas: [
      String.raw`$$x\longmapsto v_*(x)\longmapsto p\longmapsto (w_0,\zeta_0)$$`,
    ],
    sources: ["Exact inversion note, Algorithms 9 and 12"],
  },
  {
    id: "sign-clean",
    title: "Sign-clean variables",
    territory: "construction",
    level: 1,
    row: 2,
    col: 1,
    status: "proved extension",
    teaser: String.raw`Solve in $v=-w$, then return to LMS coordinates.`,
    detail: [
      String.raw`The variable $v$ is the deboost parameter used in the inverse solve. The reduced LMS boost coordinate is $w=-v$ at identity rotation.`,
    ],
    formulas: [
      String.raw`$$z=v_*$$`,
      String.raw`$$w_*=-v_*$$`,
    ],
    sources: ["Exact inversion note, Section 4"],
  },
  {
    id: "deboosted-base",
    title: "Deboosted base cloud",
    territory: "construction",
    level: 2,
    row: 2,
    col: 2,
    status: "proved extension",
    teaser: "The canonical base cloud lies in the centered slice.",
    detail: [
      String.raw`Once $v_*$ is found, every observed point is deboosted. The resulting finite list $p_i$ is frozen orbit data.`,
    ],
    formulas: [
      String.raw`$$p_i=M_{v_*}(x_i)=M_{-w_*}(x_i)$$`,
    ],
    sources: ["Exact inversion note, Corollary 10"],
  },
  {
    id: "local-initializer",
    title: "Finite-N initializer",
    territory: "construction",
    level: 1,
    row: 2,
    col: 3,
    status: "algorithm",
    teaser: "First and second moments give a small-center seed.",
    detail: [
      String.raw`The initializer is not the exact answer; it is a finite-$N$ local approximation used to start the nonlinear solve.`,
    ],
    formulas: [
      String.raw`$$v^{(0)}=\frac12(I-C+\lambda I)^{-1}\mu$$`,
    ],
    sources: ["Exact inversion note, Proposition 8"],
  },
  {
    id: "rapidity-solver",
    title: "Rapidity solver",
    territory: "construction",
    level: 2,
    row: 2,
    col: 4,
    status: "algorithm",
    markers: ["play"],
    teaser: "Unconstrained coordinates keep the center inside the ball.",
    detail: [
      String.raw`Rapidity coordinates let the optimizer run in $\R^d$ while the transformed iterate always remains in the Poincare ball $\Ball^d$.`,
    ],
    formulas: [
      String.raw`$$v(y)=\tanh(\norm{y})\frac{y}{\norm{y}}$$`,
    ],
    sources: ["Exact inversion note, Algorithm 9"],
  },
  {
    id: "orbit-projection",
    title: "Exact orbit projection",
    territory: "construction",
    level: 2,
    row: 2,
    col: 5,
    status: "algorithm",
    markers: ["visualize"],
    teaser: "Radius control is a move on the same Mobius orbit.",
    detail: [
      String.raw`A target radius or axis should not be forced into the inverse problem. Keep $p$ fixed and place the cloud at the desired reduced coordinate $w_{\mathrm{target}}$.`,
    ],
    formulas: [
      String.raw`$$x_i^{\mathrm{target}}=M_{w_{\mathrm{target}}}(p_i)$$`,
    ],
    sources: ["Exact inversion note, Theorem 11"],
  },
  {
    id: "finite-reconstruction",
    title: "Finite-N reconstruction",
    territory: "construction",
    level: 3,
    row: 2,
    col: 6,
    status: "algorithm",
    teaser: String.raw`Store $N$ constants once and evolve only low-dimensional state.`,
    detail: [
      "This is the Watanabe-Strogatz-style reconstruction layer for the sphere: the full cloud is recovered by one Mobius transformation and one rotation.",
    ],
    formulas: [
      String.raw`$$x_i(t)=M_{w(t)}(p_i^0)\zeta(t)^T$$`,
    ],
    sources: ["Gauge invariants draft, Section 5"],
  },
  {
    id: "invariants-root",
    title: "Moving frames",
    territory: "invariants",
    level: 0,
    row: 3,
    col: 0,
    status: "diagnostic",
    teaser: "After centering, the remaining motion is rotational.",
    detail: [
      "The moving-frame layer explains what is conserved after the boost gauge is removed: the centered shape rotates but does not strain.",
    ],
    formulas: [
      String.raw`$$D_t p_i^*=0$$`,
    ],
    sources: ["Gauge invariants draft, Sections 6-9"],
  },
  {
    id: "moving-frame",
    title: "Moving centered frame",
    territory: "invariants",
    level: 1,
    row: 3,
    col: 1,
    status: "proved extension",
    teaser: "Deboost at each time to recover the centered representative.",
    detail: [
      "The canonical center can be recomputed along a trajectory. On a single LMS orbit, it exactly removes the boost component.",
    ],
    formulas: [
      String.raw`$$p_i^*(t)=M_{z(t)}(x_i(t))$$`,
    ],
    sources: ["Gauge invariants draft, Section 6"],
  },
  {
    id: "covariant-constant",
    title: "Covariantly constant shape",
    territory: "invariants",
    level: 2,
    row: 3,
    col: 2,
    status: "proved extension",
    markers: ["visualize"],
    teaser: "The centered shape is the stored base cloud rotated.",
    detail: [
      "After exact Busemann centering, every particle in the centered frame evolves by the same rotation.",
    ],
    formulas: [
      String.raw`$$p_i^*(t)=p_i^0\zeta(t)^T$$`,
    ],
    sources: ["Gauge invariants draft, Section 6"],
  },
  {
    id: "gram-invariants",
    title: "Gram invariants",
    territory: "invariants",
    level: 2,
    row: 3,
    col: 3,
    status: "proved extension",
    teaser: "Pairwise centered inner products are conserved.",
    detail: [
      String.raw`The Gram matrix is the most direct finite-$N$ shape invariant left by the canonical moving frame.`,
    ],
    formulas: [
      String.raw`$$G_{ij}(t)=\inner{p_i^*(t)}{p_j^*(t)}$$`,
    ],
    sources: ["Gauge invariants draft, Section 7"],
  },
  {
    id: "moment-invariants",
    title: "Moment spectra",
    territory: "invariants",
    level: 2,
    row: 3,
    col: 4,
    status: "proved extension",
    teaser: "Weighted moment tensors rotate covariantly.",
    detail: [
      "Inertia and higher moment tensors do change by rotation, but their spectra and rotational contractions are conserved.",
    ],
    formulas: [
      String.raw`$$T=\sum_i a_i\,p_i^{*\top}p_i^*$$`,
    ],
    sources: ["Gauge invariants draft, Section 7"],
  },
  {
    id: "connection",
    title: "Connection term",
    territory: "invariants",
    level: 2,
    row: 3,
    col: 5,
    status: "proved extension",
    markers: ["play"],
    teaser: "Changing boosts induce rotational precession.",
    detail: [
      "The connection term is not an extra force. It is the curvature of the moving boost frame, analogous to Thomas precession.",
    ],
    formulas: [
      String.raw`$$\Omega=A+\alpha(z,Z)$$`,
    ],
    sources: ["Gauge invariants draft, Section 8"],
  },
  {
    id: "data-diagnostics",
    title: "Data diagnostics",
    territory: "invariants",
    level: 3,
    row: 3,
    col: 6,
    status: "diagnostic",
    teaser: "Estimate the connection and detect non-LMS strain.",
    detail: [
      "Centered trajectory data should have a skew connection and negligible symmetric strain if the motion is exactly LMS.",
    ],
    formulas: [
      String.raw`$$C=\Omega T$$`,
      String.raw`$$\operatorname{skew}(CT^+)\quad\mathrm{vs.}\quad A+\alpha(z,Z)$$`,
    ],
    sources: ["Gauge invariants draft, Section 9"],
  },
  {
    id: "local-source",
    title: "Local source exponents",
    territory: "invariants",
    level: 3,
    row: 4,
    col: 2,
    status: "diagnostic",
    teaser: "The inertia tensor controls the local boost instability.",
    detail: [
      String.raw`Near a centered representative, the reduced boost flow linearizes through $I-T$. This gives a local finite-$N$ stability diagnostic.`,
    ],
    formulas: [
      String.raw`$$\dot{u}=u(I-T)+O(\norm{u}^2)$$`,
    ],
    sources: ["Gauge invariants draft, Section 10"],
  },
  {
    id: "euler-sundman",
    title: "Euler-Sundman lift",
    territory: "invariants",
    level: 3,
    row: 4,
    col: 4,
    status: "diagnostic",
    teaser: "A time rescaling gives a Euclidean gradient diagnostic.",
    detail: [
      "The Euler-Sundman form is mainly interpretive: it turns the hyperbolic gradient branch into a Euclidean gradient equation after time reparametrization.",
    ],
    formulas: [
      String.raw`$$\frac{\dd w}{\dd\tau}=-\grad_{\euc}\Phi_P(w)$$`,
    ],
    sources: ["Gauge invariants draft, Section 11"],
  },
  {
    id: "continuum-root",
    title: "Continuum layer",
    territory: "continuum",
    level: 0,
    row: 5,
    col: 0,
    status: "bridge",
    teaser: "The finite orbit story becomes a measure flow.",
    detail: [
      "The same Mobius action pushes forward probability measures. The OA manifold appears as the orbit of the uniform boundary measure.",
    ],
    formulas: [
      String.raw`$$\rho\longmapsto g_*\rho$$`,
    ],
    sources: ["LMS.tex, Continuum limit"],
  },
  {
    id: "measure-action",
    title: "Measure pushforward",
    territory: "continuum",
    level: 1,
    row: 5,
    col: 1,
    status: "LMS theorem",
    teaser: "The group action extends from particles to measures.",
    detail: [
      "Finite clouds are empirical measures. In the continuum limit, Mobius transformations act by pushforward on probability measures over the sphere.",
    ],
    formulas: [
      String.raw`$$\int f\dd(g_*\rho)=\int f(gx)\dd\rho(x)$$`,
    ],
    sources: ["LMS.tex, Continuum limit"],
  },
  {
    id: "continuity-pde",
    title: "Continuity PDE",
    territory: "continuum",
    level: 1,
    row: 5,
    col: 2,
    status: "bridge",
    teaser: "The continuum model is transport by the same vector field.",
    detail: [
      "The PDE viewpoint keeps the geometry but trades particle coordinates for a boundary density or measure.",
    ],
    formulas: [
      String.raw`$$\partial_t\rho+\operatorname{div}_{\Sph^{d-1}}(\rho\,v[\rho])=0$$`,
    ],
    sources: ["LMS.tex, Continuum limit"],
  },
  {
    id: "oa-manifold",
    title: "OA / Poisson orbit",
    territory: "continuum",
    level: 2,
    row: 5,
    col: 3,
    status: "LMS theorem",
    markers: ["visualize"],
    teaser: "The uniform measure orbit is the OA-type manifold.",
    detail: [
      String.raw`Rotations stabilize the uniform measure, so its Mobius orbit is only $d$-dimensional and is parametrized by the boost center $z$.`,
    ],
    formulas: [
      String.raw`$$\rho_z=(M_{-z})_*\sigma$$`,
    ],
    sources: ["LMS.tex, Continuum limit"],
  },
  {
    id: "hyperbolic-poisson",
    title: "Hyperbolic Poisson kernel",
    territory: "continuum",
    level: 2,
    row: 5,
    col: 4,
    status: "LMS theorem",
    teaser: "The real higher-dimensional kernel is hyperbolic.",
    detail: [
      String.raw`The real $d$-dimensional OA analog uses the hyperbolic Poisson kernel. It agrees with the Euclidean Poisson picture only in $d=2$.`,
    ],
    formulas: [
      String.raw`$$P_{\hyp}(z,x)=\left(\frac{1-\norm{z}^2}{\norm{z-x}^2}\right)^{d-1}$$`,
    ],
    sources: ["LMS.tex, Continuum limit"],
  },
  {
    id: "centroid-gap",
    title: "Centroid closure gap",
    territory: "continuum",
    level: 3,
    row: 5,
    col: 5,
    status: "LMS theorem",
    teaser: String.raw`For $d\ge 3$, centroid radius is not simply the boost center.`,
    detail: [
      String.raw`This is why the exact finite-$N$ inverse should not be reduced to a first-moment shrink rule.`,
    ],
    formulas: [
      String.raw`$$Z(z)=K\,h_d(\norm{z}^2)z$$`,
    ],
    sources: ["LMS.tex, Equation Zevaluated"],
  },
  {
    id: "empirical-limit",
    title: "Empirical-to-continuum limit",
    territory: "continuum",
    level: 3,
    row: 5,
    col: 6,
    status: "bridge",
    teaser: String.raw`Finite-$N$ inversion discretizes a boundary Busemann functional.`,
    detail: [
      "Replacing the empirical sum by an integral suggests a continuum conformal barycenter problem for boundary measures.",
    ],
    formulas: [
      String.raw`$$\Phi_\mu(v)=\int\log\frac{1-\norm{v}^2}{\norm{v-x}^2}\dd\mu(x)$$`,
    ],
    sources: ["Synthesis from LMS and exact inversion notes"],
  },
  {
    id: "bridges-root",
    title: "Passages",
    territory: "bridges",
    level: 0,
    row: 6,
    col: 0,
    status: "bridge",
    teaser: "Where the finite exact gauge touches broader frameworks.",
    detail: [
      "These are research bridges rather than claims with the same status as LMS theorems. They organize possible future mathematical language.",
    ],
    formulas: [
      String.raw`$$\text{Busemann center}+\text{orbit projection}+\text{invariants}$$`,
    ],
    sources: ["Synthesis"],
  },
  {
    id: "conformal-barycenter",
    title: "Conformal barycenter",
    territory: "bridges",
    level: 1,
    row: 6,
    col: 1,
    status: "bridge",
    teaser: "Busemann inversion is a boundary barycenter equation.",
    detail: [
      "The exact inverse center can be read as the point where the average boosted boundary direction vanishes.",
    ],
    formulas: [
      String.raw`$$\int M_v(x)\dd\mu(x)=0$$`,
    ],
    sources: ["Synthesis from Busemann inversion"],
  },
  {
    id: "moment-map",
    title: "Moment-map reading",
    territory: "bridges",
    level: 1,
    row: 6,
    col: 2,
    status: "bridge",
    teaser: "The centered slice resembles a zero-moment condition.",
    detail: [
      "This language clarifies why centering is a gauge condition: the chosen representative lands at zero first moment after a group action.",
    ],
    formulas: [
      String.raw`$$J(g_*\mu)=\int x\dd(g_*\mu)=0$$`,
    ],
    sources: ["Synthesis"],
  },
  {
    id: "ot-orbit-projection",
    title: "Orbit projection",
    territory: "bridges",
    level: 2,
    row: 6,
    col: 3,
    status: "bridge",
    teaser: "Radius control is projection inside the Mobius orbit.",
    detail: [
      "The exact placement step is not arbitrary particle transport. It is a finite-dimensional orbit move after the canonical base has been chosen.",
    ],
    formulas: [
      String.raw`$$\mu\longmapsto (M_{w_{\mathrm{target}}})_*p$$`,
    ],
    sources: ["Exact inversion note, reframed"],
  },
  {
    id: "ot-metric-question",
    title: "OT metric question",
    territory: "bridges",
    level: 3,
    row: 6,
    col: 4,
    status: "bridge",
    teaser: "Compare Busemann projection with Wasserstein projection.",
    detail: [
      "A natural follow-up is to ask whether the canonical gauge has a variational interpretation in a transport metric on boundary measures.",
    ],
    formulas: [
      String.raw`$$\operatorname*{argmin}_{g\in G} W_2(g_*\mu,S_0)\;?$$`,
    ],
    sources: ["New synthesis"],
  },
  {
    id: "ws-constants",
    title: "WS constants in d>2",
    territory: "bridges",
    level: 2,
    row: 6,
    col: 5,
    status: "bridge",
    teaser: "Gram and moment invariants replace circle cross-ratios.",
    detail: [
      "After canonical centering, higher-dimensional shape invariants play the role that Watanabe-Strogatz constants play on the circle.",
    ],
    formulas: [
      String.raw`$$d=2:\;\text{cross-ratios}$$`,
      String.raw`$$d>2:\;\text{Gram and moment invariants}$$`,
    ],
    sources: ["Gauge invariants draft, Section 7"],
  },
  {
    id: "gradient-stability",
    title: "Boundary stability logic",
    territory: "bridges",
    level: 3,
    row: 6,
    col: 6,
    status: "bridge",
    markers: ["visualize"],
    teaser: "LMS stability, inverse uniqueness, and source exponents share one potential geometry.",
    detail: [
      "The same Busemann potential explains forward synchronization, backward centered limits, uniqueness of the inverse center, and local source exponents.",
    ],
    formulas: [
      String.raw`$$\Phi\Longrightarrow\text{unique interior source}\Longrightarrow\text{boundary synchrony}$$`,
    ],
    sources: ["Synthesis from LMS and both notes"],
  },
];

export const edges: MapEdge[] = [
  { source: "lms-root", target: "sphere-flow", kind: "foundation", label: "sets equation", levelRelation: "same", arcSide: "up" },
  { source: "sphere-flow", target: "mobius-boosts", kind: "foundation", label: "infinitesimal generators", levelRelation: "same", arcSide: "up" },
  { source: "mobius-boosts", target: "group-orbit", kind: "foundation", label: "builds orbit", levelRelation: "same", arcSide: "down" },
  { source: "group-orbit", target: "reduced-odes", kind: "foundation", label: "coordinates", levelRelation: "same", arcSide: "up" },
  { source: "reduced-odes", target: "linear-decoupling", kind: "foundation", label: String.raw`linear $Z$`, levelRelation: "same", arcSide: "down" },
  { source: "linear-decoupling", target: "lms-potential", kind: "foundation", label: "gradient identity", levelRelation: "same", arcSide: "up" },
  { source: "inverse-root", target: "gauge-dependence", kind: "exact", label: "problem statement", levelRelation: "same", arcSide: "up" },
  { source: "gauge-dependence", target: "centered-slice", kind: "exact", label: "fix boost", levelRelation: "same", arcSide: "up" },
  { source: "centered-slice", target: "residual-equation", kind: "exact", label: "zero barycenter", levelRelation: "same", arcSide: "down" },
  { source: "empirical-cloud", target: "residual-equation", kind: "exact", label: "full cloud", levelRelation: "same", arcSide: "up" },
  { source: "lms-potential", target: "cloud-potential", kind: "exact", label: "replace anchors", levelRelation: "between", lane: 1 },
  { source: "cloud-potential", target: "residual-equation", kind: "exact", label: String.raw`critical iff $R=0$`, levelRelation: "same", arcSide: "down" },
  { source: "lms-potential", target: "unique-center", kind: "exact", label: "LMS uniqueness", levelRelation: "between", lane: -1 },
  { source: "residual-equation", target: "unique-center", kind: "exact", label: "one solution", levelRelation: "same", arcSide: "up" },
  { source: "unique-center", target: "mobius-equivariance", kind: "exact", label: "canonical", levelRelation: "same", arcSide: "down" },
  { source: "unique-center", target: "sign-clean", kind: "algorithm", label: String.raw`choose $v=-w$`, levelRelation: "between", lane: 0 },
  { source: "sign-clean", target: "deboosted-base", kind: "algorithm", label: String.raw`recover $p$`, levelRelation: "same", arcSide: "up" },
  { source: "local-initializer", target: "rapidity-solver", kind: "algorithm", label: "seed solve", levelRelation: "same", arcSide: "up" },
  { source: "rapidity-solver", target: "unique-center", kind: "algorithm", label: "numerical critical point", levelRelation: "between", lane: -2 },
  { source: "deboosted-base", target: "orbit-projection", kind: "algorithm", label: "same orbit", levelRelation: "same", arcSide: "down" },
  { source: "deboosted-base", target: "finite-reconstruction", kind: "algorithm", label: "stored constants", levelRelation: "same", arcSide: "up" },
  { source: "reduced-odes", target: "finite-reconstruction", kind: "foundation", label: "evolve reduced state", levelRelation: "between", lane: 2 },
  { source: "finite-reconstruction", target: "moving-frame", kind: "invariant", label: "deboost each time", levelRelation: "between", lane: -1 },
  { source: "moving-frame", target: "covariant-constant", kind: "invariant", label: "canonical theorem", levelRelation: "same", arcSide: "up" },
  { source: "covariant-constant", target: "gram-invariants", kind: "invariant", label: "rotational invariant", levelRelation: "same", arcSide: "down" },
  { source: "covariant-constant", target: "moment-invariants", kind: "invariant", label: "tensor covariance", levelRelation: "same", arcSide: "up" },
  { source: "reduced-odes", target: "connection", kind: "invariant", label: "rotation equation", levelRelation: "between", lane: -3 },
  { source: "connection", target: "data-diagnostics", kind: "invariant", label: "estimate/check", levelRelation: "same", arcSide: "down" },
  { source: "moment-invariants", target: "local-source", kind: "invariant", label: "inertia controls", levelRelation: "between", lane: 1 },
  { source: "lms-potential", target: "euler-sundman", kind: "invariant", label: "time rescaling", levelRelation: "between", lane: 3 },
  { source: "group-orbit", target: "measure-action", kind: "bridge", label: String.raw`$N$ to measure`, levelRelation: "between", lane: -2 },
  { source: "sphere-flow", target: "continuity-pde", kind: "bridge", label: "transport limit", levelRelation: "between", lane: 2 },
  { source: "measure-action", target: "oa-manifold", kind: "foundation", label: "uniform orbit", levelRelation: "same", arcSide: "up" },
  { source: "oa-manifold", target: "hyperbolic-poisson", kind: "foundation", label: "density formula", levelRelation: "same", arcSide: "down" },
  { source: "hyperbolic-poisson", target: "centroid-gap", kind: "foundation", label: String.raw`$d\ge3$ correction`, levelRelation: "same", arcSide: "up" },
  { source: "cloud-potential", target: "empirical-limit", kind: "bridge", label: "empirical integral", levelRelation: "between", lane: -1 },
  { source: "empirical-limit", target: "conformal-barycenter", kind: "bridge", label: "boundary barycenter", levelRelation: "between", lane: 0 },
  { source: "centered-slice", target: "moment-map", kind: "bridge", label: "zero moment", levelRelation: "between", lane: 2 },
  { source: "orbit-projection", target: "ot-orbit-projection", kind: "bridge", label: "orbit move", levelRelation: "between", lane: -1 },
  { source: "ot-orbit-projection", target: "ot-metric-question", kind: "bridge", label: String.raw`compare with $W_2$`, levelRelation: "same", arcSide: "up" },
  { source: "gram-invariants", target: "ws-constants", kind: "bridge", label: "shape signatures", levelRelation: "between", lane: 1 },
  { source: "moment-invariants", target: "ws-constants", kind: "bridge", label: "higher moments", levelRelation: "between", lane: -1 },
  { source: "lms-potential", target: "gradient-stability", kind: "bridge", label: "same potential", levelRelation: "between", lane: 3 },
  { source: "unique-center", target: "gradient-stability", kind: "bridge", label: "inverse uniqueness", levelRelation: "between", lane: -2 },
  { source: "local-source", target: "gradient-stability", kind: "bridge", label: "local source", levelRelation: "between", lane: 2 },
];
