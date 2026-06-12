# Shared Backend Design Framework

Both the LMS and Peszek-Poyato parts of LMSSPP are best understood as
dynamics plus gauge transformations. The dynamics describe how a state evolves
once a gauge has been chosen. The gauge transformations reveal conserved objects
that make the motion lower-dimensional, reproducible, or easier to compare
across trajectories.

This page describes the intended common backend architecture as well as the
current mathematical framework. Some pieces exist today in the backend and
widgets; other pieces describe the refactor direction for future LMS experiment
and gauge APIs.

## Disambiguation: Dynamics vs Gauge

> [!WARNING]
> LMS evolves arbitrary gauges. The newer LMS construction chooses a canonical
> gauge by an exact Busemann inverse problem. Poisson shrink is a noncanonical
> gauge approximation and legacy gauge. The backend should allow different
> gauges as explicit transformations of the canonical reconstruction, rather
> than hardcoding the canonical gauge as the only possible state.

The distinction matters because the LMS equations do not require the reference
cloud to be canonical. They evolve a chosen reduced coordinate `w`, a frame
variable `zeta`, and a frozen reference cloud. The canonical Busemann
construction is a preferred way to obtain such a reference cloud from an
observed physical cloud.

Peszek-Poyato has an analogous split. The second-order Cucker-Smale-type
alignment system evolves a physical joint measure on `(x, v)`. The
Peszek-Poyato transform changes the second variable to a conserved label
`omega`, yielding a first-order fiberwise system with a fixed marginal
`nu(omega)`.

## Topology and State Spaces

The two systems share a gauge-theoretic pattern but live on different spaces.
For LMS, the code often uses ambient dimension `d` and points on `S^{d-1}`;
equivalently, one may write the sphere as `S^d` embedded in `R^{d+1}`.

| System | Physical space | Gauge/conserved label space | Canonical object |
|---|---|---|---|
| LMS sphere Kuramoto | `S^d` or finite clouds on the sphere | boundary reference cloud `nu(xi)` on `S^d` plus `w in B^{d+1}` | Busemann-centered reference cloud |
| Peszek-Poyato | `R^d_x x R^d_v` or measures on phase space | conserved `nu(omega)` on `R^d_omega` plus disintegration `{mu^omega}` | PP transformed joint measure `mu(x,omega)` |

> [!IMPORTANT]
> LMS can be studied in a corotating frame, and Peszek-Poyato can be studied
> in a comoving frame. Optional rotation/precession variables for LMS and
> barycenter variables for PP are important for reconstructing the full
> physical state, but they are not the central reduced objects in the gauge
> viewpoint.

## Unreduced LMS Dynamics

The finite-particle LMS sphere Kuramoto dynamics has particles
`x_i in S^{d-1}` and a common skew-symmetric rotation generator `A`:

$$
\dot x_i = A x_i + Z - \langle Z, x_i\rangle x_i,
\qquad
Z = \sum_i a_i x_i.
$$

The corresponding measure-valued form on the sphere is a continuity equation

$$
\partial_t \rho + \operatorname{div}_{S}
\left(\left(Ax + Z[\rho] - \langle Z[\rho],x\rangle x\right)\rho\right)=0,
$$

where

$$
Z[\rho]=\int_{S^{d-1}} x\,d\rho(x).
$$

LMS generalizes the Watanabe-Strogatz / Mobius transformation picture from
circle-like oscillator systems to Kuramoto dynamics on higher-dimensional
spheres. After choosing a reference cloud `p`, the physical state is
reconstructed by

$$
x_i(t)=\zeta(t)M_{w(t)}(p_i).
$$

## Canonical LMS Gauge

The canonical LMS construction starts from an observed physical cloud or
measure and chooses a distinguished reference representative.

1. Phase potential: build the Busemann phase from the physical cloud or
   physical measure.
2. Inverse problem: find the critical center `w_* in B^{d+1}`.
3. Constants of motion: deboost the physical cloud by
   `xi = M_{-w_*}(x)`, giving the conserved reference distribution `nu(xi)`.

In this gauge, the physical cloud evolution is reduced to:

- a dynamic coordinate `w_t`
- a frozen reference distribution `nu(xi)`
- an optional frame variable `zeta_t in SO(d)` for full reconstruction

The canonical reference cloud is tied to a shared backward-time limiting shape,
modulo rotation and frame conventions. This is the sense in which the
Busemann-centered cloud plays the role of Watanabe-Strogatz constants of
motion for the finite-N LMS system.

## Unreduced Peszek-Poyato Dynamics

Peszek-Poyato begins from a second-order Cucker-Smale-type alignment model on
`R^d_x x R^d_v`:

$$
\dot x_i = v_i,
\qquad
\dot v_i =
\frac1N \sum_j D^2W(x_i-x_j)(v_j-v_i).
$$

The kinetic measure form is

$$
\partial_t f + v\cdot\nabla_x f + \operatorname{div}_v(F[f]f)=0,
$$

with

$$
F[f](x,v)=\int D^2W(x-x')(v'-v)\,df(x',v').
$$

The first-order PP / Kuramoto-type formulation uses a conserved label
`omega` and fiberwise measures:

$$
\partial_t \mu^\omega
+ \operatorname{div}_x(u[\rho](x,\omega)\mu^\omega)=0,
$$

where

$$
u[\rho](x,\omega)=\omega-\nabla W*\rho(x).
$$

## Canonical PP Gauge

The PP gauge is field-based rather than center-based:

1. Phase potential: compute the convolution potential
   `Psi_rho = W * rho`.
2. Gauge field: compute `A_rho = grad Psi_rho`.
3. Constants of motion: transform
   `omega = v + grad W*rho(x)` and push forward
   `mu = T^{2->1}[rho]# f`.

The `omega`-marginal `nu(omega)` is conserved. The joint measure can be
disintegrated in the sense

$$
\mu(x,\omega) = \mu^\omega(x)\otimes\nu(\omega),
$$

or, more precisely, integrals against `mu` are computed by integrating the
fiber measures `mu^omega` against the conserved marginal `nu`.

Unlike finite LMS point constants, PP fibers may carry non-atomic probability
measures. This richer fiber structure is the reason the PP reduction preserves
a conserved marginal and a family of conditional distributions rather than
collapsing to a single vector-valued reduced coordinate.

## Parallel Pattern

| Step | LMS | Peszek-Poyato |
|---|---|---|
| Phase potential | Busemann potential on the ball | convolution potential `W*rho` |
| Gauge field/center | critical Busemann center `w_*` | field `grad W*rho` and transformed label `omega` |
| Conserved object | reference cloud/distribution `nu(xi)` | marginal `nu(omega)` |
| Reconstruction | `x = zeta M_w(xi)` | `v = omega - grad W*rho(x)` |
| Limit interpretation | backward-time shape preservation | forward-time fiberwise equilibrium/duality |

The common backend design should therefore expose:

- dynamics modules that evolve states in a chosen gauge
- gauge state modules that store reduced LMS data and enforce invariants when
  the state is canonical
- initialization modules that build raw template clouds before any gauge is
  chosen
- canonical modules that construct exact Busemann gauges
- gauge transformation modules that move between canonical and noncanonical
  representations
- widgets and YAML experiments that call backend APIs instead of duplicating
  gauge mathematics

## Backend Consequence

The implementation boundary should follow the mathematics:

- `core/lms.py` evolves arbitrary LMS reduced states.
- `core/gauge.py` holds neutral cloud/state vocabulary. Its base `GaugeState`
  is intentionally permissive, so it can represent arbitrary, legacy, or
  diagnostic gauges without enforcing a Mobius reconstruction relation.
- `core/canonical_gauge.py` constructs the exact Busemann gauge. Its
  `CanonicalGaugeState` enforces `x = M_w(xi)` by deriving observed physical
  points from the reference cloud and current reduced coordinate.
- `core/initialize.py` holds gauge-agnostic initialization procedures, including
  optical preset/template sampling and legacy Poisson shrink initialization.
- `core/gauge_transformations.py` is a compatibility and transformation layer
  that chooses between canonical Busemann construction and noncanonical
  initialization paths while widgets migrate to explicit state objects.
- PP backend modules should keep the same separation between physical dynamics,
  gauge transforms, and experiment orchestration.

This keeps canonical gauges central without making them the only states the
backend can evolve.

> [!IMPORTANT]
> Changing `w` in a `CanonicalGaugeState` changes the represented physical
> cloud, because the canonical state always maintains
> `observed_points = M_w(reference_points)`. This is not a loss of the original
> initialized cloud: the exact inversion coordinate is stored as provenance, so
> resetting `w` to that value reconstructs the original initialized physical
> cloud up to numerical tolerance.
