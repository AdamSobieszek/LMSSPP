"""LMS dynamics utilities for Kuramoto-on-sphere models.

This module implements the real-valued Lipton-Morelo-Strogatz (LMS) reduction
for Kuramoto dynamics on S^{d-1}, with PyTorch autograd support for
hyperbolic-metric gradients on the Poincare ball B^d.

Primary references:
- Eq. (13), (14), (15) in arXiv:1907.07150.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch
from torch import Tensor


DEFAULT_EPS = 1e-9


def dot(a: Tensor, b: Tensor) -> Tensor:
    """Dot product along the last axis."""
    return (a * b).sum(dim=-1)


def normalize(x: Tensor, eps: float = DEFAULT_EPS) -> Tensor:
    """Normalize vectors along the last axis."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _expand_like(w: Tensor, x: Tensor) -> Tensor:
    while w.dim() < x.dim():
        w = w.unsqueeze(0)
    return w


def clamp_to_ball(w: Tensor, radius: float = 0.999, eps: float = DEFAULT_EPS) -> Tensor:
    """Project vectors to the open Poincare ball with max norm `radius`."""
    nrm = w.norm(dim=-1, keepdim=True)
    scaled = w / (nrm + eps) * radius
    return torch.where(nrm >= radius, scaled, w)


def mobius_ball(x: Tensor, w: Tensor, eps: float = DEFAULT_EPS) -> Tensor:
    r"""Real Mobius map M_w on B^d.

    M_w(x) = ((1 - |w|^2)(x - |x|^2 w)) / (1 - 2<w,x> + |w|^2|x|^2) - w
    """
    w = _expand_like(w, x)
    x2 = dot(x, x).unsqueeze(-1)
    w2 = dot(w, w).unsqueeze(-1)
    wx = dot(w, x).unsqueeze(-1)
    num = (1.0 - w2) * (x - x2 * w)
    den = (1.0 - 2.0 * wx + w2 * x2).clamp(min=eps)
    return num / den - w


def mobius_sphere(x: Tensor, w: Tensor, eps: float = DEFAULT_EPS) -> Tensor:
    r"""Mobius map M_w restricted to S^{d-1} (|x|=1)."""
    w = _expand_like(w, x)
    w2 = dot(w, w).unsqueeze(-1)
    diff = x - w
    diff2 = dot(diff, diff).unsqueeze(-1).clamp(min=eps)
    return (1.0 - w2) * diff / diff2 - w


def random_points_on_sphere(
    n: int,
    d: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample N i.i.d. random unit vectors in R^d."""
    x = torch.randn((n, d), device=device, dtype=dtype, generator=generator)
    return normalize(x)


def phases_to_s1(theta: Tensor) -> Tensor:
    """Map phases theta -> (cos(theta), sin(theta))."""
    return torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)


def order_parameter(w: Tensor, base_points: Tensor, weights: Tensor) -> Tensor:
    r"""Z(w) = sum_i a_i M_w(p_i), with p_i on S^{d-1}."""
    x = mobius_sphere(base_points, w)
    return (weights[:, None] * x).sum(dim=0)


def lms_vector_field(w: Tensor, base_points: Tensor, weights: Tensor) -> Tensor:
    r"""Eq. (14)-style reduced field for linear order parameter.

    dw/dt = -0.5 * (1 - |w|^2) * Z(w),  Z(w)=sum_i a_i M_w(p_i)
    """
    z = order_parameter(w, base_points, weights)
    return -0.5 * (1.0 - dot(w, w)) * z


def hyperbolic_conformal_factor(w: Tensor, eps: float = DEFAULT_EPS) -> Tensor:
    r"""phi(w)=2/(1-|w|^2), where g_hyp = phi(w)^2 I."""
    return 2.0 / (1.0 - dot(w, w)).clamp(min=eps)


def hyperbolic_potential_lms(
    w: Tensor,
    base_points: Tensor,
    weights: Tensor,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    r"""LMS potential Phi(w) from Eq. (15).

    Phi(w) = sum_i a_i log((1-|w|^2)/|w-p_i|^2)
    """
    if w.dim() == 1:
        w_eval = w.unsqueeze(0)
        squeeze = True
    else:
        w_eval = w
        squeeze = False

    w2 = dot(w_eval, w_eval).unsqueeze(-1)
    numer = (1.0 - w2).clamp(min=eps)
    diff = w_eval[:, None, :] - base_points[None, :, :]
    denom = dot(diff, diff).clamp(min=eps)
    log_terms = torch.log(numer) - torch.log(denom)
    phi = (log_terms * weights[None, :]).sum(dim=-1)
    return phi.squeeze(0) if squeeze else phi


def hyperbolic_grad_from_euclidean(w: Tensor, grad_euclidean: Tensor) -> Tensor:
    r"""Convert Euclidean gradient to hyperbolic gradient in B^d."""
    factor = ((1.0 - dot(w, w)).clamp(min=DEFAULT_EPS) ** 2) / 4.0
    return factor * grad_euclidean


def hyperbolic_grad_autograd(
    w: Tensor,
    base_points: Tensor,
    weights: Tensor,
    *,
    create_graph: bool = False,
) -> Tensor:
    """Compute grad_hyp Phi(w) via torch.autograd.grad."""
    if not w.requires_grad:
        raise ValueError("`w` must have requires_grad=True.")

    phi = hyperbolic_potential_lms(w, base_points, weights)
    (grad_euc,) = torch.autograd.grad(
        phi,
        w,
        create_graph=create_graph,
        retain_graph=create_graph,
    )
    return hyperbolic_grad_from_euclidean(w, grad_euc)


def lms_vector_field_autograd(
    w: Tensor,
    base_points: Tensor,
    weights: Tensor,
    *,
    detach: bool = True,
    create_graph: bool = False,
) -> Tensor:
    r"""Compute dw/dt=-grad_hyp Phi(w) via autograd."""
    # Avoid an unnecessary clone in the runtime integrator path.
    w_req = w if w.requires_grad else w.detach().requires_grad_(True)
    grad_hyp = hyperbolic_grad_autograd(
        w_req,
        base_points,
        weights,
        create_graph=create_graph,
    )
    vec = -grad_hyp
    return vec.detach() if detach else vec


def reduced_general_vector_field(w: Tensor, omega: Tensor, x: Tensor) -> Tensor:
    r"""General reduced LMS field from Eq. (6) in the real setting.

    dw/dt = Omega w + 0.5(1+|w|^2)X - <w,X>w
    """
    w2 = dot(w, w)
    return omega @ w + 0.5 * (1.0 + w2) * x - dot(w, x) * w


def kuramoto_sphere_vector_field(x: Tensor, a_matrix: Tensor, weights: Tensor) -> Tensor:
    r"""Full Kuramoto/Lohe field on S^{d-1}.

    dx_i/dt = A x_i + Z - <Z, x_i>x_i, with Z = sum_j a_j x_j
    """
    z = (weights[:, None] * x).sum(dim=0)
    ax = x @ a_matrix.T
    proj = dot(x, z).unsqueeze(-1) * x
    return ax + z.unsqueeze(0) - proj


def kuramoto_sphere_vector_field_pairwise(x: Tensor, a_matrix: Tensor, weights: Tensor) -> Tensor:
    r"""Pairwise all-to-all Kuramoto/Lohe field on S^{d-1}.

    Equivalent to `kuramoto_sphere_vector_field` for linear mean field:
      dx_i/dt = A x_i + sum_j a_j (x_j - <x_j, x_i>x_i)
    """
    if x.dim() != 2:
        raise ValueError("x must have shape [N,d].")
    if weights.dim() != 1 or weights.shape[0] != x.shape[0]:
        raise ValueError("weights must have shape [N], matching x.")
    if a_matrix.shape != (x.shape[1], x.shape[1]):
        raise ValueError("a_matrix must have shape [d,d].")

    ax = x @ a_matrix.T
    # i,j convention:
    # xi: [N,1,d], xj: [1,N,d], dot_ji[i,j] = <x_j, x_i>.
    xi = x.unsqueeze(1)
    xj = x.unsqueeze(0)
    dot_ji = (xj * xi).sum(dim=-1, keepdim=True)
    pair_terms = xj - dot_ji * xi
    coupling = (weights.view(1, -1, 1) * pair_terms).sum(dim=1)
    return ax + coupling


def pushforward_points(w: Tensor, base_points: Tensor, *, renormalize: bool = True) -> Tensor:
    """x_i(w)=M_w(p_i) for p_i on the sphere."""
    x = mobius_sphere(base_points, w)
    return normalize(x) if renormalize else x


@dataclass
class SimulationResult:
    trajectory: Tensor
    dt: float
    steps: int
    field_name: str


@dataclass
class LMSReducedTrajectory:
    """Trajectory container for reduced LMS dynamics in any ambient dimension d."""

    w: Tensor  # [T,d]
    zeta: Tensor  # [T,d,d]
    z: Tensor  # [T,d], z = -zeta w (lab frame)
    Z: Tensor  # [T,d], lab order parameter
    Z_body: Tensor  # [T,d], body-frame order parameter = zeta^{-1} Z
    x_body: Tensor | None  # [T,N,d], optional body-frame boundary points M_w(p_i)
    x_lab: Tensor | None  # [T,N,d], optional lab-frame boundary points zeta M_w(p_i)
    base_points: Tensor  # [N,d], stored for optional point reconstruction
    dt: float
    steps: int
    coupling: float
    w_mode: str


def alpha_operator(y1: Tensor, y2: Tensor) -> Tensor:
    r"""Compute alpha(y1,y2) = y2 y1^T - y1 y2^T (skew-symmetric)."""
    return torch.outer(y2, y1) - torch.outer(y1, y2)


def project_to_so(matrix: Tensor) -> Tensor:
    """Project a matrix (or batch of matrices) to SO(d) via QR + determinant fix."""
    q, r = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs.unsqueeze(-2)  # fixes column signs

    det = torch.linalg.det(q)
    if q.dim() == 2:
        if det < 0:
            q = q.clone()
            q[:, -1] *= -1.0
        return q

    q = q.clone()
    neg = det < 0
    if neg.any():
        q[neg, :, -1] *= -1.0
    return q


def skew_symmetric_from_axis(axis: Tensor, rate: float = 1.0) -> Tensor:
    """Build 3D skew-symmetric matrix for angular velocity around `axis`."""
    if axis.numel() != 3:
        raise ValueError("axis must be a 3-vector.")
    a = normalize(axis.view(3), eps=DEFAULT_EPS)
    x, y, z = float(a[0]), float(a[1]), float(a[2])
    return torch.tensor(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=axis.dtype,
        device=axis.device,
    ) * float(rate)


def lms_reduced_observables(
    w: Tensor,
    zeta: Tensor,
    base_points: Tensor,
    weights: Tensor,
    *,
    coupling: float = 1.0,
    renormalize_points: bool = True,
    return_x_body: bool = True,
    return_x_lab: bool = True,
) -> tuple[Tensor | None, Tensor | None, Tensor, Tensor, Tensor]:
    r"""Compute reduced LMS observables for state (w,zeta).

    Returns:
    - x_body: [N,d] = M_w(p_i) (optional)
    - x_lab: [N,d] = zeta M_w(p_i) (optional)
    - Z: [d], lab-frame order parameter
    - Z_body: [d], body-frame order parameter = zeta^{-1}Z
    - z: [d], lab-frame center with z = -zeta w
    """
    x_body_all = mobius_sphere(base_points, w)
    if renormalize_points:
        x_body_all = normalize(x_body_all)

    # Compute order parameters in body frame first; this avoids constructing
    # x_lab unless explicitly requested.
    Z_body = float(coupling) * (weights[:, None] * x_body_all).sum(dim=0)
    Z = Z_body @ zeta.T
    z = -(w @ zeta.T)
    x_body = x_body_all if return_x_body else None
    x_lab = (x_body_all @ zeta.T) if return_x_lab else None
    return x_body, x_lab, Z, Z_body, z


def lms_reduced_rhs(
    w: Tensor,
    zeta: Tensor,
    base_points: Tensor,
    weights: Tensor,
    *,
    A: Tensor | None = None,
    coupling: float = 1.0,
    w_mode: Literal["explicit", "autograd"] = "explicit",
    return_x_body: bool = True,
    return_x_lab: bool = True,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor, Tensor, Tensor]:
    r"""Reduced LMS RHS for general ambient dimension.

    Eq. (7a), (7b)-style update in row-vector convention:
    - wdot = -0.5(1-|w|^2) zeta^{-1} Z
    - zetadot = (A - alpha(zeta w, Z)) zeta
    """
    if A is None:
        A = torch.zeros(
            (w.shape[-1], w.shape[-1]),
            dtype=w.dtype,
            device=w.device,
        )

    x_body, x_lab, Z, Z_body, z = lms_reduced_observables(
        w,
        zeta,
        base_points,
        weights,
        coupling=coupling,
        return_x_body=return_x_body,
        return_x_lab=return_x_lab,
    )

    if w_mode == "autograd":
        # For linear order parameters, dw/dt is hyperbolic gradient flow in w.
        wdot = lms_vector_field_autograd(w, base_points, float(coupling) * weights)
    elif w_mode == "explicit":
        wdot = -0.5 * (1.0 - dot(w, w)) * Z_body
    else:
        raise ValueError("w_mode must be 'explicit' or 'autograd'.")

    zeta_w = w @ zeta.T
    alpha = alpha_operator(zeta_w, Z)
    zetadot = (A - alpha) @ zeta
    return wdot, zetadot, x_body, x_lab, Z, Z_body, z


def reconstruct_points_at_frame(
    w_t: Tensor,
    zeta_t: Tensor,
    base_points: Tensor,
    *,
    frame: Literal["lab", "body"] = "lab",
    renormalize: bool = True,
) -> Tensor:
    """Reconstruct boundary points at one frame from reduced state."""
    x_body = mobius_sphere(base_points, w_t)
    if renormalize:
        x_body = normalize(x_body)
    if frame == "body":
        return x_body
    if frame == "lab":
        return x_body @ zeta_t.T
    raise ValueError("frame must be 'lab' or 'body'.")


def integrate_lms_reduced_euler(
    w0: Tensor,
    zeta0: Tensor,
    base_points: Tensor,
    weights: Tensor,
    *,
    A: Tensor | None = None,
    coupling: float = 1.0,
    dt: float = 1e-3,
    steps: int = 1000,
    w_mode: Literal["explicit", "autograd"] = "explicit",
    clamp_radius: float = 0.999,
    project_rotation: bool = True,
    store_points: Literal["none", "body", "lab", "both"] = "none",
    store_dtype: torch.dtype = torch.float32,
    preallocate: bool = True,
    cancel_check: Callable[[], bool] | None = None,
) -> LMSReducedTrajectory:
    """Integrate reduced LMS dynamics in ambient dimension d."""
    if base_points.dim() != 2:
        raise ValueError("base_points must have shape [N,d].")
    if w0.dim() != 1:
        raise ValueError("w0 must have shape [d].")
    if zeta0.dim() != 2:
        raise ValueError("zeta0 must have shape [d,d].")
    if weights.dim() != 1 or weights.shape[0] != base_points.shape[0]:
        raise ValueError("weights must have shape [N], matching base_points.")

    n, d = base_points.shape
    if w0.shape[0] != d or zeta0.shape != (d, d):
        raise ValueError("Incompatible shapes among w0, zeta0, and base_points.")
    if A is not None and A.shape != (d, d):
        raise ValueError("A must have shape [d,d].")
    if store_points not in {"none", "body", "lab", "both"}:
        raise ValueError("store_points must be one of: 'none', 'body', 'lab', 'both'.")

    w = w0.clone()
    zeta = zeta0.clone()

    t_count = steps + 1
    device = w0.device

    store_body = store_points in {"body", "both"}
    store_lab = store_points in {"lab", "both"}

    if preallocate:
        w_hist = torch.empty((t_count, d), dtype=store_dtype, device=device)
        zeta_hist = torch.empty((t_count, d, d), dtype=store_dtype, device=device)
        z_hist = torch.empty((t_count, d), dtype=store_dtype, device=device)
        Z_hist = torch.empty((t_count, d), dtype=store_dtype, device=device)
        Z_body_hist = torch.empty((t_count, d), dtype=store_dtype, device=device)
        x_body_hist = (
            torch.empty((t_count, n, d), dtype=store_dtype, device=device) if store_body else None
        )
        x_lab_hist = (
            torch.empty((t_count, n, d), dtype=store_dtype, device=device) if store_lab else None
        )

        for t in range(t_count):
            if cancel_check is not None and bool(cancel_check()):
                raise InterruptedError("Reduced LMS integration cancelled.")
            wdot, zetadot, x_body, x_lab, Z, Z_body, z = lms_reduced_rhs(
                w,
                zeta,
                base_points,
                weights,
                A=A,
                coupling=coupling,
                w_mode=w_mode,
                return_x_body=store_body,
                return_x_lab=store_lab,
            )

            w_hist[t].copy_(w.detach())
            zeta_hist[t].copy_(zeta.detach())
            z_hist[t].copy_(z.detach())
            Z_hist[t].copy_(Z.detach())
            Z_body_hist[t].copy_(Z_body.detach())
            if store_body and x_body_hist is not None:
                if x_body is None:
                    raise RuntimeError("x_body is None while store_points requests body points.")
                x_body_hist[t].copy_(x_body.detach())
            if store_lab and x_lab_hist is not None:
                if x_lab is None:
                    raise RuntimeError("x_lab is None while store_points requests lab points.")
                x_lab_hist[t].copy_(x_lab.detach())

            w = clamp_to_ball(w + dt * wdot, radius=clamp_radius)
            zeta = zeta + dt * zetadot
            if project_rotation:
                zeta = project_to_so(zeta)
    else:
        w_hist_l = []
        zeta_hist_l = []
        z_hist_l = []
        Z_hist_l = []
        Z_body_hist_l = []
        x_body_hist_l = [] if store_body else None
        x_lab_hist_l = [] if store_lab else None

        for _ in range(t_count):
            if cancel_check is not None and bool(cancel_check()):
                raise InterruptedError("Reduced LMS integration cancelled.")
            wdot, zetadot, x_body, x_lab, Z, Z_body, z = lms_reduced_rhs(
                w,
                zeta,
                base_points,
                weights,
                A=A,
                coupling=coupling,
                w_mode=w_mode,
                return_x_body=store_body,
                return_x_lab=store_lab,
            )

            w_hist_l.append(w.detach().to(dtype=store_dtype))
            zeta_hist_l.append(zeta.detach().to(dtype=store_dtype))
            z_hist_l.append(z.detach().to(dtype=store_dtype))
            Z_hist_l.append(Z.detach().to(dtype=store_dtype))
            Z_body_hist_l.append(Z_body.detach().to(dtype=store_dtype))
            if store_body and x_body_hist_l is not None:
                x_body_hist_l.append(x_body.detach().to(dtype=store_dtype))
            if store_lab and x_lab_hist_l is not None:
                x_lab_hist_l.append(x_lab.detach().to(dtype=store_dtype))

            w = clamp_to_ball(w + dt * wdot, radius=clamp_radius)
            zeta = zeta + dt * zetadot
            if project_rotation:
                zeta = project_to_so(zeta)

        w_hist = torch.stack(w_hist_l, dim=0)
        zeta_hist = torch.stack(zeta_hist_l, dim=0)
        z_hist = torch.stack(z_hist_l, dim=0)
        Z_hist = torch.stack(Z_hist_l, dim=0)
        Z_body_hist = torch.stack(Z_body_hist_l, dim=0)
        x_body_hist = torch.stack(x_body_hist_l, dim=0) if x_body_hist_l is not None else None
        x_lab_hist = torch.stack(x_lab_hist_l, dim=0) if x_lab_hist_l is not None else None

    return LMSReducedTrajectory(
        w=w_hist,
        zeta=zeta_hist,
        z=z_hist,
        Z=Z_hist,
        Z_body=Z_body_hist,
        x_body=x_body_hist,
        x_lab=x_lab_hist,
        base_points=base_points.detach().to(dtype=store_dtype).clone(),
        dt=dt,
        steps=steps,
        coupling=float(coupling),
        w_mode=w_mode,
    )


def _select_w_field(
    field: Literal["mobius", "autograd"] | Callable[[Tensor, Tensor, Tensor], Tensor]
) -> tuple[Callable[[Tensor, Tensor, Tensor], Tensor], str]:
    if callable(field):
        return field, getattr(field, "__name__", "custom")
    if field == "mobius":
        return lms_vector_field, "mobius"
    if field == "autograd":
        return lms_vector_field_autograd, "autograd"
    raise ValueError("field must be 'mobius', 'autograd', or a callable.")


def integrate_w_euler(
    w0: Tensor,
    base_points: Tensor,
    weights: Tensor,
    *,
    dt: float = 1e-2,
    steps: int = 200,
    field: Literal["mobius", "autograd"] | Callable[[Tensor, Tensor, Tensor], Tensor] = "mobius",
    clamp_radius: float = 0.999,
) -> SimulationResult:
    """Euler integration for reduced LMS w-dynamics."""
    field_fn, field_name = _select_w_field(field)
    w = w0.clone()
    traj = [w.detach().clone()]
    for _ in range(steps):
        dw = field_fn(w, base_points, weights)
        w = w + dt * dw
        w = clamp_to_ball(w, radius=clamp_radius)
        traj.append(w.detach().clone())
    return SimulationResult(torch.stack(traj, dim=0), dt, steps, field_name)


def integrate_w_rk4(
    w0: Tensor,
    base_points: Tensor,
    weights: Tensor,
    *,
    dt: float = 1e-2,
    steps: int = 200,
    field: Literal["mobius", "autograd"] | Callable[[Tensor, Tensor, Tensor], Tensor] = "mobius",
    clamp_radius: float = 0.999,
) -> SimulationResult:
    """RK4 integration for reduced LMS w-dynamics."""
    field_fn, field_name = _select_w_field(field)
    w = w0.clone()
    traj = [w.detach().clone()]
    for _ in range(steps):
        k1 = field_fn(w, base_points, weights)
        k2 = field_fn(w + 0.5 * dt * k1, base_points, weights)
        k3 = field_fn(w + 0.5 * dt * k2, base_points, weights)
        k4 = field_fn(w + dt * k3, base_points, weights)
        w = w + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        w = clamp_to_ball(w, radius=clamp_radius)
        traj.append(w.detach().clone())
    return SimulationResult(torch.stack(traj, dim=0), dt, steps, field_name)


def reconstruct_sphere_trajectory(
    w_trajectory: Tensor,
    base_points: Tensor,
    *,
    renormalize: bool = True,
) -> Tensor:
    """Reconstruct x_i(t)=M_{w(t)}(p_i); returns [T,N,d]."""
    states = []
    for t in range(w_trajectory.shape[0]):
        states.append(pushforward_points(w_trajectory[t], base_points, renormalize=renormalize))
    return torch.stack(states, dim=0)


def integrate_full_kuramoto_euler(
    x0: Tensor,
    a_matrix: Tensor,
    weights: Tensor,
    *,
    dt: float = 1e-3,
    steps: int = 1000,
    rhs_mode: Literal["mean_field", "pairwise"] = "mean_field",
) -> Tensor:
    """Euler simulation for full Kuramoto sphere dynamics; returns [T,N,d]."""
    if rhs_mode not in {"mean_field", "pairwise"}:
        raise ValueError("rhs_mode must be 'mean_field' or 'pairwise'.")
    rhs_fn = (
        kuramoto_sphere_vector_field
        if rhs_mode == "mean_field"
        else kuramoto_sphere_vector_field_pairwise
    )
    x = normalize(x0.clone())
    traj = [x.detach().clone()]
    for _ in range(steps):
        dx = rhs_fn(x, a_matrix, weights)
        x = normalize(x + dt * dx)
        traj.append(x.detach().clone())
    return torch.stack(traj, dim=0)


def compare_reduced_vs_full_kuramoto(
    w0: Tensor,
    zeta0: Tensor,
    base_points: Tensor,
    weights: Tensor,
    *,
    A: Tensor | None = None,
    coupling: float = 1.0,
    dt: float = 1e-3,
    steps: int = 1000,
    w_mode: Literal["explicit", "autograd"] = "explicit",
    clamp_radius: float = 0.999,
    project_rotation: bool = True,
    full_rhs_mode: Literal["mean_field", "pairwise"] = "mean_field",
) -> dict[str, Tensor]:
    r"""Compare reduced LMS evolution to full Kuramoto sphere evolution.

    Both simulations start from the same initial configuration:
      x_i(0) = zeta0 M_{w0}(p_i),
    where `p_i` are `base_points`.

    Returns a dictionary with full/reduced trajectories and error series:
    - `point_err_rms`: sqrt(mean_i ||x_full - x_reduced||^2)
    - `point_err_max`: max_i ||x_full - x_reduced||
    - `Z_err_norm`: ||Z_full_empirical - Z_reduced||
    - `rhs_pairwise_gap`: max abs diff between pairwise and mean-field full RHS
      evaluated on each full trajectory frame.
    """
    if base_points.dim() != 2:
        raise ValueError("base_points must have shape [N,d].")
    if w0.dim() != 1:
        raise ValueError("w0 must have shape [d].")
    if zeta0.dim() != 2:
        raise ValueError("zeta0 must have shape [d,d].")
    if weights.dim() != 1 or weights.shape[0] != base_points.shape[0]:
        raise ValueError("weights must have shape [N], matching base_points.")

    n, d = base_points.shape
    if w0.shape[0] != d or zeta0.shape != (d, d):
        raise ValueError("Incompatible shapes among w0, zeta0, and base_points.")

    if A is None:
        A = torch.zeros((d, d), dtype=w0.dtype, device=w0.device)
    if A.shape != (d, d):
        raise ValueError("A must have shape [d,d].")

    c = float(coupling)
    reduced = integrate_lms_reduced_euler(
        w0=w0,
        zeta0=zeta0,
        base_points=base_points,
        weights=weights,
        A=A,
        coupling=c,
        dt=float(dt),
        steps=int(steps),
        w_mode=w_mode,
        clamp_radius=float(clamp_radius),
        project_rotation=bool(project_rotation),
        store_points="lab",
        store_dtype=w0.dtype,
        preallocate=True,
    )
    if reduced.x_lab is None:
        raise RuntimeError("reduced trajectory is missing x_lab despite store_points='lab'.")

    # Build full-system initial points from the same reduced state.
    x0_body = mobius_sphere(base_points, w0)
    x0_lab = normalize(x0_body @ zeta0.T)
    weights_full = c * weights
    full = integrate_full_kuramoto_euler(
        x0=x0_lab,
        a_matrix=A,
        weights=weights_full,
        dt=float(dt),
        steps=int(steps),
        rhs_mode=full_rhs_mode,
    )

    # Empirical full-system observables.
    Z_full = (weights_full.unsqueeze(0).unsqueeze(-1) * full).sum(dim=1)
    weight_sum = float(weights.sum())
    denom = max(abs(weight_sum), DEFAULT_EPS)
    mu_full = (weights.unsqueeze(0).unsqueeze(-1) * full).sum(dim=1) / denom

    # Reduced-system observables.
    x_red = reduced.x_lab
    Z_red = reduced.Z

    point_diff = full - x_red
    point_err = torch.linalg.norm(point_diff, dim=-1)  # [T,N]
    point_err_rms = torch.sqrt(torch.mean(point_err * point_err, dim=1))
    point_err_max = torch.max(point_err, dim=1).values
    Z_err_norm = torch.linalg.norm(Z_full - Z_red, dim=-1)

    # Verify equivalence of pairwise-vs-mean-field full RHS on the same states.
    rhs_pairwise_gap = torch.empty((full.shape[0],), dtype=full.dtype, device=full.device)
    for t in range(full.shape[0]):
        vf_mean = kuramoto_sphere_vector_field(full[t], A, weights_full)
        vf_pair = kuramoto_sphere_vector_field_pairwise(full[t], A, weights_full)
        rhs_pairwise_gap[t] = torch.max(torch.abs(vf_mean - vf_pair))

    return {
        "x_full": full,
        "x_reduced_lab": x_red,
        "w_reduced": reduced.w,
        "z_reduced": reduced.z,
        "Z_reduced": Z_red,
        "Z_full_empirical": Z_full,
        "mu_full_empirical": mu_full,
        "point_err_rms": point_err_rms,
        "point_err_max": point_err_max,
        "Z_err_norm": Z_err_norm,
        "rhs_pairwise_gap": rhs_pairwise_gap,
    }


def fiber_full_cross_entropy_vector_field(
    ws: Tensor,
    fiber_points: Tensor,
    *,
    within_weight: float | None = None,
    cross_weight: float = -1.0,
) -> Tensor:
    r"""Multi-fiber variant using autograd hyperbolic gradients.

    For each fiber k:
    dw_k/dt = -grad_hyp Phi_k(w_k)
    with
      Phi_k(w) = sum_j alpha_j log((1-|w|^2)/|w-q_j|^2),
    where q_j includes:
    - within-fiber points u_k^i with weight `within_weight` (default d-1),
    - anti-points -u_l^j for l!=k with weight `cross_weight`.
    """
    k_total, n, d = fiber_points.shape
    if ws.shape != (k_total, d):
        raise ValueError("ws must have shape [K,d] matching fiber_points [K,N,d].")

    if within_weight is None:
        within_weight = float(d - 1)

    out = []
    for k in range(k_total):
        w_k = ws[k].detach().clone().requires_grad_(True)

        self_pts = fiber_points[k]
        self_w = torch.full((n,), within_weight, dtype=ws.dtype, device=ws.device)

        if k_total > 1:
            others = torch.cat([fiber_points[j] for j in range(k_total) if j != k], dim=0)
            ext_pts = -others
            ext_w = torch.full((ext_pts.shape[0],), cross_weight, dtype=ws.dtype, device=ws.device)
            points = torch.cat([self_pts, ext_pts], dim=0)
            weights = torch.cat([self_w, ext_w], dim=0)
        else:
            points = self_pts
            weights = self_w

        grad_hyp = hyperbolic_grad_autograd(w_k, points, weights, create_graph=False)
        out.append((-grad_hyp).detach())

    return torch.stack(out, dim=0)


def integrate_fiber_full_cross_entropy_euler(
    w0_all: Tensor,
    fiber_points: Tensor,
    *,
    dt: float = 5e-3,
    steps: int = 300,
    within_weight: float | None = None,
    cross_weight: float = -1.0,
    clamp_radius: float = 0.999,
) -> Tensor:
    """Euler integration for K coupled fibers; returns [T,K,d]."""
    ws = w0_all.clone()
    traj = [ws.detach().clone()]
    for _ in range(steps):
        dw = fiber_full_cross_entropy_vector_field(
            ws,
            fiber_points,
            within_weight=within_weight,
            cross_weight=cross_weight,
        )
        ws = clamp_to_ball(ws + dt * dw, radius=clamp_radius)
        traj.append(ws.detach().clone())
    return torch.stack(traj, dim=0)


def _sphere_wireframe_traces(n_lat: int = 9, n_lon: int = 18):
    import numpy as np
    import plotly.graph_objects as go

    traces = []
    lat_vals = np.linspace(-0.8 * np.pi / 2, 0.8 * np.pi / 2, n_lat)
    lon = np.linspace(0, 2 * np.pi, 200)

    for phi in lat_vals:
        x = np.cos(phi) * np.cos(lon)
        y = np.cos(phi) * np.sin(lon)
        z = np.sin(phi) * np.ones_like(lon)
        traces.append(
            go.Scatter3d(
                x=x.tolist(),
                y=y.tolist(),
                z=z.tolist(),
                mode="lines",
                line=dict(color="lightgrey", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    lon_vals = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    lat = np.linspace(-np.pi / 2, np.pi / 2, 200)
    for lam in lon_vals:
        x = np.cos(lat) * np.cos(lam)
        y = np.cos(lat) * np.sin(lam)
        z = np.sin(lat)
        traces.append(
            go.Scatter3d(
                x=x.tolist(),
                y=y.tolist(),
                z=z.tolist(),
                mode="lines",
                line=dict(color="lightgrey", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return traces


def make_disk_figure(
    traj_w: Tensor,
    base_points: Tensor,
    *,
    t_index: int = -1,
    title: str = "LMS Dynamics on S^1",
    point_size: int = 8,
):
    """Return a Plotly disk snapshot for x_i=M_{w(t)}(p_i) and w-path."""
    import numpy as np
    import plotly.graph_objects as go

    t = int(t_index)
    if t < 0:
        t = traj_w.shape[0] + t
    t = max(0, min(t, traj_w.shape[0] - 1))

    x_t = pushforward_points(traj_w[t], base_points).detach().cpu()
    path = traj_w[: t + 1].detach().cpu()

    theta = np.linspace(0.0, 2.0 * np.pi, 300)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode="lines",
            line=dict(color="black", width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_t[:, 0].tolist(),
            y=x_t[:, 1].tolist(),
            mode="markers",
            marker=dict(size=point_size, color="royalblue"),
            name="x_i(t)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=path[:, 0].tolist(),
            y=path[:, 1].tolist(),
            mode="lines",
            line=dict(color="firebrick", width=2, dash="dot"),
            name="w path",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[float(path[-1, 0])],
            y=[float(path[-1, 1])],
            mode="markers",
            marker=dict(size=point_size + 4, color="firebrick", symbol="x"),
            name="w(t)",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=620,
        height=620,
        xaxis=dict(range=[-1.1, 1.1], visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-1.1, 1.1], visible=False),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def make_sphere_figure(
    x_t: Tensor,
    *,
    w_t: Tensor | None = None,
    w_path: Tensor | None = None,
    title: str = "LMS Dynamics on S^2",
    point_size: int = 5,
):
    """Return a Plotly 3D sphere snapshot with optional w marker/path."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for tr in _sphere_wireframe_traces():
        fig.add_trace(tr)

    x_cpu = x_t.detach().cpu()
    fig.add_trace(
        go.Scatter3d(
            x=x_cpu[:, 0].tolist(),
            y=x_cpu[:, 1].tolist(),
            z=x_cpu[:, 2].tolist(),
            mode="markers",
            marker=dict(size=point_size, color="royalblue"),
            name="x_i(t)",
        )
    )

    if w_path is not None:
        p = w_path.detach().cpu()
        fig.add_trace(
            go.Scatter3d(
                x=p[:, 0].tolist(),
                y=p[:, 1].tolist(),
                z=p[:, 2].tolist(),
                mode="lines",
                line=dict(color="firebrick", width=2, dash="dot"),
                name="w path",
            )
        )

    if w_t is not None:
        w_cpu = w_t.detach().cpu()
        fig.add_trace(
            go.Scatter3d(
                x=[float(w_cpu[0])],
                y=[float(w_cpu[1])],
                z=[float(w_cpu[2])],
                mode="markers",
                marker=dict(size=point_size + 2, color="firebrick", symbol="x"),
                name="w(t)",
            )
        )

    fig.update_layout(
        title=title,
        width=680,
        height=680,
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )
    return fig


def make_kuramoto_widget_2d(traj_w: Tensor, base_points: Tensor, **kwargs):
    """Thin wrapper around the existing notebook widget class."""
    try:
        from ..integrations.kuramoto_widget import KuramotoWidget
    except Exception:
        from integrations.kuramoto_widget import KuramotoWidget  # type: ignore
    return KuramotoWidget(traj_w, base_points, **kwargs)


def make_lms_circle_plotly_widget(**kwargs):
    """Create the new Plotly-first LMS circle widget (lazy import)."""
    try:
        from ..lms_plotly_widget import LMSCirclePlotlyWidget
    except Exception:
        from lms_plotly_widget import LMSCirclePlotlyWidget  # type: ignore
    return LMSCirclePlotlyWidget(**kwargs)


def make_lms_ball3d_widget(**kwargs):
    """Create the reduced LMS B^3/S^2 Plotly widget (lazy import)."""
    try:
        from ..lms_ball3d_widget import LMSBall3DWidget
    except Exception:
        from lms_ball3d_widget import LMSBall3DWidget  # type: ignore
    return LMSBall3DWidget(**kwargs)


def make_lms_ball3d_backward_two_sheet_widget(**kwargs):
    """Create the LMS entropy-shell two-sheet widget with outer-sheet traces."""
    try:
        from ..lms_ball3d_widget import LMSBall3DEntropyShellTwoSheetWidget
    except Exception:
        from lms_ball3d_widget import LMSBall3DEntropyShellTwoSheetWidget  # type: ignore
    return LMSBall3DEntropyShellTwoSheetWidget(**kwargs)


def make_lms_ball3d_hydrodynamic_ensemble_widget(**kwargs):
    """Create the LMS entropy-shell ensemble widget."""
    try:
        from ..lms_ball3d_widget import LMSBall3DEntropyShellEnsembleWidget
    except Exception:
        from lms_ball3d_widget import LMSBall3DEntropyShellEnsembleWidget  # type: ignore
    return LMSBall3DEntropyShellEnsembleWidget(**kwargs)


def make_cucker_smale_ball3d_widget(**kwargs):
    """Create the Cucker-Smale B^3 display widget (lazy import)."""
    try:
        from ..cucker_smale_ball3d_widget import CuckerSmaleBall3DWidget
    except Exception:
        from cucker_smale_ball3d_widget import CuckerSmaleBall3DWidget  # type: ignore
    return CuckerSmaleBall3DWidget(**kwargs)


def make_cucker_smale_ball3d_hydrodynamic_ensemble_widget(**kwargs):
    """Create the Cucker-Smale hydrodynamic-ensemble widget (lazy import)."""
    try:
        from ..cucker_smale_ball3d_widget import CuckerSmaleBall3DHydrodynamicEnsembleWidget
    except Exception:
        from cucker_smale_ball3d_widget import CuckerSmaleBall3DHydrodynamicEnsembleWidget  # type: ignore
    return CuckerSmaleBall3DHydrodynamicEnsembleWidget(**kwargs)


def make_lms_iframe_widget(
    widget: Literal["ball3d", "ball3d_backward_two_sheet", "circle"] = "ball3d",
    **kwargs,
):
    """Factory for iframe-facing widgets."""
    if widget == "ball3d":
        return make_lms_ball3d_widget(**kwargs)
    if widget == "ball3d_backward_two_sheet":
        return make_lms_ball3d_backward_two_sheet_widget(**kwargs)
    if widget == "circle":
        return make_lms_circle_plotly_widget(**kwargs)
    raise ValueError("widget must be one of: 'ball3d', 'ball3d_backward_two_sheet', 'circle'.")


def export_lms_static_payload(
    *,
    widget: Literal["ball3d", "ball3d_backward_two_sheet", "circle"] = "ball3d",
    params: dict[str, Any] | None = None,
    include_points: bool = True,
    point_decimation: int = 1,
    seed: int = 0,
    w_mode: Literal["explicit", "autograd"] = "autograd",
) -> dict[str, Any]:
    """Export JSON-serializable trajectory payload for iframe/static rendering."""
    try:
        from ..export.iframe_export import export_lms_static_payload as _impl
    except Exception:
        from lms_iframe_export import export_lms_static_payload as _impl  # type: ignore
    return _impl(
        widget=widget,
        params=params,
        include_points=include_points,
        point_decimation=point_decimation,
        seed=seed,
        w_mode=w_mode,
    )


def write_lms_static_bundle(
    out_dir: str,
    *,
    widget: Literal["ball3d", "ball3d_backward_two_sheet", "circle"] = "ball3d",
    params: dict[str, Any] | None = None,
    include_points: bool = True,
    point_decimation: int = 1,
    seed: int = 0,
    w_mode: Literal["explicit", "autograd"] = "autograd",
    api_base: str = "/api/recompute",
):
    """Write standalone iframe bundle (index.html + trajectory.json)."""
    try:
        from ..export.iframe_export import write_lms_static_bundle as _impl
    except Exception:
        from lms_iframe_export import write_lms_static_bundle as _impl  # type: ignore
    return _impl(
        out_dir=out_dir,
        widget=widget,
        params=params,
        include_points=include_points,
        point_decimation=point_decimation,
        seed=seed,
        w_mode=w_mode,
        api_base=api_base,
    )


__all__ = [
    "LMSReducedTrajectory",
    "SimulationResult",
    "alpha_operator",
    "clamp_to_ball",
    "dot",
    "fiber_full_cross_entropy_vector_field",
    "hyperbolic_conformal_factor",
    "hyperbolic_grad_autograd",
    "hyperbolic_grad_from_euclidean",
    "hyperbolic_potential_lms",
    "integrate_fiber_full_cross_entropy_euler",
    "integrate_full_kuramoto_euler",
    "integrate_lms_reduced_euler",
    "integrate_w_euler",
    "integrate_w_rk4",
    "compare_reduced_vs_full_kuramoto",
    "kuramoto_sphere_vector_field",
    "kuramoto_sphere_vector_field_pairwise",
    "lms_reduced_observables",
    "lms_reduced_rhs",
    "lms_vector_field",
    "lms_vector_field_autograd",
    "make_lms_ball3d_widget",
    "make_lms_ball3d_backward_two_sheet_widget",
    "make_lms_ball3d_hydrodynamic_ensemble_widget",
    "make_lms_iframe_widget",
    "export_lms_static_payload",
    "write_lms_static_bundle",
    "make_disk_figure",
    "make_lms_circle_plotly_widget",
    "make_kuramoto_widget_2d",
    "make_sphere_figure",
    "mobius_ball",
    "mobius_sphere",
    "normalize",
    "order_parameter",
    "phases_to_s1",
    "project_to_so",
    "pushforward_points",
    "random_points_on_sphere",
    "reconstruct_points_at_frame",
    "reconstruct_sphere_trajectory",
    "reduced_general_vector_field",
    "skew_symmetric_from_axis",
]
