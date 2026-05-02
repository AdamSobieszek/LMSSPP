"""Canonical finite-N LMS gauge and moving-frame diagnostics.

This module uses the deboost convention z = c, where the canonical center
satisfies

    sum_i a_i M_z(x_i) = 0.

The widget reduced coordinate at identity rotation is w = -z.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .lms import DEFAULT_EPS, alpha_operator, clamp_to_ball, dot, mobius_sphere, normalize


@dataclass(frozen=True)
class CanonicalCenterSolve:
    z: Tensor
    residual_norm: float
    potential: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class CanonicalCloud:
    z: Tensor
    w: Tensor
    P: Tensor
    residual_norm: float
    center_error: float
    potential: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class ConnectionEstimate:
    omega: Tensor
    strain: Tensor
    raw: Tensor
    inertia: Tensor
    cross: Tensor


@dataclass(frozen=True)
class CanonicalTrajectoryData:
    z_series: Tensor
    w_series: Tensor
    P_series: Tensor
    center_errors: Tensor
    residual_norms: Tensor
    gram_series: Tensor
    inertia_spectra: Tensor
    converged: Tensor


@dataclass(frozen=True)
class MovingFrameDiagnostics:
    canonical: CanonicalTrajectoryData
    gram_drift: Tensor
    inertia_spectrum_drift: Tensor
    connection_mismatch: Tensor
    covariant_error: Tensor
    strain_error: Tensor
    z_equation_error: Tensor | None
    omega_pred_series: Tensor
    omega_data_series: Tensor


def _prepare_weights(points: Tensor, weights: Tensor) -> Tensor:
    if points.dim() != 2:
        raise ValueError("points must have shape [N,d].")
    if weights.dim() != 1 or int(weights.shape[0]) != int(points.shape[0]):
        raise ValueError("weights must have shape [N], matching points.")
    w = weights.to(dtype=points.dtype, device=points.device)
    if not torch.isfinite(w).all():
        raise ValueError("weights must be finite.")
    if bool((w < 0).any()):
        raise ValueError("weights must be nonnegative.")
    total = w.sum()
    if float(torch.abs(total)) <= DEFAULT_EPS:
        raise ValueError("weights must have positive sum.")
    return w / total


def ball_to_rapidity(v: Tensor, eps: float = 1e-12) -> Tensor:
    norm = torch.linalg.norm(v)
    if float(norm) < eps:
        return torch.zeros_like(v)
    norm_clip = torch.clamp(norm, max=1.0 - 1e-9)
    return v * (torch.atanh(norm_clip) / (norm + eps))


def rapidity_to_ball(y: Tensor, eps: float = 1e-12) -> Tensor:
    norm = torch.linalg.norm(y)
    if float(norm) < eps:
        return torch.zeros_like(y)
    return y * (torch.tanh(norm) / (norm + eps))


def weighted_barycenter(points: Tensor, weights: Tensor) -> Tensor:
    w = _prepare_weights(points, weights)
    return (w[:, None] * points).sum(dim=0)


def canonical_residual(c: Tensor, points: Tensor, weights: Tensor) -> Tensor:
    """Return B_x(c) = sum_i a_i M_c(x_i)."""
    w = _prepare_weights(points, weights)
    pushed = mobius_sphere(points, c)
    return (w[:, None] * pushed).sum(dim=0)


def busemann_cloud_potential(
    c: Tensor,
    points: Tensor,
    weights: Tensor,
    *,
    eps: float = 1e-12,
) -> Tensor:
    """Return Phi_x(c) = sum_i a_i log((1-|c|^2)/|c-x_i|^2)."""
    w = _prepare_weights(points, weights)
    c2 = dot(c, c)
    numer = (1.0 - c2).clamp(min=eps)
    diff = c.unsqueeze(0) - points
    denom = dot(diff, diff).clamp(min=eps)
    return (w * (torch.log(numer) - torch.log(denom))).sum()


def local_busemann_initializer(
    points: Tensor,
    weights: Tensor,
    *,
    lam: float = 1e-6,
) -> Tensor:
    """Small-center initializer z ~= 0.5 (I-C)^(-1) mean."""
    w = _prepare_weights(points, weights)
    d = int(points.shape[1])
    mu = (w[:, None] * points).sum(dim=0)
    C = (w[:, None, None] * (points[:, :, None] * points[:, None, :])).sum(dim=0)
    I = torch.eye(d, dtype=points.dtype, device=points.device)
    try:
        z0 = 0.5 * torch.linalg.solve(I - C + float(lam) * I, mu)
    except RuntimeError:
        z0 = 0.5 * torch.linalg.lstsq(I - C + float(lam) * I, mu.unsqueeze(-1)).solution.squeeze(-1)
    return clamp_to_ball(z0, radius=0.999999)


def _newton_polish_center(
    z: Tensor,
    points: Tensor,
    weights: Tensor,
    *,
    tol: float,
    max_steps: int = 8,
) -> tuple[Tensor, float]:
    """Polish the low-dimensional residual equation after potential ascent."""
    z_cur = z.detach()
    res_cur = canonical_residual(z_cur, points, weights)
    norm_cur = float(torch.linalg.norm(res_cur))
    for _ in range(max_steps):
        if norm_cur <= float(tol):
            break
        z_req = z_cur.detach().clone().requires_grad_(True)

        def residual_fn(inp: Tensor) -> Tensor:
            return canonical_residual(inp, points, weights)

        jac = torch.autograd.functional.jacobian(residual_fn, z_req).detach()
        res = residual_fn(z_req).detach()
        try:
            delta = torch.linalg.solve(jac, -res)
        except RuntimeError:
            delta = torch.linalg.lstsq(jac, -res.unsqueeze(-1)).solution.squeeze(-1)
        if not torch.isfinite(delta).all():
            break

        accepted = False
        for step in (1.0, 0.5, 0.25, 0.125, 0.0625):
            cand = clamp_to_ball(z_cur + float(step) * delta, radius=0.999999)
            norm_cand = float(torch.linalg.norm(canonical_residual(cand, points, weights)))
            if norm_cand < norm_cur:
                z_cur = cand.detach()
                norm_cur = norm_cand
                accepted = True
                break
        if not accepted:
            break
    return z_cur, norm_cur


def solve_canonical_center(
    points: Tensor,
    weights: Tensor,
    *,
    fallback_dir: Tensor | None = None,
    initial_center: Tensor | None = None,
    max_iters: int = 120,
    tol: float = 1e-10,
) -> CanonicalCenterSolve:
    """Solve sum_i a_i M_z(x_i)=0 by maximizing the Busemann potential."""
    w = _prepare_weights(points, weights)
    if initial_center is None:
        z0 = local_busemann_initializer(points, w)
    else:
        z0 = clamp_to_ball(initial_center.to(dtype=points.dtype, device=points.device), radius=0.999999)

    mean = (w[:, None] * points).sum(dim=0)
    if float(torch.linalg.norm(z0)) < 1e-10 and float(torch.linalg.norm(mean)) < 1e-10:
        if fallback_dir is None:
            z0 = torch.zeros(points.shape[1], dtype=points.dtype, device=points.device)
        else:
            z0 = normalize(fallback_dir.to(dtype=points.dtype, device=points.device).unsqueeze(0))[0] * 1e-4

    y = ball_to_rapidity(z0.detach()).requires_grad_(True)
    opt = torch.optim.LBFGS([y], lr=1.0, max_iter=20, history_size=30, line_search_fn="strong_wolfe")

    def closure() -> Tensor:
        opt.zero_grad(set_to_none=True)
        z = rapidity_to_ball(y)
        loss = -busemann_cloud_potential(z, points, w)
        loss.backward()
        return loss

    outer_iters = max(1, int(math.ceil(max(1, int(max_iters)) / 20.0)))
    res_norm = float("inf")
    iterations = 0
    for _ in range(outer_iters):
        opt.step(closure)
        iterations += 20
        with torch.no_grad():
            z = rapidity_to_ball(y)
            res = canonical_residual(z, points, w)
            res_norm = float(torch.linalg.norm(res))
            if res_norm <= float(tol):
                break

    with torch.no_grad():
        z_star = rapidity_to_ball(y.detach())
        if not torch.isfinite(z_star).all() or not math.isfinite(res_norm):
            raise RuntimeError("canonical Busemann center solve produced non-finite values.")
    if res_norm > float(tol):
        z_star, res_norm = _newton_polish_center(z_star, points, w, tol=float(tol))
    with torch.no_grad():
        potential = float(busemann_cloud_potential(z_star, points, w))
    return CanonicalCenterSolve(
        z=z_star,
        residual_norm=res_norm,
        potential=potential,
        iterations=min(iterations, max(1, int(max_iters))),
        converged=res_norm <= float(tol),
    )


def canonical_center(
    points: Tensor,
    weights: Tensor,
    *,
    fallback_dir: Tensor | None = None,
    initial_center: Tensor | None = None,
    max_iters: int = 120,
    tol: float = 1e-10,
) -> Tensor:
    """Return the canonical deboost center z with sum_i a_i M_z(x_i) ~= 0."""
    return solve_canonical_center(
        points,
        weights,
        fallback_dir=fallback_dir,
        initial_center=initial_center,
        max_iters=max_iters,
        tol=tol,
    ).z


def canonical_cloud(
    points: Tensor,
    weights: Tensor,
    *,
    fallback_dir: Tensor | None = None,
    initial_center: Tensor | None = None,
    max_iters: int = 120,
    tol: float = 1e-10,
) -> CanonicalCloud:
    """Return z, w=-z, and P_i=M_z(x_i) in the canonical centered gauge."""
    w = _prepare_weights(points, weights)
    solve = solve_canonical_center(
        points,
        w,
        fallback_dir=fallback_dir,
        initial_center=initial_center,
        max_iters=max_iters,
        tol=tol,
    )
    P = normalize(mobius_sphere(points, solve.z))
    center = weighted_barycenter(P, w)
    center_error = float(torch.linalg.norm(center))
    return CanonicalCloud(
        z=solve.z,
        w=-solve.z,
        P=P,
        residual_norm=solve.residual_norm,
        center_error=center_error,
        potential=solve.potential,
        iterations=solve.iterations,
        converged=solve.converged,
    )


def gram_matrix(P: Tensor) -> Tensor:
    return P @ P.T


def gram_conservation_error(G_series: Tensor) -> Tensor:
    if G_series.dim() != 3:
        raise ValueError("G_series must have shape [T,N,N].")
    return torch.amax(torch.abs(G_series - G_series[0].unsqueeze(0)), dim=(-2, -1))


def inertia_tensor(P: Tensor, weights: Tensor) -> Tensor:
    w = _prepare_weights(P, weights)
    return (w[:, None, None] * (P[:, :, None] * P[:, None, :])).sum(dim=0)


def inertia_spectrum(P: Tensor, weights: Tensor) -> Tensor:
    return torch.linalg.eigvalsh(inertia_tensor(P, weights))


def inertia_spectrum_conservation_error(spectrum_series: Tensor) -> Tensor:
    if spectrum_series.dim() != 2:
        raise ValueError("spectrum_series must have shape [T,d].")
    return torch.linalg.norm(spectrum_series - spectrum_series[0].unsqueeze(0), dim=-1)


def source_matrix(P: Tensor, weights: Tensor) -> Tensor:
    T = inertia_tensor(P, weights)
    I = torch.eye(T.shape[0], dtype=T.dtype, device=T.device)
    return I - T


def higher_moment_tensor(P: Tensor, weights: Tensor, order: int) -> Tensor:
    """Return sum_i a_i P_i^{tensor order}; intended only for small orders."""
    if order < 1:
        raise ValueError("order must be >= 1.")
    if order > 6:
        raise ValueError("higher_moment_tensor is intentionally limited to order <= 6.")
    w = _prepare_weights(P, weights)
    shape = (P.shape[0],) + (1,) * int(order)
    terms = w.reshape(shape)
    for axis in range(int(order)):
        view_shape = (P.shape[0],) + (1,) * axis + (P.shape[1],) + (1,) * (int(order) - axis - 1)
        terms = terms * P.reshape(view_shape)
    return terms.sum(dim=0)


def alpha_matrix(z: Tensor, Z: Tensor) -> Tensor:
    return alpha_operator(z, Z)


def predicted_connection(z: Tensor, Z: Tensor, A: Tensor | None = None) -> Tensor:
    omega = alpha_matrix(z, Z)
    if A is not None:
        omega = A.to(dtype=omega.dtype, device=omega.device) + omega
    return omega


def estimate_connection(P: Tensor, Pdot: Tensor, weights: Tensor, *, rcond: float = 1e-10) -> ConnectionEstimate:
    w = _prepare_weights(P, weights)
    T = inertia_tensor(P, w)
    C = (w[:, None, None] * (Pdot[:, :, None] * P[:, None, :])).sum(dim=0)
    raw = C @ torch.linalg.pinv(T, rtol=float(rcond))
    omega = 0.5 * (raw - raw.T)
    strain = 0.5 * (raw + raw.T)
    return ConnectionEstimate(omega=omega, strain=strain, raw=raw, inertia=T, cross=C)


def covariant_derivative_error(P: Tensor, Pdot: Tensor, Omega: Tensor, weights: Tensor) -> Tensor:
    w = _prepare_weights(P, weights)
    residual = Pdot - P @ Omega.T
    return torch.sqrt((w * dot(residual, residual)).sum().clamp(min=0.0))


def z_rhs(z: Tensor, Z: Tensor, A: Tensor | None = None) -> Tensor:
    if A is None:
        rot = torch.zeros_like(z)
    else:
        rot = z @ A.to(dtype=z.dtype, device=z.device).T
    return rot + 0.5 * (1.0 + dot(z, z)) * Z - dot(Z, z) * z


def _finite_difference_series(series: Tensor, dt: float) -> Tensor:
    if series.shape[0] <= 1:
        return torch.zeros_like(series)
    out = torch.empty_like(series)
    h = float(dt)
    out[0] = (series[1] - series[0]) / h
    out[-1] = (series[-1] - series[-2]) / h
    if series.shape[0] > 2:
        out[1:-1] = (series[2:] - series[:-2]) / (2.0 * h)
    return out


def canonicalize_trajectory(
    x_series: Tensor,
    weights: Tensor,
    *,
    fallback_dir: Tensor | None = None,
    max_iters: int = 120,
    tol: float = 1e-10,
    warm_start: bool = True,
) -> CanonicalTrajectoryData:
    if x_series.dim() != 3:
        raise ValueError("x_series must have shape [T,N,d].")
    w = _prepare_weights(x_series[0], weights)
    z_values = []
    w_values = []
    P_values = []
    center_errors = []
    residual_norms = []
    gram_values = []
    spectra = []
    converged = []
    z_prev = None
    for t in range(int(x_series.shape[0])):
        state = canonical_cloud(
            x_series[t],
            w,
            fallback_dir=fallback_dir,
            initial_center=z_prev if warm_start else None,
            max_iters=max_iters,
            tol=tol,
        )
        z_values.append(state.z)
        w_values.append(state.w)
        P_values.append(state.P)
        center_errors.append(state.center_error)
        residual_norms.append(state.residual_norm)
        gram_values.append(gram_matrix(state.P))
        spectra.append(inertia_spectrum(state.P, w))
        converged.append(state.converged)
        z_prev = state.z.detach()
    return CanonicalTrajectoryData(
        z_series=torch.stack(z_values, dim=0),
        w_series=torch.stack(w_values, dim=0),
        P_series=torch.stack(P_values, dim=0),
        center_errors=torch.as_tensor(center_errors, dtype=x_series.dtype, device=x_series.device),
        residual_norms=torch.as_tensor(residual_norms, dtype=x_series.dtype, device=x_series.device),
        gram_series=torch.stack(gram_values, dim=0),
        inertia_spectra=torch.stack(spectra, dim=0),
        converged=torch.as_tensor(converged, dtype=torch.bool, device=x_series.device),
    )


def _expand_A_series(A_series: Tensor | None, t_count: int, d: int, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    if A_series is None:
        return torch.zeros((t_count, d, d), dtype=dtype, device=device)
    A = A_series.to(dtype=dtype, device=device)
    if A.dim() == 2:
        return A.unsqueeze(0).expand(t_count, d, d)
    if A.dim() == 3 and int(A.shape[0]) == t_count:
        return A
    raise ValueError("A_series must have shape [d,d] or [T,d,d].")


def moving_frame_diagnostics(
    x_series: Tensor,
    weights: Tensor,
    dt: float,
    *,
    A_series: Tensor | None = None,
    Z_series: Tensor | None = None,
    fallback_dir: Tensor | None = None,
    max_iters: int = 120,
    tol: float = 1e-10,
) -> MovingFrameDiagnostics:
    canonical = canonicalize_trajectory(
        x_series,
        weights,
        fallback_dir=fallback_dir,
        max_iters=max_iters,
        tol=tol,
    )
    Pdot = _finite_difference_series(canonical.P_series, float(dt))
    zdot = _finite_difference_series(canonical.z_series, float(dt))
    t_count, _, d = x_series.shape
    w = _prepare_weights(x_series[0], weights)
    A = _expand_A_series(A_series, int(t_count), int(d), dtype=x_series.dtype, device=x_series.device)
    if Z_series is None:
        Z = torch.stack([weighted_barycenter(x_series[t], w) for t in range(int(t_count))], dim=0)
    else:
        Z = Z_series.to(dtype=x_series.dtype, device=x_series.device)
        if Z.shape != (t_count, d):
            raise ValueError("Z_series must have shape [T,d].")

    omega_pred = []
    omega_data = []
    conn_mismatch = []
    cov_errors = []
    strain_errors = []
    z_errors = []
    for t in range(int(t_count)):
        pred = predicted_connection(canonical.z_series[t], Z[t], A[t])
        est = estimate_connection(canonical.P_series[t], Pdot[t], w)
        omega_pred.append(pred)
        omega_data.append(est.omega)
        conn_mismatch.append(torch.linalg.norm(est.omega - pred))
        cov_errors.append(covariant_derivative_error(canonical.P_series[t], Pdot[t], pred, w))
        strain_errors.append(torch.linalg.norm(est.strain))
        z_errors.append(torch.linalg.norm(zdot[t] - z_rhs(canonical.z_series[t], Z[t], A[t])))

    return MovingFrameDiagnostics(
        canonical=canonical,
        gram_drift=gram_conservation_error(canonical.gram_series),
        inertia_spectrum_drift=inertia_spectrum_conservation_error(canonical.inertia_spectra),
        connection_mismatch=torch.stack(conn_mismatch, dim=0),
        covariant_error=torch.stack(cov_errors, dim=0),
        strain_error=torch.stack(strain_errors, dim=0),
        z_equation_error=torch.stack(z_errors, dim=0),
        omega_pred_series=torch.stack(omega_pred, dim=0),
        omega_data_series=torch.stack(omega_data, dim=0),
    )


def euclidean_busemann_gradient(u: Tensor, P: Tensor, weights: Tensor) -> Tensor:
    residual = canonical_residual(u, P, weights)
    factor = 2.0 / (1.0 - dot(u, u)).clamp(min=DEFAULT_EPS)
    return factor * residual


def es_energy(u: Tensor, u_tau: Tensor, P: Tensor, weights: Tensor) -> Tensor:
    grad = euclidean_busemann_gradient(u, P, weights)
    return 0.5 * dot(u_tau, u_tau) - 0.5 * dot(grad, grad)


__all__ = [
    "CanonicalCenterSolve",
    "CanonicalCloud",
    "CanonicalTrajectoryData",
    "ConnectionEstimate",
    "MovingFrameDiagnostics",
    "alpha_matrix",
    "ball_to_rapidity",
    "busemann_cloud_potential",
    "canonical_center",
    "canonical_cloud",
    "canonical_residual",
    "canonicalize_trajectory",
    "covariant_derivative_error",
    "es_energy",
    "estimate_connection",
    "euclidean_busemann_gradient",
    "gram_conservation_error",
    "gram_matrix",
    "higher_moment_tensor",
    "inertia_spectrum",
    "inertia_spectrum_conservation_error",
    "inertia_tensor",
    "local_busemann_initializer",
    "moving_frame_diagnostics",
    "predicted_connection",
    "rapidity_to_ball",
    "solve_canonical_center",
    "source_matrix",
    "weighted_barycenter",
    "z_rhs",
]
