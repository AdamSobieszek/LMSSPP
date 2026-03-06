"""Cucker-Smale dynamics utilities with singular Hessian communication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch
from torch import Tensor


DEFAULT_EPS = 1e-9


def dot(a: Tensor, b: Tensor) -> Tensor:
    return (a * b).sum(dim=-1)


def _validate_alpha(alpha: float) -> float:
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("alpha must be in (0,1).")
    return a


def _normalize_weights(weights: Tensor) -> Tensor:
    if weights.dim() != 1:
        raise ValueError("weights must have shape [N].")
    s = float(weights.sum().detach().cpu().item())
    if abs(s) < DEFAULT_EPS:
        raise ValueError("weights must not sum to zero.")
    return weights / s


def project_to_open_ball(x: Tensor, eps: float = DEFAULT_EPS) -> Tensor:
    nrm = x.norm(dim=-1, keepdim=True)
    return x / (1.0 + nrm + eps)


def grad_w_cs(
    dx: Tensor,
    alpha: float,
    eps: float = 1e-3,
    *,
    coupling: float = 1.0,
) -> Tensor:
    a = _validate_alpha(alpha)
    r = torch.sqrt(dot(dx, dx).unsqueeze(-1) + float(eps) ** 2)
    coeff = float(coupling) * (1.0 / (1.0 - a)) * torch.pow(r, -a)
    return coeff * dx


def hess_w_times_vec(
    dx: Tensor,
    dv: Tensor,
    alpha: float,
    eps: float = 1e-3,
    *,
    coupling: float = 1.0,
) -> Tensor:
    a = _validate_alpha(alpha)
    r = torch.sqrt(dot(dx, dx).unsqueeze(-1) + float(eps) ** 2)
    e = dx / r
    e_dot_dv = dot(e, dv).unsqueeze(-1)
    coeff = float(coupling) * (1.0 / (1.0 - a)) * torch.pow(r, -a)
    return coeff * (dv - a * e * e_dot_dv)


def _zero_diagonal_inplace(pair_tensor: Tensor, *, row_start: int = 0) -> None:
    if pair_tensor.dim() != 3:
        raise ValueError("pair_tensor must have shape [B,N,d].")
    b, n, _ = pair_tensor.shape
    idx_row = torch.arange(b, device=pair_tensor.device)
    idx_col = idx_row + int(row_start)
    valid = (idx_col >= 0) & (idx_col < n)
    if valid.any():
        pair_tensor[idx_row[valid], idx_col[valid], :] = 0.0


def cs_force(
    x: Tensor,
    weights: Tensor,
    *,
    alpha: float,
    eps: float = 1e-3,
    coupling: float = 1.0,
    chunk_size: int | None = None,
) -> Tensor:
    if x.dim() != 2:
        raise ValueError("x must have shape [N,d].")
    n, d = x.shape
    w = _normalize_weights(weights).to(dtype=x.dtype, device=x.device)
    if w.shape[0] != n:
        raise ValueError("weights shape must match x shape [N,d].")

    if chunk_size is None or chunk_size <= 0:
        chunk_size = n if n <= 1024 else 256

    if chunk_size >= n:
        dx = x.unsqueeze(1) - x.unsqueeze(0)
        grad = grad_w_cs(dx, alpha, eps, coupling=coupling)
        _zero_diagonal_inplace(grad, row_start=0)
        return (w.view(1, n, 1) * grad).sum(dim=1)

    out = torch.zeros((n, d), dtype=x.dtype, device=x.device)
    for row_start in range(0, n, int(chunk_size)):
        row_end = min(n, row_start + int(chunk_size))
        xi = x[row_start:row_end]
        dx = xi.unsqueeze(1) - x.unsqueeze(0)
        grad = grad_w_cs(dx, alpha, eps, coupling=coupling)
        _zero_diagonal_inplace(grad, row_start=row_start)
        out[row_start:row_end] = (w.view(1, n, 1) * grad).sum(dim=1)
    return out


def cs_first_order_rhs(
    x: Tensor,
    omega: Tensor,
    weights: Tensor,
    *,
    alpha: float,
    eps: float = 1e-3,
    coupling: float = 1.0,
    chunk_size: int | None = None,
) -> Tensor:
    if x.shape != omega.shape:
        raise ValueError("x and omega must have matching shape [N,d].")
    force = cs_force(
        x,
        weights,
        alpha=alpha,
        eps=eps,
        coupling=coupling,
        chunk_size=chunk_size,
    )
    return omega - force


def cs_reconstruct_velocity(
    x: Tensor,
    omega: Tensor,
    weights: Tensor,
    *,
    alpha: float,
    eps: float = 1e-3,
    coupling: float = 1.0,
    chunk_size: int | None = None,
) -> Tensor:
    return cs_first_order_rhs(
        x,
        omega,
        weights,
        alpha=alpha,
        eps=eps,
        coupling=coupling,
        chunk_size=chunk_size,
    )


def cs_second_order_rhs(
    x: Tensor,
    v: Tensor,
    weights: Tensor,
    *,
    alpha: float,
    eps: float = 1e-3,
    coupling: float = 1.0,
    chunk_size: int | None = None,
) -> Tensor:
    if x.shape != v.shape:
        raise ValueError("x and v must have matching shape [N,d].")
    if x.dim() != 2:
        raise ValueError("x and v must have shape [N,d].")
    n, d = x.shape
    w = _normalize_weights(weights).to(dtype=x.dtype, device=x.device)
    if w.shape[0] != n:
        raise ValueError("weights shape must match x shape [N,d].")

    if chunk_size is None or chunk_size <= 0:
        chunk_size = n if n <= 1024 else 256

    out = torch.zeros((n, d), dtype=x.dtype, device=x.device)
    for row_start in range(0, n, int(chunk_size)):
        row_end = min(n, row_start + int(chunk_size))
        xi = x[row_start:row_end]
        vi = v[row_start:row_end]
        dx = xi.unsqueeze(1) - x.unsqueeze(0)
        dv = v.unsqueeze(0) - vi.unsqueeze(1)
        hdv = hess_w_times_vec(dx, dv, alpha, eps, coupling=coupling)
        _zero_diagonal_inplace(hdv, row_start=row_start)
        out[row_start:row_end] = (w.view(1, n, 1) * hdv).sum(dim=1)
    return out


def compute_omega_invariant(
    x: Tensor,
    v: Tensor,
    weights: Tensor,
    *,
    alpha: float,
    eps: float = 1e-3,
    coupling: float = 1.0,
    chunk_size: int | None = None,
) -> Tensor:
    force = cs_force(
        x,
        weights,
        alpha=alpha,
        eps=eps,
        coupling=coupling,
        chunk_size=chunk_size,
    )
    return v + force


@dataclass
class CSFirstOrderTrajectory:
    x: Tensor
    v: Tensor | None
    force: Tensor | None
    mean_x: Tensor
    mean_v: Tensor
    omega: Tensor
    dt: float
    steps: int
    alpha: float
    eps: float
    coupling: float


@dataclass
class CSSecondOrderTrajectory:
    x: Tensor
    v: Tensor
    omega_diag: Tensor | None
    dt: float
    steps: int
    alpha: float
    eps: float
    coupling: float


def integrate_cs_first_order_euler(
    x0: Tensor,
    omega: Tensor,
    weights: Tensor,
    *,
    alpha: float,
    eps: float = 1e-3,
    coupling: float = 1.0,
    dt: float = 1e-3,
    steps: int = 1000,
    store_indices: Tensor | None = None,
    store_velocity: bool = False,
    store_force: bool = False,
    store_dtype: torch.dtype = torch.float32,
    preallocate: bool = True,
    chunk_size: int | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> CSFirstOrderTrajectory:
    if x0.shape != omega.shape or x0.dim() != 2:
        raise ValueError("x0 and omega must have shape [N,d].")
    n, d = x0.shape
    w = _normalize_weights(weights).to(dtype=x0.dtype, device=x0.device)
    if w.shape[0] != n:
        raise ValueError("weights shape must match x0 shape [N,d].")

    if store_indices is None:
        idx = torch.arange(n, device=x0.device, dtype=torch.long)
    else:
        idx = store_indices.to(device=x0.device, dtype=torch.long).reshape(-1)
        if idx.numel() == 0:
            raise ValueError("store_indices must not be empty.")
        if int(idx.min().item()) < 0 or int(idx.max().item()) >= n:
            raise ValueError("store_indices contain out-of-range values.")

    m = int(idx.numel())
    t_count = int(steps) + 1
    x = x0.clone()
    omega_const = omega.clone()

    if preallocate:
        x_hist = torch.empty((t_count, m, d), dtype=store_dtype, device=x0.device)
        mean_x_hist = torch.empty((t_count, d), dtype=store_dtype, device=x0.device)
        mean_v_hist = torch.empty((t_count, d), dtype=store_dtype, device=x0.device)
        v_hist = torch.empty((t_count, m, d), dtype=store_dtype, device=x0.device) if store_velocity else None
        f_hist = torch.empty((t_count, m, d), dtype=store_dtype, device=x0.device) if store_force else None

        for t in range(t_count):
            if cancel_check is not None and bool(cancel_check()):
                raise InterruptedError("Cucker-Smale first-order integration cancelled.")
            force_full = cs_force(
                x,
                w,
                alpha=alpha,
                eps=eps,
                coupling=coupling,
                chunk_size=chunk_size,
            )
            vel_full = omega_const - force_full
            x_sub = x.index_select(0, idx)
            x_hist[t].copy_(x_sub.detach())
            mean_x_hist[t].copy_((w.unsqueeze(-1) * x).sum(dim=0).detach())
            mean_v_hist[t].copy_((w.unsqueeze(-1) * vel_full).sum(dim=0).detach())
            if v_hist is not None:
                v_hist[t].copy_(vel_full.index_select(0, idx).detach())
            if f_hist is not None:
                f_hist[t].copy_(force_full.index_select(0, idx).detach())
            if t < steps:
                x = x + float(dt) * vel_full
    else:
        x_list = []
        mean_x_list = []
        mean_v_list = []
        v_list = [] if store_velocity else None
        f_list = [] if store_force else None
        for t in range(t_count):
            if cancel_check is not None and bool(cancel_check()):
                raise InterruptedError("Cucker-Smale first-order integration cancelled.")
            force_full = cs_force(
                x,
                w,
                alpha=alpha,
                eps=eps,
                coupling=coupling,
                chunk_size=chunk_size,
            )
            vel_full = omega_const - force_full
            x_list.append(x.index_select(0, idx).detach().to(dtype=store_dtype))
            mean_x_list.append((w.unsqueeze(-1) * x).sum(dim=0).detach().to(dtype=store_dtype))
            mean_v_list.append((w.unsqueeze(-1) * vel_full).sum(dim=0).detach().to(dtype=store_dtype))
            if v_list is not None:
                v_list.append(vel_full.index_select(0, idx).detach().to(dtype=store_dtype))
            if f_list is not None:
                f_list.append(force_full.index_select(0, idx).detach().to(dtype=store_dtype))
            if t < steps:
                x = x + float(dt) * vel_full
        x_hist = torch.stack(x_list, dim=0)
        mean_x_hist = torch.stack(mean_x_list, dim=0)
        mean_v_hist = torch.stack(mean_v_list, dim=0)
        v_hist = torch.stack(v_list, dim=0) if v_list is not None else None
        f_hist = torch.stack(f_list, dim=0) if f_list is not None else None

    return CSFirstOrderTrajectory(
        x=x_hist,
        v=v_hist,
        force=f_hist,
        mean_x=mean_x_hist,
        mean_v=mean_v_hist,
        omega=omega_const.detach().to(dtype=store_dtype),
        dt=float(dt),
        steps=int(steps),
        alpha=float(alpha),
        eps=float(eps),
        coupling=float(coupling),
    )


def integrate_cs_second_order_euler(
    x0: Tensor,
    v0: Tensor,
    weights: Tensor,
    *,
    alpha: float,
    eps: float = 1e-3,
    coupling: float = 1.0,
    dt: float = 1e-3,
    steps: int = 1000,
    store_indices: Tensor | None = None,
    store_dtype: torch.dtype = torch.float32,
    preallocate: bool = True,
    chunk_size: int | None = None,
    track_omega_invariant: bool = False,
    cancel_check: Callable[[], bool] | None = None,
) -> CSSecondOrderTrajectory:
    if x0.shape != v0.shape or x0.dim() != 2:
        raise ValueError("x0 and v0 must have shape [N,d].")
    n, d = x0.shape
    w = _normalize_weights(weights).to(dtype=x0.dtype, device=x0.device)
    if w.shape[0] != n:
        raise ValueError("weights shape must match x0 shape [N,d].")

    if store_indices is None:
        idx = torch.arange(n, device=x0.device, dtype=torch.long)
    else:
        idx = store_indices.to(device=x0.device, dtype=torch.long).reshape(-1)
        if idx.numel() == 0:
            raise ValueError("store_indices must not be empty.")
        if int(idx.min().item()) < 0 or int(idx.max().item()) >= n:
            raise ValueError("store_indices contain out-of-range values.")

    m = int(idx.numel())
    t_count = int(steps) + 1
    x = x0.clone()
    v = v0.clone()

    if preallocate:
        x_hist = torch.empty((t_count, m, d), dtype=store_dtype, device=x0.device)
        v_hist = torch.empty((t_count, m, d), dtype=store_dtype, device=x0.device)
        omega_hist = (
            torch.empty((t_count, m, d), dtype=store_dtype, device=x0.device) if track_omega_invariant else None
        )
        for t in range(t_count):
            if cancel_check is not None and bool(cancel_check()):
                raise InterruptedError("Cucker-Smale second-order integration cancelled.")
            x_hist[t].copy_(x.index_select(0, idx).detach())
            v_hist[t].copy_(v.index_select(0, idx).detach())
            if omega_hist is not None:
                omega_t = compute_omega_invariant(
                    x,
                    v,
                    w,
                    alpha=alpha,
                    eps=eps,
                    coupling=coupling,
                    chunk_size=chunk_size,
                )
                omega_hist[t].copy_(omega_t.index_select(0, idx).detach())
            if t < steps:
                vdot = cs_second_order_rhs(
                    x,
                    v,
                    w,
                    alpha=alpha,
                    eps=eps,
                    coupling=coupling,
                    chunk_size=chunk_size,
                )
                x = x + float(dt) * v
                v = v + float(dt) * vdot
    else:
        x_list = []
        v_list = []
        omega_list = [] if track_omega_invariant else None
        for t in range(t_count):
            if cancel_check is not None and bool(cancel_check()):
                raise InterruptedError("Cucker-Smale second-order integration cancelled.")
            x_list.append(x.index_select(0, idx).detach().to(dtype=store_dtype))
            v_list.append(v.index_select(0, idx).detach().to(dtype=store_dtype))
            if omega_list is not None:
                omega_t = compute_omega_invariant(
                    x,
                    v,
                    w,
                    alpha=alpha,
                    eps=eps,
                    coupling=coupling,
                    chunk_size=chunk_size,
                )
                omega_list.append(omega_t.index_select(0, idx).detach().to(dtype=store_dtype))
            if t < steps:
                vdot = cs_second_order_rhs(
                    x,
                    v,
                    w,
                    alpha=alpha,
                    eps=eps,
                    coupling=coupling,
                    chunk_size=chunk_size,
                )
                x = x + float(dt) * v
                v = v + float(dt) * vdot
        x_hist = torch.stack(x_list, dim=0)
        v_hist = torch.stack(v_list, dim=0)
        omega_hist = torch.stack(omega_list, dim=0) if omega_list is not None else None

    return CSSecondOrderTrajectory(
        x=x_hist,
        v=v_hist,
        omega_diag=omega_hist,
        dt=float(dt),
        steps=int(steps),
        alpha=float(alpha),
        eps=float(eps),
        coupling=float(coupling),
    )


@dataclass
class CuckerSmaleWidgetTrajectory:
    w: Tensor
    zeta: Tensor
    z: Tensor
    Z: Tensor
    Z_body: Tensor
    x_body: Tensor | None
    x_lab: Tensor | None
    base_points: Tensor
    dt: float
    steps: int
    coupling: float
    alpha: float
    eps: float
    omega: Tensor
    v: Tensor | None
    mean_x_raw: Tensor
    mean_v_raw: Tensor


def simulate_cs_widget_trajectory(
    *,
    x0: Tensor,
    omega: Tensor,
    weights: Tensor,
    alpha: float,
    eps: float,
    coupling: float,
    dt: float,
    steps: int,
    store_points: Literal["none", "body", "lab", "both"] = "both",
    store_indices: Tensor | None = None,
    store_dtype: torch.dtype = torch.float32,
    chunk_size: int | None = None,
    preallocate: bool = True,
    cancel_check: Callable[[], bool] | None = None,
) -> CuckerSmaleWidgetTrajectory:
    traj = integrate_cs_first_order_euler(
        x0=x0,
        omega=omega,
        weights=weights,
        alpha=alpha,
        eps=eps,
        coupling=coupling,
        dt=dt,
        steps=steps,
        store_indices=store_indices,
        store_velocity=True,
        store_force=False,
        store_dtype=store_dtype,
        preallocate=preallocate,
        chunk_size=chunk_size,
        cancel_check=cancel_check,
    )
    t_count, m, d = traj.x.shape
    x_max = float(torch.max(torch.linalg.norm(traj.x.reshape(-1, d), dim=-1)).detach().cpu().item())
    pos_scale = max(1.0, x_max)
    x_disp = project_to_open_ball(traj.x / pos_scale)
    z_disp = project_to_open_ball(traj.mean_x / pos_scale)
    Z_disp = project_to_open_ball(traj.mean_v)
    w_disp = -z_disp
    zeta = torch.eye(d, dtype=store_dtype, device=traj.x.device).unsqueeze(0).repeat(t_count, 1, 1)

    x_body: Tensor | None = None
    x_lab: Tensor | None = None
    if store_points in {"body", "both"}:
        x_body = x_disp
    if store_points in {"lab", "both"}:
        x_lab = x_disp

    base_points = x_disp[0] if m > 0 else torch.empty((0, d), dtype=store_dtype, device=traj.x.device)
    return CuckerSmaleWidgetTrajectory(
        w=w_disp,
        zeta=zeta,
        z=z_disp,
        Z=Z_disp,
        Z_body=Z_disp,
        x_body=x_body,
        x_lab=x_lab,
        base_points=base_points,
        dt=float(dt),
        steps=int(steps),
        coupling=float(coupling),
        alpha=float(alpha),
        eps=float(eps),
        omega=traj.omega,
        v=traj.v,
        mean_x_raw=traj.mean_x,
        mean_v_raw=traj.mean_v,
    )


def compare_reduced_vs_second_order_cs(
    *,
    x0: Tensor,
    v0: Tensor,
    weights: Tensor,
    alpha: float,
    eps: float = 1e-3,
    coupling: float = 1.0,
    dt: float = 1e-3,
    steps: int = 1000,
    chunk_size: int | None = None,
) -> dict[str, Tensor]:
    omega0 = compute_omega_invariant(
        x0,
        v0,
        weights,
        alpha=alpha,
        eps=eps,
        coupling=coupling,
        chunk_size=chunk_size,
    )
    first = integrate_cs_first_order_euler(
        x0=x0,
        omega=omega0,
        weights=weights,
        alpha=alpha,
        eps=eps,
        coupling=coupling,
        dt=dt,
        steps=steps,
        store_indices=None,
        store_velocity=True,
        store_force=False,
        store_dtype=x0.dtype,
        preallocate=True,
        chunk_size=chunk_size,
    )
    second = integrate_cs_second_order_euler(
        x0=x0,
        v0=v0,
        weights=weights,
        alpha=alpha,
        eps=eps,
        coupling=coupling,
        dt=dt,
        steps=steps,
        store_indices=None,
        store_dtype=x0.dtype,
        preallocate=True,
        chunk_size=chunk_size,
        track_omega_invariant=True,
    )
    x_diff = second.x - first.x
    v_recon = first.v if first.v is not None else torch.zeros_like(second.v)
    v_diff = second.v - v_recon

    omega_diag = second.omega_diag if second.omega_diag is not None else torch.zeros_like(second.v)
    omega_err = torch.linalg.norm(omega_diag - omega0.unsqueeze(0), dim=-1)
    return {
        "x_rms": torch.sqrt(torch.mean(torch.sum(x_diff * x_diff, dim=-1), dim=1)),
        "x_max": torch.max(torch.linalg.norm(x_diff, dim=-1), dim=1).values,
        "v_rms": torch.sqrt(torch.mean(torch.sum(v_diff * v_diff, dim=-1), dim=1)),
        "v_max": torch.max(torch.linalg.norm(v_diff, dim=-1), dim=1).values,
        "omega_inv_max": torch.max(omega_err, dim=1).values,
        "x_first": first.x,
        "x_second": second.x,
        "v_first_reconstructed": v_recon,
        "v_second": second.v,
    }


def make_cucker_smale_ball3d_widget(**kwargs: Any):
    try:
        from ..cucker_smale_ball3d_widget import CuckerSmaleBall3DWidget
    except Exception:
        from cucker_smale_ball3d_widget import CuckerSmaleBall3DWidget  # type: ignore
    return CuckerSmaleBall3DWidget(**kwargs)


def make_cucker_smale_ball3d_hydrodynamic_ensemble_widget(**kwargs: Any):
    try:
        from ..cucker_smale_ball3d_widget import CuckerSmaleBall3DHydrodynamicEnsembleWidget
    except Exception:
        from cucker_smale_ball3d_widget import CuckerSmaleBall3DHydrodynamicEnsembleWidget  # type: ignore
    return CuckerSmaleBall3DHydrodynamicEnsembleWidget(**kwargs)
