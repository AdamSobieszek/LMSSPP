"""
Reusable 2D Peszek--Poyato particle simulation and Hessian diagnostics.

This module evolves particles with fixed conserved labels ``omega_i`` by

    x_i' = omega_i - A_rho(x_i),

where ``A_rho`` is approximated on a regular grid by CIC deposition and FFT
convolution.  The original one-off script is kept as the default example, but
all pieces are now callable for arbitrary fiber layouts, shape families, and
prebuilt initial conditions.

Typical use:

    config = SimulationConfig(n_per_fiber=500, max_steps=200)
    initial = make_initial_condition(config)
    result = run_simulation(config, initial)
    analysis = analyze_hessian_geometry(result, config)
    fig = make_dashboard(result, analysis, config)
    save_outputs(result, analysis, config, dashboard=fig)
    open_dashboard(fig, config.out_dir)
    animation = make_dynamics_animation(result, config)
    save_animation(animation, config.out_dir)
    open_animation(animation, config.out_dir)

Run the default dashboard example with:

    python -m lmsspp.dynamics.pp_cs_equilibria
"""

from __future__ import annotations

import argparse
import json
import time
import webbrowser
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:  # Optional: only needed for the interactive notebook widget.
    import ipywidgets as widgets
except Exception:  # pragma: no cover - notebook dependency is optional
    widgets = None  # type: ignore[assignment]

Array = np.ndarray
ShapeSampler = Callable[[int, np.random.Generator], Array]

DEFAULT_SHAPES = (
    "gaussian",
    "ring",
    "arc",
    "line",
    "spiral",
    "square",
    "crescent",
    "ellipse",
    "two_mini_blobs",
    "triangle",
)

DEFAULT_PALETTE = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

DASHBOARD_FILENAME = "largeN_fft_hessian_dashboard.html"
ANIMATION_FILENAME = "largeN_fft_dynamics_animation.html"


@dataclass(frozen=True)
class FiberSpec:
    """One conserved-omega fiber in the initial condition."""

    shape: str | ShapeSampler = "gaussian"
    n_particles: int | None = None
    omega: tuple[float, float] | Array | None = None
    center: tuple[float, float] | Array | None = None
    name: str | None = None


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters for a 2D Peszek--Poyato FFT particle simulation."""

    alpha: float = 0.50
    K: float = 1.0
    n_fibers: int = 10
    n_per_fiber: int | Sequence[int] = 2000
    fibers: Sequence[FiberSpec] | None = None
    shape_names: Sequence[str] = DEFAULT_SHAPES
    omega_atoms: Array | None = None
    seed: int = 2026

    grid_size: int = 256
    domain_radius: float = 10.0
    dt: float = 0.055
    max_steps: int = 500
    tol_rms: float = 1.2e-2
    min_steps: int = 30
    record_every: int = 5

    farfield_shells: int = 18
    angles_per_shell: int = 64
    direct_hessian_chunk: int = 128

    max_plot_points_per_group: int = 1200
    max_metric_plot_points: int = 15000
    max_particle_csv_rows: int = 5000
    make_dashboard: bool = True
    make_animation: bool = False
    trajectory_frame_count: int = 0
    max_animation_points_per_group: int = 450
    animation_density_grid_size: int | None = 96
    animation_frame_duration_ms: int = 90
    animation_transition_ms: int = 0
    out_dir: Path | str = Path("pp_largeN_fft_hessian_output")


@dataclass(frozen=True)
class InitialCondition:
    """Particle positions, conserved labels, and display grouping."""

    x: Array
    omega: Array
    group_id: Array
    omega_atoms: Array
    group_names: tuple[str, ...]


@dataclass(frozen=True)
class SimulationResult:
    """State and residual diagnostics returned by ``run_simulation``."""

    initial: InitialCondition
    x_initial: Array
    x_final: Array
    A_final: Array
    residual: Array
    rho_grid: Array
    diagnostics: Array
    trajectory_x: Array | None
    trajectory_rho: Array | None
    trajectory_steps: Array | None
    trajectory_times: Array | None
    steps: int
    final_time: float
    runtime_seconds: float
    rms_residual: float
    max_residual: float


@dataclass(frozen=True)
class GeometryAnalysis:
    """Hessian metric diagnostics for a completed simulation."""

    Gxx_grid: Array
    Gxy_grid: Array
    Gyy_grid: Array
    lambda_min: Array
    lambda_max: Array
    detG: Array
    logdetG: Array
    anisotropy: Array
    lambda_radial: Array
    lambda_tangent: Array
    ratio_tangent_radial: Array
    farfield_shell: Array
    theory_lambda_radial: Array
    theory_lambda_tangent: Array
    metrics: dict[str, object]


class FFTPeszekPoyato2D:
    """Grid/FFT evaluator for the PP interaction field and Hessian metric."""

    def __init__(self, alpha: float, K: float, grid_size: int, domain_radius: float):
        if alpha >= 1:
            raise ValueError("alpha must be < 1 for the current PP kernel normalization")
        if grid_size < 4:
            raise ValueError("grid_size must be at least 4")
        if domain_radius <= 0:
            raise ValueError("domain_radius must be positive")

        self.alpha = float(alpha)
        self.K = float(K)
        self.G = int(grid_size)
        self.L = float(domain_radius)
        self.h = 2 * self.L / self.G
        self.P = 2 * self.G

        kernels = self._build_kernels()
        self.fft_Kx = np.fft.rfft2(kernels[0])
        self.fft_Ky = np.fft.rfft2(kernels[1])
        self.fft_Hxx = np.fft.rfft2(kernels[2])
        self.fft_Hxy = np.fft.rfft2(kernels[3])
        self.fft_Hyy = np.fft.rfft2(kernels[4])

    def clip_inside(self, x: Array) -> Array:
        margin = 2.1 * self.h
        return np.clip(x, -self.L + margin, self.L - margin)

    def convolve(self, rho_grid: Array, fft_kernel: Array) -> Array:
        padded = np.zeros((self.P, self.P), dtype=np.float64)
        padded[: self.G, : self.G] = rho_grid
        conv = np.fft.irfft2(np.fft.rfft2(padded) * fft_kernel, s=(self.P, self.P))
        return conv[: self.G, : self.G]

    def A_grid_from_particles(self, x: Array) -> tuple[Array, Array, Array]:
        rho_grid = deposit_mass(x, self.G, self.L)
        Ax_grid = self.convolve(rho_grid, self.fft_Kx)
        Ay_grid = self.convolve(rho_grid, self.fft_Ky)
        return rho_grid, Ax_grid, Ay_grid

    def A_at_particles(self, x: Array) -> tuple[Array, Array]:
        rho_grid, Ax_grid, Ay_grid = self.A_grid_from_particles(x)
        A = np.c_[
            interp_grid(Ax_grid, x, self.G, self.L),
            interp_grid(Ay_grid, x, self.G, self.L),
        ]
        return A, rho_grid

    def velocity(self, x: Array, omega: Array) -> tuple[Array, Array]:
        A, _ = self.A_at_particles(x)
        return omega - A, A

    def hessian_grid_from_rho(self, rho_grid: Array) -> tuple[Array, Array, Array]:
        return (
            self.convolve(rho_grid, self.fft_Hxx),
            self.convolve(rho_grid, self.fft_Hxy),
            self.convolve(rho_grid, self.fft_Hyy),
        )

    def _build_kernels(self) -> tuple[Array, Array, Array, Array, Array]:
        coords = make_lag_coords(self.P, self.h)
        Xlag, Ylag = np.meshgrid(coords, coords, indexing="ij")
        R = np.sqrt(Xlag**2 + Ylag**2)
        mask = R > 1e-14

        Kx = np.zeros((self.P, self.P), dtype=np.float64)
        Ky = np.zeros((self.P, self.P), dtype=np.float64)
        scale_grad = np.zeros_like(R)
        scale_grad[mask] = (R[mask] ** (-self.alpha)) / (1 - self.alpha)
        Kx[mask] = self.K * Xlag[mask] * scale_grad[mask]
        Ky[mask] = self.K * Ylag[mask] * scale_grad[mask]

        ex = np.zeros_like(R)
        ey = np.zeros_like(R)
        ex[mask] = Xlag[mask] / R[mask]
        ey[mask] = Ylag[mask] / R[mask]

        scale_hess = np.zeros_like(R)
        scale_hess[mask] = self.K * (R[mask] ** (-self.alpha)) / (1 - self.alpha)
        Hxx = np.zeros((self.P, self.P), dtype=np.float64)
        Hxy = np.zeros((self.P, self.P), dtype=np.float64)
        Hyy = np.zeros((self.P, self.P), dtype=np.float64)
        Hxx[mask] = scale_hess[mask] * (1 - self.alpha * ex[mask] * ex[mask])
        Hxy[mask] = scale_hess[mask] * (-self.alpha * ex[mask] * ey[mask])
        Hyy[mask] = scale_hess[mask] * (1 - self.alpha * ey[mask] * ey[mask])
        return Kx, Ky, Hxx, Hxy, Hyy


def rotate(points: Array, angle: float) -> Array:
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return points @ R.T


def sample_shape(kind: str, n: int, rng: np.random.Generator) -> Array:
    """Sample a centered 2D point cloud from a named display shape."""

    if kind == "gaussian":
        pts = rng.normal(size=(n, 2)) * np.array([0.55, 0.22])
    elif kind == "ring":
        th = rng.uniform(0, 2 * np.pi, n)
        r = 0.60 + 0.08 * rng.normal(size=n)
        pts = np.c_[r * np.cos(th), r * np.sin(th)]
    elif kind == "arc":
        th = rng.uniform(-1.35, 1.35, n)
        r = 0.70 + 0.08 * rng.normal(size=n)
        pts = np.c_[r * np.cos(th), r * np.sin(th)]
    elif kind == "line":
        t = rng.uniform(-0.72, 0.72, n)
        pts = np.c_[t, 0.08 * rng.normal(size=n)]
    elif kind == "spiral":
        th = rng.uniform(0.3, 3.2 * np.pi, n)
        r = 0.045 + 0.07 * th
        pts = np.c_[r * np.cos(th), r * np.sin(th)] + 0.05 * rng.normal(size=(n, 2))
    elif kind == "square":
        pts = rng.uniform(-0.55, 0.55, size=(n, 2))
    elif kind == "crescent":
        th = rng.uniform(-0.2, 1.45 * np.pi, n)
        r = 0.62 + 0.08 * rng.normal(size=n)
        pts = np.c_[r * np.cos(th), r * np.sin(th)]
        pts[:, 0] += 0.22 * np.sin(th)
    elif kind == "ellipse":
        pts = rng.normal(size=(n, 2)) * np.array([0.72, 0.12])
    elif kind == "two_mini_blobs":
        n1 = n // 2
        pts = np.vstack(
            [
                rng.normal(scale=0.13, size=(n1, 2)) + np.array([-0.34, 0.0]),
                rng.normal(scale=0.13, size=(n - n1, 2)) + np.array([0.34, 0.0]),
            ]
        )
    elif kind == "triangle":
        vertices = np.array([[0.0, 0.58], [-0.52, -0.36], [0.52, -0.36]])
        weights = rng.exponential(size=(n, 3))
        weights = weights / weights.sum(axis=1, keepdims=True)
        pts = weights @ vertices + 0.045 * rng.normal(size=(n, 2))
    else:
        raise ValueError(f"unknown shape kind: {kind!r}")

    pts -= pts.mean(axis=0, keepdims=True)
    return rotate(pts, rng.uniform(0, 2 * np.pi))


def make_lag_coords(P: int, h: float) -> Array:
    idx = np.arange(P)
    lag = np.where(idx <= P // 2, idx, idx - P)
    return lag * h


def cic_indices_weights(x: Array, G: int, L: float) -> tuple[Array, Array, Array, Array, Array, Array]:
    h = 2 * L / G
    u = (x[:, 0] + L) / h
    v = (x[:, 1] + L) / h
    i = np.floor(u).astype(np.int64)
    j = np.floor(v).astype(np.int64)
    i = np.clip(i, 0, G - 2)
    j = np.clip(j, 0, G - 2)
    fu = u - i
    fv = v - j
    return i, j, (1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv


def deposit_mass(x: Array, G: int, L: float) -> Array:
    if len(x) == 0:
        raise ValueError("cannot deposit an empty particle set")
    grid = np.zeros((G, G), dtype=np.float64)
    i, j, w00, w10, w01, w11 = cic_indices_weights(x, G, L)
    mass = 1.0 / len(x)
    np.add.at(grid, (i, j), mass * w00)
    np.add.at(grid, (i + 1, j), mass * w10)
    np.add.at(grid, (i, j + 1), mass * w01)
    np.add.at(grid, (i + 1, j + 1), mass * w11)
    return grid


def interp_grid(field: Array, x: Array, G: int, L: float) -> Array:
    i, j, w00, w10, w01, w11 = cic_indices_weights(x, G, L)
    return (
        field[i, j] * w00
        + field[i + 1, j] * w10
        + field[i, j + 1] * w01
        + field[i + 1, j + 1] * w11
    )


def direct_hessian_at(
    points: Array,
    sources: Array,
    alpha: float,
    K: float,
    chunk: int = 128,
) -> Array:
    """Direct finite-particle Hessian metric at query points."""

    out = np.zeros((points.shape[0], 2, 2), dtype=np.float64)
    N = sources.shape[0]
    for a in range(0, points.shape[0], chunk):
        p = points[a : a + chunk]
        diff = p[:, None, :] - sources[None, :, :]
        r = np.linalg.norm(diff, axis=-1)
        mask = r > 1e-14

        ex = np.zeros_like(r)
        ey = np.zeros_like(r)
        ex[mask] = diff[..., 0][mask] / r[mask]
        ey[mask] = diff[..., 1][mask] / r[mask]

        scale = np.zeros_like(r)
        scale[mask] = K * (r[mask] ** (-alpha)) / (1 - alpha) / N

        Hxx = scale * (1 - alpha * ex * ex)
        Hxy = scale * (-alpha * ex * ey)
        Hyy = scale * (1 - alpha * ey * ey)

        out[a : a + chunk, 0, 0] = Hxx.sum(axis=1)
        out[a : a + chunk, 0, 1] = Hxy.sum(axis=1)
        out[a : a + chunk, 1, 0] = Hxy.sum(axis=1)
        out[a : a + chunk, 1, 1] = Hyy.sum(axis=1)

    return out


def make_initial_condition(
    config: SimulationConfig,
    rng: np.random.Generator | None = None,
) -> InitialCondition:
    """Build a grouped particle cloud for arbitrary conserved omega fibers."""

    rng = np.random.default_rng(config.seed) if rng is None else rng
    fibers = _normalize_fibers(config)
    n_fibers = len(fibers)
    omega_atoms = _omega_atoms_from_config(config, fibers, rng)

    x_groups: list[Array] = []
    omega_groups: list[Array] = []
    group_ids: list[Array] = []
    group_names: list[str] = []

    for k, spec in enumerate(fibers):
        n = _fiber_count(config.n_per_fiber, k, spec.n_particles)
        if n <= 0:
            raise ValueError(f"fiber {k} has non-positive particle count {n}")

        cloud = _sample_fiber_shape(spec.shape, n, rng)
        center = _fiber_center(spec.center, rng)
        x_groups.append(cloud + center)
        omega_groups.append(np.repeat(omega_atoms[k][None, :], n, axis=0))
        group_ids.append(np.full(n, k, dtype=np.int64))
        group_names.append(spec.name or _shape_name(spec.shape, k))

    x = np.vstack(x_groups).astype(np.float64, copy=False)
    omega = np.vstack(omega_groups).astype(np.float64, copy=False)
    group_id = np.concatenate(group_ids)

    x -= x.mean(axis=0, keepdims=True)
    omega -= omega.mean(axis=0, keepdims=True)
    omega_atoms = np.array([omega[group_id == k][0] for k in range(n_fibers)])
    return InitialCondition(x=x, omega=omega, group_id=group_id, omega_atoms=omega_atoms, group_names=tuple(group_names))


def run_simulation(config: SimulationConfig, initial: InitialCondition | None = None) -> SimulationResult:
    """Evolve a PP particle system to an approximate equilibrium."""

    initial = make_initial_condition(config) if initial is None else validate_initial_condition(initial)
    solver = FFTPeszekPoyato2D(config.alpha, config.K, config.grid_size, config.domain_radius)

    x = solver.clip_inside(initial.x.copy())
    x_initial = x.copy()
    diagnostics: list[tuple[int, float, float, float]] = []
    trajectory_x: list[Array] = []
    trajectory_rho: list[Array] = []
    trajectory_steps: list[int] = []
    trajectory_times: list[float] = []
    trajectory_stride = _trajectory_stride(config)
    trajectory_limit = _trajectory_frame_limit(config)

    def record_trajectory(step_: int, x_: Array, *, force: bool = False) -> None:
        if not config.make_animation:
            return
        if trajectory_steps and not force and step_ % trajectory_stride != 0 and step_ != config.max_steps:
            return
        if trajectory_limit is not None and len(trajectory_steps) >= trajectory_limit and not force and step_ != config.max_steps:
            return
        density_x = x_
        density_grid_size = config.animation_density_grid_size
        if density_grid_size is None or int(density_grid_size) == solver.G:
            rho = deposit_mass(density_x, solver.G, solver.L)
        else:
            rho = deposit_mass(density_x, int(density_grid_size), solver.L)
        trajectory_x.append(x_.astype(np.float32, copy=True))
        trajectory_rho.append(rho.astype(np.float32, copy=False))
        trajectory_steps.append(int(step_))
        trajectory_times.append(float(step_ * config.dt))

    start = time.time()

    for step in range(config.max_steps + 1):
        record_trajectory(step, x)
        vf, _ = solver.velocity(x, initial.omega)
        speed2 = np.sum(vf * vf, axis=1)
        rms = float(np.sqrt(np.mean(speed2)))
        maxv = float(np.sqrt(np.max(speed2)))
        if step % config.record_every == 0:
            diagnostics.append((step, step * config.dt, rms, maxv))
        if rms < config.tol_rms and step > config.min_steps:
            break
        if step < config.max_steps:
            x_pred = solver.clip_inside(x + config.dt * vf)
            k2, _ = solver.velocity(x_pred, initial.omega)
            x = x + 0.5 * config.dt * (vf + k2)
            x -= x.mean(axis=0, keepdims=True)
            x = solver.clip_inside(x)

    runtime = time.time() - start
    if config.make_animation and (not trajectory_steps or trajectory_steps[-1] != step):
        record_trajectory(step, x, force=True)
    A_final, rho_grid = solver.A_at_particles(x)
    residual = initial.omega - A_final
    residual_speed2 = np.sum(residual * residual, axis=1)
    return SimulationResult(
        initial=initial,
        x_initial=x_initial,
        x_final=x.copy(),
        A_final=A_final,
        residual=residual,
        rho_grid=rho_grid,
        diagnostics=np.array(diagnostics, dtype=np.float64),
        trajectory_x=np.stack(trajectory_x) if trajectory_x else None,
        trajectory_rho=np.stack(trajectory_rho) if trajectory_rho else None,
        trajectory_steps=np.array(trajectory_steps, dtype=np.int64) if trajectory_steps else None,
        trajectory_times=np.array(trajectory_times, dtype=np.float64) if trajectory_times else None,
        steps=int(step),
        final_time=float(step * config.dt),
        runtime_seconds=float(runtime),
        rms_residual=float(np.sqrt(np.mean(residual_speed2))),
        max_residual=float(np.sqrt(np.max(residual_speed2))),
    )


def analyze_hessian_geometry(result: SimulationResult, config: SimulationConfig) -> GeometryAnalysis:
    """Compute local Hessian metric and direct far-field cone diagnostics."""

    if config.farfield_shells < 2:
        raise ValueError("farfield_shells must be at least 2")
    if config.angles_per_shell < 3:
        raise ValueError("angles_per_shell must be at least 3")

    solver = FFTPeszekPoyato2D(config.alpha, config.K, config.grid_size, config.domain_radius)
    x_final = result.x_final
    Gxxg, Gxyg, Gyyg = solver.hessian_grid_from_rho(result.rho_grid)
    Gxxp = interp_grid(Gxxg, x_final, solver.G, solver.L)
    Gxyp = interp_grid(Gxyg, x_final, solver.G, solver.L)
    Gyyp = interp_grid(Gyyg, x_final, solver.G, solver.L)

    tr = Gxxp + Gyyp
    disc = np.sqrt(np.maximum((Gxxp - Gyyp) ** 2 + 4 * Gxyp**2, 0))
    lam_min = 0.5 * (tr - disc)
    lam_max = 0.5 * (tr + disc)
    detG = Gxxp * Gyyp - Gxyp * Gxyp
    anisotropy = lam_max / np.maximum(lam_min, 1e-30)
    logdetG = np.log(np.maximum(detG, 1e-300))

    center = x_final.mean(axis=0)
    rel = x_final - center
    rr = np.linalg.norm(rel, axis=1)
    er = np.zeros_like(rel)
    mask = rr > 1e-12
    er[mask] = rel[mask] / rr[mask, None]
    et = np.c_[-er[:, 1], er[:, 0]]
    lam_rad = Gxxp * er[:, 0] ** 2 + 2 * Gxyp * er[:, 0] * er[:, 1] + Gyyp * er[:, 1] ** 2
    lam_tan = Gxxp * et[:, 0] ** 2 + 2 * Gxyp * et[:, 0] * et[:, 1] + Gyyp * et[:, 1] ** 2
    ratio_tan_rad = lam_tan / np.maximum(lam_rad, 1e-30)

    farfield_shell = compute_farfield_cone_test(result, config)
    Rvals, lr_mean, _, lt_mean, _, ratio_mean = farfield_shell.T
    slope_r, _ = np.polyfit(np.log(Rvals), np.log(lr_mean), 1)
    slope_t, _ = np.polyfit(np.log(Rvals), np.log(lt_mean), 1)

    alpha_hat_rad = -float(slope_r)
    alpha_hat_tan = -float(slope_t)
    ratio_hat = float(np.mean(ratio_mean[-max(4, config.farfield_shells // 4) :]))
    alpha_hat_ratio = 1 - 1 / ratio_hat
    gamma_theory = (2 - config.alpha) / (2 * np.sqrt(1 - config.alpha))
    gamma_hat = (2 - alpha_hat_rad) / 2 * np.sqrt(max(ratio_hat, 0))
    theory_lr = config.K * Rvals ** (-config.alpha)
    theory_lt = config.K * (1 / (1 - config.alpha)) * Rvals ** (-config.alpha)

    metrics: dict[str, object] = {
        "seed": int(config.seed),
        "alpha": float(config.alpha),
        "K": float(config.K),
        "N": int(len(x_final)),
        "n_fibers": int(len(result.initial.group_names)),
        "particles_per_fiber": _group_counts(result.initial.group_id),
        "group_names": list(result.initial.group_names),
        "grid_G": int(config.grid_size),
        "domain_L": float(config.domain_radius),
        "dt": float(config.dt),
        "steps": int(result.steps),
        "final_time": float(result.final_time),
        "runtime_seconds": float(result.runtime_seconds),
        "rms_equilibrium_residual_fft": result.rms_residual,
        "max_equilibrium_residual_fft": result.max_residual,
        "mean_x": x_final.mean(axis=0).tolist(),
        "mean_omega": result.initial.omega.mean(axis=0).tolist(),
        "hessian_eig_min_median": float(np.median(lam_min)),
        "hessian_eig_max_median": float(np.median(lam_max)),
        "hessian_anisotropy_median": float(np.median(anisotropy)),
        "hessian_anisotropy_p95": float(np.quantile(anisotropy, 0.95)),
        "logdetG_mean": float(np.mean(logdetG)),
        "logdetG_std": float(np.std(logdetG)),
        "local_tangent_radial_ratio_median": float(np.median(ratio_tan_rad[np.isfinite(ratio_tan_rad)])),
        "farfield_alpha_hat_radial_slope": alpha_hat_rad,
        "farfield_alpha_hat_tangential_slope": alpha_hat_tan,
        "farfield_ratio_tangent_over_radial_hat": ratio_hat,
        "farfield_alpha_hat_from_ratio": float(alpha_hat_ratio),
        "gamma_theory": float(gamma_theory),
        "gamma_hat_from_ratio_and_slope": float(gamma_hat),
    }

    return GeometryAnalysis(
        Gxx_grid=Gxxg,
        Gxy_grid=Gxyg,
        Gyy_grid=Gyyg,
        lambda_min=lam_min,
        lambda_max=lam_max,
        detG=detG,
        logdetG=logdetG,
        anisotropy=anisotropy,
        lambda_radial=lam_rad,
        lambda_tangent=lam_tan,
        ratio_tangent_radial=ratio_tan_rad,
        farfield_shell=farfield_shell,
        theory_lambda_radial=theory_lr,
        theory_lambda_tangent=theory_lt,
        metrics=metrics,
    )


def compute_farfield_cone_test(result: SimulationResult, config: SimulationConfig) -> Array:
    center = result.x_final.mean(axis=0)
    support_radius = float(np.max(np.linalg.norm(result.x_final - center, axis=1)))
    radii = np.geomspace(
        max(2.5 * support_radius, 1.0),
        max(10.0 * support_radius, 4.0),
        config.farfield_shells,
    )
    angles = np.linspace(0, 2 * np.pi, config.angles_per_shell, endpoint=False)
    rows = []

    for radius in radii:
        pts = center + np.c_[radius * np.cos(angles), radius * np.sin(angles)]
        H = direct_hessian_at(pts, result.x_final, config.alpha, config.K, chunk=config.direct_hessian_chunk)
        ering = np.c_[np.cos(angles), np.sin(angles)]
        eting = np.c_[-np.sin(angles), np.cos(angles)]
        lr = np.einsum("ni,nij,nj->n", ering, H, ering)
        lt = np.einsum("ni,nij,nj->n", eting, H, eting)
        rows.append(
            [
                radius,
                float(lr.mean()),
                float(lr.std()),
                float(lt.mean()),
                float(lt.std()),
                float((lt / lr).mean()),
            ]
        )

    return np.array(rows, dtype=np.float64)


def save_outputs(
    result: SimulationResult,
    analysis: GeometryAnalysis,
    config: SimulationConfig,
    dashboard: go.Figure | None = None,
) -> Path:
    """Write metrics, CSV samples, compressed arrays, and optional dashboard."""

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(_jsonable(analysis.metrics), indent=2))
    (out_dir / "config.json").write_text(json.dumps(_config_to_json(config), indent=2))

    write_time_diagnostics(out_dir / "time_diagnostics.csv", result.diagnostics)
    write_farfield_csv(out_dir / "farfield_cone_test.csv", analysis)
    write_particle_sample_csv(out_dir / "particle_geometry_sample.csv", result, analysis, config)
    write_npz(out_dir / "largeN_data_compressed.npz", result, analysis)

    if config.make_dashboard:
        fig = dashboard or make_dashboard(result, analysis, config)
        fig.write_html(str(out_dir / DASHBOARD_FILENAME), include_plotlyjs="cdn")

    return out_dir


def open_dashboard(dashboard: go.Figure, out_dir: Path | str) -> Path:
    """Open the dashboard as an HTML file in the system browser."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path = out_dir / DASHBOARD_FILENAME
    if not dashboard_path.exists():
        dashboard.write_html(str(dashboard_path), include_plotlyjs="cdn")
    webbrowser.open(dashboard_path.resolve().as_uri())
    return dashboard_path


def make_dynamics_animation(result: SimulationResult, config: SimulationConfig) -> go.Figure:
    """Create a precomputed Plotly animation of fiber and joint-density dynamics."""

    if result.trajectory_x is None or result.trajectory_rho is None:
        raise ValueError("run_simulation must be called with config.make_animation=True")
    if result.trajectory_steps is None or result.trajectory_times is None:
        raise ValueError("trajectory metadata is missing")

    trajectory_x = np.asarray(result.trajectory_x, dtype=np.float64)
    trajectory_rho = np.asarray(result.trajectory_rho, dtype=np.float64)
    steps = np.asarray(result.trajectory_steps, dtype=np.int64)
    times = np.asarray(result.trajectory_times, dtype=np.float64)
    group_id = result.initial.group_id
    group_names = result.initial.group_names
    frame_count = trajectory_x.shape[0]
    if frame_count == 0:
        raise ValueError("trajectory is empty")

    rng = np.random.default_rng(config.seed + 3)
    sampled_by_group = _animation_sample_indices(group_id, config.max_animation_points_per_group, rng)
    density_axis = _density_axis(config, trajectory_rho.shape[1])
    zmax = float(np.nanquantile(trajectory_rho, 0.995)) if trajectory_rho.size else 1.0
    zmax = max(zmax, 1e-12)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scatter"}, {"type": "heatmap"}]],
        subplot_titles=[
            "Per-omega fiber particle dynamics",
            "Precomputed joint density rho_t",
        ],
        horizontal_spacing=0.08,
    )

    for k, name in enumerate(group_names):
        idx = sampled_by_group[k]
        color = DEFAULT_PALETTE[k % len(DEFAULT_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=trajectory_x[0, idx, 0],
                y=trajectory_x[0, idx, 1],
                mode="markers",
                marker=dict(size=4, color=color, opacity=0.70),
                name=f"fiber {k + 1}: {name}",
                legendgroup=f"fiber{k}",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Heatmap(
            x=density_axis,
            y=density_axis,
            z=trajectory_rho[0].T,
            colorscale="Viridis",
            zmin=0.0,
            zmax=zmax,
            zsmooth=False,
            colorbar=dict(title="mass"),
            name="rho_t",
        ),
        row=1,
        col=2,
    )

    frames: list[go.Frame] = []
    animated_trace_indices = list(range(len(group_names) + 1))
    for frame_idx in range(frame_count):
        traces: list[go.Scatter | go.Heatmap] = []
        for k in range(len(group_names)):
            idx = sampled_by_group[k]
            traces.append(
                go.Scatter(
                    x=trajectory_x[frame_idx, idx, 0],
                    y=trajectory_x[frame_idx, idx, 1],
                    mode="markers",
                    marker=dict(size=4, color=DEFAULT_PALETTE[k % len(DEFAULT_PALETTE)], opacity=0.70),
                )
            )
        traces.append(
            go.Heatmap(
                z=trajectory_rho[frame_idx].T,
                zmin=0.0,
                zmax=zmax,
                colorscale="Viridis",
                zsmooth=False,
            )
        )
        frames.append(
            go.Frame(
                data=traces,
                name=str(frame_idx),
                traces=animated_trace_indices,
                layout=go.Layout(
                    title_text=_animation_title(config, result, int(steps[frame_idx]), float(times[frame_idx]), frame_idx, frame_count)
                ),
            )
        )

    fig.frames = frames
    fig.update_xaxes(title_text="x1", range=[-config.domain_radius/2, config.domain_radius/2], scaleanchor="y", scaleratio=1, row=1, col=1)
    fig.update_yaxes(title_text="x2", range=[-config.domain_radius/2, config.domain_radius/2], row=1, col=1)
    fig.update_xaxes(title_text="x1", range=[-config.domain_radius/2, config.domain_radius/2], row=1, col=2)
    fig.update_yaxes(title_text="x2", range=[-config.domain_radius/2, config.domain_radius/2], scaleanchor="x2", scaleratio=1, row=1, col=2)
    fig.update_layout(
        title=dict(text=_animation_title(config, result, int(steps[0]), float(times[0]), 0, frame_count), x=0.5),
        width=1450,
        height=720,
        template="plotly_white",
        legend=dict(groupclick="togglegroup", itemsizing="constant"),
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.08,
                "y": -0.11,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": config.animation_frame_duration_ms, "redraw": False},
                                "transition": {"duration": config.animation_transition_ms},
                                "fromcurrent": True,
                                "mode": "immediate",
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.08,
                "y": -0.18,
                "len": 0.84,
                "currentvalue": {"prefix": "frame "},
                "steps": [
                    {
                        "label": str(i),
                        "method": "animate",
                        "args": [
                            [str(i)],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    }
                    for i in range(frame_count)
                ],
            }
        ],
        margin=dict(l=50, r=30, t=90, b=120),
    )
    return fig


def save_animation(animation: go.Figure, out_dir: Path | str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    animation_path = out_dir / ANIMATION_FILENAME
    animation.write_html(str(animation_path), include_plotlyjs="cdn")
    return animation_path


def open_animation(animation: go.Figure, out_dir: Path | str) -> Path:
    out_dir = Path(out_dir)
    animation_path = out_dir / ANIMATION_FILENAME
    if not animation_path.exists():
        animation_path = save_animation(animation, out_dir)
    webbrowser.open(animation_path.resolve().as_uri())
    return animation_path


def _plotly_values(values: Any) -> Any:
    """Convert cached numeric arrays to FigureWidget-safe Python containers.

    ``go.FigureWidget`` accepts numpy arrays, but its ipywidgets delta-sync path
    can later evaluate them in a boolean context and raise ``ValueError: truth
    value of an array is ambiguous``. Keep numpy in caches, but convert before
    assigning to trace fields.
    """
    if isinstance(values, np.ndarray):
        return values.tolist()
    return values


def make_dynamics_widget(
    config: SimulationConfig,
    result: SimulationResult | None = None,
    *,
    width: int = 1450,
    height: int = 720,
) -> "PPDynamicsWidget":
    """Build the interactive notebook animation widget.

    Mirrors the precompute / play / step / frame-slider caching model of
    ``LMSOpticalDiskWidget``: every frame is precomputed once, then playback
    only swaps cached arrays into the existing traces, so the joint-density
    heatmap updates in place instead of being redrawn each frame.
    """

    return PPDynamicsWidget(config, result=result, width=width, height=height)


class PPDynamicsWidget:
    """Interactive ``go.FigureWidget`` animation of fiber + joint-density dynamics.

    The control set and caching model are copied from
    ``LMSOpticalDiskWidget``:

    * ``Precompute`` runs the simulation (recording every step) and builds the
      per-frame payload cache.
    * ``widgets.Play`` provides start / pause / stop transport, ``Step`` advances
      one frame, and the frame slider scrubs; all of them funnel through
      ``_set_frame_index`` -> ``_apply_cached_frame``.
    * ``_apply_cached_frame`` only writes cached arrays into existing traces via
      ``fig.batch_update()`` (scatter ``x``/``y`` per fiber, heatmap ``z``), which
      is what keeps the heatmap smooth instead of strobing.

    Display the ``.layout`` attribute (or the widget itself) in a notebook cell.
    """

    def __init__(
        self,
        config: SimulationConfig,
        result: SimulationResult | None = None,
        *,
        width: int = 1450,
        height: int = 720,
    ) -> None:
        if widgets is None:
            raise RuntimeError(
                "PPDynamicsWidget requires ipywidgets (install the 'widgets' extra) "
                "and a live notebook kernel."
            )
        self.config = config
        self.width = int(width)
        self.height = int(height)

        self._result: SimulationResult | None = None
        self._frame_payloads: list[dict[str, Any]] = []
        self._frame_index = 0
        self._cache_valid = False
        self._updating = False
        self._sampled_by_group: list[Array] = []
        self._density_axis: Array | None = None
        self._zmax = 1.0

        fibers = _normalize_fibers(config)
        self._group_names: tuple[str, ...] = tuple(
            spec.name or _shape_name(spec.shape, k) for k, spec in enumerate(fibers)
        )

        self._build_controls()
        self._build_figure()
        self._wire_events()
        self.layout = widgets.VBox(
            [
                widgets.HTML(self._header_html()),
                self.fig,
                self.controls,
                self.stats_html,
            ]
        )

        if result is not None:
            self._ingest_result(result)
        else:
            self._mark_cache_stale("Click Precompute to run the simulation and cache frames.")

    # -- construction -------------------------------------------------------

    def _header_html(self) -> str:
        return (
            "<b>Peszek--Poyato large-N FFT dynamics</b><br>"
            "Left: sampled per-omega fiber particles. Right: precomputed joint "
            "spatial density. Precompute caches every step, then Play/Step/slider "
            "swap cached frames in place."
        )

    def _build_controls(self) -> None:
        self.btn_precompute = widgets.Button(
            description="Precompute",
            button_style="warning",
            layout=widgets.Layout(width="140px"),
        )
        self.btn_step = widgets.Button(
            description="Step",
            disabled=True,
            layout=widgets.Layout(width="90px"),
        )
        self.play = widgets.Play(
            value=0,
            min=0,
            max=0,
            step=1,
            interval=max(1, int(self.config.animation_frame_duration_ms)),
            disabled=True,
            show_repeat=True,
            layout=widgets.Layout(width="160px"),
        )
        if "repeat" in self.play.traits():
            self.play.repeat = True
        self.frame_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=0,
            step=1,
            description="frame",
            continuous_update=True,
            disabled=True,
            layout=widgets.Layout(width="640px"),
        )
        self.frame_counter = widgets.HTML(value="frame 0 / 0", layout=widgets.Layout(width="120px"))
        self.cache_status_html = widgets.HTML(value="", layout=widgets.Layout(width="420px"))
        self.stats_html = widgets.HTML(value="")
        self.controls = widgets.VBox(
            [
                widgets.HBox([self.btn_step, self.play, self.btn_precompute, self.cache_status_html]),
                widgets.HBox([self.frame_slider, self.frame_counter]),
            ]
        )

    def _build_figure(self) -> None:
        fig = go.FigureWidget(
            make_subplots(
                rows=1,
                cols=2,
                specs=[[{"type": "scatter"}, {"type": "heatmap"}]],
                subplot_titles=[
                    "Per-omega fiber particle dynamics",
                    "Precomputed joint density rho_t",
                ],
                horizontal_spacing=0.08,
            )
        )
        self.fig = fig
        self.tr: dict[str, int] = {}

        for k, name in enumerate(self._group_names):
            color = DEFAULT_PALETTE[k % len(DEFAULT_PALETTE)]
            fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode="markers",
                    marker=dict(size=4, color=color, opacity=0.70),
                    name=f"fiber {k + 1}: {name}",
                    legendgroup=f"fiber{k}",
                ),
                row=1,
                col=1,
            )
            self.tr[f"fiber{k}"] = len(fig.data) - 1

        fig.add_trace(
            go.Heatmap(
                x=[],
                y=[],
                z=[[0.0]],
                colorscale="Viridis",
                zmin=0.0,
                zmax=1.0,
                zsmooth=False,
                colorbar=dict(title="mass"),
                name="rho_t",
            ),
            row=1,
            col=2,
        )
        self.tr["density"] = len(fig.data) - 1

        r = self.config.domain_radius
        fig.update_xaxes(title_text="x1", range=[-r, r], scaleanchor="y", scaleratio=1, row=1, col=1)
        fig.update_yaxes(title_text="x2", range=[-r, r], row=1, col=1)
        fig.update_xaxes(title_text="x1", range=[-r, r], row=1, col=2)
        fig.update_yaxes(title_text="x2", range=[-r, r], scaleanchor="x2", scaleratio=1, row=1, col=2)
        fig.update_layout(
            width=self.width,
            height=self.height,
            template="plotly_white",
            legend=dict(groupclick="togglegroup", itemsizing="constant"),
            margin=dict(l=50, r=30, t=70, b=40),
        )

    def _wire_events(self) -> None:
        self.btn_precompute.on_click(self._on_precompute_clicked)
        self.btn_step.on_click(self._on_step)
        self.play.observe(self._on_play_tick, names="value")
        self.frame_slider.observe(self._on_frame_slider, names="value")

    # -- cache + frame state ------------------------------------------------

    def precompute(self) -> None:
        """Run the simulation, build the frame cache, and show the first frame."""

        self._cache_valid = False
        self.cache_status_html.value = "<span style='color:#666'>Running simulation and caching frames...</span>"
        self.btn_precompute.disabled = True
        self.btn_precompute.button_style = "info"
        self.btn_step.disabled = True
        self.play.disabled = True
        self.frame_slider.disabled = True
        self._updating = True
        try:
            self.play.value = 0
            self.frame_slider.value = 0
        finally:
            self._updating = False

        run_config = self.config if self.config.make_animation else replace(self.config, make_animation=True)
        result = run_simulation(run_config)
        self._ingest_result(result)

    def _ingest_result(self, result: SimulationResult) -> None:
        if result.trajectory_x is None or result.trajectory_rho is None:
            raise ValueError("result has no trajectory; run with config.make_animation=True")
        self._result = result
        self._group_names = result.initial.group_names
        rng = np.random.default_rng(self.config.seed + 3)
        self._sampled_by_group = _animation_sample_indices(
            result.initial.group_id, self.config.max_animation_points_per_group, rng
        )
        trajectory_rho = np.asarray(result.trajectory_rho, dtype=np.float64)
        self._density_axis = _density_axis(self.config, trajectory_rho.shape[1])
        zmax = float(np.nanquantile(trajectory_rho, 0.995)) if trajectory_rho.size else 1.0
        self._zmax = max(zmax, 1e-12)
        self._frame_payloads = self._build_frame_payloads()
        self._frame_index = 0
        self._apply_static_payload_to_figure()
        self._apply_cached_frame(0)
        self._mark_cache_ready(f"Cache ready: {len(self._frame_payloads)} frames.")

    def _build_frame_payloads(self) -> list[dict[str, Any]]:
        result = self._result
        assert result is not None
        trajectory_x = np.asarray(result.trajectory_x, dtype=np.float64)
        trajectory_rho = np.asarray(result.trajectory_rho, dtype=np.float64)
        steps = np.asarray(result.trajectory_steps, dtype=np.int64)
        times = np.asarray(result.trajectory_times, dtype=np.float64)
        frame_count = trajectory_x.shape[0]

        payloads: list[dict[str, Any]] = []
        for f in range(frame_count):
            fiber_x: list[Array] = []
            fiber_y: list[Array] = []
            for k in range(len(self._group_names)):
                idx = self._sampled_by_group[k]
                fiber_x.append(trajectory_x[f, idx, 0])
                fiber_y.append(trajectory_x[f, idx, 1])
            payloads.append(
                {
                    "fiber_x": fiber_x,
                    "fiber_y": fiber_y,
                    "density_z": trajectory_rho[f].T,
                    "title": _animation_title(self.config, result, int(steps[f]), float(times[f]), f, frame_count),
                    "stats": (
                        f"frame {f + 1}/{frame_count}; step={int(steps[f])}; "
                        f"t={float(times[f]):.4g}; final residual RMS={result.rms_residual:.3g}"
                    ),
                }
            )
        return payloads

    def _apply_static_payload_to_figure(self) -> None:
        if self._density_axis is None:
            return
        axis = _plotly_values(self._density_axis)
        with self.fig.batch_update():
            density = self.fig.data[self.tr["density"]]
            density.x = axis
            density.y = axis
            density.zmin = 0.0
            density.zmax = float(self._zmax)

    def _apply_cached_frame(self, frame_idx: int) -> None:
        if not self._frame_payloads:
            return
        idx = int(np.clip(int(frame_idx), 0, len(self._frame_payloads) - 1))
        payload = self._frame_payloads[idx]
        with self.fig.batch_update():
            for k in range(len(self._group_names)):
                trace = self.fig.data[self.tr[f"fiber{k}"]]
                trace.x = _plotly_values(payload["fiber_x"][k])
                trace.y = _plotly_values(payload["fiber_y"][k])
            self.fig.data[self.tr["density"]].z = _plotly_values(payload["density_z"])
            self.fig.layout.title.text = payload["title"]
        self.stats_html.value = payload["stats"]

    def _set_frame_index(self, idx: int, *, source: Any | None = None) -> None:
        self._sync_frame_controls(idx, source=source)
        self._apply_cached_frame(self._frame_index)

    def _sync_frame_controls(self, idx: int, *, source: Any | None = None) -> None:
        count = len(self._frame_payloads)
        max_idx = max(0, count - 1)
        frame = int(np.clip(int(idx), 0, max_idx)) if count else 0
        self._frame_index = frame
        disabled = (not self._cache_valid) or count <= 1
        self._updating = True
        try:
            if int(self.frame_slider.max) != max_idx:
                self.frame_slider.max = max_idx
            if bool(self.frame_slider.disabled) != disabled:
                self.frame_slider.disabled = disabled
            if source is not self.frame_slider and int(self.frame_slider.value) != frame:
                self.frame_slider.value = frame
            if int(self.play.max) != max_idx:
                self.play.max = max_idx
            if bool(self.play.disabled) != disabled:
                self.play.disabled = disabled
            if source is not self.play and int(self.play.value) != frame:
                self.play.value = frame
            if bool(self.btn_step.disabled) != disabled:
                self.btn_step.disabled = disabled
            counter_text = f"frame {frame + 1 if count else 0} / {count}"
            if self.frame_counter.value != counter_text:
                self.frame_counter.value = counter_text
        finally:
            self._updating = False

    def _mark_cache_stale(self, message: str) -> None:
        self._cache_valid = False
        self.btn_precompute.disabled = False
        self.btn_precompute.button_style = "warning"
        self.btn_step.disabled = True
        self.play.disabled = True
        self.frame_slider.disabled = True
        self.cache_status_html.value = f"<span style='color:#9a6700'>{message}</span>"

    def _mark_cache_ready(self, message: str) -> None:
        self._cache_valid = True
        count = len(self._frame_payloads)
        self.btn_precompute.disabled = False
        self.btn_precompute.button_style = "success"
        self.btn_step.disabled = count <= 1
        self.play.disabled = count <= 1
        self.frame_slider.disabled = count <= 1
        self.cache_status_html.value = f"<span style='color:#188038'>{message}</span>"
        self._sync_frame_controls(self._frame_index)

    # -- callbacks ----------------------------------------------------------

    def _on_precompute_clicked(self, _btn: Any) -> None:
        self.precompute()

    def _on_step(self, _btn: Any) -> None:
        if not self._cache_valid or len(self._frame_payloads) <= 1:
            return
        self._set_frame_index((self._frame_index + 1) % len(self._frame_payloads))

    def _on_play_tick(self, change: dict[str, Any]) -> None:
        if self._updating or change.get("name") != "value":
            return
        if not self._cache_valid:
            self._sync_frame_controls(0)
            return
        self._set_frame_index(int(change.get("new", 0)), source=self.play)

    def _on_frame_slider(self, change: dict[str, Any]) -> None:
        if self._updating or change.get("name") != "value":
            return
        if not self._cache_valid:
            return
        self._set_frame_index(int(self.frame_slider.value), source=self.frame_slider)

    def _ipython_display_(self) -> None:  # pragma: no cover - notebook display hook
        from IPython.display import display

        display(self.layout)


def make_dashboard(result: SimulationResult, analysis: GeometryAnalysis, config: SimulationConfig) -> go.Figure:
    """Create the reusable Plotly diagnostic dashboard."""

    rng = np.random.default_rng(config.seed + 1)
    group_names = result.initial.group_names
    group_id = result.initial.group_id
    x_final = result.x_final
    omega = result.initial.omega
    N = len(x_final)

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"type": "scattergl"}, {"type": "scattergl"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scattergl"}, {"type": "histogram"}],
        ],
        subplot_titles=[
            "Final spatial marginal: selectable omega fibers",
            f"Conserved omega marginal: {len(group_names)} groups, matching colors",
            "Equilibrium residual convergence",
            "Direct far-field cone test",
            "Sampled Hessian volume distortion log det g_rho",
            "Metric anisotropy distribution lambda_max/lambda_min",
        ],
        horizontal_spacing=0.10,
        vertical_spacing=0.11,
    )

    for k, name in enumerate(group_names):
        idx_all = np.where(group_id == k)[0]
        if len(idx_all) == 0:
            continue
        idx = rng.choice(idx_all, size=min(config.max_plot_points_per_group, len(idx_all)), replace=False)
        color = DEFAULT_PALETTE[k % len(DEFAULT_PALETTE)]
        trace_name = f"omega fiber {k + 1}: {name}"

        fig.add_trace(
            go.Scattergl(
                x=x_final[idx, 0],
                y=x_final[idx, 1],
                mode="markers",
                marker=dict(size=4, color=color, opacity=0.72),
                name=trace_name,
                legendgroup=f"fiber{k}",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        jitter = 0.018 * rng.normal(size=(len(idx), 2))
        fig.add_trace(
            go.Scattergl(
                x=omega[idx, 0] + jitter[:, 0],
                y=omega[idx, 1] + jitter[:, 1],
                mode="markers",
                marker=dict(size=4, color=color, opacity=0.72, symbol="diamond"),
                name=f"{trace_name} omega atom",
                legendgroup=f"fiber{k}",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    diag = result.diagnostics
    fig.add_trace(go.Scatter(x=diag[:, 1], y=diag[:, 2], mode="lines+markers", name="RMS residual", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=diag[:, 1], y=diag[:, 3], mode="lines", name="max residual", showlegend=False), row=2, col=1)

    shell = analysis.farfield_shell
    Rvals, lr_mean, _, lt_mean, _, _ = shell.T
    fig.add_trace(go.Scatter(x=Rvals, y=lr_mean, mode="markers+lines", name="radial eig", showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=Rvals, y=lt_mean, mode="markers+lines", name="tangent eig", showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=Rvals, y=analysis.theory_lambda_radial, mode="lines", line=dict(dash="dash"), name="theory rad", showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=Rvals, y=analysis.theory_lambda_tangent, mode="lines", line=dict(dash="dash"), name="theory tan", showlegend=False), row=2, col=2)

    plot_idx = np.sort(rng.choice(N, size=min(config.max_metric_plot_points, N), replace=False))
    fig.add_trace(
        go.Scattergl(
            x=x_final[plot_idx, 0],
            y=x_final[plot_idx, 1],
            mode="markers",
            marker=dict(size=3.2, color=analysis.logdetG[plot_idx], colorscale="Viridis", showscale=True, colorbar=dict(title="log det g")),
            name="logdetG",
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    fig.add_trace(go.Histogram(x=analysis.anisotropy[plot_idx], nbinsx=60, name="anisotropy", showlegend=False), row=3, col=2)

    fig.update_xaxes(title_text="x1", scaleanchor="y", scaleratio=1, row=1, col=1)
    fig.update_yaxes(title_text="x2", row=1, col=1)
    fig.update_xaxes(title_text="omega1", scaleanchor="y2", scaleratio=1, row=1, col=2)
    fig.update_yaxes(title_text="omega2", row=1, col=2)
    fig.update_xaxes(title_text="time", row=2, col=1)
    fig.update_yaxes(title_text="|omega-A_rho(x)|", type="log", row=2, col=1)
    fig.update_xaxes(title_text="R", type="log", row=2, col=2)
    fig.update_yaxes(title_text="eigenvalue", type="log", row=2, col=2)
    fig.update_xaxes(title_text="x1", scaleanchor="y5", scaleratio=1, row=3, col=1)
    fig.update_yaxes(title_text="x2", row=3, col=1)
    fig.update_xaxes(title_text="lambda_max/lambda_min", row=3, col=2)
    fig.update_yaxes(title_text="count", row=3, col=2)

    fig.update_layout(
        title=dict(
            text=(
                "FFT Peszek--Poyato Hessian geometry diagnostics<br>"
                f"<sup>N={N:,}, fibers={len(group_names)}, grid={config.grid_size}^2, "
                f"alpha={config.alpha}, seed={config.seed}, gamma_theory={analysis.metrics['gamma_theory']:.4f}, "
                f"gamma_hat={analysis.metrics['gamma_hat_from_ratio_and_slope']:.4f}, "
                f"residual RMS={result.rms_residual:.3g}, runtime={result.runtime_seconds:.1f}s</sup>"
            ),
            x=0.5,
        ),
        width=1400,
        height=1200,
        template="plotly_white",
        legend=dict(groupclick="togglegroup", itemsizing="constant"),
    )
    return fig


def write_time_diagnostics(path: Path, diagnostics: Array) -> None:
    with path.open("w") as f:
        f.write("step,time,rms_residual,max_residual\n")
        for row in diagnostics:
            f.write(",".join(map(str, row)) + "\n")


def write_farfield_csv(path: Path, analysis: GeometryAnalysis) -> None:
    shell = analysis.farfield_shell
    with path.open("w") as f:
        f.write("R,lambda_radial_mean,lambda_radial_std,lambda_tangent_mean,lambda_tangent_std,ratio_tangent_radial_mean,theory_lambda_radial,theory_lambda_tangent\n")
        for i in range(len(shell)):
            row = list(shell[i]) + [analysis.theory_lambda_radial[i], analysis.theory_lambda_tangent[i]]
            f.write(",".join(map(str, row)) + "\n")


def write_particle_sample_csv(path: Path, result: SimulationResult, analysis: GeometryAnalysis, config: SimulationConfig) -> None:
    rng = np.random.default_rng(config.seed + 2)
    N = len(result.x_final)
    sample_idx = np.sort(rng.choice(N, size=min(config.max_particle_csv_rows, N), replace=False))
    with path.open("w") as f:
        f.write("i,group,x1,x2,omega1,omega2,A1,A2,res1,res2,lambda_min,lambda_max,anisotropy,detG,logdetG,lambda_radial,lambda_tangent,ratio_tangent_radial\n")
        for i in sample_idx:
            f.write(
                f"{i},{result.initial.group_id[i]},{result.x_final[i, 0]},{result.x_final[i, 1]},"
                f"{result.initial.omega[i, 0]},{result.initial.omega[i, 1]},{result.A_final[i, 0]},{result.A_final[i, 1]},"
                f"{result.residual[i, 0]},{result.residual[i, 1]},"
                f"{analysis.lambda_min[i]},{analysis.lambda_max[i]},{analysis.anisotropy[i]},{analysis.detG[i]},{analysis.logdetG[i]},"
                f"{analysis.lambda_radial[i]},{analysis.lambda_tangent[i]},{analysis.ratio_tangent_radial[i]}\n"
            )


def write_npz(path: Path, result: SimulationResult, analysis: GeometryAnalysis) -> None:
    trajectory_x = np.empty((0, 0, 2), dtype=np.float32) if result.trajectory_x is None else result.trajectory_x.astype(np.float32)
    trajectory_rho = np.empty((0, 0, 0), dtype=np.float32) if result.trajectory_rho is None else result.trajectory_rho.astype(np.float32)
    trajectory_steps = np.empty((0,), dtype=np.int64) if result.trajectory_steps is None else result.trajectory_steps.astype(np.int64)
    trajectory_times = np.empty((0,), dtype=np.float64) if result.trajectory_times is None else result.trajectory_times.astype(np.float64)
    np.savez_compressed(
        path,
        x_initial=result.x_initial.astype(np.float32),
        x_final=result.x_final.astype(np.float32),
        omega=result.initial.omega.astype(np.float32),
        omega_atoms=result.initial.omega_atoms.astype(np.float32),
        group_id=result.initial.group_id.astype(np.int32),
        A_final=result.A_final.astype(np.float32),
        residual=result.residual.astype(np.float32),
        lam_min=analysis.lambda_min.astype(np.float32),
        lam_max=analysis.lambda_max.astype(np.float32),
        detG=analysis.detG.astype(np.float32),
        logdetG=analysis.logdetG.astype(np.float32),
        diagnostics=result.diagnostics.astype(np.float64),
        farfield_shell=analysis.farfield_shell.astype(np.float64),
        trajectory_x=trajectory_x,
        trajectory_rho=trajectory_rho,
        trajectory_steps=trajectory_steps,
        trajectory_times=trajectory_times,
    )


def validate_initial_condition(initial: InitialCondition) -> InitialCondition:
    x = np.asarray(initial.x, dtype=np.float64)
    omega = np.asarray(initial.omega, dtype=np.float64)
    group_id = np.asarray(initial.group_id, dtype=np.int64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("initial.x must have shape (N, 2)")
    if len(x) == 0:
        raise ValueError("initial condition cannot be empty")
    if omega.shape != x.shape:
        raise ValueError("initial.omega must have the same shape as initial.x")
    if group_id.shape != (len(x),):
        raise ValueError("initial.group_id must have shape (N,)")
    unique_groups = np.unique(group_id)
    expected_groups = np.arange(len(unique_groups))
    if not np.array_equal(unique_groups, expected_groups):
        raise ValueError("initial.group_id values must be contiguous integers starting at 0")
    if len(initial.group_names) != len(unique_groups):
        raise ValueError("initial.group_names must match the number of groups")
    omega_atoms = np.asarray(initial.omega_atoms, dtype=np.float64)
    if omega_atoms.shape != (len(unique_groups), 2):
        raise ValueError("initial.omega_atoms must have shape (n_groups, 2)")
    return InitialCondition(
        x=x,
        omega=omega,
        group_id=group_id,
        omega_atoms=omega_atoms,
        group_names=tuple(initial.group_names),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-fibers", type=int, default=10, help="number of conserved omega groups")
    parser.add_argument("--n-per-fiber", type=int, default=2000, help="particles per omega group")
    parser.add_argument("--shapes", default=",".join(DEFAULT_SHAPES), help="comma-separated shape names, cycled across fibers")
    parser.add_argument("--alpha", type=float, default=0.50)
    parser.add_argument("--K", type=float, default=1.0)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--domain-radius", type=float, default=9.0)
    parser.add_argument("--dt", type=float, default=0.055)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--tol-rms", type=float, default=1.2e-2)
    parser.add_argument("--farfield-shells", type=int, default=18)
    parser.add_argument("--angles-per-shell", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None, help="random seed; omitted means choose a fresh seed for this run")
    parser.add_argument("--out-dir", type=Path, default=Path("pp_largeN_fft_hessian_output"))
    parser.add_argument("--no-dashboard", action="store_true", help="skip Plotly HTML generation")
    parser.add_argument("--no-animation", action="store_true", help="skip the dynamics animation HTML")
    parser.add_argument("--trajectory-frame-count", type=int, default=0, help="maximum precomputed animation frames; 0 records every simulation step")
    parser.add_argument("--max-animation-points-per-group", type=int, default=450, help="sampled particles per omega fiber in the animation")
    parser.add_argument("--animation-density-grid-size", type=int, default=96, help="density grid size for precomputed animation image frames")
    parser.add_argument("--animation-frame-duration-ms", type=int, default=90)
    parser.add_argument("--no-show", action="store_true", help="do not open the Plotly dashboard after the simulation")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> SimulationConfig:
    shapes = tuple(part.strip() for part in args.shapes.split(",") if part.strip())
    if not shapes:
        raise ValueError("--shapes must include at least one shape")
    return SimulationConfig(
        alpha=args.alpha,
        K=args.K,
        n_fibers=args.n_fibers,
        n_per_fiber=args.n_per_fiber,
        shape_names=shapes,
        seed=_seed_from_args(args.seed),
        grid_size=args.grid_size,
        domain_radius=args.domain_radius,
        dt=args.dt,
        max_steps=args.max_steps,
        tol_rms=args.tol_rms,
        farfield_shells=args.farfield_shells,
        angles_per_shell=args.angles_per_shell,
        out_dir=args.out_dir,
        make_dashboard=not args.no_dashboard,
        make_animation=not args.no_animation,
        trajectory_frame_count=args.trajectory_frame_count,
        max_animation_points_per_group=args.max_animation_points_per_group,
        animation_density_grid_size=args.animation_density_grid_size,
        animation_frame_duration_ms=args.animation_frame_duration_ms,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    initial = make_initial_condition(config)
    result = run_simulation(config, initial)
    analysis = analyze_hessian_geometry(result, config)
    dashboard = None
    if config.make_dashboard or not args.no_show:
        dashboard = make_dashboard(result, analysis, config)
    out_dir = save_outputs(result, analysis, config, dashboard=dashboard)
    animation = None
    if config.make_animation:
        animation = make_dynamics_animation(result, config)
        save_animation(animation, out_dir)

    if dashboard is not None and not args.no_show:
        open_dashboard(dashboard, out_dir)
    if animation is not None and not args.no_show:
        open_animation(animation, out_dir)

    print("Done.")
    print(json.dumps(_jsonable(analysis.metrics), indent=2))
    print("Output directory:", out_dir)


def _normalize_fibers(config: SimulationConfig) -> tuple[FiberSpec, ...]:
    if config.fibers is not None:
        fibers = tuple(config.fibers)
        if not fibers:
            raise ValueError("config.fibers cannot be empty")
        return fibers
    if config.n_fibers <= 0:
        raise ValueError("n_fibers must be positive")
    if not config.shape_names:
        raise ValueError("shape_names must include at least one shape")
    return tuple(FiberSpec(shape=config.shape_names[k % len(config.shape_names)]) for k in range(config.n_fibers))


def _seed_from_args(seed: int | None) -> int:
    if seed is not None:
        return int(seed)
    return int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])


def _trajectory_stride(config: SimulationConfig) -> int:
    if int(config.trajectory_frame_count) <= 0:
        return 1
    frame_count = max(2, int(config.trajectory_frame_count))
    return max(1, int(np.ceil(max(1, config.max_steps) / max(1, frame_count - 1))))


def _trajectory_frame_limit(config: SimulationConfig) -> int | None:
    if int(config.trajectory_frame_count) <= 0:
        return None
    return max(2, int(config.trajectory_frame_count))


def _animation_sample_indices(group_id: Array, max_per_group: int, rng: np.random.Generator) -> list[Array]:
    if max_per_group <= 0:
        raise ValueError("max_animation_points_per_group must be positive")
    samples: list[Array] = []
    group_count = int(group_id.max()) + 1
    for k in range(group_count):
        idx_all = np.where(group_id == k)[0]
        if len(idx_all) == 0:
            samples.append(np.array([], dtype=np.int64))
            continue
        idx = rng.choice(idx_all, size=min(int(max_per_group), len(idx_all)), replace=False)
        samples.append(np.sort(idx))
    return samples


def _density_axis(config: SimulationConfig, grid_size: int) -> Array:
    h = 2 * config.domain_radius / int(grid_size)
    return np.linspace(-config.domain_radius + 0.5 * h, config.domain_radius - 0.5 * h, int(grid_size))


def _animation_title(
    config: SimulationConfig,
    result: SimulationResult,
    step: int,
    time_value: float,
    frame_index: int,
    frame_count: int,
) -> str:
    return (
        "Large-N FFT Peszek--Poyato dynamics animation<br>"
        f"<sup>N={len(result.x_final):,}, fibers={len(result.initial.group_names)}, "
        f"grid={config.grid_size}^2, alpha={config.alpha}, seed={config.seed}, "
        f"frame={frame_index + 1}/{frame_count}, step={step}, t={time_value:.4g}, "
        f"final residual RMS={result.rms_residual:.3g}</sup>"
    )


def _omega_atoms_from_config(config: SimulationConfig, fibers: Sequence[FiberSpec], rng: np.random.Generator) -> Array:
    explicit = [spec.omega for spec in fibers]
    if any(omega is not None for omega in explicit):
        if not all(omega is not None for omega in explicit):
            raise ValueError("either provide omega for every FiberSpec or for none")
        atoms = np.asarray(explicit, dtype=np.float64)
    elif config.omega_atoms is not None:
        atoms = np.asarray(config.omega_atoms, dtype=np.float64)
        if atoms.shape != (len(fibers), 2):
            raise ValueError(f"omega_atoms must have shape ({len(fibers)}, 2)")
    else:
        atoms = sample_omega_atoms(len(fibers), rng)
    atoms = atoms.astype(np.float64, copy=True)
    atoms -= atoms.mean(axis=0, keepdims=True)
    return atoms


def sample_omega_atoms(n_fibers: int, rng: np.random.Generator) -> Array:
    """Default mildly anisotropic ring of conserved omega atoms."""

    theta = np.linspace(0, 2 * np.pi, n_fibers, endpoint=False)
    rng.shuffle(theta)
    r = rng.uniform(0.55, 1.45, size=n_fibers)
    atoms = np.c_[r * np.cos(theta), 0.78 * r * np.sin(theta)]
    atoms += 0.16 * rng.normal(size=atoms.shape)
    atoms -= atoms.mean(axis=0, keepdims=True)
    return atoms


def _fiber_count(n_per_fiber: int | Sequence[int], index: int, override: int | None) -> int:
    if override is not None:
        return int(override)
    if isinstance(n_per_fiber, int):
        return int(n_per_fiber)
    if index >= len(n_per_fiber):
        raise ValueError("n_per_fiber sequence must include one count per fiber")
    return int(n_per_fiber[index])


def _sample_fiber_shape(shape: str | ShapeSampler, n: int, rng: np.random.Generator) -> Array:
    if callable(shape):
        pts = np.asarray(shape(n, rng), dtype=np.float64)
    else:
        pts = sample_shape(shape, n, rng)
    if pts.shape != (n, 2):
        raise ValueError(f"fiber sampler returned shape {pts.shape}, expected ({n}, 2)")
    pts = pts.astype(np.float64, copy=True)
    pts -= pts.mean(axis=0, keepdims=True)
    return pts


def _fiber_center(center: tuple[float, float] | Array | None, rng: np.random.Generator) -> Array:
    if center is not None:
        value = np.asarray(center, dtype=np.float64)
        if value.shape != (2,):
            raise ValueError("fiber center must have shape (2,)")
        return value
    value = rng.normal(size=2)
    value = 2.3 * value / max(np.linalg.norm(value), 1e-9) + 0.35 * rng.normal(size=2)
    return value


def _shape_name(shape: str | ShapeSampler, index: int) -> str:
    if isinstance(shape, str):
        return shape
    return getattr(shape, "__name__", f"custom_shape_{index}")


def _group_counts(group_id: Array) -> list[int]:
    return [int(np.sum(group_id == k)) for k in range(int(group_id.max()) + 1)]


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _config_to_json(config: SimulationConfig) -> dict[str, object]:
    data = asdict(config)
    data["out_dir"] = str(config.out_dir)
    data["omega_atoms"] = None if config.omega_atoms is None else np.asarray(config.omega_atoms).tolist()
    if config.fibers is not None:
        data["fibers"] = [_fiber_spec_to_json(spec) for spec in config.fibers]
    return _jsonable(data)  # type: ignore[return-value]


def _fiber_spec_to_json(spec: FiberSpec) -> dict[str, object]:
    return {
        "shape": _shape_name(spec.shape, 0),
        "n_particles": spec.n_particles,
        "omega": None if spec.omega is None else np.asarray(spec.omega, dtype=np.float64).tolist(),
        "center": None if spec.center is None else np.asarray(spec.center, dtype=np.float64).tolist(),
        "name": spec.name,
    }


if __name__ == "__main__":
    main()
