"""Research diagnostics for PP disk/cross transient experiments.

This module intentionally keeps transient experiment machinery out of
``pp_cs_equilibria.py``.  It reuses the production PP simulator components but
adds direct-pairwise validation, optional center/clip suppression, diagnostic
time series, sweep outputs, and toy models for the near-alpha-one artifact
study.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .pp_cs_equilibria import (
    DEFAULT_SHAPES,
    Array,
    FFTPeszekPoyato2D,
    InitialCondition,
    InitializerConfig,
    IntegratorChoice,
    SimulationConfig,
    SimulationResult,
    TorchPeszekPoyato2D,
    _adaptive_step_factor,
    _apply_initialization_algorithm,
    _cfl_limited_dt,
    _clamp_dt,
    _config_to_json,
    _dt_history_summary,
    _jsonable,
    _make_pp_backend,
    _projective_external_numpy,
    _seed_from_args,
    _trajectory_frame_limit,
    _trajectory_stride,
    _validate_runtime_config,
    cic_indices_weights,
    deposit_mass,
    direct_hessian_at,
    fiber_colors,
    interp_grid_with_weights,
    make_initial_condition,
    validate_initial_condition,
    write_time_diagnostics,
)


ForceBackendChoice = Literal["fft", "direct"]

RESEARCH_DASHBOARD_FILENAME = "pp_transient_research_diagnostics.html"
FINAL_MORPHOLOGY_FILENAME = "final_morphology.html"

RESEARCH_DIAGNOSTIC_FIELDS = (
    "step",
    "time",
    "dt",
    "r_min",
    "r_nn_q01",
    "r_nn_q05",
    "r_nn_median",
    "lambda_max_max",
    "lambda_max_p99",
    "lambda_min_min",
    "trace_max",
    "trace_mean",
    "anisotropy_p99",
    "q_max",
    "q_p99",
    "rk2_shell_radius",
    "R_cont_rms",
    "R_cont_max",
    "R_disc_rms",
    "R_disc_max",
    "energy",
    "delta_energy",
    "disk_radius",
    "disk_width",
    "boundary_mass_fraction",
    "interior_uniformity_score",
    "axis_mass_fraction",
    "hyperbola_score",
    "fourfold_mode",
    "covariance_anisotropy",
    "clip_events",
)


@dataclass(frozen=True)
class TransientResearchConfig(SimulationConfig):
    """Simulation configuration with transient-research extensions."""

    force_backend: ForceBackendChoice = "fft"
    center_each_step: bool = True
    clip_each_step: bool = True
    record_research_diagnostics: bool = False
    research_diagnostics_every: int | None = None
    research_diagnostic_sample_size: int = 2500
    research_energy_sample_size: int = 1200
    research_nn_chunk: int = 512


@dataclass(frozen=True)
class ResearchSimulationResult(SimulationResult):
    """PP simulation result carrying the research diagnostics matrix."""

    research_diagnostics: Array | None


def _validate_research_config(config: TransientResearchConfig) -> None:
    if config.force_backend not in ("fft", "direct"):
        raise ValueError("force_backend must be one of 'fft' or 'direct'")
    if config.research_diagnostics_every is not None and config.research_diagnostics_every <= 0:
        raise ValueError("research_diagnostics_every must be positive when set")
    if config.research_diagnostic_sample_size <= 0:
        raise ValueError("research_diagnostic_sample_size must be positive")
    if config.research_energy_sample_size <= 0:
        raise ValueError("research_energy_sample_size must be positive")
    if config.research_nn_chunk <= 0:
        raise ValueError("research_nn_chunk must be positive")


def direct_A_at(
    points: Array,
    sources: Array,
    alpha: float,
    K: float,
    chunk: int = 128,
) -> Array:
    """Direct finite-particle PP interaction field at query points."""

    points = np.asarray(points, dtype=np.float64)
    sources = np.asarray(sources, dtype=np.float64)
    out = np.zeros((points.shape[0], 2), dtype=np.float64)
    N = sources.shape[0]
    if N == 0:
        raise ValueError("direct PP field requires at least one source particle")
    for a in range(0, points.shape[0], max(1, int(chunk))):
        p = points[a : a + max(1, int(chunk))]
        diff = p[:, None, :] - sources[None, :, :]
        r = np.linalg.norm(diff, axis=-1)
        mask = r > 1e-14
        scale = np.zeros_like(r)
        scale[mask] = float(K) * (r[mask] ** (-float(alpha))) / (1.0 - float(alpha)) / N
        out[a : a + max(1, int(chunk))] = np.sum(diff * scale[..., None], axis=1)
    return out


class DirectPeszekPoyato2D:
    """Direct O(N^2) PP evaluator for small-N integrator/backend validation."""

    def __init__(self, alpha: float, K: float, grid_size: int, domain_radius: float, *, chunk: int = 128):
        if not 0.0 <= alpha <= 2.0:
            raise ValueError("alpha must lie in [0, 2] for the PP kernel normalization")
        if abs(alpha - 1.0) < 1e-9:
            raise ValueError("alpha = 1 is the singular PP point (1/(1-alpha) diverges); choose alpha != 1")
        if grid_size < 4:
            raise ValueError("grid_size must be at least 4")
        if domain_radius <= 0:
            raise ValueError("domain_radius must be positive")
        if chunk <= 0:
            raise ValueError("chunk must be positive")

        self.alpha = float(alpha)
        self.K = float(K)
        self.G = int(grid_size)
        self.L = float(domain_radius)
        self.h = 2 * self.L / self.G
        self.chunk = int(chunk)
        self.backend_name = "direct"
        self.device_name = "cpu"
        self.dtype_name = "float64"

    def clip_inside(self, x: Array) -> Array:
        margin = 2.1 * self.h
        return np.clip(x, -self.L + margin, self.L - margin)

    def clip_inside_with_count(self, x: Array) -> tuple[Array, int]:
        margin = 2.1 * self.h
        lo = -self.L + margin
        hi = self.L - margin
        clipped = np.any((x < lo) | (x > hi), axis=1)
        return np.clip(x, lo, hi), int(np.count_nonzero(clipped))

    def asarray(self, x: Array) -> Array:
        return np.asarray(x, dtype=np.float64)

    def copy_state(self, x: Array) -> Array:
        return np.array(x, dtype=np.float64, copy=True)

    def to_numpy(self, x: Any) -> Array:
        return np.asarray(x, dtype=np.float64)

    def center(self, x: Array) -> Array:
        return x - x.mean(axis=0, keepdims=True)

    def speed_stats(self, v: Array) -> tuple[float, float]:
        speed2 = np.sum(v * v, axis=1)
        return float(np.sqrt(np.mean(speed2))), float(np.sqrt(np.max(speed2)))

    def rms_delta(self, a: Array, b: Array) -> float:
        delta = a - b
        return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))

    def synchronize(self) -> None:
        return None

    def A_grid_from_particles(self, x: Array) -> tuple[Array, Array, Array]:
        rho_grid = deposit_mass(np.asarray(x, dtype=np.float64), self.G, self.L)
        return rho_grid, np.zeros_like(rho_grid), np.zeros_like(rho_grid)

    def A_at_particles(self, x: Array) -> tuple[Array, Array]:
        x_np = np.asarray(x, dtype=np.float64)
        return direct_A_at(x_np, x_np, self.alpha, self.K, chunk=self.chunk), deposit_mass(x_np, self.G, self.L)

    def velocity(self, x: Array, omega: Array) -> tuple[Array, Array]:
        A, _ = self.A_at_particles(x)
        return omega - A, A

    def projective_external(self, x: Array, omega: Array, eps: float) -> Array:
        return _projective_external_numpy(np.asarray(x, dtype=np.float64), np.asarray(omega, dtype=np.float64), float(eps))

    def hessian_at_particles(self, x: Array) -> Array:
        x_np = np.asarray(x, dtype=np.float64)
        return direct_hessian_at(x_np, x_np, self.alpha, self.K, chunk=self.chunk)


def _make_research_backend(config: TransientResearchConfig) -> FFTPeszekPoyato2D | TorchPeszekPoyato2D | DirectPeszekPoyato2D:
    if config.force_backend == "direct":
        if config.backend != "numpy":
            warnings.warn("force_backend='direct' uses the NumPy CPU path; backend/device/dtype are ignored.", RuntimeWarning, stacklevel=2)
        return DirectPeszekPoyato2D(
            config.alpha,
            config.K,
            config.grid_size,
            config.domain_radius,
            chunk=config.direct_hessian_chunk,
        )
    return _make_pp_backend(config)


def run_research_simulation(
    config: TransientResearchConfig,
    initial: InitialCondition | None = None,
    *,
    cancel_check: Callable[[], bool] | None = None,
) -> ResearchSimulationResult:
    """Run the PP particle simulation with transient-research extensions enabled."""

    _validate_runtime_config(config)
    _validate_research_config(config)

    def _raise_if_cancelled() -> None:
        if cancel_check is not None and cancel_check():
            raise InterruptedError("PP simulation cancelled.")

    raw_initial = make_initial_condition(config) if initial is None else validate_initial_condition(initial)
    initial, initialization_meta = _apply_initialization_algorithm(config, raw_initial)
    solver = _make_research_backend(config)

    def maybe_clip_state(state_x: Any) -> tuple[Any, int]:
        if not config.clip_each_step:
            return state_x, 0
        return solver.clip_inside_with_count(state_x)

    def maybe_center_state(state_x: Any) -> Any:
        if not config.center_each_step:
            return state_x
        return solver.center(state_x)

    x = solver.asarray(initial.x)
    omega = solver.asarray(initial.omega)
    x, clip_count = maybe_clip_state(solver.copy_state(x))
    x_initial = solver.to_numpy(x).copy()
    diagnostics: list[tuple[int, float, float, float, float, int, int, int, int]] = []
    research_diagnostics: list[Array] = []
    trajectory_x: list[Array] = []
    trajectory_rho: list[Array] = []
    trajectory_steps: list[int] = []
    trajectory_times: list[float] = []
    trajectory_stride = _trajectory_stride(config)
    trajectory_limit = _trajectory_frame_limit(config)
    field_evaluations = 0
    accepted_steps = 0
    rejected_steps = 0
    clip_events = int(clip_count)
    dt_current = _clamp_dt(float(config.dt), config)
    dt_history: list[float] = []
    t = 0.0
    time_sign = -1.0 if config.time_direction == "backward" else 1.0
    research_every = int(config.record_every if config.research_diagnostics_every is None else config.research_diagnostics_every)
    last_research_energy: float | None = None

    proj_eps = float(config.projective_epsilon)
    use_projective = config.external_field == "projective" and proj_eps != 0.0

    def velocity_fn(state_x: Any) -> tuple[Any, Any]:
        if not use_projective:
            return solver.velocity(state_x, omega)
        A_loc, _ = solver.A_at_particles(state_x)
        drift = solver.projective_external(state_x, omega, proj_eps)
        return drift - A_loc, A_loc

    def record_trajectory(step_: int, x_: Any, time_: float, *, force: bool = False) -> None:
        if not config.make_animation:
            return
        if trajectory_steps and not force and step_ % trajectory_stride != 0 and step_ != config.max_steps:
            return
        if trajectory_limit is not None and len(trajectory_steps) >= trajectory_limit and not force and step_ != config.max_steps:
            return
        density_x = solver.to_numpy(x_)
        density_grid_size = config.animation_density_grid_size
        if density_grid_size is None or int(density_grid_size) == solver.G:
            rho = deposit_mass(density_x, solver.G, solver.L)
        else:
            rho = deposit_mass(density_x, int(density_grid_size), solver.L)
        trajectory_x.append(density_x.astype(np.float32, copy=True))
        trajectory_rho.append(rho.astype(np.float32, copy=False))
        trajectory_steps.append(int(step_))
        trajectory_times.append(float(time_))

    def append_diagnostic(step_: int, time_: float, rms_: float, maxv_: float, dt_: float) -> None:
        diagnostics.append(
            (
                int(step_),
                float(time_),
                float(rms_),
                float(maxv_),
                float(dt_),
                int(field_evaluations),
                int(accepted_steps),
                int(rejected_steps),
                int(clip_events),
            )
        )

    def append_research_diagnostic(step_: int, time_: float, dt_: float, x_: Any) -> None:
        nonlocal last_research_energy
        if not config.record_research_diagnostics:
            return
        row, last_research_energy = _research_diagnostic_row(
            config,
            solver.to_numpy(x_),
            initial.omega,
            step=step_,
            time_value=time_,
            dt=dt_,
            clip_events=clip_events,
            previous_energy=last_research_energy,
        )
        research_diagnostics.append(row)

    start = time.time()

    if config.integrator == "fixed_rk2":
        dt_fixed = float(config.dt)
        for _ in range(config.max_steps + 1):
            _raise_if_cancelled()
            record_trajectory(accepted_steps, x, t)
            vf, _ = velocity_fn(x)
            field_evaluations += 1
            rms, maxv = solver.speed_stats(vf)
            if accepted_steps % config.record_every == 0:
                append_diagnostic(accepted_steps, t, rms, maxv, dt_fixed)
            if accepted_steps % research_every == 0:
                append_research_diagnostic(accepted_steps, t, dt_fixed, x)
            if rms < config.tol_rms and accepted_steps > config.min_steps:
                break
            if accepted_steps >= config.max_steps:
                break
            x_pred, clipped = maybe_clip_state(x + time_sign * dt_fixed * vf)
            clip_events += clipped
            k2, _ = velocity_fn(x_pred)
            field_evaluations += 1
            x = x + 0.5 * time_sign * dt_fixed * (vf + k2)
            x = maybe_center_state(x)
            x, clipped = maybe_clip_state(x)
            clip_events += clipped
            t += dt_fixed
            accepted_steps += 1
            dt_history.append(dt_fixed)
    else:
        while True:
            _raise_if_cancelled()
            record_trajectory(accepted_steps, x, t)
            vf, _ = velocity_fn(x)
            field_evaluations += 1
            rms, maxv = solver.speed_stats(vf)
            if accepted_steps % config.record_every == 0:
                append_diagnostic(accepted_steps, t, rms, maxv, dt_current)
            if accepted_steps % research_every == 0:
                append_research_diagnostic(accepted_steps, t, dt_current, x)
            if rms < config.tol_rms and accepted_steps > config.min_steps:
                break
            if accepted_steps >= config.max_steps:
                break

            trial_dt = _cfl_limited_dt(dt_current, maxv, solver.h, config)
            while True:
                x_pred, clipped = maybe_clip_state(x + time_sign * trial_dt * vf)
                clip_events += clipped
                k2, _ = velocity_fn(x_pred)
                field_evaluations += 1
                x_euler = x + time_sign * trial_dt * vf
                x_heun = x + 0.5 * time_sign * trial_dt * (vf + k2)
                local_err = solver.rms_delta(x_heun, x_euler)
                if not np.isfinite(local_err):
                    if trial_dt <= config.dt_min * (1 + 1e-12):
                        raise FloatingPointError("adaptive RK2 local error is non-finite at dt_min")
                    rejected_steps += 1
                    trial_dt = max(float(config.dt_min), trial_dt * _adaptive_step_factor(local_err, config.adaptive_tol, grow=False))
                    continue
                if local_err <= config.adaptive_tol or trial_dt <= config.dt_min * (1 + 1e-12):
                    break
                rejected_steps += 1
                trial_dt = max(float(config.dt_min), trial_dt * _adaptive_step_factor(local_err, config.adaptive_tol, grow=False))

            x = maybe_center_state(x_heun)
            x, clipped = maybe_clip_state(x)
            clip_events += clipped
            t += trial_dt
            accepted_steps += 1
            dt_history.append(trial_dt)
            dt_current = _clamp_dt(trial_dt * _adaptive_step_factor(local_err, config.adaptive_tol, grow=True), config)

    runtime = time.time() - start
    if config.make_animation and (not trajectory_steps or trajectory_steps[-1] != accepted_steps):
        record_trajectory(accepted_steps, x, t, force=True)
    A_final_backend, rho_grid_backend = solver.A_at_particles(x)
    field_evaluations += 1
    solver.synchronize()
    x_final = solver.to_numpy(x).copy()
    A_final = solver.to_numpy(A_final_backend)
    rho_grid = solver.to_numpy(rho_grid_backend)
    if use_projective:
        drift_final = solver.to_numpy(solver.projective_external(x, omega, proj_eps))
        residual = drift_final - A_final
    else:
        residual = initial.omega - A_final
    residual_speed2 = np.sum(residual * residual, axis=1)
    dt_min_observed, dt_max_observed, dt_mean = _dt_history_summary(dt_history)
    return ResearchSimulationResult(
        initial=initial,
        x_initial=x_initial,
        x_final=x_final,
        A_final=A_final,
        residual=residual,
        rho_grid=rho_grid,
        diagnostics=np.array(diagnostics, dtype=np.float64),
        trajectory_x=np.stack(trajectory_x) if trajectory_x else None,
        trajectory_rho=np.stack(trajectory_rho) if trajectory_rho else None,
        trajectory_steps=np.array(trajectory_steps, dtype=np.int64) if trajectory_steps else None,
        trajectory_times=np.array(trajectory_times, dtype=np.float64) if trajectory_times else None,
        steps=int(accepted_steps),
        final_time=float(t),
        runtime_seconds=float(runtime),
        rms_residual=float(np.sqrt(np.mean(residual_speed2))),
        max_residual=float(np.sqrt(np.max(residual_speed2))),
        backend=solver.backend_name,
        device=solver.device_name,
        dtype=solver.dtype_name,
        field_evaluations=int(field_evaluations),
        accepted_steps=int(accepted_steps),
        rejected_steps=int(rejected_steps),
        dt_min_observed=dt_min_observed,
        dt_max_observed=dt_max_observed,
        dt_mean=dt_mean,
        clip_events=int(clip_events),
        initialization_algorithm=str(initialization_meta["algorithm"]),
        initialization_steps=int(initialization_meta["steps"]),
        initialization_time=float(initialization_meta["time"]),
        initialization_stop_metric=float(initialization_meta["stop_metric"]),
        research_diagnostics=(
            np.stack(research_diagnostics).astype(np.float64, copy=False)
            if research_diagnostics
            else (np.zeros((0, len(RESEARCH_DIAGNOSTIC_FIELDS)), dtype=np.float64) if config.record_research_diagnostics else None)
        ),
    )


def _diagnostic_sample_indices(n: int, max_size: int, seed: int) -> Array:
    if n <= 0:
        return np.empty((0,), dtype=np.int64)
    max_size = int(max_size)
    if max_size <= 0 or n <= max_size:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_size, replace=False))


def nearest_neighbor_distances(points: Array, *, chunk: int = 512) -> Array:
    """Nearest-neighbor distances inside a point cloud, computed in chunks."""

    x = np.asarray(points, dtype=np.float64)
    n = int(x.shape[0])
    if n < 2:
        return np.full((n,), np.inf, dtype=np.float64)
    chunk = max(1, int(chunk))
    nn2 = np.full((n,), np.inf, dtype=np.float64)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        diff = x[a:b, None, :] - x[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        rows = np.arange(b - a)
        d2[rows, a + rows] = np.inf
        nn2[a:b] = np.min(d2, axis=1)
    return np.sqrt(np.maximum(nn2, 0.0))


def nearest_neighbor_quantiles(points: Array, *, chunk: int = 512) -> dict[str, float]:
    nn = nearest_neighbor_distances(points, chunk=chunk)
    finite = nn[np.isfinite(nn)]
    if len(finite) == 0:
        return {"r_min": float("inf"), "r_nn_q01": float("inf"), "r_nn_q05": float("inf"), "r_nn_median": float("inf")}
    return {
        "r_min": float(np.min(finite)),
        "r_nn_q01": float(np.quantile(finite, 0.01)),
        "r_nn_q05": float(np.quantile(finite, 0.05)),
        "r_nn_median": float(np.median(finite)),
    }


def _hessian_components_to_stats(H: Array) -> dict[str, Array]:
    H = np.asarray(H, dtype=np.float64)
    if H.ndim != 3 or H.shape[1:] != (2, 2):
        raise ValueError("H must have shape (N, 2, 2)")
    hxx = H[:, 0, 0]
    hxy = 0.5 * (H[:, 0, 1] + H[:, 1, 0])
    hyy = H[:, 1, 1]
    trace = hxx + hyy
    disc = np.sqrt(np.maximum((hxx - hyy) ** 2 + 4.0 * hxy * hxy, 0.0))
    lam_min = 0.5 * (trace - disc)
    lam_max = 0.5 * (trace + disc)
    det = hxx * hyy - hxy * hxy
    anisotropy = lam_max / np.maximum(lam_min, 1e-30)
    return {
        "lambda_min": lam_min,
        "lambda_max": lam_max,
        "trace": trace,
        "det": det,
        "anisotropy": anisotropy,
    }


def evaluate_A_and_H_at_particles(
    config: TransientResearchConfig,
    x: Array,
    *,
    query: Array | None = None,
    force_backend: ForceBackendChoice | None = None,
) -> dict[str, Array]:
    """Evaluate PP force and Hessian at particles or supplied query points."""

    sources = np.asarray(x, dtype=np.float64)
    points = sources if query is None else np.asarray(query, dtype=np.float64)
    backend = config.force_backend if force_backend is None else force_backend
    if backend == "direct":
        A = direct_A_at(points, sources, config.alpha, config.K, chunk=config.direct_hessian_chunk)
        H = direct_hessian_at(points, sources, config.alpha, config.K, chunk=config.direct_hessian_chunk)
        rho = deposit_mass(sources, config.grid_size, config.domain_radius)
    elif backend == "fft":
        solver = FFTPeszekPoyato2D(config.alpha, config.K, config.grid_size, config.domain_radius)
        rho, Ax_grid, Ay_grid = solver.A_grid_from_particles(sources)
        weights = cic_indices_weights(points, solver.G, solver.L)
        A = np.c_[interp_grid_with_weights(Ax_grid, weights), interp_grid_with_weights(Ay_grid, weights)]
        Hxx, Hxy, Hyy = solver.hessian_grid_from_rho(rho)
        H = np.empty((len(points), 2, 2), dtype=np.float64)
        H[:, 0, 0] = interp_grid_with_weights(Hxx, weights)
        H[:, 0, 1] = interp_grid_with_weights(Hxy, weights)
        H[:, 1, 0] = H[:, 0, 1]
        H[:, 1, 1] = interp_grid_with_weights(Hyy, weights)
    else:
        raise ValueError(f"unknown force_backend: {backend!r}")
    stats = _hessian_components_to_stats(H)
    return {"A": A, "H": H, "rho_grid": rho, **stats}


def evaluate_A_at_particles(
    config: TransientResearchConfig,
    x: Array,
    *,
    query: Array | None = None,
    force_backend: ForceBackendChoice | None = None,
) -> Array:
    sources = np.asarray(x, dtype=np.float64)
    points = sources if query is None else np.asarray(query, dtype=np.float64)
    backend = config.force_backend if force_backend is None else force_backend
    if backend == "direct":
        return direct_A_at(points, sources, config.alpha, config.K, chunk=config.direct_hessian_chunk)
    if backend == "fft":
        solver = FFTPeszekPoyato2D(config.alpha, config.K, config.grid_size, config.domain_radius)
        rho, Ax_grid, Ay_grid = solver.A_grid_from_particles(sources)
        weights = cic_indices_weights(points, solver.G, solver.L)
        return np.c_[interp_grid_with_weights(Ax_grid, weights), interp_grid_with_weights(Ay_grid, weights)]
    raise ValueError(f"unknown force_backend: {backend!r}")


def evaluate_velocity_numpy(
    config: TransientResearchConfig,
    x: Array,
    omega: Array,
    *,
    force_backend: ForceBackendChoice | None = None,
) -> tuple[Array, Array]:
    A = evaluate_A_at_particles(config, x, force_backend=force_backend)
    if config.external_field == "projective" and float(config.projective_epsilon) != 0.0:
        drift = _projective_external_numpy(x, omega, float(config.projective_epsilon))
    else:
        drift = np.asarray(omega, dtype=np.float64)
    return drift - A, A


def _clip_numpy_with_count(x: Array, config: SimulationConfig) -> tuple[Array, int]:
    h = 2.0 * float(config.domain_radius) / int(config.grid_size)
    margin = 2.1 * h
    lo = -float(config.domain_radius) + margin
    hi = float(config.domain_radius) - margin
    clipped = np.any((x < lo) | (x > hi), axis=1)
    return np.clip(x, lo, hi), int(np.count_nonzero(clipped))


def rk2_step_numpy(
    config: TransientResearchConfig,
    x: Array,
    omega: Array,
    dt: float | None = None,
    *,
    force_backend: ForceBackendChoice | None = None,
    center_each_step: bool | None = None,
    clip_each_step: bool | None = None,
) -> Array:
    """One fixed midpoint/Heun RK2 map with the research centering/clipping convention."""

    h = float(config.dt if dt is None else dt)
    time_sign = -1.0 if config.time_direction == "backward" else 1.0
    center = bool(config.center_each_step if center_each_step is None else center_each_step)
    clip = bool(config.clip_each_step if clip_each_step is None else clip_each_step)
    x0 = np.asarray(x, dtype=np.float64)
    omega0 = np.asarray(omega, dtype=np.float64)
    k1, _ = evaluate_velocity_numpy(config, x0, omega0, force_backend=force_backend)
    x_pred = x0 + time_sign * h * k1
    if clip:
        x_pred, _ = _clip_numpy_with_count(x_pred, config)
    k2, _ = evaluate_velocity_numpy(config, x_pred, omega0, force_backend=force_backend)
    out = x0 + 0.5 * time_sign * h * (k1 + k2)
    if center:
        out = out - out.mean(axis=0, keepdims=True)
    if clip:
        out, _ = _clip_numpy_with_count(out, config)
    return out


def rk2_map_residual(
    config: TransientResearchConfig,
    x: Array,
    omega: Array,
    dt: float | None = None,
    *,
    force_backend: ForceBackendChoice | None = None,
) -> Array:
    return rk2_step_numpy(config, x, omega, dt, force_backend=force_backend) - np.asarray(x, dtype=np.float64)


def pp_particle_energy(
    x: Array,
    omega: Array,
    alpha: float,
    K: float,
    *,
    sample_size: int = 1200,
    seed: int = 0,
    chunk: int = 256,
) -> float:
    """Direct particle PP energy, sampled deterministically for large clouds."""

    x_full = np.asarray(x, dtype=np.float64)
    omega_full = np.asarray(omega, dtype=np.float64)
    idx = _diagnostic_sample_indices(len(x_full), int(sample_size), seed)
    xs = x_full[idx]
    os = omega_full[idx]
    m = int(len(xs))
    if m == 0:
        return float("nan")
    linear = -float(np.mean(np.sum(os * xs, axis=1)))
    pair_sum = 0.0
    denom = (2.0 - float(alpha)) * (1.0 - float(alpha))
    for a in range(0, m, max(1, int(chunk))):
        p = xs[a : a + max(1, int(chunk))]
        r = np.linalg.norm(p[:, None, :] - xs[None, :, :], axis=2)
        mask = r > 1e-14
        w = np.zeros_like(r)
        w[mask] = float(K) * (r[mask] ** (2.0 - float(alpha))) / denom
        pair_sum += float(np.sum(w))
    return linear + 0.5 * pair_sum / float(m * m)


def compute_morphology_metrics(x: Array, *, grid_dx: float | None = None) -> dict[str, float]:
    """Disk/cross/fourfold morphology scores for a particle cloud."""

    pts = np.asarray(x, dtype=np.float64)
    n = int(len(pts))
    if n == 0:
        return {
            "disk_radius": 0.0,
            "disk_width": 0.0,
            "boundary_mass_fraction": 0.0,
            "interior_uniformity_score": 0.0,
            "axis_mass_fraction": 0.0,
            "hyperbola_score": 0.0,
            "fourfold_mode": 0.0,
            "covariance_anisotropy": 0.0,
        }

    centered = pts - pts.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(centered, axis=1)
    cloud_radius = float(np.quantile(radius, 0.98)) if n > 1 else float(radius[0])
    max_r = max(float(np.max(radius)), 1e-12)
    bins = max(12, min(96, int(np.sqrt(n))))
    counts, edges = np.histogram(radius, bins=bins, range=(0.0, max_r))
    widths = np.maximum(np.diff(edges), 1e-12)
    density = counts / widths
    peak_idx = int(np.argmax(density)) if len(density) else 0
    disk_radius = float(0.5 * (edges[peak_idx] + edges[peak_idx + 1]))
    dist_to_peak = np.abs(radius - disk_radius)
    disk_width = float(2.0 * np.quantile(dist_to_peak, 0.5))
    dx = 0.0 if grid_dx is None else float(grid_dx)
    bandwidth = max(0.05 * max(disk_radius, max_r), 2.0 * dx, 1e-12)
    boundary_mass_fraction = float(np.mean(dist_to_peak < bandwidth))

    interior = radius <= max(disk_radius, 1e-12)
    if np.count_nonzero(interior) >= 4 and disk_radius > 0:
        scaled = np.sort((radius[interior] / disk_radius) ** 2)
        empirical = (np.arange(len(scaled)) + 0.5) / len(scaled)
        interior_rmse = float(np.sqrt(np.mean((scaled - empirical) ** 2)))
        interior_uniformity_score = float(1.0 / (interior_rmse + 1e-12))
    else:
        interior_uniformity_score = 0.0

    eps_axis = max(2.0 * dx, 0.05 * max(cloud_radius, max_r), 1e-12)
    axis_distance = np.minimum(np.abs(centered[:, 0]), np.abs(centered[:, 1]))
    axis_mass_fraction = float(np.mean(axis_distance < eps_axis))

    away = (np.abs(centered[:, 0]) > eps_axis) & (np.abs(centered[:, 1]) > eps_axis)
    quadrant_codes = (centered[:, 0] > 0).astype(np.int64) + 2 * (centered[:, 1] > 0).astype(np.int64)
    variances: list[float] = []
    weights: list[int] = []
    s = np.log(np.abs(centered[:, 0] * centered[:, 1]) + 1e-30)
    for q in range(4):
        mask = away & (quadrant_codes == q)
        count = int(np.count_nonzero(mask))
        if count >= 4:
            variances.append(float(np.var(s[mask])))
            weights.append(count)
    hyperbola_score = float(1.0 / (np.average(variances, weights=weights) + 1e-12)) if weights else 0.0

    theta_mask = radius > 1e-14
    if np.any(theta_mask):
        theta = np.arctan2(centered[theta_mask, 1], centered[theta_mask, 0])
        fourfold_mode = float(abs(np.mean(np.exp(4j * theta))))
    else:
        fourfold_mode = 0.0

    if n >= 3:
        cov = np.cov(centered.T)
        eig = np.linalg.eigvalsh(cov)
        covariance_anisotropy = float(eig[-1] / max(eig[0], 1e-30))
    else:
        covariance_anisotropy = 0.0

    return {
        "disk_radius": disk_radius,
        "disk_width": disk_width,
        "boundary_mass_fraction": boundary_mass_fraction,
        "interior_uniformity_score": interior_uniformity_score,
        "axis_mass_fraction": axis_mass_fraction,
        "hyperbola_score": hyperbola_score,
        "fourfold_mode": fourfold_mode,
        "covariance_anisotropy": covariance_anisotropy,
    }


def rk2_shell_radius(alpha: float, K: float, dt: float) -> float:
    return float(abs(dt) * abs(K) / max(2.0 * abs(1.0 - float(alpha)), 1e-30))


def _research_diagnostic_row(
    config: TransientResearchConfig,
    x: Array,
    omega: Array,
    *,
    step: int,
    time_value: float,
    dt: float,
    clip_events: int,
    previous_energy: float | None,
) -> tuple[Array, float]:
    x_np = np.asarray(x, dtype=np.float64)
    omega_np = np.asarray(omega, dtype=np.float64)
    sample_idx = _diagnostic_sample_indices(
        len(x_np),
        int(config.research_diagnostic_sample_size),
        int(config.seed) + 1009 * int(step) + 17,
    )
    x_sample = x_np[sample_idx]
    omega_sample = omega_np[sample_idx]

    nn = nearest_neighbor_quantiles(x_sample, chunk=config.research_nn_chunk)
    evaluated = evaluate_A_and_H_at_particles(config, x_np, query=x_sample, force_backend=config.force_backend)
    hstats = {key: np.asarray(evaluated[key], dtype=np.float64) for key in ("lambda_min", "lambda_max", "trace", "anisotropy")}
    lam_max = hstats["lambda_max"]
    lam_min = hstats["lambda_min"]
    trace = hstats["trace"]
    anisotropy = hstats["anisotropy"]
    lambda_max_max = float(np.max(lam_max)) if len(lam_max) else 0.0
    lambda_max_p99 = float(np.quantile(lam_max, 0.99)) if len(lam_max) else 0.0
    lambda_min_min = float(np.min(lam_min)) if len(lam_min) else 0.0
    trace_max = float(np.max(trace)) if len(trace) else 0.0
    trace_mean = float(np.mean(trace)) if len(trace) else 0.0
    anisotropy_p99 = float(np.quantile(anisotropy[np.isfinite(anisotropy)], 0.99)) if np.any(np.isfinite(anisotropy)) else 0.0

    A_sample = np.asarray(evaluated["A"], dtype=np.float64)
    if config.external_field == "projective" and float(config.projective_epsilon) != 0.0:
        drift_sample = _projective_external_numpy(x_sample, omega_sample, float(config.projective_epsilon))
    else:
        drift_sample = omega_sample
    R_cont = drift_sample - A_sample
    R_cont_norm = np.linalg.norm(R_cont, axis=1) if len(R_cont) else np.zeros((0,), dtype=np.float64)
    R_cont_rms = float(np.sqrt(np.mean(R_cont_norm * R_cont_norm))) if len(R_cont_norm) else 0.0
    R_cont_max = float(np.max(R_cont_norm)) if len(R_cont_norm) else 0.0

    if config.force_backend == "direct" and len(x_np) > int(config.research_diagnostic_sample_size):
        disc_x = x_sample
        disc_omega = omega_sample
    else:
        disc_x = x_np
        disc_omega = omega_np
    R_disc = rk2_map_residual(config, disc_x, disc_omega, dt, force_backend=config.force_backend)
    R_disc_norm = np.linalg.norm(R_disc, axis=1) if len(R_disc) else np.zeros((0,), dtype=np.float64)
    R_disc_rms = float(np.sqrt(np.mean(R_disc_norm * R_disc_norm))) if len(R_disc_norm) else 0.0
    R_disc_max = float(np.max(R_disc_norm)) if len(R_disc_norm) else 0.0

    energy = pp_particle_energy(
        x_np,
        omega_np,
        config.alpha,
        config.K,
        sample_size=config.research_energy_sample_size,
        seed=int(config.seed) + 811 * int(step) + 29,
        chunk=max(64, min(512, int(config.research_nn_chunk))),
    )
    delta_energy = 0.0 if previous_energy is None or not np.isfinite(previous_energy) else float(energy - previous_energy)

    morphology = compute_morphology_metrics(x_np, grid_dx=2.0 * float(config.domain_radius) / int(config.grid_size))
    row_values = {
        "step": float(step),
        "time": float(time_value),
        "dt": float(dt),
        **nn,
        "lambda_max_max": lambda_max_max,
        "lambda_max_p99": lambda_max_p99,
        "lambda_min_min": lambda_min_min,
        "trace_max": trace_max,
        "trace_mean": trace_mean,
        "anisotropy_p99": anisotropy_p99,
        "q_max": float(abs(dt) * lambda_max_max),
        "q_p99": float(abs(dt) * lambda_max_p99),
        "rk2_shell_radius": rk2_shell_radius(config.alpha, config.K, dt),
        "R_cont_rms": R_cont_rms,
        "R_cont_max": R_cont_max,
        "R_disc_rms": R_disc_rms,
        "R_disc_max": R_disc_max,
        "energy": float(energy),
        "delta_energy": delta_energy,
        **morphology,
        "clip_events": float(clip_events),
    }
    return np.array([row_values[name] for name in RESEARCH_DIAGNOSTIC_FIELDS], dtype=np.float64), float(energy)


def _research_diag_column(diagnostics: Array, name: str) -> Array:
    return np.asarray(diagnostics[:, RESEARCH_DIAGNOSTIC_FIELDS.index(name)], dtype=np.float64)


def make_final_morphology_figure(result: ResearchSimulationResult, config: TransientResearchConfig, *, title: str | None = None) -> go.Figure:
    """Final particle scatter with fixed axes and reproducible fiber colors."""

    rng = np.random.default_rng(config.seed + 31)
    group_id = result.initial.group_id
    group_names = result.initial.group_names
    colors = fiber_colors(config, result.initial.omega_atoms, len(group_names))
    fig = go.Figure()
    for k, name in enumerate(group_names):
        idx_all = np.where(group_id == k)[0]
        if len(idx_all) == 0:
            continue
        idx = rng.choice(idx_all, size=min(config.max_plot_points_per_group, len(idx_all)), replace=False)
        fig.add_trace(
            go.Scattergl(
                x=result.x_final[idx, 0],
                y=result.x_final[idx, 1],
                mode="markers",
                marker=dict(size=4, color=colors[k], opacity=0.72),
                name=f"fiber {k + 1}: {name}",
            )
        )
    fig.update_xaxes(range=[-config.domain_radius, config.domain_radius], title_text="x1", scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-config.domain_radius, config.domain_radius], title_text="x2")
    fig.update_layout(
        title=dict(
            text=title
            or (
                f"Final morphology: {config.integrator}, force={config.force_backend}, "
                f"alpha={config.alpha:g}, K={config.K:g}, h={config.dt:g}, grid={config.grid_size}, "
                f"steps={result.steps}, t={result.final_time:.4g}, seed={config.seed}"
            ),
            x=0.5,
        ),
        width=820,
        height=760,
        template="plotly_white",
        legend=dict(itemsizing="constant"),
    )
    return fig


def make_research_diagnostics_dashboard(result: ResearchSimulationResult, config: TransientResearchConfig) -> go.Figure:
    """Plot diagnostics used to separate RK2, grid, and continuous-time effects."""

    if result.research_diagnostics is None or len(result.research_diagnostics) == 0:
        raise ValueError("run_research_simulation must be called with record_research_diagnostics=True")
    diag = np.asarray(result.research_diagnostics, dtype=np.float64)
    t = _research_diag_column(diag, "time")
    fig = make_subplots(
        rows=4,
        cols=2,
        specs=[
            [{"type": "scattergl"}, {"type": "histogram"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
        subplot_titles=[
            "Final morphology",
            "Final radius histogram",
            "Nearest-neighbor distance floor",
            "Hessian precision and q = dt Lambda",
            "Continuous vs RK2 map residual",
            "Disk/cross morphology scores",
            "PP energy increments",
            "Observed vs predicted shell scale",
        ],
        horizontal_spacing=0.10,
        vertical_spacing=0.095,
    )

    morphology = make_final_morphology_figure(result, config)
    for trace in morphology.data:
        fig.add_trace(trace, row=1, col=1)

    radius = np.linalg.norm(result.x_final - result.x_final.mean(axis=0, keepdims=True), axis=1)
    fig.add_trace(go.Histogram(x=radius, nbinsx=60, name="radius", showlegend=False), row=1, col=2)
    final_shell = float(_research_diag_column(diag, "rk2_shell_radius")[-1])
    fig.add_vline(x=final_shell, line_dash="dash", line_color="#d62728", row=1, col=2)

    for name, label in (
        ("r_min", "r_min"),
        ("r_nn_q01", "NN q01"),
        ("r_nn_q05", "NN q05"),
        ("r_nn_median", "NN median"),
    ):
        fig.add_trace(go.Scatter(x=t, y=_research_diag_column(diag, name), mode="lines+markers", name=label), row=2, col=1)

    for name, label in (
        ("lambda_max_max", "Lambda max"),
        ("lambda_max_p99", "Lambda p99"),
        ("q_max", "q max"),
        ("q_p99", "q p99"),
    ):
        fig.add_trace(go.Scatter(x=t, y=_research_diag_column(diag, name), mode="lines+markers", name=label), row=2, col=2)
    fig.add_hline(y=1.0, line_dash="dot", line_color="#555", row=2, col=2)
    fig.add_hline(y=2.0, line_dash="dot", line_color="#999", row=2, col=2)

    for name, label in (
        ("R_cont_rms", "continuous RMS"),
        ("R_cont_max", "continuous max"),
        ("R_disc_rms", "RK2 map RMS"),
        ("R_disc_max", "RK2 map max"),
    ):
        fig.add_trace(go.Scatter(x=t, y=_research_diag_column(diag, name), mode="lines+markers", name=label), row=3, col=1)

    for name, label in (
        ("disk_radius", "disk radius"),
        ("boundary_mass_fraction", "boundary mass"),
        ("axis_mass_fraction", "axis mass"),
        ("fourfold_mode", "fourfold mode"),
        ("hyperbola_score", "hyperbola score"),
    ):
        fig.add_trace(go.Scatter(x=t, y=_research_diag_column(diag, name), mode="lines+markers", name=label), row=3, col=2)

    fig.add_trace(go.Scatter(x=t, y=_research_diag_column(diag, "energy"), mode="lines+markers", name="energy"), row=4, col=1)
    fig.add_trace(go.Bar(x=t, y=_research_diag_column(diag, "delta_energy"), name="Delta energy"), row=4, col=1)

    fig.add_trace(go.Scatter(x=t, y=_research_diag_column(diag, "disk_radius"), mode="lines+markers", name="observed disk radius"), row=4, col=2)
    fig.add_trace(go.Scatter(x=t, y=_research_diag_column(diag, "rk2_shell_radius"), mode="lines", name="hK/[2(1-alpha)]", line=dict(dash="dash")), row=4, col=2)
    fig.add_trace(go.Scatter(x=t, y=_research_diag_column(diag, "r_min"), mode="lines", name="r_min", line=dict(dash="dot")), row=4, col=2)

    fig.update_xaxes(title_text="x1", range=[-config.domain_radius, config.domain_radius], scaleanchor="y", scaleratio=1, row=1, col=1)
    fig.update_yaxes(title_text="x2", range=[-config.domain_radius, config.domain_radius], row=1, col=1)
    fig.update_xaxes(title_text="radius", row=1, col=2)
    for row in (2, 3, 4):
        fig.update_xaxes(title_text="time", row=row, col=1)
        fig.update_xaxes(title_text="time", row=row, col=2)
    fig.update_yaxes(type="log", row=2, col=1)
    fig.update_yaxes(type="log", row=2, col=2)
    fig.update_yaxes(type="log", row=3, col=1)
    fig.update_yaxes(type="log", row=4, col=2)
    fig.update_layout(
        title=dict(
            text=(
                "PP transient artifact diagnostics<br>"
                f"<sup>integrator={config.integrator}, force={config.force_backend}, alpha={config.alpha:g}, "
                f"K={config.K:g}, h={config.dt:g}, grid={config.grid_size}, seed={config.seed}, "
                f"center={config.center_each_step}, clip={config.clip_each_step}</sup>"
            ),
            x=0.5,
        ),
        width=1450,
        height=1550,
        template="plotly_white",
        legend=dict(groupclick="togglegroup", itemsizing="constant"),
    )
    return fig


def write_research_diagnostics(path: Path, diagnostics: Array) -> None:
    with path.open("w") as f:
        f.write(",".join(RESEARCH_DIAGNOSTIC_FIELDS) + "\n")
        for row in diagnostics:
            f.write(",".join(map(str, row)) + "\n")


def summarize_research_run(result: ResearchSimulationResult, config: TransientResearchConfig) -> dict[str, object]:
    """Compact JSON-friendly summary for sweep tables."""

    dx = float(2.0 * config.domain_radius / config.grid_size)
    summary: dict[str, object] = {
        "seed": int(config.seed),
        "alpha": float(config.alpha),
        "K": float(config.K),
        "delta": float(1.0 - config.alpha),
        "inv_delta": float(1.0 / max(abs(1.0 - config.alpha), 1e-30)),
        "dt": float(config.dt),
        "grid_size": int(config.grid_size),
        "dx": dx,
        "domain_radius": float(config.domain_radius),
        "n_particles": int(len(result.x_final)),
        "n_fibers": int(len(result.initial.group_names)),
        "force_backend": config.force_backend,
        "integrator": config.integrator,
        "center_each_step": bool(config.center_each_step),
        "clip_each_step": bool(config.clip_each_step),
        "steps": int(result.steps),
        "final_time": float(result.final_time),
        "runtime_seconds": float(result.runtime_seconds),
        "rms_residual": float(result.rms_residual),
        "max_residual": float(result.max_residual),
        "field_evaluations": int(result.field_evaluations),
        "accepted_steps": int(result.accepted_steps),
        "rejected_steps": int(result.rejected_steps),
        "clip_events": int(result.clip_events),
        "dt_min_observed": float(result.dt_min_observed),
        "dt_max_observed": float(result.dt_max_observed),
        "dt_mean": float(result.dt_mean),
    }
    morphology = compute_morphology_metrics(result.x_final, grid_dx=dx)
    summary.update({f"final_{key}": float(value) for key, value in morphology.items()})
    if result.research_diagnostics is not None and len(result.research_diagnostics):
        diag = np.asarray(result.research_diagnostics, dtype=np.float64)
        last = diag[-1]
        for name, value in zip(RESEARCH_DIAGNOSTIC_FIELDS, last, strict=True):
            summary[f"last_{name}"] = float(value)
        for name in ("q_max", "q_p99", "lambda_max_max", "R_cont_rms", "R_disc_rms", "energy"):
            values = _research_diag_column(diag, name)
            if name == "energy":
                summary[f"min_{name}"] = float(np.nanmin(values))
                summary[f"max_{name}"] = float(np.nanmax(values))
            else:
                summary[f"max_{name}"] = float(np.nanmax(values))
        delta_e = _research_diag_column(diag, "delta_energy")
        summary["energy_increase_fraction"] = float(np.mean(delta_e > 1e-12)) if len(delta_e) else 0.0
    return summary


def save_research_run_outputs(
    result: ResearchSimulationResult,
    config: TransientResearchConfig,
    out_dir: Path | str,
    *,
    initial: InitialCondition | None = None,
) -> dict[str, object]:
    """Save lightweight per-run artifacts for parameter sweeps."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize_research_run(result, config)
    (out / "config.json").write_text(json.dumps(_config_to_json(config), indent=2))
    (out / "metrics.json").write_text(json.dumps(_jsonable(summary), indent=2))
    write_time_diagnostics(out / "time_diagnostics.csv", result.diagnostics)
    if result.research_diagnostics is not None:
        write_research_diagnostics(out / "research_diagnostics.csv", result.research_diagnostics)
    if initial is not None:
        np.savez_compressed(
            out / "initial_condition.npz",
            x=initial.x.astype(np.float32),
            omega=initial.omega.astype(np.float32),
            group_id=initial.group_id.astype(np.int32),
            omega_atoms=initial.omega_atoms.astype(np.float32),
            group_names=np.array(initial.group_names),
        )
    np.savez_compressed(
        out / "run_state.npz",
        x_initial=result.x_initial.astype(np.float32),
        x_final=result.x_final.astype(np.float32),
        omega=result.initial.omega.astype(np.float32),
        group_id=result.initial.group_id.astype(np.int32),
        A_final=result.A_final.astype(np.float32),
        residual=result.residual.astype(np.float32),
        diagnostics=result.diagnostics.astype(np.float64),
        research_diagnostics=(
            np.empty((0, len(RESEARCH_DIAGNOSTIC_FIELDS)), dtype=np.float64)
            if result.research_diagnostics is None
            else result.research_diagnostics.astype(np.float64)
        ),
    )
    make_final_morphology_figure(result, config).write_html(str(out / FINAL_MORPHOLOGY_FILENAME), include_plotlyjs="cdn")
    if result.research_diagnostics is not None and len(result.research_diagnostics):
        make_research_diagnostics_dashboard(result, config).write_html(str(out / RESEARCH_DASHBOARD_FILENAME), include_plotlyjs="cdn")
    return summary


def _default_research_sweep_values(sweep: str) -> tuple[object, ...]:
    if sweep == "timestep":
        return (0.09, 0.055, 0.03, 0.015, 0.0075, 0.00375)
    if sweep == "alpha":
        return (0.95, 0.97, 0.98, 0.99, 0.995)
    if sweep == "k_renormalization":
        return ("1", "delta", "sqrt_delta", "0.5delta", "2delta")
    if sweep == "grid":
        return (128, 256, 512)
    if sweep == "domain":
        return (3.0, 4.0, 6.0, 8.0, 12.0)
    if sweep == "particles":
        return (25, 50, 100, 200, 500)
    if sweep == "integrator_backend":
        return (
            ("fft", "fixed_rk2"),
            ("fft", "adaptive_rk2"),
            ("direct", "fixed_rk2"),
            ("direct", "adaptive_rk2"),
        )
    raise ValueError(f"unknown research sweep {sweep!r}")


def _research_sweep_config(base: TransientResearchConfig, sweep: str, value: object) -> TransientResearchConfig:
    if sweep == "timestep":
        return replace(base, dt=float(value), integrator="fixed_rk2", dt_max=max(float(value), base.dt_max))
    if sweep == "alpha":
        return replace(base, alpha=float(value))
    if sweep == "k_renormalization":
        delta = max(abs(1.0 - float(base.alpha)), 1e-30)
        label = str(value)
        if label == "1":
            K = 1.0
        elif label == "delta":
            K = delta
        elif label == "sqrt_delta":
            K = float(np.sqrt(delta))
        elif label == "0.5delta":
            K = 0.5 * delta
        elif label == "2delta":
            K = 2.0 * delta
        else:
            K = float(value)  # type: ignore[arg-type]
        return replace(base, K=K)
    if sweep == "grid":
        return replace(base, grid_size=int(value))
    if sweep == "domain":
        return replace(base, domain_radius=float(value))
    if sweep == "particles":
        return replace(base, n_per_fiber=int(value))
    if sweep == "integrator_backend":
        force_backend, integrator = value  # type: ignore[misc]
        return replace(base, force_backend=force_backend, integrator=integrator)  # type: ignore[arg-type]
    raise ValueError(f"unknown research sweep {sweep!r}")


def _slugify_sweep_value(value: object) -> str:
    if isinstance(value, tuple):
        return "_".join(_slugify_sweep_value(part) for part in value)
    text = str(value).replace(".", "p").replace("-", "m")
    return "".join(ch if ch.isalnum() or ch in ("_", "p", "m") else "_" for ch in text)


def _same_particle_layout(a: SimulationConfig, b: SimulationConfig) -> bool:
    return (
        a.n_fibers == b.n_fibers
        and a.n_per_fiber == b.n_per_fiber
        and a.fibers == b.fibers
        and tuple(a.shape_names) == tuple(b.shape_names)
        and a.seed == b.seed
    )


def write_sweep_metrics_csv(path: Path, summaries: Sequence[dict[str, object]]) -> None:
    if not summaries:
        path.write_text("")
        return
    keys: list[str] = []
    for summary in summaries:
        for key in summary:
            if key not in keys:
                keys.append(key)
    with path.open("w") as f:
        f.write(",".join(keys) + "\n")
        for summary in summaries:
            f.write(",".join(str(summary.get(key, "")) for key in keys) + "\n")


def make_sweep_summary_figure(summaries: Sequence[dict[str, object]], *, sweep: str) -> go.Figure:
    if not summaries:
        raise ValueError("cannot plot an empty sweep")
    if sweep == "timestep":
        x_key = "dt"
        x_title = "fixed timestep h"
    elif sweep == "alpha":
        x_key = "inv_delta"
        x_title = "1 / |1 - alpha|"
    elif sweep == "grid":
        x_key = "dx"
        x_title = "grid spacing dx"
    elif sweep == "domain":
        x_key = "domain_radius"
        x_title = "domain radius"
    elif sweep == "particles":
        x_key = "n_particles"
        x_title = "particle count"
    else:
        x_key = "case_index"
        x_title = "case"

    x_vals = np.arange(len(summaries), dtype=np.float64) if x_key == "case_index" else np.array([float(s[x_key]) for s in summaries])
    labels = [str(s.get("case_label", i)) for i, s in enumerate(summaries)]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Disk radius and predicted shell",
            "Stiffness q",
            "Continuous/discrete residual",
            "Disk/cross scores",
        ],
        horizontal_spacing=0.10,
        vertical_spacing=0.12,
    )
    disk = np.array([float(s.get("last_disk_radius", s.get("final_disk_radius", np.nan))) for s in summaries])
    shell = np.array([float(s.get("last_rk2_shell_radius", np.nan)) for s in summaries])
    fig.add_trace(go.Scatter(x=x_vals, y=disk, text=labels, mode="markers+lines", name="observed disk radius"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=shell, text=labels, mode="markers+lines", name="predicted RK2 shell", line=dict(dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=[float(s.get("max_q_max", np.nan)) for s in summaries], text=labels, mode="markers+lines", name="max q"), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_vals, y=[float(s.get("max_q_p99", np.nan)) for s in summaries], text=labels, mode="markers+lines", name="max q p99"), row=1, col=2)
    fig.add_hline(y=1.0, line_dash="dot", line_color="#555", row=1, col=2)
    fig.add_hline(y=2.0, line_dash="dot", line_color="#999", row=1, col=2)
    fig.add_trace(go.Scatter(x=x_vals, y=[float(s.get("last_R_cont_rms", np.nan)) for s in summaries], text=labels, mode="markers+lines", name="continuous RMS"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=[float(s.get("last_R_disc_rms", np.nan)) for s in summaries], text=labels, mode="markers+lines", name="RK2 map RMS"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_vals, y=[float(s.get("last_boundary_mass_fraction", np.nan)) for s in summaries], text=labels, mode="markers+lines", name="boundary mass"), row=2, col=2)
    fig.add_trace(go.Scatter(x=x_vals, y=[float(s.get("last_axis_mass_fraction", np.nan)) for s in summaries], text=labels, mode="markers+lines", name="axis mass"), row=2, col=2)
    fig.add_trace(go.Scatter(x=x_vals, y=[float(s.get("last_fourfold_mode", np.nan)) for s in summaries], text=labels, mode="markers+lines", name="fourfold mode"), row=2, col=2)
    for row in (1, 2):
        for col in (1, 2):
            fig.update_xaxes(title_text=x_title, row=row, col=col)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_yaxes(type="log", row=2, col=1)
    fig.update_layout(
        title=dict(text=f"PP research sweep summary: {sweep}", x=0.5),
        width=1250,
        height=900,
        template="plotly_white",
        legend=dict(groupclick="togglegroup", itemsizing="constant"),
    )
    return fig


def make_sweep_morphology_grid(
    cases: Sequence[tuple[ResearchSimulationResult, TransientResearchConfig, dict[str, object]]],
    *,
    sweep: str,
    columns: int = 3,
) -> go.Figure:
    if not cases:
        raise ValueError("cannot plot an empty sweep")
    columns = max(1, int(columns))
    rows = int(np.ceil(len(cases) / columns))
    titles = [str(summary.get("case_label", i)) for i, (_, _, summary) in enumerate(cases)]
    fig = make_subplots(rows=rows, cols=columns, subplot_titles=titles, horizontal_spacing=0.05, vertical_spacing=0.09)
    for case_idx, (result, config, _) in enumerate(cases):
        row = case_idx // columns + 1
        col = case_idx % columns + 1
        colors = fiber_colors(config, result.initial.omega_atoms, len(result.initial.group_names))
        for k in range(len(result.initial.group_names)):
            idx = np.where(result.initial.group_id == k)[0]
            if len(idx) == 0:
                continue
            if len(idx) > config.max_plot_points_per_group:
                rng = np.random.default_rng(config.seed + 101 * case_idx + k)
                idx = np.sort(rng.choice(idx, size=config.max_plot_points_per_group, replace=False))
            fig.add_trace(
                go.Scattergl(
                    x=result.x_final[idx, 0],
                    y=result.x_final[idx, 1],
                    mode="markers",
                    marker=dict(size=3.2, color=colors[k], opacity=0.70),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
        scale_y = "y" if case_idx == 0 else f"y{case_idx + 1}"
        fig.update_xaxes(range=[-config.domain_radius, config.domain_radius], scaleanchor=scale_y, scaleratio=1, row=row, col=col)
        fig.update_yaxes(range=[-config.domain_radius, config.domain_radius], row=row, col=col)
    fig.update_layout(
        title=dict(text=f"Final morphology grid: {sweep}", x=0.5),
        width=420 * columns,
        height=390 * rows + 90,
        template="plotly_white",
    )
    return fig


def run_pp_research_sweep(
    base_config: TransientResearchConfig,
    sweep: str,
    values: Sequence[object] | None = None,
    *,
    out_dir: Path | str | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[dict[str, object]]:
    """Run a structured transient-artifact sweep and save diagnostics/figures."""

    sweep_values = tuple(_default_research_sweep_values(sweep) if values is None else values)
    root = Path(out_dir) if out_dir is not None else Path(base_config.out_dir) / f"research_{sweep}_sweep"
    root.mkdir(parents=True, exist_ok=True)
    base = replace(
        base_config,
        record_research_diagnostics=True,
        make_dashboard=False,
        make_animation=False,
        backend="numpy" if base_config.force_backend == "direct" else base_config.backend,
    )
    base_initial = make_initial_condition(base)
    np.savez_compressed(
        root / "base_initial_condition.npz",
        x=base_initial.x.astype(np.float32),
        omega=base_initial.omega.astype(np.float32),
        group_id=base_initial.group_id.astype(np.int32),
        omega_atoms=base_initial.omega_atoms.astype(np.float32),
        group_names=np.array(base_initial.group_names),
    )

    summaries: list[dict[str, object]] = []
    cases: list[tuple[ResearchSimulationResult, TransientResearchConfig, dict[str, object]]] = []
    for case_index, value in enumerate(sweep_values):
        cfg = _research_sweep_config(base, sweep, value)
        cfg = replace(
            cfg,
            record_research_diagnostics=True,
            make_dashboard=False,
            make_animation=False,
            out_dir=root / f"case_{case_index:02d}_{_slugify_sweep_value(value)}",
        )
        initial = base_initial if _same_particle_layout(base, cfg) else make_initial_condition(cfg)
        result = run_research_simulation(cfg, initial, cancel_check=cancel_check)
        summary = save_research_run_outputs(result, cfg, cfg.out_dir, initial=initial)
        summary["case_index"] = int(case_index)
        summary["case_value"] = str(value)
        summary["case_label"] = f"{sweep}={value}"
        summaries.append(summary)
        cases.append((result, cfg, summary))

    (root / "sweep_metrics.json").write_text(json.dumps(_jsonable(summaries), indent=2))
    write_sweep_metrics_csv(root / "sweep_metrics.csv", summaries)
    make_sweep_summary_figure(summaries, sweep=sweep).write_html(str(root / "sweep_summary.html"), include_plotlyjs="cdn")
    make_sweep_morphology_grid(cases, sweep=sweep).write_html(str(root / "morphology_grid.html"), include_plotlyjs="cdn")
    return summaries


def _toy_vector_field(
    x: Array,
    *,
    model: str,
    c: float,
    eps: float,
    delta: float,
    saddle_c: float,
) -> Array:
    pts = np.asarray(x, dtype=np.float64)
    r = np.linalg.norm(pts, axis=1, keepdims=True)
    safe_r = np.maximum(r, float(eps))
    if model == "conical":
        return -float(c) * pts / safe_r
    if model == "log_conical":
        coeff = (1.0 / max(float(delta), 1e-30)) + np.log(safe_r)
        return -float(c) * coeff * pts / safe_r
    if model == "saddle":
        lam = float(saddle_c) / safe_r
        return np.c_[-lam[:, 0] * pts[:, 0], lam[:, 0] * pts[:, 1]]
    if model == "combined":
        radial = -float(c) * pts / safe_r
        lam = float(saddle_c) / safe_r
        saddle = np.c_[-lam[:, 0] * pts[:, 0], lam[:, 0] * pts[:, 1]]
        return radial + saddle
    raise ValueError(f"unknown toy model {model!r}")


def _toy_q(x: Array, *, h: float, c: float, eps: float, saddle_c: float) -> float:
    r = np.linalg.norm(np.asarray(x, dtype=np.float64), axis=1)
    scale = max(abs(float(c)), abs(float(saddle_c)))
    return float(abs(h) * scale / max(float(np.min(r)) + float(eps), 1e-30))


def simulate_toy_transient(
    x0: Array,
    *,
    model: str = "conical",
    integrator: IntegratorChoice = "fixed_rk2",
    dt: float = 0.02,
    max_steps: int = 200,
    c: float = 1.0,
    eps: float = 1e-4,
    delta: float = 0.01,
    saddle_c: float = 1.0,
    adaptive_tol: float = 1e-4,
    dt_min: float = 1e-5,
    dt_max: float = 0.05,
    record_every: int = 1,
) -> dict[str, Array]:
    """Integrate conical/saddle toy fields that isolate RK2 shell and hyperbola mechanisms."""

    if integrator not in ("fixed_rk2", "adaptive_rk2"):
        raise ValueError("toy integrator must be 'fixed_rk2' or 'adaptive_rk2'")
    x = np.asarray(x0, dtype=np.float64).copy()
    h_current = float(dt)
    t = 0.0
    frames: list[Array] = []
    rows: list[tuple[float, float, float, float, float]] = []

    def rhs(state: Array) -> Array:
        return _toy_vector_field(state, model=model, c=c, eps=eps, delta=delta, saddle_c=saddle_c)

    for step in range(int(max_steps) + 1):
        if step % max(1, int(record_every)) == 0:
            r = np.linalg.norm(x, axis=1)
            frames.append(x.astype(np.float32, copy=True))
            rows.append((float(step), float(t), float(h_current), float(np.min(r)), _toy_q(x, h=h_current, c=c, eps=eps, saddle_c=saddle_c)))
        if step >= int(max_steps):
            break
        if integrator == "fixed_rk2":
            h = float(dt)
            k1 = rhs(x)
            k2 = rhs(x + h * k1)
            x = x + 0.5 * h * (k1 + k2)
            t += h
            continue

        while True:
            h = float(np.clip(h_current, dt_min, dt_max))
            k1 = rhs(x)
            x_euler = x + h * k1
            k2 = rhs(x_euler)
            x_heun = x + 0.5 * h * (k1 + k2)
            err = float(np.sqrt(np.mean(np.sum((x_heun - x_euler) ** 2, axis=1))))
            if err <= adaptive_tol or h <= dt_min * (1.0 + 1e-12):
                x = x_heun
                t += h
                h_current = float(np.clip(h * _adaptive_step_factor(err, adaptive_tol, grow=True), dt_min, dt_max))
                break
            h_current = float(np.clip(h * _adaptive_step_factor(err, adaptive_tol, grow=False), dt_min, dt_max))

    return {
        "trajectory": np.stack(frames).astype(np.float32),
        "diagnostics": np.array(rows, dtype=np.float64),
    }


def _toy_initial_cloud(n: int, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, int(n))
    radius = np.sqrt(rng.uniform(0.15**2, 1.2**2, int(n)))
    x = np.c_[radius * np.cos(theta), radius * np.sin(theta)]
    x += 0.015 * rng.normal(size=x.shape)
    return x.astype(np.float64)


def make_toy_transient_figure(toy_runs: Sequence[tuple[str, dict[str, Array]]]) -> go.Figure:
    if not toy_runs:
        raise ValueError("toy_runs cannot be empty")
    fig = make_subplots(
        rows=2,
        cols=len(toy_runs),
        specs=[[{"type": "scatter"} for _ in toy_runs], [{"type": "scatter"} for _ in toy_runs]],
        subplot_titles=[label for label, _ in toy_runs] + [f"{label}: r_min and q" for label, _ in toy_runs],
        vertical_spacing=0.12,
    )
    for col, (label, run) in enumerate(toy_runs, start=1):
        traj = np.asarray(run["trajectory"], dtype=np.float64)
        final = traj[-1]
        fig.add_trace(
            go.Scattergl(x=final[:, 0], y=final[:, 1], mode="markers", marker=dict(size=3.5, opacity=0.72), name=label, showlegend=False),
            row=1,
            col=col,
        )
        diag = np.asarray(run["diagnostics"], dtype=np.float64)
        fig.add_trace(go.Scatter(x=diag[:, 1], y=diag[:, 3], mode="lines+markers", name=f"{label} r_min", showlegend=False), row=2, col=col)
        fig.add_trace(go.Scatter(x=diag[:, 1], y=diag[:, 4], mode="lines", name=f"{label} q", showlegend=False), row=2, col=col)
        fig.update_xaxes(range=[-1.4, 1.4], scaleanchor="y" if col == 1 else f"y{col}", scaleratio=1, row=1, col=col)
        fig.update_yaxes(range=[-1.4, 1.4], row=1, col=col)
        fig.update_yaxes(type="log", row=2, col=col)
        fig.update_xaxes(title_text="time", row=2, col=col)
    fig.update_layout(title=dict(text="Toy RK2 shell and hyperbolic transient mechanisms", x=0.5), width=430 * len(toy_runs), height=780, template="plotly_white")
    return fig


def run_toy_transient_suite(
    out_dir: Path | str,
    *,
    h_values: Sequence[float] = (0.09, 0.055, 0.03, 0.015, 0.0075),
    n_points: int = 600,
    seed: int = 2026,
    max_steps: int = 180,
    c: float = 1.0,
    eps: float = 1e-4,
    delta: float = 0.01,
) -> dict[str, object]:
    """Generate toy conical shell-scaling and saddle-invariant artifacts."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not h_values:
        raise ValueError("h_values must contain at least one timestep")
    x0 = _toy_initial_cloud(int(n_points), int(seed))
    shell_rows: list[dict[str, float]] = []
    toy_runs: list[tuple[str, dict[str, Array]]] = []
    for h in h_values:
        run = simulate_toy_transient(x0, model="conical", integrator="fixed_rk2", dt=float(h), max_steps=max_steps, c=c, eps=eps, delta=delta)
        final_radius = np.linalg.norm(run["trajectory"][-1], axis=1)
        shell_rows.append(
            {
                "h": float(h),
                "observed_radius_median": float(np.median(final_radius)),
                "observed_radius_peak": float(compute_morphology_metrics(run["trajectory"][-1])["disk_radius"]),
                "predicted_hc_over_2": float(0.5 * h * c),
            }
        )
    reference_h = float(h_values[min(1, len(h_values) - 1)])
    toy_runs.append(("fixed conical", simulate_toy_transient(x0, model="conical", integrator="fixed_rk2", dt=reference_h, max_steps=max_steps, c=c, eps=eps, delta=delta)))
    toy_runs.append(("adaptive conical", simulate_toy_transient(x0, model="conical", integrator="adaptive_rk2", dt=reference_h, max_steps=max_steps, c=c, eps=eps, delta=delta)))
    toy_runs.append(("adaptive saddle", simulate_toy_transient(x0, model="saddle", integrator="adaptive_rk2", dt=reference_h, max_steps=max_steps, c=0.0, eps=eps, delta=delta, saddle_c=c)))
    make_toy_transient_figure(toy_runs).write_html(str(out / "toy_transients.html"), include_plotlyjs="cdn")

    fig = go.Figure()
    h_arr = np.array([row["h"] for row in shell_rows])
    fig.add_trace(go.Scatter(x=h_arr, y=[row["observed_radius_peak"] for row in shell_rows], mode="markers+lines", name="observed radial peak"))
    fig.add_trace(go.Scatter(x=h_arr, y=[row["observed_radius_median"] for row in shell_rows], mode="markers+lines", name="median radius"))
    fig.add_trace(go.Scatter(x=h_arr, y=[row["predicted_hc_over_2"] for row in shell_rows], mode="lines", name="h c / 2", line=dict(dash="dash")))
    fig.update_layout(title=dict(text="Toy conical fixed-RK2 shell scaling", x=0.5), xaxis_title="h", yaxis_title="radius", width=780, height=520, template="plotly_white")
    fig.write_html(str(out / "toy_shell_scaling.html"), include_plotlyjs="cdn")
    metrics = {"seed": int(seed), "n_points": int(n_points), "c": float(c), "eps": float(eps), "delta": float(delta), "shell_scaling": shell_rows}
    (out / "toy_metrics.json").write_text(json.dumps(_jsonable(metrics), indent=2))
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-fibers", type=int, default=10)
    parser.add_argument("--n-per-fiber", type=int, default=2000)
    parser.add_argument("--shapes", default=",".join(DEFAULT_SHAPES))
    parser.add_argument("--initialization-algorithm", choices=("raw", "alpha_ball"), default="raw")
    parser.add_argument("--initialization-fast-steps", type=int, default=40)
    parser.add_argument("--initialization-fast-min-steps", type=int, default=6)
    parser.add_argument("--initialization-fast-window", type=int, default=3)
    parser.add_argument("--initialization-fast-displacement-tol", type=float, default=1.5e-2)
    parser.add_argument("--color-scheme", choices=("palette", "phase_color"), default="palette")
    parser.add_argument("--initializer-alpha", type=float, default=0.99)
    parser.add_argument("--initializer-K", type=float, default=1.0)
    parser.add_argument("--initializer-grid-size", type=int, default=None)
    parser.add_argument("--initializer-domain-radius", type=float, default=None)
    parser.add_argument("--initializer-dt", type=float, default=0.055)
    parser.add_argument("--alpha", type=float, default=0.50)
    parser.add_argument("--K", type=float, default=1.0)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--domain-radius", type=float, default=9.0)
    parser.add_argument("--dt", type=float, default=0.055)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--tol-rms", type=float, default=1.2e-2)
    parser.add_argument("--backend", choices=("auto", "numpy", "torch"), default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=("auto", "float32", "float64"), default="auto")
    parser.add_argument("--force-backend", choices=("fft", "direct"), default="fft")
    parser.add_argument("--integrator", choices=("fixed_rk2", "adaptive_rk2"), default="adaptive_rk2")
    parser.add_argument("--time-direction", choices=("forward", "backward"), default="forward")
    parser.add_argument("--adaptive-tol", type=float, default=5.0e-3)
    parser.add_argument("--dt-min", type=float, default=1.0e-4)
    parser.add_argument("--dt-max", type=float, default=0.09)
    parser.add_argument("--max-displacement-per-step", type=float, default=0.75)
    parser.add_argument("--no-center-each-step", action="store_true")
    parser.add_argument("--no-clip-each-step", action="store_true")
    parser.add_argument("--research-diagnostics", action="store_true")
    parser.add_argument("--research-diagnostics-every", type=int, default=None)
    parser.add_argument("--research-diagnostic-sample-size", type=int, default=2500)
    parser.add_argument("--research-energy-sample-size", type=int, default=1200)
    parser.add_argument("--research-sweep", choices=("none", "timestep", "alpha", "k_renormalization", "grid", "domain", "particles", "integrator_backend"), default="none")
    parser.add_argument("--research-sweep-values", default=None)
    parser.add_argument("--toy-suite", action="store_true")
    parser.add_argument("--record-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("pp_transient_research_output"))
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--trajectory-frame-count", type=int, default=0)
    parser.add_argument("--max-animation-points-per-group", type=int, default=450)
    parser.add_argument("--animation-density-grid-size", type=int, default=96)
    parser.add_argument("--animation-frame-duration-ms", type=int, default=90)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TransientResearchConfig:
    shapes = tuple(part.strip() for part in args.shapes.split(",") if part.strip())
    if not shapes:
        raise ValueError("--shapes must include at least one shape")
    return TransientResearchConfig(
        alpha=args.alpha,
        K=args.K,
        n_fibers=args.n_fibers,
        n_per_fiber=args.n_per_fiber,
        shape_names=shapes,
        seed=_seed_from_args(args.seed),
        color_scheme=args.color_scheme,
        initialization_algorithm=args.initialization_algorithm,
        initializer_config=InitializerConfig(
            alpha=args.initializer_alpha,
            K=args.initializer_K,
            grid_size=args.initializer_grid_size,
            domain_radius=args.initializer_domain_radius,
            dt=args.initializer_dt,
            max_steps=args.initialization_fast_steps,
            min_steps=args.initialization_fast_min_steps,
            window=args.initialization_fast_window,
            displacement_tol=args.initialization_fast_displacement_tol,
        ),
        initialization_fast_steps=args.initialization_fast_steps,
        initialization_fast_min_steps=args.initialization_fast_min_steps,
        initialization_fast_window=args.initialization_fast_window,
        initialization_fast_displacement_tol=args.initialization_fast_displacement_tol,
        grid_size=args.grid_size,
        domain_radius=args.domain_radius,
        dt=args.dt,
        max_steps=args.max_steps,
        tol_rms=args.tol_rms,
        backend=args.backend,
        device=args.device,
        dtype=args.dtype,
        force_backend=args.force_backend,
        integrator=args.integrator,
        time_direction=args.time_direction,
        adaptive_tol=args.adaptive_tol,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        max_displacement_per_step=args.max_displacement_per_step,
        center_each_step=not args.no_center_each_step,
        clip_each_step=not args.no_clip_each_step,
        record_research_diagnostics=bool(args.research_diagnostics or args.research_sweep != "none"),
        research_diagnostics_every=args.research_diagnostics_every,
        research_diagnostic_sample_size=args.research_diagnostic_sample_size,
        research_energy_sample_size=args.research_energy_sample_size,
        record_every=args.record_every,
        out_dir=args.out_dir,
        make_dashboard=False,
        make_animation=not args.no_animation,
        trajectory_frame_count=args.trajectory_frame_count,
        max_animation_points_per_group=args.max_animation_points_per_group,
        animation_density_grid_size=args.animation_density_grid_size,
        animation_frame_duration_ms=args.animation_frame_duration_ms,
    )


def _parse_research_sweep_values(raw: str | None, sweep: str) -> tuple[object, ...] | None:
    if raw is None or not raw.strip():
        return None
    parts = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not parts:
        return None
    if sweep in ("timestep", "alpha", "domain"):
        return tuple(float(part) for part in parts)
    if sweep in ("grid", "particles"):
        return tuple(int(part) for part in parts)
    if sweep == "k_renormalization":
        return parts
    raise ValueError(f"--research-sweep-values is not supported for sweep {sweep!r}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    if args.toy_suite:
        metrics = run_toy_transient_suite(config.out_dir, seed=config.seed)
        print("Done.")
        print(json.dumps(_jsonable(metrics), indent=2))
        print("Output directory:", config.out_dir)
        return
    if args.research_sweep != "none":
        values = _parse_research_sweep_values(args.research_sweep_values, args.research_sweep)
        summaries = run_pp_research_sweep(config, args.research_sweep, values, out_dir=config.out_dir)
        print("Done.")
        print(json.dumps(_jsonable(summaries), indent=2))
        print("Output directory:", config.out_dir)
        return

    initial = make_initial_condition(config)
    result = run_research_simulation(config, initial)
    out_dir = Path(config.out_dir)
    save_research_run_outputs(result, config, out_dir, initial=initial)
    print("Done.")
    print(json.dumps(_jsonable(summarize_research_run(result, config)), indent=2))
    print("Output directory:", out_dir)


if __name__ == "__main__":
    main()
