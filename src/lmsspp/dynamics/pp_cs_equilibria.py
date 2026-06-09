"""
Reusable 2D Peszek--Poyato particle simulation and Hessian diagnostics.

This module evolves particles with fixed conserved labels ``omega_i`` by

    x_i' = omega_i - A_rho(x_i),

where ``A_rho`` is approximated on a regular grid by CIC deposition and FFT
convolution.  The hot field-evaluation loop can run on NumPy or torch
CPU/CUDA/MPS backends, while public results remain NumPy arrays.  All pieces are callable
for arbitrary fiber layouts, shape families, and prebuilt initial conditions.

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
import colorsys
import json
import threading
import time
import warnings
import webbrowser
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:  # Optional: only needed for the interactive notebook widget.
    import ipywidgets as widgets
except Exception:  # pragma: no cover - notebook dependency is optional
    widgets = None  # type: ignore[assignment]

try:  # Optional accelerator backend; NumPy remains the required baseline.
    import torch
except Exception:  # pragma: no cover - torch is an optional extra
    torch = None  # type: ignore[assignment]

Array = np.ndarray
ShapeSampler = Callable[[int, np.random.Generator], Array]
BackendChoice = Literal["auto", "numpy", "torch"]
DTypeChoice = Literal["auto", "float32", "float64"]
IntegratorChoice = Literal["fixed_rk2", "adaptive_rk2"]
TimeDirectionChoice = Literal["forward", "backward"]
ExternalFieldChoice = Literal["affine", "projective"]
SecondPanelChoice = Literal["heatmap", "velocity"]
DensitySolverChoice = Literal["explicit_fv", "split_implicit_diffusion", "chang_cooper"]
DensityBoundaryChoice = Literal["noflux", "periodic"]
ContinuousDensityPanelChoice = Literal["rho", "r_fiber", "velocity_mag", "div_A"]
InitializationAlgorithmChoice = Literal["raw", "alpha_ball"]
ColorSchemeChoice = Literal["palette", "phase_color"]

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

PP_INITIALIZATION_PRESETS: dict[str, tuple[str, ...]] = {
    "mixed": DEFAULT_SHAPES,
    "gaussian": ("gaussian",),
    "ring": ("ring",),
    "line": ("line",),
    "spiral": ("spiral",),
    "square": ("square",),
}

PP_INITIALIZER_OPTIONS: tuple[tuple[str, InitializationAlgorithmChoice], ...] = (
    ("Raw sample", "raw"),
    ("Alpha>>0 Ball", "alpha_ball"),
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
class InitializerConfig:
    """Parameters for discarded-history PP warmup initializers."""

    alpha: float = 0.99
    K: float = 1.0
    grid_size: int | None = None
    domain_radius: float | None = None
    dt: float = 0.055
    max_steps: int = 40
    min_steps: int = 6
    window: int = 3
    displacement_tol: float = 1.5e-2


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters for a 2D Peszek--Poyato FFT particle simulation."""

    alpha: float = 0.50
    K: float = 1.0
    n_fibers: int = 10
    n_per_fiber: int | Sequence[int] = 200
    fibers: Sequence[FiberSpec] | None = None
    shape_names: Sequence[str] = DEFAULT_SHAPES
    omega_atoms: Array | None = None
    seed: int = 2026
    color_scheme: ColorSchemeChoice = "phase_color"
    initialization_algorithm: InitializationAlgorithmChoice = "raw"
    initializer_config: InitializerConfig | None = None
    initialization_fast_steps: int = 40
    initialization_fast_min_steps: int = 6
    initialization_fast_window: int = 3
    initialization_fast_displacement_tol: float = 1.5e-2

    grid_size: int = 256
    domain_radius: float = 4.0
    dt: float = 0.055
    max_steps: int = 500
    tol_rms: float = 1.2e-2
    min_steps: int = 30
    record_every: int = 5
    backend: BackendChoice = "auto"
    device: str | None = None
    dtype: DTypeChoice = "auto"
    integrator: IntegratorChoice = "adaptive_rk2"
    time_direction: TimeDirectionChoice = "forward"
    external_field: ExternalFieldChoice = "affine"
    projective_epsilon: float = 0.0
    prediction_horizon_tau: float = 0.055
    hamiltonian_q: float = 2.0
    hamiltonian_epsH: float = 0.0
    adaptive_tol: float = 5.0e-3
    dt_min: float = 1.0e-4
    dt_max: float = 0.09
    max_displacement_per_step: float = 0.75

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

    density_solver: DensitySolverChoice = "split_implicit_diffusion"
    eps_entropy: float = 0.0
    density_boundary: DensityBoundaryChoice = "noflux"
    density_cfl_adv: float = 0.45
    density_cfl_diff: float = 0.24
    density_entropy_floor: float = 1.0e-12
    density_renormalize_each_step: bool = True
    record_free_energy: bool = True
    record_entropy_balance: bool = True

    density_dynamic_zoom: bool = True
    density_dynamic_zoom_mass: float = 0.995
    density_dynamic_zoom_margin: float = 1.35
    density_dynamic_zoom_min_width: float | None = None
    density_dynamic_zoom_smoothing: float = 0.25
    density_display_grid_size: int = 96
    density_heatmap_smoothing: bool = True
    density_edge_band_fraction: float = 0.15


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
    backend: str
    device: str
    dtype: str
    field_evaluations: int
    accepted_steps: int
    rejected_steps: int
    dt_min_observed: float
    dt_max_observed: float
    dt_mean: float
    clip_events: int
    initialization_algorithm: str
    initialization_steps: int
    initialization_time: float
    initialization_stop_metric: float


@dataclass(frozen=True)
class DensityInitialCondition:
    """Conditional fiber densities and fixed weights for continuous PP."""

    r_fiber: Array
    omega: Array
    nu: Array
    group_names: tuple[str, ...]


@dataclass(frozen=True)
class DensitySimulationResult:
    """State and diagnostics returned by ``run_density_simulation``."""

    initial: DensityInitialCondition
    r_fiber: Array
    rho_grid: Array
    diagnostics: Array
    trajectory_r_fiber: Array | None
    trajectory_rho: Array | None
    trajectory_steps: Array | None
    trajectory_times: Array | None
    steps: int
    final_time: float
    runtime_seconds: float
    accepted_steps: int
    rejected_steps: int
    dt_min_observed: float
    dt_max_observed: float
    dt_mean: float


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
        if not 0.0 <= alpha <= 2.0:
            raise ValueError("alpha must lie in [0, 2] for the PP kernel normalization")
        if abs(alpha - 1.0) < 1e-9:
            raise ValueError("alpha = 1 is the singular PP point (1/(1-alpha) diverges); choose alpha != 1")
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
        self.backend_name = "numpy"
        self.device_name = "cpu"
        self.dtype_name = "float64"

        kernels = self._build_kernels()
        self.fft_Kx = np.fft.rfft2(kernels[0])
        self.fft_Ky = np.fft.rfft2(kernels[1])
        self.fft_Hxx = np.fft.rfft2(kernels[2])
        self.fft_Hxy = np.fft.rfft2(kernels[3])
        self.fft_Hyy = np.fft.rfft2(kernels[4])
        self.fft_W = np.fft.rfft2(kernels[5])

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

    def convolve(self, rho_grid: Array, fft_kernel: Array) -> Array:
        padded = np.zeros((self.P, self.P), dtype=np.float64)
        padded[: self.G, : self.G] = rho_grid
        conv = np.fft.irfft2(np.fft.rfft2(padded) * fft_kernel, s=(self.P, self.P))
        return conv[: self.G, : self.G]

    def convolve_fields(self, rho_grid: Array, fft_kernels: Sequence[Array]) -> tuple[Array, ...]:
        padded = np.zeros((self.P, self.P), dtype=np.float64)
        padded[: self.G, : self.G] = rho_grid
        rho_hat = np.fft.rfft2(padded)
        return tuple(np.fft.irfft2(rho_hat * fft_kernel, s=(self.P, self.P))[: self.G, : self.G] for fft_kernel in fft_kernels)

    def A_grid_from_particles(self, x: Array) -> tuple[Array, Array, Array]:
        rho_grid = deposit_mass(x, self.G, self.L)
        Ax_grid, Ay_grid = self.convolve_fields(rho_grid, (self.fft_Kx, self.fft_Ky))
        return rho_grid, Ax_grid, Ay_grid

    def A_at_particles(self, x: Array) -> tuple[Array, Array]:
        rho_grid, Ax_grid, Ay_grid = self.A_grid_from_particles(x)
        weights = cic_indices_weights(x, self.G, self.L)
        A = np.c_[interp_grid_with_weights(Ax_grid, weights), interp_grid_with_weights(Ay_grid, weights)]
        return A, rho_grid

    def velocity(self, x: Array, omega: Array) -> tuple[Array, Array]:
        A, _ = self.A_at_particles(x)
        return omega - A, A

    def projective_external(self, x: Array, omega: Array, eps: float) -> Array:
        """Projective external drift E_eps(omega, x) = eps^{-1} S[eps*omega, x].

        Implements the section 5.2 bracket field with the canonical embedding
        ``p(omega) = eps * omega``.  As ``eps -> 0`` this returns ``omega`` (the
        affine Peszek--Poyato label), recovering the flat external gauge.
        """

        return _projective_external_numpy(np.asarray(x, dtype=np.float64), np.asarray(omega, dtype=np.float64), float(eps))

    def hessian_grid_from_rho(self, rho_grid: Array) -> tuple[Array, Array, Array]:
        return self.convolve_fields(rho_grid, (self.fft_Hxx, self.fft_Hxy, self.fft_Hyy))  # type: ignore[return-value]

    def W_grid_from_particles(self, x: Array) -> tuple[Array, Array]:
        rho_grid = deposit_mass(x, self.G, self.L)
        W_grid = self.convolve(rho_grid, self.fft_W)
        return rho_grid, W_grid

    def _build_kernels(self) -> tuple[Array, Array, Array, Array, Array, Array]:
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
        W = np.zeros((self.P, self.P), dtype=np.float64)
        if abs(self.alpha - 2.0) < 1e-9:
            W[mask] = -self.K * np.log(R[mask])
        else:
            W[mask] = self.K * (R[mask] ** (2 - self.alpha)) / ((2 - self.alpha) * (1 - self.alpha))
        return Kx, Ky, Hxx, Hxy, Hyy, W


class FFTPeszekPoyatoDensity2D:
    """Grid/FFT evaluator for continuous-density PP fields and diagnostics."""

    _BOUNDED_PAD_FACTOR = 4

    def __init__(self, alpha: float, K: float, grid_size: int, domain_radius: float):
        if not 0.0 <= alpha < 1.0:
            raise ValueError("continuous density requires 0 <= alpha < 1")
        if grid_size < 4:
            raise ValueError("grid_size must be at least 4")
        if domain_radius <= 0:
            raise ValueError("domain_radius must be positive")

        self.alpha = float(alpha)
        self.K = float(K)
        self.G = int(grid_size)
        self.L = float(domain_radius)
        self.h = 2 * self.L / self.G
        self.P = self._BOUNDED_PAD_FACTOR * self.G
        self._embed_offset = (self.P - self.G) // 2
        self.cell_area = self.h * self.h

        w_kernel = self._build_w_kernel()
        self.fft_W = np.fft.rfft2(w_kernel)

    def marginal_from_fibers(self, r_fiber: Array, nu: Array) -> Array:
        return np.tensordot(nu, r_fiber, axes=(0, 0))

    def _mass_grid_from_density(self, rho_grid: Array) -> Array:
        return rho_grid * self.cell_area

    def _embed_mass(self, mass_grid: Array) -> Array:
        padded = np.zeros((self.P, self.P), dtype=np.float64)
        off = self._embed_offset
        padded[off : off + self.G, off : off + self.G] = mass_grid
        return padded

    def convolve(self, mass_grid: Array, fft_kernel: Array) -> Array:
        padded = self._embed_mass(mass_grid)
        conv = np.fft.irfft2(np.fft.rfft2(padded) * fft_kernel, s=(self.P, self.P))
        off = self._embed_offset
        return conv[off : off + self.G, off : off + self.G]

    def W_grid_from_rho(self, rho_grid: Array) -> Array:
        return self.convolve(self._mass_grid_from_density(rho_grid), self.fft_W)

    def A_grid_from_rho(self, rho_grid: Array) -> tuple[Array, Array]:
        """A_rho = grad(W^alpha * rho) on the grid, using one scalar bounded FFT pass."""
        w_field = self.W_grid_from_rho(rho_grid)
        ax_grid = np.gradient(w_field, self.h, axis=0)
        ay_grid = np.gradient(w_field, self.h, axis=1)
        return ax_grid, ay_grid

    def hessian_grid_from_rho(self, rho_grid: Array) -> tuple[Array, Array, Array]:
        """Hessian of the scalar interaction potential, consistent with A = grad W."""
        w_field = self.W_grid_from_rho(rho_grid)
        ax_grid = np.gradient(w_field, self.h, axis=0)
        ay_grid = np.gradient(w_field, self.h, axis=1)
        hxx = np.gradient(ax_grid, self.h, axis=0)
        hxy = np.gradient(ax_grid, self.h, axis=1)
        hyy = np.gradient(ay_grid, self.h, axis=1)
        return hxx, hxy, hyy

    def _build_w_kernel(self) -> Array:
        coords = make_lag_coords(self.P, self.h)
        xlag, ylag = np.meshgrid(coords, coords, indexing="ij")
        r = np.sqrt(xlag**2 + ylag**2)
        mask = r > 1e-14
        w = np.zeros((self.P, self.P), dtype=np.float64)
        w[mask] = self.K * (r[mask] ** (2 - self.alpha)) / ((2 - self.alpha) * (1 - self.alpha))
        return w


class TorchPeszekPoyato2D:
    """Torch-backed PP field evaluator for CPU, CUDA, and MPS devices."""

    def __init__(
        self,
        alpha: float,
        K: float,
        grid_size: int,
        domain_radius: float,
        *,
        device: Any,
        dtype: Any,
    ):
        if torch is None:  # pragma: no cover - guarded by backend selection
            raise RuntimeError("torch is not available")
        if not 0.0 <= alpha <= 2.0:
            raise ValueError("alpha must lie in [0, 2] for the PP kernel normalization")
        if abs(alpha - 1.0) < 1e-9:
            raise ValueError("alpha = 1 is the singular PP point (1/(1-alpha) diverges); choose alpha != 1")
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
        self.device = torch.device(device)
        self.dtype = dtype
        self.backend_name = "torch"
        self.device_name = str(self.device)
        self.dtype_name = _torch_dtype_name(dtype)

        kernels = self._build_kernels()
        self.fft_Kx = torch.fft.rfft2(kernels[0])
        self.fft_Ky = torch.fft.rfft2(kernels[1])
        self.fft_Hxx = torch.fft.rfft2(kernels[2])
        self.fft_Hxy = torch.fft.rfft2(kernels[3])
        self.fft_Hyy = torch.fft.rfft2(kernels[4])
        self.fft_W = torch.fft.rfft2(kernels[5])

    def asarray(self, x: Array) -> Any:
        return torch.as_tensor(x, dtype=self.dtype, device=self.device)

    def copy_state(self, x: Any) -> Any:
        return x.clone()

    def to_numpy(self, x: Any) -> Array:
        self.synchronize()
        return x.detach().cpu().numpy().astype(np.float64, copy=False)

    def clip_inside(self, x: Any) -> Any:
        margin = 2.1 * self.h
        return torch.clamp(x, -self.L + margin, self.L - margin)

    def clip_inside_with_count(self, x: Any) -> tuple[Any, int]:
        margin = 2.1 * self.h
        lo = -self.L + margin
        hi = self.L - margin
        clipped = torch.any((x < lo) | (x > hi), dim=1)
        return torch.clamp(x, lo, hi), int(torch.count_nonzero(clipped).detach().cpu().item())

    def center(self, x: Any) -> Any:
        return x - x.mean(dim=0, keepdim=True)

    def speed_stats(self, v: Any) -> tuple[float, float]:
        speed2 = torch.sum(v * v, dim=1)
        rms = torch.sqrt(torch.mean(speed2))
        maxv = torch.sqrt(torch.max(speed2))
        return float(rms.detach().cpu().item()), float(maxv.detach().cpu().item())

    def rms_delta(self, a: Any, b: Any) -> float:
        delta = a - b
        rms = torch.sqrt(torch.mean(torch.sum(delta * delta, dim=1)))
        return float(rms.detach().cpu().item())

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def convolve(self, rho_grid: Any, fft_kernel: Any) -> Any:
        padded = torch.zeros((self.P, self.P), dtype=self.dtype, device=self.device)
        padded[: self.G, : self.G] = rho_grid
        conv = torch.fft.irfft2(torch.fft.rfft2(padded) * fft_kernel, s=(self.P, self.P))
        return conv[: self.G, : self.G]

    def convolve_fields(self, rho_grid: Any, fft_kernels: Sequence[Any]) -> tuple[Any, ...]:
        padded = torch.zeros((self.P, self.P), dtype=self.dtype, device=self.device)
        padded[: self.G, : self.G] = rho_grid
        rho_hat = torch.fft.rfft2(padded)
        return tuple(torch.fft.irfft2(rho_hat * fft_kernel, s=(self.P, self.P))[: self.G, : self.G] for fft_kernel in fft_kernels)

    def A_grid_from_particles(self, x: Any) -> tuple[Any, Any, Any]:
        rho_grid = self.deposit_mass(x)
        Ax_grid, Ay_grid = self.convolve_fields(rho_grid, (self.fft_Kx, self.fft_Ky))
        return rho_grid, Ax_grid, Ay_grid

    def A_at_particles(self, x: Any) -> tuple[Any, Any]:
        rho_grid, Ax_grid, Ay_grid = self.A_grid_from_particles(x)
        weights = self.cic_indices_weights(x)
        A = torch.stack(
            [
                self.interp_grid_with_weights(Ax_grid, weights),
                self.interp_grid_with_weights(Ay_grid, weights),
            ],
            dim=1,
        )
        return A, rho_grid

    def velocity(self, x: Any, omega: Any) -> tuple[Any, Any]:
        A, _ = self.A_at_particles(x)
        return omega - A, A

    def projective_external(self, x: Any, omega: Any, eps: float) -> Any:
        """Projective external drift E_eps(omega, x) = eps^{-1} S[eps*omega, x].

        Torch counterpart of the NumPy bracket field used by the projective
        Peszek--Poyato gauge; ``eps -> 0`` returns the affine label ``omega``.
        """

        eps = float(eps)
        if eps == 0.0:
            return omega
        omega_sq = torch.sum(omega * omega, dim=1, keepdim=True)
        x_sq = torch.sum(x * x, dim=1, keepdim=True)
        dot = torch.sum(omega * x, dim=1, keepdim=True)
        kappa = 1.0 - 2.0 * eps * dot + (eps * eps) * omega_sq * x_sq
        return (omega - eps * omega_sq * x) / kappa

    def hessian_grid_from_rho(self, rho_grid: Any) -> tuple[Any, Any, Any]:
        return self.convolve_fields(rho_grid, (self.fft_Hxx, self.fft_Hxy, self.fft_Hyy))  # type: ignore[return-value]

    def W_grid_from_particles(self, x: Any) -> tuple[Any, Any]:
        rho_grid = self.deposit_mass(x)
        W_grid = self.convolve(rho_grid, self.fft_W)
        return rho_grid, W_grid

    def cic_indices_weights(self, x: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
        h = 2 * self.L / self.G
        u = (x[:, 0] + self.L) / h
        v = (x[:, 1] + self.L) / h
        i = torch.floor(u).to(torch.int64).clamp(0, self.G - 2)
        j = torch.floor(v).to(torch.int64).clamp(0, self.G - 2)
        fu = u - i.to(dtype=x.dtype)
        fv = v - j.to(dtype=x.dtype)
        return i, j, (1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv

    def deposit_mass(self, x: Any) -> Any:
        if int(x.shape[0]) == 0:
            raise ValueError("cannot deposit an empty particle set")
        i, j, w00, w10, w01, w11 = self.cic_indices_weights(x)
        mass = 1.0 / int(x.shape[0])
        grid_flat = torch.zeros((self.G * self.G,), dtype=self.dtype, device=self.device)
        grid_flat.index_add_(0, i * self.G + j, mass * w00)
        grid_flat.index_add_(0, (i + 1) * self.G + j, mass * w10)
        grid_flat.index_add_(0, i * self.G + j + 1, mass * w01)
        grid_flat.index_add_(0, (i + 1) * self.G + j + 1, mass * w11)
        return grid_flat.reshape(self.G, self.G)

    def interp_grid_with_weights(self, field: Any, weights: tuple[Any, Any, Any, Any, Any, Any]) -> Any:
        i, j, w00, w10, w01, w11 = weights
        return field[i, j] * w00 + field[i + 1, j] * w10 + field[i, j + 1] * w01 + field[i + 1, j + 1] * w11

    def _build_kernels(self) -> tuple[Any, Any, Any, Any, Any, Any]:
        coords = _torch_lag_coords(self.P, self.h, self.device, self.dtype)
        Xlag, Ylag = torch.meshgrid(coords, coords, indexing="ij")
        R = torch.sqrt(Xlag * Xlag + Ylag * Ylag)
        mask = R > 1e-14

        Kx = torch.zeros((self.P, self.P), dtype=self.dtype, device=self.device)
        Ky = torch.zeros_like(Kx)
        scale_grad = torch.zeros_like(R)
        scale_grad[mask] = torch.pow(R[mask], -self.alpha) / (1 - self.alpha)
        Kx[mask] = self.K * Xlag[mask] * scale_grad[mask]
        Ky[mask] = self.K * Ylag[mask] * scale_grad[mask]

        ex = torch.zeros_like(R)
        ey = torch.zeros_like(R)
        ex[mask] = Xlag[mask] / R[mask]
        ey[mask] = Ylag[mask] / R[mask]

        scale_hess = torch.zeros_like(R)
        scale_hess[mask] = self.K * torch.pow(R[mask], -self.alpha) / (1 - self.alpha)
        Hxx = torch.zeros((self.P, self.P), dtype=self.dtype, device=self.device)
        Hxy = torch.zeros_like(Hxx)
        Hyy = torch.zeros_like(Hxx)
        Hxx[mask] = scale_hess[mask] * (1 - self.alpha * ex[mask] * ex[mask])
        Hxy[mask] = scale_hess[mask] * (-self.alpha * ex[mask] * ey[mask])
        Hyy[mask] = scale_hess[mask] * (1 - self.alpha * ey[mask] * ey[mask])
        W = torch.zeros((self.P, self.P), dtype=self.dtype, device=self.device)
        if abs(self.alpha - 2.0) < 1e-9:
            W[mask] = -self.K * torch.log(R[mask])
        else:
            W[mask] = self.K * torch.pow(R[mask], 2 - self.alpha) / ((2 - self.alpha) * (1 - self.alpha))
        return Kx, Ky, Hxx, Hxy, Hyy, W


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


def fiber_colors(config: SimulationConfig, omega_atoms: Array | None, n_groups: int) -> list[str]:
    """Return Plotly marker colors for omega-fiber groups."""

    scheme = config.color_scheme
    if scheme == "palette" or omega_atoms is None:
        return [DEFAULT_PALETTE[k % len(DEFAULT_PALETTE)] for k in range(n_groups)]
    if scheme != "phase_color":
        raise ValueError(f"unknown color_scheme: {scheme!r}")

    atoms = np.asarray(omega_atoms, dtype=np.float64)
    if atoms.shape != (n_groups, 2):
        return [DEFAULT_PALETTE[k % len(DEFAULT_PALETTE)] for k in range(n_groups)]
    radii = np.linalg.norm(atoms, axis=1)
    intensities = _phase_color_intensities(radii)
    if float(np.max(intensities)) <= 0:
        return ["#000000" for _ in range(n_groups)]

    colors: list[str] = []
    for atom, intensity in zip(atoms, intensities):
        if intensity <= 0:
            colors.append("#000000")
            continue
        hue = float((np.arctan2(atom[1], atom[0]) + 2 * np.pi) % (2 * np.pi)) / (2 * np.pi)
        red, green, blue = colorsys.hsv_to_rgb(hue, float(intensity), float(intensity))
        colors.append(f"#{round(255 * red):02x}{round(255 * green):02x}{round(255 * blue):02x}")
    return colors


def _phase_color_intensities(radii: Array) -> Array:
    intensities = np.zeros_like(radii, dtype=np.float64)
    positive = radii > 0
    if not np.any(positive):
        return intensities

    positive_radii = radii[positive]
    unique_radii = np.unique(positive_radii)
    if len(unique_radii) == 1:
        intensities[positive] = 1.0
        return intensities

    ranks = np.searchsorted(unique_radii, positive_radii, side="left").astype(np.float64)
    normalized = ranks / float(len(unique_radii) - 1)
    intensities[positive] = 0.42 + 0.58 * np.sqrt(normalized)
    return intensities


def _torch_lag_coords(P: int, h: float, device: Any, dtype: Any) -> Any:
    if torch is None:  # pragma: no cover - guarded by backend construction
        raise RuntimeError("torch is not available")
    idx = torch.arange(P, device=device)
    lag = torch.where(idx <= P // 2, idx, idx - P)
    return lag.to(dtype=dtype) * h


def _torch_dtype_name(dtype: Any) -> str:
    if torch is not None and dtype is torch.float32:
        return "float32"
    if torch is not None and dtype is torch.float64:
        return "float64"
    return str(dtype).replace("torch.", "")


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
    return interp_grid_with_weights(field, cic_indices_weights(x, G, L))


def interp_grid_with_weights(field: Array, weights: tuple[Array, Array, Array, Array, Array, Array]) -> Array:
    i, j, w00, w10, w01, w11 = weights
    return (
        field[i, j] * w00
        + field[i + 1, j] * w10
        + field[i, j + 1] * w01
        + field[i + 1, j + 1] * w11
    )


def cs_kappa(a: Array, b: Array) -> Array:
    """Conformal denominator kappa[a, b] = 1 - 2<a,b> + |a|^2 |b|^2 (rowwise)."""

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dot = np.sum(a * b, axis=-1)
    a_sq = np.sum(a * a, axis=-1)
    b_sq = np.sum(b * b, axis=-1)
    return 1.0 - 2.0 * dot + a_sq * b_sq


def cs_cost(a: Array, b: Array) -> Array:
    """Symmetric logarithmic projective cost c_S(a, b) = -1/2 log kappa[a, b]."""

    return -0.5 * np.log(cs_kappa(a, b))


def inversive_bracket(a: Array, b: Array) -> Array:
    """Inversive bracket S[a, b] = (a - |a|^2 b) / kappa[a, b] (rational extension).

    This is grad_b c_S(a, b); for nonzero a it equals (a^+ - b)^+.
    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_sq = np.sum(a * a, axis=-1, keepdims=True)
    kappa = cs_kappa(a, b)[..., None]
    return (a - a_sq * b) / kappa


def _projective_external_numpy(x: Array, omega: Array, eps: float) -> Array:
    eps = float(eps)
    if eps == 0.0:
        return omega
    omega_sq = np.sum(omega * omega, axis=1, keepdims=True)
    x_sq = np.sum(x * x, axis=1, keepdims=True)
    dot = np.sum(omega * x, axis=1, keepdims=True)
    kappa = 1.0 - 2.0 * eps * dot + (eps * eps) * omega_sq * x_sq
    return (omega - eps * omega_sq * x) / kappa


def projective_external_field(omega: Array, x: Array, eps: float) -> Array:
    """External drift of the projective Peszek--Poyato gauge.

    Returns ``E_eps(omega, x) = eps^{-1} S[eps*omega, x]`` with the canonical
    embedding ``p(omega) = eps*omega``.  By the affine-tangent corollary this
    converges to the affine label ``omega`` as ``eps -> 0``, so ``eps`` is a
    single deformation knob between the flat (affine) and projective fields.
    """

    return _projective_external_numpy(np.asarray(x, dtype=np.float64), np.asarray(omega, dtype=np.float64), float(eps))


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


def run_simulation(
    config: SimulationConfig,
    initial: InitialCondition | None = None,
    *,
    cancel_check: Callable[[], bool] | None = None,
) -> SimulationResult:
    """Evolve a PP particle system to an approximate equilibrium."""

    _validate_runtime_config(config)

    def _raise_if_cancelled() -> None:
        if cancel_check is not None and cancel_check():
            raise InterruptedError("PP simulation cancelled.")
    raw_initial = make_initial_condition(config) if initial is None else validate_initial_condition(initial)
    initial, initialization_meta = _apply_initialization_algorithm(config, raw_initial)
    solver = _make_pp_backend(config)
    x = solver.asarray(initial.x)
    omega = solver.asarray(initial.omega)
    x, clip_count = solver.clip_inside_with_count(solver.copy_state(x))
    x_initial = solver.to_numpy(x).copy()
    diagnostics: list[tuple[int, float, float, float, float, int, int, int, int]] = []
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

    proj_eps = float(config.projective_epsilon)
    use_projective = config.external_field == "projective" and proj_eps != 0.0

    def velocity_fn(state_x: Any) -> tuple[Any, Any]:
        """RHS of the (affine or projective) Peszek--Poyato field at ``state_x``.

        Affine:      v = omega - A_rho(state_x)
        Projective:  v = E_eps(omega, state_x) - A_rho(state_x),
        where the interaction term ``A_rho`` (the FFT field) is shared.
        """

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
            if rms < config.tol_rms and accepted_steps > config.min_steps:
                break
            if accepted_steps >= config.max_steps:
                break
            x_pred, clipped = solver.clip_inside_with_count(x + time_sign * dt_fixed * vf)
            clip_events += clipped
            k2, _ = velocity_fn(x_pred)
            field_evaluations += 1
            x = x + 0.5 * time_sign * dt_fixed * (vf + k2)
            x = solver.center(x)
            x, clipped = solver.clip_inside_with_count(x)
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
            if rms < config.tol_rms and accepted_steps > config.min_steps:
                break
            if accepted_steps >= config.max_steps:
                break

            trial_dt = _cfl_limited_dt(dt_current, maxv, solver.h, config)
            while True:
                x_pred, clipped = solver.clip_inside_with_count(x + time_sign * trial_dt * vf)
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

            x = solver.center(x_heun)
            x, clipped = solver.clip_inside_with_count(x)
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
    return SimulationResult(
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
    )


def finite_horizon_gauge_average_field(
    solver: Any,
    x: Any,
    omega: Any,
    tau: float,
) -> tuple[Any, Any, Any]:
    """Finite-horizon gauge-averaged PP field.

    Computes

        V_tau = omega - 1/2 A_rho(x)
                      - 1/2 A_{rho_tau}(x + tau (omega - A_rho(x))),

    where ``rho_tau`` is represented by depositing the predicted particle
    positions.  The predicted sample points are clipped to the FFT box for the
    bounded-grid implementation.
    """

    A_now, rho_now = solver.A_at_particles(x)
    tau = float(tau)
    if tau == 0.0:
        return omega - A_now, A_now, rho_now
    x_tau = solver.clip_inside(x + tau * (omega - A_now))
    A_tau, _ = solver.A_at_particles(x_tau)
    A_bar = 0.5 * (A_now + A_tau)
    return omega - A_bar, A_bar, rho_now


def hamiltonian_exponent_pp_field(
    solver: Any,
    x: Any,
    omega: Any,
    q: float,
    epsH: float,
) -> tuple[Any, Any, Any, Any]:
    """Hamiltonian-exponent PP velocity using the existing FFT interaction field.

    Returns ``(velocity, A, residual, clock)`` with
    ``residual = omega - A`` and
    ``clock = (|residual|^2 + epsH^2)^((q - 2) / 2)``.
    """

    A, _ = solver.A_at_particles(x)
    residual = omega - A
    exponent = 0.5 * (float(q) - 2.0)
    eps2 = float(epsH) * float(epsH)
    if torch is not None and torch.is_tensor(residual):
        r2 = torch.sum(residual * residual, dim=1, keepdim=True)
        clock = torch.pow(r2 + eps2, exponent)
    else:
        r2 = np.sum(residual * residual, axis=1, keepdims=True)
        clock = np.power(r2 + eps2, exponent)
    return clock * residual, A, residual, clock


def _pp_empirical_energy_from_grid(solver: FFTPeszekPoyato2D, x: Array, omega: Array) -> float:
    x = np.asarray(x, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    rho_grid, W_grid = solver.W_grid_from_particles(x)
    linear = -float(np.mean(np.sum(omega * x, axis=1)))
    interaction = 0.5 * float(np.sum(rho_grid * W_grid))
    return linear + interaction


def run_hamiltonian_exponent_simulation(
    config: SimulationConfig,
    initial: InitialCondition | None = None,
    *,
    cancel_check: Callable[[], bool] | None = None,
) -> SimulationResult:
    """Evolve PP particles with the Hamiltonian-exponent residual clock."""

    _validate_runtime_config(config)
    q = float(config.hamiltonian_q)
    epsH = float(config.hamiltonian_epsH)

    def _raise_if_cancelled() -> None:
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Hamiltonian-exponent PP simulation cancelled.")

    raw_initial = make_initial_condition(config) if initial is None else validate_initial_condition(initial)
    initial, initialization_meta = _apply_initialization_algorithm(config, raw_initial)
    solver = _make_pp_backend(config)
    x = solver.asarray(initial.x)
    omega = solver.asarray(initial.omega)
    x, clip_count = solver.clip_inside_with_count(solver.copy_state(x))
    x_initial = solver.to_numpy(x).copy()
    diagnostics: list[tuple[int, float, float, float, float, int, int, int, int]] = []
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

    def velocity_fn(state_x: Any) -> tuple[Any, Any, Any, Any]:
        return hamiltonian_exponent_pp_field(solver, state_x, omega, q, epsH)

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

    def append_diagnostic(step_: int, time_: float, rms_: float, max_residual_: float, dt_: float) -> None:
        diagnostics.append(
            (
                int(step_),
                float(time_),
                float(rms_),
                float(max_residual_),
                float(dt_),
                int(field_evaluations),
                int(accepted_steps),
                int(rejected_steps),
                int(clip_events),
            )
        )

    start = time.time()

    if config.integrator == "fixed_rk2":
        dt_fixed = float(config.dt)
        for _ in range(config.max_steps + 1):
            _raise_if_cancelled()
            record_trajectory(accepted_steps, x, t)
            vf, _, residual_now, _ = velocity_fn(x)
            field_evaluations += 1
            rms, max_residual = solver.speed_stats(residual_now)
            if accepted_steps % config.record_every == 0:
                append_diagnostic(accepted_steps, t, rms, max_residual, dt_fixed)
            if rms < config.tol_rms and accepted_steps > config.min_steps:
                break
            if accepted_steps >= config.max_steps:
                break
            x_pred, clipped = solver.clip_inside_with_count(x + time_sign * dt_fixed * vf)
            clip_events += clipped
            k2, _, _, _ = velocity_fn(x_pred)
            field_evaluations += 1
            x = x + 0.5 * time_sign * dt_fixed * (vf + k2)
            x = solver.center(x)
            x, clipped = solver.clip_inside_with_count(x)
            clip_events += clipped
            t += dt_fixed
            accepted_steps += 1
            dt_history.append(dt_fixed)
    else:
        while True:
            _raise_if_cancelled()
            record_trajectory(accepted_steps, x, t)
            vf, _, residual_now, _ = velocity_fn(x)
            field_evaluations += 1
            rms, max_residual = solver.speed_stats(residual_now)
            _, max_speed = solver.speed_stats(vf)
            if accepted_steps % config.record_every == 0:
                append_diagnostic(accepted_steps, t, rms, max_residual, dt_current)
            if rms < config.tol_rms and accepted_steps > config.min_steps:
                break
            if accepted_steps >= config.max_steps:
                break

            trial_dt = _cfl_limited_dt(dt_current, max_speed, solver.h, config)
            while True:
                x_pred, clipped = solver.clip_inside_with_count(x + time_sign * trial_dt * vf)
                clip_events += clipped
                k2, _, _, _ = velocity_fn(x_pred)
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

            x = solver.center(x_heun)
            x, clipped = solver.clip_inside_with_count(x)
            clip_events += clipped
            t += trial_dt
            accepted_steps += 1
            dt_history.append(trial_dt)
            dt_current = _clamp_dt(trial_dt * _adaptive_step_factor(local_err, config.adaptive_tol, grow=True), config)

    runtime = time.time() - start
    if config.make_animation and (not trajectory_steps or trajectory_steps[-1] != accepted_steps):
        record_trajectory(accepted_steps, x, t, force=True)
    _, A_final_backend, residual_backend, _ = velocity_fn(x)
    field_evaluations += 1
    _, rho_grid_backend = solver.A_at_particles(x)
    field_evaluations += 1
    solver.synchronize()
    x_final = solver.to_numpy(x).copy()
    A_final = solver.to_numpy(A_final_backend)
    residual = solver.to_numpy(residual_backend)
    rho_grid = solver.to_numpy(rho_grid_backend)
    residual_speed2 = np.sum(residual * residual, axis=1)
    dt_min_observed, dt_max_observed, dt_mean = _dt_history_summary(dt_history)
    return SimulationResult(
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
    )


def run_finite_horizon_gauge_averaged_simulation(
    config: SimulationConfig,
    initial: InitialCondition | None = None,
    *,
    cancel_check: Callable[[], bool] | None = None,
) -> SimulationResult:
    """Evolve finite-horizon gauge-averaged Peszek--Poyato dynamics.

    The model parameter ``config.prediction_horizon_tau`` is a physical
    prediction horizon, not the numerical integration step.  The default
    integrator remains adaptive RK2, so a run can use ``dt << tau``.
    """

    _validate_runtime_config(config)
    if float(config.prediction_horizon_tau) < 0.0:
        raise ValueError("prediction_horizon_tau must be non-negative")

    def _raise_if_cancelled() -> None:
        if cancel_check is not None and cancel_check():
            raise InterruptedError("finite-horizon PP simulation cancelled.")

    raw_initial = make_initial_condition(config) if initial is None else validate_initial_condition(initial)
    initial, initialization_meta = _apply_initialization_algorithm(config, raw_initial)
    solver = _make_pp_backend(config)
    x = solver.asarray(initial.x)
    omega = solver.asarray(initial.omega)
    x, clip_count = solver.clip_inside_with_count(solver.copy_state(x))
    x_initial = solver.to_numpy(x).copy()
    diagnostics: list[tuple[int, float, float, float, float, int, int, int, int]] = []
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
    tau = float(config.prediction_horizon_tau)

    def velocity_fn(state_x: Any) -> tuple[Any, Any, Any]:
        return finite_horizon_gauge_average_field(solver, state_x, omega, tau)

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

    start = time.time()

    if config.integrator == "fixed_rk2":
        dt_fixed = float(config.dt)
        for _ in range(config.max_steps + 1):
            _raise_if_cancelled()
            record_trajectory(accepted_steps, x, t)
            vf, _, _ = velocity_fn(x)
            field_evaluations += 2 if tau != 0.0 else 1
            rms, maxv = solver.speed_stats(vf)
            if accepted_steps % config.record_every == 0:
                append_diagnostic(accepted_steps, t, rms, maxv, dt_fixed)
            if rms < config.tol_rms and accepted_steps > config.min_steps:
                break
            if accepted_steps >= config.max_steps:
                break
            x_pred, clipped = solver.clip_inside_with_count(x + time_sign * dt_fixed * vf)
            clip_events += clipped
            k2, _, _ = velocity_fn(x_pred)
            field_evaluations += 2 if tau != 0.0 else 1
            x = x + 0.5 * time_sign * dt_fixed * (vf + k2)
            x = solver.center(x)
            x, clipped = solver.clip_inside_with_count(x)
            clip_events += clipped
            t += dt_fixed
            accepted_steps += 1
            dt_history.append(dt_fixed)
    else:
        while True:
            _raise_if_cancelled()
            record_trajectory(accepted_steps, x, t)
            vf, _, _ = velocity_fn(x)
            field_evaluations += 2 if tau != 0.0 else 1
            rms, maxv = solver.speed_stats(vf)
            if accepted_steps % config.record_every == 0:
                append_diagnostic(accepted_steps, t, rms, maxv, dt_current)
            if rms < config.tol_rms and accepted_steps > config.min_steps:
                break
            if accepted_steps >= config.max_steps:
                break

            trial_dt = _cfl_limited_dt(dt_current, maxv, solver.h, config)
            while True:
                x_pred, clipped = solver.clip_inside_with_count(x + time_sign * trial_dt * vf)
                clip_events += clipped
                k2, _, _ = velocity_fn(x_pred)
                field_evaluations += 2 if tau != 0.0 else 1
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

            x = solver.center(x_heun)
            x, clipped = solver.clip_inside_with_count(x)
            clip_events += clipped
            t += trial_dt
            accepted_steps += 1
            dt_history.append(trial_dt)
            dt_current = _clamp_dt(trial_dt * _adaptive_step_factor(local_err, config.adaptive_tol, grow=True), config)

    runtime = time.time() - start
    if config.make_animation and (not trajectory_steps or trajectory_steps[-1] != accepted_steps):
        record_trajectory(accepted_steps, x, t, force=True)
    residual_backend, A_bar_backend, rho_grid_backend = velocity_fn(x)
    field_evaluations += 2 if tau != 0.0 else 1
    solver.synchronize()
    x_final = solver.to_numpy(x).copy()
    A_final = solver.to_numpy(A_bar_backend)
    residual = solver.to_numpy(residual_backend)
    rho_grid = solver.to_numpy(rho_grid_backend)
    residual_speed2 = np.sum(residual * residual, axis=1)
    dt_min_observed, dt_max_observed, dt_mean = _dt_history_summary(dt_history)
    return SimulationResult(
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
    )


DENSITY_DIAGNOSTIC_FIELDS = (
    "step",
    "time",
    "dt",
    "min_density",
    "max_density",
    "negative_count",
    "mass_min",
    "mass_max",
    "total_mass",
    "max_mass_error",
    "renorm_correction",
    "free_energy",
    "entropy_H",
    "fisher_information",
    "trace_div_A",
    "entropy_balance_rhs",
    "rms_velocity",
)


def _validate_density_config(config: SimulationConfig) -> None:
    if config.density_solver not in ("explicit_fv", "split_implicit_diffusion", "chang_cooper"):
        raise ValueError(f"unknown density_solver: {config.density_solver!r}")
    if config.density_solver == "explicit_fv" and config.eps_entropy > 0.0:
        warnings.warn(
            "density_solver='explicit_fv' with eps_entropy > 0 can develop spurious nonlocal mass; "
            "prefer 'chang_cooper' or 'split_implicit_diffusion'.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not 0.0 <= config.alpha < 1.0:
        raise ValueError("continuous density requires 0 <= alpha < 1")
    if config.density_boundary == "periodic":
        raise ValueError("density_boundary='periodic' is not implemented; use 'noflux'")
    if config.density_boundary != "noflux":
        raise ValueError(f"unknown density_boundary: {config.density_boundary!r}")
    if config.eps_entropy > 0 and config.time_direction == "backward":
        raise ValueError("backward time is incompatible with eps_entropy > 0")
    if config.density_cfl_adv <= 0:
        raise ValueError("density_cfl_adv must be positive")
    if config.density_cfl_diff <= 0:
        raise ValueError("density_cfl_diff must be positive")
    if config.density_entropy_floor <= 0:
        raise ValueError("density_entropy_floor must be positive")
    if not 0.0 < config.density_dynamic_zoom_mass <= 1.0:
        raise ValueError("density_dynamic_zoom_mass must lie in (0, 1]")
    if config.density_dynamic_zoom_margin <= 0:
        raise ValueError("density_dynamic_zoom_margin must be positive")
    if not 0.0 <= config.density_dynamic_zoom_smoothing <= 1.0:
        raise ValueError("density_dynamic_zoom_smoothing must lie in [0, 1]")
    if not 0.0 < config.density_edge_band_fraction < 1.0:
        raise ValueError("density_edge_band_fraction must lie in (0, 1)")


def _density_grid_axis(G: int, L: float) -> Array:
    h = 2 * L / G
    return np.linspace(-L + 0.5 * h, L - 0.5 * h, G)


def _effective_density_display_grid_size(config: SimulationConfig) -> int:
    """Display-only resolution, clamped to the backend grid and a minimum of 4."""

    return int(max(4, min(int(config.density_display_grid_size), int(config.grid_size))))


@dataclass
class _DensityZoomWindow:
    """Square viewport centered at the origin with equal x/y half-extent."""

    half_extent: float

    def as_bounds(self) -> tuple[float, float, float, float]:
        h = float(self.half_extent)
        return (-h, h, -h, h)


def _mass_central_interval(marginal: Array, axis: Array, mass_fraction: float) -> tuple[float, float]:
    total = float(marginal.sum())
    if total <= 0.0:
        return float(axis[0]), float(axis[-1])
    cdf = np.cumsum(marginal) / total
    tail = max(0.0, min(0.5, 0.5 * (1.0 - float(mass_fraction))))
    lo_idx = int(np.clip(np.searchsorted(cdf, tail, side="left"), 0, len(axis) - 1))
    hi_idx = int(np.clip(np.searchsorted(cdf, 1.0 - tail, side="left"), 0, len(axis) - 1))
    return float(axis[lo_idx]), float(axis[hi_idx])


def _density_support_window(
    field: Array,
    axis: Array,
    *,
    L: float,
    mass_fraction: float,
    margin: float,
    min_half_width: float | None,
) -> _DensityZoomWindow:
    """Origin-centered square window: ±max(half_x, half_y) with half_k = max(|lo_k|, |hi_k|)."""

    h = float(axis[1] - axis[0]) if len(axis) > 1 else 2 * L
    min_half = float(2.0 * h if min_half_width is None else min_half_width)
    mx = field.sum(axis=1)
    my = field.sum(axis=0)
    x_lo, x_hi = _mass_central_interval(mx, axis, mass_fraction)
    y_lo, y_hi = _mass_central_interval(my, axis, mass_fraction)
    half_x = max(abs(x_lo), abs(x_hi))
    half_y = max(abs(y_lo), abs(y_hi))
    half_extent = max(min_half, float(margin) * max(half_x, half_y))
    half_extent = min(half_extent, float(L))
    return _DensityZoomWindow(half_extent=half_extent)


def _smooth_density_zoom_window(
    previous: _DensityZoomWindow | None,
    raw: _DensityZoomWindow,
    smoothing: float,
) -> _DensityZoomWindow:
    if previous is None or smoothing <= 0.0:
        return raw
    blend = float(np.clip(smoothing, 0.0, 1.0))
    keep = 1.0 - blend
    return _DensityZoomWindow(half_extent=keep * previous.half_extent + blend * raw.half_extent)


def _bilinear_resample_field(
    field: Array,
    axis: Array,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    display_size: int,
) -> tuple[Array, Array, Array]:
    if display_size < 2:
        raise ValueError("display_size must be at least 2")
    x_out = np.linspace(float(x_lo), float(x_hi), int(display_size))
    y_out = np.linspace(float(y_lo), float(y_hi), int(display_size))
    X_out, Y_out = np.meshgrid(x_out, y_out, indexing="ij")
    G = int(field.shape[0])
    h = float(axis[1] - axis[0]) if len(axis) > 1 else 1.0
    origin = float(axis[0])
    ui = np.clip((X_out - origin) / h, 0.0, G - 1 - 1e-12)
    vi = np.clip((Y_out - origin) / h, 0.0, G - 1 - 1e-12)
    i0 = np.floor(ui).astype(np.int64)
    j0 = np.floor(vi).astype(np.int64)
    i1 = np.minimum(i0 + 1, G - 1)
    j1 = np.minimum(j0 + 1, G - 1)
    fu = ui - i0
    fv = vi - j0
    sampled = (
        (1.0 - fu) * (1.0 - fv) * field[i0, j0]
        + fu * (1.0 - fv) * field[i1, j0]
        + (1.0 - fu) * fv * field[i0, j1]
        + fu * fv * field[i1, j1]
    )
    return sampled, x_out, y_out


def _density_edge_mass_fraction(
    rho: Array,
    axis: Array,
    L: float,
    band_fraction: float,
) -> float:
    if rho.size == 0:
        return 0.0
    h = float(axis[1] - axis[0]) if len(axis) > 1 else 2 * L / max(rho.shape[0], 1)
    total = float(rho.sum()) * h * h
    if total <= 0.0:
        return 0.0
    X, Y = np.meshgrid(axis, axis, indexing="ij")
    edge_cut = max(0.0, min(1.0, 1.0 - float(band_fraction)))
    edge = (np.abs(X) >= edge_cut * L) | (np.abs(Y) >= edge_cut * L)
    return float(rho[edge].sum() * h * h / total)


def _density_display_payload_from_grid(
    field: Array,
    axis: Array,
    cfg: SimulationConfig,
    *,
    dynamic_zoom: bool,
    window: _DensityZoomWindow | None,
) -> dict[str, Any]:
    L = float(cfg.domain_radius)
    if dynamic_zoom and window is not None:
        x_lo, x_hi, y_lo, y_hi = window.as_bounds()
        z, x_axis, y_axis = _bilinear_resample_field(
            field,
            axis,
            x_lo,
            x_hi,
            y_lo,
            y_hi,
            int(_effective_density_display_grid_size(cfg)),
        )
        return {
            "z": np.asarray(z, dtype=np.float64),
            "x": np.asarray(x_axis, dtype=np.float64),
            "y": np.asarray(y_axis, dtype=np.float64),
            "x_range": [float(x_lo), float(x_hi)],
            "y_range": [float(y_lo), float(y_hi)],
            "zoomed": True,
        }
    return {
        "z": np.asarray(field, dtype=np.float64),
        "x": np.asarray(axis, dtype=np.float64),
        "y": np.asarray(axis, dtype=np.float64),
        "x_range": [-L, L],
        "y_range": [-L, L],
        "zoomed": False,
    }


def validate_density_initial_condition(
    initial: DensityInitialCondition,
    config: SimulationConfig,
) -> DensityInitialCondition:
    r_fiber = np.asarray(initial.r_fiber, dtype=np.float64)
    omega = np.asarray(initial.omega, dtype=np.float64)
    nu = np.asarray(initial.nu, dtype=np.float64)
    G = int(config.grid_size)
    if r_fiber.ndim != 3:
        raise ValueError("r_fiber must have shape (n_fibers, G, G)")
    n_fibers = int(r_fiber.shape[0])
    if r_fiber.shape[1:] != (G, G):
        raise ValueError(f"r_fiber must have spatial shape ({G}, {G})")
    if omega.shape != (n_fibers, 2):
        raise ValueError(f"omega must have shape ({n_fibers}, 2)")
    if nu.shape != (n_fibers,):
        raise ValueError(f"nu must have shape ({n_fibers},)")
    if np.any(nu < 0):
        raise ValueError("nu must be non-negative")
    nu_sum = float(nu.sum())
    if abs(nu_sum - 1.0) > 1e-10:
        raise ValueError(f"nu must sum to 1, got {nu_sum}")
    if len(initial.group_names) != n_fibers:
        raise ValueError("group_names length must match fiber count")
    return DensityInitialCondition(
        r_fiber=r_fiber,
        omega=omega,
        nu=nu,
        group_names=tuple(initial.group_names),
    )


def make_density_initial_condition(
    config: SimulationConfig,
    initial: DensityInitialCondition | None = None,
) -> DensityInitialCondition:
    if initial is not None:
        return validate_density_initial_condition(initial, config)

    fibers = _normalize_fibers(config)
    n_fibers = len(fibers)
    rng = np.random.default_rng(config.seed)
    omega_atoms = _omega_atoms_from_config(config, fibers, rng)
    G = int(config.grid_size)
    L = float(config.domain_radius)
    h = 2 * L / G
    axis = _density_grid_axis(G, L)
    x1, x2 = np.meshgrid(axis, axis, indexing="ij")

    r_fiber = np.zeros((n_fibers, G, G), dtype=np.float64)
    group_names: list[str] = []
    for k, spec in enumerate(fibers):
        center = _fiber_center(spec.center, rng)
        sigma = 0.35 + 0.08 * float(rng.random())
        bump = np.exp(-((x1 - center[0]) ** 2 + (x2 - center[1]) ** 2) / (2.0 * sigma**2))
        mass = float(bump.sum()) * h * h
        if mass <= 0:
            raise ValueError(f"fiber {k} initial bump has zero mass")
        r_fiber[k] = bump / mass
        group_names.append(spec.name or _shape_name(spec.shape, k))

    nu = np.full(n_fibers, 1.0 / n_fibers, dtype=np.float64)
    return DensityInitialCondition(
        r_fiber=r_fiber,
        omega=omega_atoms.astype(np.float64, copy=False),
        nu=nu,
        group_names=tuple(group_names),
    )


def _face_average_x(field: Array) -> Array:
    return 0.5 * (field[:-1, :] + field[1:, :])


def _face_average_y(field: Array) -> Array:
    return 0.5 * (field[:, :-1] + field[:, 1:])


def _upwind_advective_flux_x(r: Array, u: Array) -> Array:
    return np.where(u >= 0.0, r[:-1, :] * u, r[1:, :] * u)


def _upwind_advective_flux_y(r: Array, v: Array) -> Array:
    return np.where(v >= 0.0, r[:, :-1] * v, r[:, 1:] * v)


def _centered_diffusive_flux_x(r: Array, eps: float, h: float) -> Array:
    if eps == 0.0:
        return np.zeros((r.shape[0] - 1, r.shape[1]), dtype=np.float64)
    return -eps * (r[1:, :] - r[:-1, :]) / h


def _centered_diffusive_flux_y(r: Array, eps: float, h: float) -> Array:
    if eps == 0.0:
        return np.zeros((r.shape[0], r.shape[1] - 1), dtype=np.float64)
    return -eps * (r[:, 1:] - r[:, :-1]) / h


def _divergence_from_face_fluxes(fx: Array, fy: Array, h: float) -> Array:
    G = fx.shape[1]
    div = np.zeros((G, G), dtype=np.float64)
    div[0, :] += fx[0, :] / h
    if G > 1:
        div[1:-1, :] += (fx[1:, :] - fx[:-1, :]) / h
        div[-1, :] -= fx[-1, :] / h
    div[:, 0] += fy[:, 0] / h
    if G > 1:
        div[:, 1:-1] += (fy[:, 1:] - fy[:, :-1]) / h
        div[:, -1] -= fy[:, -1] / h
    return div


def _chang_cooper_bernoulli(z: Array) -> Array:
    """Bernoulli function B(z) = z / (exp(z) - 1), with B(0) = 1."""

    values = np.asarray(z, dtype=np.float64)
    out = np.ones_like(values)
    mask = np.abs(values) > 1e-10
    scaled = values[mask]
    out[mask] = scaled / np.expm1(scaled)
    return out


def _chang_cooper_face_flux(
    r0: Array,
    r1: Array,
    phi0: Array,
    phi1: Array,
    eps: float,
    h: float,
) -> Array:
    """Scharfetter--Gummel / Chang--Cooper flux for J = r grad Phi + eps grad r across one face."""

    delta_phi = phi1 - phi0
    if eps <= 0.0:
        velocity = -delta_phi / h
        return np.where(velocity >= 0.0, r0 * velocity, r1 * velocity)
    coeff = float(eps) / h
    z = delta_phi / float(eps)
    return coeff * (_chang_cooper_bernoulli(z) * r1 - _chang_cooper_bernoulli(-z) * r0)


def _fiber_potential_grid(
    solver: FFTPeszekPoyatoDensity2D,
    rho: Array,
    omega_k: Array,
    x1: Array,
    x2: Array,
) -> Array:
    """Phi_k = K(W^alpha * rho) - omega_k . x for the entropic Fokker--Planck potential."""

    w_field = solver.W_grid_from_rho(rho)
    return w_field - float(omega_k[0]) * x1 - float(omega_k[1]) * x2


def _density_chang_cooper_step(
    r: Array,
    phi: Array,
    eps: float,
    dt: float,
    h: float,
) -> Array:
    """Conservative Fokker--Planck step for d_t r = div(r grad Phi + eps grad r) with no-flux boundaries."""

    jx = _chang_cooper_face_flux(r[:-1, :], r[1:, :], phi[:-1, :], phi[1:, :], eps, h)
    jy = _chang_cooper_face_flux(r[:, :-1], r[:, 1:], phi[:, :-1], phi[:, 1:], eps, h)
    return r - float(dt) * _divergence_from_face_fluxes(jx, jy, h)


def _implicit_diffusion_noflux_step(
    r: Array,
    eps: float,
    dt: float,
    h: float,
    *,
    iterations: int = 30,
) -> Array:
    """Solve (I - dt * eps * Lap) r_new = r with homogeneous Neumann (no-flux) boundaries."""

    if eps <= 0.0 or dt <= 0.0:
        return np.array(r, dtype=np.float64, copy=True)
    nu = float(dt) * float(eps) / (float(h) * float(h))
    if nu <= 0.0:
        return np.array(r, dtype=np.float64, copy=True)
    out = np.array(r, dtype=np.float64, copy=True)
    for _ in range(max(1, int(iterations))):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="edge")
        neighbors = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
        out = (out + nu * neighbors) / (1.0 + 4.0 * nu)
    return out


def _density_split_implicit_step(
    r: Array,
    phi: Array,
    eps: float,
    dt: float,
    h: float,
) -> Array:
    """Strang split: half implicit diffusion, full potential advection, half implicit diffusion."""

    if eps <= 0.0:
        return _density_chang_cooper_step(r, phi, 0.0, dt, h)
    half_dt = 0.5 * float(dt)
    out = _implicit_diffusion_noflux_step(r, eps, half_dt, h)
    out = _density_chang_cooper_step(out, phi, 0.0, dt, h)
    return _implicit_diffusion_noflux_step(out, eps, half_dt, h)


def _density_fv_step_single(
    r: Array,
    omega_k: Array,
    Ax: Array,
    Ay: Array,
    eps: float,
    dt: float,
    h: float,
    time_sign: float,
) -> Array:
    ux = time_sign * (float(omega_k[0]) - _face_average_x(Ax))
    uy = time_sign * (float(omega_k[1]) - _face_average_y(Ay))
    Jx = _upwind_advective_flux_x(r, ux) + _centered_diffusive_flux_x(r, eps, h)
    Jy = _upwind_advective_flux_y(r, uy) + _centered_diffusive_flux_y(r, eps, h)
    return r - dt * _divergence_from_face_fluxes(Jx, Jy, h)


def _density_cfl_dt(
    Ax: Array,
    Ay: Array,
    omega: Array,
    eps: float,
    h: float,
    config: SimulationConfig,
    time_sign: float,
) -> float:
    Ax_f = _face_average_x(Ax)
    Ay_f = _face_average_y(Ay)
    max_speed = 0.0
    for k in range(int(omega.shape[0])):
        max_speed = max(
            max_speed,
            float(np.max(np.abs(time_sign * (float(omega[k, 0]) - Ax_f)))),
            float(np.max(np.abs(time_sign * (float(omega[k, 1]) - Ay_f)))),
        )
    dt_adv = float(config.density_cfl_adv) * h / max(max_speed, 1e-14)
    if config.density_solver == "explicit_fv" and eps > 0.0:
        dt_diff = float(config.density_cfl_diff) * h * h / float(eps)
        return _clamp_dt(min(dt_adv, dt_diff), config)
    return _clamp_dt(dt_adv, config)


def _density_cfl_dt_from_potential(
    phi_fields: Sequence[Array],
    h: float,
    config: SimulationConfig,
) -> float:
    max_grad = 0.0
    for phi in phi_fields:
        grad_x = np.gradient(phi, h, axis=0)
        grad_y = np.gradient(phi, h, axis=1)
        max_grad = max(max_grad, float(np.max(np.abs(grad_x))), float(np.max(np.abs(grad_y))))
    dt_adv = float(config.density_cfl_adv) * h / max(max_grad, 1e-14)
    return _clamp_dt(dt_adv, config)


def _density_step_single(
    r: Array,
    phi: Array,
    omega_k: Array,
    Ax: Array,
    Ay: Array,
    eps: float,
    dt: float,
    h: float,
    time_sign: float,
    solver_mode: DensitySolverChoice,
) -> Array:
    signed_phi = time_sign * phi
    if solver_mode == "chang_cooper":
        # Unified SG is used for eps=0; entropic runs use Strang split for stability with
        # nonlocal interaction potentials on bounded grids.
        if eps > 0.0:
            return _density_split_implicit_step(r, signed_phi, eps, dt, h)
        return _density_chang_cooper_step(r, signed_phi, 0.0, dt, h)
    if solver_mode == "split_implicit_diffusion":
        return _density_split_implicit_step(r, signed_phi, eps, dt, h)
    return _density_fv_step_single(r, omega_k, Ax, Ay, eps, dt, h, time_sign)


def _repair_fiber_densities(
    r_fiber: Array,
    h: float,
    floor: float,
    renormalize: bool,
) -> tuple[Array, float, int]:
    out = np.array(r_fiber, dtype=np.float64, copy=True)
    negative_count = int(np.count_nonzero(out < 0.0))
    out = np.where(out < 0.0, 0.0, out)
    renorm_correction = 0.0
    area = h * h
    if renormalize:
        for k in range(out.shape[0]):
            mass = float(out[k].sum()) * area
            if mass > floor:
                renorm_correction = max(renorm_correction, abs(mass - 1.0))
                out[k] /= mass
    return out, renorm_correction, negative_count


def _density_diagnostic_row(
    *,
    step: int,
    time: float,
    dt: float,
    r_fiber: Array,
    nu: Array,
    omega: Array,
    solver: FFTPeszekPoyatoDensity2D,
    eps: float,
    floor: float,
    record_free_energy: bool,
) -> Array:
    rho = solver.marginal_from_fibers(r_fiber, nu)
    area = solver.cell_area
    h = solver.h
    masses = np.array([float(r.sum()) * area for r in r_fiber], dtype=np.float64)
    total_mass = float(rho.sum()) * area
    max_mass_error = float(np.max(np.abs(masses - 1.0))) if masses.size else 0.0

    Ax, Ay = solver.A_grid_from_rho(rho)
    Hxx, _, Hyy = solver.hessian_grid_from_rho(rho)
    trace_field = Hxx + Hyy
    trace_div_A = float(np.sum(trace_field * rho) * area / max(total_mass, floor))

    fisher_information = 0.0
    entropy_H = 0.0
    for k in range(r_fiber.shape[0]):
        r_k = r_fiber[k]
        grad_x = np.gradient(r_k, h, axis=0)
        grad_y = np.gradient(r_k, h, axis=1)
        safe_r = np.maximum(r_k, floor)
        fisher_information += float(nu[k] * np.sum((grad_x * grad_x + grad_y * grad_y) / safe_r) * area)
        entropy_H += float(nu[k] * np.sum(r_k * np.log(safe_r)) * area)

    free_energy = 0.0
    if record_free_energy:
        W_grid = solver.W_grid_from_rho(rho)
        free_energy = 0.5 * float(np.sum(W_grid * rho) * area)

    entropy_balance_rhs = trace_div_A - eps * fisher_information

    vx = omega[:, None, None, 0] - Ax[None, :, :]
    vy = omega[:, None, None, 1] - Ay[None, :, :]
    speed2 = nu[:, None, None] * (vx * vx + vy * vy) * r_fiber
    rms_velocity = float(np.sqrt(np.sum(speed2) * area / max(total_mass, floor)))

    return np.array(
        [
            float(step),
            float(time),
            float(dt),
            float(np.min(r_fiber)),
            float(np.max(r_fiber)),
            0.0,
            float(np.min(masses)),
            float(np.max(masses)),
            total_mass,
            max_mass_error,
            0.0,
            free_energy,
            entropy_H,
            fisher_information,
            trace_div_A,
            entropy_balance_rhs,
            rms_velocity,
        ],
        dtype=np.float64,
    )


def run_density_simulation(
    config: SimulationConfig,
    initial: DensityInitialCondition | None = None,
    *,
    cancel_check: Callable[[], bool] | None = None,
) -> DensitySimulationResult:
    """Evolve conditional fiber densities with conservative Fokker--Planck discretizations."""

    _validate_density_config(config)

    def _raise_if_cancelled() -> None:
        if cancel_check is not None and cancel_check():
            raise InterruptedError("Density simulation cancelled.")
    initial_cond = make_density_initial_condition(config, initial)
    solver = FFTPeszekPoyatoDensity2D(config.alpha, config.K, config.grid_size, config.domain_radius)

    r_fiber = np.array(initial_cond.r_fiber, dtype=np.float64, copy=True)
    nu = np.asarray(initial_cond.nu, dtype=np.float64)
    omega = np.asarray(initial_cond.omega, dtype=np.float64)
    eps = float(config.eps_entropy)
    floor = float(config.density_entropy_floor)
    time_sign = -1.0 if config.time_direction == "backward" else 1.0

    diagnostics: list[Array] = []
    trajectory_r: list[Array] = []
    trajectory_rho: list[Array] = []
    trajectory_steps: list[int] = []
    trajectory_times: list[float] = []
    trajectory_stride = _trajectory_stride(config)
    trajectory_limit = _trajectory_frame_limit(config)

    dt_current = _clamp_dt(float(config.dt), config)
    dt_history: list[float] = []
    accepted_steps = 0
    rejected_steps = 0
    t = 0.0
    use_fixed_dt = config.integrator == "fixed_rk2"
    solver_mode = config.density_solver
    axis = _density_grid_axis(solver.G, solver.L)
    x1, x2 = np.meshgrid(axis, axis, indexing="ij")

    def record_trajectory(step_: int, time_: float, *, force: bool = False) -> None:
        if not config.make_animation:
            return
        if trajectory_steps and not force and step_ % trajectory_stride != 0 and step_ != config.max_steps:
            return
        if trajectory_limit is not None and len(trajectory_steps) >= trajectory_limit and not force and step_ != config.max_steps:
            return
        trajectory_r.append(r_fiber.astype(np.float32, copy=True))
        trajectory_rho.append(solver.marginal_from_fibers(r_fiber, nu).astype(np.float32, copy=False))
        trajectory_steps.append(int(step_))
        trajectory_times.append(float(time_))

    start = time.time()

    while True:
        _raise_if_cancelled()
        record_trajectory(accepted_steps, t)
        rho = solver.marginal_from_fibers(r_fiber, nu)
        Ax, Ay = solver.A_grid_from_rho(rho)
        phi_fields = tuple(_fiber_potential_grid(solver, rho, omega[k], x1, x2) for k in range(omega.shape[0]))
        if solver_mode in ("chang_cooper", "split_implicit_diffusion"):
            cfl_dt = _density_cfl_dt_from_potential(phi_fields, solver.h, config)
        else:
            cfl_dt = _density_cfl_dt(Ax, Ay, omega, eps, solver.h, config, time_sign)
        if use_fixed_dt:
            step_dt = min(float(config.dt), cfl_dt)
        else:
            step_dt = min(cfl_dt, dt_current)
        row = _density_diagnostic_row(
            step=accepted_steps,
            time=t,
            dt=step_dt,
            r_fiber=r_fiber,
            nu=nu,
            omega=omega,
            solver=solver,
            eps=eps,
            floor=floor,
            record_free_energy=config.record_free_energy or config.record_entropy_balance,
        )
        rms_velocity = float(row[DENSITY_DIAGNOSTIC_FIELDS.index("rms_velocity")])

        if accepted_steps % config.record_every == 0 and (config.record_free_energy or config.record_entropy_balance):
            diagnostics.append(row.copy())

        if rms_velocity < config.tol_rms and accepted_steps > config.min_steps:
            break
        if accepted_steps >= config.max_steps:
            break

        r_next = np.empty_like(r_fiber)
        for k in range(r_fiber.shape[0]):
            r_next[k] = _density_step_single(
                r_fiber[k],
                phi_fields[k],
                omega[k],
                Ax,
                Ay,
                eps,
                step_dt,
                solver.h,
                time_sign,
                solver_mode,
            )

        r_fiber, renorm_correction, negative_count = _repair_fiber_densities(
            r_next,
            solver.h,
            floor,
            config.density_renormalize_each_step,
        )
        if diagnostics:
            diagnostics[-1][DENSITY_DIAGNOSTIC_FIELDS.index("renorm_correction")] = renorm_correction
            diagnostics[-1][DENSITY_DIAGNOSTIC_FIELDS.index("negative_count")] = float(negative_count)

        t += step_dt
        accepted_steps += 1
        dt_history.append(step_dt)
        if not use_fixed_dt:
            dt_current = _clamp_dt(step_dt, config)

    runtime = time.time() - start
    if config.make_animation and (not trajectory_steps or trajectory_steps[-1] != accepted_steps):
        record_trajectory(accepted_steps, t, force=True)

    rho_final = solver.marginal_from_fibers(r_fiber, nu)
    dt_min_observed, dt_max_observed, dt_mean = _dt_history_summary(dt_history)
    return DensitySimulationResult(
        initial=initial_cond,
        r_fiber=r_fiber,
        rho_grid=rho_final,
        diagnostics=np.stack(diagnostics) if diagnostics else np.zeros((0, len(DENSITY_DIAGNOSTIC_FIELDS))),
        trajectory_r_fiber=np.stack(trajectory_r) if trajectory_r else None,
        trajectory_rho=np.stack(trajectory_rho) if trajectory_rho else None,
        trajectory_steps=np.array(trajectory_steps, dtype=np.int64) if trajectory_steps else None,
        trajectory_times=np.array(trajectory_times, dtype=np.float64) if trajectory_times else None,
        steps=int(accepted_steps),
        final_time=float(t),
        runtime_seconds=float(runtime),
        accepted_steps=int(accepted_steps),
        rejected_steps=int(rejected_steps),
        dt_min_observed=dt_min_observed,
        dt_max_observed=dt_max_observed,
        dt_mean=dt_mean,
    )


def _apply_initialization_algorithm(
    config: SimulationConfig,
    initial: InitialCondition,
) -> tuple[InitialCondition, dict[str, object]]:
    if config.initialization_algorithm == "raw":
        return initial, {"algorithm": "raw", "steps": 0, "time": 0.0, "stop_metric": 0.0}
    if config.initialization_algorithm == "alpha_ball":
        return _alpha_ball_initial_condition(config, initial)
    raise ValueError(f"unknown initialization_algorithm: {config.initialization_algorithm!r}")


def _alpha_ball_initial_condition(
    config: SimulationConfig,
    initial: InitialCondition,
) -> tuple[InitialCondition, dict[str, object]]:
    """Discarded-history fixed-RK2 fast-phase initializer.

    This intentionally mirrors the legacy PP integrator: NumPy float64,
    forward time, fixed `config.dt`, no adaptive CFL limiting, and no trajectory
    recording.  Only the final particle locations become the new initial state.
    """

    init_config = _resolve_initializer_config(config)
    grid_size = config.grid_size if init_config.grid_size is None else int(init_config.grid_size)
    domain_radius = config.domain_radius if init_config.domain_radius is None else float(init_config.domain_radius)
    solver = FFTPeszekPoyato2D(init_config.alpha, init_config.K, grid_size, domain_radius)
    x = solver.clip_inside(np.asarray(initial.x, dtype=np.float64).copy())
    omega = np.asarray(initial.omega, dtype=np.float64)
    max_steps = max(0, int(init_config.max_steps))
    min_steps = max(0, int(init_config.min_steps))
    window = max(1, int(init_config.window))
    tol = float(init_config.displacement_tol)
    recent_metrics: list[float] = []
    stop_metric = float("inf") if max_steps else 0.0
    steps_done = 0

    for step in range(max_steps):
        vf, _ = solver.velocity(x, omega)
        x_pred = solver.clip_inside(x + float(init_config.dt) * vf)
        k2, _ = solver.velocity(x_pred, omega)
        x_next = x + 0.5 * float(init_config.dt) * (vf + k2)
        x_next -= x_next.mean(axis=0, keepdims=True)
        x_next = solver.clip_inside(x_next)

        step_rms = float(np.sqrt(np.mean(np.sum((x_next - x) ** 2, axis=1))))
        rel = x_next - x_next.mean(axis=0, keepdims=True)
        support = float(np.quantile(np.linalg.norm(rel, axis=1), 0.95))
        stop_metric = step_rms / max(support, solver.h, 1e-12)
        recent_metrics.append(stop_metric)
        if len(recent_metrics) > window:
            recent_metrics.pop(0)

        x = x_next
        steps_done = step + 1
        if steps_done >= min_steps and len(recent_metrics) == window and max(recent_metrics) <= tol:
            break

    warmed = InitialCondition(
        x=x.astype(np.float64, copy=False),
        omega=omega.copy(),
        group_id=np.asarray(initial.group_id, dtype=np.int64).copy(),
        omega_atoms=np.asarray(initial.omega_atoms, dtype=np.float64).copy(),
        group_names=tuple(initial.group_names),
    )
    return warmed, {
        "algorithm": "alpha_ball",
        "steps": int(steps_done),
        "time": float(steps_done * float(init_config.dt)),
        "stop_metric": float(stop_metric),
    }


def _resolve_initializer_config(config: SimulationConfig) -> InitializerConfig:
    if config.initializer_config is not None:
        return config.initializer_config
    return InitializerConfig(
        max_steps=config.initialization_fast_steps,
        min_steps=config.initialization_fast_min_steps,
        window=config.initialization_fast_window,
        displacement_tol=config.initialization_fast_displacement_tol,
    )


def _validate_runtime_config(config: SimulationConfig) -> None:
    if config.backend not in ("auto", "numpy", "torch"):
        raise ValueError("backend must be one of 'auto', 'numpy', or 'torch'")
    if config.initialization_algorithm not in ("raw", "alpha_ball"):
        raise ValueError("initialization_algorithm must be 'raw' or 'alpha_ball'")
    if config.initialization_fast_steps < 0:
        raise ValueError("initialization_fast_steps must be non-negative")
    if config.initialization_fast_min_steps < 0:
        raise ValueError("initialization_fast_min_steps must be non-negative")
    if config.initialization_fast_window <= 0:
        raise ValueError("initialization_fast_window must be positive")
    if config.initialization_fast_displacement_tol <= 0:
        raise ValueError("initialization_fast_displacement_tol must be positive")
    if config.color_scheme not in ("palette", "phase_color"):
        raise ValueError("color_scheme must be 'palette' or 'phase_color'")
    init_config = _resolve_initializer_config(config)
    if init_config.alpha >= 1:
        raise ValueError("initializer_config.alpha must be < 1")
    if init_config.grid_size is not None and init_config.grid_size < 4:
        raise ValueError("initializer_config.grid_size must be at least 4")
    if init_config.domain_radius is not None and init_config.domain_radius <= 0:
        raise ValueError("initializer_config.domain_radius must be positive")
    if init_config.dt <= 0:
        raise ValueError("initializer_config.dt must be positive")
    if init_config.max_steps < 0:
        raise ValueError("initializer_config.max_steps must be non-negative")
    if init_config.min_steps < 0:
        raise ValueError("initializer_config.min_steps must be non-negative")
    if init_config.window <= 0:
        raise ValueError("initializer_config.window must be positive")
    if init_config.displacement_tol <= 0:
        raise ValueError("initializer_config.displacement_tol must be positive")
    if config.dtype not in ("auto", "float32", "float64"):
        raise ValueError("dtype must be one of 'auto', 'float32', or 'float64'")
    if config.integrator not in ("fixed_rk2", "adaptive_rk2"):
        raise ValueError("integrator must be 'fixed_rk2' or 'adaptive_rk2'")
    if config.time_direction not in ("forward", "backward"):
        raise ValueError("time_direction must be 'forward' or 'backward'")
    if config.external_field not in ("affine", "projective"):
        raise ValueError("external_field must be 'affine' or 'projective'")
    if not np.isfinite(config.projective_epsilon) or config.projective_epsilon < 0:
        raise ValueError("projective_epsilon must be a finite, non-negative number")
    if not np.isfinite(config.prediction_horizon_tau) or config.prediction_horizon_tau < 0:
        raise ValueError("prediction_horizon_tau must be a finite, non-negative number")
    if not np.isfinite(config.hamiltonian_q) or not 0.0 <= config.hamiltonian_q <= 2.0:
        raise ValueError("hamiltonian_q must be a finite number in [0, 2]")
    if not np.isfinite(config.hamiltonian_epsH) or config.hamiltonian_epsH < 0:
        raise ValueError("hamiltonian_epsH must be a finite, non-negative number")
    if config.dt <= 0:
        raise ValueError("dt must be positive")
    if config.dt_min <= 0:
        raise ValueError("dt_min must be positive")
    if config.dt_max < config.dt_min:
        raise ValueError("dt_max must be at least dt_min")
    if config.adaptive_tol <= 0:
        raise ValueError("adaptive_tol must be positive")
    if config.max_displacement_per_step < 0:
        raise ValueError("max_displacement_per_step must be non-negative")
    if config.max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if config.min_steps < 0:
        raise ValueError("min_steps must be non-negative")
    if config.record_every <= 0:
        raise ValueError("record_every must be positive")


def _make_pp_backend(config: SimulationConfig) -> FFTPeszekPoyato2D | TorchPeszekPoyato2D:
    backend = str(config.backend)
    if backend == "auto":
        if torch is not None and _torch_accelerator_available():
            backend = "torch"
        else:
            if torch is None:
                warnings.warn("torch is unavailable; falling back to the NumPy PP backend.", RuntimeWarning, stacklevel=2)
            backend = "numpy"

    if backend == "numpy":
        if config.device is not None:
            warnings.warn("SimulationConfig.device is ignored by the NumPy PP backend.", RuntimeWarning, stacklevel=2)
        if config.dtype == "float32":
            warnings.warn("The NumPy PP backend preserves the legacy float64 path; dtype='float32' is ignored.", RuntimeWarning, stacklevel=2)
        return FFTPeszekPoyato2D(config.alpha, config.K, config.grid_size, config.domain_radius)

    if torch is None:
        raise RuntimeError("backend='torch' requires the optional torch dependency")
    device = _select_torch_device(config.device)
    dtype = _resolve_torch_dtype(config.dtype, device)
    return TorchPeszekPoyato2D(config.alpha, config.K, config.grid_size, config.domain_radius, device=device, dtype=dtype)


def _torch_accelerator_available() -> bool:
    if torch is None:
        return False
    return bool(torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()))


def _select_torch_device(device: str | None) -> Any:
    if torch is None:  # pragma: no cover - guarded by caller
        raise RuntimeError("torch is not available")
    if device:
        selected = torch.device(device)
    elif torch.cuda.is_available():
        selected = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        selected = torch.device("mps")
    else:
        selected = torch.device("cpu")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if selected.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is false")
    return selected


def _resolve_torch_dtype(dtype: DTypeChoice, device: Any) -> Any:
    if torch is None:  # pragma: no cover - guarded by caller
        raise RuntimeError("torch is not available")
    if dtype == "auto":
        return torch.float32 if device.type in ("cuda", "mps") else torch.float64
    if dtype == "float32":
        return torch.float32
    if device.type == "mps":
        raise ValueError("MPS does not support torch.float64; use dtype='auto' or dtype='float32'")
    return torch.float64


def _clamp_dt(dt: float, config: SimulationConfig) -> float:
    return float(np.clip(float(dt), float(config.dt_min), float(config.dt_max)))


def _cfl_limited_dt(dt: float, max_speed: float, h: float, config: SimulationConfig) -> float:
    limited = _clamp_dt(dt, config)
    if config.max_displacement_per_step > 0 and max_speed > 1e-14:
        cfl_dt = float(config.max_displacement_per_step) * float(h) / float(max_speed)
        limited = min(limited, max(float(config.dt_min), cfl_dt))
    return _clamp_dt(limited, config)


def _adaptive_step_factor(local_err: float, tol: float, *, grow: bool) -> float:
    if not np.isfinite(local_err) or local_err <= 0:
        return 2.0 if grow else 0.25
    factor = 0.92 * float(tol / local_err) ** 0.5
    if grow:
        return float(np.clip(factor, 0.5, 2.0))
    return float(np.clip(factor, 0.2, 0.8))


def _dt_history_summary(dt_history: Sequence[float]) -> tuple[float, float, float]:
    if not dt_history:
        return 0.0, 0.0, 0.0
    arr = np.asarray(dt_history, dtype=np.float64)
    return float(arr.min()), float(arr.max()), float(arr.mean())


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
        "backend": result.backend,
        "device": result.device,
        "dtype": result.dtype,
        "initialization_algorithm": result.initialization_algorithm,
        "initialization_steps": int(result.initialization_steps),
        "initialization_time": float(result.initialization_time),
        "initialization_stop_metric": float(result.initialization_stop_metric),
        "integrator": config.integrator,
        "time_direction": config.time_direction,
        "field_evaluations": int(result.field_evaluations),
        "accepted_steps": int(result.accepted_steps),
        "rejected_steps": int(result.rejected_steps),
        "dt_min_observed": float(result.dt_min_observed),
        "dt_max_observed": float(result.dt_max_observed),
        "dt_mean": float(result.dt_mean),
        "clip_events": int(result.clip_events),
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
    colors = fiber_colors(config, result.initial.omega_atoms, len(group_names))
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
        color = colors[k]
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
                    marker=dict(size=4, color=colors[k], opacity=0.70),
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
    fig.update_xaxes(title_text="x1", range=[-config.domain_radius, config.domain_radius], scaleanchor="y", scaleratio=1, row=1, col=1)
    fig.update_yaxes(title_text="x2", range=[-config.domain_radius, config.domain_radius], row=1, col=1)
    fig.update_xaxes(title_text="x1", range=[-config.domain_radius, config.domain_radius], row=1, col=2)
    fig.update_yaxes(title_text="x2", range=[-config.domain_radius, config.domain_radius], scaleanchor="x2", scaleratio=1, row=1, col=2)
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


def _initialization_preset_from_config(config: SimulationConfig) -> str:
    if config.fibers is not None:
        return "mixed"
    shape_names = tuple(config.shape_names)
    for key, shapes in PP_INITIALIZATION_PRESETS.items():
        if shape_names == tuple(shapes):
            return key
    return "mixed"


def _shape_names_for_initialization_preset(preset: str) -> tuple[str, ...]:
    return PP_INITIALIZATION_PRESETS.get(str(preset), DEFAULT_SHAPES)


def _n_per_fiber_scalar(config: SimulationConfig) -> int:
    value = config.n_per_fiber
    if isinstance(value, int):
        return int(value)
    if not value:
        return 1
    return int(value[0])


def make_dynamics_widget(
    config: SimulationConfig,
    result: SimulationResult | None = None,
    *,
    width: int = 1450,
    height: int = 720,
    second_panel: SecondPanelChoice = "velocity",
) -> "PPDynamicsWidget":
    """Build the interactive notebook animation widget.

    Mirrors the precompute / play / step / frame-slider caching model of
    ``LMSOpticalDiskWidget``: every frame is precomputed once, then playback
    only swaps cached arrays into the existing traces, so the right-hand panel
    updates in place instead of being redrawn each frame.

    ``second_panel`` selects the right-hand subplot: ``"velocity"`` (default)
    scatters the per-particle velocity ``x_dot_i`` with the same fiber colors as
    the left panel, while ``"heatmap"`` shows the precomputed joint density.
    """

    return PPDynamicsWidget(config, result=result, width=width, height=height, second_panel=second_panel)


def make_finite_horizon_gauge_averaged_widget(
    config: SimulationConfig,
    result: SimulationResult | None = None,
    *,
    width: int = 1450,
    height: int = 720,
    second_panel: SecondPanelChoice = "velocity",
) -> "FiniteHorizonGaugeAveragedPeszekPoyatoDynamicsWidget":
    """Build the finite-horizon gauge-averaged PP notebook widget."""

    return FiniteHorizonGaugeAveragedPeszekPoyatoDynamicsWidget(
        config,
        result=result,
        width=width,
        height=height,
        second_panel=second_panel,
    )


def make_hamiltonian_exponent_widget(
    config: SimulationConfig,
    result: SimulationResult | None = None,
    *,
    width: int = 1450,
    height: int = 980,
    second_panel: SecondPanelChoice = "velocity",
) -> "HamiltonianExponentPeszekPoyatoDynamicsWidget":
    """Build the Hamiltonian-exponent PP notebook widget."""

    return HamiltonianExponentPeszekPoyatoDynamicsWidget(
        config,
        result=result,
        width=width,
        height=height,
        second_panel=second_panel,
    )


class _AsyncPrecomputeControlsMixin:
    """Shared async precompute, interrupt, and playback-disable plumbing for PP widgets."""

    _precompute_worker_thread_name = "pp-precompute-worker"

    def _init_async_precompute_state(self) -> None:
        self._precompute_busy = False
        self._precompute_stale_message = ""
        self._async_lock = threading.Lock()
        self._async_seq = 0
        self._async_cancel_before = 0
        self._async_pending_job: dict[str, Any] | None = None
        self._async_worker: threading.Thread | None = None

    def _dispatch_precompute_callback(self, fn: Callable[[], None]) -> None:
        fn()

    def _release_precompute_worker_thread(self) -> None:
        self._async_worker = None

    def _is_precompute_cancelled(self, seq: int) -> bool:
        with self._async_lock:
            return int(seq) <= int(self._async_cancel_before)

    def _playback_disabled(self) -> bool:
        return bool(self._precompute_busy) or len(self._frame_payloads) <= 1

    def _set_precompute_computing(self) -> None:
        self._precompute_busy = True
        self._cache_valid = False
        self.cache_status_html.value = "<span style='color:#666'>Computing all cached frames...</span>"
        self.btn_precompute.description = "Interrupt"
        self.btn_precompute.button_style = "info"
        self.btn_precompute.disabled = False
        self.btn_step.disabled = True
        self.play.disabled = True
        self.frame_slider.disabled = True
        self._updating = True
        try:
            self.play.value = 0
            self.frame_slider.value = 0
        finally:
            self._updating = False

    def _capture_precompute_job(self) -> dict[str, Any]:
        raise NotImplementedError

    def _prepare_precompute_job(self, job: dict[str, Any]) -> None:
        return None

    def _run_precompute_job(self, job: dict[str, Any]) -> Any:
        seq = int(job["seq"])
        return run_simulation(
            job["config"],
            cancel_check=lambda: self._is_precompute_cancelled(seq),
        )

    def _ingest_precompute_result(self, result: Any) -> None:
        raise NotImplementedError

    def _queue_async_precompute(self) -> None:
        job = self._capture_precompute_job()
        self._prepare_precompute_job(job)
        self._precompute_stale_message = ""
        self._set_precompute_computing()
        with self._async_lock:
            self._async_seq += 1
            seq = int(self._async_seq)
            job["seq"] = seq
            self._async_pending_job = job
            worker_alive = self._async_worker is not None and self._async_worker.is_alive()
            if worker_alive:
                return
            self._async_worker = threading.Thread(
                target=self._async_precompute_worker_loop,
                name=self._precompute_worker_thread_name,
                daemon=True,
            )
            self._async_worker.start()

    def _async_precompute_worker_loop(self) -> None:
        try:
            while True:
                with self._async_lock:
                    job = self._async_pending_job
                    self._async_pending_job = None
                if job is None:
                    self._dispatch_precompute_callback(self._on_precompute_worker_idle)
                    return
                seq = int(job["seq"])
                try:
                    result = self._run_precompute_job(job)
                except InterruptedError:
                    continue
                except Exception as exc:
                    if self._is_precompute_cancelled(seq):
                        continue
                    err_text = str(exc)
                    self._dispatch_precompute_callback(lambda msg=err_text: self._on_precompute_worker_error(msg))
                    continue
                if self._is_precompute_cancelled(seq):
                    continue
                self._dispatch_precompute_callback(
                    lambda res=result, job_seq=seq: self._apply_async_precompute_result(res, job_seq)
                )

                with self._async_lock:
                    if self._async_pending_job is None:
                        return
        finally:
            self._dispatch_precompute_callback(self._release_precompute_worker_thread)

    def _on_precompute_worker_idle(self) -> None:
        if not self._precompute_busy:
            return
        message = self._precompute_stale_message or "Precompute interrupted."
        self._precompute_busy = False
        self._mark_cache_stale(message)

    def _on_precompute_worker_error(self, err_text: str) -> None:
        self._precompute_busy = False
        self.btn_precompute.description = "Precompute flow"
        self.btn_precompute.button_style = "warning"
        self.btn_precompute.disabled = False
        self.cache_status_html.value = f"<span style='color:#b00020'>Precompute error: {err_text}</span>"
        if hasattr(self, "_sync_config_status"):
            self._sync_config_status()
        self._sync_frame_controls(self._frame_index)

    def _apply_async_precompute_result(self, result: Any, seq: int) -> None:
        if self._is_precompute_cancelled(seq):
            return
        with self._async_lock:
            if int(seq) < int(self._async_seq):
                return
        self._precompute_busy = False
        self._precompute_stale_message = ""
        self._ingest_precompute_result(result)

    def _interrupt_precompute(self) -> None:
        with self._async_lock:
            self._async_cancel_before = int(self._async_seq)
            self._async_pending_job = None
        self._precompute_busy = False
        message = self._precompute_stale_message or "Precompute interrupted. Adjust settings and click Precompute flow."
        self._mark_cache_stale(message)

    def precompute(self) -> None:
        self._queue_async_precompute()

    def _on_precompute_clicked(self, _btn: Any) -> None:
        if self._precompute_busy:
            self._interrupt_precompute()
            return
        self.precompute()

    def _mark_cache_stale(self, message: str) -> None:
        self._cache_valid = False
        if self._precompute_busy:
            self._precompute_stale_message = message
            self.cache_status_html.value = "<span style='color:#666'>Computing all cached frames...</span>"
            self.btn_precompute.description = "Interrupt"
            self.btn_precompute.button_style = "info"
            self.btn_precompute.disabled = False
            self._sync_config_status()
            self._sync_frame_controls(self._frame_index)
            return
        self.btn_precompute.description = "Precompute flow"
        self.btn_precompute.button_style = "warning"
        self.btn_precompute.disabled = False
        self.cache_status_html.value = f"<span style='color:#9a6700'>{message}</span>"
        self._sync_config_status()
        self._sync_frame_controls(self._frame_index)

    def _mark_cache_ready(self, message: str) -> None:
        self._cache_valid = True
        self._precompute_busy = False
        self.btn_precompute.description = "Precompute flow"
        self.btn_precompute.button_style = "success"
        self.btn_precompute.disabled = False
        self.cache_status_html.value = f"<span style='color:#188038'>{message}</span>"
        self._sync_config_status()
        self._sync_frame_controls(self._frame_index)


class PeszekPoyatoDynamicsBaseWidget(_AsyncPrecomputeControlsMixin):
    """Interactive ``go.FigureWidget`` animation of fiber + joint-density dynamics.

    This base mirrors the interaction model that works well in
    ``LMSOpticalDiskBaseWidget``: controls are grouped under the figure,
    expensive simulation results are precomputed into frame payloads, and
    playback only swaps cached arrays into existing Plotly traces.

    * ``Precompute flow`` runs the simulation on a background thread (recording every step) and builds the
      per-frame payload cache. While computing, the button turns light blue and reads ``Interrupt``;
      click it to cancel and try different slider settings.
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
        second_panel: SecondPanelChoice = "velocity",
    ) -> None:
        if widgets is None:
            raise RuntimeError(
                "PeszekPoyatoDynamicsBaseWidget requires ipywidgets (install the 'widgets' extra) "
                "and a live notebook kernel."
            )
        if second_panel not in ("heatmap", "velocity"):
            raise ValueError("second_panel must be 'heatmap' or 'velocity'")
        self.config = config
        self._seed = int(config.seed)
        self._init_preset = _initialization_preset_from_config(config)
        self.width = int(width)
        self.height = int(height)
        self._second_panel: SecondPanelChoice = second_panel

        self._result: SimulationResult | None = None
        self._frame_payloads: list[dict[str, Any]] = []
        self._frame_index = 0
        self._cache_valid = False
        self._updating = False
        self._init_async_precompute_state()
        self._sampled_by_group: list[Array] = []
        self._density_axis: Array | None = None
        self._zmax = 1.0
        self._vmax = 1.0
        self._fiber_trace_count = 0

        fibers = _normalize_fibers(config)
        self._group_names: tuple[str, ...] = tuple(
            spec.name or _shape_name(spec.shape, k) for k, spec in enumerate(fibers)
        )

        self._build_controls()
        self._build_figure()
        self._bind_callbacks()
        self._sync_time_direction_label()
        self._sync_config_status()
        self.layout = widgets.VBox(
            [
                widgets.HTML(self._layout_header_html()),
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

    def _layout_header_html(self) -> str:
        if self._second_panel == "velocity":
            right = (
                "Right: negative-velocity scatter (-x_dot_i) of the same sampled particles (same colors); "
                "near the origin when settled, spread out when moving fast."
            )
        else:
            right = "Right: precomputed joint spatial density."
        return (
            "<b>Peszek--Poyato large-N FFT dynamics</b><br>"
            f"Left: sampled per-omega fiber particles. {right} "
            "Precompute caches every step, then Play/Step/slider swap cached frames in place."
        )

    def _header_html(self) -> str:
        return self._layout_header_html()

    def _build_controls(self) -> None:
        n_fibers = int(max(1, self.config.n_fibers))
        n_per_fiber = max(1, _n_per_fiber_scalar(self.config))
        frame_count = max(0, int(self.config.trajectory_frame_count))

        self.initialization_dropdown = widgets.Dropdown(
            options=[
                ("Mixed shapes", "mixed"),
                ("Gaussian", "gaussian"),
                ("Rings", "ring"),
                ("Lines", "line"),
                ("Spirals", "spiral"),
                ("Squares", "square"),
            ],
            value=self._init_preset,
            description="init",
            layout=widgets.Layout(width="190px"),
        )
        self.initializer_dropdown = widgets.Dropdown(
            options=list(PP_INITIALIZER_OPTIONS),
            value=self.config.initialization_algorithm,
            description="warmup",
            layout=widgets.Layout(width="220px"),
        )
        self.alpha_slider = widgets.FloatSlider(
            value=float(self.config.alpha),
            min=0.0,
            max=2.0,
            step=0.001,
            description="alpha",
            readout_format=".3f",
            continuous_update=False,
            tooltip="PP kernel order: alpha in [0,1) is the classic regime, (1,2] the renormalized W^alpha family (alpha->2 is the -log|x| kernel); alpha=1 is singular.",
            layout=widgets.Layout(width="780px"),
        )
        self.K_slider = widgets.FloatSlider(
            value=float(self.config.K),
            min=0.0,
            max=max(5.0, 4.0 * float(self.config.K)),
            step=0.01,
            description="K",
            readout_format=".2f",
            continuous_update=False,
            tooltip="Coupling strength multiplying the PP interaction force K grad W^alpha * rho.",
            layout=widgets.Layout(width="780px"),
        )
        self.n_fibers_slider = widgets.IntSlider(
            value=n_fibers,
            min=1,
            max=max(40, n_fibers),
            step=1,
            description="omega atoms",
            continuous_update=False,
            layout=widgets.Layout(width="410px"),
        )
        self.n_per_fiber_slider = widgets.IntSlider(
            value=n_per_fiber,
            min=1,
            max=max(5000, n_per_fiber * 4),
            step=1,
            description="N/omega",
            continuous_update=False,
            layout=widgets.Layout(width="410px"),
        )
        self.btn_resample = widgets.Button(
            description="Resample x^0",
            layout=widgets.Layout(width="140px"),
        )
        self.config_status_html = widgets.HTML(value="", layout=widgets.Layout(width="620px"))
        self.time_direction_toggle = widgets.ToggleButton(
            value=self.config.time_direction == "backward",
            description="Time: forward",
            tooltip="Switch between forward dynamics and the time-inverted vector field.",
            layout=widgets.Layout(width="150px"),
        )
        self.frame_cap_slider = widgets.IntSlider(
            value=frame_count,
            min=0,
            max=max(20000, int(self.config.max_steps) + 1, frame_count),
            step=10,
            description="frame cap",
            continuous_update=False,
            layout=widgets.Layout(width="340px"),
        )
        self.frame_cap_slider.tooltip = "0 records every accepted integration step; positive values downsample to a cap."
        self.btn_precompute = widgets.Button(
            description="Precompute flow",
            button_style="warning",
            layout=widgets.Layout(width="140px"),
        )
        self.btn_step = widgets.Button(
            description="Step flow",
            disabled=True,
            layout=widgets.Layout(width="100px"),
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
                widgets.HBox(
                    [
                        self.initialization_dropdown,
                        self.initializer_dropdown,
                        self.n_fibers_slider,
                        self.n_per_fiber_slider,
                        self.btn_resample,
                    ]
                ),
                widgets.HBox([self.alpha_slider]),
                widgets.HBox([self.K_slider]),
                widgets.HBox(
                    [
                        self.config_status_html,
                        self.time_direction_toggle,
                        self.frame_cap_slider,
                    ]
                ),
                widgets.HBox([self.btn_step, self.play, self.btn_precompute, self.cache_status_html]),
                widgets.HBox([self.frame_slider, self.frame_counter]),
            ]
        )

    def _build_figure(self) -> None:
        velocity_panel = self._second_panel == "velocity"
        second_title = (
            "Negative velocity scatter (-x_dot_i = A_rho - omega_i)" if velocity_panel else "Precomputed joint density rho_t"
        )
        second_spec = {"type": "scatter"} if velocity_panel else {"type": "heatmap"}
        fig = go.FigureWidget(
            make_subplots(
                rows=1,
                cols=2,
                specs=[[{"type": "scatter"}, second_spec]],
                subplot_titles=[
                    "Per-omega fiber particle dynamics",
                    second_title,
                ],
                horizontal_spacing=0.08,
            )
        )
        self.fig = fig
        self.tr: dict[str, int] = {}
        self._fiber_trace_count = 0
        self._ensure_fiber_traces(self._group_names)

        r = self.config.domain_radius
        fig.update_xaxes(title_text="x1", range=[-r, r], scaleanchor="y", scaleratio=1, row=1, col=1)
        fig.update_yaxes(title_text="x2", range=[-r, r], row=1, col=1)
        if velocity_panel:
            v = float(self._vmax)
            fig.update_xaxes(title_text="v1", range=[-v, v], row=1, col=2)
            fig.update_yaxes(title_text="v2", range=[-v, v], scaleanchor="x2", scaleratio=1, row=1, col=2)
        else:
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
            fig.update_xaxes(title_text="x1", range=[-r, r], row=1, col=2)
            fig.update_yaxes(title_text="x2", range=[-r, r], scaleanchor="x2", scaleratio=1, row=1, col=2)
        fig.update_layout(
            width=self.width,
            height=self.height,
            template="plotly_white",
            legend=dict(groupclick="togglegroup", itemsizing="constant"),
            margin=dict(l=50, r=30, t=70, b=40),
        )

    def _ensure_fiber_traces(self, group_names: Sequence[str], omega_atoms: Array | None = None) -> None:
        if not hasattr(self, "fig"):
            return
        velocity_panel = self._second_panel == "velocity"
        colors = fiber_colors(self.config, omega_atoms, len(group_names))
        while self._fiber_trace_count < len(group_names):
            k = self._fiber_trace_count
            color = colors[k]
            self.fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode="markers",
                    marker=dict(size=4, color=color, opacity=0.70),
                    name=f"fiber {k + 1}: {group_names[k]}",
                    legendgroup=f"fiber{k}",
                    visible=True,
                ),
                row=1,
                col=1,
            )
            self.tr[f"fiber{k}"] = len(self.fig.data) - 1
            if velocity_panel:
                self.fig.add_trace(
                    go.Scatter(
                        x=[],
                        y=[],
                        mode="markers",
                        marker=dict(size=4, color=color, opacity=0.70),
                        name=f"fiber {k + 1}: {group_names[k]}",
                        legendgroup=f"fiber{k}",
                        showlegend=False,
                        visible=True,
                    ),
                    row=1,
                    col=2,
                )
                self.tr[f"vfiber{k}"] = len(self.fig.data) - 1
            self._fiber_trace_count += 1

        with self.fig.batch_update():
            for k in range(self._fiber_trace_count):
                visible = k < len(group_names)
                for prefix in ("fiber", "vfiber"):
                    key = f"{prefix}{k}"
                    if key not in self.tr:
                        continue
                    trace = self.fig.data[self.tr[key]]
                    if visible:
                        trace.name = f"fiber {k + 1}: {group_names[k]}"
                        trace.legendgroup = f"fiber{k}"
                        trace.marker.color = colors[k]
                        trace.visible = True
                    else:
                        trace.x = []
                        trace.y = []
                        trace.visible = False

    def _bind_callbacks(self) -> None:
        for ctl in (
            self.initialization_dropdown,
            self.initializer_dropdown,
            self.alpha_slider,
            self.K_slider,
            self.n_fibers_slider,
            self.n_per_fiber_slider,
            self.frame_cap_slider,
        ):
            ctl.observe(self._on_control_change, names="value")
        self.time_direction_toggle.observe(self._on_time_direction_toggle, names="value")
        self.btn_resample.on_click(self._on_resample)
        self.btn_precompute.on_click(self._on_precompute_clicked)
        self.btn_step.on_click(self._on_step)
        self.play.observe(self._on_play_tick, names="value")
        self.frame_slider.observe(self._on_frame_slider, names="value")

    def _wire_events(self) -> None:
        self._bind_callbacks()

    def _time_direction(self) -> TimeDirectionChoice:
        return "backward" if bool(self.time_direction_toggle.value) else "forward"

    def _sync_time_direction_label(self) -> None:
        self.time_direction_toggle.description = "Time: backward" if self._time_direction() == "backward" else "Time: forward"

    def _preview_group_names_from_controls(self) -> tuple[str, ...]:
        preset = str(self.initialization_dropdown.value)
        shape_names = _shape_names_for_initialization_preset(preset)
        return tuple(shape_names[k % len(shape_names)] for k in range(int(self.n_fibers_slider.value)))

    def _config_from_controls(self, *, make_animation: bool) -> SimulationConfig:
        preset = str(self.initialization_dropdown.value)
        shape_names = _shape_names_for_initialization_preset(preset)
        return replace(
            self.config,
            alpha=float(self.alpha_slider.value),
            K=float(self.K_slider.value),
            n_fibers=int(self.n_fibers_slider.value),
            n_per_fiber=int(self.n_per_fiber_slider.value),
            fibers=None,
            shape_names=shape_names,
            omega_atoms=None,
            seed=int(self._seed),
            initialization_algorithm=self.initializer_dropdown.value,
            time_direction=self._time_direction(),
            make_animation=bool(make_animation),
            trajectory_frame_count=int(self.frame_cap_slider.value),
        )

    def _sync_config_status(self, result: SimulationResult | None = None) -> None:
        cfg = self._config_from_controls(make_animation=True)
        total = int(cfg.n_fibers) * int(cfg.n_per_fiber)
        backend_text = cfg.backend
        if result is not None:
            backend_text = f"{result.backend}/{result.device}/{result.dtype}"
        residual_text = ""
        if result is not None:
            residual_text = f"; residual RMS={result.rms_residual:.3g}; runtime={result.runtime_seconds:.2f}s"
        frame_text = "all steps" if int(cfg.trajectory_frame_count) <= 0 else f"cap {int(cfg.trajectory_frame_count):,}"
        self.config_status_html.value = (
            "<b>PP config:</b> "
            f"alpha={float(cfg.alpha):.2f}; "
            f"K={float(cfg.K):.2f}; "
            f"omega atoms={int(cfg.n_fibers)}; "
            f"N/omega={int(cfg.n_per_fiber)}; "
            f"N={total:,}; "
            f"init={self.initialization_dropdown.label}; "
            f"warmup={self.initializer_dropdown.label}; "
            f"time={cfg.time_direction}; "
            f"frames={frame_text}; "
            f"backend={backend_text}; "
            f"seed={int(cfg.seed)}"
            f"{residual_text}"
        )

    # -- cache + frame state ------------------------------------------------

    def _prepare_precompute_job(self, job: dict[str, Any]) -> None:
        run_config = job["config"]
        self.config = run_config
        self._group_names = tuple(job["group_names"])
        self._ensure_fiber_traces(self._group_names)
        self._sync_config_status()

    def _capture_precompute_job(self) -> dict[str, Any]:
        run_config = self._config_from_controls(make_animation=True)
        return {
            "config": run_config,
            "group_names": self._preview_group_names_from_controls(),
        }

    def _ingest_precompute_result(self, result: SimulationResult) -> None:
        self._ingest_result(result)

    def _ingest_result(self, result: SimulationResult) -> None:
        if result.trajectory_x is None or result.trajectory_rho is None:
            raise ValueError("result has no trajectory; run with config.make_animation=True")
        self._result = result
        self._group_names = result.initial.group_names
        self._ensure_fiber_traces(self._group_names, result.initial.omega_atoms)
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
        self._sync_config_status(result)
        self._mark_cache_ready(f"Cache ready: {len(self._frame_payloads)} frames.")

    def _build_frame_payloads(self) -> list[dict[str, Any]]:
        result = self._result
        assert result is not None
        trajectory_x = np.asarray(result.trajectory_x, dtype=np.float64)
        trajectory_rho = np.asarray(result.trajectory_rho, dtype=np.float64)
        steps = np.asarray(result.trajectory_steps, dtype=np.int64)
        times = np.asarray(result.trajectory_times, dtype=np.float64)
        frame_count = trajectory_x.shape[0]

        velocity_panel = self._second_panel == "velocity"
        velocity_frames = self._compute_trajectory_velocities(trajectory_x) if velocity_panel else None
        vmax = 1e-6

        payloads: list[dict[str, Any]] = []
        for f in range(frame_count):
            fiber_x: list[Array] = []
            fiber_y: list[Array] = []
            for k in range(len(self._group_names)):
                idx = self._sampled_by_group[k]
                fiber_x.append(trajectory_x[f, idx, 0])
                fiber_y.append(trajectory_x[f, idx, 1])
            payload: dict[str, Any] = {
                "fiber_x": fiber_x,
                "fiber_y": fiber_y,
                "title": _animation_title(self.config, result, int(steps[f]), float(times[f]), f, frame_count),
                "stats": (
                    f"frame {f + 1}/{frame_count}; step={int(steps[f])}; "
                    f"t={float(times[f]):.4g}; time={self.config.time_direction}; "
                    f"final residual RMS={result.rms_residual:.3g}; "
                    f"backend={result.backend}/{result.device}/{result.dtype}"
                ),
            }
            if velocity_panel:
                assert velocity_frames is not None
                vframe = velocity_frames[f]
                vfiber_x: list[Array] = []
                vfiber_y: list[Array] = []
                for k in range(len(self._group_names)):
                    idx = self._sampled_by_group[k]
                    vfiber_x.append(vframe[idx, 0])
                    vfiber_y.append(vframe[idx, 1])
                    if idx.size:
                        vmax = max(vmax, float(np.max(np.abs(vframe[idx]))))
                payload["vfiber_x"] = vfiber_x
                payload["vfiber_y"] = vfiber_y
            else:
                payload["density_z"] = trajectory_rho[f].T
            payloads.append(payload)

        self._vmax = vmax * 1.08 if velocity_panel else self._vmax
        return payloads

    def _compute_trajectory_velocities(self, trajectory_x: Array) -> Array:
        """Per-frame negative PP velocity -x_dot = A_rho(x) - external(x) per particle.

        Recomputes the same affine/projective field that ``run_simulation`` uses,
        from the cached frame positions, then negates it so the right-hand scatter
        shows ``-x_dot_i`` for each displayed particle.
        """

        result = self._result
        assert result is not None
        solver = FFTPeszekPoyato2D(
            self.config.alpha, self.config.K, self.config.grid_size, self.config.domain_radius
        )
        omega = np.asarray(result.initial.omega, dtype=np.float64)
        eps = float(self.config.projective_epsilon)
        use_projective = self.config.external_field == "projective" and eps != 0.0
        frames = np.empty_like(trajectory_x)
        for f in range(trajectory_x.shape[0]):
            xf = solver.clip_inside(trajectory_x[f])
            A, _ = solver.A_at_particles(xf)
            external = solver.projective_external(xf, omega, eps) if use_projective else omega
            frames[f] = A - external
        return frames

    def _apply_static_payload_to_figure(self) -> None:
        if self._second_panel == "velocity":
            v = float(self._vmax)
            with self.fig.batch_update():
                self.fig.update_xaxes(range=[-v, v], row=1, col=2)
                self.fig.update_yaxes(range=[-v, v], row=1, col=2)
            return
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
        velocity_panel = self._second_panel == "velocity"
        with self.fig.batch_update():
            for k in range(len(self._group_names)):
                trace = self.fig.data[self.tr[f"fiber{k}"]]
                trace.x = _plotly_values(payload["fiber_x"][k])
                trace.y = _plotly_values(payload["fiber_y"][k])
                if velocity_panel:
                    vtrace = self.fig.data[self.tr[f"vfiber{k}"]]
                    vtrace.x = _plotly_values(payload["vfiber_x"][k])
                    vtrace.y = _plotly_values(payload["vfiber_y"][k])
            if not velocity_panel:
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
        disabled = self._playback_disabled()
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

    # -- callbacks ----------------------------------------------------------

    def _on_control_change(self, change: dict[str, Any]) -> None:
        if self._updating or change.get("name") != "value":
            return
        owner = change.get("owner")
        if owner in (
            self.initialization_dropdown,
            self.initializer_dropdown,
            self.n_fibers_slider,
            self.n_per_fiber_slider,
            self.alpha_slider,
            self.K_slider,
            self.frame_cap_slider,
        ):
            self._group_names = self._preview_group_names_from_controls()
            self._ensure_fiber_traces(self._group_names)
            self._mark_cache_stale("Parameters changed. Click Interrupt, then Precompute flow.")
            return

    def _on_time_direction_toggle(self, change: dict[str, Any]) -> None:
        if self._updating or change.get("name") != "value":
            return
        self._sync_time_direction_label()
        self._mark_cache_stale("Time direction changed. Click Interrupt, then Precompute flow.")

    def _on_resample(self, _btn: Any) -> None:
        self._seed += 1
        if self._precompute_busy:
            self._mark_cache_stale("Resampled seed. Click Interrupt, then Precompute flow.")
            return
        self.precompute()

    def _on_step(self, _btn: Any) -> None:
        if self._playback_disabled():
            return
        self._set_frame_index((self._frame_index + 1) % len(self._frame_payloads))

    def _on_play_tick(self, change: dict[str, Any]) -> None:
        if self._updating or change.get("name") != "value":
            return
        if self._playback_disabled():
            return
        self._set_frame_index(int(change.get("new", 0)), source=self.play)

    def _on_frame_slider(self, change: dict[str, Any]) -> None:
        if self._updating or change.get("name") != "value":
            return
        if self._playback_disabled():
            return
        self._set_frame_index(int(self.frame_slider.value), source=self.frame_slider)

    def _ipython_display_(self) -> None:  # pragma: no cover - notebook display hook
        from IPython.display import display

        display(self.layout)


class PPDynamicsWidget(PeszekPoyatoDynamicsBaseWidget):
    """Backward-compatible PP widget name."""


class ProjectivePeszekPoyatoDynamicsWidget(PeszekPoyatoDynamicsBaseWidget):
    """PP dynamics widget with an optional projective (c_S) external gauge.

    This subclass is a thin extension of :class:`PeszekPoyatoDynamicsBaseWidget`.
    It inherits the entire control / cache / playback / figure machinery and only
    adds the controls and config wiring needed to select between two external
    gauges of the same fibered transport ray:

    * **affine** (default, identical to the base widget) ::

          dot x_i = omega_i - K grad W^alpha * rho(x_i)

    * **projective** (section 5.2 of the inversive gauge manuscript) replaces the
      affine label ``omega`` by the inversive bracket field
      ``S[p(omega), x]`` with the canonical embedding ``p(omega) = eps * omega`` ::

          dot x_i = E_eps(omega_i, x_i) - K grad W^alpha * rho(x_i),
          E_eps(omega, x) = eps^{-1} S[eps*omega, x]
                          = (omega - eps |omega|^2 x)
                            / (1 - 2 eps <omega, x> + eps^2 |omega|^2 |x|^2).

      The affine-tangent corollary ``eps^{-1} S[eps*omega, x] -> omega`` means the
      ``eps`` slider is a single deformation knob: ``eps = 0`` reproduces the flat
      Peszek--Poyato field exactly, ``eps > 0`` bends the external drift along the
      projective bracket ray.

    The actual change of dynamics lives entirely in the (inherited) precompute
    path: this widget only sets ``external_field`` / ``projective_epsilon`` on the
    :class:`SimulationConfig`, and ``run_simulation`` dispatches the velocity to
    the projective bracket when requested.  Any future control or rendering hook
    added to the base class is therefore inherited automatically; only the gauge
    selection is overridden here.
    """

    def _layout_header_html(self) -> str:
        return (
            "<b>Peszek--Poyato dynamics &mdash; affine / projective (c<sub>S</sub>) external gauge</b><br>"
            "Affine: x&#775;<sub>i</sub>=&omega;<sub>i</sub>&minus;K&nabla;W<sup>&alpha;</sup>&lowast;&rho;. "
            "Projective: x&#775;<sub>i</sub>=&epsilon;<sup>&minus;1</sup>S[&epsilon;&omega;<sub>i</sub>,x<sub>i</sub>]&minus;K&nabla;W<sup>&alpha;</sup>&lowast;&rho; "
            "(&epsilon;&rarr;0 recovers the affine field). "
            "Left: per-omega fiber particles. Right: precomputed joint density."
        )

    def _build_controls(self) -> None:
        super()._build_controls()
        default_eps = float(self.config.projective_epsilon)
        if default_eps <= 0.0:
            default_eps = 0.30
        self.external_field_toggle = widgets.ToggleButton(
            value=self.config.external_field == "projective",
            description="Field: affine",
            tooltip="Switch the external drive between the affine omega field and the projective S[p(omega), x] field.",
            layout=widgets.Layout(width="170px"),
        )
        self.projective_epsilon_slider = widgets.FloatSlider(
            value=default_eps,
            min=0.0,
            max=2.0,
            step=0.01,
            description="proj eps",
            readout_format=".2f",
            continuous_update=False,
            tooltip="Projective embedding scale eps in p(omega)=eps*omega; eps->0 is the affine limit.",
            layout=widgets.Layout(width="560px"),
        )
        self.projective_status_html = widgets.HTML(value="", layout=widgets.Layout(width="620px"))
        self.controls.children = tuple(self.controls.children) + (
            widgets.HBox(
                [
                    self.external_field_toggle,
                    self.projective_epsilon_slider,
                    self.projective_status_html,
                ]
            ),
        )
        self._update_projective_enabled()

    def _bind_callbacks(self) -> None:
        super()._bind_callbacks()
        self.external_field_toggle.observe(self._on_projective_change, names="value")
        self.projective_epsilon_slider.observe(self._on_projective_change, names="value")

    def _external_field_choice(self) -> ExternalFieldChoice:
        return "projective" if bool(self.external_field_toggle.value) else "affine"

    def _config_from_controls(self, *, make_animation: bool) -> SimulationConfig:
        cfg = super()._config_from_controls(make_animation=make_animation)
        return replace(
            cfg,
            external_field=self._external_field_choice(),
            projective_epsilon=float(self.projective_epsilon_slider.value),
        )

    def _update_projective_enabled(self) -> None:
        self.projective_epsilon_slider.disabled = self._external_field_choice() != "projective"

    def _sync_projective_label(self) -> None:
        mode = self._external_field_choice()
        eps = float(self.projective_epsilon_slider.value)
        self.external_field_toggle.description = "Field: projective" if mode == "projective" else "Field: affine"
        if mode == "projective" and eps != 0.0:
            self.projective_status_html.value = (
                "<b>External gauge:</b> projective; "
                f"E<sub>&epsilon;</sub>(&omega;,x)=&epsilon;<sup>&minus;1</sup>S[&epsilon;&omega;,x], "
                f"&epsilon;={eps:.2f}"
            )
        else:
            self.projective_status_html.value = (
                "<b>External gauge:</b> affine; E(&omega;,x)=&omega; (flat Peszek--Poyato field)"
            )

    def _sync_config_status(self, result: SimulationResult | None = None) -> None:
        super()._sync_config_status(result)
        self._sync_projective_label()

    def _on_projective_change(self, change: dict[str, Any]) -> None:
        if self._updating or change.get("name") != "value":
            return
        self._update_projective_enabled()
        self._sync_projective_label()
        self._mark_cache_stale("External gauge changed. Click Interrupt, then Precompute flow.")


class FiniteHorizonGaugeAveragedPeszekPoyatoDynamicsWidget(PeszekPoyatoDynamicsBaseWidget):
    """Widget for finite-horizon gauge-averaged Peszek--Poyato dynamics.

    The physical prediction horizon ``tau`` deforms the continuum vector field:

        x_dot_i = omega_i
                  - 1/2 A_rho(x_i)
                  - 1/2 A_{rho_tau}(x_i + tau (omega_i - A_rho(x_i))).

    ``tau`` is not the numerical timestep.  The precompute path calls
    :func:`run_finite_horizon_gauge_averaged_simulation`, which still defaults to
    adaptive RK2, so users can keep integration ``dt`` much smaller than the
    prediction horizon.
    """

    def _layout_header_html(self) -> str:
        return (
            "<b>Finite-Horizon Gauge-Averaged Peszek--Poyato dynamics</b><br>"
            "Agents align to the average of the present gauge field and the field sampled at a finite predicted point. "
            "The horizon &tau; is a model parameter, not the numerical integration step."
        )

    def _build_controls(self) -> None:
        super()._build_controls()
        tau_default = max(0.0, float(self.config.prediction_horizon_tau))
        self.prediction_horizon_slider = widgets.FloatSlider(
            value=tau_default,
            min=0.0,
            max=max(0.30, 4.0 * tau_default),
            step=0.001,
            description="tau",
            readout_format=".3f",
            continuous_update=False,
            tooltip="Physical prediction horizon tau in the two-point gauge average; keep numerical dt smaller than tau.",
            layout=widgets.Layout(width="560px"),
        )
        self.prediction_status_html = widgets.HTML(value="", layout=widgets.Layout(width="620px"))
        self.controls.children = tuple(self.controls.children) + (
            widgets.HBox([self.prediction_horizon_slider, self.prediction_status_html]),
        )
        self._sync_prediction_label()

    def _bind_callbacks(self) -> None:
        super()._bind_callbacks()
        self.prediction_horizon_slider.observe(self._on_prediction_horizon_change, names="value")

    def _config_from_controls(self, *, make_animation: bool) -> SimulationConfig:
        cfg = super()._config_from_controls(make_animation=make_animation)
        return replace(
            cfg,
            prediction_horizon_tau=float(self.prediction_horizon_slider.value),
            external_field="affine",
            projective_epsilon=0.0,
        )

    def _run_precompute_job(self, job: dict[str, Any]) -> SimulationResult:
        seq = int(job["seq"])
        return run_finite_horizon_gauge_averaged_simulation(
            job["config"],
            cancel_check=lambda: self._is_precompute_cancelled(seq),
        )

    def _compute_trajectory_velocities(self, trajectory_x: Array) -> Array:
        result = self._result
        assert result is not None
        solver = FFTPeszekPoyato2D(
            self.config.alpha,
            self.config.K,
            self.config.grid_size,
            self.config.domain_radius,
        )
        omega = np.asarray(result.initial.omega, dtype=np.float64)
        tau = float(self.config.prediction_horizon_tau)
        frames = np.empty_like(trajectory_x)
        for f in range(trajectory_x.shape[0]):
            xf = solver.clip_inside(trajectory_x[f])
            v, _, _ = finite_horizon_gauge_average_field(solver, xf, omega, tau)
            frames[f] = -v
        return frames

    def _sync_prediction_label(self) -> None:
        tau = float(self.prediction_horizon_slider.value)
        dt = float(self.config.dt)
        ratio = dt / tau if tau > 0 else float("inf")
        if tau > 0:
            self.prediction_status_html.value = (
                "<b>Finite horizon:</b> "
                f"&tau;={tau:.4g}; numerical dt/&tau;={ratio:.3g}; "
                "V=&omega;-&frac12;A<sub>&rho;</sub>(x)-&frac12;A<sub>&rho;<sub>&tau;</sub></sub>(P<sub>&tau;</sub>)"
            )
        else:
            self.prediction_status_html.value = "<b>Finite horizon:</b> &tau;=0, recovering ordinary PP."

    def _sync_config_status(self, result: SimulationResult | None = None) -> None:
        super()._sync_config_status(result)
        self._sync_prediction_label()

    def _on_prediction_horizon_change(self, change: dict[str, Any]) -> None:
        if self._updating or change.get("name") != "value":
            return
        self._sync_prediction_label()
        self._mark_cache_stale("Prediction horizon changed. Click Interrupt, then Precompute flow.")


class HamiltonianExponentPeszekPoyatoDynamicsWidget(PeszekPoyatoDynamicsBaseWidget):
    """PP dynamics widget with an independent Hamiltonian residual exponent.

    The interaction field remains the ordinary FFT Peszek--Poyato
    ``A_rho = K grad W^alpha * rho``.  This subclass only changes the residual
    clock in the particle velocity:

        R_i = omega_i - A_rho(x_i),
        x_dot_i = (|R_i|^2 + epsH^2)^((q - 2) / 2) R_i.

    Thus ``q=2`` recovers the base PP particle dynamics, while ``q<2`` changes
    the transient fiberwise clock without changing the zero-residual graph.
    """

    def __init__(
        self,
        config: SimulationConfig,
        result: SimulationResult | None = None,
        *,
        width: int = 1450,
        height: int = 980,
        second_panel: SecondPanelChoice = "velocity",
    ) -> None:
        self._hamiltonian_velocity_frames: Array | None = None
        self._hamiltonian_timeline: dict[str, Array] | None = None
        self._energy_cursor_y: list[float] = [0.0, 1.0]
        self._dissipation_cursor_y: list[float] = [0.0, 1.0]
        super().__init__(config, result=result, width=width, height=height, second_panel=second_panel)

    def _layout_header_html(self) -> str:
        return (
            "<b>Hamiltonian-exponent Peszek--Poyato dynamics</b><br>"
            "The FFT PP interaction exponent &alpha; is unchanged; q changes the local residual clock "
            "x&#775;<sub>i</sub>=(|R<sub>i</sub>|<sup>2</sup>+epsH<sup>2</sup>)<sup>(q-2)/2</sup>R<sub>i</sub>, "
            "R<sub>i</sub>=&omega;<sub>i</sub>&minus;A<sub>&rho;</sub>(x<sub>i</sub>)."
        )

    def _build_controls(self) -> None:
        super()._build_controls()
        q_default = float(np.clip(float(self.config.hamiltonian_q), 0.0, 2.0))
        eps_default = max(0.0, float(self.config.hamiltonian_epsH))
        eps_log = float(np.log10(eps_default))
        self.hamiltonian_q_slider = widgets.FloatSlider(
            value=q_default,
            min=0.0,
            max=2.0,
            step=0.01,
            description="q",
            readout_format=".2f",
            continuous_update=False,
            tooltip="Hamiltonian exponent q in the residual clock; q=2 is ordinary PP.",
            layout=widgets.Layout(width="430px"),
        )
        self.hamiltonian_epsH_slider = widgets.FloatLogSlider(
            value=eps_default,
            base=10,
            min=min(0.0, np.floor(eps_log) - 1.0),
            max=max(-1.0, np.ceil(eps_log) + 1.0),
            step=0.1,
            description="epsH",
            readout_format=".1e",
            continuous_update=False,
            tooltip="Positive regularization in the Hamiltonian residual clock.",
            layout=widgets.Layout(width="430px"),
        )
        self.hamiltonian_status_html = widgets.HTML(value="", layout=widgets.Layout(width="760px"))
        self.controls.children = tuple(self.controls.children) + (
            widgets.HBox(
                [
                    self.hamiltonian_q_slider,
                    self.hamiltonian_epsH_slider,
                    self.hamiltonian_status_html,
                ]
            ),
        )
        self._sync_hamiltonian_label()

    def _build_figure(self) -> None:
        velocity_panel = self._second_panel == "velocity"
        second_title = "Negative q-velocity scatter (-x_dot_i)" if velocity_panel else "Precomputed joint density rho_t"
        second_spec = {"type": "scatter"} if velocity_panel else {"type": "heatmap"}
        fig = go.FigureWidget(
            make_subplots(
                rows=3,
                cols=2,
                specs=[
                    [{"type": "scatter"}, second_spec],
                    [{"type": "scatter", "colspan": 2}, None],
                    [{"type": "scatter", "colspan": 2}, None],
                ],
                subplot_titles=[
                    "Per-omega fiber particle dynamics",
                    second_title,
                    self._energy_subplot_title(),
                    self._dissipation_subplot_title(),
                ],
                row_heights=[0.58, 0.19, 0.23],
                horizontal_spacing=0.08,
                vertical_spacing=0.09,
            )
        )
        self.fig = fig
        self.tr: dict[str, int] = {}
        self._fiber_trace_count = 0
        self._ensure_fiber_traces(self._group_names)

        r = self.config.domain_radius
        fig.update_xaxes(title_text="x1", range=[-r, r], scaleanchor="y", scaleratio=1, row=1, col=1)
        fig.update_yaxes(title_text="x2", range=[-r, r], row=1, col=1)
        if velocity_panel:
            v = float(self._vmax)
            fig.update_xaxes(title_text="v1", range=[-v, v], row=1, col=2)
            fig.update_yaxes(title_text="v2", range=[-v, v], scaleanchor="x2", scaleratio=1, row=1, col=2)
        else:
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
            fig.update_xaxes(title_text="x1", range=[-r, r], row=1, col=2)
            fig.update_yaxes(title_text="x2", range=[-r, r], scaleanchor="x2", scaleratio=1, row=1, col=2)

        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="E actual", line=dict(color="#1f77b4", width=2)), row=2, col=1)
        self.tr["energy_actual"] = len(fig.data) - 1
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="E predicted", line=dict(color="#d62728", width=2)), row=2, col=1)
        self.tr["energy_pred"] = len(fig.data) - 1
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                name="energy cursor",
                line=dict(color="#444444", width=1, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        self.tr["energy_cursor"] = len(fig.data) - 1

        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="D theory", line=dict(color="#2ca02c", width=2)), row=3, col=1)
        self.tr["d_theory"] = len(fig.data) - 1
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="D empirical", line=dict(color="#ff7f0e", width=2)), row=3, col=1)
        self.tr["d_emp"] = len(fig.data) - 1
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                name="D2 reference",
                line=dict(color="#7f7f7f", width=1.5, dash="dash"),
            ),
            row=3,
            col=1,
        )
        self.tr["d2"] = len(fig.data) - 1
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                name="dissipation cursor",
                line=dict(color="#444444", width=1, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=3,
            col=1,
        )
        self.tr["dissipation_cursor"] = len(fig.data) - 1

        fig.update_xaxes(title_text="time", row=2, col=1)
        fig.update_yaxes(title_text="E_N", row=2, col=1)
        fig.update_xaxes(title_text="time", row=3, col=1)
        fig.update_yaxes(title_text="rate", row=3, col=1)
        fig.update_layout(
            width=self.width,
            height=self.height,
            template="plotly_white",
            legend=dict(groupclick="togglegroup", itemsizing="constant"),
            margin=dict(l=55, r=35, t=78, b=45),
        )

    def _bind_callbacks(self) -> None:
        super()._bind_callbacks()
        self.hamiltonian_q_slider.observe(self._on_hamiltonian_change, names="value")
        self.hamiltonian_epsH_slider.observe(self._on_hamiltonian_change, names="value")

    def _config_from_controls(self, *, make_animation: bool) -> SimulationConfig:
        cfg = super()._config_from_controls(make_animation=make_animation)
        return replace(
            cfg,
            hamiltonian_q=float(self.hamiltonian_q_slider.value),
            hamiltonian_epsH=float(self.hamiltonian_epsH_slider.value),
            external_field="affine",
            projective_epsilon=0.0,
        )

    def _run_precompute_job(self, job: dict[str, Any]) -> SimulationResult:
        seq = int(job["seq"])
        return run_hamiltonian_exponent_simulation(
            job["config"],
            cancel_check=lambda: self._is_precompute_cancelled(seq),
        )

    def _build_frame_payloads(self) -> list[dict[str, Any]]:
        result = self._result
        assert result is not None
        trajectory_x = np.asarray(result.trajectory_x, dtype=np.float64)
        self._hamiltonian_velocity_frames, self._hamiltonian_timeline = self._compute_hamiltonian_frame_diagnostics(
            trajectory_x
        )
        payloads = super()._build_frame_payloads()
        timeline = self._hamiltonian_timeline
        if timeline is not None:
            energy = timeline["energy_actual"]
            d_theory = timeline["D_theory"]
            d_emp = timeline["D_emp"]
            for f, payload in enumerate(payloads):
                emp_text = f"{d_emp[f]:.4g}" if np.isfinite(d_emp[f]) else "n/a"
                payload["stats"] = (
                    f"{payload['stats']}; q={float(self.config.hamiltonian_q):.2f}; "
                    f"epsH={float(self.config.hamiltonian_epsH):.1e}; "
                    f"E={energy[f]:.5g}; Dq={d_theory[f]:.4g}; Demp={emp_text}"
                )
        return payloads

    def _compute_trajectory_velocities(self, trajectory_x: Array) -> Array:
        frames = self._hamiltonian_velocity_frames
        if frames is not None and frames.shape == trajectory_x.shape:
            return frames
        frames, self._hamiltonian_timeline = self._compute_hamiltonian_frame_diagnostics(trajectory_x)
        self._hamiltonian_velocity_frames = frames
        return frames

    def _compute_hamiltonian_frame_diagnostics(self, trajectory_x: Array) -> tuple[Array, dict[str, Array]]:
        result = self._result
        assert result is not None
        solver = FFTPeszekPoyato2D(
            self.config.alpha,
            self.config.K,
            self.config.grid_size,
            self.config.domain_radius,
        )
        omega = np.asarray(result.initial.omega, dtype=np.float64)
        times = np.asarray(result.trajectory_times, dtype=np.float64)
        q = float(self.config.hamiltonian_q)
        epsH = float(self.config.hamiltonian_epsH)
        exponent = 0.5 * (q - 2.0)
        time_sign = -1.0 if self.config.time_direction == "backward" else 1.0
        frame_count = trajectory_x.shape[0]
        velocity_frames = np.empty_like(trajectory_x)
        energy = np.empty(frame_count, dtype=np.float64)
        d_forward = np.empty(frame_count, dtype=np.float64)
        d2_forward = np.empty(frame_count, dtype=np.float64)
        r_l2 = np.empty(frame_count, dtype=np.float64)
        r_lq = np.empty(frame_count, dtype=np.float64)
        r_max = np.empty(frame_count, dtype=np.float64)
        r_min_nonzero = np.empty(frame_count, dtype=np.float64)
        active_fraction = np.empty(frame_count, dtype=np.float64)
        clock_mean = np.empty(frame_count, dtype=np.float64)
        clock_max = np.empty(frame_count, dtype=np.float64)
        clock_min = np.empty(frame_count, dtype=np.float64)
        threshold = max(float(epsH), 1.0e-12)

        for f in range(frame_count):
            xf = solver.clip_inside(trajectory_x[f])
            A, _ = solver.A_at_particles(xf)
            residual = omega - A
            r2 = np.sum(residual * residual, axis=1)
            r_abs = np.sqrt(r2)
            clock = np.power(r2 + epsH * epsH, exponent)
            velocity = clock[:, None] * residual
            velocity_frames[f] = -velocity
            energy[f] = _pp_empirical_energy_from_grid(solver, xf, omega)
            d_forward[f] = float(np.mean(clock * r2))
            d2_forward[f] = float(np.mean(r2))
            r_l2[f] = float(np.sqrt(np.mean(r2)))
            if q > 0.0:
                r_lq[f] = float(np.power(np.mean(np.power(r_abs, q)), 1.0 / q))
            else:
                r_lq[f] = np.nan
            r_max[f] = float(np.max(r_abs))
            active = r_abs > threshold
            r_min_nonzero[f] = float(np.min(r_abs[active])) if np.any(active) else 0.0
            active_fraction[f] = float(np.mean(active))
            clock_mean[f] = float(np.mean(clock))
            clock_max[f] = float(np.max(clock))
            clock_min[f] = float(np.min(clock))

        d_theory = time_sign * d_forward
        d2 = time_sign * d2_forward
        d_emp = np.full(frame_count, np.nan, dtype=np.float64)
        if frame_count > 1:
            dt = np.diff(times)
            valid = dt > 0.0
            d_emp_tail = d_emp[1:]
            d_emp_tail[valid] = -np.diff(energy)[valid] / dt[valid]

        e_pred = np.empty(frame_count, dtype=np.float64)
        if frame_count:
            e_pred[0] = energy[0]
            for f in range(1, frame_count):
                dt = max(0.0, float(times[f] - times[f - 1]))
                e_pred[f] = e_pred[f - 1] - dt * d_theory[f - 1]

        return velocity_frames, {
            "time": times,
            "energy_actual": energy,
            "energy_pred": e_pred,
            "D_theory": d_theory,
            "D_emp": d_emp,
            "D2": d2,
            "R_l2": r_l2,
            "R_lq": r_lq,
            "R_max": r_max,
            "R_min_nonzero": r_min_nonzero,
            "active_fraction": active_fraction,
            "clock_mean": clock_mean,
            "clock_max": clock_max,
            "clock_min": clock_min,
        }

    def _apply_static_payload_to_figure(self) -> None:
        super()._apply_static_payload_to_figure()
        timeline = self._hamiltonian_timeline
        if timeline is None:
            return
        t = timeline["time"]
        self._energy_cursor_y = list(self._finite_range(timeline["energy_actual"], timeline["energy_pred"]))
        self._dissipation_cursor_y = list(self._finite_range(timeline["D_theory"], timeline["D_emp"], timeline["D2"]))
        with self.fig.batch_update():
            self.fig.data[self.tr["energy_actual"]].x = _plotly_values(t)
            self.fig.data[self.tr["energy_actual"]].y = _plotly_values(timeline["energy_actual"])
            self.fig.data[self.tr["energy_pred"]].x = _plotly_values(t)
            self.fig.data[self.tr["energy_pred"]].y = _plotly_values(timeline["energy_pred"])
            self.fig.data[self.tr["d_theory"]].x = _plotly_values(t)
            self.fig.data[self.tr["d_theory"]].y = _plotly_values(timeline["D_theory"])
            self.fig.data[self.tr["d_emp"]].x = _plotly_values(t)
            self.fig.data[self.tr["d_emp"]].y = _plotly_values(timeline["D_emp"])
            self.fig.data[self.tr["d2"]].x = _plotly_values(t)
            self.fig.data[self.tr["d2"]].y = _plotly_values(timeline["D2"])
            self.fig.update_yaxes(range=self._energy_cursor_y, row=2, col=1)
            self.fig.update_yaxes(range=self._dissipation_cursor_y, row=3, col=1)
            self._update_timeline_subplot_titles()

    def _apply_cached_frame(self, frame_idx: int) -> None:
        super()._apply_cached_frame(frame_idx)
        self._apply_timeline_cursor(frame_idx)

    def _apply_timeline_cursor(self, frame_idx: int) -> None:
        timeline = self._hamiltonian_timeline
        if timeline is None or len(timeline["time"]) == 0:
            return
        idx = int(np.clip(int(frame_idx), 0, len(timeline["time"]) - 1))
        time_value = float(timeline["time"][idx])
        with self.fig.batch_update():
            self.fig.data[self.tr["energy_cursor"]].x = [time_value, time_value]
            self.fig.data[self.tr["energy_cursor"]].y = self._energy_cursor_y
            self.fig.data[self.tr["dissipation_cursor"]].x = [time_value, time_value]
            self.fig.data[self.tr["dissipation_cursor"]].y = self._dissipation_cursor_y

    def _sync_config_status(self, result: SimulationResult | None = None) -> None:
        super()._sync_config_status(result)
        self._sync_hamiltonian_label()

    def _sync_hamiltonian_label(self) -> None:
        q = float(self.hamiltonian_q_slider.value)
        epsH = float(self.hamiltonian_epsH_slider.value)
        self.hamiltonian_status_html.value = (
            "<b>Hamiltonian clock:</b> "
            f"q={q:.2f}; epsH={epsH:.1e}; "
            "s<sub>i</sub>=(|R<sub>i</sub>|<sup>2</sup>+epsH<sup>2</sup>)<sup>(q-2)/2</sup>"
        )
        self._update_timeline_subplot_titles()

    def _on_hamiltonian_change(self, change: dict[str, Any]) -> None:
        if self._updating or change.get("name") != "value":
            return
        self._sync_hamiltonian_label()
        self._mark_cache_stale("Hamiltonian exponent changed. Click Interrupt, then Precompute flow.")

    def _energy_subplot_title(self) -> str:
        alpha = float(self.alpha_slider.value) if hasattr(self, "alpha_slider") else float(self.config.alpha)
        q = float(self.hamiltonian_q_slider.value) if hasattr(self, "hamiltonian_q_slider") else float(self.config.hamiltonian_q)
        epsH = (
            float(self.hamiltonian_epsH_slider.value)
            if hasattr(self, "hamiltonian_epsH_slider")
            else float(self.config.hamiltonian_epsH)
        )
        return f"Energy: alpha={alpha:.2f}, q={q:.2f}, epsH={epsH:.1e}"

    def _dissipation_subplot_title(self) -> str:
        alpha = float(self.alpha_slider.value) if hasattr(self, "alpha_slider") else float(self.config.alpha)
        q = float(self.hamiltonian_q_slider.value) if hasattr(self, "hamiltonian_q_slider") else float(self.config.hamiltonian_q)
        epsH = (
            float(self.hamiltonian_epsH_slider.value)
            if hasattr(self, "hamiltonian_epsH_slider")
            else float(self.config.hamiltonian_epsH)
        )
        return (
            f"Dissipation: Dq = mean |R|^2(|R|^2+epsH^2)^((q-2)/2), "
            f"alpha={alpha:.2f}, q={q:.2f}, epsH={epsH:.1e}"
        )

    def _update_timeline_subplot_titles(self) -> None:
        if not hasattr(self, "fig"):
            return
        annotations = self.fig.layout.annotations
        if len(annotations) >= 4:
            annotations[2].text = self._energy_subplot_title()
            annotations[3].text = self._dissipation_subplot_title()

    @staticmethod
    def _finite_range(*arrays: Array) -> tuple[float, float]:
        finite_parts = []
        for arr in arrays:
            values = np.asarray(arr, dtype=np.float64)
            finite = values[np.isfinite(values)]
            if finite.size:
                finite_parts.append(finite)
        if not finite_parts:
            return 0.0, 1.0
        values = np.concatenate(finite_parts)
        lo = float(np.min(values))
        hi = float(np.max(values))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return 0.0, 1.0
        if abs(hi - lo) < 1.0e-12:
            pad = max(1.0e-6, 0.05 * max(abs(lo), 1.0))
        else:
            pad = 0.06 * abs(hi - lo)
        return lo - pad, hi + pad


class PeszekPoyatoContinuousDensityWidget(_AsyncPrecomputeControlsMixin):
    """Interactive heatmap playback for continuous-density PP simulations."""

    _precompute_worker_thread_name = "pp-density-precompute-worker"

    _PANEL_OPTIONS: tuple[tuple[str, ContinuousDensityPanelChoice], ...] = (
        ("Marginal rho", "rho"),
        ("Fiber density r_k", "r_fiber"),
        ("Velocity magnitude", "velocity_mag"),
        ("div A_rho", "div_A"),
    )

    def __init__(
        self,
        config: SimulationConfig,
        result: DensitySimulationResult | None = None,
        *,
        width: int = 900,
        height: int = 720,
    ) -> None:
        if widgets is None:
            raise RuntimeError(
                "PeszekPoyatoContinuousDensityWidget requires ipywidgets (install the 'widgets' extra) "
                "and a live notebook kernel."
            )
        self.config = config
        self._seed = int(config.seed)
        self.width = int(width)
        self.height = int(height)

        self._result: DensitySimulationResult | None = None
        self._frame_payloads: list[dict[str, Any]] = []
        self._frame_index = 0
        self._cache_valid = False
        self._updating = False
        self._init_async_precompute_state()
        self._density_axis: Array | None = None
        self._zmax = 1.0
        self._solver: FFTPeszekPoyatoDensity2D | None = None

        fibers = _normalize_fibers(config)
        self._group_names: tuple[str, ...] = tuple(
            spec.name or _shape_name(spec.shape, k) for k, spec in enumerate(fibers)
        )

        self._build_controls()
        self._build_figure()
        self._bind_callbacks()
        self._sync_config_status()
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
            self._mark_cache_stale("Click Precompute to run the density simulation and cache frames.")

    def _header_html(self) -> str:
        return (
            "<b>Peszek--Poyato continuous-density dynamics</b><br>"
            "Precompute runs the explicit finite-volume entropic solver; playback swaps cached heatmaps."
        )

    def _build_controls(self) -> None:
        n_fibers = int(max(1, self.config.n_fibers))
        frame_count = max(0, int(self.config.trajectory_frame_count))

        self.alpha_slider = widgets.FloatSlider(
            value=float(self.config.alpha),
            min=0.0,
            max=0.999,
            step=0.001,
            description="alpha",
            readout_format=".3f",
            continuous_update=False,
            layout=widgets.Layout(width="780px"),
        )
        self.K_slider = widgets.FloatSlider(
            value=float(self.config.K),
            min=0.0,
            max=max(5.0, 4.0 * float(self.config.K)),
            step=0.01,
            description="K",
            readout_format=".2f",
            continuous_update=False,
            layout=widgets.Layout(width="780px"),
        )
        self.eps_entropy_slider = widgets.FloatSlider(
            value=float(self.config.eps_entropy),
            min=0.0,
            max=2.0,
            step=0.01,
            description="eps",
            readout_format=".3f",
            continuous_update=False,
            layout=widgets.Layout(width="780px"),
        )
        self.n_fibers_slider = widgets.IntSlider(
            value=n_fibers,
            min=1,
            max=max(20, n_fibers),
            step=1,
            description="omega atoms",
            continuous_update=False,
            layout=widgets.Layout(width="410px"),
        )
        self.fiber_dropdown = widgets.Dropdown(
            options=[(f"fiber {k + 1}: {name}", k) for k, name in enumerate(self._group_names)],
            value=0,
            description="fiber",
            layout=widgets.Layout(width="260px"),
        )
        self.panel_dropdown = widgets.Dropdown(
            options=list(self._PANEL_OPTIONS),
            value="rho",
            description="panel",
            layout=widgets.Layout(width="260px"),
        )
        self.dynamic_zoom_toggle = widgets.ToggleButton(
            value=bool(self.config.density_dynamic_zoom),
            description="Zoom: on",
            tooltip="Crop/resample heatmaps around active density support (display-only).",
            layout=widgets.Layout(width="120px"),
        )
        backend_g = int(self.config.grid_size)
        display_min = min(32, max(4, backend_g))
        self.display_grid_slider = widgets.IntSlider(
            value=max(display_min, min(int(self.config.density_display_grid_size), backend_g)),
            min=display_min,
            max=backend_g,
            step=8,
            description="display px",
            continuous_update=False,
            layout=widgets.Layout(width="340px"),
        )
        self.heatmap_smooth_toggle = widgets.ToggleButton(
            value=bool(self.config.density_heatmap_smoothing),
            description="Smooth: on",
            tooltip="Plotly heatmap z-smoothing for cleaner playback.",
            layout=widgets.Layout(width="120px"),
        )
        self.frame_cap_slider = widgets.IntSlider(
            value=frame_count,
            min=0,
            max=max(20000, int(self.config.max_steps) + 1, frame_count),
            step=10,
            description="frame cap",
            continuous_update=False,
            layout=widgets.Layout(width="340px"),
        )
        self.btn_precompute = widgets.Button(
            description="Precompute flow",
            button_style="warning",
            layout=widgets.Layout(width="140px"),
        )
        self.btn_step = widgets.Button(
            description="Step flow",
            disabled=True,
            layout=widgets.Layout(width="100px"),
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
        self.config_status_html = widgets.HTML(value="", layout=widgets.Layout(width="620px"))
        self.stats_html = widgets.HTML(value="")
        self.controls = widgets.VBox(
            [
                widgets.HBox([self.alpha_slider]),
                widgets.HBox([self.K_slider, self.eps_entropy_slider]),
                widgets.HBox([self.n_fibers_slider, self.fiber_dropdown, self.panel_dropdown]),
                widgets.HBox(
                    [
                        self.dynamic_zoom_toggle,
                        self.display_grid_slider,
                        self.heatmap_smooth_toggle,
                    ]
                ),
                widgets.HBox([self.config_status_html, self.frame_cap_slider]),
                widgets.HBox([self.btn_step, self.play, self.btn_precompute, self.cache_status_html]),
                widgets.HBox([self.frame_slider, self.frame_counter]),
            ]
        )

    def _build_figure(self) -> None:
        self.fig = go.FigureWidget(
            data=[
                go.Heatmap(
                    x=[],
                    y=[],
                    z=[[0.0]],
                    colorscale="Viridis",
                    zmin=0.0,
                    zmax=1.0,
                    zsmooth="best" if self.config.density_heatmap_smoothing else False,
                    colorbar=dict(title="density"),
                )
            ],
            layout=go.Layout(
                width=self.width,
                height=self.height,
                template="plotly_white",
                title="Continuous-density PP field",
                margin=dict(l=50, r=30, t=70, b=40),
            ),
        )
        r = self.config.domain_radius
        self.fig.update_xaxes(title_text="x1", range=[-r, r])
        self.fig.update_yaxes(title_text="x2", range=[-r, r], scaleanchor="x", scaleratio=1)

    def _bind_callbacks(self) -> None:
        for control in (
            self.alpha_slider,
            self.K_slider,
            self.eps_entropy_slider,
            self.n_fibers_slider,
            self.fiber_dropdown,
            self.panel_dropdown,
            self.frame_cap_slider,
            self.dynamic_zoom_toggle,
            self.display_grid_slider,
            self.heatmap_smooth_toggle,
        ):
            control.observe(self._on_control_change, names="value")
        self.btn_precompute.on_click(self._on_precompute_clicked)
        self.btn_step.on_click(self._on_step)
        self.play.observe(self._on_play_tick, names="value")
        self.frame_slider.observe(self._on_frame_slider, names="value")

    def _sync_display_grid_slider_bounds(self) -> None:
        backend_g = int(self.config.grid_size)
        display_min = min(32, max(4, backend_g))
        display_max = max(display_min, backend_g)
        display_value = int(np.clip(int(self.display_grid_slider.value), display_min, backend_g))
        self._updating = True
        try:
            if int(self.display_grid_slider.min) != display_min:
                self.display_grid_slider.min = display_min
            if int(self.display_grid_slider.max) != display_max:
                self.display_grid_slider.max = display_max
            if int(self.display_grid_slider.value) != display_value:
                self.display_grid_slider.value = display_value
        finally:
            self._updating = False

    def _dynamic_zoom_enabled(self) -> bool:
        return bool(self.dynamic_zoom_toggle.value)

    def _sync_zoom_toggle_labels(self) -> None:
        self.dynamic_zoom_toggle.description = "Zoom: on" if self._dynamic_zoom_enabled() else "Zoom: off"
        self.heatmap_smooth_toggle.description = "Smooth: on" if bool(self.heatmap_smooth_toggle.value) else "Smooth: off"

    def _config_from_controls(self, *, make_animation: bool) -> SimulationConfig:
        n_fibers = int(self.n_fibers_slider.value)
        fibers = tuple(FiberSpec(shape=self.config.shape_names[k % len(self.config.shape_names)]) for k in range(n_fibers))
        return replace(
            self.config,
            alpha=float(self.alpha_slider.value),
            K=float(self.K_slider.value),
            eps_entropy=float(self.eps_entropy_slider.value),
            n_fibers=n_fibers,
            fibers=fibers,
            seed=self._seed,
            make_dashboard=False,
            make_animation=make_animation,
            trajectory_frame_count=int(self.frame_cap_slider.value),
            integrator="fixed_rk2",
            density_solver=self.config.density_solver,
            density_boundary="noflux",
            record_free_energy=True,
            record_entropy_balance=True,
            density_dynamic_zoom=self._dynamic_zoom_enabled(),
            density_display_grid_size=min(int(self.display_grid_slider.value), int(self.config.grid_size)),
            density_heatmap_smoothing=bool(self.heatmap_smooth_toggle.value),
        )

    def _sync_config_status(self, result: DensitySimulationResult | None = None) -> None:
        self._sync_display_grid_slider_bounds()
        cfg = self._config_from_controls(make_animation=False)
        self._sync_zoom_toggle_labels()
        zoom_text = f"zoom={cfg.density_display_grid_size}px" if cfg.density_dynamic_zoom else "zoom=off"
        msg = (
            f"<b>Config:</b> alpha={cfg.alpha:.3f}, K={cfg.K:.2f}, eps={cfg.eps_entropy:.3f}, "
            f"fibers={cfg.n_fibers}, grid={cfg.grid_size}^2, {zoom_text}"
        )
        if result is not None:
            msg += f", steps={result.steps}, t={result.final_time:.3g}"
        self.config_status_html.value = msg

    def _capture_precompute_job(self) -> dict[str, Any]:
        return {"config": self._config_from_controls(make_animation=True)}

    def _prepare_precompute_job(self, job: dict[str, Any]) -> None:
        cfg = job["config"]
        self.config = cfg
        self._group_names = tuple(
            spec.name or _shape_name(spec.shape, k) for k, spec in enumerate(_normalize_fibers(cfg))
        )
        fiber_options = [(f"fiber {k + 1}: {name}", k) for k, name in enumerate(self._group_names)]
        self._updating = True
        try:
            self.fiber_dropdown.options = fiber_options
            if int(self.fiber_dropdown.value) >= len(fiber_options):
                self.fiber_dropdown.value = 0
        finally:
            self._updating = False
        self._sync_config_status()

    def _run_precompute_job(self, job: dict[str, Any]) -> DensitySimulationResult:
        seq = int(job["seq"])
        return run_density_simulation(
            job["config"],
            cancel_check=lambda: self._is_precompute_cancelled(seq),
        )

    def _ingest_precompute_result(self, result: DensitySimulationResult) -> None:
        self._ingest_result(result)

    def _sync_frame_controls(self, frame_index: int) -> None:
        max_frame = max(0, len(self._frame_payloads) - 1)
        frame_index = int(np.clip(frame_index, 0, max_frame))
        disabled = self._playback_disabled()
        self._updating = True
        try:
            self.frame_slider.min = 0
            self.frame_slider.max = max_frame
            self.frame_slider.value = frame_index
            self.frame_slider.disabled = disabled
            self.play.min = 0
            self.play.max = max_frame
            self.play.value = frame_index
            self.play.disabled = disabled
            self.btn_step.disabled = disabled
            self.frame_counter.value = f"frame {frame_index} / {max_frame}"
        finally:
            self._updating = False

    def _set_frame_index(self, index: int, *, source: Any | None = None) -> None:
        if not self._frame_payloads:
            return
        index = int(np.clip(index, 0, len(self._frame_payloads) - 1))
        self._frame_index = index
        self._apply_cached_frame(self._frame_payloads[index])
        self._updating = True
        try:
            if source is not self.frame_slider:
                self.frame_slider.value = index
            if source is not self.play:
                pass
            elif hasattr(self.play, "value"):
                self.play.value = index
            self.frame_counter.value = f"frame {index} / {len(self._frame_payloads) - 1}"
        finally:
            self._updating = False

    def _plotly_values(self, arr: Array) -> list[float]:
        return np.asarray(arr, dtype=np.float64).tolist()

    def _apply_cached_frame(self, payload: dict[str, Any]) -> None:
        with self.fig.batch_update():
            heatmap = self.fig.data[0]
            heatmap.x = self._plotly_values(np.asarray(payload["x"], dtype=np.float64))
            heatmap.y = self._plotly_values(np.asarray(payload["y"], dtype=np.float64))
            heatmap.z = payload["z"]
            heatmap.zmin = 0.0 if payload["panel"] != "div_A" else payload.get("zmin", 0.0)
            heatmap.zmax = float(payload.get("zmax", self._zmax))
            heatmap.zsmooth = "best" if payload.get("heatmap_smoothing", True) else False
            x_range = payload.get("x_range", [-self.config.domain_radius, self.config.domain_radius])
            y_range = payload.get("y_range", [-self.config.domain_radius, self.config.domain_radius])
            self.fig.update_xaxes(range=list(x_range))
            self.fig.update_yaxes(range=list(y_range))
            self.fig.layout.title = payload.get("title", self.fig.layout.title)
        stats = payload.get("stats", "")
        if stats:
            self.stats_html.value = stats

    def _support_field_for_panel(
        self,
        panel: ContinuousDensityPanelChoice,
        r_fiber: Array,
        rho: Array,
        fiber: int,
    ) -> Array:
        if panel == "r_fiber":
            idx = int(np.clip(fiber, 0, r_fiber.shape[0] - 1))
            return r_fiber[idx]
        return rho

    def _build_frame_payloads(self, result: DensitySimulationResult, cfg: SimulationConfig) -> list[dict[str, Any]]:
        solver = FFTPeszekPoyatoDensity2D(cfg.alpha, cfg.K, cfg.grid_size, cfg.domain_radius)
        self._solver = solver
        axis = _density_axis(cfg, cfg.grid_size)
        self._density_axis = axis
        panel = self.panel_dropdown.value
        fiber = int(self.fiber_dropdown.value)
        payloads: list[dict[str, Any]] = []

        if result.trajectory_r_fiber is None or result.trajectory_rho is None:
            frames_r = [result.r_fiber]
            frames_rho = [result.rho_grid]
            frame_steps = [int(result.steps)]
        else:
            frames_r = list(result.trajectory_r_fiber)
            frames_rho = list(result.trajectory_rho)
            frame_steps = [int(s) for s in result.trajectory_steps] if result.trajectory_steps is not None else list(range(len(frames_rho)))

        raw_windows: list[_DensityZoomWindow] = []
        for frame_rho in frames_rho:
            support = self._support_field_for_panel(
                panel,
                np.asarray(frames_r[len(raw_windows)], dtype=np.float64),
                np.asarray(frame_rho, dtype=np.float64),
                fiber,
            )
            raw_windows.append(
                _density_support_window(
                    support,
                    axis,
                    L=float(cfg.domain_radius),
                    mass_fraction=float(cfg.density_dynamic_zoom_mass),
                    margin=float(cfg.density_dynamic_zoom_margin),
                    min_half_width=cfg.density_dynamic_zoom_min_width,
                )
            )

        smoothed_windows: list[_DensityZoomWindow | None] = []
        previous: _DensityZoomWindow | None = None
        for raw in raw_windows:
            if cfg.density_dynamic_zoom:
                previous = _smooth_density_zoom_window(previous, raw, float(cfg.density_dynamic_zoom_smoothing))
                smoothed_windows.append(previous)
            else:
                smoothed_windows.append(None)

        zmax = 0.0
        for frame_idx, (frame_r, frame_rho) in enumerate(zip(frames_r, frames_rho, strict=True)):
            frame_r = np.asarray(frame_r, dtype=np.float64)
            frame_rho = np.asarray(frame_rho, dtype=np.float64)
            field, title, zmin = self._panel_array(
                panel,
                frame_r,
                frame_rho,
                solver,
                result.initial.omega,
                result.initial.nu,
                fiber,
            )
            edge_mass = _density_edge_mass_fraction(
                frame_rho,
                axis,
                float(cfg.domain_radius),
                float(cfg.density_edge_band_fraction),
            )
            display = _density_display_payload_from_grid(
                field,
                axis,
                cfg,
                dynamic_zoom=bool(cfg.density_dynamic_zoom),
                window=smoothed_windows[frame_idx],
            )
            z = display["z"]
            zmax = max(zmax, float(np.max(z)))
            step = frame_steps[frame_idx] if frame_idx < len(frame_steps) else frame_idx
            zoom_note = "dynamic zoom" if display["zoomed"] else "full domain"
            stats = (
                f"frame {frame_idx + 1}/{len(frames_rho)}; step={step}; "
                f"edge_mass={edge_mass:.3e}; {zoom_note}; "
                f"x=[{display['x_range'][0]:.3g}, {display['x_range'][1]:.3g}]"
            )
            payloads.append(
                {
                    "panel": panel,
                    "z": self._plotly_values(z.T),
                    "x": self._plotly_values(display["x"]),
                    "y": self._plotly_values(display["y"]),
                    "x_range": list(display["x_range"]),
                    "y_range": list(display["y_range"]),
                    "zmax": float(np.max(z)),
                    "zmin": zmin,
                    "title": title,
                    "stats": stats,
                    "edge_mass_fraction": edge_mass,
                    "heatmap_smoothing": bool(cfg.density_heatmap_smoothing),
                }
            )
        self._zmax = max(zmax, 1e-12)
        for payload in payloads:
            payload["zmax"] = self._zmax
        return payloads

    def _panel_array(
        self,
        panel: ContinuousDensityPanelChoice,
        r_fiber: Array,
        rho: Array,
        solver: FFTPeszekPoyatoDensity2D,
        omega: Array,
        nu: Array,
        fiber: int,
    ) -> tuple[Array, str, float]:
        if panel == "rho":
            return rho, "Marginal density rho(x)", 0.0
        if panel == "r_fiber":
            idx = int(np.clip(fiber, 0, r_fiber.shape[0] - 1))
            return r_fiber[idx], f"Fiber density r_{idx + 1}(x)", 0.0
        Ax, Ay = solver.A_grid_from_rho(rho)
        if panel == "velocity_mag":
            idx = int(np.clip(fiber, 0, omega.shape[0] - 1))
            speed = np.sqrt((omega[idx, 0] - Ax) ** 2 + (omega[idx, 1] - Ay) ** 2)
            return speed, f"|omega_{idx + 1} - A_rho|", 0.0
        Hxx, _, Hyy = solver.hessian_grid_from_rho(rho)
        trace = Hxx + Hyy
        return trace, "trace div A_rho = Hxx + Hyy", float(np.min(trace))

    def _ingest_result(self, result: DensitySimulationResult) -> None:
        cfg = self._config_from_controls(make_animation=True)
        self._result = result
        self._frame_payloads = self._build_frame_payloads(result, cfg)
        self._frame_index = 0
        if self._frame_payloads:
            self._apply_cached_frame(self._frame_payloads[0])
        self._sync_config_status(result)
        self._mark_cache_ready(f"Cache ready: {len(self._frame_payloads)} frames.")

    def _on_control_change(self, change: dict[str, Any]) -> None:
        if self._updating or change.get("name") != "value":
            return
        if change.get("owner") in (
            self.panel_dropdown,
            self.fiber_dropdown,
            self.dynamic_zoom_toggle,
            self.display_grid_slider,
            self.heatmap_smooth_toggle,
        ) and self._result is not None:
            self._sync_zoom_toggle_labels()
            self._frame_payloads = self._build_frame_payloads(self._result, self._config_from_controls(make_animation=True))
            if self._frame_payloads:
                self._set_frame_index(self._frame_index)
            self._sync_frame_controls(self._frame_index)
            return
        if change.get("owner") is self.n_fibers_slider:
            n_fibers = int(self.n_fibers_slider.value)
            self._group_names = tuple(
                spec.name or _shape_name(spec.shape, k) for k, spec in enumerate(_normalize_fibers(replace(self.config, n_fibers=n_fibers)))
            )
            self.fiber_dropdown.options = [(f"fiber {k + 1}: {name}", k) for k, name in enumerate(self._group_names)]
        self._mark_cache_stale("Parameters changed. Click Interrupt, then Precompute flow.")

    def _on_step(self, _btn: Any) -> None:
        if self._playback_disabled():
            return
        self._set_frame_index((self._frame_index + 1) % len(self._frame_payloads))

    def _on_play_tick(self, change: dict[str, Any]) -> None:
        if self._updating or change.get("name") != "value":
            return
        if self._playback_disabled():
            return
        self._set_frame_index(int(change.get("new", 0)), source=self.play)

    def _on_frame_slider(self, change: dict[str, Any]) -> None:
        if self._updating or change.get("name") != "value":
            return
        if self._playback_disabled():
            return
        self._set_frame_index(int(self.frame_slider.value), source=self.frame_slider)

    def _ipython_display_(self) -> None:  # pragma: no cover - notebook display hook
        from IPython.display import display

        display(self.layout)


def make_continuous_density_widget(
    config: SimulationConfig,
    result: DensitySimulationResult | None = None,
    *,
    width: int = 900,
    height: int = 720,
) -> PeszekPoyatoContinuousDensityWidget:
    """Build the continuous-density PP notebook widget."""

    return PeszekPoyatoContinuousDensityWidget(config, result=result, width=width, height=height)


def make_dashboard(result: SimulationResult, analysis: GeometryAnalysis, config: SimulationConfig) -> go.Figure:
    """Create the reusable Plotly diagnostic dashboard."""

    rng = np.random.default_rng(config.seed + 1)
    group_names = result.initial.group_names
    group_id = result.initial.group_id
    x_final = result.x_final
    omega = result.initial.omega
    N = len(x_final)
    colors = fiber_colors(config, result.initial.omega_atoms, len(group_names))

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
        color = colors[k]
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
    header = (
        "step,time,rms_residual,max_residual,dt,field_evaluations,"
        "accepted_steps,rejected_steps,clip_events\n"
    )
    with path.open("w") as f:
        f.write(header)
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
    parser.add_argument("--device", default=None, help="torch device override, e.g. cuda, cuda:0, mps, or cpu")
    parser.add_argument("--dtype", choices=("auto", "float32", "float64"), default="auto")
    parser.add_argument("--integrator", choices=("fixed_rk2", "adaptive_rk2"), default="adaptive_rk2")
    parser.add_argument("--time-direction", choices=("forward", "backward"), default="forward")
    parser.add_argument("--adaptive-tol", type=float, default=5.0e-3)
    parser.add_argument("--dt-min", type=float, default=1.0e-4)
    parser.add_argument("--dt-max", type=float, default=0.09)
    parser.add_argument("--max-displacement-per-step", type=float, default=0.75)
    parser.add_argument("--record-every", type=int, default=5)
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
        integrator=args.integrator,
        time_direction=args.time_direction,
        adaptive_tol=args.adaptive_tol,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        max_displacement_per_step=args.max_displacement_per_step,
        record_every=args.record_every,
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
        f"time={config.time_direction}, "
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
