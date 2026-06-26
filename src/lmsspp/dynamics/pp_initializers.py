"""Peszek-Poyato initial condition and warmup initializers.

Initializers create raw joint ``(x, omega)`` particle states before the PP
dynamics integrator runs. Warmup initializers may use PP backend fields, but
they intentionally discard their transient history and return only a new
initial state plus provenance metadata.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import numpy as np

from .PP import FFTPeszekPoyato2D

Array = np.ndarray
ShapeSampler = Callable[[int, np.random.Generator], Array]
InitializationAlgorithmChoice = Literal["raw", "alpha_ball", "legacy_fast_phase"]

DEFAULT_SHAPES = (
    "gaussian",
    "ring",
    "arc",
    "line",
    "spiral",
    "square",
    "crescent",
    "ellipse",
    "two_blobs",
    "triangle",
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
    ("Legacy fast phase", "legacy_fast_phase"),
)


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
    K: float | None = 1.0
    grid_size: int | None = None
    domain_radius: float | None = None
    dt: float = 0.055
    max_steps: int = 40
    min_steps: int = 6
    window: int = 3
    displacement_tol: float = 1.5e-2


@dataclass(frozen=True)
class InitialCondition:
    """Initial particle cloud grouped by conserved omega fibers."""

    x: Array
    omega: Array
    group_id: Array
    omega_atoms: Array
    group_names: tuple[str, ...]


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
    elif kind == "two_blobs":
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


def make_initial_condition(
    config: Any,
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


def _apply_initialization_algorithm(
    config: Any,
    initial: InitialCondition,
) -> tuple[InitialCondition, dict[str, object]]:
    if config.initialization_algorithm == "raw":
        return initial, {"algorithm": "raw", "steps": 0, "time": 0.0, "stop_metric": 0.0}
    if config.initialization_algorithm in ("alpha_ball", "legacy_fast_phase"):
        return _alpha_ball_initial_condition(config, initial, algorithm_name=config.initialization_algorithm)
    raise ValueError(f"unknown initialization_algorithm: {config.initialization_algorithm!r}")


def _alpha_ball_initial_condition(
    config: Any,
    initial: InitialCondition,
    *,
    algorithm_name: str = "alpha_ball",
) -> tuple[InitialCondition, dict[str, object]]:
    """Discarded-history fixed-RK2 fast-phase initializer.

    This intentionally mirrors the legacy PP integrator: NumPy float64,
    forward time, fixed ``config.dt``, no adaptive CFL limiting, and no
    trajectory recording. Only the final particle locations become the new
    initial state.
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
        "algorithm": algorithm_name,
        "steps": int(steps_done),
        "time": float(steps_done * float(init_config.dt)),
        "stop_metric": float(stop_metric),
    }


def _resolve_initializer_config(config: Any) -> InitializerConfig:
    if config.initializer_config is not None:
        return config.initializer_config
    return InitializerConfig(
        max_steps=config.initialization_fast_steps,
        min_steps=config.initialization_fast_min_steps,
        window=config.initialization_fast_window,
        displacement_tol=config.initialization_fast_displacement_tol,
    )


def _normalize_fibers(config: Any) -> tuple[FiberSpec, ...]:
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


def _omega_atoms_from_config(config: Any, fibers: Sequence[FiberSpec], rng: np.random.Generator) -> Array:
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


def _fiber_spec_to_json(spec: FiberSpec) -> dict[str, object]:
    return {
        "shape": _shape_name(spec.shape, 0),
        "n_particles": spec.n_particles,
        "omega": None if spec.omega is None else np.asarray(spec.omega, dtype=np.float64).tolist(),
        "center": None if spec.center is None else np.asarray(spec.center, dtype=np.float64).tolist(),
        "name": spec.name,
    }

