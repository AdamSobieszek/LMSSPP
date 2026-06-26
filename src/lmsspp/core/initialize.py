"""Gauge-agnostic LMS initialization helpers."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from .gauge import (
    GaugeDiagnostics,
    GaugeState,
    physical_cloud_from_reference,
    prepare_sphere_cloud,
    reference_cloud_from_physical,
)
from .lms import clamp_to_ball


OpticalPreset = Literal["random", "balanced", "clustered", "dipole"]
TAU = 2.0 * math.pi


def random_sphere_points_np(
    n: int,
    d: int,
    rng: np.random.Generator,
    *,
    preset: OpticalPreset = "random",
) -> np.ndarray:
    n = max(2, int(n))
    d = int(d)
    if d == 2:
        if preset == "balanced":
            theta = np.linspace(0.0, TAU, n, endpoint=False)
        elif preset == "clustered":
            theta = rng.normal(0.2, 0.35, size=n)
        elif preset == "dipole":
            half = n // 2
            theta = np.concatenate(
                [
                    rng.normal(0.0, 0.16, size=half),
                    rng.normal(math.pi, 0.16, size=n - half),
                ]
            )
        else:
            theta = rng.uniform(-math.pi, math.pi, size=n)
        return np.column_stack([np.cos(theta), np.sin(theta)])
    if preset == "balanced":
        idx = np.arange(n, dtype=np.float64)
        z = 1.0 - 2.0 * (idx + 0.5) / float(n)
        phi = idx * math.pi * (3.0 - math.sqrt(5.0))
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if preset == "clustered":
        center = np.zeros(d, dtype=np.float64)
        center[0] = 1.0
        if d > 2:
            center[2] = 0.2
        x = center + 0.35 * rng.normal(size=(n, d))
    elif preset == "dipole":
        half = n // 2
        plus = np.zeros(d, dtype=np.float64)
        minus = np.zeros(d, dtype=np.float64)
        plus[0] = 1.0
        minus[0] = -1.0
        x = np.vstack(
            [
                plus + 0.18 * rng.normal(size=(half, d)),
                minus + 0.18 * rng.normal(size=(n - half, d)),
            ]
        )
    else:
        x = rng.normal(size=(n, d))
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def normalize_np(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return arr / np.maximum(np.linalg.norm(arr, axis=-1, keepdims=True), float(eps))


def unit_direction_np(direction: np.ndarray | None, d: int, *, eps: float = 1e-12) -> np.ndarray:
    if direction is None:
        u = np.zeros(int(d), dtype=np.float64)
        u[0] = 1.0
        return u
    u = np.asarray(direction, dtype=np.float64).reshape(int(d))
    n = float(np.linalg.norm(u))
    if not np.isfinite(n) or n <= float(eps):
        u = np.zeros(int(d), dtype=np.float64)
        u[0] = 1.0
        return u
    return u / n


def householder_map_e0_to(target: np.ndarray) -> np.ndarray:
    """Return an orthogonal matrix H with H @ e0 == target."""
    u = np.asarray(target, dtype=np.float64).reshape(-1)
    d = int(u.size)
    e0 = np.zeros(d, dtype=np.float64)
    e0[0] = 1.0
    v = e0 - u
    nv = float(np.linalg.norm(v))
    if nv <= 1e-15:
        return np.eye(d, dtype=np.float64)
    wv = v / nv
    return np.eye(d, dtype=np.float64) - 2.0 * np.outer(wv, wv)


def euclidean_center_unit_sphere_np(
    pts: np.ndarray,
    weights: np.ndarray,
    *,
    max_iters: int = 48,
    tol: float = 1e-13,
) -> np.ndarray:
    """Subtract the weighted Euclidean mean and renormalize to the sphere."""
    x = np.asarray(pts, dtype=np.float64).reshape(-1, pts.shape[-1])
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = w / float(np.sum(w))
    for _ in range(int(max_iters)):
        mu = np.sum(w[:, None] * x, axis=0)
        if float(np.linalg.norm(mu)) <= float(tol):
            break
        x = normalize_np(x - mu[None, :])
    return x


def reference_cloud_from_preset_np(
    n: int,
    d: int,
    rng: np.random.Generator,
    *,
    preset: OpticalPreset,
    direction: np.ndarray,
) -> np.ndarray:
    """Sample a reference-cloud template and align its e0 axis."""
    xi = random_sphere_points_np(int(n), int(d), rng, preset=preset)
    h = householder_map_e0_to(unit_direction_np(direction, int(d)))
    return (h @ xi.T).T


def shrink_fd(d: int, r: float) -> float:
    """Continuum LMS Poisson shrink factor f_d(r)."""
    r_clamped = max(0.0, min(0.999999999, float(r)))
    if int(d) == 2:
        return 1.0
    b = 1.0 - 0.5 * float(d)
    c = 1.0 + 0.5 * float(d)
    u = r_clamped * r_clamped
    fu = _hyp2f1_1b_c_u(b, c, u)
    f1 = _hyp2f1_1b_c_1(b, c)
    if abs(f1) < 1e-12:
        return 1.0
    return float(fu / f1)


def radius_to_centroid_norm(d: int, r: float) -> float:
    r_clip = float(max(0.0, min(0.999999, float(r))))
    return float(max(0.0, min(0.999999, shrink_fd(int(d), r_clip) * r_clip)))


def poisson_shrink_w_from_observed(
    points: Tensor,
    weights: Tensor,
    *,
    fallback_dir: Tensor | None = None,
) -> Tensor:
    """Estimate w from a physical cloud using the continuum Poisson shrink law."""
    observed, a = prepare_sphere_cloud(points, weights)
    mean = (a[:, None] * observed).sum(dim=0)
    q = float(torch.linalg.norm(mean))
    if q <= 1e-12:
        return torch.zeros(int(observed.shape[1]), dtype=observed.dtype, device=observed.device)

    _ = fallback_dir
    direction = -mean / max(q, 1e-12)
    q_target = min(q, 0.999999)
    d = int(observed.shape[1])
    lo = 0.0
    hi = 0.999999
    for _ in range(72):
        mid = 0.5 * (lo + hi)
        if radius_to_centroid_norm(d, mid) < q_target:
            lo = mid
        else:
            hi = mid
    return clamp_to_ball(direction * (0.5 * (lo + hi)), radius=0.999999)


def poisson_state_from_observed_cloud(
    points: Tensor,
    weights: Tensor,
    *,
    fallback_dir: Tensor | None = None,
) -> GaugeState:
    observed, a = prepare_sphere_cloud(points, weights)
    w = poisson_shrink_w_from_observed(observed, a, fallback_dir=fallback_dir)
    reference = reference_cloud_from_physical(observed, w)
    center_error = float(torch.linalg.norm((a[:, None] * reference).sum(dim=0)))
    return GaugeState(
        w=w,
        reference_points=reference,
        weights=a,
        observed_points=observed,
        mode="poisson_shrink",
        diagnostics=GaugeDiagnostics(
            mode="poisson_shrink",
            residual_norm=center_error,
            center_error=center_error,
            iterations=0,
            converged=True,
        ),
    )


def poisson_state_from_reference_cloud(
    reference_points: Tensor,
    target_w: Tensor,
    weights: Tensor,
) -> GaugeState:
    reference, a = prepare_sphere_cloud(reference_points, weights)
    w = target_w.to(dtype=reference.dtype, device=reference.device).reshape(-1)
    observed = physical_cloud_from_reference(reference, w)
    center_error = float(torch.linalg.norm((a[:, None] * reference).sum(dim=0)))
    return GaugeState(
        w=w,
        reference_points=reference,
        weights=a,
        observed_points=observed,
        mode="poisson_shrink",
        diagnostics=GaugeDiagnostics(
            mode="poisson_shrink",
            residual_norm=center_error,
            center_error=center_error,
            iterations=0,
            converged=True,
        ),
    )


def _hyp2f1_1b_c_u(b: float, c: float, u: float) -> float:
    max_n = 2500
    tol = 1e-12
    term = 1.0
    acc = 1.0
    for n in range(max_n):
        term *= ((b + n) / (c + n)) * u
        acc += term
        if abs(term) < tol * max(1.0, abs(acc)):
            break
    return float(acc)


def _hyp2f1_1b_c_1(b: float, c: float) -> float:
    return float(
        math.gamma(c) * math.gamma(c - 1.0 - b)
        / (math.gamma(c - 1.0) * math.gamma(c - b))
    )


__all__ = [
    "OpticalPreset",
    "euclidean_center_unit_sphere_np",
    "householder_map_e0_to",
    "normalize_np",
    "poisson_shrink_w_from_observed",
    "poisson_state_from_observed_cloud",
    "poisson_state_from_reference_cloud",
    "radius_to_centroid_norm",
    "random_sphere_points_np",
    "reference_cloud_from_preset_np",
    "shrink_fd",
    "unit_direction_np",
]
