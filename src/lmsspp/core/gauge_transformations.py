"""Compatibility layer for LMS gauge-state construction.

Exact canonical construction lives in :mod:`canonical_gauge`.  Gauge-agnostic
initializers, including legacy Poisson shrink, live in :mod:`initialize`.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from .canonical_gauge import CanonicalGaugeState, canonical_residual
from .gauge import GaugeState, physical_cloud_from_reference, prepare_sphere_cloud
from .initialize import (
    poisson_shrink_w_from_observed,
    poisson_state_from_observed_cloud,
    poisson_state_from_reference_cloud,
    radius_to_centroid_norm,
    shrink_fd,
)


CenterEstimationMode = Literal["busemann_exact", "poisson_shrink"]


def canonical_center_estimation_mode(mode: str) -> CenterEstimationMode:
    value = str(mode).strip().lower().replace("-", "_")
    if value in {"poisson", "poisson_shrink", "moment", "moment_shrink", "legacy"}:
        return "poisson_shrink"
    if value in {
        "busemann",
        "busemann_exact",
        "canonical",
        "canonical_busemann",
        "exact",
        "finite_n_busemann",
    }:
        return "busemann_exact"
    return "busemann_exact"


def state_from_observed_cloud(
    points: Tensor,
    weights: Tensor,
    *,
    mode: str,
    fallback_dir: Tensor | None = None,
    target_w: Tensor | None = None,
    initial_center: Tensor | None = None,
    max_iters: int = 160,
    tol: float = 1e-10,
) -> GaugeState:
    """Build an LMS gauge state from an initialized or observed cloud."""
    resolved_mode = canonical_center_estimation_mode(mode)
    if resolved_mode == "busemann_exact":
        return CanonicalGaugeState.from_initialized_cloud(
            points,
            weights,
            target_w=target_w,
            fallback_dir=fallback_dir,
            initial_center=initial_center,
            max_iters=int(max_iters),
            tol=float(tol),
        )
    _ = target_w
    return poisson_state_from_observed_cloud(points, weights, fallback_dir=fallback_dir)


def state_from_reference_cloud(
    reference_points: Tensor,
    target_w: Tensor,
    weights: Tensor,
    *,
    mode: str,
    fallback_dir: Tensor | None = None,
    initial_center: Tensor | None = None,
    max_iters: int = 160,
    tol: float = 1e-10,
) -> GaugeState:
    """Build an LMS state from a seed/reference cloud and desired w."""
    reference, a = prepare_sphere_cloud(reference_points, weights)
    w_target = target_w.to(dtype=reference.dtype, device=reference.device).reshape(-1)
    resolved_mode = canonical_center_estimation_mode(mode)
    if resolved_mode == "poisson_shrink":
        return poisson_state_from_reference_cloud(reference, w_target, a)

    initialized = physical_cloud_from_reference(reference, w_target)
    return CanonicalGaugeState.from_initialized_cloud(
        initialized,
        a,
        target_w=w_target,
        fallback_dir=fallback_dir,
        initial_center=-w_target if initial_center is None else initial_center,
        max_iters=int(max_iters),
        tol=float(tol),
    )


def canonical_reference_from_template(
    points: Tensor,
    weights: Tensor,
    *,
    max_iters: int = 160,
    tol: float = 1e-10,
) -> CanonicalGaugeState:
    """Canonically balance a template cloud for later radius selection."""
    template, a = prepare_sphere_cloud(points, weights)
    z0 = torch.zeros(int(template.shape[1]), dtype=template.dtype, device=template.device)
    residual = float(torch.linalg.norm(canonical_residual(z0, template, a)))
    if residual <= float(tol):
        return CanonicalGaugeState.from_reference_cloud(
            template,
            a,
            z0,
            require_centered=False,
            tol=float(tol),
        )
    return CanonicalGaugeState.from_initialized_cloud(
        template,
        a,
        max_iters=int(max_iters),
        tol=float(tol),
    )


__all__ = [
    "CenterEstimationMode",
    "canonical_center_estimation_mode",
    "canonical_reference_from_template",
    "poisson_shrink_w_from_observed",
    "radius_to_centroid_norm",
    "shrink_fd",
    "state_from_observed_cloud",
    "state_from_reference_cloud",
]
