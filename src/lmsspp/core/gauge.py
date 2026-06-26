"""Neutral LMS gauge primitives.

This module intentionally avoids gauge-selection policy.  It provides the
shared tensor operations used by exact canonical gauges, legacy gauges, widgets,
and future experiment APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from .lms import DEFAULT_EPS, mobius_sphere, normalize


@dataclass(frozen=True)
class GaugeDiagnostics:
    mode: str
    residual_norm: float = float("nan")
    center_error: float = float("nan")
    potential: float = float("nan")
    iterations: int = 0
    converged: bool = True


class GaugeState:
    """Permissive LMS reduced state container.

    The base class deliberately does not enforce a relation between ``w``,
    ``reference_points``, and ``observed_points``.  Canonical subclasses may
    impose stronger invariants.
    """

    def __init__(
        self,
        *,
        w: Tensor,
        reference_points: Tensor,
        weights: Tensor | None = None,
        observed_points: Tensor | None = None,
        mode: str = "arbitrary",
        diagnostics: GaugeDiagnostics | None = None,
        canonical: Any | None = None,
    ) -> None:
        ref, a = prepare_sphere_cloud(reference_points, weights)
        self._reference_points = ref
        self._weights = a
        self._w = self._coerce_w(w)
        if observed_points is None:
            observed = self.reconstructed_points()
        else:
            observed, _ = prepare_sphere_cloud(observed_points, dtype=ref.dtype)
            if observed.shape != ref.shape:
                raise ValueError("observed_points must have shape matching reference_points.")
        self._observed_points = observed
        self.mode = str(mode)
        self.diagnostics = diagnostics or GaugeDiagnostics(mode=self.mode)
        self.canonical = canonical

    def _coerce_w(self, value: Tensor) -> Tensor:
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value, dtype=self._reference_points.dtype, device=self._reference_points.device)
        w = value.to(dtype=self._reference_points.dtype, device=self._reference_points.device).reshape(-1)
        if int(w.shape[0]) != int(self._reference_points.shape[1]):
            raise ValueError("w must have shape [d], matching reference_points.")
        if not torch.isfinite(w).all():
            raise ValueError("w must be finite.")
        return w

    @property
    def w(self) -> Tensor:
        return self._w

    @w.setter
    def w(self, value: Tensor) -> None:
        self.set_w(value)

    @property
    def reference_points(self) -> Tensor:
        return self._reference_points

    @reference_points.setter
    def reference_points(self, value: Tensor) -> None:
        self.set_reference_points(value)

    @property
    def observed_points(self) -> Tensor:
        return self._observed_points

    @observed_points.setter
    def observed_points(self, value: Tensor) -> None:
        observed, _ = prepare_sphere_cloud(value, dtype=self._reference_points.dtype)
        if observed.shape != self._reference_points.shape:
            raise ValueError("observed_points must have shape matching reference_points.")
        self._observed_points = observed

    @property
    def weights(self) -> Tensor:
        return self._weights

    def set_w(self, value: Tensor) -> "GaugeState":
        self._w = self._coerce_w(value)
        return self

    def set_reference_points(
        self,
        reference_points: Tensor,
        weights: Tensor | None = None,
    ) -> "GaugeState":
        ref, a = prepare_sphere_cloud(reference_points, self._weights if weights is None else weights)
        if int(ref.shape[1]) != int(self._w.shape[0]):
            raise ValueError("reference_points dimension must match w.")
        self._reference_points = ref
        self._weights = a
        return self

    def reconstructed_points(self) -> Tensor:
        return physical_cloud_from_reference(self._reference_points, self._w)

    def as_lms_inputs(self) -> tuple[Tensor, Tensor, Tensor]:
        return self._w, self._reference_points, self._weights


ReducedGaugeState = GaugeState


def normalize_weights(points: Tensor, weights: Tensor) -> Tensor:
    """Return nonnegative weights normalized to total mass one."""
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


def prepare_sphere_cloud(
    points: Tensor,
    weights: Tensor | None = None,
    *,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    """Normalize a finite sphere cloud and its weights."""
    if not isinstance(points, Tensor):
        points = torch.as_tensor(points, dtype=dtype or torch.float64)
    elif dtype is not None:
        points = points.to(dtype=dtype)
    if points.dim() != 2:
        raise ValueError("points must have shape [N,d].")
    pts = normalize(points)
    if weights is None:
        w = torch.ones(int(pts.shape[0]), dtype=pts.dtype, device=pts.device) / float(pts.shape[0])
    else:
        if not isinstance(weights, Tensor):
            weights = torch.as_tensor(weights, dtype=pts.dtype, device=pts.device)
        w = normalize_weights(pts, weights)
    return pts, w


def physical_cloud_from_reference(reference_points: Tensor, w: Tensor) -> Tensor:
    """Push a reference cloud forward by the reduced LMS coordinate w."""
    ref, _ = prepare_sphere_cloud(reference_points)
    w_t = w.to(dtype=ref.dtype, device=ref.device)
    return normalize(mobius_sphere(ref, w_t))


def reference_cloud_from_physical(observed_points: Tensor, w: Tensor) -> Tensor:
    """Recover reference points by applying M_{-w} to a physical cloud."""
    observed, _ = prepare_sphere_cloud(observed_points)
    w_t = w.to(dtype=observed.dtype, device=observed.device)
    return normalize(mobius_sphere(observed, -w_t))


def target_w_from_radius(
    radius: float,
    direction: Tensor,
    *,
    convention: Literal["w", "physical_dipole"] = "w",
) -> Tensor:
    """Return a reduced coordinate with the requested radius and sign convention."""
    r = float(max(0.0, min(0.999999, float(radius))))
    if not isinstance(direction, Tensor):
        direction = torch.as_tensor(direction, dtype=torch.float64)
    if direction.dim() != 1:
        direction = direction.reshape(-1)
    norm = torch.linalg.norm(direction)
    if float(norm) <= DEFAULT_EPS:
        axis = torch.zeros_like(direction)
        axis[0] = 1.0
    else:
        axis = direction / norm
    if convention == "w":
        return axis * r
    if convention == "physical_dipole":
        return -axis * r
    raise ValueError(f"Unsupported target_w convention: {convention!r}")


__all__ = [
    "GaugeDiagnostics",
    "GaugeState",
    "ReducedGaugeState",
    "normalize_weights",
    "physical_cloud_from_reference",
    "prepare_sphere_cloud",
    "reference_cloud_from_physical",
    "target_w_from_radius",
]
