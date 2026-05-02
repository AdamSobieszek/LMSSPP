"""Interactive optical disk views of the finite-N LMS reduced force.

This module visualizes the Householder/geometric-optics decomposition

    R_{p^0}(w) = sum_i a_i H_{p_i^0-w}(p_i^0)
               = -sum_i a_i M_w(p_i^0),
    dw/dt = 0.5 * (1 - |w|^2) * R_{p^0}(w).

The points p_i^0 are frozen canonical shape constants on S^{d-1}.  They are
computed from an observed cloud x_i^0 by the exact gauge equation
sum_i a_i M_{-w_*}(x_i^0)=0 and p_i^0=M_{-w_*}(x_i^0).  The reflected cloud
r_i(w)=H_{p_i^0-w}(p_i^0) is the hidden optical object: its
barycenter R_{p^0}(w) is the direction of the reduced motion, |R_{p^0}(w)| is the
ray coherence, and its covariance records angular dispersion.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from .core.canonical_gauge import canonical_cloud
from .core.lms import DEFAULT_EPS, dot, mobius_sphere, normalize

try:  # Optional widget dependency.
    import ipywidgets as widgets
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:  # pragma: no cover
    widgets = None  # type: ignore[assignment]
    go = None  # type: ignore[assignment]
    make_subplots = None  # type: ignore[assignment]


TAU = 2.0 * math.pi
OpticalPreset = Literal["random", "balanced", "clustered", "dipole"]


@dataclass(frozen=True)
class OpticalState:
    """Full finite-N optical state at one reduced point w."""

    w: Tensor
    anchors: Tensor
    weights: Tensor
    reflected: Tensor
    R: Tensor
    F: Tensor
    velocity: Tensor
    phase: Tensor
    coherence: Tensor
    variance: Tensor
    covariance: Tensor
    second_moment: Tensor
    covariance_evals: Tensor
    covariance_evecs: Tensor


def _require_widgets() -> None:
    if widgets is None or go is None or make_subplots is None:  # pragma: no cover
        raise ImportError("lms_optical_widget requires plotly and ipywidgets.")


def _html(value: str = "", **kwargs: Any) -> Any:
    """Return a plain HTML widget, matching the working LMS 3D widget."""
    if widgets is None:  # pragma: no cover
        raise ImportError("ipywidgets is required.")
    return widgets.HTML(value=value, **kwargs)


def _sanitize_plot_text(text: str) -> str:
    """Sanitize Plotly/HTML math text the same way as the working 3D widget.

    FigureWidget MathJax rendering is environment-dependent in notebooks.  The
    LMS 3D widget avoids raw `$...$` labels, so the optical widgets do the same:
    keep labels readable as plain text rather than showing unrendered TeX.
    """
    out = str(text)
    replacements = {
        "\\partial": "∂",
        "\\mathbb{B}": "B",
        "\\mathbb H": "H",
        "\\ell": "ell",
        "\\lambda": "lambda",
        "\\Theta": "Theta",
        "\\Phi": "Phi",
        "\\Delta": "Delta",
        "\\beta": "beta",
        "\\xi": "xi",
        "\\ast": "*",
        "\\dot": "dot",
        "\\sum": "sum",
        "\\arg": "arg",
        "\\perp": "perp",
        "\\mathrm": "",
        "\\operatorname": "",
        "\\langle": "<",
        "\\rangle": ">",
        "\\quad": " ",
        "\\,": " ",
        "\\|": "|",
        "\\": "",
        "$": "",
        "{": "",
        "}": "",
    }
    for old, new in replacements.items():
        out = out.replace(old, new)
    out = out.replace("_perp", "_⊥").replace("_\\perp", "_⊥")
    out = out.replace("^0", "⁰").replace("^1", "¹").replace("^2", "²").replace("^3", "³")
    out = out.replace("_i", "ᵢ").replace("_t", "ₜ").replace("_p", "ₚ").replace("_⊥", "⊥")
    out = out.replace("p_i⁰", "pᵢ⁰").replace("x_i⁰", "xᵢ⁰").replace("r_i", "rᵢ")
    return " ".join(out.split())


def _sanitize_figure_text(fig: Any) -> None:
    """Apply plain-text Plotly labels after figure construction."""
    for tr in fig.data:
        if getattr(tr, "name", None) is not None:
            tr.name = _sanitize_plot_text(tr.name)
    for ann in getattr(fig.layout, "annotations", ()):
        if getattr(ann, "text", None):
            ann.text = _sanitize_plot_text(ann.text)
    for key in fig.layout:
        if key.startswith(("xaxis", "yaxis")):
            axis = getattr(fig.layout, key)
            title = getattr(axis, "title", None)
            if getattr(title, "text", None):
                title.text = _sanitize_plot_text(title.text)
    for key in fig.layout:
        if key.startswith("scene"):
            scene = getattr(fig.layout, key)
            for axis_name in ("xaxis", "yaxis", "zaxis"):
                axis = getattr(scene, axis_name, None)
                title = getattr(axis, "title", None) if axis is not None else None
                if getattr(title, "text", None):
                    title.text = _sanitize_plot_text(title.text)


def _as_tensor(x: Tensor | np.ndarray | list[float], *, dtype: torch.dtype = torch.float64) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return torch.as_tensor(x, dtype=dtype)


def _prepare_points_weights(
    points: Tensor | np.ndarray,
    weights: Tensor | np.ndarray | None = None,
    *,
    dtype: torch.dtype = torch.float64,
) -> tuple[Tensor, Tensor]:
    P = _as_tensor(points, dtype=dtype)
    if P.dim() != 2:
        raise ValueError("points must have shape [N,d].")
    P = normalize(P.to(dtype=dtype))
    if weights is None:
        a = torch.full((P.shape[0],), 1.0 / float(P.shape[0]), dtype=P.dtype, device=P.device)
    else:
        a = _as_tensor(weights, dtype=dtype).to(dtype=P.dtype, device=P.device)
        if a.dim() != 1 or int(a.shape[0]) != int(P.shape[0]):
            raise ValueError("weights must have shape [N], matching points.")
        if bool((a < 0).any()):
            raise ValueError("weights must be nonnegative.")
        total = a.sum()
        if float(torch.abs(total)) <= DEFAULT_EPS:
            raise ValueError("weights must have positive sum.")
        a = a / total
    return P, a


def householder_reflected_points(
    w: Tensor | np.ndarray,
    points: Tensor | np.ndarray,
    *,
    eps: float = DEFAULT_EPS,
) -> Tensor:
    """Return r_i(w)=H_{p_i-w}(p_i) using the Householder formula.

    The Householder normal is n_i=p_i-w and

        H_n(p) = p - 2 <p,n> n / |n|^2.

    For |p_i|=1 this equals -M_w(p_i), which is the optical sign convention.
    """
    P, _ = _prepare_points_weights(points)
    ww = _as_tensor(w, dtype=P.dtype).to(dtype=P.dtype, device=P.device).reshape(P.shape[1])
    normals = P - ww.unsqueeze(0)
    denom = dot(normals, normals).unsqueeze(-1).clamp(min=float(eps))
    scale = 2.0 * dot(P, normals).unsqueeze(-1) / denom
    return normalize(P - scale * normals)


def reflected_points(w: Tensor | np.ndarray, points: Tensor | np.ndarray) -> Tensor:
    """Return the reflected ray cloud r_i(w).

    This wrapper uses the Householder implementation, not the Mobius shortcut,
    so the optics construction remains explicit.
    """
    return householder_reflected_points(w, points)


def mobius_reflection_error(w: Tensor | np.ndarray, points: Tensor | np.ndarray) -> Tensor:
    """Return max_i |H_{p_i-w}(p_i) + M_w(p_i)| for identity checks."""
    P, _ = _prepare_points_weights(points)
    ww = _as_tensor(w, dtype=P.dtype).to(dtype=P.dtype, device=P.device).reshape(P.shape[1])
    r_house = reflected_points(ww, P)
    r_mobius = -mobius_sphere(P, ww)
    return torch.amax(torch.linalg.norm(r_house - r_mobius, dim=-1))


def reflected_barycenter(
    w: Tensor | np.ndarray,
    points: Tensor | np.ndarray,
    weights: Tensor | np.ndarray | None = None,
) -> Tensor:
    """Return R_p(w)=sum_i a_i r_i(w)."""
    P, a = _prepare_points_weights(points, weights)
    R = reflected_points(_as_tensor(w, dtype=P.dtype).to(dtype=P.dtype, device=P.device), P)
    return (a[:, None] * R).sum(dim=0)


def frozen_field(
    w: Tensor | np.ndarray,
    points: Tensor | np.ndarray,
    weights: Tensor | np.ndarray | None = None,
) -> Tensor:
    """Return F_p(w)=sum_i a_i M_w(p_i), so F_p(w)=-R_p(w)."""
    P, a = _prepare_points_weights(points, weights)
    ww = _as_tensor(w, dtype=P.dtype).to(dtype=P.dtype, device=P.device).reshape(P.shape[1])
    return (a[:, None] * mobius_sphere(P, ww)).sum(dim=0)


def optical_velocity(
    w: Tensor | np.ndarray,
    points: Tensor | np.ndarray,
    weights: Tensor | np.ndarray | None = None,
) -> Tensor:
    """Return dw/dt = 0.5 * (1-|w|^2) * R_p(w)."""
    P, a = _prepare_points_weights(points, weights)
    ww = _as_tensor(w, dtype=P.dtype).to(dtype=P.dtype, device=P.device).reshape(P.shape[1])
    R = reflected_barycenter(ww, P, a)
    return 0.5 * (1.0 - dot(ww, ww)) * R


def busemann_phase(
    w: Tensor | np.ndarray,
    points: Tensor | np.ndarray,
    weights: Tensor | np.ndarray | None = None,
    *,
    eps: float = 1e-12,
) -> Tensor:
    """Return S_p(w)=sum_i a_i log(|w-p_i|^2/(1-|w|^2))."""
    P, a = _prepare_points_weights(points, weights)
    ww = _as_tensor(w, dtype=P.dtype).to(dtype=P.dtype, device=P.device).reshape(P.shape[1])
    diff = ww.unsqueeze(0) - P
    numer = dot(diff, diff).clamp(min=float(eps))
    denom = (1.0 - dot(ww, ww)).clamp(min=float(eps))
    return (a * (torch.log(numer) - torch.log(denom))).sum()


def ray_second_moment(
    w: Tensor | np.ndarray,
    points: Tensor | np.ndarray,
    weights: Tensor | np.ndarray | None = None,
) -> Tensor:
    """Return A_p(w)=sum_i a_i r_i(w) r_i(w)^T."""
    P, a = _prepare_points_weights(points, weights)
    Rpts = reflected_points(_as_tensor(w, dtype=P.dtype).to(dtype=P.dtype, device=P.device), P)
    return (a[:, None, None] * (Rpts[:, :, None] * Rpts[:, None, :])).sum(dim=0)


def ray_covariance(
    w: Tensor | np.ndarray,
    points: Tensor | np.ndarray,
    weights: Tensor | np.ndarray | None = None,
) -> Tensor:
    """Return C_p(w)=sum_i a_i (r_i-R)(r_i-R)^T."""
    P, a = _prepare_points_weights(points, weights)
    Rpts = reflected_points(_as_tensor(w, dtype=P.dtype).to(dtype=P.dtype, device=P.device), P)
    Rbar = (a[:, None] * Rpts).sum(dim=0)
    centered = Rpts - Rbar.unsqueeze(0)
    return (a[:, None, None] * (centered[:, :, None] * centered[:, None, :])).sum(dim=0)


def ray_principal_axes(
    w: Tensor | np.ndarray,
    points: Tensor | np.ndarray,
    weights: Tensor | np.ndarray | None = None,
) -> tuple[Tensor, Tensor]:
    """Return eigenvalues/eigenvectors of the reflected-ray covariance."""
    C = ray_covariance(w, points, weights)
    vals, vecs = torch.linalg.eigh(C)
    return vals, vecs


def mirror_normals(w: Tensor | np.ndarray, points: Tensor | np.ndarray) -> Tensor:
    """Return n_i(w)=p_i-w."""
    P, _ = _prepare_points_weights(points)
    ww = _as_tensor(w, dtype=P.dtype).to(dtype=P.dtype, device=P.device).reshape(P.shape[1])
    return P - ww.unsqueeze(0)


def optical_state(
    w: Tensor | np.ndarray,
    points: Tensor | np.ndarray,
    weights: Tensor | np.ndarray | None = None,
) -> OpticalState:
    """Compute all first/second-moment optical data at w."""
    P, a = _prepare_points_weights(points, weights)
    ww = _as_tensor(w, dtype=P.dtype).to(dtype=P.dtype, device=P.device).reshape(P.shape[1])
    Rpts = reflected_points(ww, P)
    Rbar = (a[:, None] * Rpts).sum(dim=0)
    F = -Rbar
    vel = 0.5 * (1.0 - dot(ww, ww)) * Rbar
    centered = Rpts - Rbar.unsqueeze(0)
    C = (a[:, None, None] * (centered[:, :, None] * centered[:, None, :])).sum(dim=0)
    A = (a[:, None, None] * (Rpts[:, :, None] * Rpts[:, None, :])).sum(dim=0)
    evals, evecs = torch.linalg.eigh(C)
    coh = torch.linalg.norm(Rbar)
    return OpticalState(
        w=ww,
        anchors=P,
        weights=a,
        reflected=Rpts,
        R=Rbar,
        F=F,
        velocity=vel,
        phase=busemann_phase(ww, P, a),
        coherence=coh,
        variance=(1.0 - coh * coh).clamp(min=0.0),
        covariance=C,
        second_moment=A,
        covariance_evals=evals,
        covariance_evecs=evecs,
    )


def phase_grid_2d(
    points: Tensor | np.ndarray,
    weights: Tensor | np.ndarray | None = None,
    *,
    grid_size: int = 100,
    radius: float = 0.995,
    clip_quantile: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate S_p on a masked square grid for a 2D contour plot."""
    P, a = _prepare_points_weights(points, weights)
    if int(P.shape[1]) != 2:
        raise ValueError("phase_grid_2d requires points in R^2.")
    p_np = P.detach().cpu().numpy()
    a_np = a.detach().cpu().numpy()
    xs = np.linspace(-float(radius), float(radius), int(grid_size))
    ys = np.linspace(-float(radius), float(radius), int(grid_size))
    X, Y = np.meshgrid(xs, ys)
    rr = X * X + Y * Y
    mask = rr < float(radius) ** 2
    denom = np.maximum(1.0 - rr, 1e-12)
    Z = np.zeros_like(X, dtype=np.float64)
    for ai, pi in zip(a_np, p_np, strict=True):
        diff2 = np.maximum((X - pi[0]) ** 2 + (Y - pi[1]) ** 2, 1e-12)
        Z += float(ai) * (np.log(diff2) - np.log(denom))
    Z[~mask] = np.nan
    finite = Z[np.isfinite(Z)]
    if finite.size and 0.0 < float(clip_quantile) < 0.49:
        lo, hi = np.quantile(finite, [float(clip_quantile), 1.0 - float(clip_quantile)])
        Z = np.clip(Z, lo, hi)
    return xs, ys, Z


def potential_level_segments_2d(
    xs: np.ndarray,
    ys: np.ndarray,
    values: np.ndarray,
    *,
    level_count: int = 8,
) -> tuple[list[float | None], list[float | None]]:
    """Return explicit line segments for potential level sets on a regular grid.

    The implementation vectorizes marching-squares work over grid cells.  This
    is intended for precompute time; playback should only reuse the returned
    Python lists.
    """
    Z = np.asarray(values, dtype=np.float64)
    finite = Z[np.isfinite(Z)]
    if finite.size <= 0:
        return [], []
    lo, hi = np.quantile(finite, [0.08, 0.92])
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(float(hi - lo)) < 1e-12:
        return [], []
    levels = np.linspace(float(lo), float(hi), max(2, int(level_count)))
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    X0 = x_arr[:-1][None, :]
    X1 = x_arr[1:][None, :]
    Y0 = y_arr[:-1][:, None]
    Y1 = y_arr[1:][:, None]
    z0 = Z[:-1, :-1]
    z1 = Z[:-1, 1:]
    z2 = Z[1:, 1:]
    z3 = Z[1:, :-1]
    finite_cells = np.isfinite(z0) & np.isfinite(z1) & np.isfinite(z2) & np.isfinite(z3)
    out_x: list[float | None] = []
    out_y: list[float | None] = []

    def edge_points(level: float) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        eps = 1e-12
        t01 = np.clip((level - z0) / np.where(np.abs(z1 - z0) < eps, np.nan, z1 - z0), 0.0, 1.0)
        t12 = np.clip((level - z1) / np.where(np.abs(z2 - z1) < eps, np.nan, z2 - z1), 0.0, 1.0)
        t23 = np.clip((level - z2) / np.where(np.abs(z3 - z2) < eps, np.nan, z3 - z2), 0.0, 1.0)
        t30 = np.clip((level - z3) / np.where(np.abs(z0 - z3) < eps, np.nan, z0 - z3), 0.0, 1.0)
        return {
            0: (X0 + t01 * (X1 - X0), np.broadcast_to(Y0, z0.shape)),
            1: (np.broadcast_to(X1, z0.shape), Y0 + t12 * (Y1 - Y0)),
            2: (X1 + t23 * (X0 - X1), np.broadcast_to(Y1, z0.shape)),
            3: (np.broadcast_to(X0, z0.shape), Y1 + t30 * (Y0 - Y1)),
        }

    def add_segments(mask: np.ndarray, e_a: int, e_b: int, edges: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
        if not bool(mask.any()):
            return
        xa, ya = edges[e_a]
        xb, yb = edges[e_b]
        xa_v = xa[mask]
        ya_v = ya[mask]
        xb_v = xb[mask]
        yb_v = yb[mask]
        if xa_v.size <= 0:
            return
        arr_x = np.empty(xa_v.size * 3, dtype=object)
        arr_y = np.empty(ya_v.size * 3, dtype=object)
        arr_x[0::3] = xa_v
        arr_x[1::3] = xb_v
        arr_x[2::3] = None
        arr_y[0::3] = ya_v
        arr_y[1::3] = yb_v
        arr_y[2::3] = None
        out_x.extend(arr_x.tolist())
        out_y.extend(arr_y.tolist())

    case_segments = {
        1: ((3, 0),),
        2: ((0, 1),),
        3: ((3, 1),),
        4: ((1, 2),),
        5: ((3, 2), (0, 1)),
        6: ((0, 2),),
        7: ((3, 2),),
        8: ((2, 3),),
        9: ((0, 2),),
        10: ((0, 3), (1, 2)),
        11: ((1, 2),),
        12: ((1, 3),),
        13: ((0, 1),),
        14: ((3, 0),),
    }

    for level in levels:
        code = (
            (z0 > level).astype(np.uint8)
            | ((z1 > level).astype(np.uint8) << 1)
            | ((z2 > level).astype(np.uint8) << 2)
            | ((z3 > level).astype(np.uint8) << 3)
        )
        edges = edge_points(float(level))
        for case, segments in case_segments.items():
            mask = finite_cells & (code == case)
            for e_a, e_b in segments:
                add_segments(mask, e_a, e_b, edges)
    return out_x, out_y


def optical_flow_step(
    w: Tensor | np.ndarray,
    points: Tensor | np.ndarray,
    weights: Tensor | np.ndarray | None = None,
    *,
    step_size: float = 0.08,
    radius: float = 0.999,
) -> Tensor:
    """Take one explicit step along the optical reduced velocity."""
    P, a = _prepare_points_weights(points, weights)
    ww = _as_tensor(w, dtype=P.dtype).to(dtype=P.dtype, device=P.device).reshape(P.shape[1])
    nxt = ww + float(step_size) * optical_velocity(ww, P, a)
    nrm = torch.linalg.norm(nxt)
    if float(nrm) > float(radius):
        nxt = nxt * (float(radius) / (nrm + DEFAULT_EPS))
    return nxt


def _random_sphere_points_np(
    n: int,
    d: int,
    rng: np.random.Generator,
    *,
    preset: OpticalPreset = "random",
) -> np.ndarray:
    n = max(2, int(n))
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
        # Fibonacci sphere for a stable non-random reference cloud.
        idx = np.arange(n, dtype=np.float64)
        z = 1.0 - 2.0 * (idx + 0.5) / float(n)
        phi = idx * math.pi * (3.0 - math.sqrt(5.0))
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if preset == "clustered":
        x = np.array([1.0, 0.0, 0.2]) + 0.35 * rng.normal(size=(n, d))
    elif preset == "dipole":
        half = n // 2
        x = np.vstack(
            [
                np.array([1.0, 0.0, 0.0]) + 0.18 * rng.normal(size=(half, d)),
                np.array([-1.0, 0.0, 0.0]) + 0.18 * rng.normal(size=(n - half, d)),
            ]
        )
    else:
        x = rng.normal(size=(n, d))
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _initialized_observed_cloud_np(
    n: int,
    rng: np.random.Generator,
    *,
    preset: OpticalPreset,
    target_radius: float,
    direction: np.ndarray | None = None,
) -> np.ndarray:
    """Sample x_i^0 with a requested exact-gauge radius before inversion.

    This is the 2D analogue of the 3D widget's radius-driven Poisson/reduced
    initialization path: sample a base boundary cloud, exact-center it to get
    constants p_i^0, then generate the observed cloud x_i^0=M_w(p_i^0) for a
    requested |w|.  The widget still computes displayed w* from x_i^0 by exact
    Busemann inversion afterward.
    """
    base = _random_sphere_points_np(int(n), 2, rng, preset=preset)
    weights = np.full((base.shape[0],), 1.0 / float(base.shape[0]), dtype=np.float64)
    P_t, a_t = _prepare_points_weights(base, weights)
    try:
        centered = canonical_cloud(P_t, a_t, max_iters=120, tol=1e-10).P
    except Exception:
        centered = P_t
    if direction is None:
        u = np.array([1.0, 0.0], dtype=np.float64)
    else:
        u = np.asarray(direction, dtype=np.float64).reshape(2)
        u_norm = float(np.linalg.norm(u))
        u = np.array([1.0, 0.0], dtype=np.float64) if u_norm <= 1e-12 else u / u_norm
    r = float(np.clip(float(target_radius), 0.0, 0.985))
    w = torch.as_tensor(r * u, dtype=centered.dtype, device=centered.device)
    observed = normalize(mobius_sphere(centered, w))
    return observed.detach().cpu().numpy().astype(np.float64)


def _angle2(theta: float) -> np.ndarray:
    return np.array([math.cos(float(theta)), math.sin(float(theta))], dtype=np.float64)


def _line2(a: np.ndarray, b: np.ndarray) -> tuple[list[float], list[float]]:
    return [float(a[0]), float(b[0])], [float(a[1]), float(b[1])]


def _plotly_values(values: Any) -> Any:
    """Convert cached numeric arrays to widget-safe Python containers.

    Plotly FigureWidget can accept numpy arrays, but its ipywidgets delta-sync
    path may later evaluate them in boolean context, which raises
    `ValueError: truth value of an array is ambiguous`.  Keep numpy in caches,
    but never assign it directly to trace x/y fields.
    """
    if isinstance(values, np.ndarray):
        return values.tolist()
    if isinstance(values, Tensor):
        return values.detach().cpu().numpy().tolist()
    return values


def _circle_xy(samples: int = 360) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, TAU, int(samples))
    return np.cos(t), np.sin(t)


def _disk_click_grid(samples: int = 41, radius: float = 0.985) -> tuple[np.ndarray, np.ndarray]:
    vals = np.linspace(-float(radius), float(radius), int(samples))
    X, Y = np.meshgrid(vals, vals)
    mask = (X * X + Y * Y) < float(radius) ** 2
    return X[mask], Y[mask]


def _covariance_ellipse_2d(
    center: np.ndarray,
    covariance: np.ndarray,
    *,
    scale: float = 1.0,
    samples: int = 160,
) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(0.5 * (covariance + covariance.T))
    vals = np.maximum(vals, 0.0)
    theta = np.linspace(0.0, TAU, int(samples))
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    ellipse = center.reshape(2, 1) + float(scale) * vecs @ (np.sqrt(vals).reshape(2, 1) * circle)
    return ellipse[0], ellipse[1]


def _reflected_series_np(w_series: np.ndarray, points: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Vectorized r_i(w)=H_{p_i-w}(p_i) for many reduced points."""
    W = np.asarray(w_series, dtype=np.float64).reshape(-1, 2)
    P = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    normals = P[None, :, :] - W[:, None, :]
    denom = np.maximum(np.sum(normals * normals, axis=-1, keepdims=True), float(eps))
    scale = 2.0 * np.sum(P[None, :, :] * normals, axis=-1, keepdims=True) / denom
    reflected = P[None, :, :] - scale * normals
    reflected /= np.maximum(np.linalg.norm(reflected, axis=-1, keepdims=True), float(eps))
    return reflected


def _mobius_sphere_series_np(w_series: np.ndarray, points_series: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Vectorized M_w(x) for w shape [T,2] and x shape [T,N,2]."""
    W = np.asarray(w_series, dtype=np.float64).reshape(-1, 2)
    X = np.asarray(points_series, dtype=np.float64)
    diff = X - W[:, None, :]
    denom = np.maximum(np.sum(diff * diff, axis=-1, keepdims=True), float(eps))
    w2 = np.sum(W * W, axis=1).reshape(-1, 1, 1)
    return (1.0 - w2) * diff / denom - W[:, None, :]


def _spherical_inversion_chart_np(x: np.ndarray, center: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Pointwise spherical inversion chart y=(x-center)/|x-center|^2."""
    X = np.asarray(x, dtype=np.float64)
    original_shape = X.shape
    X2 = X.reshape(-1, 2)
    c = np.asarray(center, dtype=np.float64).reshape(2)
    diff = X2 - c[None, :]
    den = np.sum(diff * diff, axis=1)
    out = diff / np.maximum(den, float(eps))[:, None]
    out[den < float(eps)] = np.nan
    return out.reshape(original_shape)


def _one_sided_unit_distance_chart_np(x: np.ndarray, center: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Keep points at least unit distance from center, preserving direction."""
    X = np.asarray(x, dtype=np.float64)
    original_shape = X.shape
    X2 = X.reshape(-1, 2)
    c = np.asarray(center, dtype=np.float64).reshape(2)
    diff = X2 - c[None, :]
    dist = np.linalg.norm(diff, axis=1)
    out = X2.copy()
    mask = (dist < 1.0) & (dist >= float(eps))
    out[mask] = c[None, :] + diff[mask] / dist[mask, None]
    out[dist < float(eps)] = c
    return out.reshape(original_shape)


def _phase_grids_for_reflected_series_np(
    reflected_series: np.ndarray,
    weights: np.ndarray,
    *,
    grid_size: int,
    radius: float = 0.995,
    clip_quantile: float = 0.03,
    max_elements: int = 18_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate S_{r(w)}(u) for all frames in chunks.

    The heavy work is batched over frames/grid/reflected points.  Marching-square
    segment extraction is still per frame, but it consumes these cached grids
    and never runs in animation callbacks.
    """
    Q = np.asarray(reflected_series, dtype=np.float64)
    a = np.asarray(weights, dtype=np.float64).reshape(-1)
    frames, n_points, _ = Q.shape
    g = max(8, int(grid_size))
    xs = np.linspace(-float(radius), float(radius), g, dtype=np.float64)
    ys = np.linspace(-float(radius), float(radius), g, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    U = np.column_stack([X.ravel(), Y.ravel()])
    rr = np.sum(U * U, axis=1)
    mask = rr < float(radius) ** 2
    denom_log = np.log(np.maximum(1.0 - rr, 1e-12))
    M = U.shape[0]
    chunk = max(1, min(frames, int(max_elements // max(1, M * n_points))))
    out = np.empty((frames, g, g), dtype=np.float64)
    for start in range(0, frames, chunk):
        stop = min(frames, start + chunk)
        diff = U[None, :, None, :] - Q[start:stop, None, :, :]
        diff2 = np.maximum(np.sum(diff * diff, axis=-1), 1e-12)
        vals = np.einsum("n,bmn->bm", a, np.log(diff2), optimize=True) - denom_log[None, :]
        vals[:, ~mask] = np.nan
        if 0.0 < float(clip_quantile) < 0.49:
            lo = np.nanquantile(vals, float(clip_quantile), axis=1)
            hi = np.nanquantile(vals, 1.0 - float(clip_quantile), axis=1)
            vals = np.clip(vals, lo[:, None], hi[:, None])
        out[start:stop] = vals.reshape(stop - start, g, g)
    return xs, ys, out


def _weighted_busemann_phase_series_np(
    w_series: np.ndarray,
    points: np.ndarray,
    weights: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return S_{p^0}(w_t)=sum_i a_i log(|w_t-p_i^0|^2/(1-|w_t|^2)).

    The convention in the reconstruction note is
    Phi_{p^0}(w)=-S_{p^0}(w).
    """
    W = np.asarray(w_series, dtype=np.float64).reshape(-1, 2)
    P = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    a = np.asarray(weights, dtype=np.float64).reshape(-1)
    w2 = np.sum(W * W, axis=1)
    diff = W[:, None, :] - P[None, :, :]
    numer_log = np.log(np.maximum(np.sum(diff * diff, axis=-1), float(eps)))
    denom_log = np.log(np.maximum(1.0 - w2, float(eps)))[:, None]
    return np.einsum("n,tn->t", a, numer_log - denom_log, optimize=True)


def lorentz_inner(x: Tensor | np.ndarray, y: Tensor | np.ndarray) -> Tensor:
    """Return the Lorentz inner product -x0*y0 + x_spatial dot y_spatial."""
    X = _as_tensor(x)
    Y = _as_tensor(y, dtype=X.dtype).to(dtype=X.dtype, device=X.device)
    return -(X[..., 0] * Y[..., 0]) + (X[..., 1:] * Y[..., 1:]).sum(dim=-1)


def hyperboloid_lift(w: Tensor | np.ndarray, *, eps: float = 1e-12) -> Tensor:
    """Lift ball coordinates w in B^2 to X(w) in the hyperboloid model."""
    W = _as_tensor(w)
    if W.shape[-1] != 2:
        raise ValueError("hyperboloid_lift currently expects final dimension 2.")
    w2 = (W * W).sum(dim=-1)
    denom = (1.0 - w2).clamp(min=float(eps))
    x0 = (1.0 + w2) / denom
    xs = 2.0 * W / denom.unsqueeze(-1)
    return torch.cat([x0.unsqueeze(-1), xs], dim=-1)


def null_lift(points: Tensor | np.ndarray) -> Tensor:
    """Lift boundary constants p_i^0 in S^1 to null vectors ell_i=(1,p_i^0)."""
    P, _ = _prepare_points_weights(points)
    ones = torch.ones((P.shape[0], 1), dtype=P.dtype, device=P.device)
    return torch.cat([ones, P], dim=-1)


def hyperboloid_ray_fields(
    w: Tensor | np.ndarray,
    points: Tensor | np.ndarray,
    weights: Tensor | np.ndarray | None = None,
    *,
    eps: float = 1e-12,
) -> dict[str, Tensor]:
    """Return X, ell_i, alpha_i, u_i and U for the hyperboloid optical chart."""
    P, a = _prepare_points_weights(points, weights)
    W = _as_tensor(w, dtype=P.dtype).to(dtype=P.dtype, device=P.device)
    X = hyperboloid_lift(W, eps=eps)
    ell = null_lift(P)
    X_series = X.reshape((-1, 3))
    alpha = (-lorentz_inner(X_series[:, None, :], ell[None, :, :])).clamp(min=float(eps))
    rays = ell[None, :, :] / alpha[:, :, None] - X_series[:, None, :]
    U = (a[None, :, None] * rays).sum(dim=1)
    if X.dim() == 1:
        return {
            "X": X_series[0],
            "ell": ell,
            "alpha": alpha[0],
            "u": rays[0],
            "U": U[0],
        }
    return {
        "X": X_series.reshape(*X.shape[:-1], 3),
        "ell": ell,
        "alpha": alpha.reshape(*X.shape[:-1], P.shape[0]),
        "u": rays.reshape(*X.shape[:-1], P.shape[0], 3),
        "U": U.reshape(*X.shape[:-1], 3),
    }


def _lorentz_inner_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = np.asarray(x, dtype=np.float64)
    Y = np.asarray(y, dtype=np.float64)
    return -(X[..., 0] * Y[..., 0]) + np.sum(X[..., 1:] * Y[..., 1:], axis=-1)


def _hyperboloid_lift_series_2d(w_series: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    W = np.asarray(w_series, dtype=np.float64).reshape(-1, 2)
    w2 = np.sum(W * W, axis=1)
    denom = np.maximum(1.0 - w2, float(eps))
    return np.column_stack([(1.0 + w2) / denom, 2.0 * W[:, 0] / denom, 2.0 * W[:, 1] / denom])


def _null_lift_points_2d(points: np.ndarray) -> np.ndarray:
    P = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return np.column_stack([np.ones(P.shape[0], dtype=np.float64), P])


def _hyperboloid_ray_fields_series_2d(
    w_series: np.ndarray,
    points: np.ndarray,
    weights: np.ndarray,
    *,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    X = _hyperboloid_lift_series_2d(w_series, eps=eps)
    ell = _null_lift_points_2d(points)
    alpha = np.maximum(-_lorentz_inner_np(X[:, None, :], ell[None, :, :]), float(eps))
    rays = ell[None, :, :] / alpha[:, :, None] - X[:, None, :]
    a = np.asarray(weights, dtype=np.float64).reshape(-1)
    U = np.einsum("n,tnj->tj", a, rays, optimize=True)
    return {"X": X, "ell": ell, "alpha": alpha, "u": rays, "U": U}


def _hyperboloid_tangent_from_ball_direction_series_2d(
    w_series: np.ndarray,
    directions: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Lift Euclidean ball directions to unit hyperboloid tangent vectors."""
    W = np.asarray(w_series, dtype=np.float64).reshape(-1, 2)
    E = np.asarray(directions, dtype=np.float64).reshape(-1, 2)
    e_norm = np.maximum(np.linalg.norm(E, axis=1, keepdims=True), float(eps))
    E = E / e_norm
    wdot = np.sum(W * E, axis=1)
    w2 = np.sum(W * W, axis=1)
    denom = np.maximum(1.0 - w2, float(eps))
    dX0 = 4.0 * wdot / (denom * denom)
    dXs = 2.0 * E / denom[:, None] + 4.0 * wdot[:, None] * W / (denom * denom)[:, None]
    dX = np.column_stack([dX0, dXs])
    unit = 0.5 * denom[:, None] * dX
    norm = np.sqrt(np.maximum(_lorentz_inner_np(unit, unit), float(eps)))
    return unit / norm[:, None]


def _screen_directions_from_orbit_2d(
    w_series: np.ndarray,
    r_series: np.ndarray,
    w_star: np.ndarray,
    *,
    eps: float = 1e-10,
) -> np.ndarray:
    """Return deterministic Euclidean screen directions transverse to R."""
    W = np.asarray(w_series, dtype=np.float64).reshape(-1, 2)
    R = np.asarray(r_series, dtype=np.float64).reshape(-1, 2)
    w0 = np.asarray(w_star, dtype=np.float64).reshape(2)
    out = np.empty_like(W)
    fallback = w0.copy()
    if float(np.linalg.norm(fallback)) <= eps:
        fallback = np.array([1.0, 0.0], dtype=np.float64)
    for i in range(W.shape[0]):
        center = R[i]
        if float(np.linalg.norm(center)) <= eps:
            if i + 1 < W.shape[0] and float(np.linalg.norm(W[i + 1] - W[i])) > eps:
                center = W[i + 1] - W[i]
            elif i > 0 and float(np.linalg.norm(W[i] - W[i - 1])) > eps:
                center = W[i] - W[i - 1]
            else:
                center = fallback
        perp = np.array([-center[1], center[0]], dtype=np.float64)
        nrm = float(np.linalg.norm(perp))
        if nrm <= eps:
            perp = np.array([0.0, 1.0], dtype=np.float64)
            nrm = 1.0
        out[i] = perp / nrm
    return out


def _hyperboloid_surface_grid_2d(radius: float = 2.4, samples: int = 34) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radial = np.linspace(0.0, float(radius), max(8, int(samples)), dtype=np.float64)
    theta = np.linspace(0.0, TAU, max(24, int(samples) * 2), dtype=np.float64)
    rr, tt = np.meshgrid(radial, theta, indexing="ij")
    x = rr * np.cos(tt)
    y = rr * np.sin(tt)
    z = np.sqrt(1.0 + rr * rr)
    return x, y, z


def _hyperboloid_radial_geodesic_lines_2d(
    points: np.ndarray,
    *,
    top_height: float,
    samples: int = 90,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Return geodesics X(r p_i^0) ending at the chosen hyperboloid height."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    height = max(1.02, float(top_height))
    r_max = math.sqrt(max(0.0, (height - 1.0) / (height + 1.0)))
    r_vals = np.linspace(0.0, min(r_max, 0.999999), int(samples), dtype=np.float64)
    denom = np.maximum(1.0 - r_vals * r_vals, 1e-12)
    x0 = (1.0 + r_vals * r_vals) / denom
    spatial_scale = 2.0 * r_vals / denom
    out_x: list[float | None] = []
    out_y: list[float | None] = []
    out_z: list[float | None] = []
    for p in P:
        curve = spatial_scale[:, None] * p[None, :]
        out_x.extend(curve[:, 0].tolist())
        out_y.extend(curve[:, 1].tolist())
        out_z.extend(x0.tolist())
        out_x.append(None)
        out_y.append(None)
        out_z.append(None)
    return out_x, out_y, out_z


def _weighted_cayley_chart_series_2d(
    w_series: np.ndarray,
    points: np.ndarray,
    weights: np.ndarray,
    xi: np.ndarray,
    *,
    log_clip: float = 24.0,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Modified Cayley chart with lambda=exp(-S_{p^0}(w)).

    For the usual Cayley chart centered at xi, lambda=exp(-B_xi(w)).
    Here the horizontal direction still uses the xi-tangential normalized
    action mathfrak{S}_xi(w)=2 P_xi^perp(w)/(1-|w|^2), but lambda is replaced
    by the weighted total Busemann phase of the frozen constants p_i^0.
    """
    W = np.asarray(w_series, dtype=np.float64).reshape(-1, 2)
    xi_arr = np.asarray(xi, dtype=np.float64).reshape(2)
    xi_arr = xi_arr / max(float(np.linalg.norm(xi_arr)), eps)
    e_perp = np.array([-xi_arr[1], xi_arr[0]], dtype=np.float64)
    phase = _weighted_busemann_phase_series_np(W, points, weights, eps=eps)
    log_lambda = np.clip(-phase, -float(log_clip), float(log_clip))
    lam = np.exp(log_lambda)
    w2 = np.sum(W * W, axis=1)
    tangent_scalar = 2.0 * (W @ e_perp) / np.maximum(1.0 - w2, eps)
    u = lam * tangent_scalar
    return {
        "u": u,
        "lambda": lam,
        "phase": phase,
        "log_lambda": log_lambda,
        "xi": xi_arr,
        "e_perp": e_perp,
    }


def _orientation_series_from_w_2d(
    w_series: np.ndarray,
    fallback_xi: np.ndarray,
    *,
    eps: float = 1e-10,
) -> np.ndarray:
    """Return xi_t=w_t/|w_t| with previous/fallback orientation near zero."""
    W = np.asarray(w_series, dtype=np.float64).reshape(-1, 2)
    fb = np.asarray(fallback_xi, dtype=np.float64).reshape(2)
    fb = fb / max(float(np.linalg.norm(fb)), eps)
    out = np.empty_like(W)
    last = fb
    for i, w in enumerate(W):
        n = float(np.linalg.norm(w))
        if n > eps:
            last = w / n
        out[i] = last
    return out


def _regular_cayley_chart_series_2d(
    w_series: np.ndarray,
    points: np.ndarray,
    weights: np.ndarray,
    fallback_xi: np.ndarray,
    *,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Regular moving-puncture Cayley chart with xi_t=w_t/|w_t|.

    This uses the single Busemann factor B_{xi_t}(w_t):

        lambda_t = exp(-B_{xi_t}(w_t))
                 = (1-|w_t|^2)/|w_t-xi_t|^2.

    Since xi_t is the current w-orientation, the current point usually lies on
    the vertical axis u=0.  Frozen boundary constants p_i^0 are re-charted at
    each frame.
    """
    W = np.asarray(w_series, dtype=np.float64).reshape(-1, 2)
    P = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    Xi = _orientation_series_from_w_2d(W, fallback_xi, eps=max(eps, 1e-10))
    Eperp = np.column_stack([-Xi[:, 1], Xi[:, 0]])
    w2 = np.sum(W * W, axis=1)
    denom = np.maximum(np.sum((W - Xi) ** 2, axis=1), eps)
    lam = np.maximum(1.0 - w2, eps) / denom
    u = 2.0 * np.sum(W * Eperp, axis=1) / denom
    phase = -np.log(np.maximum(lam, eps))
    diff = P[None, :, :] - Xi[:, None, :]
    raw_boundary_denom = np.sum(diff * diff, axis=-1)
    boundary = 2.0 * (Eperp @ P.T) / np.maximum(raw_boundary_denom, eps)
    boundary[raw_boundary_denom < 1e-8] = np.nan
    finite = boundary[np.isfinite(boundary)]
    if finite.size > 0:
        cap = max(1.0, float(np.nanquantile(np.abs(finite), 0.95)))
        boundary = np.clip(boundary, -1.25 * cap, 1.25 * cap)
    return {
        "u": u,
        "lambda": lam,
        "phase": phase,
        "xi": Xi,
        "e_perp": Eperp,
        "boundary": boundary,
    }


def _cayley_boundary_coordinates_2d(
    points: np.ndarray,
    xi: np.ndarray,
    *,
    cap_quantile: float = 0.95,
    eps: float = 1e-12,
) -> np.ndarray:
    """Boundary Cayley coordinates b_eta projected onto xi^perp.

    Points too close to the puncture xi are clipped for finite display.
    """
    P = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    xi_arr = np.asarray(xi, dtype=np.float64).reshape(2)
    xi_arr = xi_arr / max(float(np.linalg.norm(xi_arr)), eps)
    e_perp = np.array([-xi_arr[1], xi_arr[0]], dtype=np.float64)
    raw_denom = np.sum((P - xi_arr[None, :]) ** 2, axis=1)
    near_puncture = raw_denom < 1e-8
    denom = np.maximum(raw_denom, eps)
    b = 2.0 * (P @ e_perp) / denom
    b[near_puncture] = np.nan
    finite = b[np.isfinite(b)]
    if finite.size > 0:
        cap = float(np.quantile(np.abs(finite), float(cap_quantile)))
        cap = max(cap, 1.0)
        b = np.clip(b, -1.25 * cap, 1.25 * cap)
    return b


class LMSOpticalDiskBaseWidget:
    """Reusable base for 2D optical widgets.

    The base class owns the expensive and interaction-sensitive parts:
    controls, exact gauge preprocessing, orbit integration, batched frame
    payload precompute, cache invalidation, and frame/play callbacks.

    Subclasses can change the visualization without copying the LMS optics
    helpers by overriding `_build_figure`, `_apply_payload_to_figure`, and
    optionally `_layout_header_html` / `_augment_frame_payload`.

    Left panel: the exact constants of motion p_i^0=M_{-w_*}(x_i^0)
    and the one-dimensional reduced orbit w(t) initialized at w(0)=w_*.

    Right panel: the reflected cloud r_i(w)=H_{p_i^0-w}(p_i^0).  The Busemann
    level sets shown there are the level sets of S_{r(w)}(u), and the current
    w is the exact Busemann balance of r_i(w) up to numerical centering error.
    The motion is still driven by the frozen constants p_i^0 through
    dw/dt=0.5*(1-|w|^2)*sum_i a_i r_i(w), not by the gradient of S_{r(w)} at w.
    """

    _ES_BOUNDARY_TARGET_RADIUS = 0.9995
    _ES_BOUNDARY_FIT_TOLERANCE = 5e-4

    def __init__(
        self,
        *,
        N: int = 14,
        preset: OpticalPreset = "random",
        seed: int = 7,
        init_radius: float = 0.35,
        grid_size: int = 110,
        width: int = 1040,
        height: int = 760,
        points: np.ndarray | Tensor | None = None,
        weights: np.ndarray | Tensor | None = None,
    ) -> None:
        _require_widgets()
        self.rng = np.random.default_rng(int(seed))
        self.init_radius_value = float(np.clip(float(init_radius), 0.0, 0.985))
        self.grid_size = int(grid_size)
        # Contours are visual context, not integration data.  Capping their
        # precompute grid keeps notebook interaction responsive even when old
        # notebooks pass large grid_size values.
        self.level_grid_size = max(36, min(int(grid_size), 80))
        self.width = int(width)
        self.height = int(height)
        self._updating = False
        self._frame_index = 0
        self._cache_valid = False
        self._frame_payloads: list[dict[str, Any]] = []
        self._es_step_fit_cache: dict[tuple[str, int, float], dict[str, float | bool]] = {}
        self._es_step_fit_last: dict[str, Any] | None = None
        if points is None:
            self.raw_points = _initialized_observed_cloud_np(
                int(N),
                self.rng,
                preset=preset,
                target_radius=self.init_radius_value,
            )
        else:
            self.raw_points = _prepare_points_weights(points, weights)[0].detach().cpu().numpy()
        self.weights = (
            np.full((self.raw_points.shape[0],), 1.0 / float(self.raw_points.shape[0]), dtype=np.float64)
            if weights is None
            else _prepare_points_weights(self.raw_points, weights)[1].detach().cpu().numpy()
        )
        self.points = self._canonicalized_points(self.raw_points)
        self._orbit = np.zeros((1, 2), dtype=np.float64)
        self._orbit_r = np.zeros(1, dtype=np.float64)
        self._build_controls(preset=preset)
        self._build_figure()
        self._bind_callbacks()
        self._sync_anchor_controls()
        self._rebuild_orbit()
        self.layout = widgets.VBox(
            [
                _html(_sanitize_plot_text(self._layout_header_html())),
                self.fig,
                self.controls,
                self.stats_html,
            ]
        )

    def _layout_header_html(self) -> str:
        return (
            "<b>LMS optical reflected-ray explorer</b><br>"
            "Input cloud $\\{x_i^0\\}$ is exactly deboosted to constants "
            "$p_i^0=M_{-w_\\ast}(x_i^0)$, then $w(0)=w_\\ast$ is evolved by the frozen LMS field."
        )

    def _canonicalized_points(self, points: np.ndarray) -> np.ndarray:
        P, a = _prepare_points_weights(points, self.weights)
        try:
            state = canonical_cloud(P, a, max_iters=120, tol=1e-10)
            self.z_star = state.z.detach().cpu().numpy().astype(np.float64)
            self.w_star = state.w.detach().cpu().numpy().astype(np.float64)
            self._center_error = float(state.center_error)
            return state.P.detach().cpu().numpy().astype(np.float64)
        except Exception:
            centered = P.detach().cpu().numpy().astype(np.float64)
            self.z_star = np.zeros(centered.shape[1], dtype=np.float64)
            self.w_star = np.zeros(centered.shape[1], dtype=np.float64)
            self._center_error = float(np.linalg.norm((self.weights[:, None] * centered).sum(axis=0)))
            return centered

    def _build_controls(self, *, preset: OpticalPreset) -> None:
        self.preset_dropdown = widgets.Dropdown(
            options=[("Random", "random"), ("Balanced", "balanced"), ("Clustered", "clustered"), ("Dipole", "dipole")],
            value=preset,
            description="x^0 preset",
            layout=widgets.Layout(width="210px"),
        )
        self.n_slider = widgets.IntSlider(
            value=int(self.points.shape[0]),
            min=3,
            max=80,
            step=1,
            description="N",
            continuous_update=False,
            layout=widgets.Layout(width="240px"),
        )
        self.init_radius_slider = widgets.FloatSlider(
            value=float(self.init_radius_value),
            min=0.0,
            max=0.985,
            step=0.005,
            description="init |w*|",
            readout_format=".3f",
            continuous_update=False,
            layout=widgets.Layout(width="250px"),
        )
        self.exact_center_html = _html(value="", layout=widgets.Layout(width="460px"))
        self.time_mode_toggle = widgets.ToggleButton(
            value=False,
            description="Time: physical",
            tooltip="Switch orbit precompute between physical time and Euler-Sundman time.",
            layout=widgets.Layout(width="150px"),
        )
        self.selected = widgets.IntSlider(value=0, min=0, max=max(0, int(self.points.shape[0]) - 1), step=1, description="selected i", continuous_update=False, layout=widgets.Layout(width="260px"))
        self.anchor_theta = widgets.FloatSlider(value=0.0, min=-math.pi, max=math.pi, step=0.01, description="edit x_i^0", readout_format=".2f", continuous_update=False, layout=widgets.Layout(width="330px"))
        self.step_size = widgets.FloatSlider(value=0.16, min=0.005, max=0.5, step=0.005, description="flow step", readout_format=".3f", continuous_update=False, layout=widgets.Layout(width="280px"))
        self.max_frames = widgets.IntSlider(value=20, min=2, max=300, step=1, description="max frames", continuous_update=False, layout=widgets.Layout(width="300px"))
        self.orbit_frames = self.max_frames
        self.frame_slider = widgets.IntSlider(value=0, min=0, max=0, step=1, description="frame", continuous_update=True, disabled=True, layout=widgets.Layout(width="640px"))
        self.frame_counter = widgets.HTML(value="frame 0 / 0", layout=widgets.Layout(width="110px"))
        self.show_contours = widgets.Checkbox(value=True, description="reflected S level sets", indent=False)
        self.btn_resample = widgets.Button(description="Resample x^0", layout=widgets.Layout(width="140px"))
        self.btn_precompute = widgets.Button(description="Precompute flow", layout=widgets.Layout(width="140px"))
        self.btn_rebuild = self.btn_precompute
        self.btn_step = widgets.Button(description="Step on orbit", disabled=True, layout=widgets.Layout(width="120px"))
        self.play = widgets.Play(
            value=0,
            min=0,
            max=0,
            step=1,
            interval=80,
            description="Play",
            disabled=True,
            show_repeat=True,
            layout=widgets.Layout(width="120px"),
        )
        if "repeat" in self.play.traits():
            self.play.repeat = True
        self.cache_status_html = widgets.HTML(value="", layout=widgets.Layout(width="360px"))
        self.stats_html = _html(value="")
        self.controls = widgets.VBox(
            [
                widgets.HBox([self.preset_dropdown, self.n_slider, self.init_radius_slider, self.btn_resample]),
                widgets.HBox([self.exact_center_html, self.time_mode_toggle, self.step_size, self.max_frames]),
                widgets.HBox([self.btn_step, self.play, self.btn_precompute, self.cache_status_html]),
                widgets.HBox([self.frame_slider, self.frame_counter]),
                widgets.HBox([self.selected, self.anchor_theta, self.show_contours]),
            ]
        )

    def _make_subplot_figure(self) -> Any:
        return make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                r"$p_i^0=M_{-w_\ast}(x_i^0),\quad w(t)$",
                r"$r_i(w)=H_{p_i^0-w}(p_i^0),\quad S_{r(w)}(u)=\mathrm{const}$",
                r"$R_{p^0}(w)=\sum_i a_i r_i(w)$",
                r"$p_i^0\mapsto r_i(w)$",
            ),
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
        )

    def _after_build_figure(self) -> None:
        """Hook for subclasses to add extra traces/layout after base traces."""

    def _reset_subplot_legend_state(self) -> None:
        self._subplot_legend_ids: dict[tuple[int, int], str] = {}
        self._trace_subplots: dict[str, tuple[int, int]] = {}

    def _legend_id_for_subplot(self, row: int, col: int) -> str:
        key = (int(row), int(col))
        if key not in self._subplot_legend_ids:
            idx = len(self._subplot_legend_ids) + 1
            self._subplot_legend_ids[key] = "legend" if idx == 1 else f"legend{idx}"
        return self._subplot_legend_ids[key]

    def _add_trace_to_subplot(
        self,
        fig: Any,
        trace: Any,
        key: str,
        row: int,
        col: int,
        *,
        secondary_y: bool = False,
    ) -> None:
        """Add a trace and attach it to the legend local to its subplot."""
        legend_id = self._legend_id_for_subplot(row, col)
        try:
            trace.legend = legend_id
        except Exception:
            pass
        fig.add_trace(trace, row=row, col=col, secondary_y=secondary_y)
        self.tr[key] = len(fig.data) - 1
        self._trace_subplots[key] = (int(row), int(col))

    @staticmethod
    def _layout_axis_name(axis_ref: str | None, axis_prefix: str) -> str:
        ref = axis_ref or axis_prefix
        if ref == axis_prefix:
            return f"{axis_prefix}axis"
        return f"{axis_prefix}axis{ref[len(axis_prefix):]}"

    def _trace_domain(self, trace: Any) -> tuple[list[float], list[float]]:
        scene_ref = getattr(trace, "scene", None)
        if scene_ref:
            scene = getattr(self.fig.layout, scene_ref)
            return list(scene.domain.x), list(scene.domain.y)
        xaxis_name = self._layout_axis_name(getattr(trace, "xaxis", None), "x")
        yaxis_name = self._layout_axis_name(getattr(trace, "yaxis", None), "y")
        xaxis = getattr(self.fig.layout, xaxis_name)
        yaxis = getattr(self.fig.layout, yaxis_name)
        return list(xaxis.domain), list(yaxis.domain)

    def _configure_subplot_legends(self) -> None:
        """Overlay one compact legend inside each subplot domain."""
        legend_layouts: dict[str, dict[str, Any]] = {}
        for subplot, legend_id in self._subplot_legend_ids.items():
            trace_indices = [
                self.tr[key]
                for key, trace_subplot in self._trace_subplots.items()
                if trace_subplot == subplot and key in self.tr
            ]
            visible_legend_indices = [
                idx for idx in trace_indices if getattr(self.fig.data[idx], "showlegend", None) is not False
            ]
            if not visible_legend_indices:
                continue
            xdom, ydom = self._trace_domain(self.fig.data[visible_legend_indices[0]])
            legend_layouts[legend_id] = dict(
                x=float(xdom[1]) - 0.012,
                y=float(ydom[1]) - 0.012,
                xanchor="right",
                yanchor="top",
                orientation="v",
                bgcolor="rgba(255,255,255,0.76)",
                bordercolor="rgba(40,40,40,0.18)",
                borderwidth=1,
                font=dict(size=10),
                itemsizing="constant",
            )
        if legend_layouts:
            self.fig.update_layout(showlegend=True, **legend_layouts)

    def _build_figure(self) -> None:
        fig = go.FigureWidget(
            self._make_subplot_figure()
        )
        self.fig = fig
        self.tr: dict[str, int] = {}
        self._reset_subplot_legend_state()

        def add(trace: Any, key: str, row: int, col: int) -> None:
            self._add_trace_to_subplot(fig, trace, key, row, col)

        cx, cy = _circle_xy()
        add(go.Scatter(x=cx.tolist(), y=cy.tolist(), mode="lines", line=dict(color="rgba(20,20,20,0.75)", width=2), name=r"$\partial\mathbb{B}^2$", hoverinfo="skip"), "disk", 1, 1)
        add(go.Scatter(x=[], y=[], mode="lines", line=dict(color="rgba(20,20,20,0.35)", width=2), name=r"$w(t)$", hoverinfo="skip"), "orbit_path", 1, 1)
        add(go.Scatter(x=[0.0], y=[0.0], mode="markers", marker=dict(size=11, color="white", line=dict(color="black", width=2)), name=r"$\sum_i a_i p_i^0=0$"), "core_balance", 1, 1)
        add(go.Scatter(x=[], y=[], mode="markers", marker=dict(size=8, color="#244C9A", opacity=0.72), name=r"$p_i^0$"), "anchors", 1, 1)
        add(go.Scatter(x=[], y=[], mode="markers", marker=dict(size=14, color="#F2A900", line=dict(color="black", width=1.4)), name=r"$\mathrm{selected}\ p_i^0$"), "selected_anchor", 1, 1)
        add(go.Scatter(x=[], y=[], mode="markers", marker=dict(size=13, color="black"), name=r"$w$"), "w", 1, 1)
        add(go.Scatter(x=[], y=[], mode="lines+markers", line=dict(color="#D72638", width=4), marker=dict(size=[0, 9], color="#D72638"), name=r"$R_{p^0}(w)=\sum_i a_i r_i(w)$"), "R_arrow", 1, 1)
        add(go.Scatter(x=[], y=[], mode="lines+markers", line=dict(color="#188038", width=3, dash="dash"), marker=dict(size=[0, 7], color="#188038"), name=r"$\dot w$"), "vel_arrow", 1, 1)

        add(go.Scatter(x=[], y=[], mode="lines", line=dict(color="rgba(78,60,150,0.82)", width=1.45), opacity=0.82, name=r"$S_{r(w)}(u)=\mathrm{const}$", hoverinfo="skip"), "ref_phase", 1, 2)
        add(go.Scatter(x=cx.tolist(), y=cy.tolist(), mode="lines", line=dict(color="rgba(20,20,20,0.75)", width=2), name=r"$\partial\mathbb{B}^2$", hoverinfo="skip", showlegend=False), "ref_circle", 1, 2)
        add(go.Scatter(x=[], y=[], mode="markers", marker=dict(size=8, color="#5B8C5A", opacity=0.78), name=r"$r_i(w)$"), "reflected", 1, 2)
        add(go.Scatter(x=[], y=[], mode="markers", marker=dict(size=13, color="black", symbol="x"), name=r"$w:\sum_i a_iM_w(r_i(w))=0$"), "ref_w", 1, 2)
        add(go.Scatter(x=[], y=[], mode="lines+markers", line=dict(color="#D72638", width=4), marker=dict(size=[0, 10], color="#D72638"), name=r"$R_{p^0}(w)$"), "R_circle", 1, 2)
        add(go.Scatter(x=[], y=[], mode="lines", line=dict(color="#5B8C5A", width=2), name=r"$\mathrm{Cov}(r_i(w))$", hoverinfo="skip"), "ellipse", 1, 2)
        add(go.Scatter(x=[], y=[], mode="lines+markers", line=dict(color="#3B5BDB", width=2), marker=dict(size=5, color="#3B5BDB"), name=r"$\sum_i a_i r_i(w)$"), "polygon", 2, 1)
        add(go.Scatter(x=[0.0], y=[0.0], mode="markers", marker=dict(size=9, color="black"), name=r"$0$", showlegend=False, hoverinfo="skip"), "polygon_origin", 2, 1)

        add(go.Scatter(x=cx.tolist(), y=cy.tolist(), mode="lines", line=dict(color="rgba(20,20,20,0.3)", width=1), name=r"$S^1$", hoverinfo="skip", showlegend=False), "geom_circle", 2, 2)
        add(go.Scatter(x=[], y=[], mode="lines", line=dict(color="#244C9A", width=3), name=r"$p_i^0$"), "geom_incoming", 2, 2)
        add(go.Scatter(x=[], y=[], mode="lines", line=dict(color="#5B8C5A", width=3), name=r"$r_i(w)$"), "geom_reflected", 2, 2)
        add(go.Scatter(x=[], y=[], mode="lines", line=dict(color="#D72638", width=3), name=r"$p_i^0-w$"), "geom_normal", 2, 2)
        add(go.Scatter(x=[], y=[], mode="lines", line=dict(color="rgba(0,0,0,0.45)", width=2, dash="dash"), name=r"$(p_i^0-w)^\perp$"), "geom_mirror", 2, 2)

        fig.update_layout(
            width=self.width,
            height=self.height,
            template="plotly_white",
            margin=dict(l=35, r=20, t=58, b=30),
            showlegend=True,
        )
        for axis in ("xaxis", "xaxis2", "xaxis3", "xaxis4"):
            fig.layout[axis].update(range=[-1.15, 1.15], zeroline=False, showgrid=False)
        for axis, anchor in (("yaxis", "x"), ("yaxis2", "x2"), ("yaxis3", "x3"), ("yaxis4", "x4")):
            fig.layout[axis].update(range=[-1.15, 1.15], zeroline=False, showgrid=False, scaleanchor=anchor, scaleratio=1)
        for row, col in ((1, 1), (1, 2), (2, 1), (2, 2)):
            fig.update_xaxes(title_text=r"$e_1$", row=row, col=col)
            fig.update_yaxes(title_text=r"$e_2$", row=row, col=col)
        self._after_build_figure()
        _sanitize_figure_text(fig)
        self._configure_subplot_legends()

    def _bind_callbacks(self) -> None:
        for ctl in [
            self.preset_dropdown,
            self.n_slider,
            self.init_radius_slider,
            self.step_size,
            self.max_frames,
            self.frame_slider,
            self.selected,
            self.anchor_theta,
            self.show_contours,
        ]:
            ctl.observe(self._on_control_change, names="value")
        self.time_mode_toggle.observe(self._on_time_mode_toggle, names="value")
        self.btn_resample.on_click(self._on_resample)
        self.btn_precompute.on_click(self._on_rebuild_orbit_clicked)
        self.btn_step.on_click(self._on_step)
        self.play.observe(self._on_play_tick, names="value")

    def _time_mode(self) -> Literal["physical", "euler_sundman"]:
        return "euler_sundman" if bool(self.time_mode_toggle.value) else "physical"

    def _sync_time_mode_label(self) -> None:
        self.time_mode_toggle.description = "Time: ES" if self._time_mode() == "euler_sundman" else "Time: physical"

    def _on_time_mode_toggle(self, change: dict[str, Any]) -> None:
        if self._updating or change.get("name") != "value":
            return
        self._sync_time_mode_label()
        self._rebuild_orbit()

    def _sync_anchor_controls(self) -> None:
        idx = int(np.clip(int(self.selected.value), 0, self.points.shape[0] - 1))
        self.selected.max = max(0, self.points.shape[0] - 1)
        theta = math.atan2(float(self.raw_points[idx, 1]), float(self.raw_points[idx, 0]))
        self._updating = True
        self.anchor_theta.value = theta
        self._updating = False
        self._sync_exact_center_label()

    def _sync_exact_center_label(self) -> None:
        w = np.asarray(getattr(self, "w_star", np.zeros(2)), dtype=np.float64).reshape(2)
        r = float(np.linalg.norm(w))
        theta = math.atan2(float(w[1]), float(w[0])) if r > 1e-12 else 0.0
        self.exact_center_html.value = (
            f"<b>Exact gauge:</b> "
            f"w*=({w[0]:+.4f},{w[1]:+.4f}), "
            f"|w*|={r:.4f}, arg(w*)={theta:+.3f}"
        )

    def _mark_cache_stale(self, message: str = "Parameters changed. Precompute flow before playback.") -> None:
        self._cache_valid = False
        self.btn_precompute.disabled = False
        self.btn_precompute.button_style = "warning"
        self.btn_step.disabled = True
        self.play.disabled = True
        self.frame_slider.disabled = True
        self.cache_status_html.value = f"<span style='color:#9a6700'>{message}</span>"
        self._sync_frame_controls(0)

    def _mark_cache_ready(self, message: str = "Flow cache ready.") -> None:
        self._cache_valid = True
        count = int(len(self._orbit))
        self.btn_precompute.disabled = False
        self.btn_precompute.button_style = "success"
        self.btn_step.disabled = count <= 1
        self.play.disabled = count <= 1
        self.frame_slider.disabled = count <= 1
        self.cache_status_html.value = f"<span style='color:#188038'>{message}</span>"
        self._sync_frame_controls(self._frame_index)

    def _initial_w(self) -> np.ndarray:
        """Return the exact reduced initial state w(0)=w_*."""
        return np.asarray(getattr(self, "w_star", np.zeros(2)), dtype=np.float64).reshape(2).copy()

    def _integrate_orbit_from_controls(self) -> np.ndarray:
        frames = max(2, int(self.max_frames.value))
        if self._time_mode() == "euler_sundman":
            return self._integrate_euler_sundman_orbit_from_controls(frames)
        pts = [self._initial_w()]
        w = pts[0].copy()
        step = float(self.step_size.value)
        parameter_times = [0.0]
        physical_times = [0.0]
        for _ in range(frames - 1):
            nxt = optical_flow_step(
                w,
                self.points,
                self.weights,
                step_size=step,
                radius=0.985,
            ).detach().cpu().numpy()
            if not np.isfinite(nxt).all():
                break
            pts.append(nxt.copy())
            parameter_times.append(parameter_times[-1] + step)
            physical_times.append(physical_times[-1] + step)
            w = nxt
            if float(np.linalg.norm(w)) >= 0.984:
                break
        self._orbit_parameter_time = np.asarray(parameter_times, dtype=np.float64)
        self._orbit_physical_time = np.asarray(physical_times, dtype=np.float64)
        return np.asarray(pts, dtype=np.float64)

    def _es_step_state_signature(self) -> str:
        h = hashlib.blake2b(digest_size=16)
        for values in (self.points, self.weights, self._initial_w()):
            arr = np.ascontiguousarray(values, dtype=np.float64)
            h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
            h.update(arr.tobytes())
        return h.hexdigest()

    def _es_step_fit_cache_key(self, frames: int, radius_stop: float) -> tuple[str, int, float]:
        return (self._es_step_state_signature(), int(frames), round(float(radius_stop), 12))

    def _remember_es_step_fit(
        self,
        key: tuple[str, int, float],
        *,
        tau_step: float,
        final_radius: float,
        boundary_fit: bool,
    ) -> None:
        self._es_step_fit_cache[key] = {
            "tau_step": float(tau_step),
            "final_radius": float(final_radius),
            "boundary_fit": bool(boundary_fit),
        }
        while len(self._es_step_fit_cache) > 64:
            self._es_step_fit_cache.pop(next(iter(self._es_step_fit_cache)))
        self._es_step_fit_last = {
            "state_signature": key[0],
            "frames": int(key[1]),
            "radius_stop": float(key[2]),
            "tau_step": float(tau_step),
        }

    def _estimate_es_boundary_tau_step(
        self,
        *,
        frames: int,
        radius_stop: float,
        points: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Return a cold-start ES step prior on the observed useful scale."""
        frame_intervals = max(1, int(frames) - 1)
        horizon_scale = 19.0 / float(frame_intervals)
        fallback = 0.02 * horizon_scale
        w0 = self._initial_w()
        r0 = float(np.linalg.norm(w0))
        if not (np.isfinite(r0) and r0 < float(radius_stop)):
            return fallback
        q0 = _reflected_series_np(w0.reshape(1, 2), points)[0]
        R0 = np.einsum("n,nj->j", weights, q0, optimize=True)
        radial = float(np.dot(R0, w0 / max(r0, 1e-12)))
        if radial <= 1e-12:
            return 0.05 * horizon_scale
        numerator = (float(radius_stop) - float(radius_stop) ** 3 / 3.0) - (r0 - r0**3 / 3.0)
        estimate = 0.42 * numerator / (2.0 * radial * float(frame_intervals))
        lo = 0.01 * horizon_scale
        hi = 0.05 * horizon_scale
        return float(np.clip(estimate, lo, hi))

    def _integrate_euler_sundman_orbit_from_controls(self, frames: int) -> np.ndarray:
        """Integrate dw/dtau with one cached frame per selected ES time step.

        The user-facing step slider is treated as the nominal ES step.  When
        that step would cross the display boundary before the requested frame
        count, bracket the crossing step and bisect to make the final requested
        frame land on the boundary threshold instead of collapsing the cache to
        the first few pre-boundary frames.
        """
        P = np.asarray(self.points, dtype=np.float64)
        a = np.asarray(self.weights, dtype=np.float64)
        requested_step = float(self.step_size.value)
        radius_stop = float(self._ES_BOUNDARY_TARGET_RADIUS)
        fit_tolerance = float(self._ES_BOUNDARY_FIT_TOLERANCE)
        cache_key = self._es_step_fit_cache_key(frames, radius_stop)
        state_signature = cache_key[0]
        fit_calls = 0
        fit_point_steps = 0

        def integrate_for_step(tau_step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, float]:
            nonlocal fit_calls, fit_point_steps
            fit_calls += 1
            pts = [self._initial_w()]
            parameter_times = [0.0]
            physical_times = [0.0]
            w = pts[0].copy()
            for _ in range(frames - 1):
                fit_point_steps += 1
                w2_now = float(np.dot(w, w))
                denom = max(1.0 - w2_now, 1e-8)
                q = _reflected_series_np(w.reshape(1, 2), P)[0]
                R = np.einsum("n,nj->j", a, q, optimize=True)
                nxt = w + tau_step * (2.0 * R / denom)
                nrm = float(np.linalg.norm(nxt))
                if (not np.isfinite(nxt).all()) or nrm >= radius_stop:
                    return (
                        np.asarray(pts, dtype=np.float64),
                        np.asarray(parameter_times, dtype=np.float64),
                        np.asarray(physical_times, dtype=np.float64),
                        True,
                        nrm,
                    )
                pts.append(nxt.copy())
                parameter_times.append(parameter_times[-1] + tau_step)
                physical_times.append(physical_times[-1] + tau_step * 4.0 / (denom * denom))
                w = nxt
            orbit = np.asarray(pts, dtype=np.float64)
            tau = np.asarray(parameter_times, dtype=np.float64)
            physical = np.asarray(physical_times, dtype=np.float64)
            final_radius = float(np.linalg.norm(orbit[-1])) if len(orbit) else float("nan")
            return orbit, tau, physical, False, final_radius

        cached_fit = self._es_step_fit_cache.get(cache_key)
        if cached_fit is not None:
            cached_step = float(cached_fit["tau_step"])
            orbit, tau, physical, crossed, final_radius = integrate_for_step(cached_step)
            if (not crossed) and len(orbit) == frames and abs(final_radius - radius_stop) <= fit_tolerance:
                self._orbit_parameter_time = tau
                self._orbit_physical_time = physical
                self._orbit_es_tau_step = cached_step
                self._orbit_es_boundary_fit = True
                self._orbit_es_fit_calls = fit_calls
                self._orbit_es_fit_point_steps = fit_point_steps
                self._orbit_es_fit_cache_hit = True
                return orbit

        initial_step = self._estimate_es_boundary_tau_step(
            frames=frames,
            radius_stop=radius_stop,
            points=P,
            weights=a,
        )
        warm_started = False
        if (
            cached_fit is None
            and self._es_step_fit_last is not None
            and self._es_step_fit_last.get("state_signature") == state_signature
            and np.isfinite(float(self._es_step_fit_last.get("tau_step", requested_step)))
        ):
            last_step = float(self._es_step_fit_last["tau_step"])
            last_frames = int(self._es_step_fit_last.get("frames", frames))
            if frames > 1 and last_frames > 1:
                initial_step = last_step * float(last_frames - 1) / float(frames - 1)
            else:
                initial_step = last_step
            warm_started = True
        elif cached_fit is not None:
            initial_step = float(cached_fit["tau_step"])
            warm_started = True

        orbit, tau, physical, crossed, final_radius = integrate_for_step(initial_step)
        if (not crossed) and len(orbit) == frames:
            lower_step = initial_step
            lower = (orbit, tau, physical)
            if final_radius < radius_stop - fit_tolerance:
                upper_step = initial_step
                upper_found = False
                for _ in range(16):
                    upper_step *= 1.15 if warm_started else 2.0
                    _, _, _, upper_crossed, upper_radius = integrate_for_step(upper_step)
                    if upper_crossed or upper_radius >= radius_stop:
                        upper_found = True
                        break
                if not upper_found:
                    best = lower
                    self._orbit_es_boundary_fit = False
                    self._orbit_es_tau_step = lower_step
                    self._orbit_parameter_time = best[1]
                    self._orbit_physical_time = best[2]
                    self._remember_es_step_fit(cache_key, tau_step=lower_step, final_radius=final_radius, boundary_fit=False)
                    self._orbit_es_fit_calls = fit_calls
                    self._orbit_es_fit_point_steps = fit_point_steps
                    self._orbit_es_fit_cache_hit = False
                    return best[0]
            else:
                best = lower
                self._orbit_es_boundary_fit = True
                self._orbit_es_tau_step = lower_step
                self._orbit_parameter_time = best[1]
                self._orbit_physical_time = best[2]
                self._remember_es_step_fit(cache_key, tau_step=lower_step, final_radius=final_radius, boundary_fit=True)
                self._orbit_es_fit_calls = fit_calls
                self._orbit_es_fit_point_steps = fit_point_steps
                self._orbit_es_fit_cache_hit = False
                return best[0]
        else:
            upper_step = initial_step
            lower_step = initial_step
            lower: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
            for _ in range(24):
                lower_step *= (1.0 / 1.15) if warm_started else 0.5
                lo_orbit, lo_tau, lo_physical, lo_crossed, _ = integrate_for_step(lower_step)
                if (not lo_crossed) and len(lo_orbit) == frames:
                    lower = (lo_orbit, lo_tau, lo_physical)
                    break
            if lower is None:
                self._orbit_es_boundary_fit = False
                self._orbit_es_tau_step = float(tau[1] - tau[0]) if tau.shape[0] > 1 else requested_step
                self._orbit_parameter_time = tau
                self._orbit_physical_time = physical
                self._remember_es_step_fit(
                    cache_key,
                    tau_step=float(tau[1] - tau[0]) if tau.shape[0] > 1 else requested_step,
                    final_radius=final_radius,
                    boundary_fit=False,
                )
                self._orbit_es_fit_calls = fit_calls
                self._orbit_es_fit_point_steps = fit_point_steps
                self._orbit_es_fit_cache_hit = False
                return orbit

        assert lower is not None
        best = lower
        for _ in range(32):
            mid_step = 0.5 * (lower_step + upper_step)
            mid_orbit, mid_tau, mid_physical, mid_crossed, _ = integrate_for_step(mid_step)
            if mid_crossed or len(mid_orbit) < frames:
                upper_step = mid_step
            else:
                lower_step = mid_step
                best = (mid_orbit, mid_tau, mid_physical)
                final_radius = float(np.linalg.norm(mid_orbit[-1]))
                if abs(final_radius - radius_stop) <= fit_tolerance:
                    break
        self._orbit_parameter_time = best[1]
        self._orbit_physical_time = best[2]
        self._orbit_es_tau_step = float(best[1][1] - best[1][0]) if best[1].shape[0] > 1 else requested_step
        self._orbit_es_boundary_fit = True
        final_radius = float(np.linalg.norm(best[0][-1])) if len(best[0]) else float("nan")
        self._remember_es_step_fit(
            cache_key,
            tau_step=self._orbit_es_tau_step,
            final_radius=final_radius,
            boundary_fit=True,
        )
        self._orbit_es_fit_calls = fit_calls
        self._orbit_es_fit_point_steps = fit_point_steps
        self._orbit_es_fit_cache_hit = False
        return best[0]

    def _build_preview_frame(self, message: str) -> None:
        w0 = self._initial_w()
        self._orbit = np.asarray([w0], dtype=np.float64)
        self._orbit_r = np.linalg.norm(self._orbit, axis=1)
        self._orbit_parameter_time = np.zeros(1, dtype=np.float64)
        self._orbit_physical_time = np.zeros(1, dtype=np.float64)
        # Preview updates must be cheap.  Level sets are part of the cached
        # animation payload and are populated only by Precompute flow.
        self._frame_payloads = self._build_frame_payloads(self._orbit, include_contours=False)
        self._frame_index = 0
        self._apply_cached_frame(0)
        self._mark_cache_stale(message)

    def _rebuild_orbit(self) -> None:
        self._cache_valid = False
        self.cache_status_html.value = "<span style='color:#666'>Computing all cached frames...</span>"
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
        self._orbit = self._integrate_orbit_from_controls()
        self._orbit_r = np.linalg.norm(self._orbit, axis=1)
        self._frame_payloads = self._build_frame_payloads(self._orbit, include_contours=bool(self.show_contours.value))
        self._frame_index = 0
        self._apply_cached_frame(0)
        self._mark_cache_ready(f"Flow cache ready: {len(self._orbit)} frames.")

    def _current_w(self) -> np.ndarray:
        if self._orbit.size == 0:
            return self._initial_w()
        idx = int(np.clip(self._frame_index, 0, len(self._orbit) - 1))
        return self._orbit[idx].copy()

    def _sync_frame_controls(self, idx: int) -> None:
        count = int(len(self._orbit)) if self._orbit is not None else 0
        max_idx = max(0, count - 1)
        frame = int(np.clip(int(idx), 0, max_idx)) if count else 0
        self._frame_index = frame
        self._updating = True
        try:
            self.frame_slider.max = max_idx
            self.frame_slider.disabled = (not self._cache_valid) or count <= 1
            self.frame_slider.value = frame
            self.play.max = max_idx
            self.play.disabled = (not self._cache_valid) or count <= 1
            self.play.value = frame
            self.btn_step.disabled = (not self._cache_valid) or count <= 1
            self.frame_counter.value = f"frame {frame + 1 if count else 0} / {count}"
        finally:
            self._updating = False

    def _set_frame_index(self, idx: int) -> None:
        self._sync_frame_controls(idx)
        self._apply_cached_frame(self._frame_index)

    def _build_frame_payloads(self, orbit: np.ndarray, *, include_contours: bool) -> list[dict[str, Any]]:
        W = np.asarray(orbit, dtype=np.float64).reshape(-1, 2)
        if W.size == 0:
            return []
        time_mode = self._time_mode()
        parameter_time = np.asarray(getattr(self, "_orbit_parameter_time", np.arange(W.shape[0])), dtype=np.float64)
        physical_time = np.asarray(getattr(self, "_orbit_physical_time", np.arange(W.shape[0])), dtype=np.float64)
        if parameter_time.shape[0] != W.shape[0]:
            parameter_time = np.arange(W.shape[0], dtype=np.float64)
        if physical_time.shape[0] != W.shape[0]:
            physical_time = parameter_time.copy()
        P = np.asarray(self.points, dtype=np.float64)
        a = np.asarray(self.weights, dtype=np.float64)
        Q = _reflected_series_np(W, P)
        R = np.einsum("n,tnj->tj", a, Q, optimize=True)
        w2 = np.sum(W * W, axis=1)
        velocity = 0.5 * (1.0 - w2)[:, None] * R
        centered = Q - R[:, None, :]
        covariance = np.einsum("n,tni,tnj->tij", a, centered, centered, optimize=True)
        evals = np.linalg.eigvalsh(0.5 * (covariance + np.swapaxes(covariance, -1, -2)))
        reflected_balance = np.linalg.norm(
            np.einsum("n,tnj->tj", a, _mobius_sphere_series_np(W, Q), optimize=True),
            axis=1,
        )
        denom = np.maximum(1.0 - w2, 1e-12)
        diffw = W[:, None, :] - Q
        reflected_phase = np.einsum(
            "n,tn->t",
            a,
            np.log(np.maximum(np.sum(diffw * diffw, axis=-1), 1e-12)) - np.log(denom)[:, None],
            optimize=True,
        )
        s_p0 = _weighted_busemann_phase_series_np(W, P, a)
        phi_p0 = -s_p0
        if include_contours:
            contour_grid_size = min(self.level_grid_size, 64 if len(W) <= 480 else 52)
            xs, ys, grids = _phase_grids_for_reflected_series_np(
                Q,
                a,
                grid_size=contour_grid_size,
            )
        else:
            xs = ys = np.asarray([], dtype=np.float64)
            grids = None
        payloads: list[dict[str, Any]] = []
        frame_arrays = {
            "W": W,
            "P": P,
            "Q": Q,
            "R": R,
            "velocity": velocity,
            "covariance": covariance,
            "covariance_evals": evals,
        }
        for i, w_np in enumerate(W):
            scale_R = 0.48
            scale_vel = 1.6
            rx, ry = _line2(w_np, w_np + scale_R * R[i])
            vx, vy = _line2(w_np, w_np + scale_vel * velocity[i])
            if grids is None:
                level_x: list[float | None] = []
                level_y: list[float | None] = []
            else:
                level_x, level_y = potential_level_segments_2d(xs, ys, grids[i], level_count=7 if len(W) <= 480 else 5)
            ex, ey = _covariance_ellipse_2d(R[i], covariance[i], scale=0.72)
            order = np.argsort(np.arctan2(Q[i, :, 1], Q[i, :, 0]))
            increments = a[order, None] * Q[i, order]
            cumulative = np.vstack([np.zeros((1, 2)), np.cumsum(increments, axis=0)])
            coh = float(np.linalg.norm(R[i]))
            var = max(0.0, 1.0 - coh * coh)
            payload = {
                "w": w_np.copy(),
                "P": P,
                "Q": Q[i].copy(),
                "R": R[i].copy(),
                "velocity": velocity[i].copy(),
                "covariance": covariance[i].copy(),
                "covariance_evals": evals[i].copy(),
                "anchors_x": P[:, 0],
                "anchors_y": P[:, 1],
                "R_arrow_x": rx,
                "R_arrow_y": ry,
                "vel_arrow_x": vx,
                "vel_arrow_y": vy,
                "ref_phase_x": level_x,
                "ref_phase_y": level_y,
                "reflected_x": Q[i, :, 0],
                "reflected_y": Q[i, :, 1],
                "ref_w_x": [float(w_np[0])],
                "ref_w_y": [float(w_np[1])],
                "R_circle_x": [0.0, float(R[i, 0])],
                "R_circle_y": [0.0, float(R[i, 1])],
                "ellipse_x": ex,
                "ellipse_y": ey,
                "polygon_x": cumulative[:, 0],
                "polygon_y": cumulative[:, 1],
                "reflected_balance": float(reflected_balance[i]),
                "reflected_phase": float(reflected_phase[i]),
                "S_p0": float(s_p0[i]),
                "Phi_p0": float(phi_p0[i]),
                "coherence": coh,
                "ray_variance": var,
                "speed": float(np.linalg.norm(velocity[i])),
                "parameter_time": float(parameter_time[i]),
                "physical_time": float(physical_time[i]),
                "stats": (
                    "<b>Optical diagnostics</b> "
                    f"$\\|\\sum_i a_i p_i^0\\|={self._center_error:.2e}$; "
                    f"$\\|\\sum_i a_i M_w(r_i)\\|={float(reflected_balance[i]):.2e}$; "
                    f"$|R_{{p^0}}(w)|={coh:.6f}$; "
                    f"$1-|R|^2={var:.6f}$; "
                    f"$\\Phi_{{p^0}}(w)={float(phi_p0[i]):.6f}$; "
                    f"$S_{{r(w)}}(w)={float(reflected_phase[i]):.6f}$; "
                    f"$|\\dot w|={float(np.linalg.norm(velocity[i])):.6f}$; "
                    f"$\\mathrm{{time}}={'ES' if time_mode == 'euler_sundman' else 'physical'}$; "
                    f"$\\tau={float(parameter_time[i]):.6g}$; "
                    f"$t={float(physical_time[i]):.6g}$; "
                    f"$\\lambda(C)=({float(evals[i, 0]):.5f},{float(evals[i, 1]):.5f})$"
                ),
            }
            payloads.append(
                self._augment_frame_payload(
                    payload,
                    frame_index=i,
                    frame_arrays=frame_arrays,
                )
            )
        return self._finalize_frame_payloads(payloads, frame_arrays=frame_arrays)

    def _augment_frame_payload(
        self,
        payload: dict[str, Any],
        *,
        frame_index: int,
        frame_arrays: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """Hook for subclasses to add cached, visualization-specific data."""
        _ = frame_index, frame_arrays
        return payload

    def _finalize_frame_payloads(
        self,
        payloads: list[dict[str, Any]],
        *,
        frame_arrays: dict[str, np.ndarray],
    ) -> list[dict[str, Any]]:
        """Hook for subclasses that need series-wide cached payload fields."""
        _ = frame_arrays
        return payloads

    def _precompute_frame_payloads(self) -> None:
        self._frame_payloads = self._build_frame_payloads(self._orbit, include_contours=bool(self.show_contours.value))

    def _apply_cached_frame(self, frame_idx: int) -> None:
        if not self._frame_payloads:
            return
        payload = self._frame_payloads[int(np.clip(int(frame_idx), 0, len(self._frame_payloads) - 1))]
        selected = self._selected_payload_geometry(payload)
        self._apply_payload_to_figure(payload, selected)
        self.stats_html.value = _sanitize_plot_text(payload["stats"])

    def _selected_payload_geometry(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return selected-anchor reflection geometry for the current payload.

        The selected constant is intentionally computed at render time, so
        changing `selected i` updates the small geometry panel without forcing
        any expensive frame-cache recompute.
        """
        P = np.asarray(payload["P"], dtype=np.float64)
        Q = np.asarray(payload["Q"], dtype=np.float64)
        w_np = np.asarray(payload["w"], dtype=np.float64)
        idx = int(np.clip(int(self.selected.value), 0, P.shape[0] - 1))
        p = P[idx]
        q = Q[idx]
        n = p - w_np
        n_unit = n / max(float(np.linalg.norm(n)), 1e-12)
        tangent = np.array([-n_unit[1], n_unit[0]])
        return {
            "index": idx,
            "p": p,
            "q": q,
            "normal": n,
            "tangent": tangent,
            "w": w_np,
        }

    def _apply_payload_to_figure(self, payload: dict[str, Any], selected: dict[str, Any]) -> None:
        """Render one cached payload.

        Subclasses that want a different visualization should override this
        method and can still reuse the precomputed payload fields from the
        base class.
        """
        p = np.asarray(selected["p"], dtype=np.float64)
        q = np.asarray(selected["q"], dtype=np.float64)
        w_np = np.asarray(selected["w"], dtype=np.float64)
        tangent = np.asarray(selected["tangent"], dtype=np.float64)
        with self.fig.batch_update():
            self.fig.data[self.tr["orbit_path"]].x = self._orbit[:, 0].tolist()
            self.fig.data[self.tr["orbit_path"]].y = self._orbit[:, 1].tolist()
            self.fig.data[self.tr["anchors"]].x = _plotly_values(payload["anchors_x"])
            self.fig.data[self.tr["anchors"]].y = _plotly_values(payload["anchors_y"])
            self.fig.data[self.tr["selected_anchor"]].x = [float(p[0])]
            self.fig.data[self.tr["selected_anchor"]].y = [float(p[1])]
            self.fig.data[self.tr["w"]].x = [float(w_np[0])]
            self.fig.data[self.tr["w"]].y = [float(w_np[1])]
            self.fig.data[self.tr["R_arrow"]].x = _plotly_values(payload["R_arrow_x"])
            self.fig.data[self.tr["R_arrow"]].y = _plotly_values(payload["R_arrow_y"])
            self.fig.data[self.tr["vel_arrow"]].x = _plotly_values(payload["vel_arrow_x"])
            self.fig.data[self.tr["vel_arrow"]].y = _plotly_values(payload["vel_arrow_y"])
            self.fig.data[self.tr["ref_phase"]].x = _plotly_values(payload["ref_phase_x"])
            self.fig.data[self.tr["ref_phase"]].y = _plotly_values(payload["ref_phase_y"])
            self.fig.data[self.tr["ref_phase"]].visible = bool(self.show_contours.value)
            self.fig.data[self.tr["ref_phase"]].opacity = 0.82 if bool(self.show_contours.value) else 0.0
            self.fig.data[self.tr["reflected"]].x = _plotly_values(payload["reflected_x"])
            self.fig.data[self.tr["reflected"]].y = _plotly_values(payload["reflected_y"])
            self.fig.data[self.tr["ref_w"]].x = _plotly_values(payload["ref_w_x"])
            self.fig.data[self.tr["ref_w"]].y = _plotly_values(payload["ref_w_y"])
            self.fig.data[self.tr["R_circle"]].x = _plotly_values(payload["R_circle_x"])
            self.fig.data[self.tr["R_circle"]].y = _plotly_values(payload["R_circle_y"])
            self.fig.data[self.tr["ellipse"]].x = _plotly_values(payload["ellipse_x"])
            self.fig.data[self.tr["ellipse"]].y = _plotly_values(payload["ellipse_y"])
            self.fig.data[self.tr["polygon"]].x = _plotly_values(payload["polygon_x"])
            self.fig.data[self.tr["polygon"]].y = _plotly_values(payload["polygon_y"])
            self.fig.data[self.tr["geom_incoming"]].x = [0.0, float(p[0])]
            self.fig.data[self.tr["geom_incoming"]].y = [0.0, float(p[1])]
            self.fig.data[self.tr["geom_reflected"]].x = [0.0, float(q[0])]
            self.fig.data[self.tr["geom_reflected"]].y = [0.0, float(q[1])]
            self.fig.data[self.tr["geom_normal"]].x = [float(w_np[0]), float(p[0])]
            self.fig.data[self.tr["geom_normal"]].y = [float(w_np[1]), float(p[1])]
            self.fig.data[self.tr["geom_mirror"]].x = [float(-1.08 * tangent[0]), float(1.08 * tangent[0])]
            self.fig.data[self.tr["geom_mirror"]].y = [float(-1.08 * tangent[1]), float(1.08 * tangent[1])]

    def _on_control_change(self, change: dict[str, Any]) -> None:
        if self._updating:
            return
        owner = change.get("owner")
        if owner in (self.preset_dropdown, self.n_slider, self.init_radius_slider):
            self._resample_anchors_and_precompute()
            return
        if owner is self.selected:
            self._sync_anchor_controls()
            self._apply_cached_frame(self._frame_index)
            return
        if owner is self.anchor_theta:
            idx = int(self.selected.value)
            self.raw_points[idx] = _angle2(float(self.anchor_theta.value))
            self.points = self._canonicalized_points(self.raw_points)
            self._sync_anchor_controls()
            self._build_preview_frame("Anchor edited. Precompute flow before playback.")
            return
        if owner is self.frame_slider:
            if self._cache_valid:
                self._set_frame_index(int(self.frame_slider.value))
            else:
                self._sync_frame_controls(0)
            return
        if owner in (self.step_size, self.max_frames):
            self._build_preview_frame("Flow parameter changed. Precompute flow before playback.")
            return
        if owner is self.show_contours:
            if bool(self.show_contours.value) and self._frame_payloads and not self._frame_payloads[0].get("ref_phase_x"):
                self._mark_cache_stale("Contours were not cached. Precompute flow to populate level sets.")
            self._apply_cached_frame(self._frame_index)
            return
        self._apply_cached_frame(self._frame_index)

    def _resample_anchors_and_precompute(self) -> None:
        self.init_radius_value = float(np.clip(float(self.init_radius_slider.value), 0.0, 0.985))
        self.raw_points = _initialized_observed_cloud_np(
            int(self.n_slider.value),
            self.rng,
            preset=self.preset_dropdown.value,
            target_radius=self.init_radius_value,
        )
        self.weights = np.full((self.raw_points.shape[0],), 1.0 / float(self.raw_points.shape[0]), dtype=np.float64)
        self.points = self._canonicalized_points(self.raw_points)
        self._updating = True
        try:
            self.selected.max = max(0, self.points.shape[0] - 1)
            self.selected.value = min(int(self.selected.value), int(self.selected.max))
        finally:
            self._updating = False
        self._sync_anchor_controls()
        self._rebuild_orbit()

    def _on_resample(self, _btn: Any) -> None:
        self._resample_anchors_and_precompute()

    def _on_rebuild_orbit_clicked(self, _btn: Any) -> None:
        self._rebuild_orbit()

    def _on_step(self, _btn: Any) -> None:
        if not self._cache_valid or len(self._orbit) <= 1:
            return
        self._set_frame_index((self._frame_index + 1) % len(self._orbit))

    def _on_play_tick(self, change: dict[str, Any]) -> None:
        if self._updating or change.get("name") != "value":
            return
        if not self._cache_valid:
            self._sync_frame_controls(0)
            return
        self._set_frame_index(int(change.get("new", 0)))

    def _on_reset(self, _btn: Any) -> None:
        self._set_frame_index(0)

    def _stats_html(self, state: OpticalState, reflected_balance_error: float, reflected_phase: float) -> str:
        """Legacy helper kept for static callers; playback uses cached HTML."""
        evals = state.covariance_evals.detach().cpu().numpy().astype(np.float64)
        return (
            "<b>Optical diagnostics</b> "
            f"$\\|\\sum_i a_i p_i^0\\|={self._center_error:.2e}$; "
            f"$\\|\\sum_i a_i M_w(r_i)\\|={reflected_balance_error:.2e}$; "
            f"$|R_{{p^0}}(w)|={float(state.coherence):.6f}$; "
            f"$1-|R|^2={float(state.variance):.6f}$; "
            f"$S_{{r(w)}}(w)={reflected_phase:.6f}$; "
            f"$|\\dot w|={float(torch.linalg.norm(state.velocity)):.6f}$; "
            f"$\\lambda(C)=({float(evals[0]):.5f},{float(evals[1]):.5f})$"
        )

    def _update(self, *, recompute_reflected_grid: bool = True) -> None:
        """Apply cached frame data only.

        Kept as a compatibility shim for older notebook state. It deliberately
        does not recompute per-frame optics.
        """
        _ = recompute_reflected_grid
        self._apply_cached_frame(self._frame_index)


class LMSOpticalDiskWidget(LMSOpticalDiskBaseWidget):
    """Default four-panel 2D optical explorer.

    This concrete subclass currently uses the base class rendering hooks
    unchanged.  New optical views should subclass `LMSOpticalDiskBaseWidget`
    and override the visualization hooks rather than duplicating the common
    frame cache and optical helper logic.
    """


class LMSOpticalDynamicInversionCayleyDiskWidget(LMSOpticalDiskBaseWidget):
    """2D optical explorer with dynamic w-relative charts.

    The top row is the standard optical disk view.  The bottom row replaces the
    base diagnostic panels with two moving charts of the first subplot's data:
    the left chart pushes points to at least unit distance from w_t, and the
    right chart maps x -> (x+w_t)/|x+w_t|^2.
    """

    _DYNAMIC_CHART_CIRCLE_SAMPLES = 4096

    def __init__(
        self,
        *,
        width: int = 1260,
        height: int = 860,
        **kwargs: Any,
    ) -> None:
        super().__init__(width=width, height=height, **kwargs)

    def _layout_header_html(self) -> str:
        return (
            "<b>LMS optical reflected-ray explorer + dynamic w-relative charts</b><br>"
            "Bottom left: one-sided unit-distance normalization around $w_t$. "
            "Bottom right: pointwise spherical inversion $(x+w_t)/|x+w_t|^2$."
        )

    def _make_subplot_figure(self) -> Any:
        return make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                r"$p_i^0=M_{-w_\ast}(x_i^0),\quad w(t)$",
                r"$r_i(w)=H_{p_i^0-w}(p_i^0),\quad S_{r(w)}(u)=\mathrm{const}$",
                r"$N_{w_t}(x):\ |N_{w_t}(x)-w_t|\ge 1$",
                r"$I_{-w_t}(x)=(x+w_t)/|x+w_t|^2$",
            ),
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
        )

    def _build_figure(self) -> None:
        fig = go.FigureWidget(self._make_subplot_figure())
        self.fig = fig
        self.tr: dict[str, int] = {}
        self._reset_subplot_legend_state()

        def add(trace: Any, key: str, row: int, col: int) -> None:
            self._add_trace_to_subplot(fig, trace, key, row, col)

        cx, cy = _circle_xy()
        add(go.Scatter(x=cx.tolist(), y=cy.tolist(), mode="lines", line=dict(color="rgba(20,20,20,0.75)", width=2), name=r"$\partial\mathbb{B}^2$", hoverinfo="skip"), "disk", 1, 1)
        add(go.Scatter(x=[], y=[], mode="lines", line=dict(color="rgba(20,20,20,0.35)", width=2), name=r"$w(t)$", hoverinfo="skip"), "orbit_path", 1, 1)
        add(go.Scatter(x=[0.0], y=[0.0], mode="markers", marker=dict(size=11, color="white", line=dict(color="black", width=2)), name=r"$\sum_i a_i p_i^0=0$"), "core_balance", 1, 1)
        add(go.Scatter(x=[], y=[], mode="markers", marker=dict(size=8, color="#244C9A", opacity=0.72), name=r"$p_i^0$"), "anchors", 1, 1)
        add(go.Scatter(x=[], y=[], mode="markers", marker=dict(size=14, color="#F2A900", line=dict(color="black", width=1.4)), name=r"$\mathrm{selected}\ p_i^0$"), "selected_anchor", 1, 1)
        add(go.Scatter(x=[], y=[], mode="markers", marker=dict(size=13, color="black"), name=r"$w$"), "w", 1, 1)
        add(go.Scatter(x=[], y=[], mode="lines+markers", line=dict(color="#D72638", width=4), marker=dict(size=[0, 9], color="#D72638"), name=r"$R_{p^0}(w)=\sum_i a_i r_i(w)$"), "R_arrow", 1, 1)
        add(go.Scatter(x=[], y=[], mode="lines+markers", line=dict(color="#188038", width=3, dash="dash"), marker=dict(size=[0, 7], color="#188038"), name=r"$\dot w$"), "vel_arrow", 1, 1)

        add(go.Scatter(x=[], y=[], mode="lines", line=dict(color="rgba(78,60,150,0.82)", width=1.45), opacity=0.82, name=r"$S_{r(w)}(u)=\mathrm{const}$", hoverinfo="skip"), "ref_phase", 1, 2)
        add(go.Scatter(x=cx.tolist(), y=cy.tolist(), mode="lines", line=dict(color="rgba(20,20,20,0.75)", width=2), name=r"$\partial\mathbb{B}^2$", hoverinfo="skip", showlegend=False), "ref_circle", 1, 2)
        add(go.Scatter(x=[], y=[], mode="markers", marker=dict(size=8, color="#5B8C5A", opacity=0.78), name=r"$r_i(w)$"), "reflected", 1, 2)
        add(go.Scatter(x=[], y=[], mode="markers", marker=dict(size=13, color="black", symbol="x"), name=r"$w:\sum_i a_iM_w(r_i(w))=0$"), "ref_w", 1, 2)
        add(go.Scatter(x=[], y=[], mode="lines+markers", line=dict(color="#D72638", width=4), marker=dict(size=[0, 10], color="#D72638"), name=r"$R_{p^0}(w)$"), "R_circle", 1, 2)
        add(go.Scatter(x=[], y=[], mode="lines", line=dict(color="#5B8C5A", width=2), name=r"$\mathrm{Cov}(r_i(w))$", hoverinfo="skip"), "ellipse", 1, 2)

        self._add_dynamic_inversion_chart_traces("inv_w", 2, 1, transform_name="N")
        self._add_dynamic_inversion_chart_traces("inv_neg_w", 2, 2, center_name=r"$c=-w_t$")

        fig.update_layout(
            width=self.width,
            height=self.height,
            template="plotly_white",
            margin=dict(l=35, r=20, t=58, b=30),
            showlegend=True,
        )
        for axis in ("xaxis", "xaxis2"):
            fig.layout[axis].update(range=[-1.15, 1.15], zeroline=False, showgrid=False)
        for axis, anchor in (("yaxis", "x"), ("yaxis2", "x2")):
            fig.layout[axis].update(range=[-1.15, 1.15], zeroline=False, showgrid=False, scaleanchor=anchor, scaleratio=1)
        for axis in ("xaxis3", "xaxis4"):
            fig.layout[axis].update(zeroline=True, showgrid=True)
        for axis, anchor in (("yaxis3", "x3"), ("yaxis4", "x4")):
            fig.layout[axis].update(zeroline=True, showgrid=True, scaleanchor=anchor, scaleratio=1)
        for row, col in ((1, 1), (1, 2)):
            fig.update_xaxes(title_text=r"$e_1$", row=row, col=col)
            fig.update_yaxes(title_text=r"$e_2$", row=row, col=col)
        fig.update_xaxes(title_text=r"$N_{w_t}(e_1)$", row=2, col=1)
        fig.update_yaxes(title_text=r"$N_{w_t}(e_2)$", row=2, col=1)
        fig.update_xaxes(title_text=r"$I_{-w_t}(e_1)$", row=2, col=2)
        fig.update_yaxes(title_text=r"$I_{-w_t}(e_2)$", row=2, col=2)
        _sanitize_figure_text(fig)
        self._configure_subplot_legends()

    def _add_dynamic_inversion_chart_traces(
        self,
        prefix: str,
        row: int,
        col: int,
        *,
        center_name: str = "",
        transform_name: str = "I",
    ) -> None:
        def add(trace: Any, suffix: str) -> None:
            self._add_trace_to_subplot(self.fig, trace, f"{prefix}_{suffix}", row, col)

        prefix_label = "N_c" if transform_name == "N" else "I_c"

        add(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color="rgba(20,20,20,0.75)", width=2),
                name=rf"${prefix_label}(\partial\mathbb{{B}}^2)$",
                hoverinfo="skip",
                showlegend=False,
            ),
            "disk",
        )
        add(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color="rgba(20,20,20,0.35)", width=2),
                name=rf"${prefix_label}(w(t))$",
                hoverinfo="skip",
            ),
            "orbit_path",
        )
        add(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                marker=dict(size=11, color="white", line=dict(color="black", width=2)),
                name=rf"${prefix_label}(0)$",
            ),
            "core_balance",
        )
        add(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                marker=dict(size=8, color="#244C9A", opacity=0.72),
                name=rf"${prefix_label}(p_i^0)$",
            ),
            "anchors",
        )
        add(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                marker=dict(size=14, color="#F2A900", line=dict(color="black", width=1.4)),
                name=rf"${prefix_label}(\mathrm{{selected}}\ p_i^0)$",
            ),
            "selected_anchor",
        )
        _ = center_name
        add(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                marker=dict(size=13, color="black"),
                name=rf"${prefix_label}(w_t)$",
            ),
            "w",
        )
        add(
            go.Scatter(
                x=[],
                y=[],
                mode="lines+markers",
                line=dict(color="#D72638", width=4),
                marker=dict(size=[0, 9], color="#D72638"),
                name=rf"${prefix_label}(w_t+R_{{p^0}}(w))$",
            ),
            "R_arrow",
        )
        add(
            go.Scatter(
                x=[],
                y=[],
                mode="lines+markers",
                line=dict(color="#188038", width=3, dash="dash"),
                marker=dict(size=[0, 7], color="#188038"),
                name=rf"${prefix_label}(w_t+\dot w)$",
            ),
            "vel_arrow",
        )

    def _dynamic_inversion_payload(
        self,
        *,
        prefix: str,
        center: np.ndarray,
        payload: dict[str, Any],
        frame_arrays: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        W = np.asarray(frame_arrays["W"], dtype=np.float64).reshape(-1, 2)
        P = np.asarray(frame_arrays["P"], dtype=np.float64).reshape(-1, 2)
        w_np = np.asarray(payload["w"], dtype=np.float64).reshape(2)
        R = np.asarray(payload["R"], dtype=np.float64).reshape(2)
        velocity = np.asarray(payload["velocity"], dtype=np.float64).reshape(2)
        c = np.asarray(center, dtype=np.float64).reshape(2)
        cx, cy = _circle_xy(samples=self._DYNAMIC_CHART_CIRCLE_SAMPLES)
        circle = np.column_stack([cx, cy])
        line_t = np.linspace(0.0, 1.0, 36, dtype=np.float64)[:, None]
        r_curve = w_np[None, :] + line_t * (0.48 * R)[None, :]
        v_curve = w_np[None, :] + line_t * (1.6 * velocity)[None, :]
        transform = _one_sided_unit_distance_chart_np if prefix == "inv_w" else _spherical_inversion_chart_np
        r_line = transform(r_curve, c)
        v_line = transform(v_curve, c)
        chart_disk = transform(circle, c)
        chart_orbit = transform(W, c)
        chart_anchors = transform(P, c)
        chart_core = transform(np.array([[0.0, 0.0]], dtype=np.float64), c)[0]
        chart_w = transform(w_np.reshape(1, 2), c)[0]
        finite = np.concatenate(
            [
                chart_disk[np.isfinite(chart_disk).all(axis=1)],
                chart_orbit[np.isfinite(chart_orbit).all(axis=1)],
                chart_anchors[np.isfinite(chart_anchors).all(axis=1)],
                r_line[np.isfinite(r_line).all(axis=1)],
                v_line[np.isfinite(v_line).all(axis=1)],
                chart_core.reshape(1, 2),
            ],
            axis=0,
        )
        finite = finite[np.isfinite(finite).all(axis=1)]
        if finite.size:
            lo = np.nanquantile(finite, 0.03, axis=0)
            hi = np.nanquantile(finite, 0.97, axis=0)
            span = np.maximum(hi - lo, 1.0)
            x_range = [float(lo[0] - 0.12 * span[0]), float(hi[0] + 0.12 * span[0])]
            y_range = [float(lo[1] - 0.12 * span[1]), float(hi[1] + 0.12 * span[1])]
        else:
            x_range = y_range = [-1.0, 1.0]
        return {
            f"{prefix}_center_source": c,
            f"{prefix}_disk_x": chart_disk[:, 0],
            f"{prefix}_disk_y": chart_disk[:, 1],
            f"{prefix}_orbit_path_x": chart_orbit[:, 0],
            f"{prefix}_orbit_path_y": chart_orbit[:, 1],
            f"{prefix}_core_balance_x": [float(chart_core[0])],
            f"{prefix}_core_balance_y": [float(chart_core[1])],
            f"{prefix}_anchors_x": chart_anchors[:, 0],
            f"{prefix}_anchors_y": chart_anchors[:, 1],
            f"{prefix}_w_x": [float(chart_w[0])],
            f"{prefix}_w_y": [float(chart_w[1])],
            f"{prefix}_R_arrow_x": r_line[:, 0],
            f"{prefix}_R_arrow_y": r_line[:, 1],
            f"{prefix}_vel_arrow_x": v_line[:, 0],
            f"{prefix}_vel_arrow_y": v_line[:, 1],
            f"{prefix}_x_range": x_range,
            f"{prefix}_y_range": y_range,
        }

    def _augment_frame_payload(
        self,
        payload: dict[str, Any],
        *,
        frame_index: int,
        frame_arrays: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        payload = super()._augment_frame_payload(payload, frame_index=frame_index, frame_arrays=frame_arrays)
        w_np = np.asarray(payload["w"], dtype=np.float64).reshape(2)
        payload.update(
            self._dynamic_inversion_payload(
                prefix="inv_w",
                center=w_np,
                payload=payload,
                frame_arrays=frame_arrays,
            )
        )
        payload.update(
            self._dynamic_inversion_payload(
                prefix="inv_neg_w",
                center=-w_np,
                payload=payload,
                frame_arrays=frame_arrays,
            )
        )
        payload["stats"] += f"; dynamic inversion centers $|w|={float(np.linalg.norm(w_np)):.6f}$"
        return payload

    def _apply_payload_to_figure(self, payload: dict[str, Any], selected: dict[str, Any]) -> None:
        p = np.asarray(selected["p"], dtype=np.float64)
        w_np = np.asarray(selected["w"], dtype=np.float64)
        selected_anchor = np.asarray(selected["p"], dtype=np.float64).reshape(2)
        with self.fig.batch_update():
            self.fig.data[self.tr["orbit_path"]].x = self._orbit[:, 0].tolist()
            self.fig.data[self.tr["orbit_path"]].y = self._orbit[:, 1].tolist()
            self.fig.data[self.tr["anchors"]].x = _plotly_values(payload["anchors_x"])
            self.fig.data[self.tr["anchors"]].y = _plotly_values(payload["anchors_y"])
            self.fig.data[self.tr["selected_anchor"]].x = [float(p[0])]
            self.fig.data[self.tr["selected_anchor"]].y = [float(p[1])]
            self.fig.data[self.tr["w"]].x = [float(w_np[0])]
            self.fig.data[self.tr["w"]].y = [float(w_np[1])]
            self.fig.data[self.tr["R_arrow"]].x = _plotly_values(payload["R_arrow_x"])
            self.fig.data[self.tr["R_arrow"]].y = _plotly_values(payload["R_arrow_y"])
            self.fig.data[self.tr["vel_arrow"]].x = _plotly_values(payload["vel_arrow_x"])
            self.fig.data[self.tr["vel_arrow"]].y = _plotly_values(payload["vel_arrow_y"])
            self.fig.data[self.tr["ref_phase"]].x = _plotly_values(payload["ref_phase_x"])
            self.fig.data[self.tr["ref_phase"]].y = _plotly_values(payload["ref_phase_y"])
            self.fig.data[self.tr["ref_phase"]].visible = bool(self.show_contours.value)
            self.fig.data[self.tr["ref_phase"]].opacity = 0.82 if bool(self.show_contours.value) else 0.0
            self.fig.data[self.tr["reflected"]].x = _plotly_values(payload["reflected_x"])
            self.fig.data[self.tr["reflected"]].y = _plotly_values(payload["reflected_y"])
            self.fig.data[self.tr["ref_w"]].x = _plotly_values(payload["ref_w_x"])
            self.fig.data[self.tr["ref_w"]].y = _plotly_values(payload["ref_w_y"])
            self.fig.data[self.tr["R_circle"]].x = _plotly_values(payload["R_circle_x"])
            self.fig.data[self.tr["R_circle"]].y = _plotly_values(payload["R_circle_y"])
            self.fig.data[self.tr["ellipse"]].x = _plotly_values(payload["ellipse_x"])
            self.fig.data[self.tr["ellipse"]].y = _plotly_values(payload["ellipse_y"])

            for prefix in ("inv_w", "inv_neg_w"):
                c = np.asarray(payload[f"{prefix}_center_source"], dtype=np.float64).reshape(2)
                transform = _one_sided_unit_distance_chart_np if prefix == "inv_w" else _spherical_inversion_chart_np
                selected_chart = transform(selected_anchor.reshape(1, 2), c)[0]
                self.fig.data[self.tr[f"{prefix}_disk"]].x = _plotly_values(payload[f"{prefix}_disk_x"])
                self.fig.data[self.tr[f"{prefix}_disk"]].y = _plotly_values(payload[f"{prefix}_disk_y"])
                self.fig.data[self.tr[f"{prefix}_orbit_path"]].x = _plotly_values(payload[f"{prefix}_orbit_path_x"])
                self.fig.data[self.tr[f"{prefix}_orbit_path"]].y = _plotly_values(payload[f"{prefix}_orbit_path_y"])
                self.fig.data[self.tr[f"{prefix}_core_balance"]].x = _plotly_values(payload[f"{prefix}_core_balance_x"])
                self.fig.data[self.tr[f"{prefix}_core_balance"]].y = _plotly_values(payload[f"{prefix}_core_balance_y"])
                self.fig.data[self.tr[f"{prefix}_anchors"]].x = _plotly_values(payload[f"{prefix}_anchors_x"])
                self.fig.data[self.tr[f"{prefix}_anchors"]].y = _plotly_values(payload[f"{prefix}_anchors_y"])
                self.fig.data[self.tr[f"{prefix}_selected_anchor"]].x = [float(selected_chart[0])]
                self.fig.data[self.tr[f"{prefix}_selected_anchor"]].y = [float(selected_chart[1])]
                self.fig.data[self.tr[f"{prefix}_w"]].x = _plotly_values(payload[f"{prefix}_w_x"])
                self.fig.data[self.tr[f"{prefix}_w"]].y = _plotly_values(payload[f"{prefix}_w_y"])
                self.fig.data[self.tr[f"{prefix}_R_arrow"]].x = _plotly_values(payload[f"{prefix}_R_arrow_x"])
                self.fig.data[self.tr[f"{prefix}_R_arrow"]].y = _plotly_values(payload[f"{prefix}_R_arrow_y"])
                self.fig.data[self.tr[f"{prefix}_vel_arrow"]].x = _plotly_values(payload[f"{prefix}_vel_arrow_x"])
                self.fig.data[self.tr[f"{prefix}_vel_arrow"]].y = _plotly_values(payload[f"{prefix}_vel_arrow_y"])
                row, col = (2, 1) if prefix == "inv_w" else (2, 2)
                self.fig.update_xaxes(range=payload[f"{prefix}_x_range"], row=row, col=col)
                self.fig.update_yaxes(range=payload[f"{prefix}_y_range"], row=row, col=col)


class LMSOpticalWeightedCayleyDiskWidget(LMSOpticalDiskBaseWidget):
    """2D optical explorer with a weighted-Busemann Cayley half-plane chart.

    The bottom row shows the modified chart

        (u, lambda) = (lambda <S_xi(w), e_perp>, lambda),
        lambda = exp(-S_{p^0}(w)) = exp(Phi_{p^0}(w)),

    where S_{p^0} is the total weighted Busemann phase of the canonical
    constants p_i^0 and xi is the normalized exact gauge direction
    w_*/|w_*|.  This keeps the Cayley puncture tied to the canonical orbit,
    not to a free UI angle.
    """

    def __init__(
        self,
        *,
        width: int = 1260,
        height: int = 1040,
        **kwargs: Any,
    ) -> None:
        super().__init__(width=width, height=height, **kwargs)

    def _build_controls(self, *, preset: OpticalPreset) -> None:
        super()._build_controls(preset=preset)
        self.cayley_chart_toggle = widgets.ToggleButton(
            value=False,
            description="Cayley: weighted total",
            tooltip="Switch between total-weighted Busemann chart and regular moving-puncture Cayley chart.",
            layout=widgets.Layout(width="220px"),
        )
        children = list(self.controls.children)
        # Place the switch on the playback/precompute row.
        playback_row = list(children[2].children)
        playback_row.insert(3, self.cayley_chart_toggle)
        children[2] = widgets.HBox(playback_row)
        self.controls.children = tuple(children)

    def _bind_callbacks(self) -> None:
        super()._bind_callbacks()
        self.cayley_chart_toggle.observe(self._on_cayley_chart_toggle, names="value")

    def _on_cayley_chart_toggle(self, change: dict[str, Any]) -> None:
        if self._updating or change.get("name") != "value":
            return
        self._sync_cayley_chart_label()
        self._apply_cached_frame(self._frame_index)

    def _sync_cayley_chart_label(self) -> None:
        self.cayley_chart_toggle.description = (
            "Cayley: regular moving ξ" if bool(self.cayley_chart_toggle.value) else "Cayley: weighted total"
        )

    def _active_cayley_prefix(self) -> str:
        return "cayley_regular" if bool(self.cayley_chart_toggle.value) else "cayley_weighted"

    def _layout_header_html(self) -> str:
        return (
            "<b>LMS optical reflected-ray explorer + weighted Cayley chart</b><br>"
            "Bottom: modified Cayley half-plane with "
            "$\\lambda=\\exp(-S_{p^0}(w))=\\exp(\\Phi_{p^0}(w))$ and puncture "
            "$\\xi=w_\\ast/|w_\\ast|$ from the exact gauge."
        )

    def _make_subplot_figure(self) -> Any:
        return make_subplots(
            rows=3,
            cols=2,
            specs=[
                [{}, {}],
                [{}, {}],
                [{"colspan": 2}, None],
            ],
            subplot_titles=(
                r"$p_i^0=M_{-w_\ast}(x_i^0),\quad w(t)$",
                r"$r_i(w)=H_{p_i^0-w}(p_i^0),\quad S_{r(w)}(u)=\mathrm{const}$",
                r"$R_{p^0}(w)=\sum_i a_i r_i(w)$",
                r"$p_i^0\mapsto r_i(w)$",
                r"$\operatorname{Cay}_\xi(w)=(u,\lambda)$",
            ),
            row_heights=[0.30, 0.28, 0.42],
            horizontal_spacing=0.08,
            vertical_spacing=0.10,
        )

    def _after_build_figure(self) -> None:
        def add(trace: Any, key: str) -> None:
            self._add_trace_to_subplot(self.fig, trace, key, 3, 1)

        add(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color="rgba(0,0,0,0.55)", width=1.5),
                name=r"$\lambda=0$",
                hoverinfo="skip",
                showlegend=False,
            ),
            "cayley_axis",
        )
        add(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                marker=dict(size=7, color="#244C9A", opacity=0.65),
                name=r"$\partial\operatorname{Cay}_\xi(p_i^0)$",
            ),
            "cayley_boundary",
        )
        add(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color="#6F4CC3", width=3),
                name=r"$\operatorname{Cay}_\xi(w(t))$",
            ),
            "cayley_path",
        )
        add(
            go.Scatter(
                x=[],
                y=[],
                mode="lines+markers",
                line=dict(color="#D72638", width=3),
                marker=dict(size=[0, 9], color="#D72638"),
                name=r"$\Delta\operatorname{Cay}$",
            ),
            "cayley_step",
        )
        add(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                marker=dict(size=13, color="black"),
                name=r"$\operatorname{Cay}_\xi(w)$",
            ),
            "cayley_point",
        )
        self.fig.update_xaxes(
            title_text=r"$u$",
            zeroline=True,
            showgrid=True,
            row=3,
            col=1,
        )
        self.fig.update_yaxes(
            title_text=r"$\lambda$",
            rangemode="tozero",
            zeroline=False,
            showgrid=True,
            row=3,
            col=1,
        )

    def _canonical_orbit_direction(self) -> np.ndarray:
        """Return the Cayley puncture direction tied to the exact gauge.

        The intended direction is xi=w_*/|w_*|.  If the exact center is
        numerically zero, fall back to the observed cloud barycenter and then
        to e_1; this fallback is deterministic and is not exposed as a free
        angle.
        """
        w = np.asarray(getattr(self, "w_star", np.zeros(2)), dtype=np.float64).reshape(2)
        n = float(np.linalg.norm(w))
        if n > 1e-10:
            return w / n
        mean_x0 = (self.weights[:, None] * np.asarray(self.raw_points, dtype=np.float64)).sum(axis=0)
        n = float(np.linalg.norm(mean_x0))
        if n > 1e-10:
            return mean_x0 / n
        return np.array([1.0, 0.0], dtype=np.float64)

    def _cayley_xi(self) -> np.ndarray:
        return self._canonical_orbit_direction()

    @staticmethod
    def _robust_cayley_ranges(
        u: np.ndarray,
        lam: np.ndarray,
        boundary: np.ndarray,
    ) -> tuple[list[float], list[float]]:
        finite_x = np.concatenate([np.asarray(u)[np.isfinite(u)], np.asarray(boundary)[np.isfinite(boundary)]])
        if finite_x.size <= 0:
            x_abs = 1.0
        else:
            x_abs = max(1.0, float(np.nanquantile(np.abs(finite_x), 0.96)))
        finite_y = np.asarray(lam)[np.isfinite(lam)]
        if finite_y.size <= 0:
            y_hi = 1.0
        else:
            y_hi = max(1e-3, float(np.nanquantile(finite_y, 0.98)))
        return [-1.18 * x_abs, 1.18 * x_abs], [0.0, 1.18 * y_hi]

    def _finalize_frame_payloads(
        self,
        payloads: list[dict[str, Any]],
        *,
        frame_arrays: dict[str, np.ndarray],
    ) -> list[dict[str, Any]]:
        payloads = super()._finalize_frame_payloads(payloads, frame_arrays=frame_arrays)
        if not payloads:
            return payloads
        W = np.asarray(frame_arrays["W"], dtype=np.float64)
        P = np.asarray(frame_arrays["P"], dtype=np.float64)
        xi = self._cayley_xi()
        weighted_chart = _weighted_cayley_chart_series_2d(W, P, self.weights, xi)
        weighted_boundary = _cayley_boundary_coordinates_2d(P, xi)
        weighted_x_range, weighted_y_range = self._robust_cayley_ranges(
            weighted_chart["u"],
            weighted_chart["lambda"],
            weighted_boundary,
        )
        regular_chart = _regular_cayley_chart_series_2d(W, P, self.weights, fallback_xi=xi)
        regular_x_range, regular_y_range = self._robust_cayley_ranges(
            regular_chart["u"],
            regular_chart["lambda"],
            regular_chart["boundary"].reshape(-1),
        )
        frame_count = len(payloads)
        for i, payload in enumerate(payloads):
            j = min(i + 1, frame_count - 1)
            if j == i and i > 0:
                j = i - 1
            payload["cayley_weighted_axis_x"] = weighted_x_range
            payload["cayley_weighted_axis_y"] = [0.0, 0.0]
            payload["cayley_weighted_boundary_x"] = weighted_boundary
            payload["cayley_weighted_boundary_y"] = np.zeros_like(weighted_boundary)
            payload["cayley_weighted_path_x"] = weighted_chart["u"]
            payload["cayley_weighted_path_y"] = weighted_chart["lambda"]
            payload["cayley_weighted_point_x"] = [float(weighted_chart["u"][i])]
            payload["cayley_weighted_point_y"] = [float(weighted_chart["lambda"][i])]
            payload["cayley_weighted_step_x"] = [float(weighted_chart["u"][i]), float(weighted_chart["u"][j])]
            payload["cayley_weighted_step_y"] = [float(weighted_chart["lambda"][i]), float(weighted_chart["lambda"][j])]
            payload["cayley_weighted_x_range"] = weighted_x_range
            payload["cayley_weighted_y_range"] = weighted_y_range
            payload["cayley_regular_axis_x"] = regular_x_range
            payload["cayley_regular_axis_y"] = [0.0, 0.0]
            payload["cayley_regular_boundary_x"] = regular_chart["boundary"][i]
            payload["cayley_regular_boundary_y"] = np.zeros_like(regular_chart["boundary"][i])
            payload["cayley_regular_path_x"] = regular_chart["u"]
            payload["cayley_regular_path_y"] = regular_chart["lambda"]
            payload["cayley_regular_point_x"] = [float(regular_chart["u"][i])]
            payload["cayley_regular_point_y"] = [float(regular_chart["lambda"][i])]
            payload["cayley_regular_step_x"] = [float(regular_chart["u"][i]), float(regular_chart["u"][j])]
            payload["cayley_regular_step_y"] = [float(regular_chart["lambda"][i]), float(regular_chart["lambda"][j])]
            payload["cayley_regular_x_range"] = regular_x_range
            payload["cayley_regular_y_range"] = regular_y_range
            payload["stats"] += (
                f"; $S_{{p^0}}(w)={float(weighted_chart['phase'][i]):.6f}$; "
                f"$\\Phi_{{p^0}}(w)={float(-weighted_chart['phase'][i]):.6f}$; "
                f"$\\lambda_{{\\mathrm{{weighted}}}}={float(weighted_chart['lambda'][i]):.6g}$; "
                f"$\\lambda_{{\\mathrm{{regular}}}}={float(regular_chart['lambda'][i]):.6g}$"
            )
        return payloads

    def _apply_payload_to_figure(self, payload: dict[str, Any], selected: dict[str, Any]) -> None:
        super()._apply_payload_to_figure(payload, selected)
        prefix = self._active_cayley_prefix()
        if prefix == "cayley_regular":
            title_x = r"$u,\quad \xi_t=w_t/|w_t|$"
            title_y = r"$\lambda=\exp(-\mathcal{B}_{\xi_t}(w_t))$"
        else:
            title_x = r"$u=\exp(-S_{p^0}(w))\,\langle\mathfrak{S}_{\xi}(w),e_\perp\rangle$"
            title_y = r"$\lambda=\exp(-S_{p^0}(w))=\exp(\Phi_{p^0}(w))$"
        with self.fig.batch_update():
            self.fig.data[self.tr["cayley_axis"]].x = _plotly_values(payload[f"{prefix}_axis_x"])
            self.fig.data[self.tr["cayley_axis"]].y = _plotly_values(payload[f"{prefix}_axis_y"])
            self.fig.data[self.tr["cayley_boundary"]].x = _plotly_values(payload[f"{prefix}_boundary_x"])
            self.fig.data[self.tr["cayley_boundary"]].y = _plotly_values(payload[f"{prefix}_boundary_y"])
            self.fig.data[self.tr["cayley_path"]].x = _plotly_values(payload[f"{prefix}_path_x"])
            self.fig.data[self.tr["cayley_path"]].y = _plotly_values(payload[f"{prefix}_path_y"])
            self.fig.data[self.tr["cayley_step"]].x = _plotly_values(payload[f"{prefix}_step_x"])
            self.fig.data[self.tr["cayley_step"]].y = _plotly_values(payload[f"{prefix}_step_y"])
            self.fig.data[self.tr["cayley_point"]].x = _plotly_values(payload[f"{prefix}_point_x"])
            self.fig.data[self.tr["cayley_point"]].y = _plotly_values(payload[f"{prefix}_point_y"])
            self.fig.update_xaxes(title_text=_sanitize_plot_text(title_x), range=payload[f"{prefix}_x_range"], row=3, col=1)
            self.fig.update_yaxes(title_text=_sanitize_plot_text(title_y), range=payload[f"{prefix}_y_range"], row=3, col=1)


class LMSOpticalHyperboloidScreenWidget(LMSOpticalDiskBaseWidget):
    """2D optical explorer using the hyperboloid lift plus a local screen chart."""

    def __init__(
        self,
        *,
        width: int = 1260,
        height: int = 900,
        **kwargs: Any,
    ) -> None:
        self._hyper_static_applied = False
        self._hyper_static: dict[str, Any] = {}
        super().__init__(width=width, height=height, **kwargs)

    def _layout_header_html(self) -> str:
        return (
            "<b>LMS hyperboloid + optical screen chart</b><br>"
            "The reflected-cloud panel is the unchanged evolution view. "
            "The other panels show $X(w)$, null constants $\\ell_i=(1,p_i^0)$, "
            "ray fields $u_i$, and the transverse spread coefficient $\\Theta_\\perp$."
        )

    def _make_subplot_figure(self) -> Any:
        return make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "scene"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy", "secondary_y": True}],
            ],
            subplot_titles=(
                r"$X(w),\quad \ell_i,\quad u_i,\quad U$",
                r"$r_i(w)=H_{p_i^0-w}(p_i^0),\quad S_{r(w)}(u)=\mathrm{const}$",
                r"$\beta_i=\langle u_i,E_\perp\rangle_L,\quad \Theta_\perp$",
                r"$|R_{p^0}(w)|,\quad \Theta_\perp,\quad \Phi_{p^0}(w)$",
            ),
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
        )

    def _build_figure(self) -> None:
        sx, sy, sz = _hyperboloid_surface_grid_2d()
        fig = go.FigureWidget(self._make_subplot_figure())
        self.fig = fig
        self.tr: dict[str, int] = {}
        self._reset_subplot_legend_state()

        def add(trace: Any, key: str, row: int, col: int, *, secondary_y: bool = False) -> None:
            self._add_trace_to_subplot(fig, trace, key, row, col, secondary_y=secondary_y)

        add(
            go.Surface(
                x=sx.tolist(),
                y=sy.tolist(),
                z=sz.tolist(),
                colorscale=[[0.0, "#D9E7FA"], [1.0, "#7EA2D6"]],
                showscale=False,
                opacity=0.64,
                showlegend=True,
                contours=dict(
                    x=dict(show=True, color="rgba(70,110,160,0.32)", width=1),
                    y=dict(show=True, color="rgba(70,110,160,0.32)", width=1),
                    z=dict(show=True, color="rgba(70,110,160,0.38)", width=1),
                ),
                name=r"$\mathbb H^2$",
                hoverinfo="skip",
            ),
            "hyper_surface",
            1,
            1,
        )
        add(go.Scatter3d(x=[], y=[], z=[], mode="lines", line=dict(color="rgba(70,70,70,0.55)", width=3), name=r"$t\ell_i=t(1,p_i^0)$", hoverinfo="skip"), "hyper_null", 1, 1)
        add(go.Scatter3d(x=[], y=[], z=[], mode="lines", line=dict(color="rgba(36,76,154,0.42)", width=3), name=r"$X(rp_i^0)$", hoverinfo="skip"), "hyper_geodesics", 1, 1)
        add(go.Scatter3d(x=[], y=[], z=[], mode="lines", line=dict(color="#6F4CC3", width=5), name=r"$X(w(t))$"), "hyper_path", 1, 1)
        add(go.Scatter3d(x=[], y=[], z=[], mode="markers", marker=dict(size=5, color="black"), name=r"$X(w)$"), "hyper_X", 1, 1)
        add(go.Scatter3d(x=[], y=[], z=[], mode="lines", line=dict(color="rgba(91,140,90,0.55)", width=3), name=r"$u_i$", hoverinfo="skip"), "hyper_u", 1, 1)
        add(go.Scatter3d(x=[], y=[], z=[], mode="lines+markers", line=dict(color="#D72638", width=7), marker=dict(size=4, color="#D72638"), name=r"$U=\sum_i a_i u_i$"), "hyper_U", 1, 1)

        cx, cy = _circle_xy()
        add(go.Scatter(x=[], y=[], mode="lines", line=dict(color="rgba(78,60,150,0.82)", width=1.45), opacity=0.82, name=r"$S_{r(w)}(u)=\mathrm{const}$", hoverinfo="skip"), "ref_phase", 1, 2)
        add(go.Scatter(x=cx.tolist(), y=cy.tolist(), mode="lines", line=dict(color="rgba(20,20,20,0.75)", width=2), name=r"$\partial\mathbb{B}^2$", hoverinfo="skip", showlegend=False), "ref_circle", 1, 2)
        add(go.Scatter(x=[], y=[], mode="markers", marker=dict(size=8, color="#5B8C5A", opacity=0.78), name=r"$r_i(w)$"), "reflected", 1, 2)
        add(go.Scatter(x=[], y=[], mode="markers", marker=dict(size=13, color="black", symbol="x"), name=r"$w:\sum_i a_iM_w(r_i(w))=0$"), "ref_w", 1, 2)
        add(go.Scatter(x=[], y=[], mode="lines+markers", line=dict(color="#D72638", width=4), marker=dict(size=[0, 10], color="#D72638"), name=r"$R_{p^0}(w)$"), "R_circle", 1, 2)
        add(go.Scatter(x=[], y=[], mode="lines", line=dict(color="#5B8C5A", width=2), name=r"$\mathrm{Cov}(r_i(w))$", hoverinfo="skip"), "ellipse", 1, 2)

        add(go.Scatter(x=[], y=[], mode="lines", line=dict(color="rgba(0,0,0,0.45)", width=2), name=r"$\beta=0$", showlegend=False, hoverinfo="skip"), "screen_axis", 2, 1)
        add(go.Scatter(x=[], y=[], mode="lines", line=dict(color="rgba(91,140,90,0.42)", width=1.5), name=r"$a_i\beta_i$", hoverinfo="skip"), "screen_stems", 2, 1)
        add(go.Scatter(x=[], y=[], mode="markers", marker=dict(size=9, color="#5B8C5A", opacity=0.82), name=r"$\beta_i$"), "screen_beta", 2, 1)
        add(go.Scatter(x=[], y=[], mode="text", text=[], textfont=dict(color="#D72638", size=13), name=r"$\Theta_\perp$", hoverinfo="skip"), "screen_theta_text", 2, 1)

        add(go.Scatter(x=[], y=[], mode="lines", line=dict(color="#D72638", width=2.5), name=r"$|R_{p^0}|$"), "diag_R", 2, 2)
        add(go.Scatter(x=[], y=[], mode="lines", line=dict(color="#188038", width=2.5), name=r"$\Theta_\perp$"), "diag_theta", 2, 2)
        add(go.Scatter(x=[], y=[], mode="lines", line=dict(color="#6F4CC3", width=2.0, dash="dot"), name=r"$\Phi_{p^0}$"), "diag_phi", 2, 2, secondary_y=True)
        add(go.Scatter(x=[], y=[], mode="markers", marker=dict(size=10, color="#D72638"), name=r"$\mathrm{current}$", showlegend=False), "diag_R_point", 2, 2)
        add(go.Scatter(x=[], y=[], mode="markers", marker=dict(size=10, color="#188038"), name=r"$\mathrm{current}$", showlegend=False), "diag_theta_point", 2, 2)
        add(go.Scatter(x=[], y=[], mode="markers", marker=dict(size=10, color="#6F4CC3"), name=r"$\mathrm{current}$", showlegend=False), "diag_phi_point", 2, 2, secondary_y=True)

        fig.update_layout(
            width=self.width,
            height=self.height,
            template="plotly_white",
            margin=dict(l=35, r=20, t=58, b=30),
            showlegend=True,
        )
        fig.update_layout(
            scene=dict(
                xaxis_title=r"$X_1$",
                yaxis_title=r"$X_2$",
                zaxis_title=r"$X_0$",
                aspectmode="data",
                camera=dict(eye=dict(x=1.55, y=1.35, z=1.05)),
            )
        )
        fig.update_xaxes(range=[-1.15, 1.15], zeroline=False, showgrid=False, title_text=r"$e_1$", row=1, col=2)
        fig.update_yaxes(range=[-1.15, 1.15], zeroline=False, showgrid=False, scaleanchor="x", scaleratio=1, title_text=r"$e_2$", row=1, col=2)
        fig.update_xaxes(title_text=r"$\beta_i=\langle u_i,E_\perp\rangle_L$", zeroline=True, showgrid=True, row=2, col=1)
        fig.update_yaxes(title_text=r"$a_i$", rangemode="tozero", showgrid=True, row=2, col=1)
        fig.update_xaxes(title_text=r"$\mathrm{frame}$", showgrid=True, row=2, col=2)
        fig.update_yaxes(title_text=r"$|R_{p^0}|,\ \Theta_\perp$", showgrid=True, row=2, col=2, secondary_y=False)
        fig.update_yaxes(title_text=r"$\Phi_{p^0}(w)$", showgrid=False, row=2, col=2, secondary_y=True)
        self._hyper_static_applied = False
        _sanitize_figure_text(fig)
        self._configure_subplot_legends()

    def _finalize_frame_payloads(
        self,
        payloads: list[dict[str, Any]],
        *,
        frame_arrays: dict[str, np.ndarray],
    ) -> list[dict[str, Any]]:
        payloads = super()._finalize_frame_payloads(payloads, frame_arrays=frame_arrays)
        if not payloads:
            return payloads
        W = np.asarray(frame_arrays["W"], dtype=np.float64)
        P = np.asarray(frame_arrays["P"], dtype=np.float64)
        R = np.asarray(frame_arrays["R"], dtype=np.float64)
        a = np.asarray(self.weights, dtype=np.float64)
        hyper = _hyperboloid_ray_fields_series_2d(W, P, a)
        X = hyper["X"]
        rays = hyper["u"]
        U = hyper["U"]
        screen_dirs = _screen_directions_from_orbit_2d(W, R, getattr(self, "w_star", np.array([1.0, 0.0])))
        E_perp = _hyperboloid_tangent_from_ball_direction_series_2d(W, screen_dirs)
        beta = _lorentz_inner_np(rays, E_perp[:, None, :])
        A_perp = np.einsum("n,tn->t", a, beta * beta, optimize=True)
        theta_perp = 1.0 - A_perp
        frames = np.arange(len(payloads), dtype=np.float64)
        coherence = np.asarray([payload["coherence"] for payload in payloads], dtype=np.float64)
        phi = np.asarray([payload["Phi_p0"] for payload in payloads], dtype=np.float64)
        trajectory_top = float(np.nanmax(X[:, 0])) if X.size else 1.5
        top_height = max(2.0, trajectory_top * 1.04)
        surface_radius = math.sqrt(max(0.0, top_height * top_height - 1.0))
        sx, sy, sz = _hyperboloid_surface_grid_2d(radius=surface_radius)
        null_scale = top_height
        null_x: list[float | None] = []
        null_y: list[float | None] = []
        null_z: list[float | None] = []
        for p in P:
            null_x.extend([0.0, float(null_scale * p[0]), None])
            null_y.extend([0.0, float(null_scale * p[1]), None])
            null_z.extend([0.0, float(null_scale), None])
        geo_x, geo_y, geo_z = _hyperboloid_radial_geodesic_lines_2d(P, top_height=top_height)
        beta_abs = max(1.0, float(np.nanquantile(np.abs(beta), 0.98)) if beta.size else 1.0)
        max_weight = max(float(np.max(a)), 1.0 / max(1, len(a)))
        self._hyper_static = {
            "surface_x": sx,
            "surface_y": sy,
            "surface_z": sz,
            "null_x": null_x,
            "null_y": null_y,
            "null_z": null_z,
            "geodesic_x": geo_x,
            "geodesic_y": geo_y,
            "geodesic_z": geo_z,
            "path_x": X[:, 1],
            "path_y": X[:, 2],
            "path_z": X[:, 0],
            "scene_range": [
                [-surface_radius, surface_radius],
                [-surface_radius, surface_radius],
                [0.0, top_height],
            ],
            "screen_axis_x": [-1.08 * beta_abs, 1.08 * beta_abs],
            "screen_axis_y": [0.0, 0.0],
            "screen_x_range": [-1.15 * beta_abs, 1.15 * beta_abs],
            "screen_y_range": [0.0, 1.35 * max_weight],
            "diag_frames": frames,
            "diag_R": coherence,
            "diag_theta": theta_perp,
            "diag_phi": phi,
            "diag_y_range": [
                min(-0.05, float(np.nanmin(theta_perp)) - 0.08 if theta_perp.size else -0.05),
                max(1.05, float(np.nanmax(theta_perp)) + 0.08 if theta_perp.size else 1.05),
            ],
            "diag_phi_range": [
                float(np.nanmin(phi)) - 0.08 * max(1.0, float(np.nanmax(phi) - np.nanmin(phi))) if phi.size else -1.0,
                float(np.nanmax(phi)) + 0.08 * max(1.0, float(np.nanmax(phi) - np.nanmin(phi))) if phi.size else 1.0,
            ],
        }
        self._hyper_static_applied = False
        ray_scale = 0.36
        mean_scale = 0.58
        for i, payload in enumerate(payloads):
            u_lines_x: list[float | None] = []
            u_lines_y: list[float | None] = []
            u_lines_z: list[float | None] = []
            for ray in rays[i]:
                end = X[i] + ray_scale * ray
                u_lines_x.extend([float(X[i, 1]), float(end[1]), None])
                u_lines_y.extend([float(X[i, 2]), float(end[2]), None])
                u_lines_z.extend([float(X[i, 0]), float(end[0]), None])
            U_end = X[i] + mean_scale * U[i]
            stems_x: list[float | None] = []
            stems_y: list[float | None] = []
            for b, weight in zip(beta[i], a, strict=True):
                stems_x.extend([float(b), float(b), None])
                stems_y.extend([0.0, float(weight), None])
            payload.update(
                {
                    "hyper_X_x": [float(X[i, 1])],
                    "hyper_X_y": [float(X[i, 2])],
                    "hyper_X_z": [float(X[i, 0])],
                    "hyper_u_x": u_lines_x,
                    "hyper_u_y": u_lines_y,
                    "hyper_u_z": u_lines_z,
                    "hyper_U_x": [float(X[i, 1]), float(U_end[1])],
                    "hyper_U_y": [float(X[i, 2]), float(U_end[2])],
                    "hyper_U_z": [float(X[i, 0]), float(U_end[0])],
                    "screen_beta_x": beta[i],
                    "screen_beta_y": a,
                    "screen_stems_x": stems_x,
                    "screen_stems_y": stems_y,
                    "screen_theta_text_x": [0.0],
                    "screen_theta_text_y": [1.18 * max_weight],
                    "screen_theta_text": [f"Theta_perp={theta_perp[i]:+.4f}"],
                    "theta_perp": float(theta_perp[i]),
                    "A_perp": float(A_perp[i]),
                    "diag_point_x": [float(frames[i])],
                    "diag_R_point_y": [float(coherence[i])],
                    "diag_theta_point_y": [float(theta_perp[i])],
                    "diag_phi_point_y": [float(phi[i])],
                }
            )
            payload["stats"] += (
                f"; $A_\\perp={float(A_perp[i]):.6f}$; "
                f"$\\Theta_\\perp={float(theta_perp[i]):.6f}$"
            )
        return payloads

    def _apply_hyper_static(self) -> None:
        if self._hyper_static_applied or not self._hyper_static:
            return
        s = self._hyper_static
        self.fig.data[self.tr["hyper_surface"]].x = _plotly_values(s["surface_x"])
        self.fig.data[self.tr["hyper_surface"]].y = _plotly_values(s["surface_y"])
        self.fig.data[self.tr["hyper_surface"]].z = _plotly_values(s["surface_z"])
        self.fig.data[self.tr["hyper_null"]].x = _plotly_values(s["null_x"])
        self.fig.data[self.tr["hyper_null"]].y = _plotly_values(s["null_y"])
        self.fig.data[self.tr["hyper_null"]].z = _plotly_values(s["null_z"])
        self.fig.data[self.tr["hyper_geodesics"]].x = _plotly_values(s["geodesic_x"])
        self.fig.data[self.tr["hyper_geodesics"]].y = _plotly_values(s["geodesic_y"])
        self.fig.data[self.tr["hyper_geodesics"]].z = _plotly_values(s["geodesic_z"])
        self.fig.data[self.tr["hyper_path"]].x = _plotly_values(s["path_x"])
        self.fig.data[self.tr["hyper_path"]].y = _plotly_values(s["path_y"])
        self.fig.data[self.tr["hyper_path"]].z = _plotly_values(s["path_z"])
        self.fig.layout.scene.xaxis.range = s["scene_range"][0]
        self.fig.layout.scene.yaxis.range = s["scene_range"][1]
        self.fig.layout.scene.zaxis.range = s["scene_range"][2]
        self.fig.data[self.tr["screen_axis"]].x = _plotly_values(s["screen_axis_x"])
        self.fig.data[self.tr["screen_axis"]].y = _plotly_values(s["screen_axis_y"])
        self.fig.data[self.tr["diag_R"]].x = _plotly_values(s["diag_frames"])
        self.fig.data[self.tr["diag_R"]].y = _plotly_values(s["diag_R"])
        self.fig.data[self.tr["diag_theta"]].x = _plotly_values(s["diag_frames"])
        self.fig.data[self.tr["diag_theta"]].y = _plotly_values(s["diag_theta"])
        self.fig.data[self.tr["diag_phi"]].x = _plotly_values(s["diag_frames"])
        self.fig.data[self.tr["diag_phi"]].y = _plotly_values(s["diag_phi"])
        self.fig.update_xaxes(range=s["screen_x_range"], row=2, col=1)
        self.fig.update_yaxes(range=s["screen_y_range"], row=2, col=1)
        self.fig.update_yaxes(range=s["diag_y_range"], row=2, col=2, secondary_y=False)
        self.fig.update_yaxes(range=s["diag_phi_range"], row=2, col=2, secondary_y=True)
        self._hyper_static_applied = True

    def _apply_payload_to_figure(self, payload: dict[str, Any], selected: dict[str, Any]) -> None:
        _ = selected
        with self.fig.batch_update():
            self._apply_hyper_static()
            self.fig.data[self.tr["hyper_X"]].x = _plotly_values(payload["hyper_X_x"])
            self.fig.data[self.tr["hyper_X"]].y = _plotly_values(payload["hyper_X_y"])
            self.fig.data[self.tr["hyper_X"]].z = _plotly_values(payload["hyper_X_z"])
            self.fig.data[self.tr["hyper_u"]].x = _plotly_values(payload["hyper_u_x"])
            self.fig.data[self.tr["hyper_u"]].y = _plotly_values(payload["hyper_u_y"])
            self.fig.data[self.tr["hyper_u"]].z = _plotly_values(payload["hyper_u_z"])
            self.fig.data[self.tr["hyper_U"]].x = _plotly_values(payload["hyper_U_x"])
            self.fig.data[self.tr["hyper_U"]].y = _plotly_values(payload["hyper_U_y"])
            self.fig.data[self.tr["hyper_U"]].z = _plotly_values(payload["hyper_U_z"])

            self.fig.data[self.tr["ref_phase"]].x = _plotly_values(payload["ref_phase_x"])
            self.fig.data[self.tr["ref_phase"]].y = _plotly_values(payload["ref_phase_y"])
            self.fig.data[self.tr["ref_phase"]].visible = bool(self.show_contours.value)
            self.fig.data[self.tr["ref_phase"]].opacity = 0.82 if bool(self.show_contours.value) else 0.0
            self.fig.data[self.tr["reflected"]].x = _plotly_values(payload["reflected_x"])
            self.fig.data[self.tr["reflected"]].y = _plotly_values(payload["reflected_y"])
            self.fig.data[self.tr["ref_w"]].x = _plotly_values(payload["ref_w_x"])
            self.fig.data[self.tr["ref_w"]].y = _plotly_values(payload["ref_w_y"])
            self.fig.data[self.tr["R_circle"]].x = _plotly_values(payload["R_circle_x"])
            self.fig.data[self.tr["R_circle"]].y = _plotly_values(payload["R_circle_y"])
            self.fig.data[self.tr["ellipse"]].x = _plotly_values(payload["ellipse_x"])
            self.fig.data[self.tr["ellipse"]].y = _plotly_values(payload["ellipse_y"])

            self.fig.data[self.tr["screen_stems"]].x = _plotly_values(payload["screen_stems_x"])
            self.fig.data[self.tr["screen_stems"]].y = _plotly_values(payload["screen_stems_y"])
            self.fig.data[self.tr["screen_beta"]].x = _plotly_values(payload["screen_beta_x"])
            self.fig.data[self.tr["screen_beta"]].y = _plotly_values(payload["screen_beta_y"])
            self.fig.data[self.tr["screen_theta_text"]].x = _plotly_values(payload["screen_theta_text_x"])
            self.fig.data[self.tr["screen_theta_text"]].y = _plotly_values(payload["screen_theta_text_y"])
            self.fig.data[self.tr["screen_theta_text"]].text = _plotly_values(payload["screen_theta_text"])

            self.fig.data[self.tr["diag_R_point"]].x = _plotly_values(payload["diag_point_x"])
            self.fig.data[self.tr["diag_R_point"]].y = _plotly_values(payload["diag_R_point_y"])
            self.fig.data[self.tr["diag_theta_point"]].x = _plotly_values(payload["diag_point_x"])
            self.fig.data[self.tr["diag_theta_point"]].y = _plotly_values(payload["diag_theta_point_y"])
            self.fig.data[self.tr["diag_phi_point"]].x = _plotly_values(payload["diag_point_x"])
            self.fig.data[self.tr["diag_phi_point"]].y = _plotly_values(payload["diag_phi_point_y"])


class LMSOpticalBall3DWidget:
    """Temporarily disabled placeholder for the removed 3D optical backend."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _ = args, kwargs
        raise NotImplementedError(
            "LMSOpticalBall3DWidget has been temporarily removed while the 2D optical widget framing is refined."
        )


__all__ = [
    "OpticalState",
    "householder_reflected_points",
    "reflected_points",
    "mobius_reflection_error",
    "reflected_barycenter",
    "frozen_field",
    "optical_velocity",
    "busemann_phase",
    "ray_second_moment",
    "ray_covariance",
    "ray_principal_axes",
    "mirror_normals",
    "optical_state",
    "lorentz_inner",
    "hyperboloid_lift",
    "null_lift",
    "hyperboloid_ray_fields",
    "phase_grid_2d",
    "potential_level_segments_2d",
    "optical_flow_step",
    "LMSOpticalDiskBaseWidget",
    "LMSOpticalDiskWidget",
    "LMSOpticalDynamicInversionCayleyDiskWidget",
    "LMSOpticalWeightedCayleyDiskWidget",
    "LMSOpticalHyperboloidScreenWidget",
]
