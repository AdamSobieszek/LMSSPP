"""Static iframe export utilities for LMS widgets.

This module provides a non-notebook export path:
- simulate LMS trajectories from core dynamics
- package them into a JSON payload schema
- write standalone HTML + trajectory bundles for iframe hosting
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
from typing import Any, Literal

import numpy as np
import torch

try:
    from ..core.lms import (
        clamp_to_ball,
        integrate_lms_reduced_euler,
        integrate_w_euler,
        order_parameter,
        phases_to_s1,
        pushforward_points,
        random_points_on_sphere,
        skew_symmetric_from_axis,
    )
except Exception:
    from LMS import (  # type: ignore
        clamp_to_ball,
        integrate_lms_reduced_euler,
        integrate_w_euler,
        order_parameter,
        phases_to_s1,
        pushforward_points,
        random_points_on_sphere,
        skew_symmetric_from_axis,
    )


WidgetKind = Literal["ball3d", "ball3d_backward_two_sheet", "circle"]


def _angles_to_unit(az: float, el: float) -> np.ndarray:
    c = math.cos(el)
    return np.array([c * math.cos(az), c * math.sin(az), math.sin(el)], dtype=np.float64)


def _jsonable(a: np.ndarray | torch.Tensor) -> list[Any]:
    if isinstance(a, torch.Tensor):
        arr = a.detach().cpu().numpy()
    else:
        arr = np.asarray(a)
    return arr.tolist()


def _bar_sheet_map(
    x: np.ndarray,
    *,
    fallback_dir: np.ndarray | None = None,
    eps: float = 1e-9,
) -> np.ndarray:
    """Second-sheet map x̄ = (x/|x| - x)/|x/|x| - x|^2 with zero fallback."""
    arr = np.asarray(x, dtype=np.float64)
    single = arr.ndim == 1
    if single:
        arr = arr[None, :]

    if fallback_dir is None:
        f = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        f = np.asarray(fallback_dir, dtype=np.float64).reshape(3)
    fn = float(np.linalg.norm(f))
    if fn < eps:
        f = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        fn = 1.0
    f = f / fn

    r = np.linalg.norm(arr, axis=1, keepdims=True)
    u = arr / np.maximum(r, eps)
    tiny = r[:, 0] < eps
    if np.any(tiny):
        u[tiny] = f[None, :]
    diff = u - arr
    den = np.sum(diff * diff, axis=1, keepdims=True)
    out = diff / np.maximum(den, eps)
    return out[0] if single else out


def _bar_cap(x: np.ndarray, max_radius: float = 8.0, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    single = arr.ndim == 1
    if single:
        arr = arr[None, :]
    r = np.linalg.norm(arr, axis=1, keepdims=True)
    s = np.minimum(1.0, float(max_radius) / np.maximum(r, eps))
    out = arr * s
    return out[0] if single else out


def _simulate_ball3d_payload(
    *,
    widget: Literal["ball3d", "ball3d_backward_two_sheet"],
    params: dict[str, Any],
    include_points: bool,
    point_decimation: int,
    seed: int,
    w_mode: Literal["explicit", "autograd"],
) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "N": 150,
        "K": 1.0,
        "omega": 3.0,
        "r0": 0.03,
        "w_az": 0.2,
        "w_el": 0.25,
        "ax_az": 0.0,
        "ax_el": 0.5 * math.pi,
        "dt": 5e-2,
        "steps": 400,
    }
    cfg = dict(defaults)
    cfg.update(params or {})

    n = int(cfg["N"])
    d = 3
    K = float(cfg["K"])
    omega = float(cfg["omega"])
    r0 = float(cfg["r0"])
    dt = float(cfg["dt"])
    if widget == "ball3d_backward_two_sheet":
        dt = -abs(dt)
    steps = int(cfg["steps"])
    w_az = float(cfg["w_az"])
    w_el = float(cfg["w_el"])
    ax_az = float(cfg["ax_az"])
    ax_el = float(cfg["ax_el"])

    gen = torch.Generator().manual_seed(int(seed))
    base_points = random_points_on_sphere(n, d=d, dtype=torch.float64, generator=gen)
    weights = torch.ones(n, dtype=torch.float64) / float(n)

    w_dir = torch.tensor(_angles_to_unit(w_az, w_el), dtype=torch.float64)
    w0 = clamp_to_ball(w_dir * float(r0), radius=0.999999)
    zeta0 = torch.eye(d, dtype=torch.float64)
    axis = torch.tensor(_angles_to_unit(ax_az, ax_el), dtype=torch.float64)
    A = skew_symmetric_from_axis(axis, rate=omega).to(dtype=torch.float64)

    traj = integrate_lms_reduced_euler(
        w0=w0,
        zeta0=zeta0,
        base_points=base_points,
        weights=weights,
        A=A,
        coupling=K,
        dt=dt,
        steps=steps,
        w_mode=w_mode,
        store_points="lab" if include_points else "none",
        store_dtype=torch.float32,
        preallocate=True,
        project_rotation=True,
    )

    w = traj.w.detach().cpu().numpy().astype(np.float64)
    z = traj.z.detach().cpu().numpy().astype(np.float64)
    z_over_k = (traj.Z.detach().cpu().numpy().astype(np.float64) / max(K, 1e-12))
    t = np.arange(steps + 1, dtype=np.float64) * dt
    metrics: dict[str, Any] = {
        "w_norm": np.linalg.norm(w, axis=1).tolist(),
        "z_norm": np.linalg.norm(z, axis=1).tolist(),
        "Z_norm_over_K": np.linalg.norm(z_over_k, axis=1).tolist(),
    }

    series: dict[str, Any] = {
        "w": w.tolist(),
        "z": z.tolist(),
        "Z_over_K": z_over_k.tolist(),
    }
    if include_points and traj.x_lab is not None:
        dec = max(1, int(point_decimation))
        pts = traj.x_lab.detach().cpu().numpy().astype(np.float64)
        series["points"] = pts[:, ::dec, :].tolist()

    if widget == "ball3d_backward_two_sheet":
        fw = np.asarray(_angles_to_unit(w_az, w_el), dtype=np.float64)
        fz = z.copy()
        fzn = np.linalg.norm(fz, axis=1, keepdims=True)
        fz[fzn[:, 0] < 1e-9] = fw[None, :]
        fZ = z_over_k.copy()
        fZn = np.linalg.norm(fZ, axis=1, keepdims=True)
        fZ[fZn[:, 0] < 1e-9] = fw[None, :]
        wb = _bar_cap(_bar_sheet_map(w, fallback_dir=fw), max_radius=8.0)
        zb = _bar_cap(_bar_sheet_map(z, fallback_dir=fw), max_radius=8.0)
        Zb = _bar_cap(_bar_sheet_map(z_over_k, fallback_dir=fw), max_radius=8.0)
        series["bar_sheet"] = {
            "w": wb.tolist(),
            "z": zb.tolist(),
            "Z_over_K": Zb.tolist(),
        }
        metrics["bar_w_norm"] = np.linalg.norm(wb, axis=1).tolist()
        metrics["bar_z_norm"] = np.linalg.norm(zb, axis=1).tolist()
        metrics["bar_Z_norm"] = np.linalg.norm(Zb, axis=1).tolist()

    payload: dict[str, Any] = {
        "schema_version": "lms_iframe_static_v1",
        "widget_type": widget,
        "params": {
            "N": n,
            "K": K,
            "omega": omega,
            "r0": r0,
            "w_az": w_az,
            "w_el": w_el,
            "ax_az": ax_az,
            "ax_el": ax_el,
            "dt": dt,
            "steps": steps,
            "spatial_dim": d,
        },
        "time": t.tolist(),
        "series": series,
        "metrics": metrics,
        "meta": {
            "seed": int(seed),
            "w_mode": str(w_mode),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    return payload


def _simulate_circle_payload(
    *,
    params: dict[str, Any],
    include_points: bool,
    point_decimation: int,
    seed: int,
    w_mode: Literal["explicit", "autograd"],
) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "N": 60,
        "K": 1.0,
        "r0": 0.02,
        "theta": 0.45,
        "dt": 0.012,
        "steps": 420,
    }
    cfg = dict(defaults)
    cfg.update(params or {})

    n = int(cfg["N"])
    K = float(cfg["K"])
    r0 = float(cfg["r0"])
    theta = float(cfg["theta"])
    dt = float(cfg["dt"])
    steps = int(cfg["steps"])

    base_theta = torch.linspace(0.0, 2.0 * math.pi, n + 1, dtype=torch.float64)[:-1]
    base_points = phases_to_s1(base_theta)
    weights = torch.ones(n, dtype=torch.float64) / float(n)

    w0 = torch.tensor([math.cos(theta), math.sin(theta)], dtype=torch.float64) * r0
    field_name: Literal["mobius", "autograd"] = "autograd" if w_mode == "autograd" else "mobius"
    sim = integrate_w_euler(
        w0=w0,
        base_points=base_points,
        weights=K * weights,
        dt=dt,
        steps=steps,
        field=field_name,
    )

    w_t = sim.trajectory.to(dtype=torch.float64)
    z_t = -w_t
    z_over_k = torch.stack([order_parameter(w_t[i], base_points, weights) for i in range(w_t.shape[0])], dim=0)
    t = torch.arange(steps + 1, dtype=torch.float64) * dt

    series: dict[str, Any] = {
        "w": _jsonable(w_t),
        "z": _jsonable(z_t),
        "Z_over_K": _jsonable(z_over_k),
    }
    if include_points:
        dec = max(1, int(point_decimation))
        points = torch.stack([pushforward_points(w_t[i], base_points) for i in range(w_t.shape[0])], dim=0)
        series["points"] = _jsonable(points[:, ::dec, :])

    metrics: dict[str, Any] = {
        "w_norm": _jsonable(torch.linalg.norm(w_t, dim=1)),
        "z_norm": _jsonable(torch.linalg.norm(z_t, dim=1)),
        "Z_norm_over_K": _jsonable(torch.linalg.norm(z_over_k, dim=1)),
    }

    payload: dict[str, Any] = {
        "schema_version": "lms_iframe_static_v1",
        "widget_type": "circle",
        "params": {
            "N": n,
            "K": K,
            "r0": r0,
            "theta": theta,
            "dt": dt,
            "steps": steps,
            "spatial_dim": 2,
        },
        "time": _jsonable(t),
        "series": series,
        "metrics": metrics,
        "meta": {
            "seed": int(seed),
            "w_mode": str(w_mode),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    return payload


def export_lms_static_payload(
    *,
    widget: WidgetKind = "ball3d",
    params: dict[str, Any] | None = None,
    include_points: bool = True,
    point_decimation: int = 1,
    seed: int = 0,
    w_mode: Literal["explicit", "autograd"] = "autograd",
) -> dict[str, Any]:
    """Build JSON-serializable trajectory payload for iframe/static rendering."""
    p = params or {}
    if widget in {"ball3d", "ball3d_backward_two_sheet"}:
        return _simulate_ball3d_payload(
            widget=widget,
            params=p,
            include_points=include_points,
            point_decimation=point_decimation,
            seed=seed,
            w_mode=w_mode,
        )
    if widget == "circle":
        return _simulate_circle_payload(
            params=p,
            include_points=include_points,
            point_decimation=point_decimation,
            seed=seed,
            w_mode=w_mode,
        )
    raise ValueError("widget must be one of: 'ball3d', 'ball3d_backward_two_sheet', 'circle'.")


def write_lms_static_bundle(
    out_dir: str | Path,
    *,
    widget: WidgetKind = "ball3d",
    params: dict[str, Any] | None = None,
    include_points: bool = True,
    point_decimation: int = 1,
    seed: int = 0,
    w_mode: Literal["explicit", "autograd"] = "autograd",
    api_base: str = "/api/recompute",
) -> Path:
    """Write standalone bundle: index.html + trajectory.json + metadata.json."""
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    payload = export_lms_static_payload(
        widget=widget,
        params=params,
        include_points=include_points,
        point_decimation=point_decimation,
        seed=seed,
        w_mode=w_mode,
    )

    traj_path = out / "trajectory.json"
    with traj_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    metadata = {
        "schema_version": "lms_iframe_bundle_v1",
        "widget": widget,
        "trajectory": "trajectory.json",
        "api_recompute_endpoint": api_base,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (out / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    root = Path(__file__).resolve().parents[2]
    template = root / "deploy" / "iframe_app" / "static" / "index.html"
    if template.exists():
        shutil.copyfile(template, out / "index.html")
    else:
        (out / "index.html").write_text(
            "<!doctype html><html><body><pre>Open trajectory.json from this directory.</pre></body></html>",
            encoding="utf-8",
        )

    # Bundle-local defaults consumed by the static player.
    with (out / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "trajectory_url": "./trajectory.json",
                "api_recompute_url": api_base,
                "widget": widget,
            },
            f,
            indent=2,
        )

    return out / "index.html"


__all__ = [
    "WidgetKind",
    "export_lms_static_payload",
    "write_lms_static_bundle",
]
