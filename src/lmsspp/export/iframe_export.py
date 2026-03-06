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


def _snapshot_player_html(payload_json: str) -> str:
    safe_payload = str(payload_json).replace("</", "<\\/")
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>LMS Snapshot Export</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body { margin: 0; font-family: "Segoe UI", Roboto, Arial, sans-serif; background: #f6f7fb; color: #1f2d4a; }
    .wrap { padding: 12px; max-width: 1440px; margin: 0 auto; }
    .card { background: #fff; border: 1px solid #d8dfed; border-radius: 10px; padding: 10px; }
    .title { margin: 0 0 8px 0; font-size: 20px; }
    .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 8px; }
    .row label { font-size: 12px; color: #54607a; }
    .row select, .row button, .row input { font-size: 12px; padding: 6px 8px; border-radius: 6px; border: 1px solid #cfd7e8; background: #fff; }
    .row button { cursor: pointer; }
    .frame { display: flex; gap: 8px; align-items: center; margin-bottom: 8px; }
    .frame input[type=range] { flex: 1; }
    #scene { height: 600px; border: 1px solid #d8dfed; border-radius: 8px; }
    #metrics { height: 520px; border: 1px solid #d8dfed; border-radius: 8px; margin-top: 8px; }
    #status { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; color: #1f5dd2; margin-bottom: 8px; }
    #meta { margin-top: 8px; border: 1px solid #d8dfed; border-radius: 8px; padding: 8px; background: #fbfcff; }
    #meta h3 { margin: 0 0 6px 0; font-size: 14px; }
    #meta pre { margin: 0; white-space: pre-wrap; font-size: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .stats { margin-top: 8px; border: 1px solid #d8dfed; border-radius: 8px; padding: 8px; background: #fff; }
  </style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1 class="title" id="title">LMS snapshot export</h1>
    <div class="row">
      <button id="playFwd">Play forward</button>
      <button id="playBack">Play backward</button>
      <button id="pause">Pause</button>
      <button id="stepBack">Step -</button>
      <button id="stepFwd">Step +</button>
      <label for="speed">Speed</label>
      <input id="speed" type="number" min="0.1" step="0.1" value="1.0" style="width:80px" />
      <label for="frameView">Frame</label>
      <select id="frameView">
        <option value="lab">Lab</option>
        <option value="body">Co-rotating</option>
      </select>
      <label for="modeSel">State</label>
      <select id="modeSel"></select>
    </div>
    <div class="frame">
      <label for="frame">Frame</label>
      <input id="frame" type="range" min="0" max="0" value="0" />
      <span id="frameLabel">0 / 0</span>
    </div>
    <div id="status"></div>
    <div id="scene"></div>
    <div id="metrics"></div>
    <div id="meta"><h3>Initialization and parameters</h3><pre id="metaText"></pre></div>
    <div class="stats" id="statsBox"></div>
  </div>
</div>
<script id="payload-json" type="application/json">__PAYLOAD_JSON__</script>
<script>
(() => {
  const payload = JSON.parse(document.getElementById("payload-json").textContent);
  const el = {
    title: document.getElementById("title"),
    playFwd: document.getElementById("playFwd"),
    playBack: document.getElementById("playBack"),
    pause: document.getElementById("pause"),
    stepBack: document.getElementById("stepBack"),
    stepFwd: document.getElementById("stepFwd"),
    speed: document.getElementById("speed"),
    frameView: document.getElementById("frameView"),
    modeSel: document.getElementById("modeSel"),
    frame: document.getElementById("frame"),
    frameLabel: document.getElementById("frameLabel"),
    status: document.getElementById("status"),
    scene: document.getElementById("scene"),
    metrics: document.getElementById("metrics"),
    metaText: document.getElementById("metaText"),
    statsBox: document.getElementById("statsBox")
  };

  const state = {
    timer: null,
    direction: 1,
    frame: 0,
    frameName: ((payload.ui_state || {}).frame_name_default || "lab") === "body" ? "body" : "lab",
    mode: ((payload.ui_state || {}).display_mode || null),
    idx: {}
  };

  function asNum(v) {
    const x = Number(v);
    return Number.isFinite(x) ? x : 0.0;
  }

  function activeBundle() {
    if (payload.ensembles) {
      const key = String(state.mode || ((payload.mode_order || [])[0] || ""));
      const entry = payload.ensembles[key];
      if (entry && entry.bundle) return entry.bundle;
    }
    return payload.bundle || null;
  }

  function sphereWire() {
    const traces = [];
    for (let i = 1; i <= 5; i++) {
      const phi = -0.5 * Math.PI + (i * Math.PI) / 6.0;
      const x = [], y = [], z = [];
      for (let k = 0; k <= 72; k++) {
        const t = (2.0 * Math.PI * k) / 72.0;
        x.push(Math.cos(phi) * Math.cos(t));
        y.push(Math.cos(phi) * Math.sin(t));
        z.push(Math.sin(phi));
      }
      traces.push({ type: "scatter3d", mode: "lines", x, y, z, hoverinfo: "skip", showlegend: false, line: { color: "rgba(140,140,140,0.24)", width: 1 } });
    }
    for (let j = 0; j < 8; j++) {
      const lam = (2.0 * Math.PI * j) / 8.0;
      const x = [], y = [], z = [];
      for (let k = 0; k <= 72; k++) {
        const t = -0.5 * Math.PI + (Math.PI * k) / 72.0;
        x.push(Math.cos(t) * Math.cos(lam));
        y.push(Math.cos(t) * Math.sin(lam));
        z.push(Math.sin(t));
      }
      traces.push({ type: "scatter3d", mode: "lines", x, y, z, hoverinfo: "skip", showlegend: false, line: { color: "rgba(140,140,140,0.24)", width: 1 } });
    }
    return traces;
  }

  function reconstructPoints(bundle, frameIdx, frameName) {
    const base = bundle.base_points_display || bundle.base_points || [];
    const w = (bundle.w || [])[frameIdx] || [0, 0, 0];
    const zeta = (bundle.zeta || [])[frameIdx] || [[1,0,0],[0,1,0],[0,0,1]];
    const w0 = asNum(w[0]), w1 = asNum(w[1]), w2 = asNum(w[2]);
    const wNorm2 = w0*w0 + w1*w1 + w2*w2;
    const out = new Array(base.length);
    for (let i = 0; i < base.length; i++) {
      const p = base[i] || [0, 0, 0];
      const d0 = asNum(p[0]) - w0;
      const d1 = asNum(p[1]) - w1;
      const d2 = asNum(p[2]) - w2;
      const den = Math.max(1e-12, d0*d0 + d1*d1 + d2*d2);
      let x0 = ((1.0 - wNorm2) / den) * d0 - w0;
      let x1 = ((1.0 - wNorm2) / den) * d1 - w1;
      let x2 = ((1.0 - wNorm2) / den) * d2 - w2;
      const n = Math.max(1e-12, Math.sqrt(x0*x0 + x1*x1 + x2*x2));
      x0 /= n; x1 /= n; x2 /= n;
      if (frameName === "body") {
        out[i] = [x0, x1, x2];
      } else {
        out[i] = [
          x0*asNum(zeta[0][0]) + x1*asNum(zeta[1][0]) + x2*asNum(zeta[2][0]),
          x0*asNum(zeta[0][1]) + x1*asNum(zeta[1][1]) + x2*asNum(zeta[2][1]),
          x0*asNum(zeta[0][2]) + x1*asNum(zeta[1][2]) + x2*asNum(zeta[2][2])
        ];
      }
    }
    return out;
  }

  function coords(vectors) {
    const x = [], y = [], z = [];
    for (const v of vectors || []) {
      x.push(asNum((v || [])[0]));
      y.push(asNum((v || [])[1]));
      z.push(asNum((v || [])[2]));
    }
    return { x, y, z };
  }

  function vectorLine(v) {
    return {
      x: [0.0, asNum((v || [])[0])],
      y: [0.0, asNum((v || [])[1])],
      z: [0.0, asNum((v || [])[2])]
    };
  }

  function status(text) {
    el.status.textContent = text || "";
  }

  function maxFrame() {
    const b = activeBundle();
    if (!b || !b.w) return 0;
    return Math.max(0, b.w.length - 1);
  }

  function clampFrame(i) {
    const last = maxFrame();
    if (i < 0) return last;
    if (i > last) return 0;
    return i;
  }

  async function drawScene() {
    const b = activeBundle();
    if (!b) return;
    const traces = sphereWire();
    const idx = {};
    const frame0 = clampFrame(state.frame);
    const points0 = reconstructPoints(b, frame0, state.frameName);
    const pointsC = coords(points0);
    idx.points = traces.length;
    traces.push({ type: "scatter3d", mode: "markers", name: "x_i(t)", marker: { size: 3, color: "royalblue" }, ...pointsC });

    idx.wMarker = traces.length;
    traces.push({ type: "scatter3d", mode: "markers", name: "w(t)", marker: { size: 7, color: "black", symbol: "x" }, x: [0], y: [0], z: [0] });
    idx.zMarker = traces.length;
    traces.push({ type: "scatter3d", mode: "markers", name: "z(t)", marker: { size: 6, color: "white", line: { color: "black", width: 2 }, symbol: "circle-open" }, x: [0], y: [0], z: [0] });
    idx.ZMarker = traces.length;
    traces.push({ type: "scatter3d", mode: "markers", name: "Z/K", marker: { size: 6, color: "firebrick", symbol: "diamond" }, x: [0], y: [0], z: [0] });

    idx.wPath = traces.length;
    traces.push({ type: "scatter3d", mode: "lines", name: "w path", line: { color: "black", width: 2, dash: "dot" }, x: [], y: [], z: [] });
    idx.zPath = traces.length;
    traces.push({ type: "scatter3d", mode: "lines", name: "z path", line: { color: "gray", width: 2, dash: "dot" }, x: [], y: [], z: [] });
    idx.ZPath = traces.length;
    traces.push({ type: "scatter3d", mode: "lines", name: "Z/K path", line: { color: "firebrick", width: 2, dash: "dot" }, x: [], y: [], z: [] });

    idx.wVec = traces.length;
    traces.push({ type: "scatter3d", mode: "lines", name: "w vector", line: { color: "black", width: 3 }, x: [0,0], y: [0,0], z: [0,0] });
    idx.zVec = traces.length;
    traces.push({ type: "scatter3d", mode: "lines", name: "z vector", line: { color: "gray", width: 3 }, x: [0,0], y: [0,0], z: [0,0] });
    idx.ZVec = traces.length;
    traces.push({ type: "scatter3d", mode: "lines", name: "Z/K vector", line: { color: "firebrick", width: 3 }, x: [0,0], y: [0,0], z: [0,0] });

    const hasBar = !!b.bar_sheet;
    if (hasBar) {
      idx.barWMarker = traces.length;
      traces.push({ type: "scatter3d", mode: "markers", name: "w_bar(t)", marker: { size: 6, color: "black", symbol: "diamond" }, x: [0], y: [0], z: [0] });
      idx.barZMarker = traces.length;
      traces.push({ type: "scatter3d", mode: "markers", name: "z_bar(t)", marker: { size: 6, color: "gray", symbol: "diamond-open" }, x: [0], y: [0], z: [0] });
      idx.barZKMarker = traces.length;
      traces.push({ type: "scatter3d", mode: "markers", name: "Z_bar/K", marker: { size: 6, color: "firebrick", symbol: "diamond" }, x: [0], y: [0], z: [0] });
      idx.barWPath = traces.length;
      traces.push({ type: "scatter3d", mode: "lines", name: "w_bar path", line: { color: "black", width: 2 }, x: [], y: [], z: [] });
      idx.barZPath = traces.length;
      traces.push({ type: "scatter3d", mode: "lines", name: "z_bar path", line: { color: "gray", width: 2 }, x: [], y: [], z: [] });
      idx.barZKPath = traces.length;
      traces.push({ type: "scatter3d", mode: "lines", name: "Z_bar/K path", line: { color: "firebrick", width: 2 }, x: [], y: [], z: [] });
      idx.barWVec = traces.length;
      traces.push({ type: "scatter3d", mode: "lines", name: "w_bar vector", line: { color: "black", width: 3 }, x: [0,0], y: [0,0], z: [0,0] });
      idx.barZVec = traces.length;
      traces.push({ type: "scatter3d", mode: "lines", name: "z_bar vector", line: { color: "gray", width: 3 }, x: [0,0], y: [0,0], z: [0,0] });
      idx.barZKVec = traces.length;
      traces.push({ type: "scatter3d", mode: "lines", name: "Z_bar/K vector", line: { color: "firebrick", width: 3 }, x: [0,0], y: [0,0], z: [0,0] });
    }

    state.idx = idx;
    await Plotly.newPlot(el.scene, traces, {
      margin: { l: 8, r: 8, t: 40, b: 8 },
      title: { text: String(payload.title || "LMS snapshot"), x: 0.01, xanchor: "left" },
      legend: { orientation: "h", x: 0, y: 1.03 },
      scene: { aspectmode: "cube", xaxis: { visible: false }, yaxis: { visible: false }, zaxis: { visible: false } }
    }, { responsive: true, displaylogo: false });
    await refreshStaticSeries();
    await updateFrame(frame0);
  }

  async function refreshStaticSeries() {
    const b = activeBundle();
    if (!b) return;
    const idx = state.idx;
    const zSeries = state.frameName === "body" ? (b.z_body || []) : (b.z_lab || []);
    const ZSeries = state.frameName === "body" ? (b.Z_body || []) : (b.Z_lab || []);
    const wC = coords(b.w || []);
    const zC = coords(zSeries);
    const ZC = coords(ZSeries);
    const tasks = [];
    tasks.push(Plotly.restyle(el.scene, { x: [wC.x], y: [wC.y], z: [wC.z] }, [idx.wPath]));
    tasks.push(Plotly.restyle(el.scene, { x: [zC.x], y: [zC.y], z: [zC.z] }, [idx.zPath]));
    tasks.push(Plotly.restyle(el.scene, { x: [ZC.x], y: [ZC.y], z: [ZC.z] }, [idx.ZPath]));
    if (b.bar_sheet) {
      const bW = coords((b.bar_sheet.w || []));
      const bZ = coords(state.frameName === "body" ? (b.bar_sheet.z_body || []) : (b.bar_sheet.z_lab || []));
      const bZK = coords(state.frameName === "body" ? (b.bar_sheet.Z_body || []) : (b.bar_sheet.Z_lab || []));
      tasks.push(Plotly.restyle(el.scene, { x: [bW.x], y: [bW.y], z: [bW.z] }, [idx.barWPath]));
      tasks.push(Plotly.restyle(el.scene, { x: [bZ.x], y: [bZ.y], z: [bZ.z] }, [idx.barZPath]));
      tasks.push(Plotly.restyle(el.scene, { x: [bZK.x], y: [bZK.y], z: [bZK.z] }, [idx.barZKPath]));
    }
    await Promise.all(tasks);
  }

  async function updateFrame(frameIndex) {
    const b = activeBundle();
    if (!b) return;
    const i = clampFrame(frameIndex);
    state.frame = i;
    const last = maxFrame();
    el.frame.value = String(i);
    el.frameLabel.textContent = i + " / " + last;

    const points = reconstructPoints(b, i, state.frameName);
    const pC = coords(points);
    const w = (b.w || [])[i] || [0,0,0];
    const z = (state.frameName === "body" ? (b.z_body || []) : (b.z_lab || []))[i] || [0,0,0];
    const Z = (state.frameName === "body" ? (b.Z_body || []) : (b.Z_lab || []))[i] || [0,0,0];
    const idx = state.idx;
    const tasks = [];
    tasks.push(Plotly.restyle(el.scene, { x: [pC.x], y: [pC.y], z: [pC.z] }, [idx.points]));
    tasks.push(Plotly.restyle(el.scene, { x: [[asNum(w[0])]], y: [[asNum(w[1])]], z: [[asNum(w[2])]] }, [idx.wMarker]));
    tasks.push(Plotly.restyle(el.scene, { x: [[asNum(z[0])]], y: [[asNum(z[1])]], z: [[asNum(z[2])]] }, [idx.zMarker]));
    tasks.push(Plotly.restyle(el.scene, { x: [[asNum(Z[0])]], y: [[asNum(Z[1])]], z: [[asNum(Z[2])]] }, [idx.ZMarker]));
    const wVec = vectorLine(w), zVec = vectorLine(z), ZVec = vectorLine(Z);
    tasks.push(Plotly.restyle(el.scene, { x: [wVec.x], y: [wVec.y], z: [wVec.z] }, [idx.wVec]));
    tasks.push(Plotly.restyle(el.scene, { x: [zVec.x], y: [zVec.y], z: [zVec.z] }, [idx.zVec]));
    tasks.push(Plotly.restyle(el.scene, { x: [ZVec.x], y: [ZVec.y], z: [ZVec.z] }, [idx.ZVec]));

    if (b.bar_sheet) {
      const bw = (b.bar_sheet.w || [])[i] || [0,0,0];
      const bz = (state.frameName === "body" ? (b.bar_sheet.z_body || []) : (b.bar_sheet.z_lab || []))[i] || [0,0,0];
      const bZ = (state.frameName === "body" ? (b.bar_sheet.Z_body || []) : (b.bar_sheet.Z_lab || []))[i] || [0,0,0];
      tasks.push(Plotly.restyle(el.scene, { x: [[asNum(bw[0])]], y: [[asNum(bw[1])]], z: [[asNum(bw[2])]] }, [idx.barWMarker]));
      tasks.push(Plotly.restyle(el.scene, { x: [[asNum(bz[0])]], y: [[asNum(bz[1])]], z: [[asNum(bz[2])]] }, [idx.barZMarker]));
      tasks.push(Plotly.restyle(el.scene, { x: [[asNum(bZ[0])]], y: [[asNum(bZ[1])]], z: [[asNum(bZ[2])]] }, [idx.barZKMarker]));
      const bwVec = vectorLine(bw), bzVec = vectorLine(bz), bZVec = vectorLine(bZ);
      tasks.push(Plotly.restyle(el.scene, { x: [bwVec.x], y: [bwVec.y], z: [bwVec.z] }, [idx.barWVec]));
      tasks.push(Plotly.restyle(el.scene, { x: [bzVec.x], y: [bzVec.y], z: [bzVec.z] }, [idx.barZVec]));
      tasks.push(Plotly.restyle(el.scene, { x: [bZVec.x], y: [bZVec.y], z: [bZVec.z] }, [idx.barZKVec]));
    }
    await Promise.all(tasks);

    const wNorm = Math.sqrt(asNum(w[0])**2 + asNum(w[1])**2 + asNum(w[2])**2);
    const zNorm = Math.sqrt(asNum(z[0])**2 + asNum(z[1])**2 + asNum(z[2])**2);
    const ZNorm = Math.sqrt(asNum(Z[0])**2 + asNum(Z[1])**2 + asNum(Z[2])**2);
    status("frame=" + i + "  |w|=" + wNorm.toFixed(5) + "  |z|=" + zNorm.toFixed(5) + "  |Z|/K=" + ZNorm.toFixed(5));
  }

  async function drawMetrics() {
    const m = payload.metrics_figure;
    if (!m || !m.data) {
      el.metrics.style.display = "none";
      return;
    }
    await Plotly.newPlot(el.metrics, m.data, m.layout || {}, { responsive: true, displaylogo: false });
  }

  function stopPlayback() {
    if (state.timer) {
      clearInterval(state.timer);
      state.timer = null;
    }
  }

  function startPlayback(direction) {
    stopPlayback();
    state.direction = direction;
    const speed = Math.max(0.1, asNum(el.speed.value) || 1.0);
    const intervalMs = Math.max(8, Math.round(33.0 / speed));
    state.timer = setInterval(() => {
      updateFrame(state.frame + state.direction).catch((err) => {
        stopPlayback();
        status("playback error: " + String(err));
      });
    }, intervalMs);
  }

  function setupControls() {
    const modes = payload.mode_order || (payload.ensembles ? Object.keys(payload.ensembles) : ["current"]);
    el.modeSel.innerHTML = "";
    for (const m of modes) {
      const opt = document.createElement("option");
      const label = payload.ensembles && payload.ensembles[m] && payload.ensembles[m].label ? payload.ensembles[m].label : m;
      opt.value = m;
      opt.textContent = String(label);
      el.modeSel.appendChild(opt);
    }
    if (!payload.ensembles) {
      el.modeSel.disabled = true;
    }
    if (!state.mode && modes.length > 0) state.mode = String(modes[0]);
    el.modeSel.value = String(state.mode || "");
    el.frameView.value = state.frameName;

    el.playFwd.addEventListener("click", () => startPlayback(+1));
    el.playBack.addEventListener("click", () => startPlayback(-1));
    el.pause.addEventListener("click", () => stopPlayback());
    el.stepBack.addEventListener("click", () => { stopPlayback(); updateFrame(state.frame - 1); });
    el.stepFwd.addEventListener("click", () => { stopPlayback(); updateFrame(state.frame + 1); });
    el.frame.addEventListener("input", () => { stopPlayback(); updateFrame(asNum(el.frame.value)); });
    el.frameView.addEventListener("change", async () => {
      stopPlayback();
      state.frameName = el.frameView.value === "body" ? "body" : "lab";
      await refreshStaticSeries();
      await updateFrame(state.frame);
    });
    el.modeSel.addEventListener("change", async () => {
      stopPlayback();
      state.mode = String(el.modeSel.value || "");
      el.frame.max = String(maxFrame());
      await drawScene();
    });
  }

  function renderMeta() {
    el.title.textContent = String(payload.title || "LMS snapshot export");
    const params = payload.params || {};
    const initInfo = payload.init_info || {};
    const text = [
      "widget_kind: " + String(payload.widget_kind || "unknown"),
      "exported_at_utc: " + String(payload.exported_at_utc || ""),
      "",
      "params:",
      JSON.stringify(params, null, 2),
      "",
      "init_info:",
      JSON.stringify(initInfo, null, 2)
    ].join("\\n");
    el.metaText.textContent = text;
    el.statsBox.innerHTML = payload.stats_html_snapshot || "";
  }

  async function init() {
    setupControls();
    renderMeta();
    const last = maxFrame();
    el.frame.max = String(last);
    const frameDefault = asNum((payload.ui_state || {}).frame_default || 0);
    state.frame = clampFrame(frameDefault);
    await drawScene();
    await drawMetrics();
    await updateFrame(state.frame);
  }

  init().catch((err) => status("initialization failed: " + String(err)));
})();
</script>
</body>
</html>
"""
    return html.replace("__PAYLOAD_JSON__", safe_payload)


def write_lms_widget_snapshot_html(out_path: str | Path, *, payload: dict[str, Any]) -> Path:
    """Write one self-contained iframe HTML from an in-memory widget snapshot."""
    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload_json = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    out.write_text(_snapshot_player_html(payload_json), encoding="utf-8")
    return out


__all__ = [
    "WidgetKind",
    "export_lms_static_payload",
    "write_lms_static_bundle",
    "write_lms_widget_snapshot_html",
]
