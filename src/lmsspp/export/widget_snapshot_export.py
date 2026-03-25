"""Standalone static snapshot export for LMS 3D widgets."""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any

from plotly.offline.offline import get_plotlyjs


def _snapshot_clone_html(payload_json: str, plotly_js: str, *, title: str) -> str:
    safe_payload = payload_json.replace("</script>", "<\\/script>")
    safe_title = escape(str(title))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title}</title>
<style>
body {{
  margin: 0;
  padding: 16px;
  background: #ffffff;
  color: #1f2937;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
.lmsspp-page {{
  display: flex;
  flex-direction: column;
  gap: 10px;
  width: max-content;
}}
.lmsspp-root {{
  display: flex;
  gap: 18px;
  align-items: flex-start;
  width: max-content;
}}
.lmsspp-root.layout-top {{
  flex-direction: column;
}}
.lmsspp-root.layout-side {{
  flex-direction: row;
}}
.lmsspp-bottom {{
  display: flex;
  flex-direction: column;
  gap: 12px;
}}
.lmsspp-bottom-row {{
  display: flex;
  gap: 16px;
  align-items: flex-start;
}}
.lmsspp-controls {{
  display: flex;
  flex-direction: column;
  gap: 8px;
}}
.lmsspp-box-vbox {{
  display: flex;
  flex-direction: column;
  gap: 8px;
  align-items: flex-start;
}}
.lmsspp-box-hbox {{
  display: flex;
  flex-direction: row;
  gap: 8px;
  align-items: flex-start;
  flex-wrap: wrap;
}}
.lmsspp-accordion {{
  width: 100%;
}}
.lmsspp-accordion > details {{
  border: 1px solid #d7dde8;
  border-radius: 6px;
  background: #fafbfd;
}}
.lmsspp-accordion > details > summary {{
  cursor: pointer;
  padding: 8px 10px;
  font-weight: 600;
}}
.lmsspp-accordion-body {{
  padding: 8px 10px 10px 10px;
}}
.lmsspp-leaf {{
  width: 100%;
  box-sizing: border-box;
}}
.lmsspp-button {{
  min-height: 32px;
  padding: 6px 10px;
  border: 1px solid #c7cfdb;
  border-radius: 4px;
  background: #f4f6fa;
  cursor: pointer;
  color: #243447;
  white-space: nowrap;
}}
.lmsspp-button:hover {{
  background: #ebeff6;
}}
.lmsspp-button.is-active {{
  box-shadow: inset 0 0 0 2px rgba(31,119,180,0.15);
}}
.lmsspp-button.style-success {{
  background: #e9f7ef;
  border-color: #9dd2b2;
}}
.lmsspp-button.style-info {{
  background: #ebf4ff;
  border-color: #9dbfe9;
}}
.lmsspp-button.style-warning {{
  background: #fff5e8;
  border-color: #e8c08a;
}}
.lmsspp-button-row {{
  display: flex;
  gap: 8px;
  align-items: center;
}}
.lmsspp-select,
.lmsspp-range {{
  width: 100%;
  box-sizing: border-box;
}}
.lmsspp-slider {{
  display: grid;
  grid-template-columns: minmax(160px, auto) minmax(240px, 1fr) auto;
  gap: 10px;
  align-items: center;
  width: 100%;
}}
.lmsspp-select-row {{
  display: grid;
  grid-template-columns: minmax(160px, auto) minmax(180px, 1fr);
  gap: 10px;
  align-items: center;
  width: 100%;
}}
.lmsspp-checkbox-row {{
  display: flex;
  align-items: center;
  gap: 8px;
}}
.lmsspp-readout {{
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  min-width: 52px;
  text-align: right;
}}
.lmsspp-html {{
  width: 100%;
}}
.lmsspp-stats {{
  width: 230px;
}}
.lmsspp-notice {{
  min-height: 22px;
  padding: 2px 0;
  font-size: 12px;
  color: #b00020;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}}
.lmsspp-backend-fixed {{
  opacity: 0.62;
  filter: grayscale(1);
}}
.lmsspp-backend-fixed .lmsspp-button,
.lmsspp-backend-fixed .lmsspp-select,
.lmsspp-backend-fixed .lmsspp-range,
.lmsspp-backend-fixed input {{
  color: #222;
}}
</style>
</head>
<body>
<div id="lmsspp-page" class="lmsspp-page">
  <div id="lmsspp-notice" class="lmsspp-notice"></div>
  <div id="lmsspp-root" class="lmsspp-root"></div>
</div>
<script>{plotly_js}</script>
<script id="lmsspp-payload" type="application/json">{safe_payload}</script>
<script>
(() => {{
  const payload = JSON.parse(document.getElementById("lmsspp-payload").textContent);
  const config = {{displaylogo: false, responsive: false, scrollZoom: false}};
  const page = document.getElementById("lmsspp-page");
  const root = document.getElementById("lmsspp-root");
  const sceneWrap = document.createElement("div");
  const bottomWrap = document.createElement("div");
  const controlsWrap = document.createElement("div");
  const noticeWrap = document.getElementById("lmsspp-notice");
  const bottomRow = document.createElement("div");
  const statsWrap = document.createElement("div");
  const metricsWrap = document.createElement("div");

  bottomWrap.className = "lmsspp-bottom";
  controlsWrap.className = "lmsspp-controls";
  bottomRow.className = "lmsspp-bottom-row";
  statsWrap.className = "lmsspp-stats";
  bottomWrap.appendChild(controlsWrap);
  bottomRow.appendChild(statsWrap);
  bottomRow.appendChild(metricsWrap);
  bottomWrap.appendChild(bottomRow);
  root.appendChild(sceneWrap);
  root.appendChild(bottomWrap);

  const controlRefs = Object.create(null);
  let noticeTimer = null;
  let metricsReady = false;
  let sceneReady = false;
  let playTimer = null;
  let playToken = 0;
  const sceneFigure = JSON.parse(JSON.stringify(payload.scene_figure_template || {{data: [], layout: {{}}}}));

  function displayModes() {{
    if (Array.isArray(payload.mode_order) && payload.mode_order.length) return payload.mode_order.slice();
    const fallback = payload.ui_state && payload.ui_state.display_mode_default;
    return fallback ? [fallback] : [];
  }}

  const state = {{
    frameName: payload.ui_state && payload.ui_state.frame_name_default === "body" ? "body" : "lab",
    showPaths: !payload.ui_state || payload.ui_state.show_paths !== false,
    showVectors: !payload.ui_state || payload.ui_state.show_vectors !== false,
    layoutMode: payload.ui_state && payload.ui_state.layout_default === "side" ? "side" : "top",
    playbackSpeed: Number((payload.ui_state && payload.ui_state.playback_speed) || 1.0) || 1.0,
    baseIntervalMs: Number((payload.ui_state && payload.ui_state.base_interval_ms) || 40) || 40,
    playDirection: 1,
    playing: false,
    displayMode: (payload.ui_state && payload.ui_state.display_mode_default) || (displayModes()[0] || null),
    frame: Number((payload.ui_state && payload.ui_state.frame_default) || 0) || 0,
  }};

  function showNotice() {{
    noticeWrap.textContent = payload.backend_notice || "This snapshot is static. This control needs a Python backend to recompute trajectories.";
    if (noticeTimer) window.clearTimeout(noticeTimer);
    noticeTimer = window.setTimeout(() => {{
      noticeWrap.textContent = "";
    }}, 2600);
  }}

  function getActiveBundle() {{
    if (payload.ensembles) {{
      const key = state.displayMode || displayModes()[0];
      const entry = payload.ensembles[key];
      return entry ? entry.bundle : null;
    }}
    return payload.bundle || null;
  }}

  function maxFrame() {{
    const bundle = getActiveBundle();
    if (!bundle || !Array.isArray(bundle.w)) return 0;
    return Math.max(0, bundle.w.length - 1);
  }}

  function clampFrame(v) {{
    const n = maxFrame();
    const x = Math.max(0, Math.min(n, Number(v) || 0));
    return Math.round(x);
  }}

  state.frame = clampFrame(state.frame);

  function formatValue(value, fmt) {{
    if (typeof value !== "number" || !isFinite(value)) return String(value);
    const match = /^\\.(\\d+)f$/.exec(String(fmt || ""));
    if (match) return value.toFixed(Number(match[1]));
    return String(value);
  }}

  function buttonClass(style, active) {{
    const parts = ["lmsspp-button"];
    if (style) parts.push("style-" + style);
    if (active) parts.push("is-active");
    return parts.join(" ");
  }}

  function modeLabel(mode) {{
    if (payload.ensembles && payload.ensembles[mode] && payload.ensembles[mode].label) return String(payload.ensembles[mode].label);
    return String(mode || "");
  }}

  function displayModeButtonStyle(mode) {{
    const tag = String(mode || "").toLowerCase();
    if (tag.includes("poisson")) return "info";
    if (tag.includes("low") || tag.includes("min")) return "success";
    return "";
  }}

  function frameButtonLabel() {{
    return state.frameName === "body" ? "Frame: Co-rotating" : "Frame: Lab";
  }}

  function applyLayoutMode() {{
    root.className = "lmsspp-root " + (state.layoutMode === "side" ? "layout-side" : "layout-top");
  }}

  function getStatsHtml() {{
    const frame = clampFrame(state.frame);
    if (payload.stats_snapshots && payload.ensembles) {{
      const perFrameName = payload.stats_snapshots[state.frameName] || {{}};
      const perMode = perFrameName[state.displayMode] || [];
      return perMode[frame] || "";
    }}
    if (payload.stats_snapshots) {{
      const perFrameName = payload.stats_snapshots[state.frameName] || [];
      return perFrameName[frame] || "";
    }}
    return "";
  }}

  function dot(a, b) {{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }}

  function norm(v) {{
    return Math.sqrt(Math.max(0, dot(v, v)));
  }}

  function normalize(v) {{
    const n = Math.max(norm(v), 1e-12);
    return [v[0] / n, v[1] / n, v[2] / n];
  }}

  function rowTimesMatrixTranspose(row, mat) {{
    return [
      row[0] * mat[0][0] + row[1] * mat[0][1] + row[2] * mat[0][2],
      row[0] * mat[1][0] + row[1] * mat[1][1] + row[2] * mat[1][2],
      row[0] * mat[2][0] + row[1] * mat[2][1] + row[2] * mat[2][2],
    ];
  }}

  function reconstructDisplayPoints(bundle, frame, frameName) {{
    const base = bundle.base_points_display || bundle.base_points || [];
    const w = bundle.w[frame];
    const zeta = bundle.zeta[frame];
    const w2 = dot(w, w);
    const pts = new Array(base.length);
    for (let i = 0; i < base.length; i += 1) {{
      const p = base[i];
      const diff = [p[0] - w[0], p[1] - w[1], p[2] - w[2]];
      const den = Math.max(dot(diff, diff), 1e-12);
      const body = normalize([
        ((1.0 - w2) / den) * diff[0] - w[0],
        ((1.0 - w2) / den) * diff[1] - w[1],
        ((1.0 - w2) / den) * diff[2] - w[2],
      ]);
      pts[i] = frameName === "body" ? body : rowTimesMatrixTranspose(body, zeta);
    }}
    return pts;
  }}

  function xyzFromPoints(points) {{
    const x = new Array(points.length);
    const y = new Array(points.length);
    const z = new Array(points.length);
    for (let i = 0; i < points.length; i += 1) {{
      x[i] = points[i][0];
      y[i] = points[i][1];
      z[i] = points[i][2];
    }}
    return [x, y, z];
  }}

  function xyzFromSeries(series, count) {{
    const n = Math.max(0, Math.min(series.length, count));
    const x = new Array(n);
    const y = new Array(n);
    const z = new Array(n);
    for (let i = 0; i < n; i += 1) {{
      x[i] = series[i][0];
      y[i] = series[i][1];
      z[i] = series[i][2];
    }}
    return [x, y, z];
  }}

  async function renderMetrics() {{
    const fig = JSON.parse(JSON.stringify((payload.metrics_figures && payload.metrics_figures[state.frameName]) || {{data: [], layout: {{}}}}));
    if (!metricsReady) {{
      await Plotly.newPlot(metricsWrap, fig.data, fig.layout, config);
      metricsReady = true;
    }} else {{
      await Plotly.react(metricsWrap, fig.data, fig.layout, config);
    }}
  }}

  async function renderScene() {{
    const bundle = getActiveBundle();
    if (!bundle) return;
    const frame = clampFrame(state.frame);
    const roles = payload.scene_trace_roles || {{}};
    const zSeries = state.frameName === "body" ? bundle.z_body : bundle.z_lab;
    const ZSeries = state.frameName === "body" ? bundle.Z_body : bundle.Z_lab;
    const points = reconstructDisplayPoints(bundle, frame, state.frameName);
    const [px, py, pz] = xyzFromPoints(points);
    const w = bundle.w[frame];
    const z = zSeries[frame];
    const Z = ZSeries[frame];
    const kTrace = [];
    const xVals = [];
    const yVals = [];
    const zVals = [];

    function pushTrace(index, xs, ys, zs) {{
      if (typeof index !== "number") return;
      kTrace.push(index);
      xVals.push(xs);
      yVals.push(ys);
      zVals.push(zs);
    }}

    pushTrace(roles.points_marker, px, py, pz);
    pushTrace(roles.w_marker, [w[0]], [w[1]], [w[2]]);
    pushTrace(roles.z_marker, [z[0]], [z[1]], [z[2]]);
    pushTrace(roles.Z_marker, [Z[0]], [Z[1]], [Z[2]]);

    const end = frame + 1;
    const [wx, wy, wz] = xyzFromSeries(bundle.w, end);
    const [zx, zy, zz] = xyzFromSeries(zSeries, end);
    const [Zx, Zy, Zz] = xyzFromSeries(ZSeries, end);
    pushTrace(roles.w_path, wx, wy, wz);
    pushTrace(roles.z_path, zx, zy, zz);
    pushTrace(roles.Z_path, Zx, Zy, Zz);
    pushTrace(roles.w_vector, [0.0, w[0]], [0.0, w[1]], [0.0, w[2]]);
    pushTrace(roles.z_vector, [0.0, z[0]], [0.0, z[1]], [0.0, z[2]]);
    pushTrace(roles.Z_vector, [0.0, Z[0]], [0.0, Z[1]], [0.0, Z[2]]);

    if (bundle.bar_sheet) {{
      const barZSeries = state.frameName === "body" ? bundle.bar_sheet.Z_body : bundle.bar_sheet.Z_lab;
      const barzSeries = state.frameName === "body" ? bundle.bar_sheet.z_body : bundle.bar_sheet.z_lab;
      const barwSeries = bundle.bar_sheet.w;
      const bw = barwSeries[frame];
      const bz = barzSeries[frame];
      const bZ = barZSeries[frame];
      pushTrace(roles.bar_w_marker, [bw[0]], [bw[1]], [bw[2]]);
      pushTrace(roles.bar_z_marker, [bz[0]], [bz[1]], [bz[2]]);
      pushTrace(roles.bar_Z_marker, [bZ[0]], [bZ[1]], [bZ[2]]);
      const [bwx, bwy, bwz] = xyzFromSeries(barwSeries, barwSeries.length);
      const [bzx, bzy, bzz] = xyzFromSeries(barzSeries, barzSeries.length);
      const [bZx, bZy, bZz] = xyzFromSeries(barZSeries, barZSeries.length);
      pushTrace(roles.bar_w_path, bwx, bwy, bwz);
      pushTrace(roles.bar_z_path, bzx, bzy, bzz);
      pushTrace(roles.bar_Z_path, bZx, bZy, bZz);
      pushTrace(roles.bar_w_vector, [0.0, bw[0]], [0.0, bw[1]], [0.0, bw[2]]);
      pushTrace(roles.bar_z_vector, [0.0, bz[0]], [0.0, bz[1]], [0.0, bz[2]]);
      pushTrace(roles.bar_Z_vector, [0.0, bZ[0]], [0.0, bZ[1]], [0.0, bZ[2]]);
    }}

    if (!sceneReady) {{
      await Plotly.newPlot(sceneWrap, sceneFigure.data, sceneFigure.layout, config);
      sceneReady = true;
    }}
    if (kTrace.length) {{
      await Plotly.restyle(sceneWrap, {{x: xVals, y: yVals, z: zVals}}, kTrace);
    }}
    const pathRoles = [roles.w_path, roles.z_path, roles.Z_path, roles.bar_w_path, roles.bar_z_path, roles.bar_Z_path]
      .filter((x) => typeof x === "number");
    const vectorRoles = [roles.w_vector, roles.z_vector, roles.Z_vector, roles.bar_w_vector, roles.bar_z_vector, roles.bar_Z_vector]
      .filter((x) => typeof x === "number");
    if (pathRoles.length) {{
      await Plotly.restyle(sceneWrap, {{visible: pathRoles.map(() => !!state.showPaths)}}, pathRoles);
    }}
    if (vectorRoles.length) {{
      await Plotly.restyle(sceneWrap, {{visible: vectorRoles.map(() => !!state.showVectors)}}, vectorRoles);
    }}
    if (payload.display_mode_colors && roles.points_marker !== undefined && payload.display_mode_colors[state.displayMode]) {{
      await Plotly.restyle(sceneWrap, {{"marker.color": [payload.display_mode_colors[state.displayMode]]}}, [roles.points_marker]);
    }}
  }}

  function renderStats() {{
    statsWrap.innerHTML = getStatsHtml();
  }}

  function syncControls() {{
    const frameRef = controlRefs.frame_slider;
    if (frameRef) {{
      frameRef.input.max = String(maxFrame());
      frameRef.input.value = String(clampFrame(state.frame));
      frameRef.readout.textContent = String(clampFrame(state.frame));
    }}
    const frameNameRef = controlRefs.view_frame_dropdown;
    if (frameNameRef) frameNameRef.input.value = state.frameName;
    const layoutRef = controlRefs.layout_dropdown;
    if (layoutRef) layoutRef.input.value = state.layoutMode;
    const pathRef = controlRefs.show_paths;
    if (pathRef) pathRef.input.checked = !!state.showPaths;
    const vectorRef = controlRefs.show_vectors;
    if (vectorRef) vectorRef.input.checked = !!state.showVectors;
    const frameBtn = controlRefs.btn_toggle_frame;
    if (frameBtn) {{
      frameBtn.input.textContent = frameButtonLabel();
      frameBtn.input.className = buttonClass(state.frameName === "body" ? "info" : "", false);
    }}
    const playRef = controlRefs.play;
    if (playRef) playRef.input.textContent = state.playing ? "Pause" : (playRef.node.description || "Play");
    const stopRef = controlRefs.play_stop;
    if (stopRef) stopRef.input.textContent = "■ Stop";
    const speedHalf = controlRefs.btn_speed_half;
    if (speedHalf) speedHalf.input.textContent = "0.5x speed (" + Number(state.playbackSpeed).toFixed(1) + ")";
    const speedDouble = controlRefs.btn_speed_double;
    if (speedDouble) speedDouble.input.textContent = "2x speed (" + Number(state.playbackSpeed).toFixed(1) + ")";
    const displayRef = controlRefs.toggle_init_state;
    if (displayRef && displayModes().length > 1) {{
      displayRef.input.textContent = "Displayed: " + modeLabel(state.displayMode);
      displayRef.input.className = buttonClass(displayModeButtonStyle(state.displayMode), false);
    }}
  }}

  function cancelPlayLoop() {{
    playToken += 1;
    if (playTimer) {{
      window.clearTimeout(playTimer);
      playTimer = null;
    }}
  }}

  function setPlaying(flag) {{
    cancelPlayLoop();
    state.playing = !!flag;
    if (state.playing) {{
      const token = playToken;
      const tick = async () => {{
        if (!state.playing || token !== playToken) return;
        const next = state.frame + state.playDirection;
        if (next < 0 || next > maxFrame()) {{
          state.frame = state.playDirection > 0 ? maxFrame() : 0;
          state.playing = false;
          syncControls();
          await renderScene();
          renderStats();
          return;
        }}
        state.frame = next;
        syncControls();
        await renderScene();
        renderStats();
        if (!state.playing || token !== playToken) return;
        const interval = Math.max(5, Math.round(state.baseIntervalMs / Math.max(state.playbackSpeed, 0.125)));
        playTimer = window.setTimeout(tick, interval);
      }};
      const interval = Math.max(5, Math.round(state.baseIntervalMs / Math.max(state.playbackSpeed, 0.125)));
      playTimer = window.setTimeout(tick, interval);
    }}
    syncControls();
  }}

  function startPlay(direction) {{
    state.playDirection = direction < 0 ? -1 : 1;
    if (state.playDirection < 0 && state.frame <= 0) state.frame = maxFrame();
    if (state.playDirection > 0 && state.frame >= maxFrame()) state.frame = 0;
    setPlaying(true);
  }}

  async function applyStateChange(kind, value) {{
    if (kind === "frame_set") {{
      state.frame = clampFrame(value);
      syncControls();
      await renderScene();
      renderStats();
      return;
    }}
    if (kind === "frame_name_set") {{
      state.frameName = value === "body" ? "body" : "lab";
      syncControls();
      await renderMetrics();
      await renderScene();
      renderStats();
      return;
    }}
    if (kind === "frame_name_toggle") {{
      state.frameName = state.frameName === "body" ? "lab" : "body";
      syncControls();
      await renderMetrics();
      await renderScene();
      renderStats();
      return;
    }}
    if (kind === "show_paths_toggle") {{
      state.showPaths = !!value;
      syncControls();
      await renderScene();
      return;
    }}
    if (kind === "show_vectors_toggle") {{
      state.showVectors = !!value;
      syncControls();
      await renderScene();
      return;
    }}
    if (kind === "layout_set") {{
      state.layoutMode = value === "side" ? "side" : "top";
      applyLayoutMode();
      syncControls();
      return;
    }}
    if (kind === "display_mode_cycle") {{
      const order = displayModes();
      if (!order.length) return;
      const idx = Math.max(0, order.indexOf(state.displayMode));
      state.displayMode = order[(idx + 1) % order.length];
      state.frame = clampFrame(state.frame);
      syncControls();
      await renderScene();
      renderStats();
      return;
    }}
    if (kind === "display_mode_set") {{
      state.displayMode = value;
      state.frame = clampFrame(state.frame);
      syncControls();
      await renderScene();
      renderStats();
      return;
    }}
    if (kind === "play") {{
      setPlaying(!state.playing);
      return;
    }}
    if (kind === "play_forward") {{
      startPlay(1);
      return;
    }}
    if (kind === "play_backward") {{
      startPlay(-1);
      return;
    }}
    if (kind === "pause") {{
      setPlaying(false);
      return;
    }}
    if (kind === "stop") {{
      setPlaying(false);
      state.frame = 0;
      syncControls();
      await renderScene();
      renderStats();
      return;
    }}
    if (kind === "step_forward") {{
      state.frame = clampFrame(state.frame + 1);
      syncControls();
      await renderScene();
      renderStats();
      return;
    }}
    if (kind === "step_backward") {{
      state.frame = clampFrame(state.frame - 1);
      syncControls();
      await renderScene();
      renderStats();
      return;
    }}
    if (kind === "speed_half") {{
      state.playbackSpeed = Math.max(0.125, Math.min(16.0, state.playbackSpeed * 0.5));
      if (state.playing) setPlaying(true); else syncControls();
      return;
    }}
    if (kind === "speed_double") {{
      state.playbackSpeed = Math.max(0.125, Math.min(16.0, state.playbackSpeed * 2.0));
      if (state.playing) setPlaying(true); else syncControls();
      return;
    }}
    showNotice();
  }}

  function applyNodeLayout(el, node) {{
    const layout = (node && node.layout) || {{}};
    if (layout.width) el.style.width = layout.width;
    if (layout.height) el.style.height = layout.height;
    if (layout.margin) el.style.margin = layout.margin;
    if (layout.justify_content) el.style.justifyContent = layout.justify_content;
    if (layout.align_items) el.style.alignItems = layout.align_items;
    if (layout.flex_flow) el.style.flexFlow = layout.flex_flow;
  }}

  function renderControlNode(node, parent) {{
    if (!node) return;
    if (node.node_type === "container") {{
      if (node.widget_type === "accordion") {{
        const wrap = document.createElement("div");
        wrap.className = "lmsspp-accordion";
        applyNodeLayout(wrap, node);
        const titles = Array.isArray(node.titles) ? node.titles : [];
        (node.children || []).forEach((child, idx) => {{
          const details = document.createElement("details");
          details.open = node.selected_index == null ? idx === 0 : Number(node.selected_index) === idx;
          const summary = document.createElement("summary");
          summary.textContent = titles[idx] || ("Section " + (idx + 1));
          const body = document.createElement("div");
          body.className = "lmsspp-accordion-body";
          renderControlNode(child, body);
          details.appendChild(summary);
          details.appendChild(body);
          wrap.appendChild(details);
        }});
        parent.appendChild(wrap);
        return;
      }}
      const box = document.createElement("div");
      box.className = node.widget_type === "hbox" ? "lmsspp-box-hbox" : "lmsspp-box-vbox";
      applyNodeLayout(box, node);
      (node.children || []).forEach((child) => renderControlNode(child, box));
      parent.appendChild(box);
      return;
    }}

    const leaf = document.createElement("div");
    leaf.className = "lmsspp-leaf";
    applyNodeLayout(leaf, node);
    const backendOnlyControl = (node.action_kind || "backend_notice") === "backend_notice";
    if (backendOnlyControl) leaf.classList.add("lmsspp-backend-fixed");

    if (node.widget_type === "html") {{
      leaf.classList.add("lmsspp-html");
      leaf.innerHTML = node.value || "";
      parent.appendChild(leaf);
      return;
    }}

    const ref = {{node, input: null, readout: null}};
    const controlId = node.control_id || null;
    if (controlId) controlRefs[controlId] = ref;

    function backendOnly(restore) {{
      return async (event) => {{
        if (restore) restore();
        showNotice();
      }};
    }}

    if (node.widget_type === "play") {{
      const row = document.createElement("div");
      row.className = "lmsspp-button-row";
      const playBtn = document.createElement("button");
      playBtn.type = "button";
      playBtn.className = buttonClass(node.button_style || "", false);
      playBtn.textContent = node.description || "Play";
      const stopBtn = document.createElement("button");
      stopBtn.type = "button";
      stopBtn.className = buttonClass("", false);
      stopBtn.textContent = "■ Stop";
      ref.input = playBtn;
      controlRefs.play_stop = {{node, input: stopBtn, readout: null}};
      playBtn.addEventListener("click", async () => {{
        await applyStateChange(node.action_kind || "play", node.value);
      }});
      stopBtn.addEventListener("click", async () => {{
        await applyStateChange("stop", null);
      }});
      row.appendChild(playBtn);
      row.appendChild(stopBtn);
      leaf.appendChild(row);
      parent.appendChild(leaf);
      return;
    }}

    if (node.widget_type === "button" || node.widget_type === "togglebutton") {{
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = buttonClass(node.button_style || "", node.widget_type === "togglebutton" && !!node.value);
      btn.textContent = node.description || "";
      if (node.disabled) btn.disabled = true;
      ref.input = btn;
      if ((node.action_kind || "backend_notice") === "backend_notice") {{
        btn.addEventListener("click", backendOnly(null));
      }} else {{
        btn.addEventListener("click", async () => {{
          await applyStateChange(node.action_kind, node.value);
        }});
      }}
      leaf.appendChild(btn);
      parent.appendChild(leaf);
      return;
    }}

    if (node.widget_type === "dropdown") {{
      const row = document.createElement("div");
      row.className = "lmsspp-select-row";
      const label = document.createElement("label");
      label.textContent = node.description || "";
      const select = document.createElement("select");
      select.className = "lmsspp-select";
      (node.options || []).forEach((opt) => {{
        const option = document.createElement("option");
        option.value = String(opt.value);
        option.textContent = opt.label;
        if (String(opt.value) === String(node.value)) option.selected = true;
        select.appendChild(option);
      }});
      ref.input = select;
      if (node.disabled) select.disabled = true;
      if ((node.action_kind || "backend_notice") === "backend_notice") {{
        select.addEventListener("change", backendOnly(() => {{ select.value = String(node.value); }}));
      }} else {{
        select.addEventListener("change", async () => {{
          await applyStateChange(node.action_kind, select.value);
        }});
      }}
      row.appendChild(label);
      row.appendChild(select);
      leaf.appendChild(row);
      parent.appendChild(leaf);
      return;
    }}

    if (node.widget_type === "checkbox") {{
      const row = document.createElement("label");
      row.className = "lmsspp-checkbox-row";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = !!node.value;
      if (node.disabled) input.disabled = true;
      const text = document.createElement("span");
      text.textContent = node.description || "";
      ref.input = input;
      if ((node.action_kind || "backend_notice") === "backend_notice") {{
        input.addEventListener("change", backendOnly(() => {{ input.checked = !!node.value; }}));
      }} else {{
        input.addEventListener("change", async () => {{
          await applyStateChange(node.action_kind, input.checked);
        }});
      }}
      row.appendChild(input);
      row.appendChild(text);
      leaf.appendChild(row);
      parent.appendChild(leaf);
      return;
    }}

    if (node.widget_type === "intslider" || node.widget_type === "floatslider") {{
      const row = document.createElement("div");
      row.className = "lmsspp-slider";
      const label = document.createElement("label");
      label.textContent = node.description || "";
      const input = document.createElement("input");
      input.type = "range";
      input.className = "lmsspp-range";
      input.min = String(node.min);
      input.max = String(node.max);
      input.step = String(node.step);
      input.value = String(node.value);
      const readout = document.createElement("span");
      readout.className = "lmsspp-readout";
      readout.textContent = formatValue(Number(node.value), node.readout_format);
      ref.input = input;
      ref.readout = readout;
      if (node.disabled) input.disabled = true;
      if ((node.action_kind || "backend_notice") === "backend_notice") {{
        input.addEventListener("input", backendOnly(() => {{
          input.value = String(node.value);
          readout.textContent = formatValue(Number(node.value), node.readout_format);
        }}));
      }} else {{
        input.addEventListener("input", async () => {{
          const numeric = node.widget_type === "intslider" ? Math.round(Number(input.value)) : Number(input.value);
          readout.textContent = formatValue(numeric, node.readout_format);
          await applyStateChange(node.action_kind, numeric);
        }});
      }}
      row.appendChild(label);
      row.appendChild(input);
      row.appendChild(readout);
      leaf.appendChild(row);
      parent.appendChild(leaf);
      return;
    }}
  }}

  renderControlNode(payload.controls_manifest, controlsWrap);
  applyLayoutMode();

  async function init() {{
    syncControls();
    await renderMetrics();
    await renderScene();
    renderStats();
  }}

  init().catch((err) => {{
    noticeWrap.textContent = "snapshot initialization failed: " + String(err);
  }});
}})();
</script>
</body>
</html>
"""


def write_lms_widget_snapshot_html(out_path: str | Path, *, payload: dict[str, Any]) -> Path:
    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload_json = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    plotly_js = get_plotlyjs()
    html = _snapshot_clone_html(payload_json, plotly_js, title=str(payload.get("title", "LMS widget snapshot")))
    out.write_text(html, encoding="utf-8")
    return out


__all__ = ["write_lms_widget_snapshot_html"]
