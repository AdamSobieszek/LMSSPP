"""3D Plotly widget for reduced LMS dynamics on S^2 / B^3."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Callable, Literal

import numpy as np
import torch

try:
    import ipywidgets as widgets
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as exc:  # pragma: no cover
    raise ImportError("lms_ball3d_widget requires plotly and ipywidgets.") from exc

try:
    from .LMS import (
        integrate_lms_reduced_euler,
        mobius_sphere,
        normalize,
        random_points_on_sphere,
        skew_symmetric_from_axis,
    )
except Exception:
    from LMS import (  # type: ignore
        integrate_lms_reduced_euler,
        mobius_sphere,
        normalize,
        random_points_on_sphere,
        skew_symmetric_from_axis,
    )


@dataclass(frozen=True)
class LMS3DControlSpec:
    key: str
    label: str
    min: float
    max: float
    step: float
    value: float
    integer: bool = False
    readout_format: str = ".2f"
    continuous_update: bool = False


DEFAULT_CONTROLS = (
    LMS3DControlSpec("r0", "Start radius |w|", 1e-4, 0.9999, 1e-4, 0.03, readout_format=".4f"),
    LMS3DControlSpec("N", "Oscillators N", 8, 300, 1, 150, integer=True, readout_format=".0f"),
    LMS3DControlSpec("K", "Coupling K", 0.0, 5.0, 0.02, 1.0, readout_format=".2f"),
    LMS3DControlSpec("omega", "Rotation rate omega", -10.0, 10.0, 0.02, 3.0, readout_format=".2f"),
    LMS3DControlSpec("dt", "Time step dt", 1e-4, 8e-2, 1e-4, 5e-2, readout_format=".4f"),
    LMS3DControlSpec("w_az", "w direction azimuth", -math.pi, math.pi, 0.02, 0.2, readout_format=".2f"),
    LMS3DControlSpec("w_el", "w direction elevation", -0.5 * math.pi, 0.5 * math.pi, 0.02, 0.25, readout_format=".2f"),
    LMS3DControlSpec("ax_az", "axis azimuth", -math.pi, math.pi, 0.02, 0.0, readout_format=".2f"),
    LMS3DControlSpec("ax_el", "axis elevation", -0.5 * math.pi, 0.5 * math.pi, 0.02, 0.5 * math.pi, readout_format=".2f"),
    LMS3DControlSpec("steps", "Integration steps", 200, 2000, 1, 400, integer=True, readout_format=".0f"),
)


HYDRO_DEFAULT_CONTROLS = (
    LMS3DControlSpec("r0", "Start radius |w|", 1e-4, 0.9999, 1e-4, 0.03, readout_format=".4f"),
    LMS3DControlSpec("N", "Oscillators N", 100, 12000, 50, 1800, integer=True, readout_format=".0f"),
    LMS3DControlSpec("K", "Coupling K", 0.0, 5.0, 0.02, 1.0, readout_format=".2f"),
    LMS3DControlSpec("omega", "Rotation rate omega", -10.0, 10.0, 0.02, 3.0, readout_format=".2f"),
    LMS3DControlSpec("dt", "Time step dt", 1e-4, 8e-2, 1e-4, 5e-2, readout_format=".4f"),
    LMS3DControlSpec("w_az", "w direction azimuth", -math.pi, math.pi, 0.02, 0.2, readout_format=".2f"),
    LMS3DControlSpec("w_el", "w direction elevation", -0.5 * math.pi, 0.5 * math.pi, 0.02, 0.25, readout_format=".2f"),
    LMS3DControlSpec("ax_az", "axis azimuth", -math.pi, math.pi, 0.02, 0.0, readout_format=".2f"),
    LMS3DControlSpec("ax_el", "axis elevation", -0.5 * math.pi, 0.5 * math.pi, 0.02, 0.5 * math.pi, readout_format=".2f"),
    LMS3DControlSpec("steps", "Integration steps", 120, 1800, 1, 420, integer=True, readout_format=".0f"),
)


InitMode = Literal[
    "entropy_high",
    "entropy_low",
    "var_perp_high",
    "var_perp_low",
    "poisson",
    "varmax_parallel_perp",
    "varmin_parallel_perp",
]
OptimizedInitMode = Literal[
    "entropy_high",
    "entropy_low",
    "var_perp_high",
    "var_perp_low",
    "varmax_parallel_perp",
    "varmin_parallel_perp",
]
ActiveInitMode = Literal["entropy_high", "entropy_low", "var_perp_high", "var_perp_low", "poisson"]
InitMetricMode = Literal["entropy", "perp_variance"]


class _HydroRecomputeCancelled(Exception):
    """Internal cooperative-cancellation signal for hydro recompute jobs."""


def _sphere_wireframe_traces(n_lat: int = 9, n_lon: int = 18) -> list[go.Scatter3d]:
    traces = []
    lat_vals = np.linspace(-0.8 * np.pi / 2, 0.8 * np.pi / 2, n_lat)
    lon = np.linspace(0, 2 * np.pi, 260)
    for phi in lat_vals:
        x = np.cos(phi) * np.cos(lon)
        y = np.cos(phi) * np.sin(lon)
        z = np.sin(phi) * np.ones_like(lon)
        traces.append(
            go.Scatter3d(
                x=x.tolist(),
                y=y.tolist(),
                z=z.tolist(),
                mode="lines",
                line=dict(color="lightgrey", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    lon_vals = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    lat = np.linspace(-np.pi / 2, np.pi / 2, 260)
    for lam in lon_vals:
        x = np.cos(lat) * np.cos(lam)
        y = np.cos(lat) * np.sin(lam)
        z = np.sin(lat)
        traces.append(
            go.Scatter3d(
                x=x.tolist(),
                y=y.tolist(),
                z=z.tolist(),
                mode="lines",
                line=dict(color="lightgrey", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return traces


def _angles_to_unit(az: float, el: float) -> np.ndarray:
    c = math.cos(el)
    return np.array([c * math.cos(az), c * math.sin(az), math.sin(el)], dtype=np.float64)


class LMSBall3DWidget:
    """Interactive reduced LMS widget for the 2D-sphere case (S^2 in R^3)."""

    def __init__(
        self,
        *,
        controls: tuple[LMS3DControlSpec, ...] = DEFAULT_CONTROLS,
        w_mode: str = "autograd",
        rng_seed: int = 0,
        point_size: int = 5,
        title: str = "Reduced LMS dynamics on S² (points) / B³ (order parameters)",
        trajectory_mode: Literal["memory", "fps", "auto"] = "memory",
        thermo_mode: Literal["recompute", "approx", "exact"] = "recompute",
        init_metric_mode: InitMetricMode = "entropy",
        display_points_cap: int | None = None,
        memory_budget_mb: int = 512,
    ) -> None:
        self.control_specs = controls
        self.w_mode = w_mode
        self.rng_seed = int(rng_seed)
        self.point_size = int(point_size)
        self.title = self._sanitize_plot_text(title)
        self.trajectory_mode = trajectory_mode
        self.thermo_mode = thermo_mode
        self.init_metric_mode = self._canonical_init_metric_mode(init_metric_mode)
        self.display_points_cap = (
            int(display_points_cap) if display_points_cap is not None and int(display_points_cap) > 0 else None
        )
        self.memory_budget_mb = int(memory_budget_mb)
        if self.trajectory_mode not in {"memory", "fps", "auto"}:
            raise ValueError("trajectory_mode must be 'memory', 'fps', or 'auto'.")
        if self.thermo_mode not in {"recompute", "approx", "exact"}:
            raise ValueError("thermo_mode must be 'recompute', 'approx', or 'exact'.")

        self._updating = False
        self._torch_gen = torch.Generator().manual_seed(self.rng_seed)
        self._controls: dict[str, widgets.Widget] = {}
        self._wire_count = 0
        self._camera_cache: dict[str, Any] | None = None
        self._default_camera_eye: dict[str, float] = {"x": 1.25, "y": 1.25, "z": 1.25}
        self._in_frame_update = False
        self._paused_for_drag = False
        self._was_playing_before_drag = False
        self._ignore_camera_pause_until = 0.0
        self._traj_cache: dict[str, np.ndarray] = {}
        self._metric_cache: dict[str, np.ndarray] = {}
        self._params_cache: dict[str, float | int] = {}
        self._last_overlay_frame = -10**9
        self._last_path_frame = -10**9
        self._steps = 0
        self._resolved_trajectory_mode: Literal["memory", "fps"] = "memory"
        self._base_points_np: np.ndarray | None = None
        self._display_indices: np.ndarray | None = None
        self._base_interval_ms = 40
        self._playback_speed = 1.0
        self._init_state_mode: InitMode = self._active_mode_order()[0]
        # Inversion mapping remains available in code, but is not user-exposed in widget controls.
        self._inversion_enabled = False

        self._build_controls()
        self._build_figures()
        self._bind_events()
        self._recompute(reset_frame=True)

        self.bottom_panel = widgets.VBox(
            [
                self.controls_box,
                widgets.HBox(
                    [
                        widgets.VBox([self.stats_html], layout=widgets.Layout(width="230px")),
                        self.metrics_fig,
                    ],
                    layout=widgets.Layout(align_items="flex-start"),
                ),
            ],
            layout=widgets.Layout(width="980px"),
        )
        self.layout = widgets.Box()
        self._apply_root_layout()

    def _build_controls(self) -> None:
        sliders = []
        slider_style = {"description_width": "160px"}
        control_width = "960px"
        self._control_width = control_width
        for spec in self.control_specs:
            if spec.integer:
                w = widgets.IntSlider(
                    value=int(spec.value),
                    min=int(spec.min),
                    max=int(spec.max),
                    step=int(spec.step),
                    description=spec.label,
                    continuous_update=spec.continuous_update,
                    layout=widgets.Layout(width="100%"),
                    style=slider_style,
                )
            else:
                w = widgets.FloatSlider(
                    value=float(spec.value),
                    min=float(spec.min),
                    max=float(spec.max),
                    step=float(spec.step),
                    description=spec.label,
                    readout_format=spec.readout_format,
                    continuous_update=spec.continuous_update,
                    layout=widgets.Layout(width="100%"),
                    style=slider_style,
                )
            self._controls[spec.key] = w
            marker = self._make_slider_reference_marker(spec)
            if marker is None:
                sliders.append(w)
            else:
                sliders.append(
                    widgets.VBox(
                        [w, marker],
                        layout=widgets.Layout(width="100%", margin="0"),
                    )
                )

        mode_row_width = "430px"
        layout_row_width = "510px"
        self.mode_dropdown = widgets.Dropdown(
            options=["explicit", "autograd"],
            value=self.w_mode,
            description="Computational backend",
            layout=widgets.Layout(width=mode_row_width),
            style={"description_width": "initial"},
        )
        self.layout_dropdown = widgets.Dropdown(
            options=[
                ("Top (3D over controls)", "top"),
                ("Side-by-side", "side"),
            ],
            value="top",
            description="Layout",
            layout=widgets.Layout(width=layout_row_width),
            style={"description_width": "initial"},
        )
        self.view_frame_dropdown = widgets.Dropdown(
            options=[
                ("Lab frame (zeta applied)", "lab"),
                ("Co-rotating frame (w-parameter view)", "body"),
            ],
            value="lab",
            description="Viewing frame",
            layout=widgets.Layout(width="360px"),
            style={"description_width": "initial"},
        )
        self.layout_top_view = widgets.Checkbox(
            value=True,
            description="Large top view layout",
            indent=False,
            layout=widgets.Layout(width="220px"),
        )
        self.show_paths = widgets.Checkbox(value=True, description="show paths", indent=False)
        self.show_vectors = widgets.Checkbox(value=True, description="show vectors", indent=False)

        self.play = widgets.Play(
            value=0,
            min=0,
            max=int(self._controls["steps"].value),
            step=1,
            interval=self._base_interval_ms,
            description="Play",
            show_repeat=False,
            layout=widgets.Layout(width="180px"),
        )
        self.frame_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=int(self._controls["steps"].value),
            step=1,
            description="Frame",
            continuous_update=False,
            layout=widgets.Layout(width=control_width),
            style={"description_width": "initial"},
        )
        widgets.jslink((self.play, "value"), (self.frame_slider, "value"))

        self.btn_play_forward = widgets.Button(description="Forward ▶")
        self.btn_play_backward = widgets.Button(description="◀ Backward")
        self.toggle_entropy = widgets.ToggleButton(value=False, description="Entropy Direction: Dissipate")
        self.toggle_time_direction = widgets.ToggleButton(value=False, description="Time Direction: Forward")
        self.btn_recompute = widgets.Button(description="Recompute")
        self.toggle_init_state = widgets.Button(description="Initial State: High Entropy")
        self.btn_toggle_frame = widgets.Button(description="Frame: Lab")
        self.btn_speed_half = widgets.Button(description="0.5x speed (1.0)")
        self.btn_speed_double = widgets.Button(description="2x speed (1.0)")

        self.stats_html = widgets.HTML("")
        button_col_layout = widgets.Layout(width="180px")
        for btn in (
            self.btn_play_forward,
            self.btn_play_backward,
            self.toggle_entropy,
            self.toggle_time_direction,
            self.btn_recompute,
            self.toggle_init_state,
            self.btn_toggle_frame,
            self.btn_speed_half,
            self.btn_speed_double,
        ):
            btn.layout = button_col_layout
        self.play.layout = button_col_layout

        button_grid = widgets.HBox(
            [
                widgets.VBox([self.btn_play_forward, self.play]),
                widgets.VBox([self.btn_play_backward, self.btn_recompute]),
                widgets.VBox([self.toggle_time_direction, self.toggle_entropy]),
                widgets.VBox([self.toggle_init_state, self.btn_toggle_frame]),
                widgets.VBox([self.btn_speed_half, self.btn_speed_double]),
            ],
            layout=widgets.Layout(align_items="flex-start", width=control_width),
        )
        params_box = widgets.VBox(sliders, layout=widgets.Layout(width=control_width))
        params_accordion = widgets.Accordion(
            children=[params_box],
            titles=("Parameters",),
            layout=widgets.Layout(width=control_width),
        )
        self.controls_box = widgets.VBox(
            [
                widgets.HTML("<b>LMS B³/S² widget controls</b>"),
                button_grid,
                self.frame_slider,
                widgets.HBox(
                    [self.mode_dropdown, self.layout_dropdown],
                    layout=widgets.Layout(width=control_width, justify_content="space-between"),
                ),
                widgets.HBox([self.show_paths, self.show_vectors]),
                params_accordion,
            ],
            layout=widgets.Layout(width=control_width, align_items="flex-start"),
        )
        self._sync_speed_button_labels()
        self._sync_entropy_button_label()
        self._sync_time_direction_button_label()
        self._sync_init_state_button_label()

    @staticmethod
    def _make_slider_reference_marker(spec: LMS3DControlSpec) -> widgets.HTML | None:
        markers: list[tuple[float, str]] = []
        min_v = float(spec.min)
        max_v = float(spec.max)
        if max_v <= min_v:
            return None

        if min_v < 0.0 < max_v:
            markers.append((0.0, "0"))

        if spec.key == "K" and min_v <= 1.0 <= max_v:
            markers.append((1.0, "1"))

        if not markers:
            return None

        line_bits: list[str] = []
        for value, label in markers:
            frac = (float(value) - min_v) / (max_v - min_v)
            frac = float(np.clip(frac, 0.0, 1.0))
            left = 100.0 * frac
            line_bits.append(
                (
                    f"<div style='position:absolute;left:{left:.2f}%;top:0;bottom:0;"
                    "border-left:1px dashed #8a8a8a;'></div>"
                )
            )
            line_bits.append(
                (
                    f"<div style='position:absolute;left:{left:.2f}%;top:-1px;"
                    "transform:translateX(-50%);font-size:10px;color:#666;'>"
                    f"{label}</div>"
                )
            )

        html = (
            "<div style='position:relative;height:8px;margin-top:-11px;margin-bottom:0;"
            "margin-left:166px;margin-right:70px;pointer-events:none;overflow:visible;'>"
            + "".join(line_bits)
            + "</div>"
        )
        return widgets.HTML(value=html)

    def _build_figures(self) -> None:
        self.sphere_fig = go.FigureWidget()
        for tr in _sphere_wireframe_traces():
            self.sphere_fig.add_trace(tr)
        self._wire_count = len(self.sphere_fig.data)

        # Dynamic traces:
        # points, w, z, Zhat, w path, z path, Zhat path, vec w, vec z, vec Zhat
        self.sphere_fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="markers",
                marker=dict(size=self.point_size, color="royalblue"),
                name="xᵢ(t)",
                showlegend=True,
            )
        )
        self.sphere_fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="markers",
                marker=dict(size=self.point_size + 2, color="black", symbol="x"),
                name="w(t)",
                showlegend=True,
            )
        )
        self.sphere_fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="markers",
                marker=dict(size=self.point_size + 1, color="white", line=dict(color="black", width=2)),
                name="z(t)",
                showlegend=True,
            )
        )
        self.sphere_fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="markers",
                marker=dict(size=self.point_size + 1, color="firebrick"),
                name="Z/K (inside B³)",
                showlegend=True,
            )
        )
        self.sphere_fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="lines",
                line=dict(color="black", width=2, dash="dot"),
                name="w path",
                showlegend=False,
            )
        )
        self.sphere_fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="lines",
                line=dict(color="gray", width=2, dash="dot"),
                name="z path",
                showlegend=False,
            )
        )
        self.sphere_fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="lines",
                line=dict(color="firebrick", width=2, dash="dot"),
                name="Z/K path",
                showlegend=False,
            )
        )
        self.sphere_fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="lines",
                line=dict(color="black", width=3),
                name="w vector",
                showlegend=False,
            )
        )
        self.sphere_fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="lines",
                line=dict(color="gray", width=3),
                name="z vector",
                showlegend=False,
            )
        )
        self.sphere_fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="lines",
                line=dict(color="firebrick", width=3),
                name="Z/K vector",
                showlegend=False,
            )
        )

        self.sphere_fig.update_layout(
            title=self.title,
            width=760,
            height=760,
            margin=dict(l=20, r=20, t=48, b=20),
            showlegend=True,
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.75)"),
            uirevision="lms-ball3d",
            scene=dict(
                aspectmode="cube",
                dragmode="orbit",
                uirevision="lms-ball3d-scene",
                camera=dict(eye=dict(self._default_camera_eye)),
                xaxis=dict(visible=False, autorange=False),
                yaxis=dict(visible=False, autorange=False),
                zaxis=dict(visible=False, autorange=False),
            ),
        )
        self._camera_cache = self._camera_to_json()
        self._apply_projection_visual_mode()

        self.metrics_fig = go.FigureWidget(
            make_subplots(
                rows=3,
                cols=1,
                subplot_titles=(
                    "Magnitudes: |Z|/K and |w|",
                    "z order parameter vector projected onto x,y,z axes",
                    f"Thermodynamics: {self._primary_metric_title().lower()} and variance diagnostics",
                ),
                vertical_spacing=0.12,
            )
        )
        # Row 1: |Z|/K and |w| on the same subplot.
        self.metrics_fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color="firebrick", width=2),
                name="|Z|/K",
                legend="legend",
            ),
            row=1,
            col=1,
        )
        self.metrics_fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color="black", width=2),
                name="|w|",
                legend="legend",
            ),
            row=1,
            col=1,
        )
        self.metrics_fig.add_trace(
            go.Scatter(x=[], y=[], mode="markers", marker=dict(size=9, color="red"), showlegend=False),
            row=1,
            col=1,
        )
        self.metrics_fig.add_trace(
            go.Scatter(x=[], y=[], mode="markers", marker=dict(size=9, color="black"), showlegend=False),
            row=1,
            col=1,
        )
        # Row 2: x/y/z phase variables (z components in lab frame).
        self.metrics_fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color="royalblue", width=2),
                name="phase x",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        self.metrics_fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color="seagreen", width=2),
                name="phase y",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        self.metrics_fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color="darkorange", width=2),
                name="phase z",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        self.metrics_fig.add_trace(
            go.Scatter(x=[], y=[], mode="markers", marker=dict(size=8, color="royalblue"), showlegend=False),
            row=2,
            col=1,
        )
        self.metrics_fig.add_trace(
            go.Scatter(x=[], y=[], mode="markers", marker=dict(size=8, color="seagreen"), showlegend=False),
            row=2,
            col=1,
        )
        self.metrics_fig.add_trace(
            go.Scatter(x=[], y=[], mode="markers", marker=dict(size=8, color="darkorange"), showlegend=False),
            row=2,
            col=1,
        )

        # Row 3: entropy and variance diagnostics.
        self.metrics_fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color="purple", width=2),
                name=self._primary_metric_series_name(),
                legend="legend2",
            ),
            row=3,
            col=1,
        )
        self.metrics_fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color="orange", width=2),
                name="variance to center",
                legend="legend2",
            ),
            row=3,
            col=1,
        )
        self.metrics_fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color="darkorange", width=2, dash="dash"),
                name="variance to conformal center",
                legend="legend2",
            ),
            row=3,
            col=1,
        )
        self.metrics_fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color="steelblue", width=2),
                name="||mean(x)|| empirical",
                legend="legend2",
            ),
            row=3,
            col=1,
        )
        self.metrics_fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color="firebrick", width=2, dash="dot"),
                name="||Z||/K conformal",
                legend="legend2",
            ),
            row=3,
            col=1,
        )
        self.metrics_fig.add_trace(
            go.Scatter(x=[], y=[], mode="markers", marker=dict(size=7, color="purple"), showlegend=False),
            row=3,
            col=1,
        )
        self.metrics_fig.add_trace(
            go.Scatter(x=[], y=[], mode="markers", marker=dict(size=7, color="orange"), showlegend=False),
            row=3,
            col=1,
        )
        self.metrics_fig.add_trace(
            go.Scatter(x=[], y=[], mode="markers", marker=dict(size=7, color="darkorange"), showlegend=False),
            row=3,
            col=1,
        )
        self.metrics_fig.add_trace(
            go.Scatter(x=[], y=[], mode="markers", marker=dict(size=7, color="steelblue"), showlegend=False),
            row=3,
            col=1,
        )
        self.metrics_fig.add_trace(
            go.Scatter(x=[], y=[], mode="markers", marker=dict(size=7, color="firebrick"), showlegend=False),
            row=3,
            col=1,
        )
        # Keep metric cursor traces allocated but hidden to avoid expensive
        # per-frame trait sync on FigureWidget during playback.
        for i in (2, 3, 7, 8, 9, 15, 16, 17, 18, 19):
            self.metrics_fig.data[i].visible = False

        self.metrics_fig.update_layout(
            width=980,
            height=700,
            margin=dict(l=20, r=20, t=90, b=90),
            showlegend=True,
            legend=dict(
                orientation="v",
                x=0.76,
                y=0.98,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.80)",
            ),
            legend2=dict(
                orientation="v",
                x=0.76,
                y=0.32,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.80)",
            ),
            template="plotly_white",
            xaxis=dict(domain=[0.0, 0.72]),
            xaxis2=dict(domain=[0.0, 0.72]),
            xaxis3=dict(domain=[0.0, 0.72]),
        )
        self.metrics_fig.update_xaxes(title_text="", row=1, col=1)
        self.metrics_fig.update_xaxes(title_text="", row=2, col=1)
        self.metrics_fig.update_xaxes(title_text="frame", row=3, col=1)
        self.metrics_fig.update_yaxes(range=[-0.02, 1.05], row=1, col=1)
        self.metrics_fig.update_yaxes(range=[-1.05, 1.05], row=2, col=1)
        self.metrics_fig.update_yaxes(range=[-0.05, 1.20], row=3, col=1)

    def _bind_events(self) -> None:
        for w in self._controls.values():
            w.observe(self._on_control_change, names="value")
        self.mode_dropdown.observe(self._on_control_change, names="value")
        self.view_frame_dropdown.observe(self._on_visual_change, names="value")
        self.layout_top_view.observe(self._on_layout_change, names="value")
        self.layout_dropdown.observe(self._on_layout_dropdown_change, names="value")
        self.show_paths.observe(self._on_visual_change, names="value")
        self.show_vectors.observe(self._on_visual_change, names="value")
        self.toggle_time_direction.observe(self._on_time_direction_change, names="value")
        self.frame_slider.observe(self._on_frame_change, names="value")
        self.sphere_fig.layout.scene.on_change(self._on_camera_changed, "camera")
        self.sphere_fig.on_edits_completed(self._on_plot_edits_completed)
        self.btn_play_forward.on_click(self._on_play_forward_clicked)
        self.btn_play_backward.on_click(self._on_play_backward_clicked)
        self.toggle_entropy.observe(self._on_entropy_direction_change, names="value")
        self.toggle_init_state.on_click(self._on_init_state_clicked)
        self.btn_toggle_frame.on_click(self._on_toggle_frame_clicked)
        self.btn_speed_half.on_click(self._on_speed_half_clicked)
        self.btn_speed_double.on_click(self._on_speed_double_clicked)
        self.btn_recompute.on_click(self._on_recompute_clicked)

    def _params(self) -> dict[str, float | int]:
        params: dict[str, float | int] = {}
        for spec in self.control_specs:
            v = self._controls[spec.key].value
            params[spec.key] = int(v) if spec.integer else float(v)
        return params

    def _apply_root_layout(self) -> None:
        """Switch between side-by-side and stacked layouts."""
        old_height = float(self.sphere_fig.layout.height or 760)
        old_plot_height = self._drawable_plot_height(old_height)
        cam = self._ensure_camera_eye(self._camera_to_json() or self._camera_cache)
        metrics_w = int(self.metrics_fig.layout.width or 740)
        stats_w = 230
        panel_w = max(980, metrics_w + stats_w + 20)
        if bool(self.layout_top_view.value):
            # Top row: larger 3D view. Bottom row: controls + stats/plots.
            self.sphere_fig.update_layout(width=1200, height=600)
            layout_w = max(1220, panel_w)
            self.bottom_panel.layout.width = f"{layout_w}px"
            self.layout.children = (self.sphere_fig, self.bottom_panel)
            self.layout.layout = widgets.Layout(
                display="flex",
                flex_flow="column",
                align_items="flex-start",
                width=f"{layout_w}px",
            )
        else:
            self.sphere_fig.update_layout(width=760, height=760)
            self.bottom_panel.layout.width = f"{panel_w}px"
            self.layout.children = (self.sphere_fig, self.bottom_panel)
            self.layout.layout = widgets.Layout(
                display="flex",
                flex_flow="row",
                align_items="flex-start",
            )

        new_height = float(self.sphere_fig.layout.height or old_height)
        new_plot_height = self._drawable_plot_height(new_height)
        if cam is not None and old_plot_height > 1e-9 and abs(new_plot_height - old_plot_height) > 1e-6:
            # Keep perceived sphere scale stable when only widget/container height changes.
            scaled_cam = self._scale_camera_eye(cam, new_plot_height / old_plot_height)
            self._ignore_camera_pause_until = time.monotonic() + 0.20
            self.sphere_fig.layout.scene.camera = scaled_cam
            self._camera_cache = scaled_cam
        self._sync_mode_button_labels()

    def _sync_mode_button_labels(self) -> None:
        if self.view_frame_dropdown.value == "body":
            self.btn_toggle_frame.description = "Frame: Co-rotating"
            self.btn_toggle_frame.button_style = "info"
        else:
            self.btn_toggle_frame.description = "Frame: Lab"
            self.btn_toggle_frame.button_style = ""

        layout_value = "top" if bool(self.layout_top_view.value) else "side"
        if self.layout_dropdown.value != layout_value:
            prev = self._updating
            self._updating = True
            self.layout_dropdown.value = layout_value
            self._updating = prev

    def _sync_time_direction_button_label(self) -> None:
        if bool(self.toggle_time_direction.value):
            self.toggle_time_direction.description = "Time Direction: Backward"
            self.toggle_time_direction.button_style = "warning"
        else:
            self.toggle_time_direction.description = "Time Direction: Forward"
            self.toggle_time_direction.button_style = "success"

    def _sync_entropy_button_label(self) -> None:
        if bool(self.toggle_entropy.value):
            self.toggle_entropy.description = "Entropy Direction: Increase"
            self.toggle_entropy.button_style = "warning"
        else:
            self.toggle_entropy.description = "Entropy Direction: Dissipate"
            self.toggle_entropy.button_style = "success"

    @staticmethod
    def _canonical_init_metric_mode(mode: str) -> InitMetricMode:
        v = str(mode).strip().lower()
        if v in {"perp_variance", "variance", "var", "var_perp", "perp"}:
            return "perp_variance"
        return "entropy"

    @staticmethod
    def _active_modes_for_init_metric_mode(mode: str) -> tuple[ActiveInitMode, ...]:
        canonical = LMSBall3DWidget._canonical_init_metric_mode(mode)
        if canonical == "perp_variance":
            return ("var_perp_high", "var_perp_low", "poisson")
        return ("entropy_high", "entropy_low", "poisson")

    def _active_mode_order(self) -> tuple[ActiveInitMode, ...]:
        return self._active_modes_for_init_metric_mode(self.init_metric_mode)

    def _coerce_mode_to_active_family(self, mode: str) -> InitMode:
        canonical = self._canonical_init_mode(mode)
        active = self._active_mode_order()
        if canonical in {"entropy_high", "var_perp_high"}:
            return active[0]
        if canonical in {"entropy_low", "var_perp_low"}:
            return active[1]
        return canonical

    def _primary_metric_title(self) -> str:
        if self.init_metric_mode == "perp_variance":
            return "Perpendicular-variance proxy"
        return "Entropy kernel proxy"

    def _primary_metric_series_name(self) -> str:
        if self.init_metric_mode == "perp_variance":
            return "perp-variance proxy"
        return "entropy proxy (kernel)"

    def _primary_rate_series_name(self) -> str:
        if self.init_metric_mode == "perp_variance":
            return "dV_perp/dt"
        return "dH/dt"

    @staticmethod
    def _canonical_init_mode(mode: str) -> InitMode:
        aliases = {
            "high": "entropy_high",
            "low": "entropy_low",
            "entropy_high": "entropy_high",
            "entropy_low": "entropy_low",
            "var_perp_high": "var_perp_high",
            "var_perp_low": "var_perp_low",
            "high_variance": "var_perp_high",
            "low_variance": "var_perp_low",
            "var_high": "var_perp_high",
            "var_low": "var_perp_low",
            "poisson": "poisson",
            "varmax_parallel_perp": "varmax_parallel_perp",
            "varmin_parallel_perp": "varmin_parallel_perp",
            "varmax": "varmax_parallel_perp",
            "varmin": "varmin_parallel_perp",
        }
        return aliases.get(str(mode), "entropy_high")  # type: ignore[return-value]

    @staticmethod
    def _init_mode_label(mode: str) -> str:
        labels = {
            "entropy_high": "High Entropy",
            "entropy_low": "Low Entropy",
            "var_perp_high": "High Perpendicular Variance",
            "var_perp_low": "Low Perpendicular Variance",
            "poisson": "Poisson",
            "varmax_parallel_perp": "Parallel/Perpendicular Max Variance",
            "varmin_parallel_perp": "Parallel/Perpendicular Min Variance",
        }
        return labels.get(str(mode), str(mode))

    @staticmethod
    def _init_mode_short_tag(mode: str) -> str:
        tags = {
            "entropy_high": "high",
            "entropy_low": "low",
            "var_perp_high": "high-var",
            "var_perp_low": "low-var",
            "poisson": "poisson",
            "varmax_parallel_perp": "varmax",
            "varmin_parallel_perp": "varmin",
        }
        return tags.get(str(mode), str(mode))

    def _sync_init_state_button_label(self) -> None:
        mode = self._coerce_mode_to_active_family(self._init_state_mode)
        self._init_state_mode = mode
        if mode == "entropy_low":
            self.toggle_init_state.description = "Initial State: Low Entropy"
            self.toggle_init_state.button_style = "success"
        elif mode == "poisson":
            self.toggle_init_state.description = "Initial State: Poisson"
            self.toggle_init_state.button_style = "info"
        elif mode == "var_perp_high":
            self.toggle_init_state.description = "Initial State: High Perp Variance"
            self.toggle_init_state.button_style = ""
        elif mode == "var_perp_low":
            self.toggle_init_state.description = "Initial State: Low Perp Variance"
            self.toggle_init_state.button_style = "success"
        elif mode == "varmax_parallel_perp":
            self.toggle_init_state.description = "Initial State: Parallel/Perp Max Variance"
            self.toggle_init_state.button_style = ""
        elif mode == "varmin_parallel_perp":
            self.toggle_init_state.description = "Initial State: Parallel/Perp Min Variance"
            self.toggle_init_state.button_style = "success"
        else:
            self.toggle_init_state.description = "Initial State: High Entropy"
            self.toggle_init_state.button_style = ""

    @staticmethod
    def _fmt_speed(speed: float) -> str:
        return f"{speed:.1f}"

    def _sync_speed_button_labels(self) -> None:
        s = self._fmt_speed(self._playback_speed)
        self.btn_speed_half.description = f"0.5x speed ({s})"
        self.btn_speed_double.description = f"2x speed ({s})"

    def _set_playback_speed(self, speed: float) -> None:
        self._playback_speed = float(min(16.0, max(0.125, speed)))
        interval = int(max(5, round(self._base_interval_ms / self._playback_speed)))
        self.play.interval = interval
        self._sync_speed_button_labels()

    def _is_playing(self) -> bool:
        """Compatibility wrapper across ipywidgets versions.

        Some environments (notably Colab + older widget stacks) do not expose
        `Play.playing`. In that case we keep a local fallback flag.
        """
        attr = getattr(self.play, "playing", None)
        if attr is None:
            return bool(getattr(self, "_play_running_fallback", False))
        try:
            return bool(attr)
        except Exception:
            return bool(getattr(self, "_play_running_fallback", False))

    def _set_playing(self, value: bool) -> None:
        self._play_running_fallback = bool(value)
        if hasattr(self.play, "playing"):
            try:
                self.play.playing = bool(value)
            except Exception:
                pass

    def _time_direction_sign(self, *, time_backward: bool | None = None) -> float:
        backward = bool(self.toggle_time_direction.value) if time_backward is None else bool(time_backward)
        return -1.0 if backward else 1.0

    def _effective_dt(self, params: dict[str, float | int], *, time_backward: bool | None = None) -> float:
        dt_mag = abs(float(params["dt"]))
        if dt_mag < 1e-12:
            dt_mag = 1e-12
        return self._time_direction_sign(time_backward=time_backward) * dt_mag

    def _overlay_stride(self) -> int:
        if not self._is_playing():
            return 1
        return max(1, min(16, int(round(self._playback_speed))))

    def _path_stride(self) -> int:
        if not self._is_playing():
            return 1
        return max(1, min(12, int(round(self._playback_speed * 0.8))))

    @staticmethod
    def _scale_camera_eye(camera: dict[str, Any], factor: float) -> dict[str, Any]:
        if factor <= 0:
            return camera
        out = dict(camera)
        eye = dict(out.get("eye", {}))
        for axis in ("x", "y", "z"):
            if axis in eye:
                try:
                    eye[axis] = float(eye[axis]) * float(factor)
                except Exception:
                    pass
        out["eye"] = eye
        return out

    @staticmethod
    def _sanitize_plot_text(text: str) -> str:
        # FigureWidget LaTeX rendering is environment-dependent; avoid raw "$...$" fallbacks.
        out = str(text).replace("$", "")
        out = out.replace("\\zeta", "zeta")
        out = out.replace("\\omega", "omega")
        out = out.replace("\\phi", "phi")
        out = out.replace("\\theta", "theta")
        out = out.replace("S^2", "S²").replace("B^3", "B³")
        return out

    def _ensure_camera_eye(self, camera: dict[str, Any] | None) -> dict[str, Any]:
        out: dict[str, Any] = dict(camera or {})
        eye = dict(out.get("eye", {}))
        for axis, default in self._default_camera_eye.items():
            val = eye.get(axis, default)
            try:
                eye[axis] = float(val)
            except Exception:
                eye[axis] = float(default)
        out["eye"] = eye
        return out

    def _drawable_plot_height(self, fig_height: float) -> float:
        margin = self.sphere_fig.layout.margin
        top = float(getattr(margin, "t", 0) or 0)
        bottom = float(getattr(margin, "b", 0) or 0)
        return max(1.0, float(fig_height) - top - bottom)

    def _scene_radius_default(self) -> float:
        return 1.1

    def _scene_radius_inversion(self) -> float:
        return 6.0

    def _apply_scene_range(self) -> None:
        r = float(self._scene_radius_inversion() if bool(self._inversion_enabled) else self._scene_radius_default())
        self.sphere_fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False, autorange=False, range=[-r, r]),
                yaxis=dict(visible=False, autorange=False, range=[-r, r]),
                zaxis=dict(visible=False, autorange=False, range=[-r, r]),
            )
        )

    def _apply_projection_visual_mode(self) -> None:
        # Hide sphere wireframe in inverted view; trajectories are shown on the plane.
        show_wire = not bool(self._inversion_enabled)
        self._in_frame_update = True
        try:
            with self.sphere_fig.batch_update():
                for i in range(self._wire_count):
                    self.sphere_fig.data[i].visible = show_wire
                self._apply_scene_range()
        finally:
            self._in_frame_update = False

    @staticmethod
    def _safe_unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(arr))
        if n < eps:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return arr / n

    def _corotating_boundary_point(self, *, frame_name: Literal["lab", "body"]) -> np.ndarray:
        z_key = "z_body" if frame_name == "body" else "z_lab"
        z_series = self._traj_cache.get(z_key)
        if z_series is not None and len(z_series) > 0:
            z_last = np.asarray(z_series[-1], dtype=np.float64)
            z_n = float(np.linalg.norm(z_last))
            if z_n > 1e-12:
                return z_last / z_n

        w_series = self._traj_cache.get("w")
        if w_series is not None and len(w_series) > 0:
            w_last = np.asarray(w_series[-1], dtype=np.float64)
            w_n = float(np.linalg.norm(w_last))
            if w_n > 1e-12:
                return -w_last / w_n
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)

    def _inversion_context(self, *, frame_name: Literal["lab", "body"]) -> tuple[np.ndarray, float]:
        # User-requested inversion center: numerical negative of the
        # co-rotating boundary alignment point.
        corot = self._corotating_boundary_point(frame_name=frame_name)
        x0 = -corot
        cap = float(self._scene_radius_inversion())
        return x0, cap

    @staticmethod
    def _literal_inversion_rows(
        x: np.ndarray,
        *,
        x0: np.ndarray,
        cap: float,
        eps: float = 1e-6,
    ) -> np.ndarray:
        # Literal inversion requested by user:
        #   x -> (x0 - x) / ||x0 - x||^2
        arr = np.asarray(x, dtype=np.float64)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        center = np.asarray(x0, dtype=np.float64).reshape(1, 3)
        diff = center - arr
        den = np.sum(diff * diff, axis=1, keepdims=True)
        out = diff / np.maximum(den, eps)
        if cap > 0.0:
            r = np.linalg.norm(out, axis=1, keepdims=True)
            s = np.minimum(1.0, cap / np.maximum(r, 1e-12))
            out = out * s
        return out[0] if single else out

    def _maybe_invert_rows(
        self,
        x: np.ndarray,
        *,
        frame_name: Literal["lab", "body"],
        inv_ctx: tuple[np.ndarray, float] | None,
    ) -> np.ndarray:
        if inv_ctx is None:
            # Fast path: inversion disabled, keep original dtype/array and avoid copies.
            return x
        x0, cap = inv_ctx
        return self._literal_inversion_rows(
            x,
            x0=x0,
            cap=cap,
        )

    def _frame_arrays(self) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
        """Return (x, z, Z) arrays according to current viewing-frame choice."""
        if self.view_frame_dropdown.value == "body":
            return (
                self._traj_cache.get("x_body"),
                self._traj_cache["z_body"],
                self._traj_cache["Z_body"],
            )
        return (
            self._traj_cache.get("x_lab"),
            self._traj_cache["z_lab"],
            self._traj_cache["Z_lab"],
        )

    def _reconstruct_points_numpy(
        self,
        *,
        w_t: np.ndarray,
        zeta_t: np.ndarray,
        frame_name: Literal["lab", "body"],
    ) -> np.ndarray:
        if self._base_points_np is None:
            raise RuntimeError("base points are not available for frame reconstruction.")

        # Real Möbius map on S^2 in vector form:
        # M_w(x) = ((1-|w|^2)(x-w))/|x-w|^2 - w
        base = self._base_points_np
        diff = base - w_t[None, :]
        den = np.einsum("ij,ij->i", diff, diff)[:, None]
        w2 = float(np.dot(w_t, w_t))
        x_body = ((1.0 - w2) / np.maximum(den, 1e-12)) * diff - w_t[None, :]
        norms = np.linalg.norm(x_body, axis=1, keepdims=True)
        x_body = x_body / np.maximum(norms, 1e-12)
        if frame_name == "body":
            return x_body
        return x_body @ zeta_t.T

    def _points_at_frame(
        self,
        t: int,
        x_series: np.ndarray | None = None,
        *,
        frame_name: Literal["lab", "body"] | None = None,
    ) -> np.ndarray:
        if x_series is not None:
            return x_series[t]

        f_name: Literal["lab", "body"] = (
            frame_name if frame_name is not None else ("body" if self.view_frame_dropdown.value == "body" else "lab")
        )
        return self._reconstruct_points_numpy(
            w_t=np.asarray(self._traj_cache["w"][t], dtype=np.float32),
            zeta_t=np.asarray(self._traj_cache["zeta"][t], dtype=np.float32),
            frame_name=f_name,
        )

    @staticmethod
    def _deterministic_subsample_indices(n: int, sample_size: int) -> np.ndarray:
        if n <= 0:
            return np.zeros((0,), dtype=np.int64)
        m = int(max(1, sample_size))
        if m >= n:
            return np.arange(n, dtype=np.int64)
        step = float(n) / float(m)
        idx = np.floor(np.arange(m, dtype=np.float64) * step).astype(np.int64)
        return np.clip(idx, 0, n - 1)

    @staticmethod
    def _kernel_uniform_density(*, kappa: float, sample_count: int) -> float:
        m = float(max(1, int(sample_count)))
        k = float(max(kappa, 0.0))
        if k < 1e-8:
            pair_expect = 1.0
        else:
            # On S^2 for u = x·y ~ Unif[-1,1], E[exp(k(u-1))] = (1-exp(-2k))/(2k).
            pair_expect = (1.0 - math.exp(-2.0 * k)) / (2.0 * k)
        return float((1.0 / m) + (1.0 - 1.0 / m) * pair_expect)

    def _kernel_entropy_proxy_numpy(
        self,
        *,
        points: np.ndarray,
        kappa: float = 14.0,
        sample_size: int = 420,
    ) -> float:
        pts = np.asarray(points, dtype=np.float64)
        n = int(pts.shape[0])
        if n <= 1:
            return 0.0
        idx = self._deterministic_subsample_indices(n, sample_size)
        xs = np.ascontiguousarray(pts[idx], dtype=np.float64)
        m = int(xs.shape[0])
        if m <= 1:
            return 0.0
        gram = np.clip(xs @ xs.T, -1.0, 1.0)
        kernel = np.exp(float(kappa) * (gram - 1.0))
        density = kernel.mean(axis=1)
        h_raw = -float(np.mean(np.log(np.maximum(density, 1e-12))))

        rho_uniform = self._kernel_uniform_density(kappa=float(kappa), sample_count=m)
        h_uniform = -math.log(max(rho_uniform, 1e-12))
        if h_uniform <= 1e-12:
            return 0.0
        return float(np.clip(h_raw / h_uniform, 0.0, 1.2))

    def _refresh_metric_series(self, params: dict[str, float | int]) -> None:
        """Refresh full metric lines according to selected view frame."""
        if not self._traj_cache:
            return
        t_axis = np.arange(self._steps + 1)
        x_series, z_series, Z_series = self._frame_arrays()
        z_norm = np.linalg.norm(Z_series, axis=1) / max(float(params["K"]), 1e-9)
        w_norm = np.linalg.norm(self._traj_cache["w"], axis=1)
        phase_xyz = z_series
        # For now, all modes compute thermo series once per recompute.
        thermo = self._compute_thermo_metrics(
            x_series=x_series,
            z_series=z_series,
            Z_series=Z_series,
            frame_name="body" if self.view_frame_dropdown.value == "body" else "lab",
        )
        self._metric_cache = thermo

        with self.metrics_fig.batch_update():
            self.metrics_fig.data[0].x = t_axis.tolist()
            self.metrics_fig.data[0].y = z_norm.tolist()
            self.metrics_fig.data[1].x = t_axis.tolist()
            self.metrics_fig.data[1].y = w_norm.tolist()
            self.metrics_fig.data[4].x = t_axis.tolist()
            self.metrics_fig.data[4].y = phase_xyz[:, 0].tolist()
            self.metrics_fig.data[5].x = t_axis.tolist()
            self.metrics_fig.data[5].y = phase_xyz[:, 1].tolist()
            self.metrics_fig.data[6].x = t_axis.tolist()
            self.metrics_fig.data[6].y = phase_xyz[:, 2].tolist()
            self.metrics_fig.data[10].x = t_axis.tolist()
            self.metrics_fig.data[10].y = thermo["entropy"].tolist()
            self.metrics_fig.data[11].x = t_axis.tolist()
            self.metrics_fig.data[11].y = thermo["var_to_center"].tolist()
            self.metrics_fig.data[12].x = t_axis.tolist()
            self.metrics_fig.data[12].y = thermo["var_to_conformal_center"].tolist()
            self.metrics_fig.data[13].x = t_axis.tolist()
            self.metrics_fig.data[13].y = thermo["bary_emp"].tolist()
            self.metrics_fig.data[14].x = t_axis.tolist()
            self.metrics_fig.data[14].y = thermo["bary_conf"].tolist()
            self.metrics_fig.update_xaxes(range=[0, int(params["steps"])], row=1, col=1)
            self.metrics_fig.update_xaxes(range=[0, int(params["steps"])], row=2, col=1)
            self.metrics_fig.update_xaxes(range=[0, int(params["steps"])], row=3, col=1)

    def _compute_thermo_metrics(
        self,
        *,
        x_series: np.ndarray | None,
        z_series: np.ndarray,
        Z_series: np.ndarray,
        frame_name: Literal["lab", "body"],
    ) -> dict[str, np.ndarray]:
        """Compute entropy and Euclidean-variance diagnostics over the trajectory."""
        if x_series is not None:
            t_count, n, d = x_series.shape
        else:
            t_count = z_series.shape[0]
            if self._base_points_np is None:
                raise RuntimeError("base points are not available for thermo metric reconstruction.")
            n, d = self._base_points_np.shape

        bary_emp = np.zeros(t_count, dtype=np.float64)
        z_norm = np.linalg.norm(z_series, axis=1)
        shrink = np.array([self._shrink_fd(d=d, r=float(r)) for r in z_norm], dtype=np.float64)
        bary_conf = shrink * z_norm

        var_to_center = np.zeros(t_count, dtype=np.float64)
        var_to_conformal_center = np.zeros(t_count, dtype=np.float64)

        # Spherical histogram entropy estimate in [0, 1] (kept for reference).
        n_u = 12
        n_phi = 24
        n_bins = n_u * n_phi
        log_bins = math.log(float(n_bins))
        counts = np.zeros(n_bins, dtype=np.float64)
        entropy_hist = np.zeros(t_count, dtype=np.float64)
        entropy_proxy = np.zeros(t_count, dtype=np.float64)
        entropy_kappa = 14.0
        entropy_sample_size = min(int(n), 420)
        var_par = np.zeros(t_count, dtype=np.float64)
        var_perp = np.zeros(t_count, dtype=np.float64)
        for t in range(t_count):
            if x_series is not None:
                pts = x_series[t]
            else:
                pts = self._points_at_frame(
                    t,
                    x_series=None,
                    frame_name=frame_name,
                )

            mu_t = pts.mean(axis=0)
            bary_emp_t = float(np.linalg.norm(mu_t))
            bary_emp[t] = bary_emp_t
            scale = bary_conf[t] / max(bary_emp_t, 1e-12)
            mu_conf_t = mu_t * scale
            centered_emp_t = pts - mu_t[None, :]
            centered_conf_t = pts - mu_conf_t[None, :]
            sq_emp_t = np.sum(centered_emp_t * centered_emp_t, axis=1)
            sq_conf_t = np.sum(centered_conf_t * centered_conf_t, axis=1)
            var_to_center[t] = float(sq_emp_t.mean())
            var_to_conformal_center[t] = float(sq_conf_t.mean())

            # Entropy decomposition aligned with the order-parameter axis:
            # parallel component (affecting dipole direction) vs perpendicular spread.
            axis = z_series[t]
            axis_n = float(np.linalg.norm(axis))
            if axis_n < 1e-12:
                if bary_emp_t > 1e-12:
                    axis = mu_t / max(bary_emp_t, 1e-12)
                else:
                    axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                    axis_n = 1.0
            axis_u = axis / max(axis_n, 1e-12)
            proj = pts @ axis_u
            proj_mean = float(proj.mean())
            par_t = float(np.mean((proj - proj_mean) ** 2))
            perp = pts - proj[:, None] * axis_u[None, :]
            perp_mean = perp.mean(axis=0, keepdims=True)
            perp_t = float(np.mean(np.sum((perp - perp_mean) ** 2, axis=1)))
            var_par[t] = par_t
            var_perp[t] = perp_t
            if self.init_metric_mode == "perp_variance":
                entropy_proxy[t] = float(np.clip(perp_t, 0.0, 1.0))
            else:
                entropy_proxy[t] = self._kernel_entropy_proxy_numpy(
                    points=pts,
                    kappa=entropy_kappa,
                    sample_size=entropy_sample_size,
                )

            x = pts[:, 0]
            y = pts[:, 1]
            z = np.clip(pts[:, 2], -1.0, 1.0)
            u = 0.5 * (z + 1.0)
            az = np.arctan2(y, x)
            iu = np.minimum(n_u - 1, np.floor(u * n_u).astype(np.int64))
            ip = np.minimum(n_phi - 1, np.floor(((az + np.pi) / (2.0 * np.pi)) * n_phi).astype(np.int64))
            flat = iu * n_phi + ip
            counts.fill(0.0)
            np.add.at(counts, flat, 1.0)
            p = counts[counts > 0.0] / float(max(n, 1))
            h = float(-(p * np.log(p)).sum()) if p.size else 0.0
            entropy_hist[t] = h / max(log_bins, 1e-12)

        return {
            "entropy": entropy_proxy,
            "entropy_hist": entropy_hist,
            "var_to_center": var_to_center,
            "var_to_conformal_center": var_to_conformal_center,
            "bary_emp": bary_emp,
            "bary_conf": bary_conf,
            "var_parallel": var_par,
            "var_perp": var_perp,
        }

    @staticmethod
    def _estimate_points_storage_mb(*, n: int, steps: int, d: int, channels: int) -> float:
        # bytes_est = T * N * d * 4 * channels, with T = steps + 1 and float32 storage.
        bytes_est = float(steps + 1) * float(n) * float(d) * 4.0 * float(channels)
        return bytes_est / (1024.0 * 1024.0)

    def _resolve_store_points(
        self, *, n: int, steps: int, d: int
    ) -> tuple[Literal["none", "body", "lab", "both"], Literal["memory", "fps"]]:
        if self.trajectory_mode == "memory":
            return "none", "memory"
        if self.trajectory_mode == "fps":
            return "both", "fps"

        # auto mode: compare estimate against configured memory budget.
        est_mb = self._estimate_points_storage_mb(n=n, steps=steps, d=d, channels=2)
        if est_mb > float(self.memory_budget_mb):
            return "none", "memory"
        return "both", "fps"

    @staticmethod
    def _hyp2f1_1b_c_u(b: float, c: float, u: float) -> float:
        """Gauss hypergeometric 2F1(1,b;c;u) for |u|<1 via series."""
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

    @staticmethod
    def _hyp2f1_1b_c_1(b: float, c: float) -> float:
        """Closed form 2F1(1,b;c;1) when Re(c-1-b)>0."""
        return float(
            math.gamma(c) * math.gamma(c - 1.0 - b)
            / (math.gamma(c - 1.0) * math.gamma(c - b))
        )

    def _shrink_fd(self, d: int, r: float) -> float:
        """Continuum LMS shrink factor f_d(r) from the hypergeometric ratio."""
        r_clamped = max(0.0, min(0.999999999, float(r)))
        if d == 2:
            return 1.0
        b = 1.0 - 0.5 * float(d)
        c = 1.0 + 0.5 * float(d)
        u = r_clamped * r_clamped
        fu = self._hyp2f1_1b_c_u(b, c, u)
        f1 = self._hyp2f1_1b_c_1(b, c)
        if abs(f1) < 1e-12:
            return 1.0
        return float(fu / f1)

    def _simulate(self, params: dict[str, float | int]):
        d = 3
        n = int(params["N"])
        K = float(params["K"])
        conformal_sign = -1.0 if bool(self.toggle_entropy.value) else 1.0
        omega = float(params["omega"])
        r0 = float(params["r0"])
        w_az = float(params["w_az"])
        w_el = float(params["w_el"])
        ax_az = float(params["ax_az"])
        ax_el = float(params["ax_el"])
        dt = self._effective_dt(params)
        steps = int(params["steps"])
        store_points, resolved_mode = self._resolve_store_points(n=n, steps=steps, d=d)
        self._resolved_trajectory_mode = resolved_mode

        weights = torch.ones(n, dtype=torch.float64) / float(n)
        center_dir = torch.tensor(_angles_to_unit(w_az, w_el), dtype=torch.float64)
        init_mode = self._coerce_mode_to_active_family(self._init_state_mode)
        self._init_state_mode = init_mode
        if init_mode == "poisson":
            # Finite-N Poisson-manifold initialization:
            # choose uniform base points and requested reduced center w0 directly.
            base_points = random_points_on_sphere(
                n,
                d=d,
                generator=self._torch_gen,
                dtype=torch.float64,
            )
            r0_clip = float(np.clip(r0, 0.0, 0.999999))
            # Keep sign convention consistent with optimized initializers:
            # requested axis controls the physical dipole direction z=-w at t=0.
            w0 = -center_dir * r0_clip
        else:
            x0_points = self._make_initial_boundary_points(
                n=n,
                d=d,
                w_az=w_az,
                w_el=w_el,
                target_r=float(r0),
                init_state=init_mode,
            )
            w0 = self._estimate_w_from_boundary_points(
                points=x0_points,
                weights=weights,
                d=d,
                fallback_dir=center_dir,
            )
            base_points = self._recover_base_points_from_state(
                x_points=x0_points,
                w0=w0,
            )

        zeta0 = torch.eye(d, dtype=torch.float64)

        axis = torch.tensor(_angles_to_unit(ax_az, ax_el), dtype=torch.float64)
        A = skew_symmetric_from_axis(axis, rate=omega).to(dtype=torch.float64)

        return integrate_lms_reduced_euler(
            w0=w0,
            zeta0=zeta0,
            base_points=base_points,
            weights=weights,
            A=A,
            coupling=conformal_sign * K,
            dt=dt,
            steps=steps,
            w_mode=str(self.mode_dropdown.value),
            project_rotation=True,
            store_points=store_points,
            store_dtype=torch.float32,
            preallocate=True,
        )

    def _make_initial_boundary_points(
        self,
        *,
        n: int,
        d: int,
        w_az: float,
        w_el: float,
        target_r: float,
        init_state: OptimizedInitMode,
    ) -> torch.Tensor:
        mode = self._canonical_init_mode(str(init_state))
        if mode in {"entropy_high", "entropy_low"}:
            return self._make_entropy_initial_boundary_points(
                n=n,
                d=d,
                w_az=w_az,
                w_el=w_el,
                target_r=target_r,
                init_state=mode,
            )
        if mode in {"var_perp_high", "var_perp_low"}:
            return self._make_perp_variance_initial_boundary_points(
                n=n,
                d=d,
                w_az=w_az,
                w_el=w_el,
                target_r=target_r,
                init_state=mode,
            )
        if mode in {"varmax_parallel_perp", "varmin_parallel_perp"}:
            return self._make_variance_initial_boundary_points(
                n=n,
                d=d,
                w_az=w_az,
                w_el=w_el,
                target_r=target_r,
                init_state=mode,
            )
        raise ValueError(f"Unsupported init state: {init_state}")

    def _make_perp_variance_initial_boundary_points(
        self,
        *,
        n: int,
        d: int,
        w_az: float,
        w_el: float,
        target_r: float,
        init_state: Literal["var_perp_high", "var_perp_low"],
    ) -> torch.Tensor:
        return self._make_variance_initial_boundary_points(
            n=n,
            d=d,
            w_az=w_az,
            w_el=w_el,
            target_r=target_r,
            init_state="varmin_parallel_perp" if init_state == "var_perp_low" else "varmax_parallel_perp",
            parallel_weight=0.0,
        )

    def _make_entropy_initial_boundary_points(
        self,
        *,
        n: int,
        d: int,
        w_az: float,
        w_el: float,
        target_r: float,
        init_state: Literal["entropy_high", "entropy_low"],
    ) -> torch.Tensor:
        """Entropy-gradient initializer from random starts with centroid constraints.

        This is a stochastic optimization of a spherical kernel-entropy surrogate.
        It keeps the requested first moment (radius/direction) while ascending or
        descending entropy, without any ad-hoc clustered templates.
        """
        target_dir = torch.tensor(_angles_to_unit(w_az, w_el), dtype=torch.float64)
        target_dir = normalize(target_dir.unsqueeze(0))[0]
        q_target = self._radius_to_q_target(d=d, r=float(target_r))
        maximize_entropy = init_state == "entropy_high"

        x0 = random_points_on_sphere(
            n,
            d=d,
            generator=self._torch_gen,
            dtype=torch.float64,
        )
        if q_target > 1e-9:
            mix = float(np.clip(q_target, 0.0, 0.95))
            x0 = normalize((1.0 - mix) * x0 + mix * target_dir.unsqueeze(0))

        x_var = torch.nn.Parameter(x0.clone())
        opt = torch.optim.Adam([x_var], lr=0.065 if maximize_entropy else 0.070)

        max_iters = 170 if maximize_entropy else 180
        min_iters = 28
        move_cap = 0.050 if maximize_entropy else 0.040
        q_tol = self._q_tolerance_from_r_tolerance(d=d, r=float(target_r), dr=0.01)

        slack = float(max(1e-3, 1.0 - q_target * q_target))
        w_q = 320.0
        w_dir = 40.0 if q_target > 1e-5 else 0.0
        w_entropy = 11.0 + 19.0 * slack
        kappa = 10.0 + 26.0 * float(np.clip(q_target, 0.0, 1.0))
        sample_size = min(n, 800 if n >= 5000 else 640)
        entropy_sign = -1.0 if maximize_entropy else 1.0

        x_prev = normalize(x_var.detach())
        h_prev = float("nan")
        stable_h = 0
        for it in range(max_iters):
            opt.zero_grad(set_to_none=True)
            x = normalize(x_var)
            mean = x.mean(dim=0)
            q = torch.linalg.norm(mean)

            q_loss = (q - q_target) ** 2
            if w_dir > 0.0:
                mean_dir = mean / (q + 1e-12)
                dir_cos = torch.clamp(torch.dot(mean_dir, target_dir), -1.0, 1.0)
                dir_loss = (1.0 - dir_cos) ** 2
            else:
                dir_loss = torch.zeros((), dtype=x.dtype, device=x.device)

            entropy_proxy = self._spherical_entropy_proxy(
                points=x,
                kappa=kappa,
                sample_size=sample_size,
            )
            loss = w_q * q_loss + w_dir * dir_loss + entropy_sign * w_entropy * entropy_proxy

            loss.backward()
            opt.step()

            with torch.no_grad():
                x_new = normalize(x_var)
                delta = x_new - x_prev
                delta_norm = torch.linalg.norm(delta, dim=1, keepdim=True)
                scale = torch.clamp(move_cap / (delta_norm + 1e-12), max=1.0)
                x_step = normalize(x_prev + delta * scale)
                x_var.copy_(x_step)
                x_prev = x_step.detach()

            h_now = float(entropy_proxy.detach())
            if math.isfinite(h_prev) and abs(h_now - h_prev) < 2.0e-4:
                stable_h += 1
            else:
                stable_h = 0
            h_prev = h_now

            if it >= min_iters and abs(float(q.detach()) - q_target) <= q_tol and stable_h >= 6:
                break

        # Radius-lock refinement pass.
        opt_refine = torch.optim.Adam([x_var], lr=0.045)
        refine_iters = 48
        refine_cap = 0.7 * move_cap
        x_prev = normalize(x_var.detach())
        for _ in range(refine_iters):
            opt_refine.zero_grad(set_to_none=True)
            x = normalize(x_var)
            mean = x.mean(dim=0)
            q = torch.linalg.norm(mean)
            q_loss = (q - q_target) ** 2
            if w_dir > 0.0:
                mean_dir = mean / (q + 1e-12)
                dir_cos = torch.clamp(torch.dot(mean_dir, target_dir), -1.0, 1.0)
                dir_loss = (1.0 - dir_cos) ** 2
            else:
                dir_loss = torch.zeros((), dtype=x.dtype, device=x.device)
            loss = 420.0 * q_loss + 20.0 * dir_loss
            loss.backward()
            opt_refine.step()

            with torch.no_grad():
                x_new = normalize(x_var)
                delta = x_new - x_prev
                delta_norm = torch.linalg.norm(delta, dim=1, keepdim=True)
                scale = torch.clamp(refine_cap / (delta_norm + 1e-12), max=1.0)
                x_step = normalize(x_prev + delta * scale)
                x_var.copy_(x_step)
                x_prev = x_step.detach()
                if abs(float(q.detach()) - q_target) <= 0.55 * q_tol:
                    break

        x_final = normalize(x_var.detach())
        x_final = self._tune_radius_by_global_shift(
            points=x_final,
            axis=target_dir,
            q_target=q_target,
        )
        return normalize(x_final)

    def _spherical_entropy_proxy(
        self,
        *,
        points: torch.Tensor,
        kappa: float,
        sample_size: int,
    ) -> torch.Tensor:
        """Kernel entropy proxy H ~ -E log rho from a stochastic self-sample."""
        n = int(points.shape[0])
        if n <= 1:
            return torch.zeros((), dtype=points.dtype, device=points.device)
        if sample_size >= n:
            xs = points
        else:
            idx = torch.randperm(n, generator=self._torch_gen, device=points.device)[:sample_size]
            xs = points[idx]
        gram = torch.clamp(xs @ xs.T, -1.0, 1.0)
        kernel = torch.exp(float(kappa) * (gram - 1.0))
        density = kernel.mean(dim=1)
        return -torch.log(density + 1e-12).mean()

    def _make_variance_initial_boundary_points(
        self,
        *,
        n: int,
        d: int,
        w_az: float,
        w_el: float,
        target_r: float,
        init_state: Literal["varmax_parallel_perp", "varmin_parallel_perp"],
        parallel_weight: float = 0.10,
    ) -> torch.Tensor:
        """Legacy variance-proxy initializer kept for future comparisons."""
        target_dir = torch.tensor(_angles_to_unit(w_az, w_el), dtype=torch.float64)
        target_dir = normalize(target_dir.unsqueeze(0))[0]
        q_target = self._radius_to_q_target(d=d, r=float(target_r))
        minimize_variance = init_state == "varmin_parallel_perp"

        x0 = random_points_on_sphere(
            n,
            d=d,
            generator=self._torch_gen,
            dtype=torch.float64,
        )
        if q_target > 1e-9:
            mix = float(np.clip(q_target, 0.0, 0.95))
            x0 = normalize((1.0 - mix) * x0 + mix * target_dir.unsqueeze(0))

        x_var = torch.nn.Parameter(x0.clone())
        opt = torch.optim.Adam([x_var], lr=0.08 if minimize_variance else 0.06)

        max_iters = 160 if minimize_variance else 140
        min_iters = 24
        move_cap = 0.035 if minimize_variance else 0.055
        q_tol = self._q_tolerance_from_r_tolerance(d=d, r=float(target_r), dr=0.01)

        slack = float(max(1e-3, 1.0 - q_target * q_target))
        w_q = 260.0
        w_dir = 34.0 if q_target > 1e-5 else 0.0
        entropy_scale = 18.0 * slack
        w_par = float(parallel_weight) * entropy_scale
        w_perp = 1.00 * entropy_scale

        x_prev = normalize(x_var.detach())
        for it in range(max_iters):
            opt.zero_grad(set_to_none=True)
            x = normalize(x_var)
            mean = x.mean(dim=0)
            q = torch.linalg.norm(mean)

            q_loss = (q - q_target) ** 2
            if w_dir > 0.0:
                mean_dir = mean / (q + 1e-12)
                dir_cos = torch.clamp(torch.dot(mean_dir, target_dir), -1.0, 1.0)
                dir_loss = (1.0 - dir_cos) ** 2
            else:
                dir_loss = torch.zeros((), dtype=x.dtype, device=x.device)

            var_parallel, var_perp = self._entropy_components(
                points=x,
                axis=target_dir,
            )
            entropy_term = w_perp * var_perp + w_par * var_parallel
            if minimize_variance:
                loss = w_q * q_loss + w_dir * dir_loss + entropy_term
            else:
                loss = w_q * q_loss + w_dir * dir_loss - entropy_term

            loss.backward()
            opt.step()

            with torch.no_grad():
                x_new = normalize(x_var)
                delta = x_new - x_prev
                delta_norm = torch.linalg.norm(delta, dim=1, keepdim=True)
                scale = torch.clamp(move_cap / (delta_norm + 1e-12), max=1.0)
                x_step = normalize(x_prev + delta * scale)
                x_var.copy_(x_step)
                x_prev = x_step.detach()

            if it >= min_iters and abs(float(q.detach()) - q_target) <= q_tol:
                break

        opt_refine = torch.optim.Adam([x_var], lr=0.045)
        refine_iters = 48
        refine_cap = 0.7 * move_cap
        x_prev = normalize(x_var.detach())
        for _ in range(refine_iters):
            opt_refine.zero_grad(set_to_none=True)
            x = normalize(x_var)
            mean = x.mean(dim=0)
            q = torch.linalg.norm(mean)
            q_loss = (q - q_target) ** 2
            if w_dir > 0.0:
                mean_dir = mean / (q + 1e-12)
                dir_cos = torch.clamp(torch.dot(mean_dir, target_dir), -1.0, 1.0)
                dir_loss = (1.0 - dir_cos) ** 2
            else:
                dir_loss = torch.zeros((), dtype=x.dtype, device=x.device)
            loss = 420.0 * q_loss + 20.0 * dir_loss
            loss.backward()
            opt_refine.step()

            with torch.no_grad():
                x_new = normalize(x_var)
                delta = x_new - x_prev
                delta_norm = torch.linalg.norm(delta, dim=1, keepdim=True)
                scale = torch.clamp(refine_cap / (delta_norm + 1e-12), max=1.0)
                x_step = normalize(x_prev + delta * scale)
                x_var.copy_(x_step)
                x_prev = x_step.detach()
                if abs(float(q.detach()) - q_target) <= 0.55 * q_tol:
                    break

        x_final = normalize(x_var.detach())
        x_final = self._tune_radius_by_global_shift(
            points=x_final,
            axis=target_dir,
            q_target=q_target,
        )
        return normalize(x_final)

    @staticmethod
    def _entropy_components(
        *,
        points: torch.Tensor,
        axis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (parallel, perpendicular) variance components vs a fixed axis."""
        axis_u = normalize(axis.unsqueeze(0))[0]
        proj = points @ axis_u  # [N]
        proj_mean = proj.mean()
        var_parallel = ((proj - proj_mean) ** 2).mean()

        perp = points - proj[:, None] * axis_u[None, :]
        perp_mean = perp.mean(dim=0, keepdim=True)
        var_perp = ((perp - perp_mean) ** 2).sum(dim=1).mean()
        return var_parallel, var_perp

    def _radius_to_q_target(self, *, d: int, r: float) -> float:
        r_clip = float(np.clip(r, 0.0, 0.999999))
        return float(np.clip(self._shrink_fd(d=d, r=r_clip) * r_clip, 0.0, 0.999999))

    def _q_tolerance_from_r_tolerance(self, *, d: int, r: float, dr: float) -> float:
        r0 = float(np.clip(r, 0.0, 0.999999))
        r_lo = float(np.clip(r0 - abs(dr), 0.0, 0.999999))
        r_hi = float(np.clip(r0 + abs(dr), 0.0, 0.999999))
        q0 = self._radius_to_q_target(d=d, r=r0)
        q_lo = self._radius_to_q_target(d=d, r=r_lo)
        q_hi = self._radius_to_q_target(d=d, r=r_hi)
        return float(max(abs(q0 - q_lo), abs(q_hi - q0), 1e-4))

    @staticmethod
    def _tune_radius_by_global_shift(
        *,
        points: torch.Tensor,
        axis: torch.Tensor,
        q_target: float,
    ) -> torch.Tensor:
        """Shift all points along axis and renormalize; pick shift closest to q_target."""
        axis_u = normalize(axis.unsqueeze(0))[0]
        x = normalize(points)
        best = x
        best_err = float("inf")

        def _scan(lo: float, hi: float, steps: int, current_best_err: float) -> tuple[torch.Tensor, float, float]:
            best_local = best
            best_err_local = current_best_err
            best_delta = 0.0
            for delta in np.linspace(lo, hi, steps):
                y = normalize(x + float(delta) * axis_u.unsqueeze(0))
                m = y.mean(dim=0)
                q = float(torch.linalg.norm(m))
                # Penalize sign flips against the requested axis direction.
                sign_pen = 1.0 if float(torch.dot(m, axis_u)) < 0.0 else 0.0
                err = abs(q - q_target) + sign_pen
                if err < best_err_local:
                    best_err_local = err
                    best_local = y
                    best_delta = float(delta)
            return best_local, best_err_local, best_delta

        # Coarse-to-fine search over a monotone-like 1D control.
        best, best_err, center = _scan(-3.0, 3.0, 121, best_err)
        for span, steps in ((0.8, 81), (0.25, 81), (0.08, 81)):
            best, best_err, center = _scan(center - span, center + span, steps, best_err)
        return best

    def _estimate_w_from_boundary_points(
        self,
        *,
        points: torch.Tensor,
        weights: torch.Tensor,
        d: int,
        fallback_dir: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate reduced w directly from boundary points via first moment.

        Uses direction of weighted centroid and radial inversion of
        q(r) = f_d(r) * r in [0, 1), where q = ||sum_i a_i x_i||.
        """
        m = (weights[:, None] * points).sum(dim=0)
        q = float(torch.linalg.norm(m))
        q = float(np.clip(q, 0.0, 0.999999))
        if q < 1e-12:
            return fallback_dir * 0.0

        # In reduced coordinates z = -zeta w, so for zeta=I at initialization
        # the physical centroid direction aligns with -w (not +w).
        direction = -m / max(q, 1e-12)

        def q_of_r(r: float) -> float:
            return float(self._shrink_fd(d=d, r=r) * r)

        lo = 0.0
        hi = 0.999999
        q_hi = q_of_r(hi)
        if q >= q_hi:
            r = hi
        else:
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if q_of_r(mid) < q:
                    lo = mid
                else:
                    hi = mid
            r = 0.5 * (lo + hi)
        return direction * float(r)

    @staticmethod
    def _recover_base_points_from_state(
        *,
        x_points: torch.Tensor,
        w0: torch.Tensor,
    ) -> torch.Tensor:
        """Recover body-frame base points p_i so that x_i(0)=M_{w0}(p_i)."""
        p = mobius_sphere(x_points, -w0)
        return normalize(p)

    def _recompute(self, *, reset_frame: bool) -> None:
        params = self._params()
        self._params_cache = dict(params)
        prev_frame = int(self.frame_slider.value)
        traj = self._simulate(params)
        self._steps = int(traj.steps)
        self._base_points_np = np.ascontiguousarray(
            np.asarray(traj.base_points.detach().cpu().numpy(), dtype=np.float32)
        )

        w_np = np.ascontiguousarray(np.asarray(traj.w.detach().cpu().numpy(), dtype=np.float32))
        zeta_np = np.ascontiguousarray(np.asarray(traj.zeta.detach().cpu().numpy(), dtype=np.float32))
        z_np = np.ascontiguousarray(np.asarray(traj.z.detach().cpu().numpy(), dtype=np.float32))
        z_body_np = np.ascontiguousarray(np.asarray((-traj.w).detach().cpu().numpy(), dtype=np.float32))
        Z_np = np.ascontiguousarray(np.asarray(traj.Z.detach().cpu().numpy(), dtype=np.float32))
        Z_body_np = np.ascontiguousarray(np.asarray(traj.Z_body.detach().cpu().numpy(), dtype=np.float32))

        x_lab_np = (
            np.ascontiguousarray(np.asarray(traj.x_lab.detach().cpu().numpy(), dtype=np.float32))
            if traj.x_lab is not None
            else None
        )
        x_body_np = (
            np.ascontiguousarray(np.asarray(traj.x_body.detach().cpu().numpy(), dtype=np.float32))
            if traj.x_body is not None
            else None
        )

        self._traj_cache = {
            "w": w_np,
            "zeta": zeta_np,
            "z_lab": z_np,
            "z_body": z_body_np,
            "Z_lab": Z_np,
            "Z_body": Z_body_np,
        }
        if x_lab_np is not None:
            self._traj_cache["x_lab"] = x_lab_np
        if x_body_np is not None:
            self._traj_cache["x_body"] = x_body_np
        n_points = int(self._base_points_np.shape[0]) if self._base_points_np is not None else 0
        if self.display_points_cap is None or n_points <= self.display_points_cap:
            self._display_indices = None
        else:
            stride = max(1, int(math.ceil(n_points / float(self.display_points_cap))))
            self._display_indices = np.arange(0, n_points, stride, dtype=np.int32)[: self.display_points_cap]
        del traj
        self._refresh_metric_series(params)
        self._last_overlay_frame = -10**9
        self._last_path_frame = -10**9

        frame_target = 0 if reset_frame else max(0, min(prev_frame, self._steps))
        self._updating = True
        try:
            self.frame_slider.max = self._steps
            self.play.max = self._steps
            self.frame_slider.value = frame_target
            self.play.value = frame_target
        finally:
            self._updating = False

        self._render_frame(int(self.frame_slider.value))

    def _render_frame(self, t: int) -> None:
        if self._steps <= 0 or not self._traj_cache:
            return
        params = self._params_cache if self._params_cache else self._params()
        t = max(0, min(t, self._steps))

        frame_name: Literal["lab", "body"] = "body" if self.view_frame_dropdown.value == "body" else "lab"
        x_series, z_series, Z_series = self._frame_arrays()
        x = self._points_at_frame(t, x_series, frame_name=frame_name)
        if self._display_indices is not None and len(self._display_indices) < x.shape[0]:
            x_plot = x[self._display_indices]
        else:
            x_plot = x
        w = self._traj_cache["w"][t]
        z = z_series[t]
        Z = Z_series[t]
        K = max(float(params["K"]), 1e-9)
        Z_hat = Z / K
        inv_ctx = self._inversion_context(frame_name=frame_name) if bool(self._inversion_enabled) else None
        x_plot_disp = self._maybe_invert_rows(x_plot, frame_name=frame_name, inv_ctx=inv_ctx)
        w_disp = self._maybe_invert_rows(w, frame_name=frame_name, inv_ctx=inv_ctx)
        z_disp = self._maybe_invert_rows(z, frame_name=frame_name, inv_ctx=inv_ctx)
        Z_hat_disp = self._maybe_invert_rows(Z_hat, frame_name=frame_name, inv_ctx=inv_ctx)

        show_paths = bool(self.show_paths.value)
        show_vectors = bool(self.show_vectors.value)
        force_update = (t == 0) or (t == self._steps)
        path_update = force_update or (abs(t - self._last_path_frame) >= self._path_stride())

        idx = self._wire_count
        self._in_frame_update = True
        try:
            with self.sphere_fig.batch_update():
                self.sphere_fig.data[idx + 0].x = x_plot_disp[:, 0].tolist()
                self.sphere_fig.data[idx + 0].y = x_plot_disp[:, 1].tolist()
                self.sphere_fig.data[idx + 0].z = x_plot_disp[:, 2].tolist()

                self.sphere_fig.data[idx + 1].x = [float(w_disp[0])]
                self.sphere_fig.data[idx + 1].y = [float(w_disp[1])]
                self.sphere_fig.data[idx + 1].z = [float(w_disp[2])]

                self.sphere_fig.data[idx + 2].x = [float(z_disp[0])]
                self.sphere_fig.data[idx + 2].y = [float(z_disp[1])]
                self.sphere_fig.data[idx + 2].z = [float(z_disp[2])]

                self.sphere_fig.data[idx + 3].x = [float(Z_hat_disp[0])]
                self.sphere_fig.data[idx + 3].y = [float(Z_hat_disp[1])]
                self.sphere_fig.data[idx + 3].z = [float(Z_hat_disp[2])]

                if show_paths and path_update:
                    path_decim = 1 if not self._is_playing() else max(1, (t + 1) // 1200)
                    if path_decim > 1:
                        path_idx = np.arange(0, t + 1, path_decim, dtype=np.int32)
                        if path_idx[-1] != t:
                            path_idx = np.concatenate([path_idx, np.array([t], dtype=np.int32)])
                        wp = self._traj_cache["w"][path_idx]
                        zp = z_series[path_idx]
                        Zp = Z_series[path_idx] / K
                    else:
                        wp = self._traj_cache["w"][: t + 1]
                        zp = z_series[: t + 1]
                        Zp = Z_series[: t + 1] / K

                    wp_disp = self._maybe_invert_rows(wp, frame_name=frame_name, inv_ctx=inv_ctx)
                    zp_disp = self._maybe_invert_rows(zp, frame_name=frame_name, inv_ctx=inv_ctx)
                    Zp_disp = self._maybe_invert_rows(Zp, frame_name=frame_name, inv_ctx=inv_ctx)

                    self.sphere_fig.data[idx + 4].x = wp_disp[:, 0].tolist()
                    self.sphere_fig.data[idx + 4].y = wp_disp[:, 1].tolist()
                    self.sphere_fig.data[idx + 4].z = wp_disp[:, 2].tolist()

                    self.sphere_fig.data[idx + 5].x = zp_disp[:, 0].tolist()
                    self.sphere_fig.data[idx + 5].y = zp_disp[:, 1].tolist()
                    self.sphere_fig.data[idx + 5].z = zp_disp[:, 2].tolist()

                    self.sphere_fig.data[idx + 6].x = Zp_disp[:, 0].tolist()
                    self.sphere_fig.data[idx + 6].y = Zp_disp[:, 1].tolist()
                    self.sphere_fig.data[idx + 6].z = Zp_disp[:, 2].tolist()

                self.sphere_fig.data[idx + 7].x = [0.0, float(w_disp[0])]
                self.sphere_fig.data[idx + 7].y = [0.0, float(w_disp[1])]
                self.sphere_fig.data[idx + 7].z = [0.0, float(w_disp[2])]

                self.sphere_fig.data[idx + 8].x = [0.0, float(z_disp[0])]
                self.sphere_fig.data[idx + 8].y = [0.0, float(z_disp[1])]
                self.sphere_fig.data[idx + 8].z = [0.0, float(z_disp[2])]

                self.sphere_fig.data[idx + 9].x = [0.0, float(Z_hat_disp[0])]
                self.sphere_fig.data[idx + 9].y = [0.0, float(Z_hat_disp[1])]
                self.sphere_fig.data[idx + 9].z = [0.0, float(Z_hat_disp[2])]

                self.sphere_fig.data[idx + 4].visible = show_paths
                self.sphere_fig.data[idx + 5].visible = show_paths
                self.sphere_fig.data[idx + 6].visible = show_paths
                self.sphere_fig.data[idx + 7].visible = show_vectors
                self.sphere_fig.data[idx + 8].visible = show_vectors
                self.sphere_fig.data[idx + 9].visible = show_vectors
        finally:
            self._in_frame_update = False

        if path_update:
            self._last_path_frame = t
        overlay_update = force_update or (abs(t - self._last_overlay_frame) >= self._overlay_stride())
        if not overlay_update:
            return
        self._last_overlay_frame = t

        z_norm = float(np.linalg.norm(Z) / K)
        w_norm = float(np.linalg.norm(w))
        z_abs = float(np.linalg.norm(z))
        def _metric_at(key: str) -> float:
            arr = self._metric_cache.get(key) if self._metric_cache else None
            if arr is None or len(arr) <= t:
                return 0.0
            return float(arr[t])

        entropy_t = _metric_at("entropy")
        var_to_center_t = _metric_at("var_to_center")
        var_to_conformal_center_t = _metric_at("var_to_conformal_center")
        bary_emp_t = _metric_at("bary_emp")
        bary_conf_t = _metric_at("bary_conf")
        conformal_sign = -1.0 if bool(self.toggle_entropy.value) else 1.0
        time_sign = self._time_direction_sign()
        dt_eff = self._effective_dt(params)

        self.stats_html.value = (
            "<b>State statistics</b>"
            "<table style='font-family:monospace;font-size:12px;margin-top:6px'>"
            f"<tr><td style='padding-right:14px'>N</td><td>{int(params['N'])}</td></tr>"
            f"<tr><td style='padding-right:14px'>K</td><td>{float(params['K']):.3f}</td></tr>"
            f"<tr><td style='padding-right:14px'>omega</td><td>{float(params['omega']):.3f}</td></tr>"
            f"<tr><td style='padding-right:14px'>dt eff</td><td>{dt_eff:+.4f}</td></tr>"
            f"<tr><td style='padding-right:14px'>Time direction</td><td>{'backward' if time_sign < 0 else 'forward'}</td></tr>"
            f"<tr><td style='padding-right:14px'>|w|,|z|</td><td>{w_norm:.5f}</td></tr>"
            f"<tr><td style='padding-right:14px'>|Z|/K</td><td>{z_norm:.5f}</td></tr>"
            f"<tr><td style='padding-right:14px'>Entropy direction</td>"
            f"<td>{'increase' if bool(self.toggle_entropy.value) else 'dissipate'}</td></tr>"
            f"<tr><td style='padding-right:14px'>Alignment force sign</td><td>{conformal_sign:+.0f}</td></tr>"
            f"<tr><td style='padding-right:14px'>Trajectory mode</td><td>{self._resolved_trajectory_mode}</td></tr>"
            f"<tr><td style='padding-right:14px'>Thermo mode</td><td>{self.thermo_mode}</td></tr>"
            f"<tr><td style='padding-right:14px'>Initial state</td>"
            f"<td>{self._init_mode_label(self._init_state_mode)}</td></tr>"
            f"<tr><td style='padding-right:14px'>{self._primary_metric_title()}</td><td>{entropy_t:.5f}</td></tr>"
            f"<tr><td style='padding-right:14px'>Variance to center</td><td>{var_to_center_t:.5f}</td></tr>"
            f"<tr><td style='padding-right:14px'>Variance to conformal center</td><td>{var_to_conformal_center_t:.5f}</td></tr>"
            f"<tr><td style='padding-right:14px'>||mean(x)|| emp</td><td>{bary_emp_t:.5f}</td></tr>"
            "</table>"
        )

    def _on_control_change(self, _change: dict[str, Any]) -> None:
        if self._updating:
            return
        # Recompute trajectory on parameter changes (explicitly finite-N trajectories).
        self._recompute(reset_frame=False)

    def _on_visual_change(self, _change: dict[str, Any]) -> None:
        if self._updating:
            return
        if _change.get("owner") is self.view_frame_dropdown:
            self._refresh_metric_series(self._params())
            self._sync_mode_button_labels()
        self._render_frame(int(self.frame_slider.value))

    def _on_layout_change(self, _change: dict[str, Any]) -> None:
        if self._updating:
            return
        self._apply_root_layout()

    def _on_layout_dropdown_change(self, change: dict[str, Any]) -> None:
        if self._updating or change.get("name") != "value":
            return
        self.layout_top_view.value = bool(change.get("new") == "top")

    def _on_toggle_frame_clicked(self, _btn: widgets.Button) -> None:
        self.view_frame_dropdown.value = "body" if self.view_frame_dropdown.value == "lab" else "lab"

    def _on_toggle_layout_clicked(self, _btn: widgets.Button) -> None:
        self.layout_top_view.value = not bool(self.layout_top_view.value)

    def _on_time_direction_change(self, change: dict[str, Any]) -> None:
        if self._updating or change.get("name") != "value":
            return
        self._sync_time_direction_button_label()
        self._recompute(reset_frame=False)

    def _on_entropy_direction_change(self, change: dict[str, Any]) -> None:
        if self._updating or change.get("name") != "value":
            return
        self._sync_entropy_button_label()
        self._recompute(reset_frame=False)

    def _on_init_state_change(self, _change: dict[str, Any]) -> None:
        # Legacy no-op for backward compatibility with older bindings.
        return

    def _on_init_state_clicked(self, _btn: widgets.Button) -> None:
        if self._updating:
            return
        order = self._active_mode_order()
        mode = self._coerce_mode_to_active_family(self._init_state_mode)
        if mode not in order:
            mode = order[0]
        i = order.index(mode)
        self._init_state_mode = order[(i + 1) % len(order)]
        self._sync_init_state_button_label()
        self._recompute(reset_frame=False)

    def _on_speed_half_clicked(self, _btn: widgets.Button) -> None:
        self._set_playback_speed(self._playback_speed * 0.5)

    def _on_speed_double_clicked(self, _btn: widgets.Button) -> None:
        self._set_playback_speed(self._playback_speed * 2.0)

    def _on_frame_change(self, change: dict[str, Any]) -> None:
        if self._updating:
            return
        self._render_frame(int(change["new"]))

    def _on_recompute_clicked(self, _btn: widgets.Button) -> None:
        self._recompute(reset_frame=False)

    def _start_play(self, direction: int) -> None:
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")

        # Reset transient drag-pause state and restart play deterministically.
        self._paused_for_drag = False
        self._was_playing_before_drag = False
        self._set_playing(False)
        self.play.step = direction
        cur = int(self.frame_slider.value)
        if direction < 0 and cur <= self.frame_slider.min:
            # Requested behavior: reverse from frame 0 starts at the last frame.
            self.play.value = self.play.max
        elif direction > 0 and cur >= self.frame_slider.max:
            # Symmetric behavior for forward playback from the last frame.
            self.play.value = self.play.min
        else:
            self.play.value = cur

        # Guard against false camera-change pauses right after starting playback.
        self._ignore_camera_pause_until = time.monotonic() + 0.30
        self._set_playing(True)

    def _on_play_forward_clicked(self, _btn: widgets.Button) -> None:
        self._start_play(direction=1)

    def _on_play_backward_clicked(self, _btn: widgets.Button) -> None:
        self._start_play(direction=-1)

    def _camera_to_json(self) -> dict[str, Any] | None:
        cam = self.sphere_fig.layout.scene.camera
        if cam is None:
            return None
        if hasattr(cam, "to_plotly_json"):
            try:
                return self._ensure_camera_eye(cam.to_plotly_json())
            except Exception:
                return None
        try:
            return self._ensure_camera_eye(dict(cam))
        except Exception:
            return None

    @staticmethod
    def _camera_changed_materially(prev: dict[str, Any] | None, cur: dict[str, Any] | None) -> bool:
        if prev is None or cur is None:
            return True
        for group in ("eye", "up", "center"):
            p = prev.get(group, {}) or {}
            c = cur.get(group, {}) or {}
            for axis in ("x", "y", "z"):
                pv = float(p.get(axis, 0.0))
                cv = float(c.get(axis, 0.0))
                if abs(pv - cv) > 1e-7:
                    return True
        return False

    def _on_camera_changed(self, _obj: Any, camera: Any) -> None:
        if self._in_frame_update:
            return
        now = time.monotonic()
        if hasattr(camera, "to_plotly_json"):
            try:
                cam_json = camera.to_plotly_json()
            except Exception:
                cam_json = None
        else:
            try:
                cam_json = dict(camera)
            except Exception:
                cam_json = None

        changed = self._camera_changed_materially(self._camera_cache, cam_json)
        if cam_json is not None:
            self._camera_cache = cam_json

        if now < self._ignore_camera_pause_until:
            return
        if changed and self._is_playing() and not self._paused_for_drag:
            self._was_playing_before_drag = True
            self._paused_for_drag = True
            self._set_playing(False)

    def _on_plot_edits_completed(self) -> None:
        if self._paused_for_drag and self._was_playing_before_drag:
            self._set_playing(True)
        self._paused_for_drag = False
        self._was_playing_before_drag = False


class LMSBall3DBackwardTwoSheetWidget(LMSBall3DWidget):
    """LMS widget with native backward-time simulation and outer-sheet traces.

    The additional sheet map is:
      x̄ = (x/|x| - x) / |x/|x| - x|^2,
    interpreted with a fallback direction when |x|≈0.
    """

    def __init__(
        self,
        *,
        outer_radius_display: float = 3.0,
        outer_radius_cap: float = 6.0,
        display_points_cap: int | None = 1200,
        force_backward_time: bool = True,
        **kwargs: Any,
    ) -> None:
        self.outer_radius_display = float(max(1.2, outer_radius_display))
        self.outer_radius_cap = float(max(self.outer_radius_display, outer_radius_cap))
        self.force_backward_time = bool(force_backward_time)
        self._bar_start_idx = -1
        self._last_bar_path_frame = -10**9
        self._last_bar_overlay_frame = -10**9
        self._bar_cache: dict[str, np.ndarray] | None = None
        self._bar_paths_initialized = False
        self._bar_paths_frame_name: Literal["lab", "body"] | None = None
        self._bar_paths_inversion_on: bool | None = None
        self._phase_plane_cache: dict[str, np.ndarray] | None = None
        if "trajectory_mode" not in kwargs:
            kwargs["trajectory_mode"] = "auto"
        super().__init__(display_points_cap=display_points_cap, **kwargs)
        if self.force_backward_time:
            # Keep backward widget defaults distinct via explicit button states
            # rather than hidden dt-sign overrides.
            prev = self._updating
            self._updating = True
            self.toggle_time_direction.value = True
            self.toggle_entropy.value = True
            self._updating = prev
            self._sync_time_direction_button_label()
            self._sync_entropy_button_label()
            self._recompute(reset_frame=False)

    def _path_stride(self) -> int:
        """Backward/two-sheet view needs stronger throttling for base path traces."""
        if not self._is_playing():
            return 1
        return max(8, min(48, int(round(self._playback_speed * 6.0))))

    def _scene_radius_default(self) -> float:
        return max(self.outer_radius_display, self.outer_radius_cap)

    def _scene_radius_inversion(self) -> float:
        return max(self.outer_radius_cap, 8.0)

    @staticmethod
    def _unit_or_none(v: np.ndarray, eps: float = 1e-12) -> np.ndarray | None:
        arr = np.asarray(v, dtype=np.float64).reshape(-1)
        n = float(np.linalg.norm(arr))
        if n < eps:
            return None
        return arr / n

    @classmethod
    def _build_w_plane_basis(
        cls,
        *,
        w_series: np.ndarray,
        z_series: np.ndarray,
        zhat_series: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        eps = 1e-12
        w0 = np.asarray(w_series[0], dtype=np.float64)
        winf = np.asarray(w_series[-1], dtype=np.float64)

        e1 = cls._unit_or_none(winf, eps=eps)
        if e1 is None:
            for cand in (w0, z_series[-1], zhat_series[-1], np.array([1.0, 0.0, 0.0], dtype=np.float64)):
                e1 = cls._unit_or_none(cand, eps=eps)
                if e1 is not None:
                    break
        if e1 is None:
            e1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        seed = cls._unit_or_none(w0, eps=eps)
        if seed is None:
            for cand in (z_series[0], zhat_series[0], np.array([0.0, 1.0, 0.0], dtype=np.float64)):
                seed = cls._unit_or_none(cand, eps=eps)
                if seed is not None:
                    break
        if seed is None:
            seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        e2_raw = seed - float(np.dot(seed, e1)) * e1
        if float(np.linalg.norm(e2_raw)) < eps:
            basis = (
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
                np.array([0.0, 1.0, 0.0], dtype=np.float64),
                np.array([0.0, 0.0, 1.0], dtype=np.float64),
            )
            best = min(basis, key=lambda a: abs(float(np.dot(a, e1))))
            e2_raw = best - float(np.dot(best, e1)) * e1
        e2 = e2_raw / max(float(np.linalg.norm(e2_raw)), eps)
        return e1, e2

    @classmethod
    def _project_series_to_w_plane(
        cls,
        *,
        w_series: np.ndarray,
        z_series: np.ndarray,
        zhat_series: np.ndarray,
    ) -> dict[str, np.ndarray]:
        e1, e2 = cls._build_w_plane_basis(
            w_series=w_series,
            z_series=z_series,
            zhat_series=zhat_series,
        )
        return {
            "basis_e1": e1,
            "basis_e2": e2,
            "w_x": np.asarray(w_series @ e1, dtype=np.float64),
            "w_y": np.asarray(w_series @ e2, dtype=np.float64),
            "z_x": np.asarray(z_series @ e1, dtype=np.float64),
            "z_y": np.asarray(z_series @ e2, dtype=np.float64),
            "Z_x": np.asarray(zhat_series @ e1, dtype=np.float64),
            "Z_y": np.asarray(zhat_series @ e2, dtype=np.float64),
        }

    def _refresh_metric_series(self, params: dict[str, float | int]) -> None:
        # Preserve base rows 1 and 3 behavior, then swap row 2 to basis-plane projections.
        super()._refresh_metric_series(params)
        if not self._traj_cache:
            self._phase_plane_cache = None
            return

        _, z_series, Z_series = self._frame_arrays()
        w_series = np.asarray(self._traj_cache["w"], dtype=np.float64)
        K = max(float(params["K"]), 1e-9)
        zhat_series = np.asarray(Z_series, dtype=np.float64) / K

        plane = self._project_series_to_w_plane(
            w_series=w_series,
            z_series=np.asarray(z_series, dtype=np.float64),
            zhat_series=zhat_series,
        )
        self._phase_plane_cache = plane
        lim = 1.05

        with self.metrics_fig.batch_update():
            # Row 2 traces 4..6 are lines; 7..9 are current-frame markers.
            self.metrics_fig.data[4].x = plane["w_x"].tolist()
            self.metrics_fig.data[4].y = plane["w_y"].tolist()
            self.metrics_fig.data[5].x = plane["z_x"].tolist()
            self.metrics_fig.data[5].y = plane["z_y"].tolist()
            self.metrics_fig.data[6].x = plane["Z_x"].tolist()
            self.metrics_fig.data[6].y = plane["Z_y"].tolist()
            self.metrics_fig.update_xaxes(
                title_text="",
                range=[-lim, lim],
                automargin=False,
                row=2,
                col=1,
            )
            self.metrics_fig.update_yaxes(
                title_text="",
                range=[-lim, lim],
                automargin=False,
                row=2,
                col=1,
            )

    def _update_phase_plane_markers(self, t: int) -> None:
        # Disabled for smoother playback: updating cursor markers every frame
        # is a major FigureWidget sync bottleneck.
        return

    @staticmethod
    def _bar_sheet_map(
        x: np.ndarray,
        *,
        fallback_dir: np.ndarray | None = None,
        eps: float = 1e-9,
    ) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]

        if fallback_dir is None:
            f = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            f = np.asarray(fallback_dir, dtype=np.float64).reshape(3)
        f_norm = float(np.linalg.norm(f))
        if f_norm < eps:
            f = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            f_norm = 1.0
        f = f / f_norm

        r = np.linalg.norm(arr, axis=1, keepdims=True)
        unit = arr / np.maximum(r, eps)
        tiny = (r[:, 0] < eps)
        if np.any(tiny):
            unit[tiny] = f[None, :]
        diff = unit - arr
        den = np.sum(diff * diff, axis=1, keepdims=True)
        out = diff / np.maximum(den, eps)
        return out[0] if single else out

    @staticmethod
    def _bar_sheet_map_rows(
        x: np.ndarray,
        *,
        fallback_rows: np.ndarray | None = None,
        eps: float = 1e-9,
    ) -> np.ndarray:
        """Vectorized outer-sheet map with optional per-row fallback directions."""
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("x must have shape [N,3] or [3].")

        n = int(arr.shape[0])
        r = np.linalg.norm(arr, axis=1, keepdims=True)
        unit = arr / np.maximum(r, eps)
        tiny = r[:, 0] < eps
        if np.any(tiny):
            if fallback_rows is None:
                fb_unit = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (n, 1))
            else:
                fb = np.asarray(fallback_rows, dtype=np.float64)
                if fb.ndim == 1:
                    fb = np.tile(fb[None, :], (n, 1))
                if fb.shape != arr.shape:
                    raise ValueError("fallback_rows must match x shape or be a single [3] vector.")
                fb_n = np.linalg.norm(fb, axis=1, keepdims=True)
                fb_unit = fb / np.maximum(fb_n, eps)
                tiny_fb = fb_n[:, 0] < eps
                if np.any(tiny_fb):
                    fb_unit[tiny_fb] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            unit[tiny] = fb_unit[tiny]

        diff = unit - arr
        den = np.sum(diff * diff, axis=1, keepdims=True)
        return diff / np.maximum(den, eps)

    @staticmethod
    def _radial_cap(x: np.ndarray, max_radius: float, eps: float = 1e-12) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        r = np.linalg.norm(arr, axis=1, keepdims=True)
        s = np.minimum(1.0, float(max_radius) / np.maximum(r, eps))
        out = arr * s
        return out[0] if single else out

    def _simulate(self, params: dict[str, float | int]):
        # Time-direction sign is controlled globally by the dedicated toggle button.
        return super()._simulate(params)

    def _build_outer_sheet_cache(self) -> None:
        if not self._traj_cache:
            self._bar_cache = None
            self._bar_paths_initialized = False
            self._bar_paths_frame_name = None
            self._bar_paths_inversion_on = None
            return
        params = self._params_cache if self._params_cache else self._params()
        K = max(float(params["K"]), 1e-9)

        w_series = np.asarray(self._traj_cache["w"], dtype=np.float64)
        z_lab_series = np.asarray(self._traj_cache["z_lab"], dtype=np.float64)
        z_body_series = np.asarray(self._traj_cache["z_body"], dtype=np.float64)
        Zhat_lab_series = np.asarray(self._traj_cache["Z_lab"], dtype=np.float64) / K
        Zhat_body_series = np.asarray(self._traj_cache["Z_body"], dtype=np.float64) / K

        self._bar_cache = {
            "w": self._radial_cap(
                self._bar_sheet_map_rows(w_series, fallback_rows=w_series),
                self.outer_radius_cap,
            ),
            "z_lab": self._radial_cap(
                self._bar_sheet_map_rows(z_lab_series, fallback_rows=z_lab_series),
                self.outer_radius_cap,
            ),
            "z_body": self._radial_cap(
                self._bar_sheet_map_rows(z_body_series, fallback_rows=z_body_series),
                self.outer_radius_cap,
            ),
            "Zhat_lab": self._radial_cap(
                self._bar_sheet_map_rows(Zhat_lab_series, fallback_rows=Zhat_lab_series),
                self.outer_radius_cap,
            ),
            "Zhat_body": self._radial_cap(
                self._bar_sheet_map_rows(Zhat_body_series, fallback_rows=Zhat_body_series),
                self.outer_radius_cap,
            ),
        }
        self._bar_paths_initialized = False
        self._bar_paths_frame_name = None
        self._bar_paths_inversion_on = None

    def _recompute(self, *, reset_frame: bool) -> None:
        self._last_bar_path_frame = -10**9
        self._last_bar_overlay_frame = -10**9
        self._bar_paths_initialized = False
        self._bar_paths_frame_name = None
        self._bar_paths_inversion_on = None
        super()._recompute(reset_frame=reset_frame)
        self._build_outer_sheet_cache()

    def _build_figures(self) -> None:
        super()._build_figures()
        if len(self.metrics_fig.layout.annotations) >= 2:
            self.metrics_fig.layout.annotations[1].text = (
                "Projected phase plane of w, z, Z/K in basis (w∞, GS(w₀ ⟂ w∞))"
            )
        # Keep w/z/Z color semantics consistent across sphere + metrics.
        self.metrics_fig.data[4].name = "w projection"
        self.metrics_fig.data[4].line.color = "black"
        self.metrics_fig.data[4].showlegend = True
        self.metrics_fig.data[4].legend = "legend4"
        self.metrics_fig.data[5].name = "z projection"
        self.metrics_fig.data[5].line.color = "gray"
        self.metrics_fig.data[5].showlegend = True
        self.metrics_fig.data[5].legend = "legend4"
        self.metrics_fig.data[6].name = "Z/K projection"
        self.metrics_fig.data[6].line.color = "firebrick"
        self.metrics_fig.data[6].showlegend = True
        self.metrics_fig.data[6].legend = "legend4"
        self.metrics_fig.data[7].marker.color = "black"
        self.metrics_fig.data[8].marker.color = "gray"
        self.metrics_fig.data[9].marker.color = "firebrick"
        # Keep all rows full-width in the metrics figure.
        self.metrics_fig.update_layout(
            xaxis=dict(domain=[0.0, 0.72]),
            xaxis2=dict(domain=[0.0, 0.72]),
            xaxis3=dict(domain=[0.0, 0.72]),
            legend4=dict(
                orientation="v",
                x=0.76,
                y=0.66,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.80)",
            ),
        )
        self.metrics_fig.update_yaxes(automargin=False, row=2, col=1)
        self._bar_start_idx = len(self.sphere_fig.data)

        # Outer-sheet markers.
        self.sphere_fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="markers",
                marker=dict(size=self.point_size + 2, color="black", symbol="diamond"),
                name="w̄(t)",
                showlegend=True,
            )
        )
        self.sphere_fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="markers",
                marker=dict(size=self.point_size + 1, color="white", line=dict(color="black", width=2), symbol="diamond-open"),
                name="z̄(t)",
                showlegend=True,
            )
        )
        self.sphere_fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="markers",
                marker=dict(size=self.point_size + 1, color="firebrick", symbol="diamond"),
                name="Z̄/K (outer sheet)",
                showlegend=True,
            )
        )

        # Outer-sheet paths.
        self.sphere_fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="lines",
                line=dict(color="black", width=2, dash="solid"),
                name="w̄ path",
                showlegend=False,
            )
        )
        self.sphere_fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="lines",
                line=dict(color="gray", width=2, dash="solid"),
                name="z̄ path",
                showlegend=False,
            )
        )
        self.sphere_fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="lines",
                line=dict(color="firebrick", width=2, dash="solid"),
                name="Z̄/K path",
                showlegend=False,
            )
        )

        # Outer-sheet vectors.
        self.sphere_fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="lines",
                line=dict(color="black", width=3),
                name="w̄ vector",
                showlegend=False,
            )
        )
        self.sphere_fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="lines",
                line=dict(color="gray", width=3),
                name="z̄ vector",
                showlegend=False,
            )
        )
        self.sphere_fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="lines",
                line=dict(color="firebrick", width=3),
                name="Z̄/K vector",
                showlegend=False,
            )
        )

        self._apply_scene_range()

    def _render_frame(self, t: int) -> None:
        super()._render_frame(t)
        self._update_phase_plane_markers(t)
        if self._steps <= 0 or not self._traj_cache or self._bar_start_idx < 0:
            return

        t = max(0, min(int(t), self._steps))
        if self._bar_cache is None:
            self._build_outer_sheet_cache()
        if self._bar_cache is None:
            return

        frame_name: Literal["lab", "body"] = "body" if self.view_frame_dropdown.value == "body" else "lab"
        wbar_series = self._bar_cache["w"]
        zbar_series = self._bar_cache["z_body"] if frame_name == "body" else self._bar_cache["z_lab"]
        Zbar_series = self._bar_cache["Zhat_body"] if frame_name == "body" else self._bar_cache["Zhat_lab"]

        idx = self._bar_start_idx
        show_paths = bool(self.show_paths.value)
        show_vectors = bool(self.show_vectors.value)
        inversion_on = bool(self._inversion_enabled)
        path_refresh = show_paths and (
            (not self._bar_paths_initialized)
            or (self._bar_paths_frame_name != frame_name)
            or (self._bar_paths_inversion_on != inversion_on)
        )
        inv_ctx = self._inversion_context(frame_name=frame_name) if inversion_on else None

        # Markers are updated per-frame; outer paths are refreshed from cache only when needed.
        wb = np.asarray(wbar_series[t], dtype=np.float64)
        zb = np.asarray(zbar_series[t], dtype=np.float64)
        Zb = np.asarray(Zbar_series[t], dtype=np.float64)
        wb_disp = self._maybe_invert_rows(wb, frame_name=frame_name, inv_ctx=inv_ctx)
        zb_disp = self._maybe_invert_rows(zb, frame_name=frame_name, inv_ctx=inv_ctx)
        Zb_disp = self._maybe_invert_rows(Zb, frame_name=frame_name, inv_ctx=inv_ctx)
        self._in_frame_update = True
        try:
            with self.sphere_fig.batch_update():
                # markers
                self.sphere_fig.data[idx + 0].x = [float(wb_disp[0])]
                self.sphere_fig.data[idx + 0].y = [float(wb_disp[1])]
                self.sphere_fig.data[idx + 0].z = [float(wb_disp[2])]

                self.sphere_fig.data[idx + 1].x = [float(zb_disp[0])]
                self.sphere_fig.data[idx + 1].y = [float(zb_disp[1])]
                self.sphere_fig.data[idx + 1].z = [float(zb_disp[2])]

                self.sphere_fig.data[idx + 2].x = [float(Zb_disp[0])]
                self.sphere_fig.data[idx + 2].y = [float(Zb_disp[1])]
                self.sphere_fig.data[idx + 2].z = [float(Zb_disp[2])]

                if path_refresh:
                    # Precomputed full trajectories: no per-frame path growth payload.
                    wbp = wbar_series
                    zbp = zbar_series
                    Zbp = Zbar_series
                    wbp_disp = self._maybe_invert_rows(wbp, frame_name=frame_name, inv_ctx=inv_ctx)
                    zbp_disp = self._maybe_invert_rows(zbp, frame_name=frame_name, inv_ctx=inv_ctx)
                    Zbp_disp = self._maybe_invert_rows(Zbp, frame_name=frame_name, inv_ctx=inv_ctx)

                    self.sphere_fig.data[idx + 3].x = wbp_disp[:, 0].tolist()
                    self.sphere_fig.data[idx + 3].y = wbp_disp[:, 1].tolist()
                    self.sphere_fig.data[idx + 3].z = wbp_disp[:, 2].tolist()

                    self.sphere_fig.data[idx + 4].x = zbp_disp[:, 0].tolist()
                    self.sphere_fig.data[idx + 4].y = zbp_disp[:, 1].tolist()
                    self.sphere_fig.data[idx + 4].z = zbp_disp[:, 2].tolist()

                    self.sphere_fig.data[idx + 5].x = Zbp_disp[:, 0].tolist()
                    self.sphere_fig.data[idx + 5].y = Zbp_disp[:, 1].tolist()
                    self.sphere_fig.data[idx + 5].z = Zbp_disp[:, 2].tolist()

                if show_vectors:
                    self.sphere_fig.data[idx + 6].x = [0.0, float(wb_disp[0])]
                    self.sphere_fig.data[idx + 6].y = [0.0, float(wb_disp[1])]
                    self.sphere_fig.data[idx + 6].z = [0.0, float(wb_disp[2])]

                    self.sphere_fig.data[idx + 7].x = [0.0, float(zb_disp[0])]
                    self.sphere_fig.data[idx + 7].y = [0.0, float(zb_disp[1])]
                    self.sphere_fig.data[idx + 7].z = [0.0, float(zb_disp[2])]

                    self.sphere_fig.data[idx + 8].x = [0.0, float(Zb_disp[0])]
                    self.sphere_fig.data[idx + 8].y = [0.0, float(Zb_disp[1])]
                    self.sphere_fig.data[idx + 8].z = [0.0, float(Zb_disp[2])]

                self.sphere_fig.data[idx + 3].visible = show_paths
                self.sphere_fig.data[idx + 4].visible = show_paths
                self.sphere_fig.data[idx + 5].visible = show_paths
                self.sphere_fig.data[idx + 6].visible = show_vectors
                self.sphere_fig.data[idx + 7].visible = show_vectors
                self.sphere_fig.data[idx + 8].visible = show_vectors
        finally:
            self._in_frame_update = False
        if path_refresh:
            self._bar_paths_initialized = True
            self._bar_paths_frame_name = frame_name
            self._bar_paths_inversion_on = inversion_on
            self._last_bar_path_frame = t
        self._last_bar_overlay_frame = t


class LMSBall3DHydrodynamicEnsembleWidget(LMSBall3DWidget):
    """High-N ensemble widget comparing active high/low mode plus poisson.

    This class approximates the hydrodynamic (large-N) regime by running three
    large finite-N reduced simulations and plotting ensemble diagnostics
    simultaneously. The sphere panel shows one selected ensemble at a time.
    """

    def __init__(
        self,
        *,
        controls: tuple[LMS3DControlSpec, ...] = HYDRO_DEFAULT_CONTROLS,
        init_metric_mode: InitMetricMode = "entropy",
        display_points_cap: int = 1400,
        **kwargs: Any,
    ) -> None:
        canonical_mode = self._canonical_init_metric_mode(init_metric_mode)
        self.display_points_cap = int(max(150, display_points_cap))
        self._ensemble_modes: tuple[ActiveInitMode, ...] = self._active_modes_for_init_metric_mode(canonical_mode)
        self._mode_colors: dict[str, str] = {
            "entropy_high": "royalblue",
            "entropy_low": "seagreen",
            "var_perp_high": "royalblue",
            "var_perp_low": "seagreen",
            "poisson": "firebrick",
        }
        self._display_mode: ActiveInitMode = self._ensemble_modes[0]
        self._ensemble_state: dict[str, dict[str, Any]] = {}
        self._ensemble_metrics: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        self._ensemble_runtime: dict[str, float] = {}
        self._display_indices: np.ndarray | None = None
        self._metric_trace_index: dict[str, dict[str, int]] = {}
        self._async_lock = threading.Lock()
        self._async_pending_job: dict[str, Any] | None = None
        self._async_worker: threading.Thread | None = None
        self._async_seq = 0
        self._async_cancel_before = 0
        super().__init__(
            controls=controls,
            trajectory_mode="memory",
            thermo_mode="recompute",
            init_metric_mode=canonical_mode,
            display_points_cap=self.display_points_cap,
            **kwargs,
        )

    def _build_controls(self) -> None:
        super()._build_controls()
        mode_hi, mode_lo, mode_poi = self._ensemble_modes
        self.ensemble_dropdown = widgets.Dropdown(
            options=[
                (f"Display: {self._init_mode_label(mode_hi)}", mode_hi),
                (f"Display: {self._init_mode_label(mode_lo)}", mode_lo),
                (f"Display: {self._init_mode_label(mode_poi)}", mode_poi),
            ],
            value=self._display_mode,
            description="Displayed initialization",
            layout=widgets.Layout(width=self._control_width),
            style={"description_width": "initial"},
        )
        children = list(self.controls_box.children)
        children.insert(3, widgets.HBox([self.ensemble_dropdown], layout=widgets.Layout(width=self._control_width)))
        self.controls_box.children = tuple(children)

        # This widget computes all three modes in one recompute; the old init button
        # is repurposed to choose which ensemble is displayed on the sphere panel.
        self.toggle_init_state.disabled = False
        self.toggle_init_state.layout.display = ""
        self._sync_init_state_button_label()
        self.btn_recompute.description = "Recompute all ensembles"

    def _build_figures(self) -> None:
        super()._build_figures()
        mode_hi, mode_lo, mode_poi = self._ensemble_modes

        self.metrics_fig = go.FigureWidget(
            make_subplots(
                rows=3,
                cols=1,
                subplot_titles=(
                    (
                        "Magnitudes: |w| and |Z|/K "
                        f"({self._init_mode_label(mode_hi).lower()}/"
                        f"{self._init_mode_label(mode_lo).lower()}/"
                        f"{self._init_mode_label(mode_poi).lower()})"
                    ),
                    f"{self._primary_metric_title()} and rate {self._primary_rate_series_name()} (all ensembles)",
                    "Variance vs final-axis hyperplane: total / perp / aligned",
                ),
                vertical_spacing=0.12,
            )
        )
        self._metric_trace_index = {}
        for mode in self._ensemble_modes:
            color = self._mode_colors[mode]
            mode_tag = self._init_mode_short_tag(mode)
            self._metric_trace_index[mode] = {}

            # Row 1
            self._metric_trace_index[mode]["w_norm"] = len(self.metrics_fig.data)
            self.metrics_fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=f"{mode_tag}: |w|",
                    legend="legend",
                ),
                row=1,
                col=1,
            )
            self._metric_trace_index[mode]["z_norm"] = len(self.metrics_fig.data)
            self.metrics_fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode="lines",
                    line=dict(color=color, width=2, dash="dot"),
                    name=f"{mode_tag}: |Z|/K",
                    legend="legend",
                ),
                row=1,
                col=1,
            )

            # Row 2
            self._metric_trace_index[mode]["entropy"] = len(self.metrics_fig.data)
            self.metrics_fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=f"{mode_tag}: {self._primary_metric_series_name()}",
                    legend="legend2",
                ),
                row=2,
                col=1,
            )
            self._metric_trace_index[mode]["entropy_rate"] = len(self.metrics_fig.data)
            self.metrics_fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode="lines",
                    line=dict(color=color, width=2, dash="dash"),
                    name=f"{mode_tag}: {self._primary_rate_series_name()}",
                    legend="legend2",
                ),
                row=2,
                col=1,
            )

            # Row 3
            self._metric_trace_index[mode]["var_total"] = len(self.metrics_fig.data)
            self.metrics_fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=f"{mode_tag}: var total",
                    legend="legend3",
                ),
                row=3,
                col=1,
            )
            self._metric_trace_index[mode]["var_perp"] = len(self.metrics_fig.data)
            self.metrics_fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode="lines",
                    line=dict(color=color, width=2, dash="dash"),
                    name=f"{mode_tag}: var perp",
                    legend="legend3",
                ),
                row=3,
                col=1,
            )
            self._metric_trace_index[mode]["var_aligned"] = len(self.metrics_fig.data)
            self.metrics_fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode="lines",
                    line=dict(color=color, width=2, dash="dot"),
                    name=f"{mode_tag}: var aligned",
                    legend="legend3",
                ),
                row=3,
                col=1,
            )

        self.metrics_fig.update_layout(
            width=980,
            height=760,
            margin=dict(l=24, r=24, t=96, b=80),
            template="plotly_white",
            showlegend=True,
            legend=dict(
                orientation="v",
                x=0.70,
                y=0.98,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.82)",
            ),
            legend2=dict(
                orientation="v",
                x=0.70,
                y=0.64,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.82)",
            ),
            legend3=dict(
                orientation="v",
                x=0.70,
                y=0.30,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.82)",
            ),
            xaxis=dict(domain=[0.0, 0.66]),
            xaxis2=dict(domain=[0.0, 0.66]),
            xaxis3=dict(domain=[0.0, 0.66]),
        )
        self.metrics_fig.update_xaxes(title_text="", row=1, col=1)
        self.metrics_fig.update_xaxes(title_text="", row=2, col=1)
        self.metrics_fig.update_xaxes(title_text="frame", row=3, col=1)

    def _bind_events(self) -> None:
        super()._bind_events()
        self.ensemble_dropdown.observe(self._on_ensemble_display_change, names="value")

    def _schedule_on_main_thread(self, fn: Callable[[], None]) -> None:
        try:
            from IPython import get_ipython  # type: ignore

            ip = get_ipython()
            kernel = getattr(ip, "kernel", None)
            io_loop = getattr(kernel, "io_loop", None)
            if io_loop is not None:
                io_loop.add_callback(fn)
                return
        except Exception:
            pass
        fn()

    def _capture_async_recompute_job(self, *, reset_frame: bool) -> dict[str, Any]:
        params = dict(self._params())
        target_mode = self._coerce_mode_to_active_family(
            str(getattr(self.ensemble_dropdown, "value", self._display_mode))
        )
        return {
            "params": params,
            "prev_frame": int(self.frame_slider.value),
            "reset_frame": bool(reset_frame),
            "target_mode": target_mode,
            "entropy_increase": bool(self.toggle_entropy.value),
            "time_backward": bool(self.toggle_time_direction.value),
            "w_mode": str(self.mode_dropdown.value),
        }

    def _is_async_cancelled(self, seq: int) -> bool:
        with self._async_lock:
            return int(seq) <= int(self._async_cancel_before)

    def _queue_async_recompute(self, *, reset_frame: bool) -> None:
        job = self._capture_async_recompute_job(reset_frame=reset_frame)
        with self._async_lock:
            self._async_seq += 1
            seq = int(self._async_seq)
            job["seq"] = seq
            self._async_pending_job = job
            # Any older running/pending job is now stale.
            self._async_cancel_before = seq - 1
            worker_alive = self._async_worker is not None and self._async_worker.is_alive()
            if worker_alive:
                return
            self._async_worker = threading.Thread(
                target=self._async_recompute_worker_loop,
                name="hydro-recompute-worker",
                daemon=True,
            )
            self._async_worker.start()

    def _async_recompute_worker_loop(self) -> None:
        while True:
            with self._async_lock:
                job = self._async_pending_job
                self._async_pending_job = None
            if job is None:
                return
            seq = int(job["seq"])
            try:
                result = self._compute_hydro_job_result(job=job, seq=seq)
            except _HydroRecomputeCancelled:
                continue
            except Exception as exc:
                def _report_error(err_text: str = str(exc)) -> None:
                    self.stats_html.value = (
                        "<b>Hydrodynamic Ensemble Stats</b>"
                        "<div style='margin-top:6px;color:#b00020;font-family:monospace'>"
                        f"recompute error: {err_text}</div>"
                    )

                self._schedule_on_main_thread(_report_error)
                continue
            if self._is_async_cancelled(seq):
                continue

            def _apply(seq_local: int = seq, job_local: dict[str, Any] = job, result_local: dict[str, Any] = result) -> None:
                if self._is_async_cancelled(seq_local):
                    return
                self._apply_hydro_job_result(job=job_local, result=result_local, seq=seq_local)

            self._schedule_on_main_thread(_apply)

    def _on_control_change(self, _change: dict[str, Any]) -> None:
        if self._updating:
            return
        self._queue_async_recompute(reset_frame=False)

    def _on_time_direction_change(self, change: dict[str, Any]) -> None:
        if self._updating or change.get("name") != "value":
            return
        self._sync_time_direction_button_label()
        self._queue_async_recompute(reset_frame=False)

    def _on_entropy_direction_change(self, change: dict[str, Any]) -> None:
        if self._updating or change.get("name") != "value":
            return
        self._sync_entropy_button_label()
        self._queue_async_recompute(reset_frame=False)

    def _on_recompute_clicked(self, _btn: widgets.Button) -> None:
        self._queue_async_recompute(reset_frame=False)

    def _sync_init_state_button_label(self) -> None:
        if self._display_mode == self._ensemble_modes[1]:
            self.toggle_init_state.description = f"Displayed: {self._init_mode_label(self._display_mode)}"
            self.toggle_init_state.button_style = "success"
        elif self._display_mode == "poisson":
            self.toggle_init_state.description = "Displayed: Poisson"
            self.toggle_init_state.button_style = "info"
        else:
            self.toggle_init_state.description = f"Displayed: {self._init_mode_label(self._display_mode)}"
            self.toggle_init_state.button_style = ""

    def _on_init_state_clicked(self, _btn: widgets.Button) -> None:
        if self._updating:
            return
        order = self._ensemble_modes
        cur = self._display_mode
        i = order.index(cur)
        nxt = order[(i + 1) % len(order)]
        if hasattr(self, "ensemble_dropdown"):
            self.ensemble_dropdown.value = nxt
        else:
            self._select_display_mode(nxt)
            self._sync_init_state_button_label()
            self._render_frame(int(self.frame_slider.value))

    def _on_ensemble_display_change(self, change: dict[str, Any]) -> None:
        if self._updating or change.get("name") != "value":
            return
        mode = self._coerce_mode_to_active_family(str(change.get("new", self._display_mode)))
        if mode not in self._ensemble_state:
            return
        self._select_display_mode(mode)
        self._sync_init_state_button_label()
        self._render_frame(int(self.frame_slider.value))

    def _make_fast_initial_boundary_points(
        self,
        *,
        n: int,
        d: int,
        w_az: float,
        w_el: float,
        target_r: float,
        init_state: OptimizedInitMode,
    ) -> torch.Tensor:
        """Random-start entropy-gradient initializer (no ad-hoc clustered templates)."""
        return self._make_initial_boundary_points(
            n=n,
            d=d,
            w_az=w_az,
            w_el=w_el,
            target_r=target_r,
            init_state=init_state,
        )

    def _simulate_mode(
        self,
        mode: ActiveInitMode,
        params: dict[str, float | int],
        *,
        entropy_increase: bool | None = None,
        time_backward: bool | None = None,
        w_mode: str | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ):
        d = 3
        n = int(params["N"])
        K = float(params["K"])
        entropy_flag = bool(self.toggle_entropy.value) if entropy_increase is None else bool(entropy_increase)
        conformal_sign = -1.0 if entropy_flag else 1.0
        omega = float(params["omega"])
        r0 = float(params["r0"])
        w_az = float(params["w_az"])
        w_el = float(params["w_el"])
        ax_az = float(params["ax_az"])
        ax_el = float(params["ax_el"])
        dt = self._effective_dt(params, time_backward=time_backward)
        steps = int(params["steps"])

        weights = torch.ones(n, dtype=torch.float64) / float(n)
        center_dir = torch.tensor(_angles_to_unit(w_az, w_el), dtype=torch.float64)
        if mode == "poisson":
            base_points = random_points_on_sphere(
                n,
                d=d,
                generator=self._torch_gen,
                dtype=torch.float64,
            )
            r0_clip = float(np.clip(r0, 0.0, 0.999999))
            # Keep sign convention consistent with optimized initializers.
            w0 = -center_dir * r0_clip
        else:
            x0_points = self._make_fast_initial_boundary_points(
                n=n,
                d=d,
                w_az=w_az,
                w_el=w_el,
                target_r=float(r0),
                init_state=mode,
            )
            w0 = self._estimate_w_from_boundary_points(
                points=x0_points,
                weights=weights,
                d=d,
                fallback_dir=center_dir,
            )
            base_points = self._recover_base_points_from_state(
                x_points=x0_points,
                w0=w0,
            )

        zeta0 = torch.eye(d, dtype=torch.float64)
        axis = torch.tensor(_angles_to_unit(ax_az, ax_el), dtype=torch.float64)
        A = skew_symmetric_from_axis(axis, rate=omega).to(dtype=torch.float64)

        return integrate_lms_reduced_euler(
            w0=w0,
            zeta0=zeta0,
            base_points=base_points,
            weights=weights,
            A=A,
            coupling=conformal_sign * K,
            dt=dt,
            steps=steps,
            w_mode=str(self.mode_dropdown.value) if w_mode is None else str(w_mode),
            project_rotation=True,
            store_points="none",
            store_dtype=torch.float32,
            preallocate=True,
            cancel_check=cancel_check,
        )

    @staticmethod
    def _reconstruct_points_from_cache(
        *,
        traj_cache: dict[str, np.ndarray],
        base_points: np.ndarray,
        t: int,
        frame_name: Literal["lab", "body"],
    ) -> np.ndarray:
        w_t = np.asarray(traj_cache["w"][t], dtype=np.float64)
        zeta_t = np.asarray(traj_cache["zeta"][t], dtype=np.float64)
        diff = np.asarray(base_points, dtype=np.float64) - w_t[None, :]
        den = np.einsum("ij,ij->i", diff, diff)[:, None]
        w2 = float(np.dot(w_t, w_t))
        x_body = ((1.0 - w2) / np.maximum(den, 1e-12)) * diff - w_t[None, :]
        norms = np.linalg.norm(x_body, axis=1, keepdims=True)
        x_body = x_body / np.maximum(norms, 1e-12)
        if frame_name == "body":
            return x_body
        return x_body @ zeta_t.T

    def _compute_mode_metrics(
        self,
        *,
        traj_cache: dict[str, np.ndarray],
        base_points: np.ndarray,
        params: dict[str, float | int],
        frame_name: Literal["lab", "body"],
        sample_idx: np.ndarray | None,
        time_backward: bool | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, np.ndarray]:
        z_series = traj_cache["z_body"] if frame_name == "body" else traj_cache["z_lab"]
        Z_series = traj_cache["Z_body"] if frame_name == "body" else traj_cache["Z_lab"]
        w_series = traj_cache["w"]
        t_count = z_series.shape[0]
        K = max(float(params["K"]), 1e-9)

        w_norm = np.linalg.norm(w_series, axis=1)
        z_norm = np.linalg.norm(Z_series, axis=1) / K
        entropy = np.zeros(t_count, dtype=np.float64)
        var_total = np.zeros(t_count, dtype=np.float64)
        var_perp = np.zeros(t_count, dtype=np.float64)
        var_aligned = np.zeros(t_count, dtype=np.float64)
        entropy_kappa = 14.0

        axis_final = np.asarray(z_series[-1], dtype=np.float64)
        axis_final_n = float(np.linalg.norm(axis_final))
        if axis_final_n < 1e-12:
            axis_final = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            axis_final_n = 1.0
        axis_final_u = axis_final / max(axis_final_n, 1e-12)

        for t in range(t_count):
            if cancel_check is not None and bool(cancel_check()):
                raise _HydroRecomputeCancelled("Hydro metrics computation cancelled.")
            pts = self._reconstruct_points_from_cache(
                traj_cache=traj_cache,
                base_points=base_points,
                t=t,
                frame_name=frame_name,
            )
            if sample_idx is not None:
                pts = pts[sample_idx]

            mu = pts.mean(axis=0, keepdims=True)
            centered = pts - mu

            proj_final = centered @ axis_final_u
            perp_final = centered - proj_final[:, None] * axis_final_u[None, :]
            var_aligned[t] = float(np.mean(proj_final**2))
            var_perp[t] = float(np.mean(np.sum(perp_final * perp_final, axis=1)))
            var_total[t] = var_aligned[t] + var_perp[t]

            if self.init_metric_mode == "perp_variance":
                axis_dyn = np.asarray(z_series[t], dtype=np.float64)
                axis_dyn_n = float(np.linalg.norm(axis_dyn))
                if axis_dyn_n < 1e-12:
                    mu_vec = mu[0]
                    mu_n = float(np.linalg.norm(mu_vec))
                    if mu_n > 1e-12:
                        axis_dyn = mu_vec / mu_n
                        axis_dyn_n = 1.0
                    else:
                        axis_dyn = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                        axis_dyn_n = 1.0
                axis_u = axis_dyn / max(axis_dyn_n, 1e-12)
                proj_dyn = pts @ axis_u
                perp_dyn = pts - proj_dyn[:, None] * axis_u[None, :]
                perp_mean = perp_dyn.mean(axis=0, keepdims=True)
                perp_dyn_var = float(np.mean(np.sum((perp_dyn - perp_mean) ** 2, axis=1)))
                entropy[t] = float(np.clip(perp_dyn_var, 0.0, 1.0))
            else:
                entropy[t] = self._kernel_entropy_proxy_numpy(
                    points=pts,
                    kappa=entropy_kappa,
                    sample_size=min(int(pts.shape[0]), 420),
                )

        dt_eff = self._effective_dt(params, time_backward=time_backward)
        if abs(dt_eff) < 1e-12:
            dt_eff = 1.0
        entropy_rate = np.gradient(entropy, dt_eff)
        return {
            "w_norm": w_norm,
            "z_norm": z_norm,
            "entropy": entropy,
            "entropy_rate": np.asarray(entropy_rate, dtype=np.float64),
            "var_total": var_total,
            "var_perp": var_perp,
            "var_aligned": var_aligned,
        }

    def _compute_hydro_job_result(self, *, job: dict[str, Any], seq: int) -> dict[str, Any]:
        params = dict(job["params"])
        entropy_increase = bool(job["entropy_increase"])
        time_backward = bool(job["time_backward"])
        w_mode = str(job["w_mode"])
        steps = int(params["steps"])

        ensemble_state: dict[str, dict[str, Any]] = {}
        ensemble_metrics: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        ensemble_runtime: dict[str, float] = {}

        def _cancel_check() -> bool:
            return self._is_async_cancelled(seq)

        for mode in self._ensemble_modes:
            if _cancel_check():
                raise _HydroRecomputeCancelled("Hydro recompute cancelled before simulation.")
            t0 = time.perf_counter()
            traj = self._simulate_mode(
                mode,
                params,
                entropy_increase=entropy_increase,
                time_backward=time_backward,
                w_mode=w_mode,
                cancel_check=_cancel_check,
            )
            elapsed = float(time.perf_counter() - t0)
            ensemble_runtime[mode] = elapsed
            if _cancel_check():
                raise _HydroRecomputeCancelled("Hydro recompute cancelled after simulation.")

            base_points_np = np.ascontiguousarray(
                np.asarray(traj.base_points.detach().cpu().numpy(), dtype=np.float32)
            )
            traj_cache = {
                "w": np.ascontiguousarray(np.asarray(traj.w.detach().cpu().numpy(), dtype=np.float32)),
                "zeta": np.ascontiguousarray(np.asarray(traj.zeta.detach().cpu().numpy(), dtype=np.float32)),
                "z_lab": np.ascontiguousarray(np.asarray(traj.z.detach().cpu().numpy(), dtype=np.float32)),
                "z_body": np.ascontiguousarray(np.asarray((-traj.w).detach().cpu().numpy(), dtype=np.float32)),
                "Z_lab": np.ascontiguousarray(np.asarray(traj.Z.detach().cpu().numpy(), dtype=np.float32)),
                "Z_body": np.ascontiguousarray(np.asarray(traj.Z_body.detach().cpu().numpy(), dtype=np.float32)),
            }

            n_points = int(base_points_np.shape[0])
            if n_points <= self.display_points_cap:
                display_idx = np.arange(n_points, dtype=np.int32)
            else:
                stride = max(1, int(math.ceil(n_points / float(self.display_points_cap))))
                display_idx = np.arange(0, n_points, stride, dtype=np.int32)[: self.display_points_cap]

            metric_cap = min(n_points, max(2400, self.display_points_cap * 2))
            metric_idx: np.ndarray | None
            if n_points <= metric_cap:
                metric_idx = None
            else:
                stride_m = max(1, int(math.ceil(n_points / float(metric_cap))))
                metric_idx = np.arange(0, n_points, stride_m, dtype=np.int32)[:metric_cap]

            metrics_lab = self._compute_mode_metrics(
                traj_cache=traj_cache,
                base_points=base_points_np,
                params=params,
                frame_name="lab",
                sample_idx=metric_idx,
                time_backward=time_backward,
                cancel_check=_cancel_check,
            )
            metrics_body = self._compute_mode_metrics(
                traj_cache=traj_cache,
                base_points=base_points_np,
                params=params,
                frame_name="body",
                sample_idx=metric_idx,
                time_backward=time_backward,
                cancel_check=_cancel_check,
            )

            ensemble_state[mode] = {
                "traj": traj_cache,
                "base_points": base_points_np,
                "display_idx": display_idx,
            }
            ensemble_metrics[mode] = {"lab": metrics_lab, "body": metrics_body}
            del traj

        return {
            "params": params,
            "steps": steps,
            "ensemble_state": ensemble_state,
            "ensemble_metrics": ensemble_metrics,
            "ensemble_runtime": ensemble_runtime,
        }

    def _apply_hydro_job_result(self, *, job: dict[str, Any], result: dict[str, Any], seq: int) -> None:
        if self._is_async_cancelled(seq):
            return
        params = dict(result["params"])
        self._params_cache = dict(params)
        self._ensemble_state = result["ensemble_state"]
        self._ensemble_metrics = result["ensemble_metrics"]
        self._ensemble_runtime = result["ensemble_runtime"]
        self._steps = int(result["steps"])

        prev_frame = int(job["prev_frame"])
        reset_frame = bool(job["reset_frame"])
        target_mode = self._coerce_mode_to_active_family(str(job["target_mode"]))

        self._last_overlay_frame = -10**9
        self._last_path_frame = -10**9

        frame_target = 0 if reset_frame else max(0, min(prev_frame, self._steps))
        self._updating = True
        try:
            self.frame_slider.max = self._steps
            self.play.max = self._steps
            self.frame_slider.value = frame_target
            self.play.value = frame_target
            if target_mode not in self._ensemble_state:
                target_mode = self._ensemble_modes[0]
                self.ensemble_dropdown.value = target_mode
        finally:
            self._updating = False

        if target_mode not in self._ensemble_state:
            target_mode = self._ensemble_modes[0]
        self._select_display_mode(target_mode)
        self._refresh_metric_series(params)
        self._render_frame(int(self.frame_slider.value))

    def _select_display_mode(self, mode: str) -> None:
        mode = self._coerce_mode_to_active_family(mode)
        if mode not in self._ensemble_state:
            return
        self._display_mode = mode  # type: ignore[assignment]
        self._init_state_mode = self._display_mode
        bundle = self._ensemble_state[mode]
        self._traj_cache = bundle["traj"]
        self._base_points_np = bundle["base_points"]
        self._display_indices = bundle["display_idx"]
        frame_name: Literal["lab", "body"] = "body" if self.view_frame_dropdown.value == "body" else "lab"
        self._metric_cache = self._ensemble_metrics[mode][frame_name]

        idx = self._wire_count
        color = self._mode_colors.get(mode, "royalblue")
        with self.sphere_fig.batch_update():
            self.sphere_fig.data[idx + 0].marker.color = color
        self._sync_init_state_button_label()

    def _refresh_metric_series(self, params: dict[str, float | int]) -> None:
        if not self._ensemble_metrics:
            return
        frame_name: Literal["lab", "body"] = "body" if self.view_frame_dropdown.value == "body" else "lab"
        t_axis = np.arange(self._steps + 1).tolist()
        with self.metrics_fig.batch_update():
            for mode in self._ensemble_modes:
                metric = self._ensemble_metrics[mode][frame_name]
                idx_map = self._metric_trace_index[mode]
                for key in ("w_norm", "z_norm", "entropy", "entropy_rate", "var_total", "var_perp", "var_aligned"):
                    i = idx_map[key]
                    self.metrics_fig.data[i].x = t_axis
                    self.metrics_fig.data[i].y = metric[key].tolist()
            self.metrics_fig.update_xaxes(range=[0, int(params["steps"])], row=1, col=1)
            self.metrics_fig.update_xaxes(range=[0, int(params["steps"])], row=2, col=1)
            self.metrics_fig.update_xaxes(range=[0, int(params["steps"])], row=3, col=1)
        self._metric_cache = self._ensemble_metrics[self._display_mode][frame_name]

    def _recompute(self, *, reset_frame: bool) -> None:
        params = self._params()
        self._params_cache = dict(params)
        prev_frame = int(self.frame_slider.value)
        self._ensemble_state = {}
        self._ensemble_metrics = {}
        self._ensemble_runtime = {}
        self._steps = int(params["steps"])

        for mode in self._ensemble_modes:
            t0 = time.perf_counter()
            traj = self._simulate_mode(mode, params)
            elapsed = float(time.perf_counter() - t0)
            self._ensemble_runtime[mode] = elapsed

            base_points_np = np.ascontiguousarray(
                np.asarray(traj.base_points.detach().cpu().numpy(), dtype=np.float32)
            )
            traj_cache = {
                "w": np.ascontiguousarray(np.asarray(traj.w.detach().cpu().numpy(), dtype=np.float32)),
                "zeta": np.ascontiguousarray(np.asarray(traj.zeta.detach().cpu().numpy(), dtype=np.float32)),
                "z_lab": np.ascontiguousarray(np.asarray(traj.z.detach().cpu().numpy(), dtype=np.float32)),
                "z_body": np.ascontiguousarray(np.asarray((-traj.w).detach().cpu().numpy(), dtype=np.float32)),
                "Z_lab": np.ascontiguousarray(np.asarray(traj.Z.detach().cpu().numpy(), dtype=np.float32)),
                "Z_body": np.ascontiguousarray(np.asarray(traj.Z_body.detach().cpu().numpy(), dtype=np.float32)),
            }

            n_points = int(base_points_np.shape[0])
            if n_points <= self.display_points_cap:
                display_idx = np.arange(n_points, dtype=np.int32)
            else:
                stride = max(1, int(math.ceil(n_points / float(self.display_points_cap))))
                display_idx = np.arange(0, n_points, stride, dtype=np.int32)[: self.display_points_cap]

            metric_cap = min(n_points, max(2400, self.display_points_cap * 2))
            metric_idx: np.ndarray | None
            if n_points <= metric_cap:
                metric_idx = None
            else:
                stride_m = max(1, int(math.ceil(n_points / float(metric_cap))))
                metric_idx = np.arange(0, n_points, stride_m, dtype=np.int32)[:metric_cap]

            metrics_lab = self._compute_mode_metrics(
                traj_cache=traj_cache,
                base_points=base_points_np,
                params=params,
                frame_name="lab",
                sample_idx=metric_idx,
            )
            metrics_body = self._compute_mode_metrics(
                traj_cache=traj_cache,
                base_points=base_points_np,
                params=params,
                frame_name="body",
                sample_idx=metric_idx,
            )
            self._ensemble_state[mode] = {
                "traj": traj_cache,
                "base_points": base_points_np,
                "display_idx": display_idx,
            }
            self._ensemble_metrics[mode] = {"lab": metrics_lab, "body": metrics_body}
            del traj

        self._last_overlay_frame = -10**9
        self._last_path_frame = -10**9

        frame_target = 0 if reset_frame else max(0, min(prev_frame, self._steps))
        self._updating = True
        try:
            self.frame_slider.max = self._steps
            self.play.max = self._steps
            self.frame_slider.value = frame_target
            self.play.value = frame_target
        finally:
            self._updating = False

        target_mode = self._coerce_mode_to_active_family(str(getattr(self.ensemble_dropdown, "value", self._display_mode)))
        if target_mode not in self._ensemble_state:
            target_mode = self._ensemble_modes[0]
            self.ensemble_dropdown.value = target_mode
        self._select_display_mode(target_mode)
        self._refresh_metric_series(params)
        self._render_frame(int(self.frame_slider.value))

    def _render_frame(self, t: int) -> None:
        if self._steps <= 0 or not self._traj_cache:
            return
        params = self._params_cache if self._params_cache else self._params()
        t = max(0, min(t, self._steps))

        frame_name: Literal["lab", "body"] = "body" if self.view_frame_dropdown.value == "body" else "lab"
        x_series, z_series, Z_series = self._frame_arrays()
        x = self._points_at_frame(t, x_series, frame_name=frame_name)
        if self._display_indices is not None and len(self._display_indices) < x.shape[0]:
            x_plot = x[self._display_indices]
        else:
            x_plot = x

        w = self._traj_cache["w"][t]
        z = z_series[t]
        Z = Z_series[t]
        K = max(float(params["K"]), 1e-9)
        Z_hat = Z / K
        inv_ctx = self._inversion_context(frame_name=frame_name) if bool(self._inversion_enabled) else None
        x_plot_disp = self._maybe_invert_rows(x_plot, frame_name=frame_name, inv_ctx=inv_ctx)
        w_disp = self._maybe_invert_rows(w, frame_name=frame_name, inv_ctx=inv_ctx)
        z_disp = self._maybe_invert_rows(z, frame_name=frame_name, inv_ctx=inv_ctx)
        Z_hat_disp = self._maybe_invert_rows(Z_hat, frame_name=frame_name, inv_ctx=inv_ctx)

        show_paths = bool(self.show_paths.value)
        show_vectors = bool(self.show_vectors.value)
        force_update = (t == 0) or (t == self._steps)
        path_update = force_update or (abs(t - self._last_path_frame) >= self._path_stride())

        idx = self._wire_count
        self._in_frame_update = True
        try:
            with self.sphere_fig.batch_update():
                self.sphere_fig.data[idx + 0].x = x_plot_disp[:, 0].tolist()
                self.sphere_fig.data[idx + 0].y = x_plot_disp[:, 1].tolist()
                self.sphere_fig.data[idx + 0].z = x_plot_disp[:, 2].tolist()

                self.sphere_fig.data[idx + 1].x = [float(w_disp[0])]
                self.sphere_fig.data[idx + 1].y = [float(w_disp[1])]
                self.sphere_fig.data[idx + 1].z = [float(w_disp[2])]

                self.sphere_fig.data[idx + 2].x = [float(z_disp[0])]
                self.sphere_fig.data[idx + 2].y = [float(z_disp[1])]
                self.sphere_fig.data[idx + 2].z = [float(z_disp[2])]

                self.sphere_fig.data[idx + 3].x = [float(Z_hat_disp[0])]
                self.sphere_fig.data[idx + 3].y = [float(Z_hat_disp[1])]
                self.sphere_fig.data[idx + 3].z = [float(Z_hat_disp[2])]

                if show_paths and path_update:
                    path_decim = 1 if not self._is_playing() else max(1, (t + 1) // 1200)
                    if path_decim > 1:
                        path_idx = np.arange(0, t + 1, path_decim, dtype=np.int32)
                        if path_idx[-1] != t:
                            path_idx = np.concatenate([path_idx, np.array([t], dtype=np.int32)])
                        wp = self._traj_cache["w"][path_idx]
                        zp = z_series[path_idx]
                        Zp = Z_series[path_idx] / K
                    else:
                        wp = self._traj_cache["w"][: t + 1]
                        zp = z_series[: t + 1]
                        Zp = Z_series[: t + 1] / K

                    wp_disp = self._maybe_invert_rows(wp, frame_name=frame_name, inv_ctx=inv_ctx)
                    zp_disp = self._maybe_invert_rows(zp, frame_name=frame_name, inv_ctx=inv_ctx)
                    Zp_disp = self._maybe_invert_rows(Zp, frame_name=frame_name, inv_ctx=inv_ctx)

                    self.sphere_fig.data[idx + 4].x = wp_disp[:, 0].tolist()
                    self.sphere_fig.data[idx + 4].y = wp_disp[:, 1].tolist()
                    self.sphere_fig.data[idx + 4].z = wp_disp[:, 2].tolist()

                    self.sphere_fig.data[idx + 5].x = zp_disp[:, 0].tolist()
                    self.sphere_fig.data[idx + 5].y = zp_disp[:, 1].tolist()
                    self.sphere_fig.data[idx + 5].z = zp_disp[:, 2].tolist()

                    self.sphere_fig.data[idx + 6].x = Zp_disp[:, 0].tolist()
                    self.sphere_fig.data[idx + 6].y = Zp_disp[:, 1].tolist()
                    self.sphere_fig.data[idx + 6].z = Zp_disp[:, 2].tolist()

                self.sphere_fig.data[idx + 7].x = [0.0, float(w_disp[0])]
                self.sphere_fig.data[idx + 7].y = [0.0, float(w_disp[1])]
                self.sphere_fig.data[idx + 7].z = [0.0, float(w_disp[2])]

                self.sphere_fig.data[idx + 8].x = [0.0, float(z_disp[0])]
                self.sphere_fig.data[idx + 8].y = [0.0, float(z_disp[1])]
                self.sphere_fig.data[idx + 8].z = [0.0, float(z_disp[2])]

                self.sphere_fig.data[idx + 9].x = [0.0, float(Z_hat_disp[0])]
                self.sphere_fig.data[idx + 9].y = [0.0, float(Z_hat_disp[1])]
                self.sphere_fig.data[idx + 9].z = [0.0, float(Z_hat_disp[2])]

                self.sphere_fig.data[idx + 4].visible = show_paths
                self.sphere_fig.data[idx + 5].visible = show_paths
                self.sphere_fig.data[idx + 6].visible = show_paths
                self.sphere_fig.data[idx + 7].visible = show_vectors
                self.sphere_fig.data[idx + 8].visible = show_vectors
                self.sphere_fig.data[idx + 9].visible = show_vectors
        finally:
            self._in_frame_update = False

        if path_update:
            self._last_path_frame = t
        overlay_update = force_update or (abs(t - self._last_overlay_frame) >= self._overlay_stride())
        if not overlay_update:
            return
        self._last_overlay_frame = t

        metric = self._metric_cache if self._metric_cache else {}
        def _metric_at(key: str) -> float:
            arr = metric.get(key)
            if arr is None or len(arr) <= t:
                return 0.0
            return float(arr[t])

        z_norm = float(np.linalg.norm(Z) / K)
        w_norm = float(np.linalg.norm(w))
        entropy_t = _metric_at("entropy")
        entropy_rate_t = _metric_at("entropy_rate")
        var_total_t = _metric_at("var_total")
        var_perp_t = _metric_at("var_perp")
        var_aligned_t = _metric_at("var_aligned")
        conformal_sign = -1.0 if bool(self.toggle_entropy.value) else 1.0
        time_sign = self._time_direction_sign()
        dt_eff = self._effective_dt(params)

        n_total = int(self._base_points_np.shape[0]) if self._base_points_np is not None else 0
        n_display = int(len(self._display_indices)) if self._display_indices is not None else n_total
        mode_hi, mode_lo, _ = self._ensemble_modes
        rt_high = float(self._ensemble_runtime.get(mode_hi, 0.0))
        rt_low = float(self._ensemble_runtime.get(mode_lo, 0.0))
        rt_poi = float(self._ensemble_runtime.get("poisson", 0.0))
        self.stats_html.value = (
            "<b>Hydrodynamic Ensemble Stats</b>"
            "<table style='font-family:monospace;font-size:12px;margin-top:6px'>"
            f"<tr><td style='padding-right:14px'>Displayed mode</td><td>{self._init_mode_label(self._display_mode)}</td></tr>"
            f"<tr><td style='padding-right:14px'>N total / shown</td><td>{n_total} / {n_display}</td></tr>"
            f"<tr><td style='padding-right:14px'>Runtime high/low/poisson [s]</td>"
            f"<td>{rt_high:.2f} / {rt_low:.2f} / {rt_poi:.2f}</td></tr>"
            f"<tr><td style='padding-right:14px'>dt eff</td><td>{dt_eff:+.4f}</td></tr>"
            f"<tr><td style='padding-right:14px'>Time direction</td><td>{'backward' if time_sign < 0 else 'forward'}</td></tr>"
            f"<tr><td style='padding-right:14px'>|w|</td><td>{w_norm:.5f}</td></tr>"
            f"<tr><td style='padding-right:14px'>|Z|/K</td><td>{z_norm:.5f}</td></tr>"
            f"<tr><td style='padding-right:14px'>{self._primary_metric_title()}</td><td>{entropy_t:.5f}</td></tr>"
            f"<tr><td style='padding-right:14px'>{self._primary_rate_series_name()}</td><td>{entropy_rate_t:.5f}</td></tr>"
            f"<tr><td style='padding-right:14px'>Var total</td><td>{var_total_t:.5f}</td></tr>"
            f"<tr><td style='padding-right:14px'>Var perp(final-axis)</td><td>{var_perp_t:.5f}</td></tr>"
            f"<tr><td style='padding-right:14px'>Var aligned(final-axis)</td><td>{var_aligned_t:.5f}</td></tr>"
            f"<tr><td style='padding-right:14px'>Entropy direction</td>"
            f"<td>{'increase' if bool(self.toggle_entropy.value) else 'dissipate'}</td></tr>"
            f"<tr><td style='padding-right:14px'>Alignment force sign</td><td>{conformal_sign:+.0f}</td></tr>"
            "</table>"
        )


__all__ = [
    "LMSBall3DWidget",
    "LMSBall3DBackwardTwoSheetWidget",
    "LMSBall3DHydrodynamicEnsembleWidget",
    "LMS3DControlSpec",
    "DEFAULT_CONTROLS",
    "HYDRO_DEFAULT_CONTROLS",
]
