"""Plotly + ipywidgets LMS widget for circle/simplified disk dynamics.

This is a Plotly-first interactive widget inspired by widgets_legacy/LMS.html.
It keeps distinct reduced variables (w, z, Z) and exposes an extensible
slider/toggle schema for future controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any, Iterable

import numpy as np

try:
    import ipywidgets as widgets
    import plotly.graph_objects as go
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "lms_plotly_widget requires `ipywidgets` and `plotly`."
    ) from exc


TAU = 2.0 * math.pi
EPS = 1e-12


@dataclass(frozen=True)
class SliderSpec:
    key: str
    label: str
    min: float
    max: float
    step: float
    value: float
    integer: bool = False
    continuous_update: bool = False
    readout_format: str = ".2f"


@dataclass(frozen=True)
class ToggleSpec:
    key: str
    label: str
    value: bool = False


DEFAULT_SLIDERS: tuple[SliderSpec, ...] = (
    SliderSpec("N", "Oscillators N", 8, 200, 1, 60, integer=True, readout_format=".0f"),
    SliderSpec("r0", "Start radius r0", 1e-4, 0.95, 0.01, 0.02, readout_format=".2f"),
    SliderSpec("theta", "zeta_inf angle (rad)", -math.pi, math.pi, 0.01, 0.45, readout_format=".2f"),
    SliderSpec("d", "Dimension d", 2, 12, 1, 2, integer=True, readout_format=".0f"),
    SliderSpec("K", "Coupling K", 0.0, 6.0, 0.05, 1.0, readout_format=".2f"),
    SliderSpec("omega", "Rotation rate omega", -6.0, 6.0, 0.05, 1.0, readout_format=".2f"),
    SliderSpec("warp_p", "Time-warp exponent p", -2.0, 2.0, 0.05, 1.0, readout_format=".2f"),
    SliderSpec("warp_mul", "Time-warp multiplier a", -1.0, 1.0, 0.01, 1.0, readout_format=".2f"),
    SliderSpec("snr", "Signal-to-noise ratio", 0.0, 80.0, 0.1, 10.0, readout_format=".1f"),
)


DEFAULT_TOGGLES: tuple[ToggleSpec, ...] = (
    ToggleSpec("noise", "Noise", False),
    ToggleSpec("show_ray", "Show ray + handles", True),
    ToggleSpec("show_centers", "Show w and z", True),
    ToggleSpec("show_osc_ticks", "Oscillator ticks", True),
    ToggleSpec("show_z_arrow", "Show Z direction", True),
)


def _as_vec(theta: float) -> np.ndarray:
    return np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _clamp_to_disk(v: np.ndarray, max_r: float = 0.999995) -> np.ndarray:
    r = _norm(v)
    if r <= max_r:
        return v
    return (v / max(r, EPS)) * max_r


def _rot(v: np.ndarray, ang: float) -> np.ndarray:
    c, s = math.cos(ang), math.sin(ang)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=np.float64)


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _warp_factor(r: float, p: float) -> float:
    # Kept consistent with widgets_legacy/LMS.html behavior.
    u = max(1e-3, 1.0 - r)
    return min(6.0, u ** (-p))


def _hyp2f1_1b_c_u(b: float, c: float, u: float, *, max_n: int = 4000, tol: float = 1e-13) -> float:
    # 2F1(1,b;c;u) = sum_{n>=0} (b)_n/(c)_n * u^n
    term = 1.0
    acc = 1.0
    for n in range(max_n):
        term *= ((b + n) / (c + n)) * u
        acc += term
        if abs(term) < tol * max(1.0, abs(acc)):
            break
    return float(acc)


@lru_cache(maxsize=64)
def _fd_table(d: int, table_size: int = 2048) -> np.ndarray:
    if d == 2:
        return np.ones(table_size + 1, dtype=np.float64)

    b = 1.0 - 0.5 * d
    c = 1.0 + 0.5 * d
    denom = _hyp2f1_1b_c_u(b, c, 1.0)
    denom = denom if abs(denom) > EPS else 1.0

    table = np.empty(table_size + 1, dtype=np.float64)
    for i in range(table_size + 1):
        r = i / table_size
        u = min(0.999999999, r * r)
        table[i] = _hyp2f1_1b_c_u(b, c, u) / denom
    return table


def _f_shrink(d: int, r: float, table_size: int = 2048) -> float:
    table = _fd_table(int(d), table_size)
    rr = min(1.0, max(0.0, float(r)))
    x = rr * table_size
    i0 = int(x)
    i1 = min(table_size, i0 + 1)
    t = x - i0
    return float((1.0 - t) * table[i0] + t * table[i1])


def _mobius_circle_points(w: np.ndarray, phi: float, base_angles: np.ndarray) -> np.ndarray:
    # Complex-circle expression used in the HTML widget:
    # M_w(x) = (w + x) / (1 + conj(w) x), then y = exp(i phi) M_w(x).
    w_c = complex(float(w[0]), float(w[1]))
    x = np.exp(1j * base_angles)
    y = np.exp(1j * phi) * ((w_c + x) / (1.0 + np.conj(w_c) * x))
    return np.column_stack((np.real(y), np.imag(y)))


def _rays_polyline(unit_dirs: np.ndarray) -> tuple[list[float | None], list[float | None]]:
    xs: list[float | None] = []
    ys: list[float | None] = []
    for d in unit_dirs:
        xs.extend([0.0, float(d[0]), None])
        ys.extend([0.0, float(d[1]), None])
    return xs, ys


class LMSCirclePlotlyWidget:
    """Interactive Plotly widget for LMS reduced dynamics on the disk."""

    def __init__(
        self,
        *,
        steps: int = 420,
        dt: float = 0.012,
        slider_specs: Iterable[SliderSpec] | None = None,
        toggle_specs: Iterable[ToggleSpec] | None = None,
        rng_seed: int | None = 0,
        title: str = "LMS Kuramoto on the Poincare disk",
        width: int = 640,
        height: int = 640,
    ) -> None:
        self.steps = int(steps)
        self.dt = float(dt)
        self.title = title
        self.width = int(width)
        self.height = int(height)
        self._rng = np.random.default_rng(rng_seed)
        self._updating = False

        self.slider_specs = self._merge_specs(DEFAULT_SLIDERS, slider_specs)
        self.toggle_specs = self._merge_specs(DEFAULT_TOGGLES, toggle_specs)

        self._slider_widgets: dict[str, widgets.Widget] = {}
        self._toggle_widgets: dict[str, widgets.Widget] = {}
        self._kick_w0: np.ndarray | None = None
        self._kick_phi0: float | None = None

        self._build_controls()
        self._build_figures()
        self._bind_callbacks()
        self._recompute(reset_frame=True)

        self.layout = widgets.HBox(
            [
                self.disk_fig,
                widgets.VBox(
                    [
                        self.controls_box,
                        self.stats_html,
                        self.metrics_fig,
                    ],
                    layout=widgets.Layout(width="560px"),
                ),
            ]
        )

    @staticmethod
    def _merge_specs(defaults: tuple[Any, ...], overrides: Iterable[Any] | None) -> list[Any]:
        merged = {s.key: s for s in defaults}
        if overrides is not None:
            for s in overrides:
                merged[s.key] = s
        # Keep deterministic order: defaults first, then new keys.
        ordered = [merged[s.key] for s in defaults]
        extra = [v for k, v in merged.items() if k not in {s.key for s in defaults}]
        ordered.extend(extra)
        return ordered

    def _build_controls(self) -> None:
        slider_rows: list[widgets.Widget] = []
        for spec in self.slider_specs:
            if spec.integer:
                w = widgets.IntSlider(
                    value=int(spec.value),
                    min=int(spec.min),
                    max=int(spec.max),
                    step=int(spec.step),
                    description=spec.label,
                    continuous_update=spec.continuous_update,
                    layout=widgets.Layout(width="520px"),
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
                    layout=widgets.Layout(width="520px"),
                )
            self._slider_widgets[spec.key] = w
            slider_rows.append(w)

        toggle_rows: list[widgets.Widget] = []
        for spec in self.toggle_specs:
            w = widgets.Checkbox(
                value=bool(spec.value),
                description=spec.label,
                indent=False,
            )
            self._toggle_widgets[spec.key] = w
            toggle_rows.append(w)

        self.play = widgets.Play(
            value=0,
            min=0,
            max=self.steps,
            step=1,
            interval=35,
            description="Play",
            show_repeat=False,
        )
        self.frame_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=self.steps,
            step=1,
            description="Frame",
            layout=widgets.Layout(width="520px"),
            continuous_update=False,
        )
        widgets.jslink((self.play, "value"), (self.frame_slider, "value"))

        self.btn_reset = widgets.Button(description="Reset")
        self.btn_kick = widgets.Button(description="Random kick")
        self.btn_reset.layout.width = "255px"
        self.btn_kick.layout.width = "255px"

        self.stats_html = widgets.HTML(value="")

        self.controls_box = widgets.VBox(
            [
                widgets.HTML("<b>LMS controls (extensible schema)</b>"),
                widgets.HBox([self.play, self.btn_reset, self.btn_kick]),
                self.frame_slider,
                widgets.Accordion(
                    children=[
                        widgets.VBox(slider_rows),
                        widgets.VBox(toggle_rows),
                    ],
                    titles=("Sliders", "Toggles"),
                ),
            ]
        )

    def _build_figures(self) -> None:
        # Disk panel
        circle_t = np.linspace(0.0, TAU, 500)
        self.disk_fig = go.FigureWidget()
        self.disk_fig.add_trace(
            go.Scatter(
                x=np.cos(circle_t),
                y=np.sin(circle_t),
                mode="lines",
                line=dict(color="black", width=2),
                hoverinfo="skip",
                name="boundary",
            )
        )
        self.disk_fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color="rgba(0,0,0,0.28)", width=1),
                hoverinfo="skip",
                name="osc ticks",
            )
        )
        self.disk_fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                marker=dict(size=7, color="black", opacity=0.75),
                name="oscillators",
            )
        )
        self.disk_fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                marker=dict(size=12, color="black"),
                name="w",
            )
        )
        self.disk_fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                marker=dict(size=12, color="white", line=dict(color="black", width=2)),
                name="z",
            )
        )
        self.disk_fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color="rgba(0,0,0,0.45)", width=2),
                hoverinfo="skip",
                name="Z direction",
            )
        )
        self.disk_fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color="black", width=1.5, dash="dash"),
                hoverinfo="skip",
                name="ray",
            )
        )
        self.disk_fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                marker=dict(size=11, color="black"),
                name="zeta_inf",
            )
        )
        self.disk_fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                marker=dict(size=9, color="black"),
                name="r handle",
            )
        )

        self.disk_fig.update_layout(
            title=self.title,
            width=self.width,
            height=self.height,
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False,
            xaxis=dict(range=[-1.1, 1.1], visible=False, scaleanchor="y", scaleratio=1),
            yaxis=dict(range=[-1.1, 1.1], visible=False),
        )

        # Metrics panel
        self.metrics_fig = go.FigureWidget()
        self.metrics_fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color="rgba(20,60,90,0.92)", width=2),
                name="|Z|/K",
            )
        )
        self.metrics_fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color="rgba(10,120,80,0.88)", width=2),
                name="|w|",
            )
        )
        self.metrics_fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                marker=dict(size=9, color="rgba(20,60,90,1.0)"),
                name="|Z|/K (t)",
                showlegend=False,
            )
        )
        self.metrics_fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                marker=dict(size=9, color="rgba(10,120,80,1.0)"),
                name="|w| (t)",
                showlegend=False,
            )
        )
        self.metrics_fig.update_layout(
            width=540,
            height=280,
            template="plotly_white",
            margin=dict(l=24, r=24, t=30, b=24),
            showlegend=True,
            legend=dict(orientation="h", x=0.0, y=1.12),
            xaxis=dict(title="Frame"),
            yaxis=dict(title="Magnitude", range=[-0.02, 1.05]),
        )

    def _bind_callbacks(self) -> None:
        for w in self._slider_widgets.values():
            w.observe(self._on_controls_change, names="value")
        for w in self._toggle_widgets.values():
            w.observe(self._on_controls_change, names="value")
        self.frame_slider.observe(self._on_frame_change, names="value")
        self.btn_reset.on_click(self._on_reset_clicked)
        self.btn_kick.on_click(self._on_kick_clicked)

    def _params(self) -> dict[str, float | int]:
        out: dict[str, float | int] = {}
        for spec in self.slider_specs:
            v = self._slider_widgets[spec.key].value
            out[spec.key] = int(v) if spec.integer else float(v)
        return out

    def _toggles(self) -> dict[str, bool]:
        return {spec.key: bool(self._toggle_widgets[spec.key].value) for spec in self.toggle_specs}

    def _simulate(self, params: dict[str, float | int], toggles: dict[str, bool]) -> dict[str, np.ndarray]:
        n = int(params["N"])
        d = int(params["d"])
        K = float(params["K"])
        omega = float(params["omega"])
        warp_p = float(params["warp_p"])
        warp_mul = float(params["warp_mul"])
        snr = float(params["snr"])
        theta = float(params["theta"])
        r0 = float(params["r0"])
        use_noise = bool(toggles.get("noise", False))

        base_angles = np.linspace(0.0, TAU, n, endpoint=False, dtype=np.float64)
        T = self.steps + 1

        w_arr = np.zeros((T, 2), dtype=np.float64)
        z_arr = np.zeros((T, 2), dtype=np.float64)
        Z_arr = np.zeros((T, 2), dtype=np.float64)
        fd_arr = np.zeros(T, dtype=np.float64)
        phi_arr = np.zeros(T, dtype=np.float64)
        osc_arr = np.zeros((T, n, 2), dtype=np.float64)
        zn_arr = np.zeros(T, dtype=np.float64)
        wn_arr = np.zeros(T, dtype=np.float64)

        if self._kick_w0 is not None:
            w = _clamp_to_disk(self._kick_w0.copy(), 0.98)
            phi = float(self._kick_phi0 or 0.0)
        else:
            w = _clamp_to_disk(_as_vec(theta) * r0, 0.98)
            phi = 0.0

        for t in range(T):
            z = -_rot(w, phi)
            rz = _norm(z)
            fd = _f_shrink(d, rz)
            Z = z * (K * fd)

            y = _mobius_circle_points(w, phi, base_angles)
            yn = np.linalg.norm(y, axis=1, keepdims=True)
            y = y / np.maximum(yn, EPS)

            w_arr[t] = w
            z_arr[t] = z
            Z_arr[t] = Z
            fd_arr[t] = fd
            phi_arr[t] = phi
            osc_arr[t] = y
            wn_arr[t] = _norm(w)
            zn_arr[t] = _norm(Z) / max(abs(K), EPS)

            if t == T - 1:
                break

            one_minus = max(1e-9, 1.0 - float(np.dot(w, w)))
            Z_body = _rot(Z, -phi)
            wdot0 = -0.5 * one_minus * Z_body

            time_scale = _warp_factor(_norm(w), warp_p) * warp_mul
            dt_eff = self.dt * time_scale

            if use_noise:
                drift_mag = max(1e-6, _norm(wdot0))
                sigma = drift_mag / max(0.5, snr)
                dW = self._rng.normal(size=2) * sigma * math.sqrt(abs(dt_eff))
                w = w + wdot0 * dt_eff + dW
            else:
                w = w + wdot0 * dt_eff
            w = _clamp_to_disk(w, 0.999995)

            zeta_w = _rot(w, phi)
            precess = _cross2(zeta_w, Z)
            phi += (omega - precess) * dt_eff

            if phi > math.pi:
                phi -= TAU
            if phi < -math.pi:
                phi += TAU

        return {
            "w": w_arr,
            "z": z_arr,
            "Z": Z_arr,
            "fd": fd_arr,
            "phi": phi_arr,
            "osc": osc_arr,
            "zn": zn_arr,
            "wn": wn_arr,
        }

    def _recompute(self, *, reset_frame: bool) -> None:
        params = self._params()
        toggles = self._toggles()
        self._traj = self._simulate(params, toggles)

        frames = np.arange(self.steps + 1)
        with self.metrics_fig.batch_update():
            self.metrics_fig.data[0].x = frames
            self.metrics_fig.data[0].y = self._traj["zn"]
            self.metrics_fig.data[1].x = frames
            self.metrics_fig.data[1].y = self._traj["wn"]
            self.metrics_fig.update_xaxes(range=[0, self.steps])

        if reset_frame:
            self._updating = True
            self.frame_slider.max = self.steps
            self.play.max = self.steps
            self.frame_slider.value = 0
            self.play.value = 0
            self._updating = False

        self._render_frame(int(self.frame_slider.value))

    def _render_frame(self, t: int) -> None:
        params = self._params()
        toggles = self._toggles()
        t = max(0, min(t, self.steps))

        w = self._traj["w"][t]
        z = self._traj["z"][t]
        Z = self._traj["Z"][t]
        osc = self._traj["osc"][t]
        theta = float(params["theta"])
        r0 = float(params["r0"])
        d = int(params["d"])
        K = float(params["K"])
        omega = float(params["omega"])
        p = float(params["warp_p"])
        snr = float(params["snr"])

        zeta_dir = _as_vec(theta)
        zeta_handle = zeta_dir * 0.999
        rho_handle = zeta_dir * r0
        rays_x, rays_y = _rays_polyline(osc)

        z_mag = _norm(Z)
        z_dir = Z / max(z_mag, EPS)
        z_tip = z_dir * min(0.72, 0.12 + 0.6 * self._traj["zn"][t])

        with self.disk_fig.batch_update():
            self.disk_fig.data[1].x = rays_x
            self.disk_fig.data[1].y = rays_y
            self.disk_fig.data[2].x = osc[:, 0]
            self.disk_fig.data[2].y = osc[:, 1]
            self.disk_fig.data[3].x = [w[0]]
            self.disk_fig.data[3].y = [w[1]]
            self.disk_fig.data[4].x = [z[0]]
            self.disk_fig.data[4].y = [z[1]]
            self.disk_fig.data[5].x = [0.0, z_tip[0]]
            self.disk_fig.data[5].y = [0.0, z_tip[1]]
            self.disk_fig.data[6].x = [0.0, zeta_handle[0]]
            self.disk_fig.data[6].y = [0.0, zeta_handle[1]]
            self.disk_fig.data[7].x = [zeta_handle[0]]
            self.disk_fig.data[7].y = [zeta_handle[1]]
            self.disk_fig.data[8].x = [rho_handle[0]]
            self.disk_fig.data[8].y = [rho_handle[1]]
            self.disk_fig.data[1].visible = toggles.get("show_osc_ticks", True)
            self.disk_fig.data[3].visible = toggles.get("show_centers", True)
            self.disk_fig.data[4].visible = toggles.get("show_centers", True)
            self.disk_fig.data[5].visible = toggles.get("show_z_arrow", True)
            self.disk_fig.data[6].visible = toggles.get("show_ray", True)
            self.disk_fig.data[7].visible = toggles.get("show_ray", True)
            self.disk_fig.data[8].visible = toggles.get("show_ray", True)

        with self.metrics_fig.batch_update():
            self.metrics_fig.data[2].x = [t]
            self.metrics_fig.data[2].y = [self._traj["zn"][t]]
            self.metrics_fig.data[3].x = [t]
            self.metrics_fig.data[3].y = [self._traj["wn"][t]]

        self.stats_html.value = (
            "<b>Statistics</b>"
            "<table style='font-family:monospace;font-size:12px;margin-top:6px'>"
            f"<tr><td style='padding-right:14px'>d</td><td>{d}</td></tr>"
            f"<tr><td style='padding-right:14px'>K</td><td>{K:.3f}</td></tr>"
            f"<tr><td style='padding-right:14px'>omega</td><td>{omega:.3f}</td></tr>"
            f"<tr><td style='padding-right:14px'>|w|</td><td>{self._traj['wn'][t]:.5f}</td></tr>"
            f"<tr><td style='padding-right:14px'>|z|</td><td>{_norm(z):.5f}</td></tr>"
            f"<tr><td style='padding-right:14px'>f_d(|z|)</td><td>{self._traj['fd'][t]:.5f}</td></tr>"
            f"<tr><td style='padding-right:14px'>|Z|/K</td><td>{self._traj['zn'][t]:.5f}</td></tr>"
            f"<tr><td style='padding-right:14px'>warp p</td><td>{p:.3f}</td></tr>"
            f"<tr><td style='padding-right:14px'>snr</td><td>{snr:.2f}</td></tr>"
            "</table>"
        )

    def _on_controls_change(self, _change: Any) -> None:
        if self._updating:
            return
        self._kick_w0 = None
        self._kick_phi0 = None
        self._recompute(reset_frame=False)

    def _on_frame_change(self, change: dict[str, Any]) -> None:
        if self._updating:
            return
        self._render_frame(int(change["new"]))

    def _on_reset_clicked(self, _btn: widgets.Button) -> None:
        self._kick_w0 = None
        self._kick_phi0 = None
        self._recompute(reset_frame=True)

    def _on_kick_clicked(self, _btn: widgets.Button) -> None:
        t = int(self.frame_slider.value)
        w = self._traj["w"][t].copy()
        phi = float(self._traj["phi"][t])
        self._kick_w0 = _clamp_to_disk(w + 0.10 * self._rng.normal(size=2), 0.98)
        self._kick_phi0 = phi + 0.8 * float(self._rng.normal())
        self._recompute(reset_frame=True)


__all__ = [
    "LMSCirclePlotlyWidget",
    "SliderSpec",
    "ToggleSpec",
    "DEFAULT_SLIDERS",
    "DEFAULT_TOGGLES",
]

