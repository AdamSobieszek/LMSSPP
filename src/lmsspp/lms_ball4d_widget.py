"""4D Plotly widget for reduced LMS dynamics on S^3 / B^4.

This widget keeps LMS dynamics in ambient dimension d=4 and renders the S^3
boundary into R^3 via stereographic inversion charts.

Enhancements over the single-chart view:
- Dipole charts: two simultaneous inversion charts with opposite poles.
- Covariance panel: switch right panel to a 4x4 covariance matrix view in
  lab or co-rotating coordinates.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import torch

try:
    import ipywidgets as widgets
    import plotly.graph_objects as go
except Exception as exc:  # pragma: no cover
    raise ImportError("lms_ball4d_widget requires ipywidgets and plotly.") from exc

try:
    from .LMS import (
        integrate_lms_reduced_euler,
        normalize,
        random_points_on_sphere,
    )
    from .lms_ball3d_widget import LMS3DControlSpec, LMSBall3DWidget
except Exception:
    from LMS import (  # type: ignore
        integrate_lms_reduced_euler,
        normalize,
        random_points_on_sphere,
    )
    from lms_ball3d_widget import LMS3DControlSpec, LMSBall3DWidget  # type: ignore


LMS4D_DEFAULT_CONTROLS = (
    LMS3DControlSpec("r0", "Start radius |w|", 1e-4, 0.9999, 1e-4, 0.03, readout_format=".4f"),
    LMS3DControlSpec("N", "Oscillators N", 8, 300, 1, 150, integer=True, readout_format=".0f"),
    LMS3DControlSpec("K", "Coupling K", 0.0, 5.0, 0.02, 1.0, readout_format=".2f"),
    LMS3DControlSpec("omega", "Rotation rate omega", -10.0, 10.0, 0.02, 3.0, readout_format=".2f"),
    LMS3DControlSpec("dt", "Time step dt", 1e-4, 8e-2, 1e-4, 5e-2, readout_format=".4f"),
    LMS3DControlSpec("w_az", "w direction azimuth", -math.pi, math.pi, 0.02, 0.2, readout_format=".2f"),
    LMS3DControlSpec(
        "w_el",
        "w direction elevation",
        -0.5 * math.pi,
        0.5 * math.pi,
        0.02,
        0.25,
        readout_format=".2f",
    ),
    LMS3DControlSpec("w_4", "w direction 4th component", -1.0, 1.0, 0.01, 0.10, readout_format=".2f"),
    LMS3DControlSpec("ax_az", "rotation azimuth", -math.pi, math.pi, 0.02, 0.0, readout_format=".2f"),
    LMS3DControlSpec(
        "ax_el",
        "rotation elevation",
        -0.5 * math.pi,
        0.5 * math.pi,
        0.02,
        0.5 * math.pi,
        readout_format=".2f",
    ),
    LMS3DControlSpec("ax_4", "rotation 4th component", -1.0, 1.0, 0.01, 0.0, readout_format=".2f"),
    LMS3DControlSpec("steps", "Integration steps", 200, 2000, 1, 400, integer=True, readout_format=".0f"),
)


def _angles_to_unit4(az: float, el: float, fourth: float) -> np.ndarray:
    c = math.cos(float(el))
    vec = np.array(
        [c * math.cos(float(az)), c * math.sin(float(az)), math.sin(float(el)), float(fourth)],
        dtype=np.float64,
    )
    nrm = float(np.linalg.norm(vec))
    if nrm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return vec / nrm


class LMSBall4DWidget(LMSBall3DWidget):
    """Reduced LMS widget in B^4 with S^3 rendered through inversion charts."""

    def __init__(
        self,
        *,
        controls: tuple[LMS3DControlSpec, ...] = LMS4D_DEFAULT_CONTROLS,
        title: str = "Reduced LMS dynamics on S³ (dipole charts in R³) / B⁴",
        **kwargs: Any,
    ) -> None:
        super().__init__(controls=controls, title=title, **kwargs)
        self._inversion_enabled = True
        self._apply_projection_visual_mode()
        self._sync_secondary_panel()
        self._render_frame(int(self.frame_slider.value))

    def _build_controls(self) -> None:
        super()._build_controls()
        if self.controls_box.children and isinstance(self.controls_box.children[0], widgets.HTML):
            self.controls_box.children[0].value = "<b>LMS B⁴/S³ dipole-inversion widget controls</b>"

        self.secondary_panel_dropdown = widgets.Dropdown(
            options=[
                ("Dipole inversion chart", "dipole"),
                ("Covariance matrix (lab frame)", "cov_lab"),
                ("Covariance matrix (co-rotating frame)", "cov_body"),
            ],
            value="dipole",
            description="Right panel",
            layout=widgets.Layout(width=self._control_width),
            style={"description_width": "initial"},
        )

        children = list(self.controls_box.children)
        row = widgets.HBox([self.secondary_panel_dropdown], layout=widgets.Layout(width=self._control_width))
        children.insert(5, row)
        self.controls_box.children = tuple(children)

        self.toggle_init_state.disabled = True
        self.toggle_init_state.tooltip = "4D widget currently uses Poisson initialization only."

    @staticmethod
    def _make_projection_figure(*, point_size: int, title: str, uirev: str) -> go.FigureWidget:
        fig = go.FigureWidget()
        fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="markers",
                marker=dict(size=point_size, color="royalblue"),
                name="xᵢ(t)",
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="markers",
                marker=dict(size=point_size + 2, color="black", symbol="x"),
                name="w(t)",
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="markers",
                marker=dict(size=point_size + 1, color="white", line=dict(color="black", width=2)),
                name="z(t)",
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="markers",
                marker=dict(size=point_size + 1, color="firebrick"),
                name="Z/K",
                showlegend=True,
            )
        )
        fig.add_trace(
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
        fig.add_trace(
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
        fig.add_trace(
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
        fig.add_trace(
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
        fig.add_trace(
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
        fig.add_trace(
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
        fig.update_layout(
            title=title,
            width=760,
            height=760,
            margin=dict(l=20, r=20, t=48, b=20),
            showlegend=True,
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.75)"),
            uirevision=uirev,
            scene=dict(
                aspectmode="cube",
                dragmode="orbit",
                uirevision=f"{uirev}-scene",
                xaxis=dict(visible=False, autorange=False),
                yaxis=dict(visible=False, autorange=False),
                zaxis=dict(visible=False, autorange=False),
            ),
        )
        return fig

    @staticmethod
    def _make_covariance_figure(*, uirev: str) -> go.FigureWidget:
        fig = go.FigureWidget()
        fig.add_trace(
            go.Heatmap(
                z=np.zeros((4, 4), dtype=np.float64),
                colorscale="RdBu",
                zmid=0.0,
                zmin=-1.0,
                zmax=1.0,
                colorbar=dict(title="cov"),
            )
        )
        labels = ["x₁", "x₂", "x₃", "x₄"]
        fig.update_layout(
            title="Covariance around empirical center",
            template="plotly_white",
            width=760,
            height=760,
            margin=dict(l=58, r=32, t=64, b=52),
            uirevision=uirev,
            xaxis=dict(tickmode="array", tickvals=[0, 1, 2, 3], ticktext=labels, side="top"),
            yaxis=dict(
                tickmode="array",
                tickvals=[0, 1, 2, 3],
                ticktext=labels,
                autorange="reversed",
                scaleanchor="x",
                scaleratio=1,
            ),
        )
        return fig

    def _build_figures(self) -> None:
        super()._build_figures()
        if len(self.metrics_fig.layout.annotations) >= 2:
            self.metrics_fig.layout.annotations[1].text = (
                "Projected z(t) coordinates in the primary inversion chart"
            )
        self.metrics_fig.data[4].name = "proj-x"
        self.metrics_fig.data[5].name = "proj-y"
        self.metrics_fig.data[6].name = "proj-z"

        self.sphere_fig_dual = self._make_projection_figure(
            point_size=int(self.point_size),
            title="Dipole chart (antipode inversion pole)",
            uirev="lms-ball4d-dipole",
        )
        self.cov_fig = self._make_covariance_figure(uirev="lms-ball4d-cov")
        self.secondary_panel_box = widgets.Box(
            children=(self.sphere_fig_dual,),
            layout=widgets.Layout(width="760px", height="760px"),
        )
        self.projection_row = widgets.HBox(
            [self.sphere_fig, self.secondary_panel_box],
            layout=widgets.Layout(align_items="flex-start"),
        )
        self._dual_last_path_frame = -10**9
        self._apply_dual_scene_range()

    def _bind_events(self) -> None:
        super()._bind_events()
        self.secondary_panel_dropdown.observe(self._on_secondary_panel_change, names="value")

    def _on_secondary_panel_change(self, change: dict[str, Any]) -> None:
        if self._updating or change.get("name") != "value":
            return
        self._sync_secondary_panel()
        self._render_frame(int(self.frame_slider.value))

    def _secondary_panel_mode(self) -> str:
        dropdown = getattr(self, "secondary_panel_dropdown", None)
        if dropdown is None:
            return "dipole"
        return str(dropdown.value)

    def _sync_secondary_panel(self) -> None:
        mode = self._secondary_panel_mode()
        if mode == "dipole":
            self.secondary_panel_box.children = (self.sphere_fig_dual,)
        else:
            self.secondary_panel_box.children = (self.cov_fig,)

    def _apply_root_layout(self) -> None:
        fig_w = 590 if bool(self.layout_top_view.value) else 520
        fig_h = 600 if bool(self.layout_top_view.value) else 520
        for fig in (self.sphere_fig, self.sphere_fig_dual, self.cov_fig):
            fig.update_layout(width=fig_w, height=fig_h)
        self.secondary_panel_box.layout.width = f"{fig_w}px"
        self.secondary_panel_box.layout.height = f"{fig_h}px"

        row_w = 2 * fig_w + 22
        self.projection_row.children = (self.sphere_fig, self.secondary_panel_box)
        self.projection_row.layout = widgets.Layout(
            display="flex",
            flex_flow="row",
            align_items="flex-start",
            width=f"{row_w}px",
        )
        self.bottom_panel.layout.width = f"{max(980, row_w)}px"
        self.layout.children = (self.projection_row, self.bottom_panel)
        self.layout.layout = widgets.Layout(
            display="flex",
            flex_flow="column",
            align_items="flex-start",
            width=f"{max(980, row_w)}px",
        )
        self._sync_secondary_panel()
        self._sync_mode_button_labels()

    def _sync_init_state_button_label(self) -> None:
        self._init_state_mode = "poisson"
        self.toggle_init_state.description = "Initial State: Poisson (fixed)"
        self.toggle_init_state.button_style = "info"

    def _on_init_state_clicked(self, _btn: widgets.Button) -> None:
        return

    @staticmethod
    def _safe_unit4(v: np.ndarray) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float64).reshape(4)
        n = float(np.linalg.norm(arr))
        if n < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return arr / n

    @staticmethod
    def _orthonormal_tangent_basis(pole: np.ndarray) -> np.ndarray:
        p = LMSBall4DWidget._safe_unit4(pole)
        basis: list[np.ndarray] = []
        for cand in np.eye(4, dtype=np.float64):
            v = cand - float(np.dot(cand, p)) * p
            for b in basis:
                v = v - float(np.dot(v, b)) * b
            n = float(np.linalg.norm(v))
            if n > 1e-10:
                basis.append(v / n)
            if len(basis) == 3:
                break
        if len(basis) < 3:
            projector = np.eye(4, dtype=np.float64) - np.outer(p, p)
            u, s, _ = np.linalg.svd(projector)
            for i, sv in enumerate(s):
                if sv <= 1e-10:
                    continue
                v = np.asarray(u[:, i], dtype=np.float64)
                for b in basis:
                    v = v - float(np.dot(v, b)) * b
                n = float(np.linalg.norm(v))
                if n > 1e-10:
                    basis.append(v / n)
                if len(basis) == 3:
                    break
        if len(basis) < 3:
            basis = [
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
                np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64),
            ]
        return np.stack(basis[:3], axis=1)

    def _majority_cluster_direction(self, *, frame_name: Literal["lab", "body"]) -> np.ndarray:
        z_key = "z_body" if frame_name == "body" else "z_lab"
        z_series = self._traj_cache.get(z_key)
        if z_series is not None and len(z_series) > 0:
            z_last = np.asarray(z_series[-1], dtype=np.float64).reshape(-1)
            if z_last.shape[0] == 4 and float(np.linalg.norm(z_last)) > 1e-12:
                return self._safe_unit4(z_last)

        w_series = self._traj_cache.get("w")
        if w_series is not None and len(w_series) > 0:
            w_last = np.asarray(w_series[-1], dtype=np.float64).reshape(-1)
            if w_last.shape[0] == 4 and float(np.linalg.norm(w_last)) > 1e-12:
                return self._safe_unit4(-w_last)
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _inversion_context_pair(
        self,
        *,
        frame_name: Literal["lab", "body"],
    ) -> tuple[tuple[np.ndarray, np.ndarray, float], tuple[np.ndarray, np.ndarray, float]]:
        cluster = self._majority_cluster_direction(frame_name=frame_name)
        cap = float(self._scene_radius_inversion())
        pole_primary = -cluster
        pole_secondary = cluster
        ctx_primary = (pole_primary, self._orthonormal_tangent_basis(pole_primary), cap)
        ctx_secondary = (pole_secondary, self._orthonormal_tangent_basis(pole_secondary), cap)
        return ctx_primary, ctx_secondary

    def _inversion_context(self, *, frame_name: Literal["lab", "body"]) -> tuple[np.ndarray, np.ndarray, float]:
        primary, _ = self._inversion_context_pair(frame_name=frame_name)
        return primary

    def _secondary_inversion_context(
        self,
        *,
        frame_name: Literal["lab", "body"],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        _, secondary = self._inversion_context_pair(frame_name=frame_name)
        return secondary

    @staticmethod
    def _stereographic_project_rows(
        x: np.ndarray,
        *,
        pole: np.ndarray,
        basis: np.ndarray,
        cap: float,
        eps: float = 1e-6,
    ) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        if arr.shape[1] != 4:
            raise ValueError("stereographic projection expects vectors in R^4.")

        pole_u = LMSBall4DWidget._safe_unit4(pole)
        basis_43 = np.asarray(basis, dtype=np.float64).reshape(4, 3)
        y3 = arr @ basis_43
        y4 = arr @ pole_u
        den = 1.0 - y4[:, None]
        den_safe = np.where(np.abs(den) < eps, np.where(den >= 0.0, eps, -eps), den)
        out = y3 / den_safe
        if cap > 0.0:
            r = np.linalg.norm(out, axis=1, keepdims=True)
            s = np.minimum(1.0, float(cap) / np.maximum(r, 1e-12))
            out = out * s
        return out[0] if single else out

    def _maybe_invert_rows(
        self,
        x: np.ndarray,
        *,
        frame_name: Literal["lab", "body"],
        inv_ctx: tuple[np.ndarray, np.ndarray, float] | None,
    ) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1 and arr.shape[0] == 3:
            return arr
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr
        if arr.shape[-1] != 4:
            return arr
        if inv_ctx is None:
            inv_ctx = self._inversion_context(frame_name=frame_name)
        pole, basis, cap = inv_ctx
        return self._stereographic_project_rows(arr, pole=pole, basis=basis, cap=cap)

    def _scene_radius_default(self) -> float:
        return 8.0

    def _scene_radius_inversion(self) -> float:
        return 8.0

    def _apply_dual_scene_range(self) -> None:
        r = float(self._scene_radius_inversion())
        self.sphere_fig_dual.update_layout(
            scene=dict(
                xaxis=dict(visible=False, autorange=False, range=[-r, r]),
                yaxis=dict(visible=False, autorange=False, range=[-r, r]),
                zaxis=dict(visible=False, autorange=False, range=[-r, r]),
            )
        )

    def _apply_projection_visual_mode(self) -> None:
        self._in_frame_update = True
        try:
            with self.sphere_fig.batch_update():
                for i in range(self._wire_count):
                    self.sphere_fig.data[i].visible = False
                self._apply_scene_range()
        finally:
            self._in_frame_update = False
        if hasattr(self, "sphere_fig_dual"):
            self._apply_dual_scene_range()

    @staticmethod
    def _modified_bessel_i1(x: float) -> float:
        ax = abs(float(x))
        if ax < 1e-14:
            return 0.0
        half = 0.5 * ax
        term = half
        acc = term
        for m in range(1, 256):
            term *= (half * half) / (float(m) * float(m + 1))
            acc += term
            if abs(term) < 1e-14 * max(1.0, abs(acc)):
                break
        return -acc if x < 0.0 else acc

    @staticmethod
    def _kernel_uniform_density(*, kappa: float, sample_count: int) -> float:
        m = float(max(1, int(sample_count)))
        k = float(max(kappa, 0.0))
        if k < 1e-8:
            pair_expect = 1.0
        else:
            i1 = LMSBall4DWidget._modified_bessel_i1(k)
            pair_expect = float(math.exp(-k) * (2.0 * i1 / max(k, 1e-12)))
        return float((1.0 / m) + (1.0 - 1.0 / m) * pair_expect)

    def _compute_thermo_metrics(
        self,
        *,
        x_series: np.ndarray | None,
        z_series: np.ndarray,
        Z_series: np.ndarray,
        frame_name: Literal["lab", "body"],
    ) -> dict[str, np.ndarray]:
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
        entropy_proxy = np.zeros(t_count, dtype=np.float64)
        entropy_hist = np.zeros(t_count, dtype=np.float64)
        var_par = np.zeros(t_count, dtype=np.float64)
        var_perp = np.zeros(t_count, dtype=np.float64)
        entropy_kappa = 14.0
        entropy_sample_size = min(int(n), 420)

        for t in range(t_count):
            if x_series is not None:
                pts = np.asarray(x_series[t], dtype=np.float64)
            else:
                pts = np.asarray(
                    self._points_at_frame(t, x_series=None, frame_name=frame_name),
                    dtype=np.float64,
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

            axis = np.asarray(z_series[t], dtype=np.float64)
            axis_n = float(np.linalg.norm(axis))
            if axis_n < 1e-12:
                if bary_emp_t > 1e-12:
                    axis = mu_t / max(bary_emp_t, 1e-12)
                else:
                    axis = np.zeros((d,), dtype=np.float64)
                    axis[0] = 1.0
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

    def _refresh_metric_series(self, params: dict[str, float | int]) -> None:
        super()._refresh_metric_series(params)
        if not self._traj_cache:
            return
        frame_name: Literal["lab", "body"] = "body" if self.view_frame_dropdown.value == "body" else "lab"
        _, z_series, _ = self._frame_arrays()
        inv_ctx = self._inversion_context(frame_name=frame_name)
        z_proj = self._maybe_invert_rows(
            np.asarray(z_series, dtype=np.float64),
            frame_name=frame_name,
            inv_ctx=inv_ctx,
        )
        if z_proj.ndim != 2 or z_proj.shape[1] != 3:
            return
        t_axis = np.arange(self._steps + 1)
        lim = float(np.nanpercentile(np.abs(z_proj), 98.0)) if z_proj.size else 1.0
        lim = float(np.clip(max(1.0, 1.15 * lim), 1.0, 12.0))
        with self.metrics_fig.batch_update():
            self.metrics_fig.data[4].x = t_axis.tolist()
            self.metrics_fig.data[4].y = z_proj[:, 0].tolist()
            self.metrics_fig.data[5].x = t_axis.tolist()
            self.metrics_fig.data[5].y = z_proj[:, 1].tolist()
            self.metrics_fig.data[6].x = t_axis.tolist()
            self.metrics_fig.data[6].y = z_proj[:, 2].tolist()
            self.metrics_fig.update_yaxes(range=[-lim, lim], row=2, col=1)

    @staticmethod
    def _covariance_matrix(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 4:
            raise ValueError("covariance expects points with shape [N,4].")
        if pts.shape[0] <= 1:
            return np.zeros((4, 4), dtype=np.float64)
        mu = pts.mean(axis=0, keepdims=True)
        centered = pts - mu
        return (centered.T @ centered) / float(max(pts.shape[0], 1))

    def _update_covariance_panel(self, t: int, *, frame_name: Literal["lab", "body"]) -> None:
        x_key = "x_body" if frame_name == "body" else "x_lab"
        x_series = self._traj_cache.get(x_key)
        if x_series is not None:
            pts = np.asarray(x_series[t], dtype=np.float64)
        else:
            pts = np.asarray(self._points_at_frame(t, None, frame_name=frame_name), dtype=np.float64)
        cov = self._covariance_matrix(pts)
        eig = np.linalg.eigvalsh(cov)
        vmax = float(max(np.max(np.abs(cov)), 1e-9))
        title = (
            f"Covariance ({frame_name} frame) at t={int(t)} "
            f"| eig=[{eig[0]:.3f}, {eig[1]:.3f}, {eig[2]:.3f}, {eig[3]:.3f}]"
        )
        with self.cov_fig.batch_update():
            self.cov_fig.data[0].z = cov.tolist()
            self.cov_fig.data[0].zmin = -vmax
            self.cov_fig.data[0].zmax = vmax
            self.cov_fig.layout.title = title

    def _update_dipole_projection_frame(self, t: int) -> None:
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

        inv_ctx = self._secondary_inversion_context(frame_name=frame_name)
        x_plot_disp = self._maybe_invert_rows(x_plot, frame_name=frame_name, inv_ctx=inv_ctx)
        w_disp = self._maybe_invert_rows(w, frame_name=frame_name, inv_ctx=inv_ctx)
        z_disp = self._maybe_invert_rows(z, frame_name=frame_name, inv_ctx=inv_ctx)
        Z_hat_disp = self._maybe_invert_rows(Z_hat, frame_name=frame_name, inv_ctx=inv_ctx)

        show_paths = bool(self.show_paths.value)
        show_vectors = bool(self.show_vectors.value)
        force_update = (t == 0) or (t == self._steps)
        path_update = force_update or (abs(t - self._dual_last_path_frame) >= self._path_stride())

        with self.sphere_fig_dual.batch_update():
            self.sphere_fig_dual.data[0].x = x_plot_disp[:, 0].tolist()
            self.sphere_fig_dual.data[0].y = x_plot_disp[:, 1].tolist()
            self.sphere_fig_dual.data[0].z = x_plot_disp[:, 2].tolist()

            self.sphere_fig_dual.data[1].x = [float(w_disp[0])]
            self.sphere_fig_dual.data[1].y = [float(w_disp[1])]
            self.sphere_fig_dual.data[1].z = [float(w_disp[2])]

            self.sphere_fig_dual.data[2].x = [float(z_disp[0])]
            self.sphere_fig_dual.data[2].y = [float(z_disp[1])]
            self.sphere_fig_dual.data[2].z = [float(z_disp[2])]

            self.sphere_fig_dual.data[3].x = [float(Z_hat_disp[0])]
            self.sphere_fig_dual.data[3].y = [float(Z_hat_disp[1])]
            self.sphere_fig_dual.data[3].z = [float(Z_hat_disp[2])]

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

                self.sphere_fig_dual.data[4].x = wp_disp[:, 0].tolist()
                self.sphere_fig_dual.data[4].y = wp_disp[:, 1].tolist()
                self.sphere_fig_dual.data[4].z = wp_disp[:, 2].tolist()

                self.sphere_fig_dual.data[5].x = zp_disp[:, 0].tolist()
                self.sphere_fig_dual.data[5].y = zp_disp[:, 1].tolist()
                self.sphere_fig_dual.data[5].z = zp_disp[:, 2].tolist()

                self.sphere_fig_dual.data[6].x = Zp_disp[:, 0].tolist()
                self.sphere_fig_dual.data[6].y = Zp_disp[:, 1].tolist()
                self.sphere_fig_dual.data[6].z = Zp_disp[:, 2].tolist()

            self.sphere_fig_dual.data[7].x = [0.0, float(w_disp[0])]
            self.sphere_fig_dual.data[7].y = [0.0, float(w_disp[1])]
            self.sphere_fig_dual.data[7].z = [0.0, float(w_disp[2])]

            self.sphere_fig_dual.data[8].x = [0.0, float(z_disp[0])]
            self.sphere_fig_dual.data[8].y = [0.0, float(z_disp[1])]
            self.sphere_fig_dual.data[8].z = [0.0, float(z_disp[2])]

            self.sphere_fig_dual.data[9].x = [0.0, float(Z_hat_disp[0])]
            self.sphere_fig_dual.data[9].y = [0.0, float(Z_hat_disp[1])]
            self.sphere_fig_dual.data[9].z = [0.0, float(Z_hat_disp[2])]

            self.sphere_fig_dual.data[4].visible = show_paths
            self.sphere_fig_dual.data[5].visible = show_paths
            self.sphere_fig_dual.data[6].visible = show_paths
            self.sphere_fig_dual.data[7].visible = show_vectors
            self.sphere_fig_dual.data[8].visible = show_vectors
            self.sphere_fig_dual.data[9].visible = show_vectors

        if path_update:
            self._dual_last_path_frame = t

    def _render_frame(self, t: int) -> None:
        super()._render_frame(t)
        if self._steps <= 0 or not self._traj_cache:
            return
        t = max(0, min(int(t), self._steps))
        mode = self._secondary_panel_mode()
        if mode == "dipole":
            self._update_dipole_projection_frame(t)
            return
        if mode == "cov_body":
            self._update_covariance_panel(t, frame_name="body")
            return
        self._update_covariance_panel(t, frame_name="lab")

    def _recompute(self, *, reset_frame: bool) -> None:
        self._dual_last_path_frame = -10**9
        super()._recompute(reset_frame=reset_frame)

    @staticmethod
    def _skew_plane_rotation(axis: torch.Tensor, *, rate: float) -> torch.Tensor:
        if axis.numel() != 4:
            raise ValueError("axis must be a 4-vector.")
        a = normalize(axis.view(1, 4))[0]
        e4 = torch.zeros((4,), dtype=axis.dtype, device=axis.device)
        e4[3] = 1.0
        b = e4 - torch.dot(e4, a) * a
        b_norm = float(torch.linalg.norm(b))
        if b_norm < 1e-10:
            e1 = torch.zeros((4,), dtype=axis.dtype, device=axis.device)
            e1[0] = 1.0
            b = e1 - torch.dot(e1, a) * a
        b = normalize(b.view(1, 4))[0]
        return (torch.outer(b, a) - torch.outer(a, b)) * float(rate)

    def _simulate(self, params: dict[str, float | int]):
        d = 4
        n = int(params["N"])
        K = float(params["K"])
        conformal_sign = -1.0 if bool(self.toggle_entropy.value) else 1.0
        omega = float(params["omega"])
        r0 = float(params["r0"])
        w_az = float(params["w_az"])
        w_el = float(params["w_el"])
        w_4 = float(params.get("w_4", 0.0))
        ax_az = float(params["ax_az"])
        ax_el = float(params["ax_el"])
        ax_4 = float(params.get("ax_4", 0.0))
        dt = self._effective_dt(params)
        steps = int(params["steps"])
        store_points, resolved_mode = self._resolve_store_points(n=n, steps=steps, d=d)
        self._resolved_trajectory_mode = resolved_mode
        self._init_state_mode = "poisson"

        weights = torch.ones(n, dtype=torch.float64) / float(n)
        center_dir = torch.tensor(_angles_to_unit4(w_az, w_el, w_4), dtype=torch.float64)
        base_points = random_points_on_sphere(
            n,
            d=d,
            generator=self._torch_gen,
            dtype=torch.float64,
        )
        r0_clip = float(np.clip(r0, 0.0, 0.999999))
        w0 = -center_dir * r0_clip

        zeta0 = torch.eye(d, dtype=torch.float64)
        axis = torch.tensor(_angles_to_unit4(ax_az, ax_el, ax_4), dtype=torch.float64)
        A = self._skew_plane_rotation(axis, rate=omega).to(dtype=torch.float64)

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

    def _export_iframe_payload(self) -> dict[str, Any]:
        payload = super()._export_iframe_payload()
        payload["widget_kind"] = "lms_ball4d"
        return payload


__all__ = [
    "LMSBall4DWidget",
    "LMS4D_DEFAULT_CONTROLS",
]
