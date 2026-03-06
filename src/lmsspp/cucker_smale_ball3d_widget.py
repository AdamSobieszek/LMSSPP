"""3D Plotly widgets for Cucker-Smale dynamics reusing LMS widget UX."""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Protocol

import numpy as np
import torch

try:
    import ipywidgets as widgets
except Exception as exc:  # pragma: no cover
    raise ImportError("cucker_smale_ball3d_widget requires ipywidgets.") from exc

try:
    from .core import cucker_smale as cs_backend
    from .lms_ball3d_widget import (
        LMS3DControlSpec,
        ActiveInitMode,
        InitMetricMode,
        LMSBall3DHydrodynamicEnsembleWidget,
        LMSBall3DWidget,
        _HydroRecomputeCancelled,
        _angles_to_unit,
    )
except Exception:
    import cucker_smale as cs_backend  # type: ignore
    from lms_ball3d_widget import (  # type: ignore
        LMS3DControlSpec,
        ActiveInitMode,
        InitMetricMode,
        LMSBall3DHydrodynamicEnsembleWidget,
        LMSBall3DWidget,
        _HydroRecomputeCancelled,
        _angles_to_unit,
    )


class CuckerSmaleBackendContract(Protocol):
    """Backend API contract needed by Cucker-Smale widget subclasses."""

    def simulate_cs_widget_trajectory(
        self,
        *,
        x0: torch.Tensor,
        omega: torch.Tensor,
        weights: torch.Tensor,
        alpha: float,
        eps: float,
        coupling: float,
        dt: float,
        steps: int,
        store_points: str,
        store_indices: torch.Tensor | None,
        store_dtype: torch.dtype,
        chunk_size: int | None,
        preallocate: bool,
        cancel_check: Callable[[], bool] | None,
    ) -> Any:
        ...


CS_DEFAULT_CONTROLS = (
    LMS3DControlSpec("r0", "Start spread r0", 1e-4, 4.0, 1e-4, 0.12, readout_format=".4f"),
    LMS3DControlSpec("N", "Particles N", 10, 1200, 1, 220, integer=True, readout_format=".0f"),
    LMS3DControlSpec("K", "Coupling K", 0.0, 5.0, 0.02, 1.0, readout_format=".2f"),
    LMS3DControlSpec("omega", "Mean |omega|", -10.0, 10.0, 0.02, 2.0, readout_format=".2f"),
    LMS3DControlSpec("omega_std", "omega spread", 0.0, 6.0, 0.02, 0.7, readout_format=".2f"),
    LMS3DControlSpec("alpha", "Singularity alpha", 0.01, 0.95, 0.01, 0.35, readout_format=".2f"),
    LMS3DControlSpec("eps", "Regularization eps", 1e-4, 3e-1, 1e-4, 2e-2, readout_format=".4f"),
    LMS3DControlSpec("dt", "Time step dt", 1e-4, 8e-2, 1e-4, 1e-2, readout_format=".4f"),
    LMS3DControlSpec("w_az", "omega direction azimuth", -math.pi, math.pi, 0.02, 0.2, readout_format=".2f"),
    LMS3DControlSpec(
        "w_el",
        "omega direction elevation",
        -0.5 * math.pi,
        0.5 * math.pi,
        0.02,
        0.25,
        readout_format=".2f",
    ),
    LMS3DControlSpec("ax_az", "x center azimuth", -math.pi, math.pi, 0.02, 0.0, readout_format=".2f"),
    LMS3DControlSpec(
        "ax_el",
        "x center elevation",
        -0.5 * math.pi,
        0.5 * math.pi,
        0.02,
        0.5 * math.pi,
        readout_format=".2f",
    ),
    LMS3DControlSpec("steps", "Integration steps", 60, 1600, 1, 260, integer=True, readout_format=".0f"),
)


CS_HYDRO_DEFAULT_CONTROLS = (
    LMS3DControlSpec("r0", "Start spread r0", 1e-4, 4.0, 1e-4, 0.12, readout_format=".4f"),
    LMS3DControlSpec("N", "Particles N", 80, 1800, 20, 320, integer=True, readout_format=".0f"),
    LMS3DControlSpec("K", "Coupling K", 0.0, 5.0, 0.02, 1.0, readout_format=".2f"),
    LMS3DControlSpec("omega", "Mean |omega|", -10.0, 10.0, 0.02, 2.0, readout_format=".2f"),
    LMS3DControlSpec("omega_std", "omega spread", 0.0, 6.0, 0.02, 0.7, readout_format=".2f"),
    LMS3DControlSpec("alpha", "Singularity alpha", 0.01, 0.95, 0.01, 0.35, readout_format=".2f"),
    LMS3DControlSpec("eps", "Regularization eps", 1e-4, 3e-1, 1e-4, 2e-2, readout_format=".4f"),
    LMS3DControlSpec("dt", "Time step dt", 1e-4, 6e-2, 1e-4, 8e-3, readout_format=".4f"),
    LMS3DControlSpec("w_az", "omega direction azimuth", -math.pi, math.pi, 0.02, 0.2, readout_format=".2f"),
    LMS3DControlSpec(
        "w_el",
        "omega direction elevation",
        -0.5 * math.pi,
        0.5 * math.pi,
        0.02,
        0.25,
        readout_format=".2f",
    ),
    LMS3DControlSpec("ax_az", "x center azimuth", -math.pi, math.pi, 0.02, 0.0, readout_format=".2f"),
    LMS3DControlSpec(
        "ax_el",
        "x center elevation",
        -0.5 * math.pi,
        0.5 * math.pi,
        0.02,
        0.5 * math.pi,
        readout_format=".2f",
    ),
    LMS3DControlSpec("steps", "Integration steps", 40, 900, 1, 180, integer=True, readout_format=".0f"),
)


class _CuckerSmaleWidgetMixin:
    backend: CuckerSmaleBackendContract = cs_backend  # type: ignore[assignment]

    def _cs_mode_scales(self, mode: str) -> tuple[float, float]:
        mode_key = self._canonical_init_mode(mode)
        if mode_key in {"entropy_low", "var_perp_low", "varmin_parallel_perp"}:
            return 0.55, 0.55
        if mode_key in {"entropy_high", "var_perp_high", "varmax_parallel_perp"}:
            return 1.35, 1.35
        return 1.0, 1.0

    def _sample_store_indices(self, *, n: int, target_count: int) -> torch.Tensor | None:
        m = int(max(1, min(n, target_count)))
        if m >= n:
            return None
        idx_np = self._deterministic_subsample_indices(n, m)
        idx = torch.from_numpy(idx_np.astype(np.int64))
        return idx

    def _simulate_cs_with_mode(
        self,
        mode: str,
        params: dict[str, float | int],
        *,
        entropy_increase: bool | None = None,
        time_backward: bool | None = None,
        cancel_check: Callable[[], bool] | None = None,
        store_points: str = "both",
        store_indices: torch.Tensor | None = None,
    ):
        n = int(params["N"])
        d = 3
        r0 = float(params["r0"])
        coupling_mag = float(params["K"])
        omega_mag = float(params["omega"])
        omega_std = float(params.get("omega_std", 0.7))
        alpha = float(params.get("alpha", 0.35))
        eps = float(params.get("eps", 2e-2))
        dt = self._effective_dt(params, time_backward=time_backward)
        steps = int(params["steps"])
        entropy_flag = bool(self.toggle_entropy.value) if entropy_increase is None else bool(entropy_increase)
        coupling_sign = -1.0 if entropy_flag else 1.0

        omega_dir = torch.tensor(
            _angles_to_unit(float(params["w_az"]), float(params["w_el"])),
            dtype=torch.float64,
        )
        x_dir = torch.tensor(
            _angles_to_unit(float(params["ax_az"]), float(params["ax_el"])),
            dtype=torch.float64,
        )
        pos_scale, vel_scale = self._cs_mode_scales(mode)
        pos_sigma = max(1e-4, abs(r0) * pos_scale)
        vel_sigma = max(1e-4, abs(omega_std) * vel_scale)

        mean_x = abs(r0) * x_dir
        mean_omega = omega_mag * omega_dir
        x0 = mean_x.unsqueeze(0) + pos_sigma * torch.randn(
            (n, d),
            generator=self._torch_gen,
            dtype=torch.float64,
        )
        omega = mean_omega.unsqueeze(0) + vel_sigma * torch.randn(
            (n, d),
            generator=self._torch_gen,
            dtype=torch.float64,
        )
        weights = torch.ones((n,), dtype=torch.float64) / float(n)

        return self.backend.simulate_cs_widget_trajectory(
            x0=x0,
            omega=omega,
            weights=weights,
            alpha=alpha,
            eps=eps,
            coupling=coupling_sign * coupling_mag,
            dt=dt,
            steps=steps,
            store_points=store_points,
            store_indices=store_indices,
            store_dtype=torch.float32,
            chunk_size=None,
            preallocate=True,
            cancel_check=cancel_check,
        )


class CuckerSmaleBall3DWidget(_CuckerSmaleWidgetMixin, LMSBall3DWidget):
    """LMSBall3DWidget-compatible class backed by Cucker-Smale simulations."""

    def __init__(
        self,
        *,
        controls: tuple[LMS3DControlSpec, ...] = CS_DEFAULT_CONTROLS,
        title: str = "Cucker-Smale dynamics in R³ (projected to B³)",
        **kwargs: Any,
    ) -> None:
        super().__init__(controls=controls, title=title, **kwargs)

    def _build_controls(self) -> None:
        super()._build_controls()
        if self.controls_box.children and isinstance(self.controls_box.children[0], widgets.HTML):
            self.controls_box.children[0].value = "<b>Cucker-Smale B³ display widget controls</b>"
        self.mode_dropdown.options = [("cucker-smale backend", "cs")]
        self.mode_dropdown.value = "cs"
        self.mode_dropdown.disabled = True

    def _simulate(self, params: dict[str, float | int]):
        n = int(params["N"])
        steps = int(params["steps"])
        store_points, resolved_mode = self._resolve_store_points(n=n, steps=steps, d=3)
        self._resolved_trajectory_mode = resolved_mode
        if store_points == "none":
            store_points = "both"
        mode = self._coerce_mode_to_active_family(self._init_state_mode)
        self._init_state_mode = mode
        return self._simulate_cs_with_mode(
            mode,
            params,
            store_points=store_points,
            store_indices=None,
        )


class CuckerSmaleBall3DHydrodynamicEnsembleWidget(
    _CuckerSmaleWidgetMixin,
    LMSBall3DHydrodynamicEnsembleWidget,
):
    """Hydrodynamic-style ensemble widget using Cucker-Smale backend trajectories."""

    def __init__(
        self,
        *,
        controls: tuple[LMS3DControlSpec, ...] = CS_HYDRO_DEFAULT_CONTROLS,
        init_metric_mode: InitMetricMode = "entropy",
        **kwargs: Any,
    ) -> None:
        super().__init__(controls=controls, init_metric_mode=init_metric_mode, **kwargs)

    def _build_controls(self) -> None:
        super()._build_controls()
        if self.controls_box.children and isinstance(self.controls_box.children[0], widgets.HTML):
            self.controls_box.children[0].value = "<b>Cucker-Smale B³ ensemble widget controls</b>"
        self.mode_dropdown.options = [("cucker-smale backend", "cs")]
        self.mode_dropdown.value = "cs"
        self.mode_dropdown.disabled = True

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
        n = int(params["N"])
        target_count = min(n, max(1400, int(self.display_points_cap) * 2))
        store_indices = self._sample_store_indices(n=n, target_count=target_count)
        return self._simulate_cs_with_mode(
            mode,
            params,
            entropy_increase=entropy_increase,
            time_backward=time_backward,
            cancel_check=cancel_check,
            store_points="both",
            store_indices=store_indices,
        )

    def _compute_mode_metrics(
        self,
        *,
        traj_cache: dict[str, np.ndarray],
        base_points: np.ndarray,
        params: dict[str, float | int],
        frame_name: str,
        sample_idx: np.ndarray | None,
        time_backward: bool | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, np.ndarray]:
        x_key = "x_body" if frame_name == "body" else "x_lab"
        x_series = traj_cache.get(x_key)
        if x_series is None:
            return super()._compute_mode_metrics(
                traj_cache=traj_cache,
                base_points=base_points,
                params=params,
                frame_name=frame_name,  # type: ignore[arg-type]
                sample_idx=sample_idx,
                time_backward=time_backward,
                cancel_check=cancel_check,
            )

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

        axis_final = np.asarray(z_series[-1], dtype=np.float64)
        axis_final_n = float(np.linalg.norm(axis_final))
        if axis_final_n < 1e-12:
            axis_final = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            axis_final_n = 1.0
        axis_final_u = axis_final / max(axis_final_n, 1e-12)
        entropy_kappa = 14.0

        for t in range(t_count):
            if cancel_check is not None and bool(cancel_check()):
                raise _HydroRecomputeCancelled("Cucker-Smale ensemble metrics computation cancelled.")
            pts = x_series[t]
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
                entropy[t] = float(np.clip(np.mean(np.sum((perp_dyn - perp_mean) ** 2, axis=1)), 0.0, 1.0))
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

    @staticmethod
    def _to_numpy_traj_cache(traj: Any) -> tuple[dict[str, np.ndarray], np.ndarray]:
        base_points_np = np.ascontiguousarray(np.asarray(traj.base_points.detach().cpu().numpy(), dtype=np.float32))
        traj_cache = {
            "w": np.ascontiguousarray(np.asarray(traj.w.detach().cpu().numpy(), dtype=np.float32)),
            "zeta": np.ascontiguousarray(np.asarray(traj.zeta.detach().cpu().numpy(), dtype=np.float32)),
            "z_lab": np.ascontiguousarray(np.asarray(traj.z.detach().cpu().numpy(), dtype=np.float32)),
            "z_body": np.ascontiguousarray(np.asarray((-traj.w).detach().cpu().numpy(), dtype=np.float32)),
            "Z_lab": np.ascontiguousarray(np.asarray(traj.Z.detach().cpu().numpy(), dtype=np.float32)),
            "Z_body": np.ascontiguousarray(np.asarray(traj.Z_body.detach().cpu().numpy(), dtype=np.float32)),
        }
        if traj.x_lab is not None:
            traj_cache["x_lab"] = np.ascontiguousarray(np.asarray(traj.x_lab.detach().cpu().numpy(), dtype=np.float32))
        if traj.x_body is not None:
            traj_cache["x_body"] = np.ascontiguousarray(np.asarray(traj.x_body.detach().cpu().numpy(), dtype=np.float32))
        return traj_cache, base_points_np

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
                raise _HydroRecomputeCancelled("Cucker-Smale hydro recompute cancelled before simulation.")
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
                raise _HydroRecomputeCancelled("Cucker-Smale hydro recompute cancelled after simulation.")

            traj_cache, base_points_np = self._to_numpy_traj_cache(traj)
            x_ref = traj_cache.get("x_lab")
            n_points = int(x_ref.shape[1]) if x_ref is not None else int(base_points_np.shape[0])
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

            traj_cache, base_points_np = self._to_numpy_traj_cache(traj)
            x_ref = traj_cache.get("x_lab")
            n_points = int(x_ref.shape[1]) if x_ref is not None else int(base_points_np.shape[0])
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
