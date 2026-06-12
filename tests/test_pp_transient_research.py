import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from lmsspp.dynamics.pp_cs_equilibria import (
    FFTPeszekPoyato2D,
    InitialCondition,
    SimulationConfig,
    _adaptive_step_factor,
    _cfl_limited_dt,
    _clamp_dt,
    _dt_history_summary,
    _predictive_tau_theta_from_config,
    _predictive_theta_from_config,
    compute_finite_horizon_negative_velocities,
    finite_horizon_gauge_average_field,
    finite_horizon_negative_velocity_at,
    make_initial_condition,
    run_finite_horizon_gauge_averaged_simulation,
    run_simulation,
)
from lmsspp.dynamics.pp_transient_research import (
    RESEARCH_DIAGNOSTIC_FIELDS,
    DirectPeszekPoyato2D,
    TransientResearchConfig,
    compute_morphology_metrics,
    evaluate_A_and_H_at_particles,
    nearest_neighbor_quantiles,
    run_finite_horizon_animation_batch,
    run_finite_horizon_comparison,
    run_long_cross_reference,
    run_pp_research_sweep,
    run_predictive_velocity_animation_batch,
    run_research_simulation,
    _smooth_delayed_zoom_scales,
)


def _direct_A(query: np.ndarray, sources: np.ndarray, alpha: float, K: float) -> np.ndarray:
    diff = query[:, None, :] - sources[None, :, :]
    r = np.linalg.norm(diff, axis=-1)
    mask = r > 1e-14
    scale = np.zeros_like(r)
    scale[mask] = K * (r[mask] ** (-alpha)) / (1 - alpha) / len(sources)
    return (diff * scale[..., None]).sum(axis=1)


def _reference_finite_horizon_field(
    solver: FFTPeszekPoyato2D,
    x: np.ndarray,
    omega: np.ndarray,
    tau: float,
    theta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    A_now, rho_now = solver.A_at_particles(x)
    if tau == 0.0 or theta == 0.0:
        return omega - A_now, A_now, rho_now
    x_tau = solver.clip_inside(x + tau * (omega - A_now))
    A_tau, _ = solver.A_at_particles(x_tau)
    A_model = (1.0 - theta) * A_now + theta * A_tau
    return omega - A_model, A_model, rho_now


def _reference_finite_horizon_numpy_run(
    config: SimulationConfig,
    initial: InitialCondition,
) -> dict[str, np.ndarray | int | float]:
    solver = FFTPeszekPoyato2D(config.alpha, config.K, config.grid_size, config.domain_radius)
    x = solver.clip_inside(np.asarray(initial.x, dtype=np.float64).copy())
    omega = np.asarray(initial.omega, dtype=np.float64)
    tau, theta = _predictive_tau_theta_from_config(config)
    time_sign = -1.0 if config.time_direction == "backward" else 1.0
    accepted_steps = 0
    rejected_steps = 0
    dt_current = _clamp_dt(float(config.dt), config)
    dt_history: list[float] = []
    t = 0.0

    if config.integrator == "fixed_rk2":
        dt_fixed = float(config.dt)
        for _ in range(config.max_steps + 1):
            vf, _, _ = _reference_finite_horizon_field(solver, x, omega, tau, theta)
            rms, _ = solver.speed_stats(vf)
            if rms < config.tol_rms and accepted_steps > config.min_steps:
                break
            if accepted_steps >= config.max_steps:
                break
            x_pred = solver.clip_inside(x + time_sign * dt_fixed * vf)
            k2, _, _ = _reference_finite_horizon_field(solver, x_pred, omega, tau, theta)
            x = solver.center(x + 0.5 * time_sign * dt_fixed * (vf + k2))
            x = solver.clip_inside(x)
            t += dt_fixed
            accepted_steps += 1
            dt_history.append(dt_fixed)
    else:
        while True:
            vf, _, _ = _reference_finite_horizon_field(solver, x, omega, tau, theta)
            rms, maxv = solver.speed_stats(vf)
            if rms < config.tol_rms and accepted_steps > config.min_steps:
                break
            if accepted_steps >= config.max_steps:
                break
            trial_dt = _cfl_limited_dt(dt_current, maxv, solver.h, config)
            while True:
                x_pred = solver.clip_inside(x + time_sign * trial_dt * vf)
                k2, _, _ = _reference_finite_horizon_field(solver, x_pred, omega, tau, theta)
                x_euler = x + time_sign * trial_dt * vf
                x_heun = x + 0.5 * time_sign * trial_dt * (vf + k2)
                local_err = solver.rms_delta(x_heun, x_euler)
                if not np.isfinite(local_err):
                    rejected_steps += 1
                    trial_dt = max(float(config.dt_min), trial_dt * _adaptive_step_factor(local_err, config.adaptive_tol, grow=False))
                    continue
                if local_err <= config.adaptive_tol or trial_dt <= config.dt_min * (1 + 1e-12):
                    break
                rejected_steps += 1
                trial_dt = max(float(config.dt_min), trial_dt * _adaptive_step_factor(local_err, config.adaptive_tol, grow=False))
            x = solver.center(x_heun)
            x = solver.clip_inside(x)
            t += trial_dt
            accepted_steps += 1
            dt_history.append(trial_dt)
            dt_current = _clamp_dt(trial_dt * _adaptive_step_factor(local_err, config.adaptive_tol, grow=True), config)

    residual, A_bar, rho_grid = _reference_finite_horizon_field(solver, x, omega, tau, theta)
    dt_min, dt_max, dt_mean = _dt_history_summary(dt_history)
    return {
        "x_final": x,
        "A_final": A_bar,
        "residual": residual,
        "rho_grid": rho_grid,
        "steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "final_time": t,
        "dt_min": dt_min,
        "dt_max": dt_max,
        "dt_mean": dt_mean,
    }


class PPTransientResearchTests(unittest.TestCase):
    def test_dynamic_zoom_scales_are_smoothed_delayed_and_non_clipping(self) -> None:
        raw = np.array([8.0, 8.0, 8.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        delayed = _smooth_delayed_zoom_scales(raw, smoothing_radius=1, delay_frames=2)
        self.assertEqual(delayed.shape, raw.shape)
        self.assertTrue(np.all(delayed >= raw))
        self.assertGreater(delayed[3], raw[3])
        self.assertGreater(delayed[4], raw[4])
        self.assertLess(delayed[-1], delayed[4])

    def test_predictive_modes_use_expected_A_tau_combination(self) -> None:
        rng = np.random.default_rng(16)
        x = rng.normal(scale=0.35, size=(8, 2))
        omega = rng.normal(scale=0.2, size=(8, 2))
        solver = FFTPeszekPoyato2D(alpha=0.45, K=0.7, grid_size=24, domain_radius=3.0)
        tau = 0.03
        A_now, _ = solver.A_at_particles(x)
        x_tau = solver.clip_inside(x + tau * (omega - A_now))
        A_tau, _ = solver.A_at_particles(x_tau)
        v_avg, A_avg, _ = finite_horizon_gauge_average_field(
            solver,
            x,
            omega,
            tau,
            "averaged_predictive",
        )
        v_pure, A_pure, _ = finite_horizon_gauge_average_field(
            solver,
            x,
            omega,
            tau,
            "pure_predictive",
        )
        v_quarter, A_quarter, _ = finite_horizon_gauge_average_field(
            solver,
            x,
            omega,
            tau,
            0.25,
        )
        v_fraction, A_fraction, _ = finite_horizon_gauge_average_field(
            solver,
            x,
            omega,
            tau,
            "1/2",
        )
        np.testing.assert_allclose(A_avg, 0.5 * (A_now + A_tau), rtol=0.0, atol=1e-13)
        np.testing.assert_allclose(v_avg, omega - 0.5 * (A_now + A_tau), rtol=0.0, atol=1e-13)
        np.testing.assert_allclose(A_pure, A_tau, rtol=0.0, atol=1e-13)
        np.testing.assert_allclose(v_pure, omega - A_tau, rtol=0.0, atol=1e-13)
        np.testing.assert_allclose(A_quarter, 0.75 * A_now + 0.25 * A_tau, rtol=0.0, atol=1e-13)
        np.testing.assert_allclose(v_quarter, omega - (0.75 * A_now + 0.25 * A_tau), rtol=0.0, atol=1e-13)
        np.testing.assert_allclose(A_fraction, A_avg, rtol=0.0, atol=1e-13)

    def test_predictive_pp_weight_config_parses_numeric_fractions(self) -> None:
        base = SimulationConfig()
        self.assertAlmostEqual(_predictive_theta_from_config(replace(base, predictive_pp_weight="1/2")), 0.5)
        self.assertAlmostEqual(_predictive_theta_from_config(replace(base, predictive_pp_weight=0.25)), 0.25)
        self.assertAlmostEqual(_predictive_theta_from_config(replace(base, predictive_pp_weight="0.75")), 0.75)
        self.assertAlmostEqual(_predictive_theta_from_config(replace(base, predictive_pp_weight=-0.25)), 0.25)
        self.assertEqual(
            _predictive_tau_theta_from_config(replace(base, prediction_horizon_tau=0.2, predictive_pp_weight=-0.25)),
            (-0.2, 0.25),
        )
        with self.assertRaisesRegex(ValueError, "\\[-1, 1\\]"):
            _predictive_theta_from_config(replace(base, predictive_pp_weight=1.5))
        with self.assertRaisesRegex(TypeError, "numeric"):
            _predictive_theta_from_config(replace(base, predictive_pp_weight="not-a-number"))

    def test_convolve_fields_matches_scalar_fft_reference(self) -> None:
        rng = np.random.default_rng(19)
        x = rng.normal(scale=0.35, size=(10, 2))
        solver = FFTPeszekPoyato2D(alpha=0.45, K=0.7, grid_size=24, domain_radius=3.0)
        rho = solver.A_grid_from_particles(x)[0]
        padded = np.zeros((solver.P, solver.P), dtype=np.float64)
        padded[: solver.G, : solver.G] = rho
        rho_hat = np.fft.rfft2(padded)

        fields = solver.convolve_fields(rho, (solver.fft_Kx, solver.fft_Ky, solver.fft_Hxx))
        reference = tuple(
            np.fft.irfft2(rho_hat * kernel, s=(solver.P, solver.P))[: solver.G, : solver.G]
            for kernel in (solver.fft_Kx, solver.fft_Ky, solver.fft_Hxx)
        )
        for actual, expected in zip(fields, reference):
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    def test_finite_horizon_integration_matches_reference_loop(self) -> None:
        base = SimulationConfig(
            n_fibers=2,
            n_per_fiber=5,
            grid_size=16,
            domain_radius=3.0,
            alpha=0.45,
            K=0.7,
            max_steps=3,
            min_steps=100,
            dt=0.01,
            backend="numpy",
            prediction_horizon_tau=0.03,
            predictive_pp_weight=0.65,
            make_dashboard=False,
            make_animation=False,
            seed=22,
        )
        initial = make_initial_condition(base)
        for integrator in ("fixed_rk2", "adaptive_rk2"):
            config = replace(base, integrator=integrator)
            result = run_finite_horizon_gauge_averaged_simulation(config, initial)
            reference = _reference_finite_horizon_numpy_run(config, initial)
            np.testing.assert_allclose(result.x_final, reference["x_final"], rtol=0.0, atol=1e-13)
            np.testing.assert_allclose(result.A_final, reference["A_final"], rtol=0.0, atol=1e-13)
            np.testing.assert_allclose(result.residual, reference["residual"], rtol=0.0, atol=1e-13)
            np.testing.assert_allclose(result.rho_grid, reference["rho_grid"], rtol=0.0, atol=1e-13)
            self.assertEqual(result.steps, reference["steps"])
            self.assertEqual(result.rejected_steps, reference["rejected_steps"])
            self.assertAlmostEqual(result.final_time, float(reference["final_time"]))
            self.assertAlmostEqual(result.dt_mean, float(reference["dt_mean"]))

    def test_finite_horizon_tau_zero_recovers_ordinary_pp(self) -> None:
        config = SimulationConfig(
            n_fibers=2,
            n_per_fiber=4,
            grid_size=16,
            domain_radius=3.0,
            alpha=0.45,
            K=0.7,
            max_steps=2,
            min_steps=100,
            dt=0.01,
            backend="numpy",
            integrator="fixed_rk2",
            prediction_horizon_tau=0.0,
            make_dashboard=False,
            make_animation=False,
            seed=21,
        )
        initial = make_initial_condition(config)
        ordinary = run_simulation(config, initial)
        finite_horizon = run_finite_horizon_gauge_averaged_simulation(config, initial)
        np.testing.assert_allclose(finite_horizon.x_final, ordinary.x_final, rtol=0.0, atol=1e-13)
        np.testing.assert_allclose(finite_horizon.residual, ordinary.residual, rtol=0.0, atol=1e-13)

    def test_direct_backend_matches_direct_formula_and_runs(self) -> None:
        rng = np.random.default_rng(17)
        x = rng.normal(scale=0.5, size=(24, 2))
        solver = DirectPeszekPoyato2D(alpha=0.45, K=0.8, grid_size=24, domain_radius=3.0)
        A, rho = solver.A_at_particles(x)
        np.testing.assert_allclose(A, _direct_A(x, x, 0.45, 0.8), rtol=0.0, atol=1e-13)
        self.assertEqual(rho.shape, (24, 24))

        config = TransientResearchConfig(
            n_fibers=2,
            n_per_fiber=6,
            grid_size=24,
            domain_radius=3.0,
            max_steps=2,
            min_steps=100,
            dt=0.01,
            backend="numpy",
            force_backend="direct",
            integrator="fixed_rk2",
            make_dashboard=False,
            make_animation=False,
            seed=18,
        )
        result = run_research_simulation(config)
        self.assertEqual(result.backend, "direct")
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.research_diagnostics, None)

    def test_research_diagnostics_records_spacing_hessian_residuals_and_scores(self) -> None:
        config = TransientResearchConfig(
            n_fibers=2,
            n_per_fiber=6,
            grid_size=24,
            domain_radius=3.0,
            max_steps=2,
            min_steps=100,
            dt=0.01,
            dt_min=0.01,
            dt_max=0.01,
            max_displacement_per_step=0.0,
            backend="numpy",
            integrator="fixed_rk2",
            make_dashboard=False,
            make_animation=False,
            record_research_diagnostics=True,
            record_every=1,
            research_diagnostic_sample_size=20,
            research_energy_sample_size=20,
            seed=19,
        )
        result = run_research_simulation(config)
        self.assertIsNotNone(result.research_diagnostics)
        assert result.research_diagnostics is not None
        self.assertEqual(result.research_diagnostics.shape[1], len(RESEARCH_DIAGNOSTIC_FIELDS))
        self.assertGreaterEqual(result.research_diagnostics.shape[0], 2)
        q_max = result.research_diagnostics[:, RESEARCH_DIAGNOSTIC_FIELDS.index("q_max")]
        disk_radius = result.research_diagnostics[:, RESEARCH_DIAGNOSTIC_FIELDS.index("disk_radius")]
        self.assertTrue(np.all(np.isfinite(q_max)))
        self.assertTrue(np.all(disk_radius >= 0.0))

        evaluated = evaluate_A_and_H_at_particles(config, result.x_final)
        self.assertEqual(evaluated["H"].shape, (len(result.x_final), 2, 2))
        self.assertTrue(np.all(np.isfinite(evaluated["lambda_max"])))
        self.assertGreaterEqual(compute_morphology_metrics(result.x_final)["fourfold_mode"], 0.0)
        self.assertGreaterEqual(nearest_neighbor_quantiles(result.x_final)["r_min"], 0.0)

    def test_center_each_step_flag_controls_barycenter_projection(self) -> None:
        initial = InitialCondition(
            x=np.array([[-0.2, 0.0], [0.2, 0.0]], dtype=np.float64),
            omega=np.array([[0.1, 0.0], [0.1, 0.0]], dtype=np.float64),
            group_id=np.array([0, 0], dtype=np.int64),
            omega_atoms=np.array([[0.1, 0.0]], dtype=np.float64),
            group_names=("custom",),
        )
        base = TransientResearchConfig(
            alpha=0.5,
            K=0.0,
            n_fibers=1,
            n_per_fiber=2,
            grid_size=16,
            domain_radius=2.0,
            max_steps=1,
            min_steps=100,
            dt=0.1,
            backend="numpy",
            force_backend="direct",
            integrator="fixed_rk2",
            clip_each_step=False,
            make_dashboard=False,
            make_animation=False,
        )
        centered = run_research_simulation(replace(base, center_each_step=True), initial)
        uncentered = run_research_simulation(replace(base, center_each_step=False), initial)
        np.testing.assert_allclose(centered.x_final.mean(axis=0), 0.0, rtol=0.0, atol=1e-14)
        self.assertGreater(float(uncentered.x_final.mean(axis=0)[0]), 5e-3)

    def test_tiny_research_sweep_writes_case_and_summary_artifacts(self) -> None:
        config = TransientResearchConfig(
            n_fibers=2,
            n_per_fiber=4,
            grid_size=16,
            domain_radius=3.0,
            max_steps=1,
            min_steps=100,
            dt=0.01,
            dt_min=0.01,
            dt_max=0.02,
            max_displacement_per_step=0.0,
            backend="numpy",
            integrator="fixed_rk2",
            make_dashboard=False,
            make_animation=False,
            record_every=1,
            research_diagnostic_sample_size=20,
            research_energy_sample_size=20,
            seed=20,
        )
        with tempfile.TemporaryDirectory() as tmp:
            summaries = run_pp_research_sweep(config, "timestep", values=(0.01, 0.005), out_dir=tmp)
            self.assertEqual(len(summaries), 2)
            self.assertTrue((Path(tmp) / "sweep_metrics.json").exists())
            self.assertTrue((Path(tmp) / "sweep_summary.html").exists())
            self.assertTrue((Path(tmp) / "case_00_0p01" / "research_diagnostics.csv").exists())

    def test_tiny_finite_horizon_comparison_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_finite_horizon_comparison(
                tmp,
                n_fibers=2,
                n_per_fiber=3,
                alpha=0.45,
                K=0.2,
                grid_size=16,
                domain_radius=3.0,
                tau=0.01,
                old_fixed_steps=1,
                adaptive_steps_per_horizon=1,
                seed=22,
            )
            self.assertIn("new_model_residual_rms", metrics)
            self.assertTrue((Path(tmp) / "comparison_metrics.json").exists())
            self.assertTrue((Path(tmp) / "finite_horizon_vs_fixed_rk2.html").exists())
            self.assertTrue((Path(tmp) / "finite_horizon_residual_split.html").exists())

    def test_tiny_animation_batch_supports_adaptive_pp_right_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summaries = run_finite_horizon_animation_batch(
                tmp,
                right_model="ordinary_pp_adaptive",
                left_dynamic_zoom=True,
                right_dynamic_zoom=True,
                left_zoom_smoothing_radius=3,
                left_zoom_delay_frames=0,
                n_fibers=2,
                n_per_fiber=3,
                grid_size=16,
                domain_radius=3.0,
                old_fixed_steps=1,
                adaptive_steps=1,
                animation_frames=2,
                fps=2,
                record_every=1,
                research_diagnostics_every=1,
                research_diagnostic_sample_size=20,
                research_energy_sample_size=20,
                research_nn_chunk=32,
                cases=[{"label": "tiny_pp_adaptive", "seed": 25, "alpha": 0.45, "K": 0.2, "tau": 0.01}],
            )
            case_dir = Path(tmp) / "tiny_pp_adaptive"
            self.assertEqual(summaries[0]["right_model"], "ordinary_pp_adaptive")
            self.assertTrue((case_dir / "time_aligned_fixed_vs_adaptive_pp.mp4").exists())
            self.assertTrue((case_dir / "fixed_vs_adaptive_pp.png").exists())
            self.assertTrue((case_dir / "adaptive_pp_residual_split.html").exists())
            self.assertTrue((case_dir / "adaptive_rk2_pp" / "metrics.json").exists())

    def test_finite_horizon_negative_velocity_matches_widget_field(self) -> None:
        rng = np.random.default_rng(18)
        x = rng.normal(scale=0.35, size=(8, 2))
        omega = rng.normal(scale=0.2, size=(8, 2))
        config = SimulationConfig(
            alpha=0.45,
            K=0.7,
            grid_size=24,
            domain_radius=3.0,
            prediction_horizon_tau=0.03,
            predictive_pp_weight=0.65,
        )
        solver = FFTPeszekPoyato2D(config.alpha, config.K, config.grid_size, config.domain_radius)
        velocity, _, _ = finite_horizon_gauge_average_field(
            solver,
            x,
            omega,
            config.prediction_horizon_tau,
            _predictive_theta_from_config(config),
        )
        np.testing.assert_allclose(
            finite_horizon_negative_velocity_at(config, x, omega),
            -velocity,
            rtol=0.0,
            atol=1e-13,
        )
        trajectory = np.stack([x, x + 0.01], axis=0)
        batch = compute_finite_horizon_negative_velocities(config, trajectory, omega)
        expected_batch = np.stack(
            [finite_horizon_negative_velocity_at(config, trajectory[f], omega) for f in range(trajectory.shape[0])],
            axis=0,
        )
        np.testing.assert_allclose(batch, expected_batch, rtol=0.0, atol=1e-13)

    def test_tiny_predictive_velocity_animation_batch_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summaries = run_predictive_velocity_animation_batch(
                tmp,
                left_dynamic_zoom=True,
                right_dynamic_zoom=True,
                n_fibers=2,
                n_per_fiber=3,
                grid_size=16,
                domain_radius=3.0,
                adaptive_steps=1,
                adaptive_steps_per_horizon=1,
                animation_frames=2,
                fps=1,
                cases=[
                    {
                        "label": "tiny_theta",
                        "seed": 25,
                        "alpha": 0.45,
                        "K": 0.2,
                        "tau": 0.01,
                        "predictive_pp_weight": 0.4,
                    }
                ],
            )
            case_dir = Path(tmp) / "tiny_theta"
            self.assertAlmostEqual(float(summaries[0]["predictive_pp_weight"]), 0.4)
            self.assertTrue((case_dir / "time_aligned_morphology_vs_negative_velocity.mp4").exists())
            self.assertTrue((case_dir / "morphology_vs_negative_velocity.png").exists())
            self.assertTrue((case_dir / "predictive_system_statistics.png").exists())
            self.assertTrue((Path(tmp) / "REPORT_PREDICTIVE_VELOCITY_BATCH.md").exists())

    def test_tiny_long_cross_reference_writes_zoom_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_long_cross_reference(
                tmp,
                n_fibers=2,
                n_per_fiber=3,
                alpha=0.45,
                K=0.2,
                grid_size=16,
                domain_radius=3.0,
                dt=0.01,
                dt_min=0.01,
                dt_max=0.01,
                max_steps=1,
                min_steps=2,
                seed=23,
                trajectory_frame_count=2,
                record_every=1,
                research_diagnostics_every=1,
                research_diagnostic_sample_size=20,
                research_energy_sample_size=20,
                research_nn_chunk=32,
            )
            self.assertEqual(metrics["experiment"], "long_cross_reference")
            self.assertTrue((Path(tmp) / "REPORT_LONG_1.md").exists())
            self.assertTrue((Path(tmp) / "fig_long_1_final_full_and_zoom.png").exists())
            self.assertTrue((Path(tmp) / "fig_long_1_montage_fixed_rk2.png").exists())
            self.assertTrue((Path(tmp) / "fig_long_1_montage_adaptive_rk2.png").exists())
            self.assertTrue((Path(tmp) / "fig_long_1_adaptive_cross_details.png").exists())
            self.assertTrue((Path(tmp) / "long_cross_reference_metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
