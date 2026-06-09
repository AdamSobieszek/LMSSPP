import unittest
import time
from dataclasses import replace
from unittest import mock as unittest_mock

import numpy as np

from lmsspp.dynamics.pp_cs_equilibria import (
    DENSITY_DIAGNOSTIC_FIELDS,
    FFTPeszekPoyato2D,
    FFTPeszekPoyatoDensity2D,
    InitializerConfig,
    PeszekPoyatoDynamicsBaseWidget,
    SimulationConfig,
    TorchPeszekPoyato2D,
    _bilinear_resample_field,
    _density_display_payload_from_grid,
    _density_edge_mass_fraction,
    _density_grid_axis,
    _nyquist_checkerboard_amplitude,
    _density_support_window,
    _effective_density_display_grid_size,
    _make_pp_backend,
    fiber_colors,
    interp_grid,
    make_continuous_density_widget,
    make_density_initial_condition,
    make_dynamics_widget,
    make_initial_condition,
    run_density_simulation,
    run_simulation,
    torch,
    widgets,
)


def _brute_grid_convolution(rho: np.ndarray, kernel: np.ndarray, G: int, P: int) -> np.ndarray:
    out = np.zeros((G, G), dtype=np.float64)
    for i in range(G):
        for j in range(G):
            total = 0.0
            for a in range(G):
                for b in range(G):
                    total += rho[a, b] * kernel[(i - a) % P, (j - b) % P]
            out[i, j] = total
    return out


def _direct_A(query: np.ndarray, sources: np.ndarray, alpha: float, K: float) -> np.ndarray:
    diff = query[:, None, :] - sources[None, :, :]
    r = np.linalg.norm(diff, axis=-1)
    mask = r > 1e-14
    scale = np.zeros_like(r)
    scale[mask] = K * (r[mask] ** (-alpha)) / (1 - alpha) / len(sources)
    return (diff * scale[..., None]).sum(axis=1)


def _hex_rgb(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


class PPBackendTests(unittest.TestCase):
    def test_numpy_fft_convolution_matches_brute_grid_convolution(self) -> None:
        rng = np.random.default_rng(11)
        solver = FFTPeszekPoyato2D(alpha=0.5, K=1.0, grid_size=8, domain_radius=3.0)
        rho = rng.random((solver.G, solver.G))
        rho /= rho.sum()
        kernels = solver._build_kernels()

        for fft_kernel, kernel in ((solver.fft_Kx, kernels[0]), (solver.fft_Ky, kernels[1])):
            np.testing.assert_allclose(
                solver.convolve(rho, fft_kernel),
                _brute_grid_convolution(rho, kernel, solver.G, solver.P),
                rtol=0.0,
                atol=2e-14,
            )

    @unittest.skipIf(torch is None, "torch optional dependency is unavailable")
    def test_torch_cpu_backend_matches_numpy_field_and_hessian(self) -> None:
        rng = np.random.default_rng(12)
        x = rng.normal(scale=0.8, size=(180, 2))
        x -= x.mean(axis=0, keepdims=True)

        numpy_solver = FFTPeszekPoyato2D(alpha=0.55, K=0.9, grid_size=32, domain_radius=4.5)
        torch_solver = TorchPeszekPoyato2D(
            alpha=0.55,
            K=0.9,
            grid_size=32,
            domain_radius=4.5,
            device="cpu",
            dtype=torch.float64,
        )

        A_np, rho_np = numpy_solver.A_at_particles(x)
        A_t, rho_t = torch_solver.A_at_particles(torch_solver.asarray(x))
        np.testing.assert_allclose(torch_solver.to_numpy(rho_t), rho_np, rtol=0.0, atol=2e-14)
        np.testing.assert_allclose(torch_solver.to_numpy(A_t), A_np, rtol=0.0, atol=2e-13)

        H_np = numpy_solver.hessian_grid_from_rho(rho_np)
        H_t = torch_solver.hessian_grid_from_rho(rho_t)
        for actual, expected in zip(H_t, H_np, strict=True):
            np.testing.assert_allclose(torch_solver.to_numpy(actual), expected, rtol=0.0, atol=2e-12)

    def test_cic_fft_particle_field_converges_with_grid_refinement(self) -> None:
        rng = np.random.default_rng(13)
        x = rng.normal(scale=0.75, size=(220, 2))
        x -= x.mean(axis=0, keepdims=True)
        query = rng.normal(scale=0.65, size=(80, 2))
        alpha = 0.5
        K = 1.0
        direct = _direct_A(query, x, alpha, K)

        errors = []
        for G in (32, 64, 128):
            solver = FFTPeszekPoyato2D(alpha=alpha, K=K, grid_size=G, domain_radius=4.0)
            _, Ax, Ay = solver.A_grid_from_particles(x)
            fft_A = np.c_[interp_grid(Ax, query, solver.G, solver.L), interp_grid(Ay, query, solver.G, solver.L)]
            errors.append(float(np.sqrt(np.mean(np.sum((fft_A - direct) ** 2, axis=1)))))

        self.assertLess(errors[1], 0.55 * errors[0])
        self.assertLess(errors[2], 0.55 * errors[1])

    def test_adaptive_rk2_matches_fixed_rk2_when_dt_is_fixed(self) -> None:
        base = SimulationConfig(
            n_fibers=3,
            n_per_fiber=12,
            grid_size=32,
            domain_radius=5.0,
            dt=0.025,
            dt_min=0.025,
            dt_max=0.025,
            max_steps=6,
            min_steps=100,
            max_displacement_per_step=0.0,
            backend="numpy",
            make_dashboard=False,
            make_animation=False,
            seed=14,
        )
        initial = make_initial_condition(base)
        fixed = run_simulation(replace(base, integrator="fixed_rk2"), initial)
        adaptive = run_simulation(replace(base, integrator="adaptive_rk2", adaptive_tol=1.0), initial)

        np.testing.assert_allclose(adaptive.x_final, fixed.x_final, rtol=0.0, atol=1e-12)
        self.assertLessEqual(adaptive.field_evaluations, fixed.field_evaluations)
        self.assertEqual(adaptive.rejected_steps, 0)

    def test_backward_time_uses_opposite_vector_field(self) -> None:
        config = SimulationConfig(
            n_fibers=2,
            n_per_fiber=8,
            grid_size=24,
            domain_radius=5.0,
            dt=0.01,
            dt_min=0.01,
            dt_max=0.01,
            max_steps=1,
            min_steps=100,
            max_displacement_per_step=0.0,
            backend="numpy",
            integrator="fixed_rk2",
            make_dashboard=False,
            make_animation=False,
            seed=15,
        )
        initial = make_initial_condition(config)
        forward = run_simulation(replace(config, time_direction="forward"), initial)
        backward = run_simulation(replace(config, time_direction="backward"), initial)
        forward_delta = forward.x_final - forward.x_initial
        backward_delta = backward.x_final - backward.x_initial

        self.assertLess(float(np.sum(forward_delta * backward_delta)), 0.0)
        self.assertEqual(backward.steps, 1)
        self.assertEqual(backward.final_time, config.dt)

    @unittest.skipIf(widgets is None, "ipywidgets optional dependency is unavailable")
    def test_precompute_completes_when_event_loop_only_queues_callbacks(self) -> None:
        cfg = SimulationConfig(
            n_fibers=2,
            n_per_fiber=12,
            grid_size=24,
            domain_radius=4.0,
            max_steps=20,
            make_animation=True,
            trajectory_frame_count=5,
            seed=21,
        )
        try:
            widget = PeszekPoyatoDynamicsBaseWidget(cfg, width=400, height=300, second_panel="heatmap")
        except ImportError as exc:
            self.skipTest(str(exc))

        queued: list[Any] = []
        io_loop = unittest_mock.MagicMock()
        io_loop.add_callback = queued.append
        kernel = unittest_mock.MagicMock()
        kernel.io_loop = io_loop
        ipython = unittest_mock.MagicMock()
        ipython.kernel = kernel

        with unittest_mock.patch("IPython.get_ipython", return_value=ipython):
            widget._on_precompute_clicked(None)
            deadline = time.time() + 15.0
            while widget._precompute_busy and time.time() < deadline:
                time.sleep(0.05)
        self.assertFalse(widget._precompute_busy)
        self.assertTrue(widget._cache_valid)
        self.assertGreater(len(widget._frame_payloads), 0)
        self.assertEqual(widget.btn_precompute.description, "Precompute flow")

    @unittest.skipIf(widgets is None, "ipywidgets optional dependency is unavailable")
    def test_precompute_interrupt_resets_button_immediately(self) -> None:
        cfg = SimulationConfig(
            n_fibers=2,
            n_per_fiber=20,
            grid_size=24,
            domain_radius=4.0,
            max_steps=100_000,
            min_steps=100_000,
            tol_rms=0.0,
            make_animation=True,
            trajectory_frame_count=0,
            seed=22,
        )
        try:
            widget = PeszekPoyatoDynamicsBaseWidget(cfg, width=400, height=300, second_panel="heatmap")
        except ImportError as exc:
            self.skipTest(str(exc))
        widget._on_precompute_clicked(None)
        time.sleep(0.05)
        widget._on_precompute_clicked(None)
        self.assertEqual(widget.btn_precompute.description, "Precompute flow")
        self.assertEqual(widget.btn_precompute.button_style, "warning")
        self.assertFalse(widget._cache_valid)

    @unittest.skipIf(widgets is None, "ipywidgets optional dependency is unavailable")
    def test_stale_cache_keeps_playback_controls_enabled(self) -> None:
        cfg = SimulationConfig(
            n_fibers=2,
            n_per_fiber=12,
            grid_size=24,
            domain_radius=4.0,
            max_steps=20,
            make_animation=True,
            trajectory_frame_count=5,
            seed=23,
        )
        try:
            widget = PeszekPoyatoDynamicsBaseWidget(cfg, width=400, height=300, second_panel="heatmap")
        except ImportError as exc:
            self.skipTest(str(exc))
        widget._on_precompute_clicked(None)
        deadline = time.time() + 15.0
        while widget._precompute_busy and time.time() < deadline:
            time.sleep(0.05)
        self.assertTrue(widget._cache_valid)
        self.assertGreater(len(widget._frame_payloads), 1)
        widget._mark_cache_stale("Parameters changed.")
        self.assertFalse(widget._cache_valid)
        self.assertFalse(widget.play.disabled)
        self.assertFalse(widget.frame_slider.disabled)
        self.assertFalse(widget.btn_step.disabled)

    def test_run_simulation_cancel_check_interrupts_long_run(self) -> None:
        config = SimulationConfig(
            n_fibers=2,
            n_per_fiber=8,
            grid_size=24,
            domain_radius=5.0,
            dt=0.01,
            dt_min=0.01,
            dt_max=0.01,
            max_steps=10_000,
            min_steps=10_000,
            tol_rms=0.0,
            max_displacement_per_step=0.0,
            backend="numpy",
            integrator="fixed_rk2",
            make_dashboard=False,
            make_animation=False,
            seed=16,
        )
        calls = {"n": 0}

        def cancel_check() -> bool:
            calls["n"] += 1
            return calls["n"] > 3

        with self.assertRaises(InterruptedError):
            run_simulation(config, cancel_check=cancel_check)
        self.assertGreater(calls["n"], 3)

    def test_legacy_fast_phase_initializer_discards_warmup_history(self) -> None:
        config = SimulationConfig(
            n_fibers=3,
            n_per_fiber=10,
            grid_size=32,
            domain_radius=5.0,
            dt=0.02,
            max_steps=0,
            min_steps=100,
            backend="numpy",
            integrator="fixed_rk2",
            initialization_algorithm="legacy_fast_phase",
            initialization_fast_steps=3,
            initialization_fast_min_steps=99,
            make_dashboard=False,
            make_animation=True,
            seed=16,
        )
        raw = make_initial_condition(replace(config, initialization_algorithm="raw"))
        result = run_simulation(config, raw)

        self.assertEqual(result.initialization_algorithm, "legacy_fast_phase")
        self.assertEqual(result.initialization_steps, 3)
        self.assertEqual(result.steps, 0)
        self.assertEqual(result.trajectory_x.shape[0], 1)
        self.assertEqual(int(result.trajectory_steps[0]), 0)
        np.testing.assert_allclose(result.x_initial, result.initial.x)
        self.assertGreater(float(np.sqrt(np.mean(np.sum((result.x_initial - raw.x) ** 2, axis=1)))), 1e-6)

    def test_legacy_fast_phase_uses_initializer_alpha_not_simulation_alpha(self) -> None:
        init_config = InitializerConfig(alpha=0.99, max_steps=2, min_steps=99)
        base = SimulationConfig(
            n_fibers=3,
            n_per_fiber=12,
            grid_size=32,
            domain_radius=5.0,
            max_steps=0,
            min_steps=100,
            backend="numpy",
            integrator="fixed_rk2",
            initialization_algorithm="legacy_fast_phase",
            initializer_config=init_config,
            make_dashboard=False,
            make_animation=True,
            seed=23,
        )
        raw = make_initial_condition(replace(base, initialization_algorithm="raw"))

        low_alpha_run = run_simulation(replace(base, alpha=0.2), raw)
        high_alpha_run = run_simulation(replace(base, alpha=0.8), raw)
        changed_initializer_run = run_simulation(replace(base, initializer_config=replace(init_config, alpha=0.35)), raw)

        np.testing.assert_allclose(low_alpha_run.x_initial, high_alpha_run.x_initial, rtol=0.0, atol=1e-14)
        self.assertGreater(
            float(np.sqrt(np.mean(np.sum((low_alpha_run.x_initial - changed_initializer_run.x_initial) ** 2, axis=1)))),
            1e-7,
        )

    def test_phase_color_scheme_uses_omega_phase_and_norm_ticks(self) -> None:
        atoms = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.0, 0.2],
                [-0.3, 0.0],
                [0.0, -100.0],
            ],
            dtype=np.float64,
        )
        palette_config = SimulationConfig(color_scheme="palette")
        phase_config = SimulationConfig(color_scheme="phase_color")

        self.assertEqual(fiber_colors(palette_config, atoms, 5)[0], "#1f77b4")
        colors = fiber_colors(phase_config, atoms, 5)
        self.assertEqual(colors[0], "#000000")

        low = _hex_rgb(colors[1])
        middle = _hex_rgb(colors[3])
        high = _hex_rgb(colors[4])
        self.assertGreater(low[0], low[1])
        self.assertGreater(low[0], low[2])
        self.assertGreater(max(middle), max(low))
        self.assertGreater(max(high), max(middle))
        self.assertEqual(max(high), 255)
        self.assertGreaterEqual(max(low), 100)

    @unittest.skipIf(widgets is None, "ipywidgets optional dependency is unavailable")
    def test_widget_controls_build_runtime_config(self) -> None:
        config = SimulationConfig(
            n_fibers=2,
            n_per_fiber=4,
            grid_size=16,
            max_steps=2,
            backend="numpy",
            make_dashboard=False,
            make_animation=False,
        )
        try:
            widget = make_dynamics_widget(config, width=500, height=320)
        except ImportError as exc:
            self.skipTest(str(exc))

        widget.initialization_dropdown.value = "ring"
        widget.initializer_dropdown.value = "legacy_fast_phase"
        widget.alpha_slider.value = 0.72
        widget.n_fibers_slider.value = 5
        widget.n_per_fiber_slider.value = 7
        self.assertEqual(widget.frame_cap_slider.value, 0)
        widget.frame_cap_slider.value = 900
        widget.time_direction_toggle.value = True
        runtime = widget._config_from_controls(make_animation=True)

        self.assertEqual(runtime.shape_names, ("ring",))
        self.assertEqual(runtime.initialization_algorithm, "legacy_fast_phase")
        self.assertEqual(runtime.n_fibers, 5)
        self.assertEqual(runtime.n_per_fiber, 7)
        self.assertEqual(runtime.trajectory_frame_count, 900)
        self.assertEqual(runtime.time_direction, "backward")
        self.assertAlmostEqual(runtime.alpha, 0.72)

    @unittest.skipIf(
        torch is None or not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "MPS backend is unavailable",
    )
    def test_mps_rejects_explicit_float64(self) -> None:
        config = SimulationConfig(
            backend="torch",
            device="mps",
            dtype="float64",
            n_fibers=2,
            n_per_fiber=4,
            grid_size=16,
            make_dashboard=False,
        )
        with self.assertRaisesRegex(ValueError, "MPS does not support torch.float64"):
            _make_pp_backend(config)


def _small_density_config(**overrides) -> SimulationConfig:
    base = dict(
        n_fibers=2,
        grid_size=24,
        domain_radius=3.0,
        alpha=0.0,
        K=1.0,
        eps_entropy=0.0,
        dt=0.01,
        dt_min=0.001,
        dt_max=0.02,
        max_steps=8,
        min_steps=100,
        tol_rms=0.0,
        integrator="fixed_rk2",
        make_dashboard=False,
        make_animation=False,
        record_free_energy=True,
        record_entropy_balance=True,
        seed=31,
    )
    base.update(overrides)
    grid = int(base["grid_size"])
    display = int(overrides.get("density_display_grid_size", min(96, grid)))
    base["density_display_grid_size"] = min(display, grid)
    return SimulationConfig(**base)


class ContinuousDensityTests(unittest.TestCase):
    def test_mass_conservation_per_fiber_and_global(self) -> None:
        config = _small_density_config(max_steps=12)
        initial = make_density_initial_condition(config)
        result = run_density_simulation(config, initial)
        area = (2 * config.domain_radius / config.grid_size) ** 2
        masses = np.array([float(r.sum()) * area for r in result.r_fiber])
        np.testing.assert_allclose(masses, 1.0, rtol=0.0, atol=5e-3)
        total_mass = float(result.rho_grid.sum()) * area
        np.testing.assert_allclose(total_mass, 1.0, rtol=0.0, atol=5e-3)

    def test_positivity_on_stable_step(self) -> None:
        config = _small_density_config(max_steps=5, dt=0.005, eps_entropy=0.05)
        result = run_density_simulation(config)
        self.assertGreaterEqual(float(np.min(result.r_fiber)), 0.0)
        if result.diagnostics.size:
            negative_idx = DENSITY_DIAGNOSTIC_FIELDS.index("negative_count")
            self.assertLessEqual(float(np.max(result.diagnostics[:, negative_idx])), float(config.grid_size**2))

    def test_alpha_zero_affine_field_on_interior(self) -> None:
        config = _small_density_config(n_fibers=1, grid_size=32, domain_radius=4.0, K=2.0)
        initial = make_density_initial_condition(config)
        solver = FFTPeszekPoyatoDensity2D(config.alpha, config.K, config.grid_size, config.domain_radius)
        rho = solver.marginal_from_fibers(initial.r_fiber, initial.nu)
        Ax, Ay = solver.A_grid_from_rho(rho)
        axis = _density_grid_axis(config.grid_size, config.domain_radius)
        x1, x2 = np.meshgrid(axis, axis, indexing="ij")
        total = float(rho.sum())
        mean = np.array([float((rho * x1).sum()), float((rho * x2).sum())]) / total
        expected_x = config.K * (x1 - mean[0])
        expected_y = config.K * (x2 - mean[1])
        margin = 4
        sl = slice(margin, -margin)
        np.testing.assert_allclose(Ax[sl, sl], expected_x[sl, sl], rtol=0.0, atol=0.12)
        np.testing.assert_allclose(Ay[sl, sl], expected_y[sl, sl], rtol=0.0, atol=0.12)

    def test_entropy_diffusion_broadens_bump(self) -> None:
        config = _small_density_config(
            n_fibers=1,
            grid_size=28,
            K=0.0,
            eps_entropy=0.08,
            max_steps=15,
            dt=0.004,
        )
        initial = make_density_initial_condition(config)
        initial = replace(initial, omega=np.zeros((1, 2), dtype=np.float64))
        result = run_density_simulation(config, initial)

        def second_moment(field: np.ndarray) -> float:
            axis = _density_grid_axis(config.grid_size, config.domain_radius)
            x1, x2 = np.meshgrid(axis, axis, indexing="ij")
            h = 2 * config.domain_radius / config.grid_size
            mass = float(field.sum()) * h * h
            cx = float((field * x1).sum()) * h * h / mass
            cy = float((field * x2).sum()) * h * h / mass
            return float((((x1 - cx) ** 2 + (x2 - cy) ** 2) * field).sum() * h * h / mass)

        self.assertGreater(second_moment(result.r_fiber[0]), 1.05 * second_moment(initial.r_fiber[0]))

    def test_alpha_zero_trace_div_a_is_about_twice_k(self) -> None:
        config = _small_density_config(n_fibers=1, K=1.7, max_steps=1)
        result = run_density_simulation(config)
        trace_idx = DENSITY_DIAGNOSTIC_FIELDS.index("trace_div_A")
        trace_div_a = float(result.diagnostics[-1, trace_idx])
        np.testing.assert_allclose(trace_div_a, 2.0 * config.K, rtol=0.0, atol=0.25)

    def test_rejects_unimplemented_density_solvers(self) -> None:
        for solver in ("split_implicit_diffusion", "chang_cooper"):
            config = _small_density_config(density_solver=solver)
            with self.assertRaises(ValueError):
                run_density_simulation(config)

    def test_rejects_backward_time_with_entropy(self) -> None:
        config = _small_density_config(eps_entropy=0.01, time_direction="backward")
        with self.assertRaisesRegex(ValueError, "backward time is incompatible"):
            run_density_simulation(config)

    @unittest.skipIf(widgets is None, "ipywidgets optional dependency is unavailable")
    def test_continuous_density_widget_config_wiring(self) -> None:
        config = _small_density_config(max_steps=2, grid_size=16)
        try:
            widget = make_continuous_density_widget(config, width=500, height=320)
        except ImportError as exc:
            self.skipTest(str(exc))

        widget.alpha_slider.value = 0.25
        widget.K_slider.value = 1.3
        widget.eps_entropy_slider.value = 0.04
        widget.n_fibers_slider.value = 4
        widget.fiber_dropdown.value = 1
        widget.panel_dropdown.value = "velocity_mag"
        widget.frame_cap_slider.value = 120
        runtime = widget._config_from_controls(make_animation=True)

        self.assertAlmostEqual(runtime.alpha, 0.25)
        self.assertAlmostEqual(runtime.K, 1.3)
        self.assertAlmostEqual(runtime.eps_entropy, 0.04)
        self.assertEqual(runtime.n_fibers, 4)
        self.assertEqual(runtime.trajectory_frame_count, 120)
        self.assertEqual(runtime.density_solver, "explicit_fv")


class DensityDynamicZoomTests(unittest.TestCase):
    def test_support_window_contains_gaussian_mass(self) -> None:
        G = 48
        L = 4.0
        axis = _density_grid_axis(G, L)
        X, Y = np.meshgrid(axis, axis, indexing="ij")
        field = np.exp(-((X - 0.7) ** 2 + (Y + 0.4) ** 2) / 0.25)
        window = _density_support_window(
            field,
            axis,
            L=L,
            mass_fraction=0.995,
            margin=1.2,
            min_half_width=None,
        )
        x_lo, x_hi, y_lo, y_hi = window.as_bounds()
        self.assertAlmostEqual(x_lo, -x_hi, places=12)
        self.assertAlmostEqual(y_lo, -y_hi, places=12)
        self.assertAlmostEqual(x_lo, y_lo, places=12)
        inside = field[(X >= x_lo) & (X <= x_hi) & (Y >= y_lo) & (Y <= y_hi)]
        contained = float(inside.sum() / field.sum())
        self.assertGreater(contained, 0.99)
        self.assertLessEqual(x_hi, L)
        self.assertGreaterEqual(x_lo, -L)

    def test_bilinear_resample_preserves_peak_location(self) -> None:
        G = 32
        L = 3.0
        axis = _density_grid_axis(G, L)
        X, Y = np.meshgrid(axis, axis, indexing="ij")
        field = np.exp(-((X - 1.0) ** 2 + (Y + 0.5) ** 2) / 0.2)
        sampled, x_out, y_out = _bilinear_resample_field(field, axis, -1.5, 1.5, -1.5, 1.5, 64)
        self.assertEqual(sampled.shape, (64, 64))
        peak = np.unravel_index(int(np.argmax(sampled)), sampled.shape)
        self.assertLess(abs(float(x_out[peak[0]]) - 1.0), 0.15)
        self.assertLess(abs(float(y_out[peak[1]]) + 0.5), 0.15)

    def test_density_fft_fields_have_no_nyquist_checkerboard(self) -> None:
        config = SimulationConfig(
            n_fibers=8,
            alpha=0.5,
            K=2.0,
            eps_entropy=0.15,
            grid_size=64,
            domain_radius=5.0,
            dt=0.008,
            max_steps=120,
            make_animation=False,
            seed=2026,
        )
        result = run_density_simulation(config, make_density_initial_condition(config))
        solver = FFTPeszekPoyatoDensity2D(config.alpha, config.K, config.grid_size, config.domain_radius)
        rho = result.rho_grid
        Ax, Ay = solver.A_grid_from_rho(rho)
        Hxx, _, Hyy = solver.hessian_grid_from_rho(rho)

        rel = max(float(np.max(np.abs(rho))), 1.0)
        self.assertLess(_nyquist_checkerboard_amplitude(rho) / rel, 1e-6)
        self.assertLess(_nyquist_checkerboard_amplitude(Ax) / max(float(np.max(np.abs(Ax))), 1.0), 1e-4)
        self.assertLess(_nyquist_checkerboard_amplitude(Ay) / max(float(np.max(np.abs(Ay))), 1.0), 1e-6)
        trace = Hxx + Hyy
        self.assertLess(_nyquist_checkerboard_amplitude(trace) / max(float(np.max(np.abs(trace))), 1.0), 1e-6)

    def test_display_payload_applies_display_antialias_when_smoothing_enabled(self) -> None:
        G = 16
        L = 2.0
        axis = _density_grid_axis(G, L)
        X, Y = np.meshgrid(axis, axis, indexing="ij")
        field = np.exp(-((X**2 + Y**2) / 0.25))
        cfg = SimulationConfig(grid_size=G, domain_radius=L, density_heatmap_smoothing=True)
        raw = _density_display_payload_from_grid(field, axis, cfg, dynamic_zoom=False, window=None, heatmap_smoothing=False)
        smooth = _density_display_payload_from_grid(field, axis, cfg, dynamic_zoom=False, window=None, heatmap_smoothing=True)
        self.assertFalse(np.allclose(raw["z"], smooth["z"]))
        peak_raw = np.unravel_index(int(np.argmax(raw["z"])), raw["z"].shape)
        peak_smooth = np.unravel_index(int(np.argmax(smooth["z"])), smooth["z"].shape)
        self.assertLess(abs(peak_raw[0] - peak_smooth[0]), 2)
        self.assertLess(abs(peak_raw[1] - peak_smooth[1]), 2)

    def test_display_payload_zoom_off_uses_full_grid(self) -> None:
        G = 16
        L = 2.0
        axis = _density_grid_axis(G, L)
        field = np.ones((G, G), dtype=np.float64)
        cfg = SimulationConfig(grid_size=G, domain_radius=L, density_dynamic_zoom=False)
        payload = _density_display_payload_from_grid(
            field, axis, cfg, dynamic_zoom=False, window=None, heatmap_smoothing=False
        )
        self.assertFalse(payload["zoomed"])
        np.testing.assert_array_equal(payload["z"], field)
        self.assertEqual(payload["x_range"], [-L, L])

    def test_display_payload_zoom_on_resamples_to_display_grid(self) -> None:
        G = 32
        L = 5.0
        axis = _density_grid_axis(G, L)
        X, Y = np.meshgrid(axis, axis, indexing="ij")
        field = np.exp(-(X**2 + Y**2))
        window = _density_support_window(
            field,
            axis,
            L=L,
            mass_fraction=0.995,
            margin=1.35,
            min_half_width=None,
        )
        cfg = SimulationConfig(
            grid_size=G,
            domain_radius=L,
            density_dynamic_zoom=True,
            density_display_grid_size=48,
        )
        payload = _density_display_payload_from_grid(field, axis, cfg, dynamic_zoom=True, window=window)
        self.assertTrue(payload["zoomed"])
        self.assertEqual(payload["z"].shape, (_effective_density_display_grid_size(cfg),) * 2)
        self.assertLess(payload["x_range"][1] - payload["x_range"][0], 2 * L)
        self.assertAlmostEqual(payload["x_range"][0], -payload["x_range"][1], places=12)
        self.assertEqual(payload["y_range"], payload["x_range"])

    def test_edge_mass_fraction_is_zero_for_interior_blob(self) -> None:
        G = 32
        L = 5.0
        axis = _density_grid_axis(G, L)
        X, Y = np.meshgrid(axis, axis, indexing="ij")
        rho = np.exp(-((X**2 + Y**2) / 0.5))
        edge = _density_edge_mass_fraction(rho, axis, L, band_fraction=0.15)
        self.assertLess(edge, 1e-6)

    def test_effective_display_grid_size_clamped_to_backend(self) -> None:
        cfg = SimulationConfig(grid_size=64, density_display_grid_size=96)
        self.assertEqual(_effective_density_display_grid_size(cfg), 64)
        run_density_simulation(_small_density_config(grid_size=64, max_steps=2))

    @unittest.skipIf(widgets is None, "ipywidgets optional dependency is unavailable")
    def test_continuous_density_widget_display_slider_max_is_backend_grid(self) -> None:
        config = _small_density_config(grid_size=48, density_display_grid_size=40)
        try:
            widget = make_continuous_density_widget(config, width=500, height=320)
        except ImportError as exc:
            self.skipTest(str(exc))
        widget._sync_display_grid_slider_bounds()
        self.assertEqual(int(widget.display_grid_slider.max), 48)
        self.assertLessEqual(int(widget.display_grid_slider.value), 48)

    @unittest.skipIf(widgets is None, "ipywidgets optional dependency is unavailable")
    def test_continuous_density_widget_dynamic_zoom_defaults_and_payload(self) -> None:
        config = _small_density_config(
            max_steps=4,
            grid_size=32,
            domain_radius=4.0,
            make_animation=True,
            trajectory_frame_count=0,
            density_dynamic_zoom=True,
            density_display_grid_size=32,
        )
        try:
            widget = make_continuous_density_widget(config, width=500, height=320)
        except ImportError as exc:
            self.skipTest(str(exc))

        self.assertTrue(widget.dynamic_zoom_toggle.value)
        self.assertTrue(widget.heatmap_smooth_toggle.value)
        result = run_density_simulation(config)
        payloads = widget._build_frame_payloads(result, config)
        self.assertTrue(payloads)
        payload = payloads[-1]
        self.assertEqual(len(payload["z"]), config.density_display_grid_size)
        self.assertEqual(len(payload["z"][0]), config.density_display_grid_size)
        span = payload["x_range"][1] - payload["x_range"][0]
        self.assertLessEqual(span, 2 * config.domain_radius + 1e-9)
        self.assertAlmostEqual(payload["x_range"][0], -payload["x_range"][1], places=12)
        self.assertEqual(payload["y_range"], payload["x_range"])
        self.assertIn("edge_mass", payload["stats"])


if __name__ == "__main__":
    unittest.main()
