import unittest
from dataclasses import replace

import numpy as np

from lmsspp.dynamics.pp_cs_equilibria import (
    FFTPeszekPoyato2D,
    InitializerConfig,
    SimulationConfig,
    TorchPeszekPoyato2D,
    _make_pp_backend,
    fiber_colors,
    interp_grid,
    make_dynamics_widget,
    make_initial_condition,
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


if __name__ == "__main__":
    unittest.main()
