import unittest

import numpy as np
import torch

from lmsspp.LMS import integrate_lms_reduced_euler, mobius_sphere, normalize, random_points_on_sphere, skew_symmetric_from_axis
from lmsspp.core.canonical_gauge import canonical_cloud, canonical_residual
from lmsspp.core.gauge_transformations import state_from_observed_cloud
from lmsspp.lms_ball4d_widget import LMSBall4DWidget
from lmsspp.lms_ball3d_widget import LMSBall3DWidget


class LMSBall3DExactGaugeTests(unittest.TestCase):
    def test_core_exact_reduced_state_satisfies_gauge_equation_3d(self) -> None:
        n = 18
        d = 3
        weights = torch.full((n,), 1.0 / float(n), dtype=torch.float64)
        observed = normalize(
            mobius_sphere(
                random_points_on_sphere(n, d=d, dtype=torch.float64),
                torch.tensor([0.22, -0.11, 0.07], dtype=torch.float64),
            )
        )
        state = state_from_observed_cloud(
            observed,
            weights,
            mode="busemann_exact",
            fallback_dir=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )
        w0 = state.w
        base_points = state.reference_points
        x_reconstructed = state.observed_points

        inverse = (weights[:, None] * mobius_sphere(observed, -w0)).sum(dim=0)
        reconstructed = normalize(mobius_sphere(base_points, w0))
        base_bary = (weights[:, None] * base_points).sum(dim=0)

        self.assertLess(float(torch.linalg.norm(inverse)), 1e-7)
        self.assertLess(float(torch.linalg.norm(base_bary)), 1e-7)
        self.assertLess(
            float(torch.amax(torch.linalg.norm(reconstructed - observed, dim=1))),
            1e-6,
        )
        self.assertTrue(bool(state.diagnostics.converged))

    def test_widget_exact_wrapper_records_diagnostics(self) -> None:
        n = 12
        d = 3
        weights = torch.full((n,), 1.0 / float(n), dtype=torch.float64)
        observed = normalize(
            mobius_sphere(
                random_points_on_sphere(n, d=d, dtype=torch.float64),
                torch.tensor([0.15, -0.08, 0.04], dtype=torch.float64),
            )
        )
        probe = object.__new__(LMSBall3DWidget)
        probe._last_gauge_residual_norm = float("nan")
        probe._last_gauge_center_error = float("nan")
        probe._last_gauge_converged = False

        w0, base_points, x_reconstructed = LMSBall3DWidget._exact_reduced_state_from_observed_cloud(
            probe,
            points=observed,
            weights=weights,
            fallback_dir=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )

        self.assertLess(float(torch.linalg.norm((weights[:, None] * mobius_sphere(observed, -w0)).sum(dim=0))), 1e-7)
        self.assertLess(float(torch.linalg.norm((weights[:, None] * base_points).sum(dim=0))), 1e-7)
        self.assertLess(float(torch.amax(torch.linalg.norm(x_reconstructed - observed, dim=1))), 1e-6)
        self.assertTrue(bool(probe._last_gauge_converged))

    def test_center_estimation_dropdown_mode_defaults_and_selects_poisson(self) -> None:
        class Dropdown:
            value = "busemann_exact"

        probe = object.__new__(LMSBall3DWidget)
        probe._job_center_estimation_mode = None
        probe.center_estimation_mode = "busemann_exact"
        probe.center_estimation_dropdown = Dropdown()

        self.assertEqual(probe._current_center_estimation_mode(), "busemann_exact")
        probe.center_estimation_dropdown.value = "poisson_shrink"
        self.assertEqual(probe._current_center_estimation_mode(), "poisson_shrink")

    def test_lms_ball4d_inherits_gauge_estimator(self) -> None:
        self.assertTrue(callable(getattr(LMSBall4DWidget, "_estimate_w_from_boundary_points", None)))

    def test_short_trajectory_preserves_z_magnitude_invariant(self) -> None:
        n = 12
        d = 3
        weights = torch.ones(n, dtype=torch.float64) / float(n)
        observed = normalize(random_points_on_sphere(n, d=d, dtype=torch.float64))
        state = canonical_cloud(observed, weights, max_iters=160, tol=1e-10)
        w0 = state.w
        base_points = state.P

        axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        A = skew_symmetric_from_axis(axis, rate=2.5).to(dtype=torch.float64)
        traj = integrate_lms_reduced_euler(
            w0=w0,
            zeta0=torch.eye(d, dtype=torch.float64),
            base_points=base_points,
            weights=weights,
            A=A,
            coupling=1.0,
            dt=0.02,
            steps=24,
            w_mode="explicit",
            project_rotation=True,
            store_points="none",
        )

        w_np = traj.w.detach().cpu().numpy()
        z_lab_np = traj.z.detach().cpu().numpy()
        z_body_np = (-traj.w).detach().cpu().numpy()

        w_norm = np.linalg.norm(w_np, axis=1)
        z_lab_norm = np.linalg.norm(z_lab_np, axis=1)
        z_body_norm = np.linalg.norm(z_body_np, axis=1)

        np.testing.assert_allclose(z_body_norm, w_norm, rtol=0.0, atol=2e-5)
        np.testing.assert_allclose(z_lab_norm, w_norm, rtol=0.0, atol=2e-5)
        np.testing.assert_allclose(z_body_np, -w_np, rtol=0.0, atol=2e-5)

        # Lab-frame z rotates with zeta while magnitude stays tied to |w|.
        if np.linalg.norm(z_lab_np[0]) > 1e-8 and np.linalg.norm(z_lab_np[-1]) > 1e-8:
            cos_align = float(
                np.dot(z_lab_np[0], z_lab_np[-1])
                / (np.linalg.norm(z_lab_np[0]) * np.linalg.norm(z_lab_np[-1]))
            )
            self.assertLess(abs(cos_align), 0.999)

    def test_canonical_residual_matches_negative_w(self) -> None:
        n = 10
        weights = torch.full((n,), 1.0 / float(n), dtype=torch.float64)
        points = normalize(torch.randn((n, 3), dtype=torch.float64))
        state = canonical_cloud(points, weights, max_iters=160, tol=1e-10)
        res_w = canonical_residual(-state.w, points, weights)
        res_z = canonical_residual(state.z, points, weights)
        self.assertLess(float(torch.linalg.norm(res_w)), 1e-8)
        self.assertLess(float(torch.linalg.norm(res_z)), 1e-8)

    def test_spherical_inversion_projection_forgets_constant_normal_component(self) -> None:
        pole = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        theta = np.linspace(0.22, np.pi - 0.22, 24)
        phi = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        points = np.array(
            [
                [np.cos(t), np.sin(t) * np.cos(p), np.sin(t) * np.sin(p)]
                for t, p in zip(theta, phi, strict=True)
            ],
            dtype=np.float64,
        )

        a, e1, e2 = LMSBall3DWidget._projection_basis_from_pole(pole)
        unscaled = LMSBall3DWidget._spherical_inversion_project_2d(points, pole=pole, omega=1.0)
        scaled = LMSBall3DWidget._spherical_inversion_project_2d(points, pole=pole, omega=0.25)
        diff = points - a[None, :]
        y3 = diff / np.sum(diff * diff, axis=1, keepdims=True)

        np.testing.assert_allclose(y3 @ a, -0.5 * np.ones(points.shape[0]), atol=1e-12)
        np.testing.assert_allclose(unscaled[:, 0], y3 @ e1, atol=1e-12)
        np.testing.assert_allclose(unscaled[:, 1], y3 @ e2, atol=1e-12)
        np.testing.assert_allclose(scaled, 0.25 * unscaled, atol=1e-12)

    def test_spherical_inversion_grid_lines_are_finite_2d_segments(self) -> None:
        grid_x, grid_y = LMSBall3DWidget._spherical_inversion_grid_2d(
            pole=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            omega=0.6,
            theta_count=3,
            phi_count=4,
            samples=20,
        )
        finite_x = np.asarray([x for x in grid_x if x is not None], dtype=np.float64)
        finite_y = np.asarray([y for y in grid_y if y is not None], dtype=np.float64)
        self.assertGreater(finite_x.size, 0)
        self.assertEqual(finite_x.size, finite_y.size)
        self.assertTrue(np.isfinite(finite_x).all())
        self.assertTrue(np.isfinite(finite_y).all())

    def test_projection_omega_is_state_dependent_even_for_unscaled_chart(self) -> None:
        probe = object.__new__(LMSBall3DWidget)
        probe._projection_chart_mode = "unscaled"
        state = {"w_raw": np.array([0.3, 0.4, 0.0], dtype=np.float64)}

        self.assertAlmostEqual(probe._projection_chart_omega(state), 0.75)
        self.assertAlmostEqual(probe._projection_chart_scale(state), 1.0)

        probe._projection_chart_mode = "rescaled"
        self.assertAlmostEqual(probe._projection_chart_omega(state), 0.75)
        self.assertAlmostEqual(probe._projection_chart_scale(state), 0.75)


if __name__ == "__main__":
    unittest.main()
