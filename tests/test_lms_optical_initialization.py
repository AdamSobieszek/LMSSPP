import unittest

import numpy as np

from lmsspp.lms_optical_widget import (
    _canonical_initialization_np,
    _mobius_sphere_series_np,
    _reflected_series_np,
)


class LMSOpticalCanonicalInitializationTests(unittest.TestCase):
    def test_exact_inverse_pipeline_for_all_presets_in_2d_and_3d(self) -> None:
        for dim in (2, 3):
            for preset in ("random", "balanced", "clustered", "dipole"):
                with self.subTest(dim=dim, preset=preset):
                    state = _canonical_initialization_np(
                        12,
                        np.random.default_rng(11),
                        preset=preset,
                        target_radius=0.35,
                        dimension=dim,
                    )
                    observed = state.observed_points.reshape(1, -1, dim)
                    reference = state.reference_points.reshape(1, -1, dim)
                    inverse = _mobius_sphere_series_np((-state.w_star).reshape(1, dim), observed)[0]
                    reconstructed = _mobius_sphere_series_np(state.w_star.reshape(1, dim), reference)[0]

                    self.assertLess(np.linalg.norm(state.weights @ inverse), 1e-7)
                    self.assertLess(np.linalg.norm(state.weights @ state.reference_points), 1e-7)
                    self.assertLess(
                        np.max(np.linalg.norm(reconstructed - state.observed_points, axis=1)),
                        1e-7,
                    )
                    self.assertLess(abs(np.linalg.norm(state.w_star) - 0.35), 1e-6)

    def test_observed_cloud_changes_with_radius_for_fixed_seed(self) -> None:
        for dim in (2, 3):
            for preset in ("random", "balanced", "clustered", "dipole"):
                with self.subTest(dim=dim, preset=preset):
                    low = _canonical_initialization_np(
                        14,
                        np.random.default_rng(5),
                        preset=preset,
                        target_radius=0.15,
                        dimension=dim,
                    )
                    high = _canonical_initialization_np(
                        14,
                        np.random.default_rng(5),
                        preset=preset,
                        target_radius=0.75,
                        dimension=dim,
                    )
                    pointwise_delta = np.max(
                        np.linalg.norm(low.observed_points - high.observed_points, axis=1)
                    )
                    self.assertGreater(pointwise_delta, 1e-4)

    def test_reference_shape_axis_is_opposite_recovered_w_star(self) -> None:
        for preset in ("clustered", "dipole"):
            with self.subTest(preset=preset):
                state = _canonical_initialization_np(
                    30,
                    np.random.default_rng(1),
                    preset=preset,
                    target_radius=0.4,
                    dimension=2,
                )
                w_dir = state.w_star / np.linalg.norm(state.w_star)
                reference_projection = state.candidate_points @ w_dir

                self.assertLess(float(np.median(reference_projection)), 0.0)
                np.testing.assert_allclose(w_dir, np.array([1.0, 0.0]), atol=1e-7)

    def test_initial_velocity_uses_recovered_w_star_and_reference_cloud(self) -> None:
        state = _canonical_initialization_np(
            12,
            np.random.default_rng(19),
            preset="random",
            target_radius=0.42,
            dimension=2,
        )
        w0 = state.w_star.reshape(1, 2)
        reflected = _reflected_series_np(w0, state.reference_points)[0]
        optical_velocity = 0.5 * (1.0 - float(np.dot(state.w_star, state.w_star))) * (state.weights @ reflected)
        current_cloud = _mobius_sphere_series_np(w0, state.reference_points.reshape(1, -1, 2))[0]
        expected_velocity = -0.5 * (1.0 - float(np.dot(state.w_star, state.w_star))) * (
            state.weights @ current_cloud
        )
        inverse_balance = state.weights @ _mobius_sphere_series_np(-w0, current_cloud.reshape(1, -1, 2))[0]

        np.testing.assert_allclose(w0[0], state.w_star, atol=0.0)
        np.testing.assert_allclose(optical_velocity, expected_velocity, atol=1e-12)
        self.assertLess(np.linalg.norm(inverse_balance), 1e-7)


if __name__ == "__main__":
    unittest.main()
