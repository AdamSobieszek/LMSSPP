import unittest

import torch

from lmsspp.core.canonical_gauge import CanonicalGaugeState, canonical_residual
from lmsspp.core.gauge import (
    GaugeState,
    physical_cloud_from_reference,
    prepare_sphere_cloud,
    reference_cloud_from_physical,
    target_w_from_radius,
)
from lmsspp.core.gauge_transformations import (
    state_from_observed_cloud,
    state_from_reference_cloud,
)
from lmsspp.core.initialize import poisson_shrink_w_from_observed
from lmsspp.core.lms import normalize, random_points_on_sphere


def _axis_cloud(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    eye = torch.eye(3, dtype=dtype)
    return torch.cat([eye, -eye], dim=0)


class LMSGaugeCoreTests(unittest.TestCase):
    def test_base_gauge_state_permits_mismatched_observed_points(self) -> None:
        reference = _axis_cloud()
        weights = torch.full((6,), 1.0 / 6.0, dtype=torch.float64)
        w0 = torch.tensor([0.10, 0.0, 0.0], dtype=torch.float64)
        observed = physical_cloud_from_reference(reference, torch.tensor([-0.20, 0.0, 0.0], dtype=torch.float64))
        state = GaugeState(
            w=w0,
            reference_points=reference,
            weights=weights,
            observed_points=observed,
            mode="broken",
        )

        self.assertGreater(float(torch.max(torch.linalg.norm(state.reconstructed_points() - state.observed_points, dim=1))), 1e-3)
        state.w = torch.tensor([0.25, 0.0, 0.0], dtype=torch.float64)
        self.assertLess(float(torch.max(torch.linalg.norm(state.observed_points - observed, dim=1))), 1e-12)
        w_lms, ref_lms, weights_lms = state.as_lms_inputs()
        torch.testing.assert_close(w_lms, state.w)
        torch.testing.assert_close(ref_lms, state.reference_points)
        torch.testing.assert_close(weights_lms, state.weights)

    def test_canonical_gauge_state_updates_observed_points_when_w_changes(self) -> None:
        weights = torch.full((6,), 1.0 / 6.0, dtype=torch.float64)
        initialized = physical_cloud_from_reference(
            _axis_cloud(),
            torch.tensor([0.21, -0.13, 0.08], dtype=torch.float64),
        )
        target_w = torch.tensor([-0.32, 0.0, 0.0], dtype=torch.float64)
        state = CanonicalGaugeState.from_initialized_cloud(
            initialized,
            weights,
            target_w=target_w,
            fallback_dir=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )

        torch.testing.assert_close(state.w, target_w, rtol=0.0, atol=1e-12)
        self.assertLess(float(torch.linalg.norm((state.weights[:, None] * state.reference_points).sum(dim=0))), 1e-8)
        self.assertLess(
            float(torch.max(torch.linalg.norm(state.observed_points - physical_cloud_from_reference(state.reference_points, target_w), dim=1))),
            1e-12,
        )
        state.w = state.canonical_initial_w
        self.assertLess(float(torch.max(torch.linalg.norm(state.observed_points - initialized, dim=1))), 1e-7)
        with self.assertRaises(AttributeError):
            state.observed_points = initialized

    def test_prepare_sphere_cloud_normalizes_weights(self) -> None:
        points = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=torch.float64)
        weights = torch.tensor([2.0, 6.0], dtype=torch.float64)
        prepared, normalized_weights = prepare_sphere_cloud(points, weights)

        self.assertLess(float(torch.max(torch.abs(torch.linalg.norm(prepared, dim=1) - 1.0))), 1e-8)
        self.assertAlmostEqual(float(normalized_weights.sum()), 1.0, places=14)
        torch.testing.assert_close(normalized_weights, torch.tensor([0.25, 0.75], dtype=torch.float64))

    def test_physical_and_reference_cloud_round_trip(self) -> None:
        reference = _axis_cloud()
        w = torch.tensor([0.18, -0.07, 0.11], dtype=torch.float64)
        observed = physical_cloud_from_reference(reference, w)
        recovered = reference_cloud_from_physical(observed, w)

        self.assertLess(float(torch.max(torch.linalg.norm(recovered - reference, dim=1))), 1e-8)

    def test_target_w_from_radius_conventions(self) -> None:
        direction = torch.tensor([0.0, 2.0, 0.0], dtype=torch.float64)
        torch.testing.assert_close(
            target_w_from_radius(0.4, direction, convention="w"),
            torch.tensor([0.0, 0.4, 0.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            target_w_from_radius(0.4, direction, convention="physical_dipole"),
            torch.tensor([0.0, -0.4, 0.0], dtype=torch.float64),
        )

    def test_exact_busemann_observed_state_is_centered(self) -> None:
        weights = torch.full((6,), 1.0 / 6.0, dtype=torch.float64)
        observed = physical_cloud_from_reference(
            _axis_cloud(),
            torch.tensor([0.21, -0.13, 0.08], dtype=torch.float64),
        )
        state = state_from_observed_cloud(
            observed,
            weights,
            mode="busemann_exact",
            fallback_dir=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )

        residual = canonical_residual(-state.w, observed, weights)
        self.assertLess(float(torch.linalg.norm(residual)), 1e-8)
        self.assertLess(float(torch.linalg.norm((weights[:, None] * state.reference_points).sum(dim=0))), 1e-8)

    def test_exact_target_radius_projection_uses_requested_w(self) -> None:
        weights = torch.full((6,), 1.0 / 6.0, dtype=torch.float64)
        observed = physical_cloud_from_reference(
            _axis_cloud(),
            torch.tensor([0.18, 0.04, -0.09], dtype=torch.float64),
        )
        target_w = torch.tensor([-0.32, 0.0, 0.0], dtype=torch.float64)
        state = state_from_observed_cloud(
            observed,
            weights,
            mode="busemann_exact",
            target_w=target_w,
            fallback_dir=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )

        torch.testing.assert_close(state.w, target_w, rtol=0.0, atol=1e-12)
        self.assertLess(float(torch.linalg.norm((weights[:, None] * state.reference_points).sum(dim=0))), 1e-8)
        reconstructed = physical_cloud_from_reference(state.reference_points, target_w)
        self.assertLess(float(torch.max(torch.linalg.norm(reconstructed - state.observed_points, dim=1))), 1e-12)

    def test_poisson_reference_mode_keeps_target_and_reference(self) -> None:
        gen = torch.Generator().manual_seed(2)
        reference = random_points_on_sphere(9, d=3, dtype=torch.float64, generator=gen)
        reference = normalize(reference)
        weights = torch.full((9,), 1.0 / 9.0, dtype=torch.float64)
        target_w = torch.tensor([-0.25, 0.04, 0.0], dtype=torch.float64)
        state = state_from_reference_cloud(
            reference,
            target_w,
            weights,
            mode="poisson_shrink",
            fallback_dir=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )

        torch.testing.assert_close(state.w, target_w, rtol=0.0, atol=1e-12)
        torch.testing.assert_close(state.reference_points, reference, rtol=0.0, atol=1e-12)

    def test_poisson_observed_mode_recovers_finite_reconstructing_state(self) -> None:
        weights = torch.full((6,), 1.0 / 6.0, dtype=torch.float64)
        observed = physical_cloud_from_reference(
            _axis_cloud(),
            torch.tensor([-0.28, 0.0, 0.0], dtype=torch.float64),
        )
        w = poisson_shrink_w_from_observed(
            observed,
            weights,
            fallback_dir=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )
        state = state_from_observed_cloud(
            observed,
            weights,
            mode="poisson_shrink",
            fallback_dir=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )

        self.assertTrue(bool(torch.isfinite(w).all()))
        self.assertLess(float(torch.linalg.norm(state.w)), 1.0)
        self.assertLess(float(torch.max(torch.linalg.norm(state.observed_points - observed, dim=1))), 1e-10)


if __name__ == "__main__":
    unittest.main()
