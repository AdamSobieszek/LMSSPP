import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from lmsspp.dynamics.pp_cs_equilibria import InitialCondition
from lmsspp.dynamics.pp_transient_research import (
    RESEARCH_DIAGNOSTIC_FIELDS,
    DirectPeszekPoyato2D,
    TransientResearchConfig,
    compute_morphology_metrics,
    evaluate_A_and_H_at_particles,
    nearest_neighbor_quantiles,
    run_pp_research_sweep,
    run_research_simulation,
)


def _direct_A(query: np.ndarray, sources: np.ndarray, alpha: float, K: float) -> np.ndarray:
    diff = query[:, None, :] - sources[None, :, :]
    r = np.linalg.norm(diff, axis=-1)
    mask = r > 1e-14
    scale = np.zeros_like(r)
    scale[mask] = K * (r[mask] ** (-alpha)) / (1 - alpha) / len(sources)
    return (diff * scale[..., None]).sum(axis=1)


class PPTransientResearchTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
