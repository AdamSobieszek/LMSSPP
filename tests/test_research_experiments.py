import tempfile
import unittest
import importlib.util
from pathlib import Path

from lmsspp.research_experiments import apply_overrides, run_experiment_file


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENT_PATH = REPO_ROOT / "scripts" / "run_experiment.py"
_RUN_EXPERIMENT_SPEC = importlib.util.spec_from_file_location("run_experiment_script", RUN_EXPERIMENT_PATH)
assert _RUN_EXPERIMENT_SPEC is not None
assert _RUN_EXPERIMENT_SPEC.loader is not None
run_experiment_script = importlib.util.module_from_spec(_RUN_EXPERIMENT_SPEC)
_RUN_EXPERIMENT_SPEC.loader.exec_module(run_experiment_script)


class ResearchExperimentOrchestrationTests(unittest.TestCase):
    def test_apply_overrides_supports_dotted_mappings_and_lists(self) -> None:
        config = {
            "experiment": "finite_horizon_animation_batch",
            "params": {"n_per_fiber": 40},
            "cases": [{"seed": 1, "tau": 0.055}],
        }
        resolved = apply_overrides(config, ["params.n_per_fiber=12", "cases.0.seed=2031", "cases.0.tau=0.045"])
        self.assertEqual(resolved["params"]["n_per_fiber"], 12)
        self.assertEqual(resolved["cases"][0]["seed"], 2031)
        self.assertEqual(resolved["cases"][0]["tau"], 0.045)

    def test_repo_level_script_resolves_experiment_names(self) -> None:
        resolved = run_experiment_script.resolve_config_path("finite_horizon_comparison", cwd=Path("/tmp"))
        self.assertEqual(resolved, REPO_ROOT / "experiments" / "finite_horizon_comparison.yaml")

    def test_yaml_finite_horizon_comparison_runs_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "comparison"
            config_path = Path(tmp) / "comparison.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "experiment: finite_horizon_comparison",
                        f"output_dir: {out_dir}",
                        "params:",
                        "  n_fibers: 2",
                        "  n_per_fiber: 3",
                        "  alpha: 0.45",
                        "  K: 0.2",
                        "  grid_size: 16",
                        "  domain_radius: 3.0",
                        "  tau: 0.01",
                        "  old_fixed_steps: 1",
                        "  adaptive_steps_per_horizon: 1",
                        "  seed: 22",
                    ]
                )
            )
            result = run_experiment_file(config_path, overrides=["params.seed=23"])
            self.assertIn("new_model_residual_rms", result)
            self.assertTrue((out_dir / "resolved_experiment.yaml").exists())
            self.assertTrue((out_dir / "orchestration_result.json").exists())
            self.assertTrue((out_dir / "finite_horizon_vs_fixed_rk2.png").exists())


if __name__ == "__main__":
    unittest.main()
