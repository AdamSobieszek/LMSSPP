"""Repository-level runner for YAML research experiments.

Examples:

    python scripts/run_experiment.py pp_finite_horizon_comparison
    python scripts/run_experiment.py experiments/pp_finite_horizon_animation_batch.yaml params.n_per_fiber=80

The script is intentionally outside ``src/lmsspp`` so it can resolve both the
repo-local ``experiments/`` directory and the ``src`` package without requiring
an editable install or a manually configured ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def bootstrap_src_path() -> None:
    """Make the local ``src`` layout importable when run from a checkout."""

    src = str(SRC_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)


def resolve_config_path(raw: str | Path, *, cwd: Path | None = None) -> Path:
    """Resolve an experiment YAML path or a name under ``experiments/``."""

    cwd = Path.cwd() if cwd is None else cwd
    candidate = Path(raw).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if not candidate.is_absolute():
        for base in (cwd, REPO_ROOT):
            path = base / candidate
            if path.exists():
                return path

    experiment_name = candidate.name
    experiment_candidates = [EXPERIMENTS_DIR / experiment_name]
    if not Path(experiment_name).suffix:
        experiment_candidates.extend(
            [
                EXPERIMENTS_DIR / f"{experiment_name}.yaml",
                EXPERIMENTS_DIR / f"{experiment_name}.yml",
            ]
        )
    for path in experiment_candidates:
        if path.exists():
            return path

    searched = [str(path) for path in experiment_candidates]
    raise FileNotFoundError(
        f"could not find experiment config {raw!r}; checked current directory, repo root, and {searched}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        help="Experiment YAML path, or a config name under experiments/ without the .yaml suffix",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra-style dotted overrides, for example params.n_per_fiber=80 cases.0.seed=2031",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    bootstrap_src_path()

    from lmsspp.dynamics.pp_cs_equilibria import _jsonable
    from lmsspp.research_experiments import run_experiment_file

    config_path = resolve_config_path(args.config)
    os.chdir(REPO_ROOT)
    result = run_experiment_file(config_path, args.overrides)
    print("Done.")
    print(json.dumps(_jsonable(result), indent=2))


if __name__ == "__main__":
    main()
