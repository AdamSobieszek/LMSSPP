"""YAML-driven orchestration for Peszek--Poyato research experiments.

This module is intentionally one package level above ``lmsspp.dynamics``.  The
dynamics/transient-research module keeps the ad hoc numerical machinery; this
module provides a reproducible experiment runner around those functions.

Recommended repository-level entry point:

    python scripts/run_experiment.py pp_finite_horizon_animation_batch

Optional dotted overrides use Hydra-style syntax:

    python scripts/run_experiment.py pp_finite_horizon_animation_batch params.n_per_fiber=80 cases.0.seed=2031

This module remains the importable orchestration library used by the script and
tests. Running it directly with ``python -m`` requires the package to be
installed or ``PYTHONPATH=src`` to be set.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from .dynamics.pp_cs_equilibria import InitializerConfig, _jsonable, make_initial_condition
from .dynamics.pp_transient_research import (
    TransientResearchConfig,
    run_finite_horizon_animation_batch,
    run_finite_horizon_comparison,
    run_long_cross_reference,
    run_pp_research_sweep,
    run_research_simulation,
    run_toy_transient_suite,
    save_research_run_outputs,
)

try:
    import yaml
except Exception as exc:  # pragma: no cover - import error path is environment-dependent
    yaml = None  # type: ignore[assignment]
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None


SUPPORTED_EXPERIMENTS = (
    "finite_horizon_animation_batch",
    "finite_horizon_comparison",
    "long_cross_reference",
    "research_sweep",
    "research_run",
    "toy_suite",
)


def _require_yaml() -> Any:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required for YAML experiment files. Install the research extras or add pyyaml to the environment."
        ) from _YAML_IMPORT_ERROR
    return yaml


def load_yaml_config(path: Path | str) -> dict[str, Any]:
    """Load an experiment YAML file into a plain dictionary."""

    yaml_module = _require_yaml()
    config_path = Path(path)
    data = yaml_module.safe_load(config_path.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"experiment YAML must contain a mapping at top level: {config_path}")
    return data


def _parse_override_value(raw: str) -> Any:
    yaml_module = _require_yaml()
    return yaml_module.safe_load(raw)


def _set_path_value(data: MutableMapping[str, Any] | list[Any], dotted_path: str, value: Any) -> None:
    parts = [part for part in dotted_path.split(".") if part]
    if not parts:
        raise ValueError("override path cannot be empty")
    current: Any = data
    for part in parts[:-1]:
        if isinstance(current, list):
            idx = int(part)
            current = current[idx]
            continue
        if not isinstance(current, MutableMapping):
            raise ValueError(f"cannot set override {dotted_path!r}: {part!r} does not address a mapping/list")
        if part not in current or current[part] is None:
            current[part] = {}
        current = current[part]
    leaf = parts[-1]
    if isinstance(current, list):
        current[int(leaf)] = value
    elif isinstance(current, MutableMapping):
        current[leaf] = value
    else:
        raise ValueError(f"cannot set override {dotted_path!r}: parent is not a mapping/list")


def apply_overrides(config: dict[str, Any], overrides: Sequence[str] | None = None) -> dict[str, Any]:
    """Apply Hydra-style ``a.b=value`` overrides to an experiment config."""

    import copy

    resolved = copy.deepcopy(config)
    for override in overrides or ():
        if "=" not in override:
            raise ValueError(f"override must have the form key=value, got {override!r}")
        key, raw_value = override.split("=", 1)
        _set_path_value(resolved, key.strip(), _parse_override_value(raw_value.strip()))
    return resolved


def _dataclass_kwargs(cls: type[Any], data: Mapping[str, Any]) -> dict[str, Any]:
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    names = {field.name for field in fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {', '.join(unknown)}")
    kwargs = dict(data)
    if "out_dir" in kwargs and kwargs["out_dir"] is not None:
        kwargs["out_dir"] = Path(kwargs["out_dir"])
    if "shape_names" in kwargs and kwargs["shape_names"] is not None:
        kwargs["shape_names"] = tuple(kwargs["shape_names"])
    if "initializer_config" in kwargs and isinstance(kwargs["initializer_config"], Mapping):
        kwargs["initializer_config"] = InitializerConfig(**dict(kwargs["initializer_config"]))
    return kwargs


def transient_config_from_mapping(data: Mapping[str, Any]) -> TransientResearchConfig:
    """Build ``TransientResearchConfig`` from YAML data with field validation."""

    return TransientResearchConfig(**_dataclass_kwargs(TransientResearchConfig, data))


def _experiment_name(config: Mapping[str, Any]) -> str:
    raw = config.get("experiment", config.get("name", None))
    if isinstance(raw, Mapping):
        raw = raw.get("kind", raw.get("name"))
    if not raw:
        raise ValueError(f"experiment config must define one of: {', '.join(SUPPORTED_EXPERIMENTS)}")
    kind = str(raw)
    if kind not in SUPPORTED_EXPERIMENTS:
        raise ValueError(f"unsupported experiment {kind!r}; expected one of {', '.join(SUPPORTED_EXPERIMENTS)}")
    return kind


def _output_dir(config: Mapping[str, Any], config_path: Path | None = None) -> Path:
    raw = config.get("output_dir")
    if raw is None and isinstance(config.get("experiment"), Mapping):
        raw = config["experiment"].get("output_dir")  # type: ignore[index]
    if raw is None:
        stem = "experiment" if config_path is None else config_path.stem
        raw = Path("temp_research") / stem
    return Path(raw)


def _write_resolved_config(out_dir: Path, config: Mapping[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_module = _require_yaml()
    (out_dir / "resolved_experiment.yaml").write_text(yaml_module.safe_dump(_jsonable(config), sort_keys=False))


def run_experiment_config(
    config: Mapping[str, Any],
    *,
    config_path: Path | None = None,
) -> Any:
    """Run a resolved experiment mapping and write outputs under its output dir."""

    kind = _experiment_name(config)
    out_dir = _output_dir(config, config_path=config_path)
    _write_resolved_config(out_dir, config)
    params = dict(config.get("params", {}))

    if kind == "finite_horizon_animation_batch":
        cases = config.get("cases")
        summaries = run_finite_horizon_animation_batch(
            out_dir,
            cases=cases if cases is not None else None,
            **params,
        )
        return summaries

    if kind == "finite_horizon_comparison":
        metrics = run_finite_horizon_comparison(out_dir, **params)
        return metrics

    if kind == "long_cross_reference":
        metrics = run_long_cross_reference(out_dir, **params)
        return metrics

    if kind == "toy_suite":
        metrics = run_toy_transient_suite(out_dir, **params)
        return metrics

    if kind == "research_sweep":
        base_data = dict(config.get("base_config", {}))
        base_data.setdefault("out_dir", out_dir)
        base_config = transient_config_from_mapping(base_data)
        sweep = str(config.get("sweep", params.pop("sweep", "none")))
        values = config.get("values", params.pop("values", None))
        summaries = run_pp_research_sweep(base_config, sweep, values=values, out_dir=out_dir)
        return summaries

    if kind == "research_run":
        run_data = dict(config.get("config", config.get("base_config", {})))
        run_data.setdefault("out_dir", out_dir)
        run_config = transient_config_from_mapping(run_data)
        initial = make_initial_condition(run_config)
        result = run_research_simulation(run_config, initial)
        summary = save_research_run_outputs(result, run_config, out_dir, initial=initial)
        return summary

    raise AssertionError(f"unhandled experiment kind: {kind}")


def run_experiment_file(path: Path | str, overrides: Sequence[str] | None = None) -> Any:
    """Load, override, and run a YAML experiment file."""

    config_path = Path(path)
    raw_config = load_yaml_config(config_path)
    resolved = apply_overrides(raw_config, overrides)
    result = run_experiment_config(resolved, config_path=config_path)
    out_dir = _output_dir(resolved, config_path=config_path)
    (out_dir / "orchestration_result.json").write_text(json.dumps(_jsonable(result), indent=2))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="YAML experiment file to execute")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra-style dotted overrides, for example params.n_per_fiber=80 cases.0.seed=2031",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_experiment_file(args.config, args.overrides)
    print("Done.")
    print(json.dumps(_jsonable(result), indent=2))


if __name__ == "__main__":
    main()
