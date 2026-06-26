# Reproducible Research Experiments

This directory contains YAML experiment files for the Peszek--Poyato research
workflows. A YAML file is meant to be the reproducible record of a run: it names
the experiment type, fixes the output directory, and stores the parameters that
were used to generate the artifacts under `temp_research/`.

Use the repository-level runner:

```bash
/opt/anaconda3/envs/manip311/bin/python scripts/run_experiment.py pp_finite_horizon_comparison
```

The runner lives outside `src/lmsspp` so it can see both sides of the project:
the `experiments/` YAML files and the package code under `src/`. You can pass
either a config name under this directory or an explicit YAML path:

```bash
/opt/anaconda3/envs/manip311/bin/python scripts/run_experiment.py pp_finite_horizon_animation_batch
/opt/anaconda3/envs/manip311/bin/python scripts/run_experiment.py experiments/pp_timestep_sweep.yaml
```

## YAML Workflow

Each YAML file has a small top-level schema:

```yaml
experiment: finite_horizon_comparison
output_dir: temp_research/pp_finite_horizon_comparison_yaml

params:
  n_fibers: 10
  n_per_fiber: 50
  alpha: 0.99
  K: 1.0
```

The current dispatcher in `src/lmsspp/research_experiments.py` supports:

- `finite_horizon_animation_batch`: run several finite-horizon comparison cases
  and write per-case plots, metrics, animations, and a batch report. This
  runner can also set `right_model: ordinary_pp_adaptive` to compare original
  PP fixed RK2 against original PP adaptive RK2 with dynamic zoom.
- `finite_horizon_comparison`: compare the old fixed RK2 PP run against the
  finite-horizon adaptive run for one parameter set.
- `k_scale_calibration`: multiplicatively bracket and refine alpha-dependent
  deviations around a baseline `K(alpha)`, then pick the smallest safe
  contracting value: final cloud scale no larger than the initial scale, with no
  intermediate FFT-boundary hit.
- `long_cross_reference`: reproduce the long fixed-RK2 disk versus adaptive-RK2
  cross comparison with separate full-domain and adaptive dynamic-zoom figures.
- `research_sweep`: build a `TransientResearchConfig`, sweep one parameter, and
  write case metrics plus a sweep summary.
- `research_run`: run one `TransientResearchConfig` and save the standard
  research artifacts.
- `toy_suite`: run the reduced conical/saddle transient toy models.

The runner writes two provenance files into the selected output directory:

- `resolved_experiment.yaml`: the YAML after command-line overrides were
  applied.
- `orchestration_result.json`: the JSON-serializable result returned by the
  experiment function.

## Overrides

Command-line overrides use dotted paths and are parsed as YAML values. This is
useful for small parameter changes without editing the source YAML:

```bash
/opt/anaconda3/envs/manip311/bin/python scripts/run_experiment.py pp_finite_horizon_animation_batch \
  params.n_per_fiber=80 \
  cases.0.seed=2031 \
  cases.0.tau=0.045
```

List indices are supported with numeric path components such as `cases.0.seed`.

## Adding A New YAML

Add a new file under `experiments/` when the experiment already matches one of
the supported experiment types. Prefer to make the file self-contained:

- Set `experiment` to one of the supported names.
- Set an explicit `output_dir` under `temp_research/`.
- Put function keyword arguments under `params`.
- For `research_sweep`, put the shared simulation dataclass fields under
  `base_config`, then set `sweep` and `values`.
- For `research_run`, put the simulation dataclass fields under `config` or
  `base_config`.
- For `finite_horizon_animation_batch`, put shared knobs under `params` and
  per-case overrides under `cases`. Use the default `right_model:
  finite_horizon` for the predictive model comparison, or `right_model:
  ordinary_pp_adaptive` with `right_dynamic_zoom: true` for same-equation
  fixed-vs-adaptive PP animations.
- For `long_cross_reference`, use the default `max_steps: 2000` scale when
  reproducing the reference figure; smoke-test changes with a much smaller
  `max_steps` override first.

Run the new YAML once with a small smoke-sized parameter set before launching a
long run. Keep the generated artifacts out of source control unless they are
intentional research outputs.

## Adding A New Experiment Type

Use a new experiment type when a YAML file should invoke a new orchestration
function, not merely a new parameter set.

The extension path has four parts:

1. Implement the research function in
   `src/lmsspp/dynamics/pp_transient_research.py`.

   This module contains the PP transient research machinery and imports the
   lower-level simulation primitives from `src/lmsspp/dynamics/pp_cs_equilibria.py`.
   Keep the new function callable from Python with explicit keyword arguments
   and an `out_dir` argument. It should write durable artifacts and return a
   JSON-friendly summary dictionary or list.

2. Add or expand tests in `tests/test_pp_transient_research.py`.

   This test module is the executable specification for the research simulation
   layer. Add tiny, fast configurations that exercise the new function and prove
   it writes the expected artifacts. Keep these tests small enough for routine
   development: low `n_fibers`, low `n_per_fiber`, small `grid_size`, and one or
   two steps are usually enough.

3. Expose the new type in `src/lmsspp/research_experiments.py`.

   Add the name to `SUPPORTED_EXPERIMENTS`, import the new function, and add a
   branch in `run_experiment_config()`. The branch should read the same YAML
   shape documented here: top-level `output_dir`, optional `params`, and any
   additional top-level sections needed by that experiment.

4. Add orchestration tests in `tests/test_research_experiments.py`.

   This verifies that YAML loading, overrides, dataclass validation, dispatch,
   and provenance writing work for the new type. Use a temporary output
   directory and smoke-sized parameters.

## Testing

Run the focused tests with the local Python environment:

```bash
PYTHONPATH=src /opt/anaconda3/envs/manip311/bin/python -m unittest discover -s tests -p 'test_pp_transient_research.py'
PYTHONPATH=src /opt/anaconda3/envs/manip311/bin/python -m unittest discover -s tests -p 'test_research_experiments.py'
```

The `scripts/run_experiment.py` runner handles `src` path bootstrapping for
actual experiment runs, but tests are still run with `PYTHONPATH=src` because
the package uses a `src/` layout.

## Design Boundary

Keep the responsibilities separate:

- `experiments/*.yaml` captures reproducible parameter choices.
- `scripts/run_experiment.py` is the repository-level command-line entry point.
- `src/lmsspp/research_experiments.py` loads YAML, applies overrides, dispatches
  experiment types, and writes provenance files.
- `src/lmsspp/dynamics/pp_transient_research.py` implements research
  simulations, diagnostics, plots, and reports.
- `tests/test_pp_transient_research.py` validates the research simulation layer.
- `tests/test_research_experiments.py` validates YAML orchestration.

This split keeps the YAML files lightweight while still making new experiment
families reproducible once they are promoted into the Python dispatcher.
