# LMSSPP Documentation

LMSSPP is a research codebase for two related families of alignment dynamics:
the Lipton-Mirollo-Strogatz (LMS) reduction of Kuramoto dynamics on spheres,
and Peszek-Poyato / Cucker-Smale-type dynamics in Euclidean phase space. The
package is organized around reusable Python backends, notebook widgets, and
reproducible experiment runners. Mathematical background and design notes live
in the LMS and PP documentation chapters here and in the [`papers/`](../papers/)
directory.

## Shared Package Pattern

The repository uses the same broad pattern for both dynamics families:

- backend modules under [`src/lmsspp/`](../src/lmsspp/)
- interactive notebook widgets in notebook-oriented modules
- YAML experiment specifications under [`experiments/`](../experiments/)
- command-line entrypoints under [`scripts/`](../scripts/)
- notebooks under [`notebooks/`](../notebooks/) for running visual tools

| Layer | LMS | Peszek-Poyato |
|---|---|---|
| Backend | [`src/lmsspp/core/lms.py`](../src/lmsspp/core/lms.py), canonical gauge helpers | [`src/lmsspp/dynamics/pp_cs_equilibria.py`](../src/lmsspp/dynamics/pp_cs_equilibria.py) |
| Widgets | `lms_ball3d_widget.py`, `lms_optical_widget.py` | PP dynamics widgets in `pp_cs_equilibria.py` |
| YAML experiments | planned LMS experiment API | existing `experiments/*.yaml` + `research_experiments.py` |
| Papers/math | `papers/original_work/*LMS*` | PP notes, implementation notes, future manuscript |

> [!NOTE]
> The PP YAML experiment API exists today. The LMS YAML experiment API is a
> planned extension of the same backend pattern.

## Using the Backends from Python

The LMS backend evolves reduced states of the form `(w, zeta, reference cloud)`
in arbitrary ambient dimension.

```python
import torch

from lmsspp.core.lms import (
    integrate_lms_reduced_euler,
    random_points_on_sphere,
)

n = 64
d = 3
base_points = random_points_on_sphere(n, d, dtype=torch.float64)
weights = torch.full((n,), 1.0 / n, dtype=torch.float64)
w0 = torch.tensor([0.15, 0.0, 0.0], dtype=torch.float64)
zeta0 = torch.eye(d, dtype=torch.float64)

trajectory = integrate_lms_reduced_euler(
    w0=w0,
    zeta0=zeta0,
    base_points=base_points,
    weights=weights,
    dt=0.02,
    steps=100,
    store_points="body",
)
```

The Peszek-Poyato backend exposes dataclass configs and pure run functions.

```python
from lmsspp.dynamics.pp_cs_equilibria import (
    SimulationConfig,
    make_initial_condition,
    run_simulation,
)

config = SimulationConfig(
    n_fibers=4,
    n_per_fiber=50,
    max_steps=20,
    grid_size=64,
    make_dashboard=False,
    make_animation=False,
)
initial = make_initial_condition(config)
result = run_simulation(config, initial)
```

## Running Widgets in Notebooks

The [`notebooks/`](../notebooks/) directory is the place to run the visual
widget types. Current starting points include:

- [`notebooks/lms_optical_widgets.ipynb`](../notebooks/lms_optical_widgets.ipynb)
- [`notebooks/new_LMS.ipynb`](../notebooks/new_LMS.ipynb)
- [`notebooks/PP-W2nu-equilibria.ipynb`](../notebooks/PP-W2nu-equilibria.ipynb)

These notebooks exercise the widget modules and are the most convenient entry
point for interactive inspection of trajectories, gauges, optical charts, and
large-N PP geometry.

## Running Reproducible Experiments

The [`experiments/`](../experiments/) directory contains YAML files for
reproducible PP research workflows. See
[`experiments/README.md`](../experiments/README.md) for the full schema and
extension guide.

Run an experiment by name:

```bash
/opt/anaconda3/envs/manip311/bin/python scripts/run_experiment.py pp_finite_horizon_comparison
```

Run an explicit YAML file:

```bash
/opt/anaconda3/envs/manip311/bin/python scripts/run_experiment.py experiments/pp_timestep_sweep.yaml
```

Apply Hydra-style dotted command-line overrides:

```bash
/opt/anaconda3/envs/manip311/bin/python scripts/run_experiment.py pp_finite_horizon_animation_batch params.n_per_fiber=80 cases.0.seed=2031
```

Overrides are parsed as YAML values, and list indices are supported with numeric
path components such as `cases.0.seed`. Each run writes provenance files into
the selected output directory:

- `resolved_experiment.yaml`: the YAML after command-line overrides
- `orchestration_result.json`: the JSON-serializable result returned by the
  experiment dispatcher

## Where the Mathematics Lives

Start with the shared framework:

- [`docs/shared_backend_design.md`](shared_backend_design.md)
- [`docs/LMS/`](LMS/)
- [`docs/PP/`](PP/)

Then use the research notes and implementation notes:

- [`papers/original_work/LMS_EXACT_BUSEMANN_INVERSION_NOTE.md`](../papers/original_work/LMS_EXACT_BUSEMANN_INVERSION_NOTE.md)
- [`papers/original_work/LMS_GAUGE_INVARIANTS_DRAFT.md`](../papers/original_work/LMS_GAUGE_INVARIANTS_DRAFT.md)
- [`papers/implementation_notes/`](../papers/implementation_notes/)

The documentation here is intended to connect the executable backend layout to
the mathematical picture developed in those notes.
