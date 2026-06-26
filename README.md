# LMSSPP

Standalone repository layout for LMS/Kuramoto widgets, Peszek--Poyato dynamics
experiments, notebooks, and deploy artifacts.

## Layout

```
LMSSPP/
  src/
    lmsspp/
      lms_ball3d_widget.py
      lms_plotly_widget.py
      core/
        lms.py
      export/
        iframe_export.py
      integrations/
        ...
  notebooks/
    *.ipynb
    docs/
  deploy/
    iframe_app/
      app.py
      static/
```

Main widget code stays at package top-level. Non-widget logic is organized in subpackages:

- `lmsspp.core`: LMS dynamics and simulation primitives
- `lmsspp.export`: static/iframe payload and bundle export
- `lmsspp.integrations`: notebook integration widgets

Backward-compatible import shims are kept:

- `lmsspp.LMS` -> `lmsspp.core.lms`
- `lmsspp.lms_iframe_export` -> `lmsspp.export.iframe_export`

## Local Setup

The project uses a `src/` Python package layout and requires Python 3.11 or
newer. In this repository, the working environment commonly used for scripts is:

```bash
/opt/anaconda3/envs/manip311/bin/python
```

### Option A: Editable Install

Use this when you want package imports, scripts, and local edits to work
together:

```bash
cd pitch-website/public/notebooks/kuramoto/LMSSPP
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[backend,widgets,torch,live,export_media,research]"
```

### Option B: Requirements File

Use this for a simple all-in-one notebook/research environment:

```bash
cd pitch-website/public/notebooks/kuramoto/LMSSPP
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

If you are using the project's conda environment:

```bash
/opt/anaconda3/envs/manip311/bin/python -m pip install -r requirements.txt
/opt/anaconda3/envs/manip311/bin/python -m pip install -e .
```

## Where To Go

### Peszek--Poyato Dynamics

Core numerical PP kernels, FFT backends, adaptive timestepping, and automatic
`K` handling live in:

```text
src/lmsspp/dynamics/PP.py
```

Interactive PP widgets and simulation orchestration live in:

```text
src/lmsspp/dynamics/pp_cs_equilibria.py
```

Useful notebook constructors include:

```python
from lmsspp.dynamics.pp_cs_equilibria import (
    SimulationConfig,
    make_dynamics_widget,
    make_finite_horizon_gauge_averaged_widget,
    make_hamiltonian_exponent_widget,
)
```

### Reproducible Research Experiments

YAML experiment files live under:

```text
experiments/
```

Run them through the repository-level runner:

```bash
/opt/anaconda3/envs/manip311/bin/python scripts/run_experiment.py pp_k_scale_calibration
```

Experiment orchestration is in:

```text
src/lmsspp/research_experiments.py
```

Transient PP research utilities, sweeps, diagnostics, and MP4/PNG generation are
in:

```text
src/lmsspp/dynamics/pp_transient_research.py
```

See [experiments/README.md](experiments/README.md) for the YAML workflow and
supported experiment types.

### LMS / Cucker--Smale Widgets

Core LMS/gauge code lives in:

```text
src/lmsspp/core/
```

Main notebook widgets live at package top level:

```text
src/lmsspp/lms_ball3d_widget.py
src/lmsspp/lms_ball4d_widget.py
src/lmsspp/lms_optical_widget.py
src/lmsspp/cucker_smale_ball3d_widget.py
```

Notebook integration examples live in:

```text
src/lmsspp/integrations/
notebooks/
```

### Exports And Media

Static iframe/widget export helpers live in:

```text
src/lmsspp/export/
```

Generated/exported artifacts are generally written under:

```text
artifacts/
exports/
temp_research/
```

### Backend / Iframe App

```bash
uvicorn deploy.iframe_app.app:app --host 0.0.0.0 --port 8000 --reload
```

The backend app is in:

```text
deploy/iframe_app/app.py
```

### Tests

Run the Python test suite with:

```bash
PYTHONPATH=src /opt/anaconda3/envs/manip311/bin/python -m unittest discover -s tests
```

## Docker + GHCR

- Docker build context: repository root
- Runtime app: `deploy/iframe_app/app.py`
- GHCR workflow: `.github/workflows/build-and-push-ghcr.yml`

### Build locally

```bash
docker build -t lmsspp-iframe:latest .
docker run --rm -p 8000:8000 lmsspp-iframe:latest
```
