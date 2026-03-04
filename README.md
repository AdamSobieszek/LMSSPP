# LMSSPP

Standalone repository layout for LMS Kuramoto widgets, notebooks, and deploy artifacts.

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

## Local setup

```bash
cd pitch-website/public/notebooks/kuramoto/LMSSPP
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install -e ".[backend,widgets,torch]"
```

Run backend:

```bash
uvicorn deploy.iframe_app.app:app --host 0.0.0.0 --port 8000 --reload
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
