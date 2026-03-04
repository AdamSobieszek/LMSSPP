# LMS widget export and iframe hosting (non-notebook)

This project now supports two export modes that do not depend on serving `.ipynb` notebooks.

## Mode A: static bundle (no Python runtime at playback time)

Generate `index.html + trajectory.json + metadata.json + config.json`:

```bash
python scripts/export_lms_iframe_static.py \
  --widget ball3d \
  --out dist/lms-ball3d \
  --set N=150 \
  --set steps=600 \
  --set dt=0.05 \
  --seed 0 \
  --w-mode autograd
```

Serve statically:

```bash
cd dist/lms-ball3d
python -m http.server 9000
```

Iframe:

```html
<iframe src="https://<host>/index.html" width="1400" height="980" style="border:0"></iframe>
```

## Mode B: static player + Python recompute backend

Run FastAPI app:

```bash
pip install -r deploy/iframe_app/requirements.txt "numpy>=1.26" "torch>=2.2"
uvicorn deploy.iframe_app.app:app --host 0.0.0.0 --port 8000 --reload
```

The page at `/` can:

- load `trajectory.json` if present
- call `POST /api/recompute` to regenerate trajectories from controls

For Firebase path-proxy setup (`/lms-widget/**`), run with:

```bash
LMS_BASE_PATH=/lms-widget uvicorn deploy.iframe_app.app:app --host 0.0.0.0 --port 8000 --reload
```

## Python APIs

From `/Users/adamsobieszek/PycharmProjects/ManipyTraversal/notebooks/kuramoto/LMS.py`:

- `export_lms_static_payload(...) -> dict`
- `write_lms_static_bundle(out_dir=..., ...) -> Path`
- `make_lms_iframe_widget(widget=..., **kwargs)` (widget factory routing)

## Supported widget types

- `ball3d`
- `ball3d_backward_two_sheet`
- `circle`

## Notes

- Static mode is best for CDN-style hosting and iframes.
- Recompute requires Python backend (`/api/recompute`), but still no notebook server.
- Bundle schema is versioned as `lms_iframe_static_v1`.
- Standalone deploy repo for RunPod/GHCR lives at `deploy/iframe_app_repo/`.
