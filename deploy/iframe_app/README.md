# LMS iframe backend

This FastAPI app serves:

- static iframe player at `/`
- recompute API at `/api/recompute`
- health check at `/healthz`

## Run locally

From `pitch-website/public/notebooks/kuramoto/LMSSPP`:

```bash
pip install -U pip
pip install -e .
pip install -r deploy/iframe_app/requirements.txt "numpy>=1.26" "torch>=2.2"
uvicorn deploy.iframe_app.app:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- `http://localhost:8000/`
- `http://localhost:8000/healthz`

## API

- `POST /api/recompute`

Example request:

```json
{
  "widget": "ball3d",
  "params": {"N": 150, "steps": 400, "dt": 0.05},
  "include_points": true,
  "point_decimation": 1,
  "seed": 0,
  "w_mode": "autograd"
}
```

Response schema: `lms_iframe_static_v1`.

## Iframe usage

```html
<iframe
  src="https://<your-host>/"
  width="1400"
  height="980"
  style="border:0"
  loading="lazy">
</iframe>
```

## Base path mode

If deployed behind a prefixed path (for example a proxy rewrite), set:

```bash
LMS_BASE_PATH=/lms-widget
```

Then the app serves:

- `/lms-widget/`
- `/lms-widget/api/recompute`
- `/lms-widget/static/...`

## Docker

Build from repository root:

```bash
docker build -t lmsspp-iframe:latest .
docker run --rm -p 8000:8000 lmsspp-iframe:latest
```

Alternative:

```bash
docker build -f deploy/iframe_app/Dockerfile -t lmsspp-iframe:latest .
```

## GHCR automation

Workflow file:

- `.github/workflows/build-and-push-ghcr.yml`
