# Live Notebook Widgets via ngrok

This flow serves the **original Python widget classes** through a notebook kernel.
No static bundle build is used.

## 1. Install runtime deps

From `LMSSPP` root:

```bash
pip install -e .
pip install "torch>=2.2" "ipywidgets>=8.0" "plotly>=5.0" "voila>=0.5" "jupyterlab>=4.0"
```

Also install/configure ngrok CLI separately.

## 2. Configure slots

Edit:

- `deploy/voila/slots.json`

Each slot creates one public subpage.

Supported widget values:

- `ball3d`
- `ball3d_backward_two_sheet`
- `ball3d_hydrodynamic_ensemble`
- `circle`

## 3. Run live server + ngrok

```bash
python scripts/run_live_ngrok.py
```

Optional reserved ngrok domain:

```bash
python scripts/run_live_ngrok.py --ngrok-domain your-domain.ngrok.app
```

The script prints URLs like:

- `https://<ngrok>/voila/render/deploy/voila/slots/0.ipynb`
- `https://<ngrok>/voila/render/deploy/voila/slots/1.ipynb`

## Useful flags

- `--print-only` generate notebooks and print commands/URLs only
- `--server lab` use JupyterLab instead of Voila
- `--no-ngrok` run local server only

