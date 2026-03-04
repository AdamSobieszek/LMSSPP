FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV LMS_BOOTSTRAP_WIDGET=ball3d
ENV LMS_BOOTSTRAP_API_BASE=./api/recompute
ENV LMS_BOOTSTRAP_SEED=0
ENV LMS_BOOTSTRAP_W_MODE=autograd
ENV LMS_BOOTSTRAP_INCLUDE_POINTS=true
ENV LMS_BOOTSTRAP_POINT_DECIMATION=1

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir -r deploy/iframe_app/requirements.txt "numpy>=1.26" "torch>=2.2"

EXPOSE 8000

CMD ["bash", "-lc", "python scripts/bootstrap_bundle.py && uvicorn deploy.iframe_app.app:app --host 0.0.0.0 --port ${PORT}"]

