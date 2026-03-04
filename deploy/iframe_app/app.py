"""FastAPI app for iframe-hosted LMS widgets (non-notebook backend)."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
for candidate in (SRC_DIR, ROOT):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from lmsspp.LMS import export_lms_static_payload  # type: ignore  # noqa: E402


STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="LMS iframe app", version="1.0.0")


def _normalize_base_path(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        return ""
    if not value.startswith("/"):
        value = f"/{value}"
    value = value.rstrip("/")
    return "" if value == "/" else value


BASE_PATH = _normalize_base_path(os.environ.get("LMS_BASE_PATH", ""))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_iframe_headers(request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = "frame-ancestors *;"
    if "X-Frame-Options" in response.headers:
        del response.headers["X-Frame-Options"]
    return response


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if BASE_PATH:
    app.mount(
        f"{BASE_PATH}/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static_prefixed",
    )


class RecomputeRequest(BaseModel):
    widget: Literal["ball3d", "ball3d_backward_two_sheet", "circle"] = "ball3d"
    params: dict[str, Any] = Field(default_factory=dict)
    include_points: bool = True
    point_decimation: int = 1
    seed: int = 0
    w_mode: Literal["explicit", "autograd"] = "autograd"


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/index.html")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/recompute")
def recompute(req: RecomputeRequest) -> JSONResponse:
    payload = export_lms_static_payload(
        widget=req.widget,
        params=req.params,
        include_points=bool(req.include_points),
        point_decimation=max(1, int(req.point_decimation)),
        seed=int(req.seed),
        w_mode=req.w_mode,
    )
    return JSONResponse(payload)


if BASE_PATH:
    @app.get(BASE_PATH, include_in_schema=False)
    def root_prefixed_redirect() -> RedirectResponse:
        return RedirectResponse(url=f"{BASE_PATH}/", status_code=307)

    app.add_api_route(
        f"{BASE_PATH}/",
        root,
        methods=["GET"],
        include_in_schema=False,
    )
    app.add_api_route(
        f"{BASE_PATH}/index.html",
        index,
        methods=["GET"],
        include_in_schema=False,
    )
    app.add_api_route(
        f"{BASE_PATH}/api/recompute",
        recompute,
        methods=["POST"],
        include_in_schema=False,
    )
    app.add_api_route(
        f"{BASE_PATH}/healthz",
        healthz,
        methods=["GET"],
        include_in_schema=False,
    )
