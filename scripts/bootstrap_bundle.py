#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lmsspp.LMS import write_lms_static_bundle  # noqa: E402


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _parse_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except Exception:
        return default


def main() -> None:
    widget = os.environ.get("LMS_BOOTSTRAP_WIDGET", "ball3d").strip() or "ball3d"
    w_mode = os.environ.get("LMS_BOOTSTRAP_W_MODE", "autograd").strip() or "autograd"
    api_base = os.environ.get("LMS_BOOTSTRAP_API_BASE", "./api/recompute")
    seed = _parse_int(os.environ.get("LMS_BOOTSTRAP_SEED"), 0)
    point_decimation = _parse_int(os.environ.get("LMS_BOOTSTRAP_POINT_DECIMATION"), 1)
    include_points = _parse_bool(os.environ.get("LMS_BOOTSTRAP_INCLUDE_POINTS"), True)

    params_raw = os.environ.get("LMS_BOOTSTRAP_PARAMS_JSON", "{}").strip() or "{}"
    try:
        params = json.loads(params_raw)
        if not isinstance(params, dict):
            raise ValueError("LMS_BOOTSTRAP_PARAMS_JSON must decode to an object.")
    except Exception as exc:
        raise SystemExit(f"Invalid LMS_BOOTSTRAP_PARAMS_JSON: {exc}") from exc

    out_dir = ROOT / "deploy" / "iframe_app" / "static"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_lms_static_bundle(
        out_dir=str(out_dir),
        widget=widget,  # type: ignore[arg-type]
        params=params,
        include_points=include_points,
        point_decimation=max(1, point_decimation),
        seed=seed,
        w_mode=w_mode,  # type: ignore[arg-type]
        api_base=api_base,
    )
    print(f"Bootstrapped LMS static bundle: widget={widget} out={out_dir}")


if __name__ == "__main__":
    main()
