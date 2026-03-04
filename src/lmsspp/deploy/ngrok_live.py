"""Helpers for serving live LMS widgets through notebook/Voila routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SUPPORTED_WIDGETS = {
    "ball3d",
    "ball3d_backward_two_sheet",
    "ball3d_hydrodynamic_ensemble",
    "circle",
}


def default_slots_config(max_slot: int = 10) -> dict[str, Any]:
    slots: dict[str, dict[str, Any]] = {}
    for idx in range(max_slot + 1):
        slots[str(idx)] = {
            "enabled": True,
            "widget": "ball3d",
            "kwargs": {"rng_seed": idx},
        }
    return {
        "slots": slots,
    }


def _validate_slots_config(config: dict[str, Any]) -> dict[str, Any]:
    if "slots" not in config or not isinstance(config["slots"], dict):
        raise ValueError("slots config must include object key 'slots'.")

    validated: dict[str, dict[str, Any]] = {}
    for slot_id, spec in config["slots"].items():
        if not isinstance(slot_id, str) or not slot_id:
            raise ValueError("slot id must be a non-empty string.")
        if not isinstance(spec, dict):
            raise ValueError(f"slot '{slot_id}' must be an object.")
        enabled = bool(spec.get("enabled", True))
        widget = str(spec.get("widget", "ball3d")).strip()
        if widget not in SUPPORTED_WIDGETS:
            raise ValueError(
                f"slot '{slot_id}' uses unsupported widget '{widget}'. "
                f"Supported: {sorted(SUPPORTED_WIDGETS)}"
            )
        kwargs = spec.get("kwargs", {})
        if not isinstance(kwargs, dict):
            raise ValueError(f"slot '{slot_id}' kwargs must be an object.")
        validated[slot_id] = {
            "enabled": enabled,
            "widget": widget,
            "kwargs": kwargs,
        }

    return {"slots": validated}


def load_slots_config(path: str | Path, *, default_max_slot: int = 10) -> dict[str, Any]:
    path_obj = Path(path).expanduser().resolve()
    if not path_obj.exists():
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        config = default_slots_config(max_slot=default_max_slot)
        path_obj.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return config

    raw = json.loads(path_obj.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("slots config JSON must decode to an object.")
    config = _validate_slots_config(raw)
    path_obj.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def active_slots(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    slots = config.get("slots", {})
    if not isinstance(slots, dict):
        return {}
    return {
        slot_id: spec
        for slot_id, spec in slots.items()
        if isinstance(spec, dict) and bool(spec.get("enabled", True))
    }


def _slot_notebook_json(slot_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    widget = str(spec["widget"])
    kwargs = spec.get("kwargs", {})
    kwargs_json = json.dumps(kwargs, sort_keys=True)

    code = f"""import json
from IPython.display import display
from lmsspp.core.lms import (
    make_lms_ball3d_backward_two_sheet_widget,
    make_lms_ball3d_hydrodynamic_ensemble_widget,
    make_lms_ball3d_widget,
    make_lms_circle_plotly_widget,
)

SLOT_ID = {slot_id!r}
WIDGET = {widget!r}
KWARGS = json.loads({kwargs_json!r})

if WIDGET == "ball3d":
    obj = make_lms_ball3d_widget(**KWARGS)
elif WIDGET == "ball3d_backward_two_sheet":
    obj = make_lms_ball3d_backward_two_sheet_widget(**KWARGS)
elif WIDGET == "ball3d_hydrodynamic_ensemble":
    obj = make_lms_ball3d_hydrodynamic_ensemble_widget(**KWARGS)
elif WIDGET == "circle":
    obj = make_lms_circle_plotly_widget(**KWARGS)
else:
    raise ValueError(f"Unsupported widget type in slot {{SLOT_ID}}: {{WIDGET}}")

display(obj.layout if hasattr(obj, "layout") else obj)
"""

    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# LMSSPP Live Slot {slot_id}\n",
                    "\n",
                    f"- widget: `{widget}`\n",
                    "- this notebook is auto-generated from `deploy/voila/slots.json`\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code.splitlines()],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_slot_notebooks(
    config: dict[str, Any],
    notebooks_dir: str | Path,
) -> list[Path]:
    out_dir = Path(notebooks_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for slot_id, spec in sorted(active_slots(config).items(), key=lambda item: item[0]):
        nb = _slot_notebook_json(slot_id, spec)
        target = out_dir / f"{slot_id}.ipynb"
        target.write_text(json.dumps(nb, indent=2), encoding="utf-8")
        written.append(target)

    return written


def local_slot_urls(
    slot_ids: list[str],
    *,
    port: int,
    server: str,
    notebooks_rel_dir: str,
) -> dict[str, str]:
    return _slot_urls(
        base_url=f"http://127.0.0.1:{port}",
        slot_ids=slot_ids,
        server=server,
        notebooks_rel_dir=notebooks_rel_dir,
    )


def ngrok_slot_urls(
    base_url: str,
    slot_ids: list[str],
    *,
    server: str,
    notebooks_rel_dir: str,
) -> dict[str, str]:
    return _slot_urls(
        base_url=base_url,
        slot_ids=slot_ids,
        server=server,
        notebooks_rel_dir=notebooks_rel_dir,
    )


def _slot_urls(
    *,
    base_url: str,
    slot_ids: list[str],
    server: str,
    notebooks_rel_dir: str,
) -> dict[str, str]:
    base = base_url.rstrip("/")
    rel_dir = notebooks_rel_dir.strip("/").replace("\\", "/")
    urls: dict[str, str] = {}
    for slot in slot_ids:
        if server == "voila":
            urls[slot] = f"{base}/voila/render/{rel_dir}/{slot}.ipynb"
        elif server == "lab":
            urls[slot] = f"{base}/lab/tree/{rel_dir}/{slot}.ipynb"
        else:
            raise ValueError("server must be 'voila' or 'lab'.")
    return urls
