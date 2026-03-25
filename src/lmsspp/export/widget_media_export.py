"""WebM media bundle export for LMS widget snapshots."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import numpy as np


def _require_media_dependencies() -> tuple[Any, Any]:
    missing: list[str] = []
    if importlib.util.find_spec("kaleido") is None:
        missing.append("kaleido")
    if importlib.util.find_spec("imageio_ffmpeg") is None:
        missing.append("imageio-ffmpeg")

    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception:
        imageio = None
        if "imageio" not in missing:
            missing.append("imageio")

    try:
        from PIL import Image  # type: ignore
    except Exception:
        Image = None
        if "Pillow" not in missing:
            missing.append("Pillow")

    if missing:
        deps = ", ".join(sorted(set(missing)))
        raise RuntimeError(
            "Missing export dependencies: "
            + deps
            + ". Install extras with: pip install 'lmsspp[widgets,export_media]'"
        )

    return imageio, Image


def _as_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _as_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json(v) for v in value]
    return value


def _ensure_even(value: Any, default: int) -> int:
    try:
        iv = int(round(float(value)))
    except Exception:
        iv = int(default)
    if iv < 2:
        iv = int(default)
    if iv % 2 != 0:
        iv += 1
    return max(2, iv)


def _figure_size(fig: Any, *, default_width: int, default_height: int) -> tuple[int, int]:
    layout = getattr(fig, "layout", None)
    width = _ensure_even(getattr(layout, "width", default_width), default_width)
    height = _ensure_even(getattr(layout, "height", default_height), default_height)
    return width, height


def _frame_sequence(start: int, max_frame: int, direction: int) -> list[int]:
    frame_max = max(0, int(max_frame))
    start_clamped = max(0, min(int(start), frame_max))
    step = 1 if int(direction) >= 0 else -1
    if step > 0:
        seq = list(range(start_clamped, frame_max + 1))
    else:
        seq = list(range(start_clamped, -1, -1))
    if not seq:
        return [start_clamped]
    return seq


def _next_run_tag(out_dir: Path) -> str:
    base = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = base
    idx = 1
    while (out_dir / f"{tag}_scene.webm").exists() or (out_dir / f"{tag}_metrics.webm").exists():
        tag = f"{base}_{idx:02d}"
        idx += 1
    return tag


def _load_manifest(path: Path, *, title: str) -> dict[str, Any]:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    else:
        data = {}
    if not isinstance(data, dict):
        data = {}
    if not isinstance(data.get("items"), list):
        data["items"] = []
    if not data.get("title"):
        data["title"] = str(title)
    if "description" not in data:
        data["description"] = "LMS widget trajectory media exports"
    return data


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_as_json(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _append_manifest_items(
    *,
    out_dir: Path,
    title: str,
    scene_name: str,
    metrics_name: str,
    run_meta: dict[str, Any],
) -> None:
    manifest_path = out_dir / "manifest.json"
    manifest = _load_manifest(manifest_path, title=title)
    items = manifest.get("items")
    if not isinstance(items, list):
        items = []
        manifest["items"] = items

    scene_item = {
        "src": scene_name,
        "label": f"{run_meta['run_tag']} scene",
        "kind": "video",
        "meta": {**run_meta, "panel": "scene"},
    }
    metrics_item = {
        "src": metrics_name,
        "label": f"{run_meta['run_tag']} metrics",
        "kind": "video",
        "meta": {**run_meta, "panel": "metrics"},
    }
    items.extend([scene_item, metrics_item])
    _write_manifest(manifest_path, manifest)


def write_lms_widget_webm_bundle(widget: Any, out_dir: str | Path) -> Path:
    """Export current widget playback segment into WebM files and manifest."""
    if bool(getattr(widget, "_recompute_busy", False)):
        raise RuntimeError("Widget recompute is still in progress. Wait for completion and retry.")
    if not getattr(widget, "_traj_cache", None):
        raise RuntimeError("No computed trajectory is available to export.")

    imageio, Image = _require_media_dependencies()

    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    params = dict(widget._params_cache if getattr(widget, "_params_cache", None) else widget._params())
    ui_state = dict(widget._export_capture_ui_state() if hasattr(widget, "_export_capture_ui_state") else {})
    init_info = dict(widget._export_init_info(params) if hasattr(widget, "_export_init_info") else {})
    camera = widget._camera_to_json() if hasattr(widget, "_camera_to_json") else None
    title = str(getattr(widget, "title", "LMS widget media export"))

    frame_start = int(getattr(widget.frame_slider, "value", 0))
    frame_max = int(getattr(widget, "_steps", int(getattr(widget.frame_slider, "max", 0))))
    direction_sign = 1
    try:
        play_step = int(getattr(widget.play, "step", 1))
        direction_sign = 1 if play_step >= 0 else -1
    except Exception:
        direction_sign = 1
    frame_indices = _frame_sequence(frame_start, frame_max, direction_sign)

    scene_w, scene_h = _figure_size(widget.sphere_fig, default_width=960, default_height=760)
    metrics_w, metrics_h = _figure_size(widget.metrics_fig, default_width=980, default_height=760)

    run_tag = _next_run_tag(out_path)
    scene_name = f"{run_tag}_scene.webm"
    metrics_name = f"{run_tag}_metrics.webm"
    scene_path = out_path / scene_name
    metrics_path = out_path / metrics_name
    scene_thumb_path = out_path / f"{run_tag}_scene.jpg"
    metrics_thumb_path = out_path / f"{run_tag}_metrics.jpg"

    saved_ui_state = dict(ui_state)
    saved_camera = camera
    saved_play_step = int(getattr(widget.play, "step", 1))
    was_playing = bool(widget._is_playing()) if hasattr(widget, "_is_playing") else False

    scene_writer = None
    metrics_writer = None
    first_scene = None
    first_metrics = None
    try:
        if hasattr(widget, "_set_playing"):
            widget._set_playing(False)
        elif hasattr(widget.play, "playing"):
            try:
                widget.play.playing = False
            except Exception:
                pass

        writer_kwargs = {
            "format": "FFMPEG",
            "mode": "I",
            "fps": 20,
            "codec": "libvpx-vp9",
            "output_params": ["-pix_fmt", "yuv420p", "-b:v", "0", "-crf", "35", "-an"],
        }
        scene_writer = imageio.get_writer(str(scene_path), **writer_kwargs)
        metrics_writer = imageio.get_writer(str(metrics_path), **writer_kwargs)

        for frame_idx in frame_indices:
            widget._render_frame(int(frame_idx))
            scene_png = widget.sphere_fig.to_image(
                format="png",
                engine="kaleido",
                width=scene_w,
                height=scene_h,
                scale=1,
            )
            metrics_png = widget.metrics_fig.to_image(
                format="png",
                engine="kaleido",
                width=metrics_w,
                height=metrics_h,
                scale=1,
            )
            scene_arr = np.asarray(Image.open(BytesIO(scene_png)).convert("RGB"), dtype=np.uint8)
            metrics_arr = np.asarray(Image.open(BytesIO(metrics_png)).convert("RGB"), dtype=np.uint8)

            if first_scene is None:
                first_scene = scene_arr
            if first_metrics is None:
                first_metrics = metrics_arr

            scene_writer.append_data(scene_arr)
            metrics_writer.append_data(metrics_arr)

    finally:
        if scene_writer is not None:
            scene_writer.close()
        if metrics_writer is not None:
            metrics_writer.close()

        if hasattr(widget, "_export_restore_ui_state"):
            widget._export_restore_ui_state(saved_ui_state)
        if saved_camera is not None:
            try:
                widget.sphere_fig.layout.scene.camera = saved_camera
            except Exception:
                pass
        try:
            widget.play.step = int(saved_play_step)
        except Exception:
            pass
        if was_playing and hasattr(widget, "_set_playing"):
            widget._set_playing(True)

    if first_scene is None or first_metrics is None:
        raise RuntimeError("No frames were captured for media export.")

    Image.fromarray(first_scene).save(scene_thumb_path, format="JPEG", quality=82)
    Image.fromarray(first_metrics).save(metrics_thumb_path, format="JPEG", quality=82)

    run_meta = {
        "run_tag": run_tag,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "widget_kind": str(widget._export_widget_kind() if hasattr(widget, "_export_widget_kind") else "lms_widget"),
        "params": _as_json(params),
        "ui_state": _as_json(ui_state),
        "init_info": _as_json(init_info),
        "camera": _as_json(camera),
        "frame_span": {
            "start_frame": int(frame_indices[0]),
            "end_frame": int(frame_indices[-1]),
            "count": int(len(frame_indices)),
            "direction": "forward" if direction_sign >= 0 else "backward",
        },
        "video_profile": {
            "codec": "libvpx-vp9",
            "pixel_format": "yuv420p",
            "fps": 20,
            "crf": 35,
            "audio": "none",
        },
    }
    _append_manifest_items(
        out_dir=out_path,
        title=title,
        scene_name=scene_name,
        metrics_name=metrics_name,
        run_meta=run_meta,
    )
    return out_path


__all__ = ["write_lms_widget_webm_bundle"]
