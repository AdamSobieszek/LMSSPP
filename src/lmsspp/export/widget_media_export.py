"""WebM media bundle export for LMS widget snapshots."""

from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import tempfile
import threading
import time
from typing import Any, Callable

import numpy as np


StatusCallback = Callable[[str], None]


def _require_media_dependencies() -> tuple[Any, Any, Any]:
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

    try:
        import kaleido  # type: ignore
    except Exception:
        kaleido = None
        if "kaleido" not in missing:
            missing.append("kaleido")

    if missing:
        deps = ", ".join(sorted(set(missing)))
        raise RuntimeError(
            "Missing export dependencies: "
            + deps
            + ". Install extras with: pip install 'lmsspp[widgets,export_media]'"
        )

    return imageio, Image, kaleido


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


def _figure_size(fig_json: dict[str, Any], *, default_width: int, default_height: int) -> tuple[int, int]:
    layout = fig_json.get("layout", {}) if isinstance(fig_json, dict) else {}
    width = _ensure_even(layout.get("width", default_width), default_width)
    height = _ensure_even(layout.get("height", default_height), default_height)
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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _export_worker_count(default: int) -> int:
    raw = os.environ.get("LMSSPP_EXPORT_WEBM_WORKERS")
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except Exception:
        return int(default)
    return max(1, min(16, value))


def _next_run_tag(out_dir: Path) -> str:
    base = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = base
    idx = 1
    while (out_dir / f"{tag}_scene.webm").exists() or (out_dir / f"{tag}_metrics.png").exists():
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


def _append_manifest_item(
    *,
    out_dir: Path,
    title: str,
    scene_name: str,
    metrics_image_name: str,
    item_meta: dict[str, Any],
    run_tag: str,
) -> None:
    manifest_path = out_dir / "manifest.json"
    manifest = _load_manifest(manifest_path, title=title)
    items = manifest.get("items")
    if not isinstance(items, list):
        items = []
        manifest["items"] = items
    else:
        filtered: list[Any] = []
        for entry in items:
            if isinstance(entry, dict):
                src = str(entry.get("src", "")).strip().lower()
                if src.endswith("metrics.webm") or src.endswith("_metrics.webm"):
                    continue
            filtered.append(entry)
        items[:] = filtered

    scene_item = {
        "src": scene_name,
        "label": f"Scene {run_tag}",
        "kind": "video",
        "meta": item_meta,
        "tabs": [
            {
                "id": "metrics",
                "label": "Metrics",
                "src": metrics_image_name,
                "kind": "image",
            }
        ],
    }
    items.append(scene_item)
    _write_manifest(manifest_path, manifest)


def _set_trace_xyz(fig_json: dict[str, Any], index: int | None, rows: Any) -> None:
    if index is None or not isinstance(index, int):
        return
    data = fig_json.get("data")
    if not isinstance(data, list) or index < 0 or index >= len(data):
        return
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, 3)
    if arr.size == 0:
        x: list[float] = []
        y: list[float] = []
        z: list[float] = []
    else:
        x = arr[:, 0].tolist()
        y = arr[:, 1].tolist()
        z = arr[:, 2].tolist()
    data[index]["x"] = x
    data[index]["y"] = y
    data[index]["z"] = z


def _set_trace_visible(fig_json: dict[str, Any], index: int | None, visible: bool) -> None:
    if index is None or not isinstance(index, int):
        return
    data = fig_json.get("data")
    if not isinstance(data, list) or index < 0 or index >= len(data):
        return
    data[index]["visible"] = bool(visible)


def _apply_scene_state_to_figure_json(fig_json: dict[str, Any], roles: dict[str, Any], state: dict[str, Any]) -> None:
    _set_trace_xyz(fig_json, roles.get("points_marker"), state.get("points"))
    _set_trace_xyz(fig_json, roles.get("w_marker"), state.get("w"))
    _set_trace_xyz(fig_json, roles.get("z_marker"), state.get("z"))
    _set_trace_xyz(fig_json, roles.get("Z_marker"), state.get("Z"))
    _set_trace_xyz(fig_json, roles.get("w_path"), state.get("w_path"))
    _set_trace_xyz(fig_json, roles.get("z_path"), state.get("z_path"))
    _set_trace_xyz(fig_json, roles.get("Z_path"), state.get("Z_path"))
    _set_trace_xyz(fig_json, roles.get("w_vector"), state.get("w_vector"))
    _set_trace_xyz(fig_json, roles.get("z_vector"), state.get("z_vector"))
    _set_trace_xyz(fig_json, roles.get("Z_vector"), state.get("Z_vector"))
    _set_trace_visible(fig_json, roles.get("w_path"), bool(state.get("show_paths", True)))
    _set_trace_visible(fig_json, roles.get("z_path"), bool(state.get("show_paths", True)))
    _set_trace_visible(fig_json, roles.get("Z_path"), bool(state.get("show_paths", True)))
    _set_trace_visible(fig_json, roles.get("w_vector"), bool(state.get("show_vectors", True)))
    _set_trace_visible(fig_json, roles.get("z_vector"), bool(state.get("show_vectors", True)))
    _set_trace_visible(fig_json, roles.get("Z_vector"), bool(state.get("show_vectors", True)))

    _set_trace_xyz(fig_json, roles.get("bar_w_marker"), state.get("bar_w"))
    _set_trace_xyz(fig_json, roles.get("bar_z_marker"), state.get("bar_z"))
    _set_trace_xyz(fig_json, roles.get("bar_Z_marker"), state.get("bar_Z"))
    _set_trace_xyz(fig_json, roles.get("bar_w_path"), state.get("bar_w_path"))
    _set_trace_xyz(fig_json, roles.get("bar_z_path"), state.get("bar_z_path"))
    _set_trace_xyz(fig_json, roles.get("bar_Z_path"), state.get("bar_Z_path"))
    _set_trace_xyz(fig_json, roles.get("bar_w_vector"), state.get("bar_w_vector"))
    _set_trace_xyz(fig_json, roles.get("bar_z_vector"), state.get("bar_z_vector"))
    _set_trace_xyz(fig_json, roles.get("bar_Z_vector"), state.get("bar_Z_vector"))
    _set_trace_visible(fig_json, roles.get("bar_w_path"), bool(state.get("show_paths", True)))
    _set_trace_visible(fig_json, roles.get("bar_z_path"), bool(state.get("show_paths", True)))
    _set_trace_visible(fig_json, roles.get("bar_Z_path"), bool(state.get("show_paths", True)))
    _set_trace_visible(fig_json, roles.get("bar_w_vector"), bool(state.get("show_vectors", True)))
    _set_trace_visible(fig_json, roles.get("bar_z_vector"), bool(state.get("show_vectors", True)))
    _set_trace_visible(fig_json, roles.get("bar_Z_vector"), bool(state.get("show_vectors", True)))


def _build_scene_figure_json(widget: Any, payload: dict[str, Any], frame_idx: int) -> dict[str, Any]:
    ui_state = dict(payload.get("ui_state", {}))
    fig_json = copy.deepcopy(payload["scene_figure_template"])
    scene_state = widget._export_media_scene_state(
        bundle=payload["bundle"],
        t=int(frame_idx),
        params=dict(payload["params"]),
        frame_name=widget._coerce_frame_name(ui_state.get("frame_name")),
        show_paths=bool(ui_state.get("show_paths", True)),
        show_vectors=bool(ui_state.get("show_vectors", True)),
        inversion_enabled=bool(ui_state.get("inversion_enabled", False)),
    )
    _apply_scene_state_to_figure_json(fig_json, dict(payload["scene_trace_roles"]), scene_state)
    return fig_json


class _ProgressTicker:
    def __init__(self, callback: StatusCallback | None, *, prefix: str) -> None:
        self.callback = callback
        self.prefix = prefix
        self._last_emit = 0.0

    def update(self, current: int, total: int) -> None:
        if self.callback is None:
            return
        now = time.perf_counter()
        if current != total and current != 1 and (now - self._last_emit) < 0.25:
            return
        self._last_emit = now
        self.callback(f"{self.prefix} {current}/{total}...")


def _status_emitter(widget: Any, callback: StatusCallback | None) -> StatusCallback | None:
    if callback is None:
        return None
    if not hasattr(widget, "_schedule_on_main_thread"):
        return callback

    def emit(message: str) -> None:
        if threading.current_thread() is threading.main_thread():
            callback(str(message))
            return
        widget._schedule_on_main_thread(lambda: callback(str(message)))

    return emit


def _run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover
            error["exc"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "exc" in error:
        raise error["exc"]
    return result.get("value")


async def _render_png_assets_async(
    *,
    KaleidoClass: Any,
    widget: Any,
    payload: dict[str, Any],
    frame_indices: list[int],
    frames_dir: Path,
    metrics_image_path: Path,
    scene_opts: dict[str, Any],
    metrics_opts: dict[str, Any],
    workers: int,
    enable_gpu: bool,
    status_callback: StatusCallback | None,
) -> None:
    frame_paths = [frames_dir / f"frame_{i:05d}.png" for i in range(1, len(frame_indices) + 1)]
    progress = _ProgressTicker(status_callback, prefix="rendering scene frames")
    metrics_fig = copy.deepcopy(payload["metrics_figure_template"])

    async def _render_batch(worker_count: int) -> None:
        async with KaleidoClass(n=int(worker_count), headless=True, enable_gpu=bool(enable_gpu), timeout=180) as renderer:
            if status_callback is not None:
                status_callback("rendering metrics image...")
            await renderer.write_fig(metrics_fig, path=metrics_image_path, opts=metrics_opts)

            def generator():
                for i, frame_idx in enumerate(frame_indices, start=1):
                    progress.update(i, len(frame_indices))
                    yield {
                        "fig": _build_scene_figure_json(widget, payload, int(frame_idx)),
                        "path": frame_paths[i - 1],
                        "opts": scene_opts,
                    }

            await renderer.write_fig_from_object(generator())

    async def _render_single_session() -> None:
        async with KaleidoClass(n=1, headless=True, enable_gpu=bool(enable_gpu), timeout=180) as renderer:
            if status_callback is not None:
                status_callback("rendering metrics image...")
            await renderer.write_fig(metrics_fig, path=metrics_image_path, opts=metrics_opts)
            for i, frame_idx in enumerate(frame_indices, start=1):
                progress.update(i, len(frame_indices))
                fig = _build_scene_figure_json(widget, payload, int(frame_idx))
                await renderer.write_fig(fig, path=frame_paths[i - 1], opts=scene_opts)

    try:
        await _render_batch(max(1, int(workers)))
    except Exception:
        if workers <= 1:
            raise
        if status_callback is not None:
            status_callback("batch render fallback: retrying with one persistent renderer...")
        await _render_single_session()


def _build_kaleido_specs(
    *,
    widget: Any,
    payload: dict[str, Any],
    frame_indices: list[int],
    frame_paths: list[Path],
    metrics_image_path: Path,
    scene_w: int,
    scene_h: int,
    metrics_w: int,
    metrics_h: int,
    status_callback: StatusCallback | None,
) -> list[dict[str, Any]]:
    progress = _ProgressTicker(status_callback, prefix="building frame figures")
    specs: list[dict[str, Any]] = [
        {
            "fig": copy.deepcopy(payload["metrics_figure_template"]),
            "path": metrics_image_path,
            "opts": {"format": "png", "width": metrics_w, "height": metrics_h, "scale": 1},
        }
    ]
    total = len(frame_indices)
    for i, (frame_idx, frame_path) in enumerate(zip(frame_indices, frame_paths), start=1):
        progress.update(i, total)
        specs.append(
            {
                "fig": _build_scene_figure_json(widget, payload, int(frame_idx)),
                "path": frame_path,
                "opts": {"format": "png", "width": scene_w, "height": scene_h, "scale": 1},
            }
        )
    return specs


def _render_png_assets_kaleido_batch(
    *,
    kaleido_mod: Any,
    widget: Any,
    payload: dict[str, Any],
    frame_indices: list[int],
    frames_dir: Path,
    metrics_image_path: Path,
    scene_w: int,
    scene_h: int,
    metrics_w: int,
    metrics_h: int,
    workers: int,
    enable_gpu: bool,
    status_callback: StatusCallback | None,
) -> list[Path]:
    frame_paths = [frames_dir / f"frame_{i:05d}.png" for i in range(1, len(frame_indices) + 1)]
    specs = _build_kaleido_specs(
        widget=widget,
        payload=payload,
        frame_indices=frame_indices,
        frame_paths=frame_paths,
        metrics_image_path=metrics_image_path,
        scene_w=scene_w,
        scene_h=scene_h,
        metrics_w=metrics_w,
        metrics_h=metrics_h,
        status_callback=status_callback,
    )
    kopts = {
        "n": max(1, int(workers)),
        "headless": True,
        "enable_gpu": bool(enable_gpu),
        "timeout": 180,
    }
    if status_callback is not None:
        gpu_tag = "gpu" if enable_gpu else "cpu"
        status_callback(f"batch rendering {len(specs)} figures with kaleido ({gpu_tag}, workers={kopts['n']})...")
    kaleido_mod.write_fig_from_object_sync(specs, kopts=kopts)
    return frame_paths


def _encode_scene_webm(
    *,
    imageio: Any,
    Image: Any,
    frame_paths: list[Path],
    scene_path: Path,
    status_callback: StatusCallback | None,
) -> np.ndarray:
    if status_callback is not None:
        status_callback("encoding webm...")
    writer = imageio.get_writer(
        str(scene_path),
        format="FFMPEG",
        mode="I",
        fps=20,
        codec="libvpx-vp9",
        output_params=["-pix_fmt", "yuv420p", "-b:v", "0", "-crf", "35", "-an"],
    )
    first_scene: np.ndarray | None = None
    try:
        for path in frame_paths:
            with Image.open(path) as img:
                arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
            if first_scene is None:
                first_scene = arr
            writer.append_data(arr)
    finally:
        writer.close()
    if first_scene is None:
        raise RuntimeError("No frames were rendered for WebM export.")
    return first_scene


def write_lms_widget_webm_bundle(
    widget: Any,
    out_dir: str | Path,
    *,
    status_callback: StatusCallback | None = None,
) -> Path:
    """Export current widget playback segment into WebM files and manifest."""
    if bool(getattr(widget, "_recompute_busy", False)):
        raise RuntimeError("Widget recompute is still in progress. Wait for completion and retry.")
    if not getattr(widget, "_traj_cache", None):
        raise RuntimeError("No computed trajectory is available to export.")

    imageio, Image, kaleido_mod = _require_media_dependencies()
    emit_status = _status_emitter(widget, status_callback)

    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    for old_metrics in out_path.glob("*metrics.webm"):
        try:
            old_metrics.unlink()
        except Exception:
            pass

    if emit_status is not None:
        emit_status("preparing export...")
    payload_t0 = time.perf_counter()
    payload = widget._export_media_payload()
    title = str(payload.get("title", getattr(widget, "title", "LMS widget media export")))

    ui_state = dict(payload.get("ui_state", {}))
    params = dict(payload.get("params", {}))
    init_info = dict(payload.get("init_info", {}))

    frame_start = int(ui_state.get("frame", 0))
    frame_max = int(max(0, len(np.asarray(payload["bundle"]["w"])) - 1))
    play_step = int(ui_state.get("play_step", 1))
    direction_sign = 1 if play_step >= 0 else -1
    frame_indices = _frame_sequence(frame_start, frame_max, direction_sign)

    scene_template = dict(payload["scene_figure_template"])
    metrics_template = dict(payload["metrics_figure_template"])
    scene_w, scene_h = _figure_size(scene_template, default_width=960, default_height=760)
    metrics_w, metrics_h = _figure_size(metrics_template, default_width=980, default_height=760)

    run_tag = _next_run_tag(out_path)
    scene_name = f"{run_tag}_scene.webm"
    metrics_image_name = f"{run_tag}_metrics.png"
    scene_path = out_path / scene_name
    metrics_image_path = out_path / metrics_image_name
    scene_thumb_path = out_path / f"{run_tag}_scene.jpg"

    saved_play_step = int(getattr(widget.play, "step", 1))
    was_playing = bool(widget._is_playing()) if hasattr(widget, "_is_playing") else False
    timings: dict[str, float] = {}
    timings["prepare_s"] = time.perf_counter() - payload_t0
    first_scene: np.ndarray | None = None

    render_t0 = time.perf_counter()
    try:
        if hasattr(widget, "_set_playing"):
            widget._set_playing(False)
        elif hasattr(widget.play, "playing"):
            try:
                widget.play.playing = False
            except Exception:
                pass

        default_workers = min(6, max(1, int(os.cpu_count() or 1)))
        workers = _export_worker_count(default_workers)
        gpu_default = bool(platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"})
        gpu_enabled = _env_bool("LMSSPP_EXPORT_ENABLE_GPU", gpu_default)
        with tempfile.TemporaryDirectory(prefix=f"{run_tag}_frames_", dir=str(out_path)) as tmp_dir_name:
            frames_dir = Path(tmp_dir_name)
            try:
                frame_paths = _render_png_assets_kaleido_batch(
                    kaleido_mod=kaleido_mod,
                    widget=widget,
                    payload=payload,
                    frame_indices=frame_indices,
                    frames_dir=frames_dir,
                    metrics_image_path=metrics_image_path,
                    scene_w=scene_w,
                    scene_h=scene_h,
                    metrics_w=metrics_w,
                    metrics_h=metrics_h,
                    workers=workers,
                    enable_gpu=gpu_enabled,
                    status_callback=emit_status,
                )
            except Exception:
                if gpu_enabled:
                    if emit_status is not None:
                        emit_status("kaleido gpu batch failed, retrying without gpu...")
                    try:
                        frame_paths = _render_png_assets_kaleido_batch(
                            kaleido_mod=kaleido_mod,
                            widget=widget,
                            payload=payload,
                            frame_indices=frame_indices,
                            frames_dir=frames_dir,
                            metrics_image_path=metrics_image_path,
                            scene_w=scene_w,
                            scene_h=scene_h,
                            metrics_w=metrics_w,
                            metrics_h=metrics_h,
                            workers=workers,
                            enable_gpu=False,
                            status_callback=emit_status,
                        )
                    except Exception:
                        scene_opts = {"format": "png", "width": scene_w, "height": scene_h, "scale": 1}
                        metrics_opts = {"format": "png", "width": metrics_w, "height": metrics_h, "scale": 1}
                        _run_async(
                            _render_png_assets_async(
                                KaleidoClass=kaleido_mod.Kaleido,
                                widget=widget,
                                payload=payload,
                                frame_indices=frame_indices,
                                frames_dir=frames_dir,
                                metrics_image_path=metrics_image_path,
                                scene_opts=scene_opts,
                                metrics_opts=metrics_opts,
                                workers=workers,
                                enable_gpu=False,
                                status_callback=emit_status,
                            )
                        )
                        frame_paths = [frames_dir / f"frame_{i:05d}.png" for i in range(1, len(frame_indices) + 1)]
                else:
                    scene_opts = {"format": "png", "width": scene_w, "height": scene_h, "scale": 1}
                    metrics_opts = {"format": "png", "width": metrics_w, "height": metrics_h, "scale": 1}
                    _run_async(
                        _render_png_assets_async(
                            KaleidoClass=kaleido_mod.Kaleido,
                            widget=widget,
                            payload=payload,
                            frame_indices=frame_indices,
                            frames_dir=frames_dir,
                            metrics_image_path=metrics_image_path,
                            scene_opts=scene_opts,
                            metrics_opts=metrics_opts,
                            workers=workers,
                            enable_gpu=False,
                            status_callback=emit_status,
                        )
                    )
                    frame_paths = [frames_dir / f"frame_{i:05d}.png" for i in range(1, len(frame_indices) + 1)]
            timings["render_png_s"] = time.perf_counter() - render_t0

            encode_t0 = time.perf_counter()
            first_scene = _encode_scene_webm(
                imageio=imageio,
                Image=Image,
                frame_paths=frame_paths,
                scene_path=scene_path,
                status_callback=emit_status,
            )
            timings["encode_webm_s"] = time.perf_counter() - encode_t0
    finally:
        try:
            widget.play.step = int(saved_play_step)
        except Exception:
            pass
        if was_playing and hasattr(widget, "_set_playing"):
            widget._set_playing(True)

    if first_scene is None:
        raise RuntimeError("No frames were rendered for WebM export.")

    thumb_t0 = time.perf_counter()
    Image.fromarray(first_scene).save(scene_thumb_path, format="JPEG", quality=82)
    timings["thumbnail_s"] = time.perf_counter() - thumb_t0

    manifest_t0 = time.perf_counter()
    item_meta = {
        "widget_kind": str(payload.get("widget_kind", "lms_widget")),
        "params": _as_json(params),
        "init_state_mode": init_info.get("init_state_mode"),
        "init_state_label": init_info.get("init_state_label"),
    }
    if emit_status is not None:
        emit_status("writing manifest...")
    _append_manifest_item(
        out_dir=out_path,
        title=title,
        scene_name=scene_name,
        metrics_image_name=metrics_image_name,
        item_meta=_as_json(item_meta),
        run_tag=run_tag,
    )
    timings["write_manifest_s"] = time.perf_counter() - manifest_t0
    timings["total_s"] = sum(timings.values())
    timings["workers"] = workers
    timings["apple_silicon_gpu_hint"] = gpu_enabled
    widget._last_media_export_profile = timings
    return out_path


__all__ = ["write_lms_widget_webm_bundle"]
