#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lmsspp.deploy.ngrok_live import (  # noqa: E402
    active_slots,
    load_slots_config,
    local_slot_urls,
    ngrok_slot_urls,
    write_slot_notebooks,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run live LMS notebook widgets through Voila/Jupyter and ngrok.",
    )
    parser.add_argument(
        "--slots-config",
        default="deploy/voila/slots.json",
        help="Path to slot config JSON (created automatically if missing).",
    )
    parser.add_argument(
        "--slots-dir",
        default="deploy/voila/slots",
        help="Directory where generated slot notebooks are written.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8866,
        help="Local port for Voila/Jupyter server.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind host for Voila/Jupyter server.",
    )
    parser.add_argument(
        "--server",
        choices=("voila", "lab"),
        default="voila",
        help="Server backend: 'voila' (recommended) or 'lab'.",
    )
    parser.add_argument(
        "--ngrok-domain",
        default="",
        help="Optional reserved ngrok domain (requires paid plan).",
    )
    parser.add_argument(
        "--ngrok-authtoken",
        default="",
        help="Optional ngrok authtoken; overrides existing env.",
    )
    parser.add_argument(
        "--no-ngrok",
        action="store_true",
        help="Start only local server, skip ngrok tunnel.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Only generate notebooks + print commands/URLs.",
    )
    parser.add_argument(
        "--default-max-slot",
        type=int,
        default=10,
        help="Used only when creating a new config file.",
    )
    return parser.parse_args()


def _server_cmd(server: str, host: str, port: int) -> list[str]:
    if server == "voila":
        return [
            "voila",
            "--no-browser",
            f"--Voila.ip={host}",
            f"--port={port}",
            f"--Voila.root_dir={ROOT}",
        ]
    if server == "lab":
        return [
            "jupyter",
            "lab",
            "--no-browser",
            f"--ip={host}",
            f"--port={port}",
            "--ServerApp.allow_origin=*",
            "--ServerApp.token=",
            "--ServerApp.password=",
            f"--ServerApp.root_dir={ROOT}",
        ]
    raise ValueError(f"Unsupported server mode: {server}")


def _ngrok_cmd(port: int, domain: str) -> list[str]:
    cmd = ["ngrok", "http", str(port)]
    d = (domain or "").strip()
    if d:
        cmd.extend(["--domain", d])
    return cmd


def _wait_for_ngrok_url(timeout_s: int = 25) -> str | None:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        try:
            with urlopen("http://127.0.0.1:4040/api/tunnels", timeout=1.5) as response:
                payload = json.load(response)
            for tunnel in payload.get("tunnels", []):
                if tunnel.get("proto") == "https":
                    url = str(tunnel.get("public_url", "")).strip()
                    if url:
                        return url
        except Exception:
            pass
        time.sleep(0.4)
    return None


def _print_slot_urls(title: str, urls: dict[str, str]) -> None:
    print(f"\n{title}")
    for slot_id in sorted(urls.keys(), key=lambda s: (len(s), s)):
        print(f"  slot {slot_id}: {urls[slot_id]}")


def _terminate(proc: subprocess.Popen[bytes], name: str) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=6)
    except subprocess.TimeoutExpired:
        print(f"{name} did not terminate in time; killing.")
        proc.kill()


def main() -> int:
    args = _parse_args()

    config_path = (ROOT / args.slots_config).resolve()
    slots_dir = (ROOT / args.slots_dir).resolve()
    config = load_slots_config(config_path, default_max_slot=max(0, int(args.default_max_slot)))
    notebooks = write_slot_notebooks(config, slots_dir)
    slot_specs = active_slots(config)
    slot_ids = sorted(slot_specs.keys(), key=lambda s: (len(s), s))

    if not slot_ids:
        print("No active slots. Set at least one slot with enabled=true in:", config_path)
        return 1

    rel_slots_dir = slots_dir.relative_to(ROOT).as_posix()

    print("Generated slot notebooks:")
    for nb in notebooks:
        print(f"  {nb}")
    print(f"\nSlots config: {config_path}")

    local_urls = local_slot_urls(
        slot_ids=slot_ids,
        port=int(args.port),
        server=args.server,
        notebooks_rel_dir=rel_slots_dir,
    )
    _print_slot_urls("Local URLs", local_urls)

    server_cmd = _server_cmd(args.server, args.host, int(args.port))
    ngrok_cmd = _ngrok_cmd(int(args.port), args.ngrok_domain)

    print("\nServer command:")
    print("  " + " ".join(server_cmd))
    if not args.no_ngrok:
        print("Ngrok command:")
        print("  " + " ".join(ngrok_cmd))

    if args.print_only:
        return 0

    env = os.environ.copy()
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + existing_path if existing_path else "")
    if args.ngrok_authtoken:
        env["NGROK_AUTHTOKEN"] = args.ngrok_authtoken

    server_proc = subprocess.Popen(server_cmd, cwd=str(ROOT), env=env)
    print(f"\nStarted {args.server} (pid={server_proc.pid})")

    ngrok_proc: subprocess.Popen[bytes] | None = None
    public_url: str | None = None

    try:
        # Small startup grace period; fail early if the server exits.
        time.sleep(2.0)
        if server_proc.poll() is not None:
            return int(server_proc.returncode or 1)

        if not args.no_ngrok:
            ngrok_proc = subprocess.Popen(ngrok_cmd, cwd=str(ROOT), env=env)
            print(f"Started ngrok (pid={ngrok_proc.pid})")
            public_url = _wait_for_ngrok_url()
            if public_url:
                urls = ngrok_slot_urls(
                    base_url=public_url,
                    slot_ids=slot_ids,
                    server=args.server,
                    notebooks_rel_dir=rel_slots_dir,
                )
                _print_slot_urls("Public ngrok URLs", urls)
            else:
                print("Could not read ngrok tunnel URL from http://127.0.0.1:4040/api/tunnels")

        print("\nRunning. Press Ctrl+C to stop both processes.")
        while True:
            if server_proc.poll() is not None:
                print(f"{args.server} exited with code {server_proc.returncode}")
                return int(server_proc.returncode or 1)
            if ngrok_proc is not None and ngrok_proc.poll() is not None:
                print(f"ngrok exited with code {ngrok_proc.returncode}")
                return int(ngrok_proc.returncode or 1)
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nStopping...")
        return 0
    finally:
        if ngrok_proc is not None:
            _terminate(ngrok_proc, "ngrok")
        _terminate(server_proc, args.server)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    raise SystemExit(main())
