#!/usr/bin/env python
"""Benchmark PP CIC/FFT field evaluation backends.

Example:
    python scripts/benchmark_pp_cs_backend.py --backends numpy,torch --grids 256,512 --counts 20000,100000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lmsspp.dynamics.pp_cs_equilibria import SimulationConfig, _make_pp_backend, make_initial_condition  # noqa: E402


def parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", default="numpy,torch", help="comma-separated backend list")
    parser.add_argument("--grids", default="256,512", help="comma-separated grid sizes")
    parser.add_argument("--counts", default="20000,100000", help="comma-separated total particle counts")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--domain-radius", type=float, default=9.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--K", type=float, default=1.0)
    parser.add_argument("--device", default=None, help="torch device override")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "float64"))
    args = parser.parse_args()

    backends = parse_csv_strings(args.backends)
    grids = parse_csv_ints(args.grids)
    counts = parse_csv_ints(args.counts)
    repeats = max(1, int(args.repeats))

    print("backend,device,dtype,N,G,seconds_per_eval")
    for backend in backends:
        for G in grids:
            for N in counts:
                config = SimulationConfig(
                    alpha=args.alpha,
                    K=args.K,
                    n_fibers=1,
                    n_per_fiber=N,
                    grid_size=G,
                    domain_radius=args.domain_radius,
                    backend=backend,  # type: ignore[arg-type]
                    device=args.device,
                    dtype=args.dtype,  # type: ignore[arg-type]
                    make_dashboard=False,
                    make_animation=False,
                )
                initial = make_initial_condition(config)
                solver = _make_pp_backend(config)
                x = solver.asarray(initial.x)
                solver.A_at_particles(x)
                solver.synchronize()
                t0 = time.perf_counter()
                for _ in range(repeats):
                    solver.A_at_particles(x)
                solver.synchronize()
                elapsed = (time.perf_counter() - t0) / repeats
                print(f"{solver.backend_name},{solver.device_name},{solver.dtype_name},{N},{G},{elapsed:.8f}")


if __name__ == "__main__":
    main()
