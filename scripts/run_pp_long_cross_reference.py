"""Compatibility runner for the long PP disk-vs-cross reference experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lmsspp.dynamics.pp_transient_research import run_long_cross_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("pp_transient_research_results_2000"))
    parser.add_argument("--max-steps", type=int, default=2000)
    args = parser.parse_args()
    run_long_cross_reference(args.out_dir, max_steps=args.max_steps, min_steps=args.max_steps + 10)
    print(f"Wrote long reference to {args.out_dir}")


if __name__ == "__main__":
    main()
