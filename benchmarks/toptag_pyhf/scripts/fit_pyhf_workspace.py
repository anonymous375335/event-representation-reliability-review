#!/usr/bin/env python3
"""Wrapper for E81 shifted-pseudo-data profile-stress fits."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e79-run-dir", required=True, help="Run directory containing score_templates.csv")
    parser.add_argument("--count-floor", type=float, default=1e-6)
    args = parser.parse_args()
    cmd = [
        sys.executable,
        str(ROOT / "scripts/e81_toptag_profile_stress.py"),
        "--e79-run-dir",
        args.e79_run_dir,
        "--count-floor",
        str(args.count_floor),
    ]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())

