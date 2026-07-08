#!/usr/bin/env python3
"""Wrapper for building and smoke-fitting nominal pyhf workspaces from E79 templates."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e79-run-dir", required=True, help="Run directory produced by e79_toptag_score_template_export.py")
    parser.add_argument("--count-floor", type=float, default=1e-6)
    args = parser.parse_args()
    cmd = [
        sys.executable,
        str(ROOT / "scripts/e80_toptag_pyhf_workspace_smoke.py"),
        "--e79-run-dir",
        args.e79_run_dir,
        "--count-floor",
        str(args.count_floor),
    ]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())

