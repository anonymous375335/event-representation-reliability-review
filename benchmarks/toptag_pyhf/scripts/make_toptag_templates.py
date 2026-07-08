#!/usr/bin/env python3
"""Wrapper for rebuilding TopTag score templates from selected public shards."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MANIFEST_DIR = ROOT / "benchmarks" / "toptag_pyhf" / "manifests"
MANIFESTS = {
    "shard000": MANIFEST_DIR / "record80030_files.csv",
    "shard001": MANIFEST_DIR / "selected_shard001.csv",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard", choices=sorted(MANIFESTS), default="shard000")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--max-events-per-domain", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "scripts/e79_toptag_score_template_export.py"),
        "--manifest",
        str(MANIFESTS[args.shard]),
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--max-events-per-domain",
        str(args.max_events_per_domain),
        "--batch-size",
        str(args.batch_size),
    ]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
