#!/usr/bin/env python3
"""Print the packaged E91d cross-shard summary for reviewer inspection."""

from __future__ import annotations

import csv
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE.parents[1]


def repo_path(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def main() -> int:
    summary = PACKAGE / "outputs/e91d_cross_shard_profile_summary.csv"
    rows = list(csv.DictReader(summary.open(newline="", encoding="utf-8")))
    print("E91d packaged profile-stress summary")
    print("settings:", len(rows))
    print("AUC gate passed:", all(float(row["delta_auc"]) >= -0.002 for row in rows))
    print("mean-bias improvements:", sum(float(row["mean_improvement"]) > 0 for row in rows), "/", len(rows))
    print("max-bias improvements:", sum(float(row["max_improvement"]) > 0 for row in rows), "/", len(rows))
    print("See:", repo_path(summary))
    print("Aggregate:", repo_path(PACKAGE / "outputs/e91d_aggregate_summary.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
