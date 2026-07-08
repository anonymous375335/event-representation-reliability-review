#!/usr/bin/env python3
"""E75a preflight for CERN Open Data record 80030 TopTag systematic dataset."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
DEFAULT_API_URL = "https://opendata.cern.ch/api/records/80030"


def create_run_dir(prefix: str) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    for suffix in [""] + [f"-{index:02d}" for index in range(1, 100)]:
        run_dir = RUNS / f"{timestamp}-{prefix}{suffix}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"could not create unique run directory for {prefix}")


def fetch_json(url: str, timeout: int) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "year1-e75a-preflight/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def file_indices(record: dict) -> list[dict]:
    return record.get("metadata", {}).get("_file_indices", [])


def iter_files(index: dict):
    for item in index.get("files", []):
        yield {
            "index_description": index.get("description", ""),
            "filename": item.get("filename", ""),
            "size_bytes": int(item.get("size", 0) or 0),
            "checksum": item.get("checksum", ""),
            "uri": item.get("uri", ""),
            "availability": item.get("availability", ""),
        }


def classify_index(description: str) -> str:
    name = description.replace("_file_index.json", "").replace(".json", "")
    if name in {"nominal_train", "nominal_test", "train", "test"}:
        return "nominal"
    if "nominal" in name:
        return "nominal"
    return "systematic"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sample-per-index", type=int, default=2)
    args = parser.parse_args()

    run_dir = create_run_dir("e75a-toptag-record80030-preflight")
    REPORTS.mkdir(parents=True, exist_ok=True)
    record = fetch_json(args.api_url, args.timeout)
    raw_json = run_dir / "record_80030.json"
    raw_json.write_text(json.dumps(record, indent=2), encoding="utf-8")

    index_rows = []
    file_rows = []
    sample_rows = []
    for index in file_indices(record):
        files = list(iter_files(index))
        total_size = sum(row["size_bytes"] for row in files)
        description = index.get("description", "")
        index_rows.append(
            {
                "description": description,
                "kind": classify_index(description),
                "n_files": len(files),
                "total_size_bytes": total_size,
                "total_size_gib": total_size / (1024**3),
                "first_file": files[0]["filename"] if files else "",
                "first_uri": files[0]["uri"] if files else "",
            }
        )
        file_rows.extend(files)
        for row in files[: args.sample_per_index]:
            sample_rows.append(row)

    index_csv = run_dir / "record80030_file_indices.csv"
    file_csv = run_dir / "record80030_files.csv"
    sample_csv = run_dir / "record80030_small_sample_manifest.csv"
    write_csv(index_csv, index_rows)
    write_csv(file_csv, file_rows)
    write_csv(sample_csv, sample_rows)

    title = record.get("metadata", {}).get("title", "CERN Open Data record 80030")
    total_files = len(file_rows)
    total_size = sum(row["size_bytes"] for row in file_rows)
    report_path = REPORTS / f"e75a_toptag_record80030_preflight_{dt.datetime.now():%Y%m%d}.md"
    lines = [
        "# E75a TopTag Record 80030 Preflight",
        "",
        f"- run_dir: `{run_dir}`",
        f"- generated_at: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- api_url: `{args.api_url}`",
        f"- title: {title}",
        f"- total_files_seen: {total_files}",
        f"- total_size_gib_seen: {total_size / (1024**3):.2f}",
        f"- index_csv: `{index_csv}`",
        f"- file_csv: `{file_csv}`",
        f"- small_sample_manifest: `{sample_csv}`",
        "- run note: preflight indexes record metadata and sample manifests.",
        "",
        "## File Indices",
        "",
        "| description | kind | files | size GiB | first file |",
        "|---|---|---:|---:|---|",
    ]
    for row in index_rows:
        lines.append(
            f"| {row['description']} | {row['kind']} | {row['n_files']} | "
            f"{row['total_size_gib']:.2f} | `{row['first_file']}` |"
        )
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "- Inspect `record80030_small_sample_manifest.csv` and choose one nominal index plus one or two systematic indices.",
            "- Download only the selected first files for E75b schema/loader smoke.",
            "- Do not start full-data TopTag training until the small loader verifies schema, labels, weights, and domain tags.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "experiment": "E75a TopTag record 80030 preflight",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "api_url": args.api_url,
        "raw_json": str(raw_json),
        "index_csv": str(index_csv),
        "file_csv": str(file_csv),
        "sample_csv": str(sample_csv),
        "report": str(report_path),
        "total_files_seen": total_files,
        "total_size_gib_seen": total_size / (1024**3),
        "status": "done",
    }
    (run_dir / "metrics.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "status.txt").write_text(f"status: done\nreport: {report_path}\n", encoding="utf-8")
    print(f"E75a TopTag preflight done: {run_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
