#!/usr/bin/env python3
"""E75b schema/loader smoke for CERN Open Data record 80030 TopTag files."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import os
import shutil
import subprocess
from pathlib import Path

import h5py


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
DATA_RAW_ROOT = Path(
    os.environ.get(
        "YEAR1_DATA_RAW",
        str(ROOT / "data_raw"),
    )
)
DATA_RAW = DATA_RAW_ROOT / "toptag_record80030_e75b"
DEFAULT_RECORD80030_MANIFEST = ROOT / "benchmarks" / "toptag_pyhf" / "manifests" / "record80030_files.csv"


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


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_first_files(rows: list[dict], indices: list[str]) -> list[dict]:
    selected = []
    for index in indices:
        matches = [row for row in rows if row["index_description"] == index]
        if not matches:
            raise ValueError(f"no rows found for index {index}")
        selected.append(matches[0])
    return selected


def root_uri_to_https(uri: str) -> str:
    prefix = "root://eospublic.cern.ch//"
    if uri.startswith(prefix):
        return "https://eospublic.cern.ch//" + uri[len(prefix) :]
    return uri


def download_file(row: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / row["filename"]
    expected_size = int(row["size_bytes"])
    if path.exists() and path.stat().st_size == expected_size:
        return path
    tmp = path.with_suffix(path.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    url = root_uri_to_https(row["uri"])
    cmd = [
        "aria2c",
        "--check-certificate=false",
        "-x",
        "16",
        "-s",
        "16",
        "-k",
        "1M",
        "--file-allocation=none",
        "--allow-overwrite=true",
        "-d",
        str(out_dir),
        "-o",
        tmp.name,
        url,
    ]
    subprocess.run(cmd, check=True)
    actual_size = tmp.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(f"size mismatch for {row['filename']}: got {actual_size}, expected {expected_size}")
    control_file = tmp.with_name(tmp.name + ".aria2")
    if control_file.exists():
        control_file.unlink()
    tmp.replace(path)
    return path


def inspect_h5_gz(path: Path, work_dir: Path, max_datasets: int) -> dict:
    h5_path = work_dir / path.name.removesuffix(".gz")
    with gzip.open(path, "rb") as src, h5_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)

    datasets = []
    top_keys = []
    with h5py.File(h5_path, "r") as handle:
        top_keys = list(handle.keys())

        def visit(name: str, obj) -> None:
            if isinstance(obj, h5py.Dataset) and len(datasets) < max_datasets:
                datasets.append(
                    {
                        "name": name,
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                        "compression": str(obj.compression),
                    }
                )

        handle.visititems(visit)
    h5_size = h5_path.stat().st_size
    h5_path.unlink()
    return {
        "filename": path.name,
        "compressed_size_bytes": path.stat().st_size,
        "decompressed_h5_size_bytes": h5_size,
        "top_keys": top_keys,
        "datasets": datasets,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_RECORD80030_MANIFEST,
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        default=["test_nominal_file_index.json", "esup_file_index.json", "esdown_file_index.json"],
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_RAW)
    parser.add_argument("--max-datasets", type=int, default=200)
    args = parser.parse_args()

    run_dir = create_run_dir("e75b-toptag-record80030-schema-smoke")
    REPORTS.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.manifest)
    selected = select_first_files(rows, args.indices)
    downloaded = []
    schema = []
    for row in selected:
        file_path = download_file(row, args.data_dir)
        downloaded.append(
            {
                "index_description": row["index_description"],
                "filename": row["filename"],
                "size_bytes": file_path.stat().st_size,
                "checksum_expected": row["checksum"],
                "uri": row["uri"],
                "local_path": str(file_path),
            }
        )
        schema.append(inspect_h5_gz(file_path, run_dir, args.max_datasets))

    manifest_csv = run_dir / "downloaded_files.csv"
    schema_json = run_dir / "schema_summary.json"
    write_csv(manifest_csv, downloaded)
    schema_json.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    report_path = REPORTS / f"e75b_toptag_record80030_schema_smoke_{dt.datetime.now():%Y%m%d}.md"
    lines = [
        "# E75b TopTag Record 80030 Schema/Loader Smoke",
        "",
        f"- run_dir: `{run_dir}`",
        f"- generated_at: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- source_manifest: `{args.manifest}`",
        f"- data_dir: `{args.data_dir}`",
        f"- downloaded_files_csv: `{manifest_csv}`",
        f"- schema_json: `{schema_json}`",
        "- run note: schema and loader check for downloaded HDF5 files.",
        "",
        "## Downloaded Files",
        "",
        "| index | filename | size MB | checksum expected |",
        "|---|---|---:|---|",
    ]
    for row in downloaded:
        lines.append(
            f"| {row['index_description']} | `{row['filename']}` | "
            f"{row['size_bytes'] / 1_000_000:.2f} | `{row['checksum_expected']}` |"
        )
    lines.extend(["", "## Schema Summary", ""])
    for item in schema:
        lines.append(f"### `{item['filename']}`")
        lines.append("")
        lines.append(f"- compressed_size_MB: {item['compressed_size_bytes'] / 1_000_000:.2f}")
        lines.append(f"- decompressed_h5_size_MB: {item['decompressed_h5_size_bytes'] / 1_000_000:.2f}")
        lines.append(f"- top_keys: `{', '.join(item['top_keys'])}`")
        lines.append("")
        lines.append("| dataset | shape | dtype |")
        lines.append("|---|---|---|")
        for dataset in item["datasets"][:20]:
            lines.append(f"| `{dataset['name']}` | `{dataset['shape']}` | `{dataset['dtype']}` |")
        if len(item["datasets"]) > 20:
            lines.append(f"| ... | {len(item['datasets']) - 20} additional datasets omitted | ... |")
        lines.append("")
    lines.extend(
        [
            "## Next Step",
            "",
            "- Build the minimal tensor loader only after confirming which datasets contain constituents, labels, weights, and event/jet identifiers.",
            "- Keep the next E75c run small: nominal plus one systematic variation first.",
            "- Do not download full record 80030 until a small branch-protocol run shows a clear reason to scale.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    metrics = {
        "experiment": "E75b TopTag record 80030 schema smoke",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "source_manifest": str(args.manifest),
        "data_dir": str(args.data_dir),
        "downloaded_files": downloaded,
        "schema_json": str(schema_json),
        "report": str(report_path),
        "status": "done",
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "status.txt").write_text(f"status: done\nreport: {report_path}\n", encoding="utf-8")
    print(f"E75b TopTag schema smoke done: {run_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
