#!/usr/bin/env python3
"""Observed-data H4l m4l/template sanity check on local CMS 2012 reduced NanoAOD."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import awkward as ak
import numpy as np

import e64_cms_h4l_baseline_selection_smoke as smoke


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"

DATA_SAMPLES = [
    "Run2012B_DoubleMuParked",
    "Run2012C_DoubleMuParked",
    "Run2012B_DoubleElectron",
    "Run2012C_DoubleElectron",
]

FINAL_STATE_NAMES = {
    0: "FourMuons",
    1: "FourElectrons",
    2: "TwoMuonsTwoElectrons",
}

M4L_BINS = np.linspace(70.0, 180.0, 37)


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


def tree_entries(sample: str) -> int:
    source, _ = smoke.source_for_sample(sample)
    with smoke.uproot.open(source, timeout=300) as root_file:
        return int(root_file["Events"].num_entries)


def select_chunk(sample: str, final_state: str, entry_start: int, entry_stop: int) -> list[dict]:
    arrays, stop, source_mode, _ = smoke.load_sample_arrays(sample, [final_state], entry_start, entry_stop)
    prefiltered = smoke.prefilter_arrays(arrays, final_state)
    selected = []
    for row in ak.to_list(prefiltered):
        event = smoke.select_event(type("Row", (), row), final_state)
        if event is not None:
            event["sample"] = sample
            event["final_state"] = final_state
            event["source_mode"] = source_mode
            selected.append(event)
    return selected


def scan_data_samples(chunk_size: int) -> tuple[list[dict], list[dict]]:
    events = []
    scan_rows = []
    for sample in DATA_SAMPLES:
        entries = tree_entries(sample)
        for final_state in smoke.SAMPLES[sample]:
            selected_count = 0
            for start in range(0, entries, chunk_size):
                stop = min(start + chunk_size, entries)
                chunk_events = select_chunk(sample, final_state, start, stop)
                events.extend(chunk_events)
                selected_count += len(chunk_events)
                print(
                    f"[data-scan] {sample}:{final_state} {stop}/{entries} selected_so_far={selected_count}",
                    flush=True,
                )
            scan_rows.append(
                {
                    "sample": sample,
                    "final_state": final_state,
                    "entries": entries,
                    "selected": selected_count,
                }
            )
    return events, scan_rows


def mc_expected_from_tensor(path: Path) -> list[dict]:
    with np.load(path, allow_pickle=False) as payload:
        labels = payload["labels"].astype(np.int64)
        subprocess_ids = payload["subprocess_ids"].astype(np.int64)
        conditions = payload["conditions"].astype(np.float32)
    masses = conditions[:, 0]
    weights = conditions[:, 4]
    rows = []
    for state_id, state_name in FINAL_STATE_NAMES.items():
        state_mask = subprocess_ids == state_id
        for label, component in [(1, "signal"), (0, "zz_background")]:
            mask = state_mask & (labels == label)
            hist, edges = np.histogram(masses[mask], bins=M4L_BINS, weights=weights[mask])
            raw, _ = np.histogram(masses[mask], bins=M4L_BINS)
            for idx, value in enumerate(hist):
                rows.append(
                    {
                        "sample": component,
                        "final_state": state_name,
                        "bin_low": float(edges[idx]),
                        "bin_high": float(edges[idx + 1]),
                        "weighted_count": float(value),
                        "raw_count": int(raw[idx]),
                    }
                )
    return rows


def observed_hist_rows(events: list[dict]) -> list[dict]:
    rows = []
    for sample in DATA_SAMPLES:
        for final_state in smoke.SAMPLES[sample]:
            values = np.asarray(
                [event["Higgs_mass"] for event in events if event["sample"] == sample and event["final_state"] == final_state],
                dtype=np.float32,
            )
            hist, edges = np.histogram(values, bins=M4L_BINS)
            for idx, value in enumerate(hist):
                rows.append(
                    {
                        "sample": sample,
                        "final_state": final_state,
                        "bin_low": float(edges[idx]),
                        "bin_high": float(edges[idx + 1]),
                        "observed_count": int(value),
                    }
                )
    return rows


def summarize_regions(events: list[dict], mc_rows: list[dict]) -> list[dict]:
    regions = {
        "low_sideband_70_115": (70.0, 115.0),
        "higgs_window_115_135": (115.0, 135.0),
        "high_sideband_135_180": (135.0, 180.0),
    }
    rows = []
    for sample in DATA_SAMPLES:
        for final_state in smoke.SAMPLES[sample]:
            masses = np.asarray(
                [event["Higgs_mass"] for event in events if event["sample"] == sample and event["final_state"] == final_state],
                dtype=np.float32,
            )
            for region, (low, high) in regions.items():
                observed = int(((masses >= low) & (masses < high)).sum())
                rows.append(
                    {
                        "source": "observed_data_not_deduplicated",
                        "sample": sample,
                        "final_state": final_state,
                        "region": region,
                        "count": observed,
                    }
                )
    for final_state in FINAL_STATE_NAMES.values():
        for component in ["signal", "zz_background"]:
            subset = [row for row in mc_rows if row["sample"] == component and row["final_state"] == final_state]
            for region, (low, high) in regions.items():
                expected = sum(
                    row["weighted_count"]
                    for row in subset
                    if row["bin_low"] >= low and row["bin_high"] <= high
                )
                rows.append(
                    {
                        "source": "mc_expected",
                        "sample": component,
                        "final_state": final_state,
                        "region": region,
                        "count": float(expected),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(run_dir: Path, scan_rows: list[dict], region_rows: list[dict], hist_csv: Path, mc_csv: Path, region_csv: Path) -> Path:
    report_path = REPORTS / f"e69b_cms_h4l_observed_data_sanity_{dt.datetime.now():%Y%m%d}.md"
    lines = [
        "# E69b CMS H4l Observed-Data Sanity",
        "",
        f"- run_dir: `{run_dir}`",
        f"- generated_at: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- observed_histograms_csv: `{hist_csv}`",
        f"- mc_expected_histograms_csv: `{mc_csv}`",
        f"- region_summary_csv: `{region_csv}`",
        "- dataset note: observed collision data use selection and template summaries without truth labels.",
        "- trigger streams are reported by dataset/final_state and are not de-duplicated across DoubleElectron/DoubleMu streams.",
        "",
        "## Selection Counts",
        "",
        "| sample | final state | entries | selected |",
        "|---|---|---:|---:|",
    ]
    for row in scan_rows:
        lines.append(f"| {row['sample']} | {row['final_state']} | {row['entries']} | {row['selected']} |")
    lines.extend(
        [
            "",
            "## Region Counts",
            "",
            "| source | sample/component | final state | region | count |",
            "|---|---|---|---|---:|",
        ]
    )
    for row in region_rows:
        count = row["count"]
        count_text = f"{count:.4g}" if isinstance(count, float) else str(count)
        lines.append(f"| {row['source']} | {row['sample']} | {row['final_state']} | {row['region']} | {count_text} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Use this artifact to check that observed-data m4l shapes and selected-event counts are not pathological before adding score/template overlays.",
            "- This artifact deliberately avoids signal/background labels for real data.",
            "- The next optional sanity layer is to train a fixed MC score model and overlay observed score distributions by trigger stream.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-npz", default=str(ROOT / "data_processed" / "cms_h4l_e65" / "cms_h4l_mc_candidates_e65.npz"))
    parser.add_argument("--chunk-size", type=int, default=500_000)
    args = parser.parse_args()

    run_dir = create_run_dir("e69b-cms-h4l-observed-data-sanity")
    REPORTS.mkdir(parents=True, exist_ok=True)

    events, scan_rows = scan_data_samples(args.chunk_size)
    observed_rows = observed_hist_rows(events)
    mc_rows = mc_expected_from_tensor(Path(args.tensor_npz))
    region_rows = summarize_regions(events, mc_rows)

    observed_csv = run_dir / "observed_m4l_histograms.csv"
    mc_csv = run_dir / "mc_expected_m4l_histograms.csv"
    region_csv = run_dir / "region_summary.csv"
    scan_csv = run_dir / "selection_counts.csv"
    write_csv(observed_csv, observed_rows)
    write_csv(mc_csv, mc_rows)
    write_csv(region_csv, region_rows)
    write_csv(scan_csv, scan_rows)

    report_path = write_report(run_dir, scan_rows, region_rows, observed_csv, mc_csv, region_csv)
    manifest = {
        "experiment": "E69b CMS H4l observed-data sanity",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "tensor_npz": args.tensor_npz,
        "chunk_size": args.chunk_size,
        "selected_events": len(events),
        "selection_counts_csv": str(scan_csv),
        "observed_histograms_csv": str(observed_csv),
        "mc_expected_histograms_csv": str(mc_csv),
        "region_summary_csv": str(region_csv),
        "report": str(report_path),
        "status": "done",
    }
    (run_dir / "metrics.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "status.txt").write_text(f"status: done\nreport: {report_path}\n", encoding="utf-8")
    print(f"E69b observed-data sanity done: {run_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
