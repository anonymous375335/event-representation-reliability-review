#!/usr/bin/env python3
"""E75e TopTag scale-up across more record-80030 systematic families."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import e68c_cms_h4l_split_branch_disentanglement as e68c
import e75c_toptag_branch_protocol_smoke as e75c
import e75d_toptag_domain_routing_grid as e75d
from e66_cms_h4l_readout_smoke import weighted_auc


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
DEFAULT_RECORD80030_MANIFEST = ROOT / "benchmarks" / "toptag_pyhf" / "manifests" / "record80030_files.csv"
DEFAULT_INDICES = [
    "test_nominal_file_index.json",
    "esup_file_index.json",
    "esdown_file_index.json",
    "angular_file_index.json",
    "cluster_file_index.json",
    "cer_file_index.json",
    "cpos_file_index.json",
    "bias_file_index.json",
]


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


def domain_from_index(index_description: str) -> str:
    name = index_description.replace("_file_index.json", "").replace(".json", "")
    if name == "test_nominal":
        return "nominal"
    return name


def select_first_files(rows: list[dict], indices: list[str]) -> list[dict]:
    selected = []
    for index in indices:
        matches = [row for row in rows if row["index_description"] == index]
        if not matches:
            raise ValueError(f"no rows found for index {index}")
        row = dict(matches[0])
        row["domain"] = domain_from_index(index)
        selected.append(row)
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
    control_file = tmp.with_name(tmp.name + ".aria2")
    if control_file.exists():
        control_file.unlink()
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
        root_uri_to_https(row["uri"]),
    ]
    subprocess.run(cmd, check=True)
    actual_size = tmp.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(f"size mismatch for {row['filename']}: got {actual_size}, expected {expected_size}")
    if control_file.exists():
        control_file.unlink()
    tmp.replace(path)
    return path


def load_domain(
    data_dir: Path,
    cache_dir: Path,
    domain: str,
    filename: str,
    domain_id: int,
    max_events_per_domain: int,
    max_constituents: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    gz_path = data_dir / filename
    h5_path = e75c.ensure_h5_cache(gz_path, cache_dir)
    import h5py

    with h5py.File(h5_path, "r") as handle:
        labels_all = handle["labels"][:].astype(np.int64)
        indices = e75c.select_stratified_indices(labels_all, max_events_per_domain, seed + domain_id * 101)
        indices = np.sort(indices)
        features = e75c.build_features(handle, indices, max_constituents)
        labels = labels_all[indices].astype(np.int64)
        domains = np.full(len(indices), domain_id, dtype=np.int64)
        weights = np.ones(len(indices), dtype=np.float32)
        summary = {
            "domain": domain,
            "filename": filename,
            "source_path": str(gz_path),
            "cache_h5": str(h5_path),
            "events": int(len(indices)),
            "label_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        }
    return features, labels, domains, weights, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def evaluate_model(
    row_prefix: dict,
    model: e68c.SplitBranchNet,
    train_x: torch.Tensor,
    val_x: torch.Tensor,
    train_z_source: np.ndarray | None,
    train_y_np: np.ndarray,
    val_y_np: np.ndarray,
    train_domain_np: np.ndarray,
    val_domain_np: np.ndarray,
    val_weights_np: np.ndarray,
    probe_epochs: int,
    device: torch.device,
) -> dict:
    val_scores, val_z_phys, val_z_nuis, val_nuis_logits = e68c.embed_and_score(model, val_x, device)
    _, train_z_phys, train_z_nuis, _ = e68c.embed_and_score(model, train_x, device)
    return {
        **row_prefix,
        "physics_auc": weighted_auc(val_y_np, val_scores, val_weights_np),
        "background_rejection_at_30pct_signal_eff": e75c.background_rejection_at_signal_eff(
            val_y_np, val_scores, 0.30
        ),
        "score_domain_drift_max": e68c.domain_score_drift(val_scores, val_domain_np),
        "nuisance_head_acc": float((val_nuis_logits.argmax(axis=1) == val_domain_np).mean()),
        "z_phys_domain_probe_acc": e68c.train_domain_probe(
            train_z_phys, train_domain_np, val_z_phys, val_domain_np, probe_epochs, device
        ),
        "z_nuis_domain_probe_acc": e68c.train_domain_probe(
            train_z_nuis, train_domain_np, val_z_nuis, val_domain_np, probe_epochs, device
        ),
        "z_nuis_physics_probe_auc": e68c.train_physics_probe_auc(
            train_z_nuis, train_y_np, val_z_nuis, val_y_np, val_weights_np, probe_epochs, device
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_RECORD80030_MANIFEST,
    )
    parser.add_argument("--indices", nargs="+", default=DEFAULT_INDICES)
    parser.add_argument("--data-dir", type=Path, default=e75c.DATA_RAW_ROOT / "toptag_record80030_e75b")
    parser.add_argument("--cache-dir", type=Path, default=e75c.DATA_PROCESSED_ROOT / "toptag_record80030_e75c" / "h5_cache")
    parser.add_argument("--max-events-per-domain", type=int, default=100000)
    parser.add_argument("--max-constituents", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--probe-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--orth-lambda", type=float, default=0.25)
    args = parser.parse_args()

    e68c.set_seed(args.seed)
    REPORTS.mkdir(parents=True, exist_ok=True)
    run_dir = create_run_dir("e75e-toptag-systematic-family-scaleup")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selected = select_first_files(read_rows(args.manifest), args.indices)
    downloaded_rows = []
    for row in selected:
        path = download_file(row, args.data_dir)
        downloaded_rows.append(
            {
                "domain": row["domain"],
                "index_description": row["index_description"],
                "filename": row["filename"],
                "size_bytes": path.stat().st_size,
                "checksum_expected": row["checksum"],
                "local_path": str(path),
            }
        )

    domain_names = [row["domain"] for row in selected]
    e68c.DOMAIN_NAMES = domain_names.copy()
    feature_parts = []
    label_parts = []
    domain_parts = []
    weight_parts = []
    summaries = []
    for domain_id, row in enumerate(selected):
        features, labels, domains, weights, summary = load_domain(
            args.data_dir,
            args.cache_dir,
            row["domain"],
            row["filename"],
            domain_id,
            args.max_events_per_domain,
            args.max_constituents,
            args.seed,
        )
        feature_parts.append(features)
        label_parts.append(labels)
        domain_parts.append(domains)
        weight_parts.append(weights)
        summaries.append(summary)

    features = np.concatenate(feature_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    domains = np.concatenate(domain_parts, axis=0)
    weights = np.concatenate(weight_parts, axis=0)
    train_idx, val_idx = e75c.joint_stratified_split(labels, domains, args.val_ratio, args.seed)
    train_x_np, val_x_np, _, _ = e75c.standardize(features[train_idx], features[val_idx])
    train_y_np = labels[train_idx]
    val_y_np = labels[val_idx]
    train_domain_np = domains[train_idx]
    val_domain_np = domains[val_idx]
    val_weights_np = weights[val_idx]
    train_x = torch.tensor(train_x_np, dtype=torch.float32)
    train_y = torch.tensor(train_y_np, dtype=torch.float32)
    train_domains = torch.tensor(train_domain_np, dtype=torch.long)
    val_x = torch.tensor(val_x_np, dtype=torch.float32)
    val_y = torch.tensor(val_y_np, dtype=torch.float32)
    val_domains = torch.tensor(val_domain_np, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(train_x, train_y, train_domains), batch_size=args.batch_size, shuffle=True)
    pos = max(float((train_y_np == 1).sum()), 1.0)
    neg = max(float((train_y_np == 0).sum()), 1.0)
    pos_weight = neg / pos
    input_domain_probe_acc = e68c.train_domain_probe(
        train_x_np, train_domain_np, val_x_np, val_domain_np, args.probe_epochs, device
    )

    rows = []
    history_rows = []
    print(f"[fit-start] shared_baseline {dt.datetime.now().isoformat(timespec='seconds')}", flush=True)
    baseline, baseline_history = e68c.train_one(
        mode="shared_baseline",
        train_loader=train_loader,
        val_x=val_x,
        val_y=val_y,
        val_domains=val_domains,
        input_dim=train_x.shape[1],
        pos_weight=pos_weight,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        adv_lambda=0.0,
        orth_lambda=args.orth_lambda,
        device=device,
        nuisance_latent_dim=64,
    )
    print(f"[fit-done] shared_baseline {dt.datetime.now().isoformat(timespec='seconds')}", flush=True)
    history_rows.extend([{**row, "candidate": "shared_baseline"} for row in baseline_history])
    rows.append(
        evaluate_model(
            {"candidate": "shared_baseline", "nuisance_weight": 0.0, "nuisance_latent_dim": 64},
            baseline,
            train_x,
            val_x,
            None,
            train_y_np,
            val_y_np,
            train_domain_np,
            val_domain_np,
            val_weights_np,
            args.probe_epochs,
            device,
        )
    )

    candidates = [
        {"candidate": "split_orth_domain_best", "nuisance_weight": 5.0, "nuisance_latent_dim": 32},
        {"candidate": "split_orth_physics_best", "nuisance_weight": 1.0, "nuisance_latent_dim": 64},
    ]
    for candidate in candidates:
        print(
            f"[fit-start] {candidate['candidate']} {dt.datetime.now().isoformat(timespec='seconds')}",
            flush=True,
        )
        model, history = e75d.train_split_orth(
            train_loader=train_loader,
            val_x=val_x,
            val_y=val_y,
            val_domains=val_domains,
            input_dim=train_x.shape[1],
            pos_weight=pos_weight,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            nuisance_weight=candidate["nuisance_weight"],
            orth_lambda=args.orth_lambda,
            nuisance_latent_dim=candidate["nuisance_latent_dim"],
            device=device,
        )
        print(
            f"[fit-done] {candidate['candidate']} {dt.datetime.now().isoformat(timespec='seconds')}",
            flush=True,
        )
        history_rows.extend([{**row, "candidate": candidate["candidate"]} for row in history])
        rows.append(
            evaluate_model(
                candidate,
                model,
                train_x,
                val_x,
                None,
                train_y_np,
                val_y_np,
                train_domain_np,
                val_domain_np,
                val_weights_np,
                args.probe_epochs,
                device,
            )
        )

    downloaded_csv = run_dir / "downloaded_files.csv"
    metrics_csv = run_dir / "systematic_family_scaleup_metrics.csv"
    history_csv = run_dir / "training_history.csv"
    write_csv(downloaded_csv, downloaded_rows)
    write_csv(metrics_csv, rows)
    write_csv(history_csv, history_rows)
    best = max(rows[1:], key=lambda row: (row["z_nuis_domain_probe_acc"], row["physics_auc"]))
    config = {
        "manifest": str(args.manifest),
        "indices": args.indices,
        "data_dir": str(args.data_dir),
        "cache_dir": str(args.cache_dir),
        "domains": domain_names,
        "max_events_per_domain": args.max_events_per_domain,
        "max_constituents": args.max_constituents,
        "feature_dim": int(features.shape[1]),
        "total_events_loaded": int(len(labels)),
        "seed": args.seed,
        "epochs": args.epochs,
        "probe_epochs": args.probe_epochs,
        "orth_lambda": args.orth_lambda,
        "input_domain_probe_acc": input_domain_probe_acc,
        "domain_summaries": summaries,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    domain_tag = f"d{len(domain_names)}"
    report_path = REPORTS / (
        f"e75e_toptag_systematic_family_scaleup_{domain_tag}_{dt.datetime.now():%Y%m%d}_seed{args.seed}.md"
    )
    lines = [
        "# E75e TopTag Systematic-Family Scale-Up",
        "",
        f"- run_dir: `{run_dir}`",
        f"- generated_at: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- data_dir: `{args.data_dir}`",
        f"- cache_dir: `{args.cache_dir}`",
        f"- device: `{device}`",
        f"- domains: `{', '.join(domain_names)}`",
        f"- random_domain_baseline: {1 / len(domain_names):.4f}",
        f"- input_domain_probe_acc: {input_domain_probe_acc:.4f}",
        f"- events_per_domain_cap: {args.max_events_per_domain}",
        f"- total_events_loaded: {len(labels)}",
        f"- feature_dim: {features.shape[1]}",
        f"- epochs: {args.epochs}",
        f"- probe_epochs: {args.probe_epochs}",
        "- run note: first-shard scale-up across systematic families.",
        "",
        "## Downloaded/Used Files",
        "",
        "| domain | file | events | label counts |",
        "|---|---|---:|---|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['domain']} | `{summary['filename']}` | {summary['events']} | `{summary['label_counts']}` |"
        )
    lines.extend(
        [
            "",
            "## Best Split Candidate",
            "",
            (
                f"- {best['candidate']}: physics AUC={format_float(best['physics_auc'])}, "
                f"z_phys domain={format_float(best['z_phys_domain_probe_acc'])}, "
                f"z_nuis domain={format_float(best['z_nuis_domain_probe_acc'])}, "
                f"z_nuis physics={format_float(best['z_nuis_physics_probe_auc'])}"
            ),
            "",
            "## Metrics",
            "",
            "| candidate | nuisance weight | nuisance dim | physics AUC | bkg rejection @30% sig eff | score domain drift | nuisance head acc | z_phys domain probe | z_nuis domain probe | z_nuis physics probe AUC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {row['nuisance_weight']:g} | {row['nuisance_latent_dim']} | "
            f"{format_float(row['physics_auc'])} | "
            f"{format_float(row['background_rejection_at_30pct_signal_eff'])} | "
            f"{format_float(row['score_domain_drift_max'])} | {format_float(row['nuisance_head_acc'])} | "
            f"{format_float(row['z_phys_domain_probe_acc'])} | {format_float(row['z_nuis_domain_probe_acc'])} | "
            f"{format_float(row['z_nuis_physics_probe_auc'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- Transfer support is read from physics AUC near baseline together with improved `z_nuis -> domain` or reduced `z_nuis -> physics`.",
            "- Domains use the first shard for the cross-family scale-up comparison.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "experiment": "E75e TopTag systematic-family scale-up",
                "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
                "run_dir": str(run_dir),
                "report": str(report_path),
                "downloaded_csv": str(downloaded_csv),
                "metrics_csv": str(metrics_csv),
                "history_csv": str(history_csv),
                "config": config,
                "best_split_candidate": best,
                "metrics": rows,
                "status": "done",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "status.txt").write_text(f"status: done\nreport: {report_path}\n", encoding="utf-8")
    print(f"E75e TopTag systematic-family scale-up done: {run_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
