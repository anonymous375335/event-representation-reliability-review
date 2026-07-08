#!/usr/bin/env python3
"""E75c TopTag record 80030 small branch-protocol smoke."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import e68c_cms_h4l_split_branch_disentanglement as e68c
from e66_cms_h4l_readout_smoke import weighted_auc


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
DATA_RAW_ROOT = Path(
    os.environ.get(
        "YEAR1_DATA_RAW",
        str(ROOT / "data_raw"),
    )
)
DATA_PROCESSED_ROOT = Path(
    os.environ.get(
        "YEAR1_DATA_PROCESSED",
        str(ROOT / "data_processed"),
    )
)

DOMAIN_FILES = {
    "nominal": "test_nominal_000.h5.gz",
    "esup": "esup_000.h5.gz",
    "esdown": "esdown_000.h5.gz",
}
DOMAIN_NAMES = list(DOMAIN_FILES)

HIGH_LEVEL_KEYS = [
    "fjet_pt",
    "fjet_m",
    "fjet_eta",
    "fjet_phi",
    "fjet_C2",
    "fjet_D2",
    "fjet_ECF1",
    "fjet_ECF2",
    "fjet_ECF3",
    "fjet_L2",
    "fjet_L3",
    "fjet_Qw",
    "fjet_Split12",
    "fjet_Split23",
    "fjet_Tau1_wta",
    "fjet_Tau2_wta",
    "fjet_Tau3_wta",
    "fjet_Tau4_wta",
    "fjet_ThrustMaj",
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


def ensure_h5_cache(gz_path: Path, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    h5_path = cache_dir / gz_path.name.removesuffix(".gz")
    if h5_path.exists() and h5_path.stat().st_mtime >= gz_path.stat().st_mtime:
        return h5_path
    tmp_path = h5_path.with_suffix(h5_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with gzip.open(gz_path, "rb") as src, tmp_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    tmp_path.replace(h5_path)
    return h5_path


def wrap_delta_phi(phi: np.ndarray) -> np.ndarray:
    return (phi + np.pi) % (2 * np.pi) - np.pi


def select_stratified_indices(labels: np.ndarray, max_events: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if max_events <= 0 or max_events >= len(labels):
        indices = np.arange(len(labels))
        rng.shuffle(indices)
        return indices
    parts = []
    for label in sorted(np.unique(labels).tolist()):
        label_idx = np.flatnonzero(labels == label)
        take = max(1, int(round(max_events * len(label_idx) / len(labels))))
        parts.append(rng.choice(label_idx, size=min(take, len(label_idx)), replace=False))
    indices = np.concatenate(parts)
    rng.shuffle(indices)
    return indices


def build_features(handle: h5py.File, indices: np.ndarray, max_constituents: int) -> np.ndarray:
    jet_eta = handle["fjet_eta"][indices].astype(np.float32)
    jet_phi = handle["fjet_phi"][indices].astype(np.float32)
    clus_pt = handle["fjet_clus_pt"][indices, :max_constituents].astype(np.float32)
    clus_eta = handle["fjet_clus_eta"][indices, :max_constituents].astype(np.float32)
    clus_phi = handle["fjet_clus_phi"][indices, :max_constituents].astype(np.float32)
    clus_e = handle["fjet_clus_E"][indices, :max_constituents].astype(np.float32)
    mask = (clus_pt > 0).astype(np.float32)
    constituent = np.stack(
        [
            np.log1p(np.maximum(clus_pt, 0.0) / 1000.0),
            clus_eta - jet_eta[:, None],
            wrap_delta_phi(clus_phi - jet_phi[:, None]),
            np.log1p(np.maximum(clus_e, 0.0) / 1000.0),
            mask,
        ],
        axis=-1,
    ).reshape(len(indices), -1)

    high_parts = []
    for key in HIGH_LEVEL_KEYS:
        values = handle[key][indices].astype(np.float32)
        if key in {"fjet_pt", "fjet_m", "fjet_ECF1", "fjet_ECF2", "fjet_ECF3"}:
            values = np.log1p(np.maximum(values, 0.0) / 1000.0)
        high_parts.append(values[:, None])
    high = np.concatenate(high_parts, axis=1)
    features = np.concatenate([constituent, high], axis=1)
    return np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def load_domain(
    data_dir: Path,
    cache_dir: Path,
    domain: str,
    domain_id: int,
    max_events_per_domain: int,
    max_constituents: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    gz_path = data_dir / DOMAIN_FILES[domain]
    if not gz_path.exists():
        raise FileNotFoundError(gz_path)
    h5_path = ensure_h5_cache(gz_path, cache_dir)
    with h5py.File(h5_path, "r") as handle:
        labels_all = handle["labels"][:].astype(np.int64)
        indices = select_stratified_indices(labels_all, max_events_per_domain, seed + domain_id * 101)
        indices = np.sort(indices)
        features = build_features(handle, indices, max_constituents)
        labels = labels_all[indices].astype(np.int64)
        domains = np.full(len(indices), domain_id, dtype=np.int64)
        weights = np.ones(len(indices), dtype=np.float32)
        summary = {
            "domain": domain,
            "filename": DOMAIN_FILES[domain],
            "source_path": str(gz_path),
            "cache_h5": str(h5_path),
            "events": int(len(indices)),
            "label_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        }
    return features, labels, domains, weights, summary


def joint_stratified_split(labels: np.ndarray, domains: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    for domain in sorted(np.unique(domains).tolist()):
        for label in sorted(np.unique(labels).tolist()):
            indices = np.flatnonzero((domains == domain) & (labels == label))
            rng.shuffle(indices)
            val_count = max(1, int(round(len(indices) * val_ratio)))
            val_parts.append(indices[:val_count])
            train_parts.append(indices[val_count:])
    train = np.concatenate(train_parts)
    val = np.concatenate(val_parts)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def standardize(train_x: np.ndarray, val_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, (val_x - mean) / std, mean.squeeze(0), std.squeeze(0)


def background_rejection_at_signal_eff(labels: np.ndarray, scores: np.ndarray, signal_eff: float) -> float | None:
    signal_scores = scores[labels == 1]
    background_scores = scores[labels == 0]
    if len(signal_scores) == 0 or len(background_scores) == 0:
        return None
    threshold = float(np.quantile(signal_scores, 1.0 - signal_eff))
    background_eff = float((background_scores >= threshold).mean())
    if background_eff <= 0:
        return None
    return 1.0 / background_eff


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_RAW_ROOT / "toptag_record80030_e75b")
    parser.add_argument("--cache-dir", type=Path, default=DATA_PROCESSED_ROOT / "toptag_record80030_e75c" / "h5_cache")
    parser.add_argument("--max-events-per-domain", type=int, default=20000)
    parser.add_argument("--max-constituents", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--probe-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--adv-lambda", type=float, default=0.5)
    parser.add_argument("--orth-lambda", type=float, default=0.25)
    parser.add_argument("--nuisance-latent-dim", type=int, default=16)
    args = parser.parse_args()

    e68c.set_seed(args.seed)
    e68c.DOMAIN_NAMES = DOMAIN_NAMES.copy()
    REPORTS.mkdir(parents=True, exist_ok=True)
    run_dir = create_run_dir("e75c-toptag-branch-protocol-smoke")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_parts = []
    label_parts = []
    domain_parts = []
    weight_parts = []
    summaries = []
    for domain_id, domain in enumerate(DOMAIN_NAMES):
        features, labels, domains, weights, summary = load_domain(
            args.data_dir,
            args.cache_dir,
            domain,
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
    train_idx, val_idx = joint_stratified_split(labels, domains, args.val_ratio, args.seed)
    train_x_np, val_x_np, mean, std = standardize(features[train_idx], features[val_idx])
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
    pos = max(float((train_y_np == 1).sum()), 1.0)
    neg = max(float((train_y_np == 0).sum()), 1.0)
    pos_weight = neg / pos
    train_loader = DataLoader(TensorDataset(train_x, train_y, train_domains), batch_size=args.batch_size, shuffle=True)
    input_domain_probe_acc = e68c.train_domain_probe(
        train_x_np,
        train_domain_np,
        val_x_np,
        val_domain_np,
        args.probe_epochs,
        device,
    )

    rows = []
    history_rows = []
    for mode in ["shared_baseline", "split_no_orth", "split_orth", "split_orth_adv"]:
        print(f"[fit-start] {mode} {dt.datetime.now().isoformat(timespec='seconds')}", flush=True)
        model, history = e68c.train_one(
            mode=mode,
            train_loader=train_loader,
            val_x=val_x,
            val_y=val_y,
            val_domains=val_domains,
            input_dim=train_x.shape[1],
            pos_weight=pos_weight,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            adv_lambda=args.adv_lambda,
            orth_lambda=args.orth_lambda,
            device=device,
            nuisance_latent_dim=args.nuisance_latent_dim,
        )
        print(f"[fit-done] {mode} {dt.datetime.now().isoformat(timespec='seconds')}", flush=True)
        history_rows.extend(history)
        val_scores, val_z_phys, val_z_nuis, val_nuis_logits = e68c.embed_and_score(model, val_x, device)
        _, train_z_phys, train_z_nuis, _ = e68c.embed_and_score(model, train_x, device)
        row = {
            "mode": mode,
            "physics_auc": weighted_auc(val_y_np, val_scores, val_weights_np),
            "background_rejection_at_30pct_signal_eff": background_rejection_at_signal_eff(val_y_np, val_scores, 0.30),
            "score_domain_drift_max": e68c.domain_score_drift(val_scores, val_domain_np),
            "nuisance_head_acc": float((val_nuis_logits.argmax(axis=1) == val_domain_np).mean()),
            "z_phys_domain_probe_acc": e68c.train_domain_probe(
                train_z_phys, train_domain_np, val_z_phys, val_domain_np, args.probe_epochs, device
            ),
            "z_nuis_domain_probe_acc": e68c.train_domain_probe(
                train_z_nuis, train_domain_np, val_z_nuis, val_domain_np, args.probe_epochs, device
            ),
            "z_nuis_physics_probe_auc": e68c.train_physics_probe_auc(
                train_z_nuis,
                train_y_np,
                val_z_nuis,
                val_y_np,
                val_weights_np,
                args.probe_epochs,
                device,
            ),
        }
        rows.append(row)

    metrics_csv = run_dir / "toptag_branch_protocol_metrics.csv"
    history_csv = run_dir / "training_history.csv"
    write_csv(metrics_csv, rows)
    write_csv(history_csv, history_rows)
    config = {
        "data_dir": str(args.data_dir),
        "cache_dir": str(args.cache_dir),
        "domains": DOMAIN_NAMES,
        "domain_files": DOMAIN_FILES,
        "max_events_per_domain": args.max_events_per_domain,
        "max_constituents": args.max_constituents,
        "feature_dim": int(features.shape[1]),
        "seed": args.seed,
        "epochs": args.epochs,
        "probe_epochs": args.probe_epochs,
        "adv_lambda": args.adv_lambda,
        "orth_lambda": args.orth_lambda,
        "nuisance_latent_dim": args.nuisance_latent_dim,
        "input_domain_probe_acc": input_domain_probe_acc,
        "domain_summaries": summaries,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    np.savez_compressed(run_dir / "standardization_stats.npz", mean=mean, std=std)

    report_path = REPORTS / f"e75c_toptag_branch_protocol_smoke_{dt.datetime.now():%Y%m%d}_seed{args.seed}.md"
    lines = [
        "# E75c TopTag Branch-Protocol Smoke",
        "",
        f"- run_dir: `{run_dir}`",
        f"- generated_at: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- data_dir: `{args.data_dir}`",
        f"- cache_dir: `{args.cache_dir}`",
        f"- device: `{device}`",
        f"- domains: `{', '.join(DOMAIN_NAMES)}`",
        f"- random_domain_baseline: {1 / len(DOMAIN_NAMES):.4f}",
        f"- events_per_domain_cap: {args.max_events_per_domain}",
        f"- total_events_loaded: {len(labels)}",
        f"- train_events: {len(train_idx)}",
        f"- val_events: {len(val_idx)}",
        f"- feature_dim: {features.shape[1]}",
        f"- max_constituents: {args.max_constituents}",
        f"- epochs: {args.epochs}",
        f"- probe_epochs: {args.probe_epochs}",
        f"- adv_lambda: {args.adv_lambda}",
        f"- orth_lambda: {args.orth_lambda}",
        f"- nuisance_latent_dim: {args.nuisance_latent_dim}",
        f"- input_domain_probe_acc: {input_domain_probe_acc:.4f}",
        "- run note: TopTag record-80030 transfer protocol with file-level `esup/esdown` labels and uniform variation weights.",
        "",
        "## Domain Samples",
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
            "## Metrics",
            "",
            "| mode | physics AUC | bkg rejection @30% sig eff | score domain drift | nuisance head acc | z_phys domain probe | z_nuis domain probe | z_nuis physics probe AUC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['mode']} | {fmt(row['physics_auc'])} | "
            f"{fmt(row['background_rejection_at_30pct_signal_eff'])} | "
            f"{fmt(row['score_domain_drift_max'])} | {fmt(row['nuisance_head_acc'])} | "
            f"{fmt(row['z_phys_domain_probe_acc'])} | {fmt(row['z_nuis_domain_probe_acc'])} | "
            f"{fmt(row['z_nuis_physics_probe_auc'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- A favorable result keeps physics AUC close to the shared baseline while `z_phys -> domain` decreases or remains controlled.",
            "- `z_nuis -> domain` above the 0.333 random baseline indicates systematic information in the nuisance branch.",
            "- The three-file protocol supports feasibility and direction for the larger TopTag sequence.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "experiment": "E75c TopTag branch-protocol smoke",
                "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
                "run_dir": str(run_dir),
                "report": str(report_path),
                "metrics_csv": str(metrics_csv),
                "history_csv": str(history_csv),
                "config": config,
                "metrics": rows,
                "status": "done",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "status.txt").write_text(f"status: done\nreport: {report_path}\n", encoding="utf-8")
    print(f"E75c TopTag branch-protocol smoke done: {run_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
