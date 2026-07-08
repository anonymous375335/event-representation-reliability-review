#!/usr/bin/env python3
"""E75d TopTag domain-routing grid on record 80030 smoke data."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

import e68c_cms_h4l_split_branch_disentanglement as e68c
import e75c_toptag_branch_protocol_smoke as e75c
from e66_cms_h4l_readout_smoke import weighted_auc


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"


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


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_split_orth(
    train_loader: DataLoader,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    val_domains: torch.Tensor,
    input_dim: int,
    pos_weight: float,
    epochs: int,
    learning_rate: float,
    nuisance_weight: float,
    orth_lambda: float,
    nuisance_latent_dim: int,
    device: torch.device,
) -> tuple[e68c.SplitBranchNet, list[dict]]:
    model = e68c.SplitBranchNet(
        input_dim=input_dim,
        hidden_dim=128,
        latent_dim=64,
        num_domains=len(e68c.DOMAIN_NAMES),
        nuisance_latent_dim=nuisance_latent_dim,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)
    val_domains = val_domains.to(device)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"loss": 0.0, "physics": 0.0, "nuisance": 0.0, "orth": 0.0, "count": 0}
        for batch_x, batch_y, batch_domain in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_domain = batch_domain.to(device)
            out = model(batch_x, grl_lambda=0.0)
            physics_loss = F.binary_cross_entropy_with_logits(
                out["physics_logits"], batch_y, pos_weight=pos_weight_tensor
            )
            nuisance_loss = F.cross_entropy(out["nuisance_logits"], batch_domain)
            orth_loss = e68c.orthogonal_penalty(out["z_phys"], out["z_nuis"])
            loss = physics_loss + nuisance_weight * nuisance_loss + orth_lambda * orth_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            count = batch_y.shape[0]
            totals["loss"] += float(loss.detach().cpu()) * count
            totals["physics"] += float(physics_loss.detach().cpu()) * count
            totals["nuisance"] += float(nuisance_loss.detach().cpu()) * count
            totals["orth"] += float(orth_loss.detach().cpu()) * count
            totals["count"] += count
        model.eval()
        with torch.no_grad():
            val_out = model(val_x, grl_lambda=0.0)
            val_phys = F.binary_cross_entropy_with_logits(
                val_out["physics_logits"], val_y, pos_weight=pos_weight_tensor
            )
            val_nuis = F.cross_entropy(val_out["nuisance_logits"], val_domains)
            val_nuis_acc = float((val_out["nuisance_logits"].argmax(dim=1) == val_domains).float().mean().cpu())
            val_orth = float(e68c.orthogonal_penalty(val_out["z_phys"], val_out["z_nuis"]).cpu())
        denom = max(totals.pop("count"), 1)
        history.append(
            {
                "epoch": epoch,
                "nuisance_weight": nuisance_weight,
                "nuisance_latent_dim": nuisance_latent_dim,
                "train_loss": totals["loss"] / denom,
                "train_physics_loss": totals["physics"] / denom,
                "train_nuisance_loss": totals["nuisance"] / denom,
                "train_orthogonal_penalty": totals["orth"] / denom,
                "val_physics_loss": float(val_phys.cpu()),
                "val_nuisance_loss": float(val_nuis.cpu()),
                "val_nuisance_head_acc": val_nuis_acc,
                "val_orthogonal_penalty": val_orth,
            }
        )
    return model, history


def load_e75c_arrays(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    feature_parts = []
    label_parts = []
    domain_parts = []
    weight_parts = []
    summaries = []
    for domain_id, domain in enumerate(e75c.DOMAIN_NAMES):
        features, labels, domains, weights, summary = e75c.load_domain(
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
    return (
        np.concatenate(feature_parts, axis=0),
        np.concatenate(label_parts, axis=0),
        np.concatenate(domain_parts, axis=0),
        np.concatenate(weight_parts, axis=0),
        summaries,
    )


def format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=e75c.DATA_RAW_ROOT / "toptag_record80030_e75b")
    parser.add_argument("--cache-dir", type=Path, default=e75c.DATA_PROCESSED_ROOT / "toptag_record80030_e75c" / "h5_cache")
    parser.add_argument("--max-events-per-domain", type=int, default=20000)
    parser.add_argument("--max-constituents", type=int, default=80)
    parser.add_argument("--nuisance-weights", type=float, nargs="+", default=[1.0, 2.0, 5.0])
    parser.add_argument("--nuisance-latent-dims", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--probe-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--orth-lambda", type=float, default=0.25)
    args = parser.parse_args()

    e68c.set_seed(args.seed)
    e68c.DOMAIN_NAMES = e75c.DOMAIN_NAMES.copy()
    REPORTS.mkdir(parents=True, exist_ok=True)
    run_dir = create_run_dir("e75d-toptag-domain-routing-grid")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features, labels, domains, weights, summaries = load_e75c_arrays(args)
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
    pos = max(float((train_y_np == 1).sum()), 1.0)
    neg = max(float((train_y_np == 0).sum()), 1.0)
    pos_weight = neg / pos
    train_loader = DataLoader(TensorDataset(train_x, train_y, train_domains), batch_size=args.batch_size, shuffle=True)
    input_domain_probe_acc = e68c.train_domain_probe(
        train_x_np, train_domain_np, val_x_np, val_domain_np, args.probe_epochs, device
    )

    rows = []
    history_rows = []
    for nuisance_dim in args.nuisance_latent_dims:
        for nuisance_weight in args.nuisance_weights:
            print(
                f"[fit-start] nuisance_dim={nuisance_dim} nuisance_weight={nuisance_weight:g} "
                f"{dt.datetime.now().isoformat(timespec='seconds')}",
                flush=True,
            )
            model, history = train_split_orth(
                train_loader=train_loader,
                val_x=val_x,
                val_y=val_y,
                val_domains=val_domains,
                input_dim=train_x.shape[1],
                pos_weight=pos_weight,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                nuisance_weight=nuisance_weight,
                orth_lambda=args.orth_lambda,
                nuisance_latent_dim=nuisance_dim,
                device=device,
            )
            print(
                f"[fit-done] nuisance_dim={nuisance_dim} nuisance_weight={nuisance_weight:g} "
                f"{dt.datetime.now().isoformat(timespec='seconds')}",
                flush=True,
            )
            history_rows.extend(history)
            val_scores, val_z_phys, val_z_nuis, val_nuis_logits = e68c.embed_and_score(model, val_x, device)
            _, train_z_phys, train_z_nuis, _ = e68c.embed_and_score(model, train_x, device)
            rows.append(
                {
                    "mode": "split_orth",
                    "nuisance_weight": nuisance_weight,
                    "nuisance_latent_dim": nuisance_dim,
                    "physics_auc": weighted_auc(val_y_np, val_scores, val_weights_np),
                    "background_rejection_at_30pct_signal_eff": e75c.background_rejection_at_signal_eff(
                        val_y_np, val_scores, 0.30
                    ),
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
            )

    metrics_csv = run_dir / "domain_routing_grid_metrics.csv"
    history_csv = run_dir / "training_history.csv"
    write_csv(metrics_csv, rows)
    write_csv(history_csv, history_rows)
    best = max(rows, key=lambda row: (row["z_nuis_domain_probe_acc"], row["physics_auc"]))
    config = {
        "data_dir": str(args.data_dir),
        "cache_dir": str(args.cache_dir),
        "domains": e75c.DOMAIN_NAMES,
        "max_events_per_domain": args.max_events_per_domain,
        "max_constituents": args.max_constituents,
        "feature_dim": int(features.shape[1]),
        "seed": args.seed,
        "epochs": args.epochs,
        "probe_epochs": args.probe_epochs,
        "orth_lambda": args.orth_lambda,
        "nuisance_weights": args.nuisance_weights,
        "nuisance_latent_dims": args.nuisance_latent_dims,
        "input_domain_probe_acc": input_domain_probe_acc,
        "domain_summaries": summaries,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    report_path = REPORTS / f"e75d_toptag_domain_routing_grid_{dt.datetime.now():%Y%m%d}_seed{args.seed}.md"
    lines = [
        "# E75d TopTag Domain-Routing Grid",
        "",
        f"- run_dir: `{run_dir}`",
        f"- generated_at: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- data_dir: `{args.data_dir}`",
        f"- cache_dir: `{args.cache_dir}`",
        f"- device: `{device}`",
        f"- domains: `{', '.join(e75c.DOMAIN_NAMES)}`",
        f"- random_domain_baseline: {1 / len(e75c.DOMAIN_NAMES):.4f}",
        f"- input_domain_probe_acc: {input_domain_probe_acc:.4f}",
        f"- events_per_domain_cap: {args.max_events_per_domain}",
        f"- total_events_loaded: {len(labels)}",
        f"- feature_dim: {features.shape[1]}",
        f"- epochs: {args.epochs}",
        f"- probe_epochs: {args.probe_epochs}",
        f"- nuisance_weights: `{args.nuisance_weights}`",
        f"- nuisance_latent_dims: `{args.nuisance_latent_dims}`",
        "- run note: TopTag domain-routing grid with uniform weights.",
        "",
        "## Best By `z_nuis -> domain`",
        "",
        (
            f"- nuisance_weight={best['nuisance_weight']:g}, nuisance_latent_dim={best['nuisance_latent_dim']}: "
            f"physics AUC={format_float(best['physics_auc'])}, "
            f"z_phys domain={format_float(best['z_phys_domain_probe_acc'])}, "
            f"z_nuis domain={format_float(best['z_nuis_domain_probe_acc'])}, "
            f"z_nuis physics={format_float(best['z_nuis_physics_probe_auc'])}"
        ),
        "",
        "## Metrics",
        "",
        "| nuisance weight | nuisance dim | physics AUC | bkg rejection @30% sig eff | score domain drift | nuisance head acc | z_phys domain probe | z_nuis domain probe | z_nuis physics probe AUC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['nuisance_weight']:g} | {row['nuisance_latent_dim']} | "
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
            "- A useful grid point should keep physics AUC near or above the E75c shared baseline (`~0.8846`).",
            "- It should move `z_nuis -> domain` materially above E75c (`~0.353`) and ideally toward the raw-input probe (`0.4354`).",
            "- If `z_nuis -> domain` stays near random, prioritize the E76 representation update.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "experiment": "E75d TopTag domain-routing grid",
                "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
                "run_dir": str(run_dir),
                "report": str(report_path),
                "metrics_csv": str(metrics_csv),
                "history_csv": str(history_csv),
                "config": config,
                "best_by_z_nuis_domain": best,
                "metrics": rows,
                "status": "done",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "status.txt").write_text(f"status: done\nreport: {report_path}\n", encoding="utf-8")
    print(f"E75d TopTag domain-routing grid done: {run_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
