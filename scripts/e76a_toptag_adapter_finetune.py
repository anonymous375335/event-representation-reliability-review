#!/usr/bin/env python3
"""E76a TopTag adapter fine-tune over the E75e balanced systematic set."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

import e68c_cms_h4l_split_branch_disentanglement as e68c
import e75c_toptag_branch_protocol_smoke as e75c
import e75e_toptag_systematic_family_scaleup as e75e
from e66_cms_h4l_readout_smoke import weighted_auc


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
DEFAULT_BALANCED_INDICES = [
    "test_nominal_file_index.json",
    "esup_file_index.json",
    "esdown_file_index.json",
    "cer_file_index.json",
    "cpos_file_index.json",
    "bias_file_index.json",
]
E75E_REFERENCE = {
    "shared_baseline_physics_auc": 0.9432,
    "shared_baseline_z_nuis_physics_auc": 0.9059,
    "best_split_physics_auc": 0.9299,
    "best_split_z_nuis_domain_acc": 0.4120,
    "best_split_z_nuis_physics_auc": 0.8040,
}


class AdapterSplitBranchNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        adapter_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_domains: int,
        nuisance_latent_dim: int,
        adapter_scale: float,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, input_dim),
        )
        self.adapter_scale = nn.Parameter(torch.tensor(adapter_scale, dtype=torch.float32))
        self.split = e68c.SplitBranchNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_domains=num_domains,
            nuisance_latent_dim=nuisance_latent_dim,
        )

    def forward(self, x: torch.Tensor, grl_lambda: float = 0.0) -> dict[str, torch.Tensor]:
        adapted = x + self.adapter_scale * self.adapter(self.input_norm(x))
        return self.split(adapted, grl_lambda=grl_lambda)


def train_adapter(
    train_loader: DataLoader,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    val_domains: torch.Tensor,
    input_dim: int,
    pos_weight: float,
    epochs: int,
    learning_rate: float,
    adapter_dim: int,
    nuisance_weight: float,
    nuisance_latent_dim: int,
    orth_lambda: float,
    phys_adv_weight: float,
    device: torch.device,
) -> tuple[AdapterSplitBranchNet, list[dict]]:
    model = AdapterSplitBranchNet(
        input_dim=input_dim,
        adapter_dim=adapter_dim,
        hidden_dim=128,
        latent_dim=64,
        num_domains=len(e68c.DOMAIN_NAMES),
        nuisance_latent_dim=nuisance_latent_dim,
        adapter_scale=0.1,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)
    val_domains = val_domains.to(device)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"loss": 0.0, "physics": 0.0, "nuisance": 0.0, "adv": 0.0, "orth": 0.0, "count": 0}
        for batch_x, batch_y, batch_domain in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_domain = batch_domain.to(device)
            out = model(batch_x, grl_lambda=phys_adv_weight)
            physics_loss = F.binary_cross_entropy_with_logits(
                out["physics_logits"], batch_y, pos_weight=pos_weight_tensor
            )
            nuisance_loss = F.cross_entropy(out["nuisance_logits"], batch_domain)
            adv_loss = F.cross_entropy(out["phys_adv_logits"], batch_domain)
            orth_loss = e68c.orthogonal_penalty(out["z_phys"], out["z_nuis"])
            loss = physics_loss + nuisance_weight * nuisance_loss + phys_adv_weight * adv_loss + orth_lambda * orth_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            count = batch_y.shape[0]
            totals["loss"] += float(loss.detach().cpu()) * count
            totals["physics"] += float(physics_loss.detach().cpu()) * count
            totals["nuisance"] += float(nuisance_loss.detach().cpu()) * count
            totals["adv"] += float(adv_loss.detach().cpu()) * count
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
                "adapter_dim": adapter_dim,
                "nuisance_weight": nuisance_weight,
                "nuisance_latent_dim": nuisance_latent_dim,
                "phys_adv_weight": phys_adv_weight,
                "adapter_scale": float(model.adapter_scale.detach().cpu()),
                "train_loss": totals["loss"] / denom,
                "train_physics_loss": totals["physics"] / denom,
                "train_nuisance_loss": totals["nuisance"] / denom,
                "train_adv_loss": totals["adv"] / denom,
                "train_orthogonal_penalty": totals["orth"] / denom,
                "val_physics_loss": float(val_phys.cpu()),
                "val_nuisance_loss": float(val_nuis.cpu()),
                "val_nuisance_head_acc": val_nuis_acc,
                "val_orthogonal_penalty": val_orth,
            }
        )
    return model, history


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "runs" / "20260621-110653-e75a-toptag-record80030-preflight" / "record80030_files.csv",
    )
    parser.add_argument("--indices", nargs="+", default=DEFAULT_BALANCED_INDICES)
    parser.add_argument("--data-dir", type=Path, default=e75c.DATA_RAW_ROOT / "toptag_record80030_e75b")
    parser.add_argument("--cache-dir", type=Path, default=e75c.DATA_PROCESSED_ROOT / "toptag_record80030_e75c" / "h5_cache")
    parser.add_argument("--max-events-per-domain", type=int, default=100000)
    parser.add_argument("--max-constituents", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--probe-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--orth-lambda", type=float, default=0.25)
    args = parser.parse_args()

    e68c.set_seed(args.seed)
    REPORTS.mkdir(parents=True, exist_ok=True)
    run_dir = e75e.create_run_dir("e76a-toptag-adapter-finetune")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selected = e75e.select_first_files(e75e.read_rows(args.manifest), args.indices)
    domain_names = [row["domain"] for row in selected]
    e68c.DOMAIN_NAMES = domain_names.copy()
    feature_parts = []
    label_parts = []
    domain_parts = []
    weight_parts = []
    summaries = []
    for domain_id, row in enumerate(selected):
        path = args.data_dir / row["filename"]
        if not path.exists():
            path = e75e.download_file(row, args.data_dir)
        features, labels, domains, weights, summary = e75e.load_domain(
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

    candidates = [
        {"candidate": "adapter_domain_push64", "adapter_dim": 64, "nuisance_weight": 5.0, "nuisance_latent_dim": 32, "phys_adv_weight": 0.5},
        {"candidate": "adapter_domain_push128", "adapter_dim": 128, "nuisance_weight": 5.0, "nuisance_latent_dim": 32, "phys_adv_weight": 0.5},
        {"candidate": "adapter_balanced128", "adapter_dim": 128, "nuisance_weight": 2.0, "nuisance_latent_dim": 32, "phys_adv_weight": 0.5},
        {"candidate": "adapter_physics_preserve128", "adapter_dim": 128, "nuisance_weight": 1.0, "nuisance_latent_dim": 64, "phys_adv_weight": 0.25},
    ]
    rows = []
    history_rows = []
    for candidate in candidates:
        print(f"[fit-start] {candidate['candidate']} {dt.datetime.now().isoformat(timespec='seconds')}", flush=True)
        model, history = train_adapter(
            train_loader=train_loader,
            val_x=val_x,
            val_y=val_y,
            val_domains=val_domains,
            input_dim=train_x.shape[1],
            pos_weight=pos_weight,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            adapter_dim=candidate["adapter_dim"],
            nuisance_weight=candidate["nuisance_weight"],
            nuisance_latent_dim=candidate["nuisance_latent_dim"],
            orth_lambda=args.orth_lambda,
            phys_adv_weight=candidate["phys_adv_weight"],
            device=device,
        )
        print(f"[fit-done] {candidate['candidate']} {dt.datetime.now().isoformat(timespec='seconds')}", flush=True)
        history_rows.extend([{**row, "candidate": candidate["candidate"]} for row in history])
        rows.append(
            e75e.evaluate_model(
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

    metrics_csv = run_dir / "adapter_finetune_metrics.csv"
    history_csv = run_dir / "training_history.csv"
    e75e.write_csv(metrics_csv, rows)
    e75e.write_csv(history_csv, history_rows)
    best_leakage = min(rows, key=lambda row: (row["z_nuis_physics_probe_auc"], -row["physics_auc"]))
    best_tradeoff = max(rows, key=lambda row: (row["physics_auc"] - 0.5 * max(row["z_nuis_physics_probe_auc"] - 0.65, 0.0)))
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
        "learning_rate": args.learning_rate,
        "e75e_reference": E75E_REFERENCE,
        "domain_summaries": summaries,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    report_path = REPORTS / f"e76a_toptag_adapter_finetune_d{len(domain_names)}_{dt.datetime.now():%Y%m%d}_seed{args.seed}.md"
    lines = [
        "# E76a TopTag Adapter Fine-Tune",
        "",
        f"- run_dir: `{run_dir}`",
        f"- generated_at: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- data_dir: `{args.data_dir}`",
        f"- cache_dir: `{args.cache_dir}`",
        f"- device: `{device}`",
        f"- domains: `{', '.join(domain_names)}`",
        f"- random_domain_baseline: {1 / len(domain_names):.4f}",
        f"- events_per_domain_cap: {args.max_events_per_domain}",
        f"- total_events_loaded: {len(labels)}",
        f"- feature_dim: {features.shape[1]}",
        f"- epochs: {args.epochs}",
        f"- probe_epochs: {args.probe_epochs}",
        "- run note: adapter/readout fine-tune over first balanced shards.",
        "",
        "## E75e Reference",
        "",
        f"- shared baseline physics AUC: {E75E_REFERENCE['shared_baseline_physics_auc']:.4f}",
        f"- shared baseline `z_nuis -> physics`: {E75E_REFERENCE['shared_baseline_z_nuis_physics_auc']:.4f}",
        f"- best E75e split physics AUC: {E75E_REFERENCE['best_split_physics_auc']:.4f}",
        f"- best E75e split `z_nuis -> domain`: {E75E_REFERENCE['best_split_z_nuis_domain_acc']:.4f}",
        f"- best E75e split `z_nuis -> physics`: {E75E_REFERENCE['best_split_z_nuis_physics_auc']:.4f}",
        "",
        "## Best Candidates",
        "",
        (
            f"- lowest leakage: {best_leakage['candidate']}, physics AUC={e75e.format_float(best_leakage['physics_auc'])}, "
            f"z_nuis domain={e75e.format_float(best_leakage['z_nuis_domain_probe_acc'])}, "
            f"z_nuis physics={e75e.format_float(best_leakage['z_nuis_physics_probe_auc'])}"
        ),
        (
            f"- best tradeoff: {best_tradeoff['candidate']}, physics AUC={e75e.format_float(best_tradeoff['physics_auc'])}, "
            f"z_nuis domain={e75e.format_float(best_tradeoff['z_nuis_domain_probe_acc'])}, "
            f"z_nuis physics={e75e.format_float(best_tradeoff['z_nuis_physics_probe_auc'])}"
        ),
        "",
        "## Metrics",
        "",
        "| candidate | adapter dim | nuisance weight | nuisance dim | phys adv | physics AUC | bkg rejection @30% sig eff | score domain drift | nuisance head acc | z_phys domain probe | z_nuis domain probe | z_nuis physics probe AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {row['adapter_dim']} | {row['nuisance_weight']:g} | "
            f"{row['nuisance_latent_dim']} | {row['phys_adv_weight']:g} | "
            f"{e75e.format_float(row['physics_auc'])} | "
            f"{e75e.format_float(row['background_rejection_at_30pct_signal_eff'])} | "
            f"{e75e.format_float(row['score_domain_drift_max'])} | {e75e.format_float(row['nuisance_head_acc'])} | "
            f"{e75e.format_float(row['z_phys_domain_probe_acc'])} | {e75e.format_float(row['z_nuis_domain_probe_acc'])} | "
            f"{e75e.format_float(row['z_nuis_physics_probe_auc'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- Upgrade signal: `z_nuis -> physics` should fall materially below E75e 0.8040 while physics AUC stays close to 0.94.",
            "- If leakage remains above roughly 0.75, E76a is not strong enough and E76b should move to a small constituent encoder.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "experiment": "E76a TopTag adapter fine-tune",
                "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
                "run_dir": str(run_dir),
                "report": str(report_path),
                "metrics_csv": str(metrics_csv),
                "history_csv": str(history_csv),
                "config": config,
                "best_lowest_leakage": best_leakage,
                "best_tradeoff": best_tradeoff,
                "metrics": rows,
                "status": "done",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "status.txt").write_text(f"status: done\nreport: {report_path}\n", encoding="utf-8")
    print(f"E76a TopTag adapter fine-tune done: {run_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
