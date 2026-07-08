#!/usr/bin/env python3
"""E91 train with a frozen-template, post-hoc calibrated residual target."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

import e68c_cms_h4l_split_branch_disentanglement as e68c
import e75c_toptag_branch_protocol_smoke as e75c
import e75e_toptag_systematic_family_scaleup as e75e
import e76b_toptag_constituent_encoder as e76b
import e79_toptag_score_template_export as e79
import e88_toptag_signal_tail_protected_split as e88


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_RECORD80030_MANIFEST = ROOT / "benchmarks" / "toptag_pyhf" / "manifests" / "record80030_files.csv"


def candidate_configs(residual_weight: float) -> list[dict]:
    return [
        {
            "candidate": "constituent_balanced_reference",
            "mode": "split",
            "point_dim": 64,
            "nuisance_weight": 2.0,
            "nuisance_latent_dim": 32,
            "phys_adv_weight": 0.5,
            "residual_weight": 0.0,
        },
        {
            "candidate": "constituent_frozen_residual_target",
            "mode": "split",
            "point_dim": 64,
            "nuisance_weight": 2.0,
            "nuisance_latent_dim": 32,
            "phys_adv_weight": 0.5,
            "residual_weight": residual_weight,
        },
    ]


def load_frozen_model(
    checkpoint_dir: Path,
    checkpoint_name: str,
    high_dim: int,
    num_domains: int,
    device: torch.device,
) -> e76b.ConstituentSplitNet:
    model = e76b.ConstituentSplitNet(
        constituent_dim=4,
        high_dim=high_dim,
        point_dim=64,
        hidden_dim=128,
        latent_dim=64,
        nuisance_latent_dim=32,
        num_domains=num_domains,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_dir / checkpoint_name, map_location=device))
    model.eval()
    return model


def frozen_residual_weights(
    frozen_scores: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    bins: int,
    min_score: float,
) -> tuple[np.ndarray, list[dict]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_ids = np.clip(np.digitize(frozen_scores, edges, right=False) - 1, 0, bins - 1)
    nominal = domains == 0
    nominal_bkg = nominal & (labels == 0)
    nominal_sig = nominal & (labels == 1)
    nominal_bkg_counts = np.bincount(bin_ids[nominal_bkg], minlength=bins).astype(float)
    nominal_sig_counts = np.bincount(bin_ids[nominal_sig], minlength=bins).astype(float)
    nominal_bkg_density = nominal_bkg_counts / max(nominal_bkg_counts.sum(), 1.0)
    signal_fraction = nominal_sig_counts / np.maximum(nominal_sig_counts + nominal_bkg_counts, 1.0)

    weights = np.zeros_like(frozen_scores, dtype=np.float32)
    rows = []
    for domain_id in sorted(set(domains.tolist())):
        if domain_id == 0:
            continue
        selected_bkg = (domains == domain_id) & (labels == 0)
        counts = np.bincount(bin_ids[selected_bkg], minlength=bins).astype(float)
        density = counts / max(counts.sum(), 1.0)
        residual = np.maximum(density - nominal_bkg_density, 0.0) * signal_fraction * (centers >= min_score)
        max_residual = float(residual.max())
        if max_residual > 0:
            residual = residual / max_residual
        for bin_id in range(bins):
            mask = selected_bkg & (bin_ids == bin_id)
            weights[mask] = max(weights[mask].max(initial=0.0), residual[bin_id])
            rows.append(
                {
                    "domain_id": domain_id,
                    "bin_index": bin_id,
                    "bin_center": float(centers[bin_id]),
                    "nominal_background_density": float(nominal_bkg_density[bin_id]),
                    "domain_background_density": float(density[bin_id]),
                    "signal_fraction": float(signal_fraction[bin_id]),
                    "calibrated_weight": float(residual[bin_id]),
                    "events": int(mask.sum()),
                }
            )
    return weights, rows


def residual_target_loss(logits: torch.Tensor, weights: torch.Tensor, min_score: float, temperature: float) -> torch.Tensor:
    if float(weights.sum().detach().cpu()) <= 0:
        return logits.new_tensor(0.0)
    scores = torch.sigmoid(logits)
    soft_high = torch.sigmoid((scores - min_score) / temperature)
    return (weights * soft_high).sum() / weights.sum().clamp_min(1e-12)


def train_one(
    candidate: dict,
    train_loader: DataLoader,
    val_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    pos_weight: float,
    epochs: int,
    learning_rate: float,
    orth_lambda: float,
    residual_min_score: float,
    residual_temperature: float,
    device: torch.device,
) -> tuple[e76b.ConstituentSplitNet, list[dict]]:
    val_const, val_mask, val_high, val_y, val_domains = [tensor.to(device) for tensor in val_tensors]
    model = e76b.ConstituentSplitNet(
        constituent_dim=4,
        high_dim=val_high.shape[1],
        point_dim=candidate["point_dim"],
        hidden_dim=128,
        latent_dim=64,
        nuisance_latent_dim=candidate["nuisance_latent_dim"],
        num_domains=len(e68c.DOMAIN_NAMES),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"loss": 0.0, "physics": 0.0, "nuisance": 0.0, "adv": 0.0, "orth": 0.0, "residual": 0.0, "count": 0}
        for batch_const, batch_mask, batch_high, batch_y, batch_domain, batch_residual_w in train_loader:
            batch_const = batch_const.to(device)
            batch_mask = batch_mask.to(device)
            batch_high = batch_high.to(device)
            batch_y = batch_y.to(device)
            batch_domain = batch_domain.to(device)
            batch_residual_w = batch_residual_w.to(device)
            out = model(batch_const, batch_mask, batch_high, grl_lambda=candidate["phys_adv_weight"])
            physics_loss = F.binary_cross_entropy_with_logits(
                out["physics_logits"], batch_y, pos_weight=pos_weight_tensor
            )
            nuisance_loss = F.cross_entropy(out["nuisance_logits"], batch_domain)
            adv_loss = F.cross_entropy(out["phys_adv_logits"], batch_domain)
            orth_loss = e68c.orthogonal_penalty(out["z_phys"], out["z_nuis"])
            residual_loss = residual_target_loss(
                out["physics_logits"], batch_residual_w, residual_min_score, residual_temperature
            )
            loss = (
                physics_loss
                + candidate["nuisance_weight"] * nuisance_loss
                + candidate["phys_adv_weight"] * adv_loss
                + orth_lambda * orth_loss
                + candidate["residual_weight"] * residual_loss
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            count = batch_y.shape[0]
            totals["loss"] += float(loss.detach().cpu()) * count
            totals["physics"] += float(physics_loss.detach().cpu()) * count
            totals["nuisance"] += float(nuisance_loss.detach().cpu()) * count
            totals["adv"] += float(adv_loss.detach().cpu()) * count
            totals["orth"] += float(orth_loss.detach().cpu()) * count
            totals["residual"] += float(residual_loss.detach().cpu()) * count
            totals["count"] += count
        model.eval()
        with torch.no_grad():
            val_out = model(val_const, val_mask, val_high, grl_lambda=0.0)
            val_phys = F.binary_cross_entropy_with_logits(
                val_out["physics_logits"], val_y, pos_weight=pos_weight_tensor
            )
            val_nuis = F.cross_entropy(val_out["nuisance_logits"], val_domains)
            val_nuis_acc = float((val_out["nuisance_logits"].argmax(dim=1) == val_domains).float().mean().cpu())
            val_orth = float(e68c.orthogonal_penalty(val_out["z_phys"], val_out["z_nuis"]).cpu())
        denom = max(totals.pop("count"), 1)
        history.append(
            {
                "candidate": candidate["candidate"],
                "epoch": epoch,
                "residual_weight": candidate["residual_weight"],
                "train_loss": totals["loss"] / denom,
                "train_physics_loss": totals["physics"] / denom,
                "train_nuisance_loss": totals["nuisance"] / denom,
                "train_adv_loss": totals["adv"] / denom,
                "train_orthogonal_penalty": totals["orth"] / denom,
                "train_residual_target_loss": totals["residual"] / denom,
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
        default=DEFAULT_RECORD80030_MANIFEST,
    )
    parser.add_argument("--indices", nargs="+", default=e76b.DEFAULT_BALANCED_INDICES)
    parser.add_argument("--data-dir", type=Path, default=e75c.DATA_RAW_ROOT / "toptag_record80030_e75b")
    parser.add_argument("--cache-dir", type=Path, default=e75c.DATA_PROCESSED_ROOT / "toptag_record80030_e75c" / "h5_cache")
    parser.add_argument("--max-events-per-domain", type=int, default=100000)
    parser.add_argument("--max-constituents", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--orth-lambda", type=float, default=0.25)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--frozen-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--frozen-checkpoint-name", default="constituent_balanced.pt")
    parser.add_argument("--residual-weight", type=float, default=0.5)
    parser.add_argument("--residual-min-score", type=float, default=0.75)
    parser.add_argument("--residual-temperature", type=float, default=0.04)
    args = parser.parse_args()

    e68c.set_seed(args.seed)
    REPORTS.mkdir(parents=True, exist_ok=True)
    run_dir = e75e.create_run_dir("e91-toptag-frozen-template-residual-target")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain_names, constituents, masks, high, labels, domains, weights, summaries = e88.load_data(args)
    train_idx, val_idx = e75c.joint_stratified_split(labels, domains, args.val_ratio, args.seed)
    train_const_np, val_const_np = e76b.standardize_constituents(
        constituents[train_idx], constituents[val_idx], masks[train_idx], masks[val_idx]
    )
    train_high_np, val_high_np = e76b.standardize_high(high[train_idx], high[val_idx])
    train_mask_np = masks[train_idx].astype(np.float32)
    val_mask_np = masks[val_idx].astype(np.float32)
    train_y_np = labels[train_idx]
    val_y_np = labels[val_idx]
    train_domain_np = domains[train_idx]
    val_domain_np = domains[val_idx]
    val_weights_np = weights[val_idx]

    e68c.DOMAIN_NAMES = domain_names.copy()
    frozen = load_frozen_model(
        args.frozen_checkpoint_dir,
        args.frozen_checkpoint_name,
        train_high_np.shape[1],
        len(domain_names),
        device,
    )
    frozen_scores, _, _, _ = e76b.embed_and_score(
        frozen,
        torch.tensor(train_const_np, dtype=torch.float32),
        torch.tensor(train_mask_np, dtype=torch.float32),
        torch.tensor(train_high_np, dtype=torch.float32),
        device,
        args.batch_size,
    )
    residual_weights_np, residual_rows = frozen_residual_weights(
        frozen_scores, train_y_np, train_domain_np, args.bins, args.residual_min_score
    )

    train_const = torch.tensor(train_const_np, dtype=torch.float32)
    train_mask = torch.tensor(train_mask_np, dtype=torch.float32)
    train_high = torch.tensor(train_high_np, dtype=torch.float32)
    train_y = torch.tensor(train_y_np, dtype=torch.float32)
    train_domains = torch.tensor(train_domain_np, dtype=torch.long)
    train_residual_weights = torch.tensor(residual_weights_np, dtype=torch.float32)
    val_const = torch.tensor(val_const_np, dtype=torch.float32)
    val_mask = torch.tensor(val_mask_np, dtype=torch.float32)
    val_high = torch.tensor(val_high_np, dtype=torch.float32)
    val_y = torch.tensor(val_y_np, dtype=torch.float32)
    val_domains = torch.tensor(val_domain_np, dtype=torch.long)
    train_loader = DataLoader(
        TensorDataset(train_const, train_mask, train_high, train_y, train_domains, train_residual_weights),
        batch_size=args.batch_size,
        shuffle=True,
    )
    pos_weight = max(float((train_y_np == 0).sum()), 1.0) / max(float((train_y_np == 1).sum()), 1.0)
    edges = np.linspace(0.0, 1.0, args.bins + 1)

    template_rows = []
    shape_rows = []
    fixed_rows = []
    summary_rows = []
    history_rows = []
    tail_rows = []
    score_columns = {
        "event_index": np.arange(len(val_y_np)),
        "domain": np.array([domain_names[index] for index in val_domain_np]),
        "domain_id": val_domain_np,
        "label": val_y_np,
    }
    val_train_tensors = (val_const, val_mask, val_high, val_y, val_domains)
    for candidate in candidate_configs(args.residual_weight):
        print(f"[fit-start] {candidate['candidate']} {dt.datetime.now().isoformat(timespec='seconds')}", flush=True)
        model, history = train_one(
            candidate,
            train_loader,
            val_train_tensors,
            pos_weight,
            args.epochs,
            args.learning_rate,
            args.orth_lambda,
            args.residual_min_score,
            args.residual_temperature,
            device,
        )
        print(f"[fit-done] {candidate['candidate']} {dt.datetime.now().isoformat(timespec='seconds')}", flush=True)
        torch.save(model.state_dict(), run_dir / f"{candidate['candidate']}.pt")
        history_rows.extend(history)
        scores, _, _, _ = e76b.embed_and_score(model, val_const, val_mask, val_high, device, args.batch_size)
        score_columns[f"score_{candidate['candidate']}"] = scores
        template_rows.extend(e79.histogram_rows(candidate["candidate"], domain_names, scores, val_y_np, val_domain_np, edges))
        shape_rows.extend(e79.shape_metric_rows(candidate["candidate"], domain_names, scores, val_y_np, val_domain_np, edges))
        fixed_rows.extend(
            e79.fixed_efficiency_rows(candidate["candidate"], domain_names, scores, val_y_np, val_domain_np, [0.30, 0.50])
        )
        summary_rows.extend(e79.score_summary_rows(candidate, domain_names, scores, val_y_np, val_domain_np, val_weights_np))
        tail_rows.extend(e88.tail_metric_rows(candidate["candidate"], domain_names, scores, val_y_np, val_domain_np, 0.30))

    validation_score_rows = []
    for row_index in range(len(val_y_np)):
        row = {}
        for key, values in score_columns.items():
            value = values[row_index]
            row[key] = value.item() if hasattr(value, "item") else value
        validation_score_rows.append(row)

    e75e.write_csv(run_dir / "score_templates.csv", template_rows)
    e75e.write_csv(run_dir / "shape_metrics.csv", shape_rows)
    e75e.write_csv(run_dir / "fixed_efficiency_summary.csv", fixed_rows)
    e75e.write_csv(run_dir / "score_summary.csv", summary_rows)
    e75e.write_csv(run_dir / "training_history.csv", history_rows)
    e75e.write_csv(run_dir / "tail_stability_summary.csv", tail_rows)
    e75e.write_csv(run_dir / "frozen_residual_targets.csv", residual_rows)
    e75e.write_csv(run_dir / "validation_scores.csv", validation_score_rows)
    config = vars(args).copy()
    config.update(
        {
            "domains": domain_names,
            "total_events_loaded": int(len(labels)),
            "validation_events": int(len(val_y_np)),
            "device": str(device),
            "n_residual_weighted_train_events": int((residual_weights_np > 0).sum()),
            "mean_positive_residual_weight": None
            if not (residual_weights_np > 0).any()
            else float(residual_weights_np[residual_weights_np > 0].mean()),
            "domain_summaries": summaries,
            "loss_boundary": "Residual targets are frozen from a baseline checkpoint on the training split; profile-stress validation remains the decision gate.",
        }
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    (run_dir / "status.txt").write_text("completed\n", encoding="utf-8")

    report_path = REPORTS / f"e91_toptag_frozen_template_residual_target_{dt.datetime.now():%Y%m%d}_seed{args.seed}.md"
    lines = [
        "# E91 TopTag Frozen-Template Residual Target",
        "",
        f"- run_dir: `{run_dir}`",
        f"- generated_at: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- device: `{device}`",
        f"- frozen checkpoint: `{args.frozen_checkpoint_dir / args.frozen_checkpoint_name}`",
        f"- residual weighted train events: `{int((residual_weights_np > 0).sum())}`",
        f"- residual weight: `{args.residual_weight}`",
        f"- residual min score: `{args.residual_min_score}`",
        "",
        "## Score summaries",
        "",
        "| candidate | domain | AUC | rejection@30% sig eff | score mean | score std |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {candidate} | {domain} | {auc} | {rej} | {mean} | {std} |".format(
                candidate=row["candidate"],
                domain=row["domain"],
                auc=e79.format_float(row["physics_auc"]),
                rej=e79.format_float(row["background_rejection_at_30pct_signal_eff"]),
                mean=e79.format_float(row["score_mean"]),
                std=e79.format_float(row["score_std"]),
            )
        )
    lines.extend(
        [
            "",
            "## Run Note",
            "",
            "Frozen-template target test tracking E81-style profile stress alongside AUC cost.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"E91 TopTag frozen-template residual target done: {run_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
