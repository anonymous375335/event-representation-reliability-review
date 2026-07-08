#!/usr/bin/env python3
"""E88 train a signal-tail-protected TopTag split model and export templates."""

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
import e76b_toptag_constituent_encoder as e76b
import e79_toptag_score_template_export as e79


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def candidate_configs(tail_weight: float) -> list[dict]:
    return [
        {
            "candidate": "constituent_balanced_reference",
            "mode": "split",
            "point_dim": 64,
            "nuisance_weight": 2.0,
            "nuisance_latent_dim": 32,
            "phys_adv_weight": 0.5,
            "tail_weight": 0.0,
        },
        {
            "candidate": "constituent_tail_protected",
            "mode": "split",
            "point_dim": 64,
            "nuisance_weight": 2.0,
            "nuisance_latent_dim": 32,
            "phys_adv_weight": 0.5,
            "tail_weight": tail_weight,
        },
    ]


def tail_protection_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor,
    threshold: float,
    temperature: float,
    min_events: int,
    label_mode: str,
) -> torch.Tensor:
    scores = torch.sigmoid(logits)
    soft_tail = torch.sigmoid((scores - threshold) / temperature)
    losses = []
    if label_mode == "background":
        label_values = [0.0]
    elif label_mode == "signal":
        label_values = [1.0]
    else:
        label_values = [0.0, 1.0]
    for label_value in label_values:
        nominal_mask = (domains == 0) & (labels == label_value)
        if int(nominal_mask.sum().item()) < min_events:
            continue
        nominal_rate = soft_tail[nominal_mask].mean().detach()
        for domain_id in torch.unique(domains):
            if int(domain_id.item()) == 0:
                continue
            selected = (domains == domain_id) & (labels == label_value)
            if int(selected.sum().item()) < min_events:
                continue
            losses.append((soft_tail[selected].mean() - nominal_rate).pow(2))
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def train_one(
    candidate: dict,
    train_loader: DataLoader,
    val_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    pos_weight: float,
    epochs: int,
    learning_rate: float,
    orth_lambda: float,
    tail_threshold: float,
    tail_temperature: float,
    tail_min_events: int,
    tail_label_mode: str,
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
        totals = {
            "loss": 0.0,
            "physics": 0.0,
            "nuisance": 0.0,
            "adv": 0.0,
            "orth": 0.0,
            "tail": 0.0,
            "count": 0,
        }
        for batch_const, batch_mask, batch_high, batch_y, batch_domain in train_loader:
            batch_const = batch_const.to(device)
            batch_mask = batch_mask.to(device)
            batch_high = batch_high.to(device)
            batch_y = batch_y.to(device)
            batch_domain = batch_domain.to(device)
            out = model(batch_const, batch_mask, batch_high, grl_lambda=candidate["phys_adv_weight"])
            physics_loss = F.binary_cross_entropy_with_logits(
                out["physics_logits"], batch_y, pos_weight=pos_weight_tensor
            )
            nuisance_loss = F.cross_entropy(out["nuisance_logits"], batch_domain)
            adv_loss = F.cross_entropy(out["phys_adv_logits"], batch_domain)
            orth_loss = e68c.orthogonal_penalty(out["z_phys"], out["z_nuis"])
            tail_loss = tail_protection_loss(
                out["physics_logits"],
                batch_y,
                batch_domain,
                tail_threshold,
                tail_temperature,
                tail_min_events,
                tail_label_mode,
            )
            loss = (
                physics_loss
                + candidate["nuisance_weight"] * nuisance_loss
                + candidate["phys_adv_weight"] * adv_loss
                + orth_lambda * orth_loss
                + candidate["tail_weight"] * tail_loss
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
            totals["tail"] += float(tail_loss.detach().cpu()) * count
            totals["count"] += count
        model.eval()
        with torch.no_grad():
            val_out = model(val_const, val_mask, val_high, grl_lambda=0.0)
            val_tail = tail_protection_loss(
                val_out["physics_logits"],
                val_y,
                val_domains,
                tail_threshold,
                tail_temperature,
                tail_min_events,
                tail_label_mode,
            )
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
                "tail_weight": candidate["tail_weight"],
                "tail_threshold": tail_threshold,
                "tail_temperature": tail_temperature,
                "tail_label_mode": tail_label_mode,
                "train_loss": totals["loss"] / denom,
                "train_physics_loss": totals["physics"] / denom,
                "train_nuisance_loss": totals["nuisance"] / denom,
                "train_adv_loss": totals["adv"] / denom,
                "train_orthogonal_penalty": totals["orth"] / denom,
                "train_tail_loss": totals["tail"] / denom,
                "val_physics_loss": float(val_phys.cpu()),
                "val_nuisance_loss": float(val_nuis.cpu()),
                "val_nuisance_head_acc": val_nuis_acc,
                "val_orthogonal_penalty": val_orth,
                "val_tail_loss": float(val_tail.cpu()),
            }
        )
    return model, history


def tail_metric_rows(
    candidate: str,
    domain_names: list[str],
    scores: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    signal_eff: float,
) -> list[dict]:
    nominal_id = domain_names.index("nominal")
    nominal_signal_scores = scores[(domains == nominal_id) & (labels == 1)]
    threshold = float(np.quantile(nominal_signal_scores, 1.0 - signal_eff))
    rows = []
    nominal_rates = {}
    for label_value in [0, 1]:
        selected = (domains == nominal_id) & (labels == label_value)
        nominal_rates[label_value] = float((scores[selected] >= threshold).mean())
    for domain_id, domain in enumerate(domain_names):
        for label_value in [0, 1]:
            selected = (domains == domain_id) & (labels == label_value)
            rate = float((scores[selected] >= threshold).mean())
            rows.append(
                {
                    "candidate": candidate,
                    "domain": domain,
                    "label": label_value,
                    "nominal_signal_eff_threshold": signal_eff,
                    "threshold": threshold,
                    "tail_rate": rate,
                    "delta_tail_rate_vs_nominal": rate - nominal_rates[label_value],
                    "events": int(selected.sum()),
                }
            )
    return rows


def load_data(args: argparse.Namespace) -> tuple:
    selected = e75e.select_first_files(e75e.read_rows(args.manifest), args.indices)
    domain_names = [row["domain"] for row in selected]
    e68c.DOMAIN_NAMES = domain_names.copy()
    parts = {"const": [], "mask": [], "high": [], "label": [], "domain": [], "weight": []}
    summaries = []
    for domain_id, row in enumerate(selected):
        path = args.data_dir / row["filename"]
        if not path.exists():
            e75e.download_file(row, args.data_dir)
        constituents, mask, high, labels, domains, weights, summary = e76b.load_domain_sequences(
            args.data_dir,
            args.cache_dir,
            row["domain"],
            row["filename"],
            domain_id,
            args.max_events_per_domain,
            args.max_constituents,
            args.seed,
        )
        parts["const"].append(constituents)
        parts["mask"].append(mask)
        parts["high"].append(high)
        parts["label"].append(labels)
        parts["domain"].append(domains)
        parts["weight"].append(weights)
        summaries.append(summary)
    return (
        domain_names,
        np.concatenate(parts["const"], axis=0),
        np.concatenate(parts["mask"], axis=0),
        np.concatenate(parts["high"], axis=0),
        np.concatenate(parts["label"], axis=0),
        np.concatenate(parts["domain"], axis=0),
        np.concatenate(parts["weight"], axis=0),
        summaries,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "runs" / "20260621-110653-e75a-toptag-record80030-preflight" / "record80030_files.csv",
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
    parser.add_argument("--tail-weight", type=float, default=2.0)
    parser.add_argument("--tail-threshold", type=float, default=0.8)
    parser.add_argument("--tail-temperature", type=float, default=0.06)
    parser.add_argument("--tail-min-events", type=int, default=32)
    parser.add_argument("--tail-label-mode", choices=["background", "signal", "both"], default="both")
    args = parser.parse_args()

    e68c.set_seed(args.seed)
    REPORTS.mkdir(parents=True, exist_ok=True)
    run_dir = e75e.create_run_dir("e88-toptag-signal-tail-protected-split")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain_names, constituents, masks, high, labels, domains, weights, summaries = load_data(args)
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

    train_const = torch.tensor(train_const_np, dtype=torch.float32)
    train_mask = torch.tensor(train_mask_np, dtype=torch.float32)
    train_high = torch.tensor(train_high_np, dtype=torch.float32)
    train_y = torch.tensor(train_y_np, dtype=torch.float32)
    train_domains = torch.tensor(train_domain_np, dtype=torch.long)
    val_const = torch.tensor(val_const_np, dtype=torch.float32)
    val_mask = torch.tensor(val_mask_np, dtype=torch.float32)
    val_high = torch.tensor(val_high_np, dtype=torch.float32)
    val_y = torch.tensor(val_y_np, dtype=torch.float32)
    val_domains = torch.tensor(val_domain_np, dtype=torch.long)
    train_loader = DataLoader(
        TensorDataset(train_const, train_mask, train_high, train_y, train_domains),
        batch_size=args.batch_size,
        shuffle=True,
    )
    pos = max(float((train_y_np == 1).sum()), 1.0)
    neg = max(float((train_y_np == 0).sum()), 1.0)
    pos_weight = neg / pos
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
    for candidate in candidate_configs(args.tail_weight):
        print(f"[fit-start] {candidate['candidate']} {dt.datetime.now().isoformat(timespec='seconds')}", flush=True)
        model, history = train_one(
            candidate,
            train_loader,
            val_train_tensors,
            pos_weight,
            args.epochs,
            args.learning_rate,
            args.orth_lambda,
            args.tail_threshold,
            args.tail_temperature,
            args.tail_min_events,
            args.tail_label_mode,
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
        tail_rows.extend(tail_metric_rows(candidate["candidate"], domain_names, scores, val_y_np, val_domain_np, 0.30))

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
    e75e.write_csv(run_dir / "validation_scores.csv", validation_score_rows)
    config = vars(args).copy()
    config.update(
        {
            "domains": domain_names,
            "total_events_loaded": int(len(labels)),
            "validation_events": int(len(val_y_np)),
            "device": str(device),
            "domain_summaries": summaries,
            "outputs": {
                "score_templates": "score_templates.csv",
                "tail_stability_summary": "tail_stability_summary.csv",
                "validation_scores": "validation_scores.csv",
            },
        }
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    (run_dir / "status.txt").write_text("completed\n", encoding="utf-8")

    report_path = REPORTS / f"e88_toptag_signal_tail_protected_split_{dt.datetime.now():%Y%m%d}_seed{args.seed}.md"
    lines = [
        "# E88 TopTag Signal-Tail-Protected Split",
        "",
        f"- run_dir: `{run_dir}`",
        f"- generated_at: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- device: `{device}`",
        f"- domains: `{', '.join(domain_names)}`",
        f"- tail protection: threshold `{args.tail_threshold}`, temperature `{args.tail_temperature}`, weight `{args.tail_weight}`",
        f"- tail label mode: `{args.tail_label_mode}`",
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
            "Training-objective test tracking profile-stress, high-score-tail diagnostics and physics AUC across shards.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"E88 TopTag signal-tail-protected split done: {run_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
