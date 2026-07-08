#!/usr/bin/env python3
"""E79 export TopTag score templates for a later pyhf workspace smoke."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import e68c_cms_h4l_split_branch_disentanglement as e68c
import e75c_toptag_branch_protocol_smoke as e75c
import e75e_toptag_systematic_family_scaleup as e75e
import e76b_toptag_constituent_encoder as e76b
from e66_cms_h4l_readout_smoke import weighted_auc


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_RECORD80030_MANIFEST = ROOT / "benchmarks" / "toptag_pyhf" / "manifests" / "record80030_files.csv"
DEFAULT_INDICES = e76b.DEFAULT_BALANCED_INDICES
TARGET_SIGNAL_EFFS = [0.30, 0.50]


def candidate_configs() -> list[dict]:
    return [
        {
            "candidate": "constituent_shared_baseline",
            "mode": "shared_baseline",
            "point_dim": 64,
            "nuisance_weight": 0.0,
            "nuisance_latent_dim": 64,
            "phys_adv_weight": 0.0,
        },
        {
            "candidate": "constituent_balanced",
            "mode": "split",
            "point_dim": 64,
            "nuisance_weight": 2.0,
            "nuisance_latent_dim": 32,
            "phys_adv_weight": 0.5,
        },
    ]


def format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def histogram_rows(
    candidate: str,
    domain_names: list[str],
    scores: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    edges: np.ndarray,
) -> list[dict]:
    rows = []
    for domain_id, domain in enumerate(domain_names):
        for label in [0, 1]:
            selected = (domains == domain_id) & (labels == label)
            counts, _ = np.histogram(scores[selected], bins=edges)
            total = int(counts.sum())
            for bin_index, count in enumerate(counts.tolist()):
                rows.append(
                    {
                        "candidate": candidate,
                        "domain": domain,
                        "label": label,
                        "bin_index": bin_index,
                        "bin_low": float(edges[bin_index]),
                        "bin_high": float(edges[bin_index + 1]),
                        "count": int(count),
                        "total": total,
                        "density": 0.0 if total == 0 else float(count / total),
                    }
                )
    return rows


def total_variation_distance(a: np.ndarray, b: np.ndarray) -> float | None:
    total_a = float(a.sum())
    total_b = float(b.sum())
    if total_a <= 0 or total_b <= 0:
        return None
    pa = a / total_a
    pb = b / total_b
    return float(0.5 * np.abs(pa - pb).sum())


def shape_metric_rows(
    candidate: str,
    domain_names: list[str],
    scores: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    edges: np.ndarray,
) -> list[dict]:
    rows = []
    nominal_id = domain_names.index("nominal")
    for label in [0, 1]:
        nominal_counts, _ = np.histogram(scores[(domains == nominal_id) & (labels == label)], bins=edges)
        for domain_id, domain in enumerate(domain_names):
            counts, _ = np.histogram(scores[(domains == domain_id) & (labels == label)], bins=edges)
            rows.append(
                {
                    "candidate": candidate,
                    "domain": domain,
                    "label": label,
                    "reference_domain": "nominal",
                    "tvd_vs_nominal": total_variation_distance(counts.astype(float), nominal_counts.astype(float)),
                    "events": int(counts.sum()),
                    "nominal_events": int(nominal_counts.sum()),
                }
            )
    return rows


def fixed_efficiency_rows(
    candidate: str,
    domain_names: list[str],
    scores: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    target_signal_effs: list[float],
) -> list[dict]:
    rows = []
    nominal_id = domain_names.index("nominal")
    nominal_signal_scores = scores[(domains == nominal_id) & (labels == 1)]
    if len(nominal_signal_scores) == 0:
        return rows
    for target_eff in target_signal_effs:
        threshold = float(np.quantile(nominal_signal_scores, 1.0 - target_eff))
        nominal_sig_eff = None
        nominal_bkg_eff = None
        for domain_id, domain in enumerate(domain_names):
            signal = scores[(domains == domain_id) & (labels == 1)]
            background = scores[(domains == domain_id) & (labels == 0)]
            signal_eff = None if len(signal) == 0 else float((signal >= threshold).mean())
            background_eff = None if len(background) == 0 else float((background >= threshold).mean())
            if domain == "nominal":
                nominal_sig_eff = signal_eff
                nominal_bkg_eff = background_eff
            rows.append(
                {
                    "candidate": candidate,
                    "domain": domain,
                    "target_signal_eff_nominal": target_eff,
                    "threshold": threshold,
                    "signal_efficiency": signal_eff,
                    "background_efficiency": background_eff,
                    "background_rejection": None
                    if background_eff is None or background_eff <= 0
                    else float(1.0 / background_eff),
                    "delta_signal_eff_vs_nominal": None
                    if signal_eff is None or nominal_sig_eff is None
                    else float(signal_eff - nominal_sig_eff),
                    "delta_background_eff_vs_nominal": None
                    if background_eff is None or nominal_bkg_eff is None
                    else float(background_eff - nominal_bkg_eff),
                    "signal_events": int(len(signal)),
                    "background_events": int(len(background)),
                }
            )
    return rows


def score_summary_rows(
    candidate: dict,
    domain_names: list[str],
    scores: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    weights: np.ndarray,
) -> list[dict]:
    rows = [
        {
            **candidate,
            "domain": "all",
            "events": int(len(labels)),
            "physics_auc": weighted_auc(labels, scores, weights),
            "background_rejection_at_30pct_signal_eff": e75c.background_rejection_at_signal_eff(labels, scores, 0.30),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
        }
    ]
    for domain_id, domain in enumerate(domain_names):
        selected = domains == domain_id
        rows.append(
            {
                **candidate,
                "domain": domain,
                "events": int(selected.sum()),
                "physics_auc": weighted_auc(labels[selected], scores[selected], weights[selected]),
                "background_rejection_at_30pct_signal_eff": e75c.background_rejection_at_signal_eff(
                    labels[selected], scores[selected], 0.30
                ),
                "score_mean": float(scores[selected].mean()),
                "score_std": float(scores[selected].std()),
            }
        )
    return rows


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
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--orth-lambda", type=float, default=0.25)
    parser.add_argument("--bins", type=int, default=20)
    args = parser.parse_args()

    e68c.set_seed(args.seed)
    REPORTS.mkdir(parents=True, exist_ok=True)
    run_dir = e75e.create_run_dir("e79-toptag-score-template-export")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selected = e75e.select_first_files(e75e.read_rows(args.manifest), args.indices)
    domain_names = [row["domain"] for row in selected]
    e68c.DOMAIN_NAMES = domain_names.copy()

    const_parts = []
    mask_parts = []
    high_parts = []
    label_parts = []
    domain_parts = []
    weight_parts = []
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
        const_parts.append(constituents)
        mask_parts.append(mask)
        high_parts.append(high)
        label_parts.append(labels)
        domain_parts.append(domains)
        weight_parts.append(weights)
        summaries.append(summary)

    constituents = np.concatenate(const_parts, axis=0)
    masks = np.concatenate(mask_parts, axis=0)
    high = np.concatenate(high_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    domains = np.concatenate(domain_parts, axis=0)
    weights = np.concatenate(weight_parts, axis=0)

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
    score_columns = {
        "event_index": np.arange(len(val_y_np)),
        "domain": np.array([domain_names[index] for index in val_domain_np]),
        "domain_id": val_domain_np,
        "label": val_y_np,
    }
    template_rows = []
    shape_rows = []
    fixed_rows = []
    summary_rows = []
    history_rows = []

    val_train_tensors = (val_const, val_mask, val_high, val_y, val_domains)
    candidates = candidate_configs()
    for candidate in candidates:
        print(f"[fit-start] {candidate['candidate']} {dt.datetime.now().isoformat(timespec='seconds')}", flush=True)
        model, history = e76b.train_one(
            candidate=candidate,
            train_loader=train_loader,
            val_tensors=val_train_tensors,
            pos_weight=pos_weight,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            orth_lambda=args.orth_lambda,
            device=device,
        )
        print(f"[fit-done] {candidate['candidate']} {dt.datetime.now().isoformat(timespec='seconds')}", flush=True)
        torch.save(model.state_dict(), run_dir / f"{candidate['candidate']}.pt")
        history_rows.extend(history)
        scores, _, _, _ = e76b.embed_and_score(model, val_const, val_mask, val_high, device, args.batch_size)
        score_columns[f"score_{candidate['candidate']}"] = scores
        template_rows.extend(histogram_rows(candidate["candidate"], domain_names, scores, val_y_np, val_domain_np, edges))
        shape_rows.extend(shape_metric_rows(candidate["candidate"], domain_names, scores, val_y_np, val_domain_np, edges))
        fixed_rows.extend(
            fixed_efficiency_rows(candidate["candidate"], domain_names, scores, val_y_np, val_domain_np, TARGET_SIGNAL_EFFS)
        )
        summary_rows.extend(score_summary_rows(candidate, domain_names, scores, val_y_np, val_domain_np, val_weights_np))

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
    e75e.write_csv(run_dir / "validation_scores.csv", validation_score_rows)

    config = {
        "manifest": str(args.manifest),
        "indices": args.indices,
        "data_dir": str(args.data_dir),
        "cache_dir": str(args.cache_dir),
        "domains": domain_names,
        "max_events_per_domain": args.max_events_per_domain,
        "max_constituents": args.max_constituents,
        "total_events_loaded": int(len(labels)),
        "validation_events": int(len(val_y_np)),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "orth_lambda": args.orth_lambda,
        "learning_rate": args.learning_rate,
        "bins": args.bins,
        "target_signal_efficiencies": TARGET_SIGNAL_EFFS,
        "domain_summaries": summaries,
        "outputs": {
            "score_templates": "score_templates.csv",
            "shape_metrics": "shape_metrics.csv",
            "fixed_efficiency_summary": "fixed_efficiency_summary.csv",
            "score_summary": "score_summary.csv",
            "training_history": "training_history.csv",
            "validation_scores": "validation_scores.csv",
            "model_checkpoints": [f"{candidate['candidate']}.pt" for candidate in candidates],
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "status.txt").write_text("completed\n", encoding="utf-8")

    max_tvd = {}
    for candidate in candidates:
        values = [
            row["tvd_vs_nominal"]
            for row in shape_rows
            if row["candidate"] == candidate["candidate"] and row["domain"] != "nominal" and row["tvd_vs_nominal"] is not None
        ]
        max_tvd[candidate["candidate"]] = None if not values else max(values)

    report_path = REPORTS / f"e79_toptag_score_template_export_{dt.datetime.now():%Y%m%d}_seed{args.seed}.md"
    lines = [
        "# E79 TopTag Score Template Export",
        "",
        f"- run_dir: `{run_dir}`",
        f"- generated_at: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- device: `{device}`",
        f"- domains: `{', '.join(domain_names)}`",
        f"- events_per_domain_cap: {args.max_events_per_domain}",
        f"- total_events_loaded: {len(labels)}",
        f"- validation_events: {len(val_y_np)}",
        f"- histogram_bins: {args.bins}",
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
                auc=format_float(row["physics_auc"]),
                rej=format_float(row["background_rejection_at_30pct_signal_eff"]),
                mean=format_float(row["score_mean"]),
                std=format_float(row["score_std"]),
            )
        )
    lines.extend(
        [
            "",
            "## Shape drift",
            "",
            "| candidate | max TVD vs nominal across non-nominal domains/labels |",
            "|---|---:|",
        ]
    )
    for candidate in candidates:
        lines.append(f"| {candidate['candidate']} | {format_float(max_tvd[candidate['candidate']])} |")
    lines.extend(
        [
            "",
            "## Exported artifacts",
            "",
            "- `score_templates.csv`: binned score templates by candidate, domain, and label.",
            "- `fixed_efficiency_summary.csv`: domain shifts at nominal signal-efficiency thresholds.",
            "- `shape_metrics.csv`: total-variation distance against nominal templates.",
            "- `validation_scores.csv`: event-level validation scores for direct workspace construction.",
            "- `*.pt`: model checkpoints used to generate the templates.",
            "",
            "## Run Note",
            "",
            "The exported templates support pyhf/HistFactory workspace construction and profiling of the exposed systematic labels against the score-only comparison.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"E79 TopTag score template export done: {run_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
