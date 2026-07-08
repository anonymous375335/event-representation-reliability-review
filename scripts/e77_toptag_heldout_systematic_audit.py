#!/usr/bin/env python3
"""E77 held-out systematic audit for the E76b TopTag constituent encoder."""

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
from e66_cms_h4l_readout_smoke import weighted_auc


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_RECORD80030_MANIFEST = ROOT / "benchmarks" / "toptag_pyhf" / "manifests" / "record80030_files.csv"
TRAIN_INDICES = [
    "test_nominal_file_index.json",
    "esup_file_index.json",
    "esdown_file_index.json",
    "cer_file_index.json",
    "cpos_file_index.json",
]
HOLDOUT_INDEX = "bias_file_index.json"


def transform_with_train_stats(
    train_const: np.ndarray,
    eval_const: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    train_high: np.ndarray,
    eval_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = train_mask > 0
    if valid.any():
        const_mean = train_const[valid].mean(axis=0, keepdims=True)
        const_std = train_const[valid].std(axis=0, keepdims=True)
        const_std[const_std < 1e-6] = 1.0
    else:
        const_mean = np.zeros((1, train_const.shape[-1]), dtype=np.float32)
        const_std = np.ones((1, train_const.shape[-1]), dtype=np.float32)
    high_mean = train_high.mean(axis=0, keepdims=True)
    high_std = train_high.std(axis=0, keepdims=True)
    high_std[high_std < 1e-6] = 1.0
    train_const_out = ((train_const - const_mean) / const_std).astype(np.float32) * train_mask[:, :, None]
    eval_const_out = ((eval_const - const_mean) / const_std).astype(np.float32) * eval_mask[:, :, None]
    train_high_out = ((train_high - high_mean) / high_std).astype(np.float32)
    eval_high_out = ((eval_high - high_mean) / high_std).astype(np.float32)
    return train_const_out, eval_const_out, train_high_out, eval_high_out


def train_holdout_probe_auc(
    seen_z: np.ndarray,
    holdout_z: np.ndarray,
    epochs: int,
    seed: int,
    device: torch.device,
) -> float:
    rng = np.random.default_rng(seed)
    seen_idx = rng.permutation(len(seen_z))
    hold_idx = rng.permutation(len(holdout_z))
    seen_mid = len(seen_idx) // 2
    hold_mid = len(hold_idx) // 2
    train_z = np.concatenate([seen_z[seen_idx[:seen_mid]], holdout_z[hold_idx[:hold_mid]]], axis=0)
    train_y = np.concatenate([np.zeros(seen_mid), np.ones(hold_mid)]).astype(np.float32)
    val_z = np.concatenate([seen_z[seen_idx[seen_mid:]], holdout_z[hold_idx[hold_mid:]]], axis=0)
    val_y = np.concatenate([np.zeros(len(seen_idx) - seen_mid), np.ones(len(hold_idx) - hold_mid)]).astype(np.int64)
    probe = nn.Linear(train_z.shape[1], 1).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=2e-3, weight_decay=1e-4)
    ds = TensorDataset(torch.tensor(train_z, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=8192, shuffle=True)
    for _ in range(epochs):
        probe.train()
        for batch_z, batch_y in loader:
            batch_z = batch_z.to(device)
            batch_y = batch_y.to(device)
            loss = F.binary_cross_entropy_with_logits(probe(batch_z).squeeze(-1), batch_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    probe.eval()
    with torch.no_grad():
        logits = probe(torch.tensor(val_z, dtype=torch.float32, device=device)).squeeze(-1).cpu().numpy()
    scores = 1.0 / (1.0 + np.exp(-logits))
    return weighted_auc(val_y, scores, np.ones_like(val_y, dtype=np.float32))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_RECORD80030_MANIFEST,
    )
    parser.add_argument("--data-dir", type=Path, default=e75c.DATA_RAW_ROOT / "toptag_record80030_e75b")
    parser.add_argument("--cache-dir", type=Path, default=e75c.DATA_PROCESSED_ROOT / "toptag_record80030_e75c" / "h5_cache")
    parser.add_argument("--max-events-per-domain", type=int, default=100000)
    parser.add_argument("--max-constituents", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--probe-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--orth-lambda", type=float, default=0.25)
    args = parser.parse_args()

    e68c.set_seed(args.seed)
    REPORTS.mkdir(parents=True, exist_ok=True)
    run_dir = e75e.create_run_dir("e77-toptag-heldout-systematic-audit")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = e75e.read_rows(args.manifest)
    train_selected = e75e.select_first_files(rows, TRAIN_INDICES)
    holdout_selected = e75e.select_first_files(rows, [HOLDOUT_INDEX])[0]
    train_domain_names = [row["domain"] for row in train_selected]
    holdout_domain = holdout_selected["domain"]
    e68c.DOMAIN_NAMES = train_domain_names.copy()

    const_parts = []
    mask_parts = []
    high_parts = []
    label_parts = []
    domain_parts = []
    weight_parts = []
    summaries = []
    for domain_id, row in enumerate(train_selected):
        if not (args.data_dir / row["filename"]).exists():
            e75e.download_file(row, args.data_dir)
        loaded = e76b.load_domain_sequences(
            args.data_dir,
            args.cache_dir,
            row["domain"],
            row["filename"],
            domain_id,
            args.max_events_per_domain,
            args.max_constituents,
            args.seed,
        )
        constituents, mask, high, labels, domains, weights, summary = loaded
        const_parts.append(constituents)
        mask_parts.append(mask)
        high_parts.append(high)
        label_parts.append(labels)
        domain_parts.append(domains)
        weight_parts.append(weights)
        summaries.append(summary)

    if not (args.data_dir / holdout_selected["filename"]).exists():
        e75e.download_file(holdout_selected, args.data_dir)
    holdout_const, holdout_mask, holdout_high, holdout_y_np, _, holdout_weights_np, holdout_summary = (
        e76b.load_domain_sequences(
            args.data_dir,
            args.cache_dir,
            holdout_domain,
            holdout_selected["filename"],
            0,
            args.max_events_per_domain,
            args.max_constituents,
            args.seed + 911,
        )
    )

    constituents = np.concatenate(const_parts, axis=0)
    masks = np.concatenate(mask_parts, axis=0)
    high = np.concatenate(high_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    domains = np.concatenate(domain_parts, axis=0)
    weights = np.concatenate(weight_parts, axis=0)
    train_idx, val_idx = e75c.joint_stratified_split(labels, domains, args.val_ratio, args.seed)
    train_const_np, eval_const_np, train_high_np, eval_high_np = transform_with_train_stats(
        constituents[train_idx],
        np.concatenate([constituents[val_idx], holdout_const], axis=0),
        masks[train_idx],
        np.concatenate([masks[val_idx], holdout_mask], axis=0),
        high[train_idx],
        np.concatenate([high[val_idx], holdout_high], axis=0),
    )
    seen_val_count = len(val_idx)
    val_const_np = eval_const_np[:seen_val_count]
    holdout_const_np = eval_const_np[seen_val_count:]
    val_high_np = eval_high_np[:seen_val_count]
    holdout_high_np = eval_high_np[seen_val_count:]
    train_mask_np = masks[train_idx].astype(np.float32)
    val_mask_np = masks[val_idx].astype(np.float32)
    holdout_mask_np = holdout_mask.astype(np.float32)
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
    holdout_const_t = torch.tensor(holdout_const_np, dtype=torch.float32)
    holdout_mask_t = torch.tensor(holdout_mask_np, dtype=torch.float32)
    holdout_high_t = torch.tensor(holdout_high_np, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(train_const, train_mask, train_high, train_y, train_domains),
        batch_size=args.batch_size,
        shuffle=True,
    )
    pos = max(float((train_y_np == 1).sum()), 1.0)
    neg = max(float((train_y_np == 0).sum()), 1.0)
    pos_weight = neg / pos
    candidates = [
        {
            "candidate": "heldout_shared_baseline",
            "mode": "shared_baseline",
            "point_dim": 64,
            "nuisance_weight": 0.0,
            "nuisance_latent_dim": 64,
            "phys_adv_weight": 0.0,
        },
        {
            "candidate": "heldout_constituent_balanced",
            "mode": "split",
            "point_dim": 64,
            "nuisance_weight": 2.0,
            "nuisance_latent_dim": 32,
            "phys_adv_weight": 0.5,
        },
        {
            "candidate": "heldout_physics_preserve",
            "mode": "split",
            "point_dim": 96,
            "nuisance_weight": 1.0,
            "nuisance_latent_dim": 64,
            "phys_adv_weight": 0.25,
        },
    ]
    train_eval_tensors = (train_const, train_mask, train_high)
    val_eval_tensors = (val_const, val_mask, val_high)
    val_train_tensors = (val_const, val_mask, val_high, val_y, val_domains)
    result_rows = []
    history_rows = []
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
        history_rows.extend(history)
        seen_row = e76b.evaluate(
            candidate,
            model,
            train_eval_tensors,
            val_eval_tensors,
            train_y_np,
            val_y_np,
            train_domain_np,
            val_domain_np,
            val_weights_np,
            args.probe_epochs,
            args.batch_size,
            device,
        )
        holdout_scores, holdout_z_phys, holdout_z_nuis, _ = e76b.embed_and_score(
            model, holdout_const_t, holdout_mask_t, holdout_high_t, device, args.batch_size
        )
        _, train_z_phys, train_z_nuis, _ = e76b.embed_and_score(
            model, train_const, train_mask, train_high, device, args.batch_size
        )
        _, seen_z_phys, seen_z_nuis, _ = e76b.embed_and_score(
            model, val_const, val_mask, val_high, device, args.batch_size
        )
        nominal_scores = seen_row["score_domain_drift_max"]
        nominal_val_scores, _, _, _ = e76b.embed_and_score(
            model,
            val_const[val_domain_np == 0],
            val_mask[val_domain_np == 0],
            val_high[val_domain_np == 0],
            device,
            args.batch_size,
        )
        holdout_physics_probe = e68c.train_physics_probe_auc(
            train_z_nuis,
            train_y_np,
            holdout_z_nuis,
            holdout_y_np,
            holdout_weights_np,
            args.probe_epochs,
            device,
        )
        result_rows.append(
            {
                **candidate,
                "seen_physics_auc": seen_row["physics_auc"],
                "seen_bkg_rejection_at_30pct_signal_eff": seen_row[
                    "background_rejection_at_30pct_signal_eff"
                ],
                "seen_z_nuis_physics_probe_auc": seen_row["z_nuis_physics_probe_auc"],
                "seen_z_nuis_domain_probe_acc": seen_row["z_nuis_domain_probe_acc"],
                "seen_z_phys_domain_probe_acc": seen_row["z_phys_domain_probe_acc"],
                "seen_score_domain_drift_max": nominal_scores,
                "holdout_physics_auc": weighted_auc(holdout_y_np, holdout_scores, holdout_weights_np),
                "holdout_bkg_rejection_at_30pct_signal_eff": e75c.background_rejection_at_signal_eff(
                    holdout_y_np, holdout_scores, 0.30
                ),
                "holdout_score_drift_vs_nominal": abs(float(holdout_scores.mean()) - float(nominal_val_scores.mean())),
                "holdout_z_nuis_physics_probe_auc": holdout_physics_probe,
                "z_phys_seen_vs_holdout_probe_auc": train_holdout_probe_auc(
                    seen_z_phys, holdout_z_phys, args.probe_epochs, args.seed + 17, device
                ),
                "z_nuis_seen_vs_holdout_probe_auc": train_holdout_probe_auc(
                    seen_z_nuis, holdout_z_nuis, args.probe_epochs, args.seed + 23, device
                ),
            }
        )

    metrics_csv = run_dir / "heldout_systematic_metrics.csv"
    history_csv = run_dir / "training_history.csv"
    e75e.write_csv(metrics_csv, result_rows)
    e75e.write_csv(history_csv, history_rows)
    config = {
        "train_domains": train_domain_names,
        "holdout_domain": holdout_domain,
        "max_events_per_domain": args.max_events_per_domain,
        "total_seen_events": int(len(labels)),
        "holdout_events": int(len(holdout_y_np)),
        "seed": args.seed,
        "epochs": args.epochs,
        "probe_epochs": args.probe_epochs,
        "batch_size": args.batch_size,
        "run_note": "holdout domain is excluded from model training and seen-domain objectives",
        "train_domain_summaries": summaries,
        "holdout_summary": holdout_summary,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    best = min(result_rows[1:], key=lambda row: (row["holdout_z_nuis_physics_probe_auc"], -row["holdout_physics_auc"]))
    report_path = REPORTS / f"e77_toptag_heldout_systematic_audit_{dt.datetime.now():%Y%m%d}_seed{args.seed}.md"
    lines = [
        "# E77 TopTag Held-Out Systematic Audit",
        "",
        f"- run_dir: `{run_dir}`",
        f"- generated_at: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- train_domains: `{', '.join(train_domain_names)}`",
        f"- holdout_domain: `{holdout_domain}`",
        f"- device: `{device}`",
        f"- seen_events: {len(labels)}",
        f"- holdout_events: {len(holdout_y_np)}",
        f"- epochs: {args.epochs}",
        f"- probe_epochs: {args.probe_epochs}",
        "- run note: holdout domain is excluded from model training and seen-domain objectives.",
        "",
        "## Best Held-Out Candidate",
        "",
        (
            f"- {best['candidate']}: holdout physics AUC={e75e.format_float(best['holdout_physics_auc'])}, "
            f"holdout z_nuis physics={e75e.format_float(best['holdout_z_nuis_physics_probe_auc'])}, "
            f"seen z_nuis domain={e75e.format_float(best['seen_z_nuis_domain_probe_acc'])}"
        ),
        "",
        "## Metrics",
        "",
        "| candidate | seen physics AUC | holdout physics AUC | holdout bkg rej | seen z_nuis physics | holdout z_nuis physics | seen z_nuis domain | holdout score drift | z_phys seen/holdout AUC | z_nuis seen/holdout AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result_rows:
        lines.append(
            f"| {row['candidate']} | {e75e.format_float(row['seen_physics_auc'])} | "
            f"{e75e.format_float(row['holdout_physics_auc'])} | "
            f"{e75e.format_float(row['holdout_bkg_rejection_at_30pct_signal_eff'])} | "
            f"{e75e.format_float(row['seen_z_nuis_physics_probe_auc'])} | "
            f"{e75e.format_float(row['holdout_z_nuis_physics_probe_auc'])} | "
            f"{e75e.format_float(row['seen_z_nuis_domain_probe_acc'])} | "
            f"{e75e.format_float(row['holdout_score_drift_vs_nominal'])} | "
            f"{e75e.format_float(row['z_phys_seen_vs_holdout_probe_auc'])} | "
            f"{e75e.format_float(row['z_nuis_seen_vs_holdout_probe_auc'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- This is the first judge-athlete separation audit: the held-out domain is not part of the training objective.",
            "- Useful evidence requires held-out physics AUC to remain competitive while held-out `z_nuis -> physics` leakage falls relative to the shared baseline.",
            "- Seen/holdout probe AUC is diagnostic only; it tests whether a branch carries held-out shift information after training.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "experiment": "E77 TopTag held-out systematic audit",
                "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
                "run_dir": str(run_dir),
                "report": str(report_path),
                "metrics_csv": str(metrics_csv),
                "history_csv": str(history_csv),
                "config": config,
                "best_heldout_candidate": best,
                "metrics": result_rows,
                "status": "done",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "status.txt").write_text(f"status: done\nreport: {report_path}\n", encoding="utf-8")
    print(f"E77 TopTag held-out systematic audit done: {run_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
