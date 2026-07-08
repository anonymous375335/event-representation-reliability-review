#!/usr/bin/env python3
"""E73 domain-design repeat for frozen EveNet H4l split-branch results."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import e68c_cms_h4l_split_branch_disentanglement as e68c
import e72_cms_h4l_evenet_embedding_split_branch as e72
from e66_cms_h4l_readout_smoke import stratified_split, weighted_auc


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_TEACHER_CKPT = Path(
    os.environ.get("EVENET_TEACHER_CKPT", str(ROOT / "data" / "checkpoints" / "teachers" / "evenet_public.ckpt"))
)
DEFAULT_EVENET_REPO = Path(os.environ.get("EVENET_REPO", str(ROOT / "external" / "EveNet_Public")))

DOMAIN_SETS = {
    "pt_fine": [
        "nominal",
        "pt_scale_up_1pct",
        "pt_scale_down_1pct",
        "pt_scale_up_3pct",
        "pt_scale_down_3pct",
    ],
    "angular": [
        "nominal",
        "eta_shift_up_0p02",
        "eta_shift_down_0p02",
        "phi_shift_up_0p02",
        "phi_shift_down_0p02",
    ],
    "mixed_visible": [
        "nominal",
        "pt_up_1pct_eta_up_0p01",
        "pt_down_1pct_eta_down_0p01",
        "pt_up_3pct_phi_up_0p02",
        "pt_down_3pct_phi_down_0p02",
    ],
}


def transform_visible_features(features: np.ndarray, domain: str) -> np.ndarray:
    shifted = features.copy()
    if domain == "nominal":
        return shifted
    if domain == "pt_scale_up_1pct":
        shifted[:, :, 0] *= 1.01
    elif domain == "pt_scale_down_1pct":
        shifted[:, :, 0] *= 0.99
    elif domain == "pt_scale_up_3pct":
        shifted[:, :, 0] *= 1.03
    elif domain == "pt_scale_down_3pct":
        shifted[:, :, 0] *= 0.97
    elif domain == "eta_shift_up_0p02":
        shifted[:, :, 1] += 0.02
    elif domain == "eta_shift_down_0p02":
        shifted[:, :, 1] -= 0.02
    elif domain == "phi_shift_up_0p02":
        shifted[:, :, 2] += 0.02
    elif domain == "phi_shift_down_0p02":
        shifted[:, :, 2] -= 0.02
    elif domain == "pt_up_1pct_eta_up_0p01":
        shifted[:, :, 0] *= 1.01
        shifted[:, :, 1] += 0.01
    elif domain == "pt_down_1pct_eta_down_0p01":
        shifted[:, :, 0] *= 0.99
        shifted[:, :, 1] -= 0.01
    elif domain == "pt_up_3pct_phi_up_0p02":
        shifted[:, :, 0] *= 1.03
        shifted[:, :, 2] += 0.02
    elif domain == "pt_down_3pct_phi_down_0p02":
        shifted[:, :, 0] *= 0.97
        shifted[:, :, 2] -= 0.02
    else:
        raise ValueError(f"unknown domain {domain}")
    return shifted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-npz", default=str(ROOT / "data_processed" / "cms_h4l_e65" / "cms_h4l_mc_candidates_e65.npz"))
    parser.add_argument("--teacher-ckpt", default=str(DEFAULT_TEACHER_CKPT))
    parser.add_argument("--evenet-repo", default=str(DEFAULT_EVENET_REPO))
    parser.add_argument("--domain-set", choices=sorted(DOMAIN_SETS), required=True)
    parser.add_argument("--max-events", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--embedding-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--adv-lambda", type=float, default=0.5)
    parser.add_argument("--orth-lambda", type=float, default=0.25)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    domain_names = DOMAIN_SETS[args.domain_set]
    e72.DOMAIN_NAMES = domain_names.copy()
    e72.transform_visible_features = transform_visible_features
    e68c.DOMAIN_NAMES = domain_names.copy()

    e72.set_seed(args.seed)
    REPORTS.mkdir(parents=True, exist_ok=True)
    run_dir = e72.create_run_dir(f"e73-cms-h4l-evenet-domain-design-{args.domain_set}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features, valid_masks, labels, conditions, conditions_mask = e72.load_h4l_subset(
        Path(args.tensor_npz), args.max_events, args.seed
    )
    train_idx, val_idx = stratified_split(labels, val_ratio=args.val_ratio, seed=args.seed)
    teacher = e72.build_teacher(
        features,
        labels,
        conditions,
        Path(args.teacher_ckpt),
        Path(args.evenet_repo),
        device,
    )

    train_x_np, train_y_np, train_domain_np, _, train_weights_np, train_raw_domains = e72.collect_embeddings_for_indices(
        teacher, features, valid_masks, labels, conditions, conditions_mask, train_idx, args.embedding_batch_size, device
    )
    val_x_np, val_y_np, val_domain_np, val_masses_np, val_weights_np, val_raw_domains = e72.collect_embeddings_for_indices(
        teacher, features, valid_masks, labels, conditions, conditions_mask, val_idx, args.embedding_batch_size, device
    )
    train_x_np, val_x_np = e72.standardize(train_x_np, val_x_np)

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
        )
        print(f"[fit-done] {mode} {dt.datetime.now().isoformat(timespec='seconds')}", flush=True)
        history_rows.extend(history)
        val_scores, val_z_phys, val_z_nuis, val_nuis_logits = e68c.embed_and_score(model, val_x, device)
        _, train_z_phys, train_z_nuis, _ = e68c.embed_and_score(model, train_x, device)
        rows.append(
            {
                "mode": mode,
                "physics_auc": weighted_auc(val_y_np.astype(np.int64), val_scores, val_weights_np),
                "score_m4l_corr": e68c.corrcoef(val_scores, val_masses_np),
                "score_domain_drift_max": e68c.domain_score_drift(val_scores, val_raw_domains),
                "nuisance_head_acc": float((val_nuis_logits.argmax(axis=1) == val_raw_domains).mean()),
                "z_phys_domain_probe_acc": e68c.train_domain_probe(
                    train_z_phys, train_raw_domains, val_z_phys, val_raw_domains, args.probe_epochs, device
                ),
                "z_nuis_domain_probe_acc": e68c.train_domain_probe(
                    train_z_nuis, train_raw_domains, val_z_nuis, val_raw_domains, args.probe_epochs, device
                ),
                "z_nuis_physics_probe_auc": e68c.train_physics_probe_auc(
                    train_z_nuis,
                    train_y_np.astype(np.int64),
                    val_z_nuis,
                    val_y_np.astype(np.int64),
                    val_weights_np,
                    args.probe_epochs,
                    device,
                ),
            }
        )

    metrics_csv = run_dir / "domain_design_metrics.csv"
    history_csv = run_dir / "training_history.csv"
    e68c.write_csv(metrics_csv, rows)
    e68c.write_csv(history_csv, history_rows)

    event_tag = "fullmc" if args.max_events == 0 else f"{args.max_events}"
    report_path = REPORTS / f"e73_cms_h4l_evenet_domain_design_{args.domain_set}_{dt.datetime.now():%Y%m%d}_seed{args.seed}_{event_tag}.md"
    lines = [
        "# E73 CMS H4l EveNet Domain-Design Repeat",
        "",
        f"- run_dir: `{run_dir}`",
        f"- generated_at: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- domain_set: `{args.domain_set}`",
        f"- tensor_npz: `{args.tensor_npz}`",
        f"- teacher_ckpt: `{args.teacher_ckpt}`",
        f"- device: `{device}`",
        f"- source_events: {len(labels)}",
        f"- expanded_train_events: {len(train_y_np)}",
        f"- expanded_val_events: {len(val_y_np)}",
        f"- domains: `{', '.join(domain_names)}`",
        f"- random_domain_baseline: {1 / len(domain_names):.4f}",
        f"- epochs: {args.epochs}",
        f"- probe_epochs: {args.probe_epochs}",
        f"- adv_lambda: {args.adv_lambda}",
        f"- orth_lambda: {args.orth_lambda}",
        "- dataset note: frozen EveNet embedding bridge with visible-schema domain variants",
        "",
        "| mode | physics AUC | score-m4l corr | score domain drift | nuisance head acc | z_phys domain probe | z_nuis domain probe | z_nuis physics probe AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['physics_auc']:.4f} | {row['score_m4l_corr']:.4f} | "
            f"{row['score_domain_drift_max']:.4f} | {row['nuisance_head_acc']:.4f} | "
            f"{row['z_phys_domain_probe_acc']:.4f} | {row['z_nuis_domain_probe_acc']:.4f} | "
            f"{row['z_nuis_physics_probe_auc']:.4f} |"
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    metrics = {
        "experiment": "E73 CMS H4l EveNet domain-design repeat",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "domain_set": args.domain_set,
        "report": str(report_path),
        "metrics_csv": str(metrics_csv),
        "history_csv": str(history_csv),
        "domain_names": domain_names,
        "random_domain_baseline": 1 / len(domain_names),
        "metrics": rows,
        "status": "done",
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "status.txt").write_text(f"status: done\nreport: {report_path}\n", encoding="utf-8")
    print(f"E73 domain-design repeat done: {run_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
