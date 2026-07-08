#!/usr/bin/env python3
"""E72 split-branch systematic test on frozen EveNet embeddings for CMS H4l."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import e68c_cms_h4l_split_branch_disentanglement as e68c
from e66_cms_h4l_readout_smoke import stratified_split, weighted_auc
from e71_cms_h4l_evenet_bridge_feasibility import (
    adapt_h4l_to_evenet_public_schema,
    build_evenet_passthrough_normalization,
    EveNetOfficialTeacherAdapter,
)


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
DEFAULT_TEACHER_CKPT = Path(
    os.environ.get("EVENET_TEACHER_CKPT", str(ROOT / "data" / "checkpoints" / "teachers" / "evenet_public.ckpt"))
)
DEFAULT_EVENET_REPO = Path(os.environ.get("EVENET_REPO", str(ROOT / "external" / "EveNet_Public")))

DOMAIN_NAMES = [
    "nominal",
    "pt_scale_up_3pct",
    "pt_scale_down_3pct",
    "eta_shift_up_0p02",
    "eta_shift_down_0p02",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def transform_visible_features(features: np.ndarray, domain: str) -> np.ndarray:
    shifted = features.copy()
    if domain == "nominal":
        return shifted
    if domain == "pt_scale_up_3pct":
        shifted[:, :, 0] *= 1.03
    elif domain == "pt_scale_down_3pct":
        shifted[:, :, 0] *= 0.97
    elif domain == "eta_shift_up_0p02":
        shifted[:, :, 1] += 0.02
    elif domain == "eta_shift_down_0p02":
        shifted[:, :, 1] -= 0.02
    else:
        raise ValueError(f"unknown domain {domain}")
    return shifted


def load_h4l_subset(path: Path, max_events: int, seed: int):
    with np.load(path, allow_pickle=False) as payload:
        features = payload["features"].astype(np.float32)
        valid_masks = payload["valid_masks"].astype(np.float32)
        labels = payload["labels"].astype(np.int64)
        conditions = payload["conditions"].astype(np.float32)
        conditions_mask = payload["conditions_mask"].astype(np.float32)

    if max_events > 0 and max_events < len(labels):
        rng = np.random.default_rng(seed)
        keep_parts = []
        for label in sorted(np.unique(labels).tolist()):
            label_idx = np.flatnonzero(labels == label)
            take = max(1, int(round(max_events * len(label_idx) / len(labels))))
            keep_parts.append(rng.choice(label_idx, size=min(take, len(label_idx)), replace=False))
        keep = np.concatenate(keep_parts)
        rng.shuffle(keep)
        features = features[keep]
        valid_masks = valid_masks[keep]
        labels = labels[keep]
        conditions = conditions[keep]
        conditions_mask = conditions_mask[keep]

    return features, valid_masks, labels, conditions, conditions_mask


def build_teacher(features, labels, conditions, teacher_ckpt: Path, evenet_repo: Path, device: torch.device):
    evenet_features, evenet_conditions = adapt_h4l_to_evenet_public_schema(features, conditions)
    normalization = build_evenet_passthrough_normalization(
        feature_dim=evenet_features.shape[-1],
        condition_dim=evenet_conditions.shape[-1],
        labels=torch.as_tensor(labels),
        device=device,
    )
    teacher = EveNetOfficialTeacherAdapter(
        checkpoint_path=teacher_ckpt,
        normalization_dict=normalization,
        device=device,
        repo_root=evenet_repo,
    ).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def collect_embeddings_for_indices(
    teacher,
    features: np.ndarray,
    valid_masks: np.ndarray,
    labels: np.ndarray,
    conditions: np.ndarray,
    conditions_mask: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
):
    embedding_parts = []
    label_parts = []
    domain_parts = []
    mass_parts = []
    weight_parts = []
    raw_domain_parts = []

    for domain_id, domain in enumerate(DOMAIN_NAMES):
        shifted = transform_visible_features(features[indices], domain)
        evenet_features, evenet_conditions = adapt_h4l_to_evenet_public_schema(shifted, conditions[indices])
        ds = TensorDataset(
            torch.as_tensor(evenet_features, dtype=torch.float32),
            torch.as_tensor(valid_masks[indices], dtype=torch.float32),
            torch.as_tensor(evenet_conditions, dtype=torch.float32),
            torch.as_tensor(conditions_mask[indices], dtype=torch.float32),
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        domain_embeddings = []
        with torch.no_grad():
            for batch_features, batch_masks, batch_conditions, batch_conditions_mask in loader:
                embedding, _ = teacher(
                    batch_features.to(device),
                    batch_masks.to(device),
                    batch_conditions.to(device),
                    batch_conditions_mask.to(device),
                )
                domain_embeddings.append(embedding.detach().cpu())
        embedding_parts.append(torch.cat(domain_embeddings, dim=0).numpy().astype(np.float32))
        label_parts.append(labels[indices].astype(np.float32))
        domain_parts.append(np.full(len(indices), domain_id, dtype=np.int64))
        mass_parts.append(conditions[indices, 0].astype(np.float32))
        weight_parts.append(conditions[indices, 4].astype(np.float32))
        raw_domain_parts.append(np.full(len(indices), domain_id, dtype=np.int64))

    return (
        np.concatenate(embedding_parts),
        np.concatenate(label_parts),
        np.concatenate(domain_parts),
        np.concatenate(mass_parts),
        np.concatenate(weight_parts),
        np.concatenate(raw_domain_parts),
    )


def standardize(train_x: np.ndarray, val_x: np.ndarray):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train_x - mean).astype(np.float32) / std.astype(np.float32), (val_x - mean).astype(np.float32) / std.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-npz", default=str(ROOT / "data_processed" / "cms_h4l_e65" / "cms_h4l_mc_candidates_e65.npz"))
    parser.add_argument("--teacher-ckpt", default=str(DEFAULT_TEACHER_CKPT))
    parser.add_argument("--evenet-repo", default=str(DEFAULT_EVENET_REPO))
    parser.add_argument("--max-events", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--embedding-batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--adv-lambda", type=float, default=0.5)
    parser.add_argument("--orth-lambda", type=float, default=0.25)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    set_seed(args.seed)
    e68c.DOMAIN_NAMES = DOMAIN_NAMES.copy()
    REPORTS.mkdir(parents=True, exist_ok=True)
    run_dir = create_run_dir("e72-cms-h4l-evenet-embedding-split-branch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features, valid_masks, labels, conditions, conditions_mask = load_h4l_subset(Path(args.tensor_npz), args.max_events, args.seed)
    train_idx, val_idx = stratified_split(labels, val_ratio=args.val_ratio, seed=args.seed)
    teacher = build_teacher(
        features,
        labels,
        conditions,
        Path(args.teacher_ckpt),
        Path(args.evenet_repo),
        device,
    )

    train_x_np, train_y_np, train_domain_np, _, train_weights_np, train_raw_domains = collect_embeddings_for_indices(
        teacher, features, valid_masks, labels, conditions, conditions_mask, train_idx, args.embedding_batch_size, device
    )
    val_x_np, val_y_np, val_domain_np, val_masses_np, val_weights_np, val_raw_domains = collect_embeddings_for_indices(
        teacher, features, valid_masks, labels, conditions, conditions_mask, val_idx, args.embedding_batch_size, device
    )
    train_x_np, val_x_np = standardize(train_x_np, val_x_np)

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

    metrics_csv = run_dir / "evenet_embedding_split_branch_metrics.csv"
    history_csv = run_dir / "training_history.csv"
    e68c.write_csv(metrics_csv, rows)
    e68c.write_csv(history_csv, history_rows)

    event_tag = "fullmc" if args.max_events == 0 else f"{args.max_events}"
    report_path = REPORTS / f"e72_cms_h4l_evenet_embedding_split_branch_{dt.datetime.now():%Y%m%d}_seed{args.seed}_{event_tag}.md"
    lines = [
        "# E72 CMS H4l EveNet Embedding Split-Branch",
        "",
        f"- run_dir: `{run_dir}`",
        f"- generated_at: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- tensor_npz: `{args.tensor_npz}`",
        f"- teacher_ckpt: `{args.teacher_ckpt}`",
        f"- device: `{device}`",
        f"- source_events: {len(labels)}",
        f"- expanded_train_events: {len(train_y_np)}",
        f"- expanded_val_events: {len(val_y_np)}",
        f"- domains: `{', '.join(DOMAIN_NAMES)}`",
        f"- random_domain_baseline: {1 / len(DOMAIN_NAMES):.4f}",
        f"- epochs: {args.epochs}",
        f"- probe_epochs: {args.probe_epochs}",
        f"- adv_lambda: {args.adv_lambda}",
        f"- orth_lambda: {args.orth_lambda}",
        "- dataset note: frozen EveNet embedding bridge with kinematic visible-domain variants",
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
        "experiment": "E72 CMS H4l EveNet embedding split-branch",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "report": str(report_path),
        "metrics_csv": str(metrics_csv),
        "history_csv": str(history_csv),
        "domain_names": DOMAIN_NAMES,
        "random_domain_baseline": 1 / len(DOMAIN_NAMES),
        "metrics": rows,
        "status": "done",
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "status.txt").write_text(f"status: done\nreport: {report_path}\n", encoding="utf-8")
    print(f"E72 EveNet embedding split-branch done: {run_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
