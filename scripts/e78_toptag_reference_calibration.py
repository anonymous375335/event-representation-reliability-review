#!/usr/bin/env python3
"""E78 calibration on the standard Top Quark Tagging Reference Dataset."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from e66_cms_h4l_readout_smoke import weighted_auc
import e75c_toptag_branch_protocol_smoke as e75c
import e75e_toptag_systematic_family_scaleup as e75e
import e76b_toptag_constituent_encoder as e76b


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
DATA_RAW_ROOT = Path(os.environ.get("YEAR1_DATA_RAW", str(ROOT / "data_raw")))
DATA_ROOT = Path(os.environ.get("YEAR1_TOPTAG_REFERENCE_ROOT", str(DATA_RAW_ROOT / "toptag_reference_2603256")))


def create_run_dir(prefix: str) -> Path:
    return e75e.create_run_dir(prefix)


def infer_h5_schema(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        keys = list(handle.keys())
        shapes = {key: getattr(handle[key], "shape", None) for key in keys}
    return {"keys": keys, "shapes": {key: tuple(value) if value is not None else None for key, value in shapes.items()}}


def read_direct_arrays(path: Path, max_events: int | None) -> tuple[np.ndarray, np.ndarray] | None:
    with h5py.File(path, "r") as handle:
        keys = set(handle.keys())
        x_key = next((key for key in ["X", "x", "features", "data", "particles"] if key in keys), None)
        y_key = next((key for key in ["y", "Y", "labels", "label", "is_signal_new"] if key in keys), None)
        if x_key is None or y_key is None:
            return None
        stop = min(max_events or handle[x_key].shape[0], handle[x_key].shape[0])
        features = handle[x_key][:stop].astype(np.float32)
        labels = handle[y_key][:stop].astype(np.int64).reshape(-1)
    if features.ndim == 2 and features.shape[1] % 4 == 0:
        features = features.reshape(features.shape[0], features.shape[1] // 4, 4)
    if features.ndim != 3:
        raise ValueError(f"unsupported direct feature shape {features.shape}")
    return features, labels


def read_table_arrays(path: Path, max_events: int | None, max_constituents: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    with h5py.File(path, "r") as handle:
        table_key = "table/table" if "table" in handle and "table" in handle["table"] else None
        if table_key is None:
            raise ValueError(f"cannot infer PyTables table dataset in {path}: {list(handle.keys())}")
        table = handle[table_key]
        stop = min(max_events or table.shape[0], table.shape[0])
        block = table[:stop]
    if "values_block_0" not in block.dtype.names or "values_block_1" not in block.dtype.names:
        raise ValueError(f"unsupported PyTables block dtype {block.dtype}")
    flat = block["values_block_0"].astype(np.float32)
    if flat.shape[1] % 4 != 0:
        raise ValueError(f"expected flattened four-vectors, got {flat.shape}")
    features = flat.reshape(flat.shape[0], flat.shape[1] // 4, 4)
    label_block = block["values_block_1"]
    labels = label_block[:, -1].astype(np.int64)
    features = features[:, :max_constituents, :]
    return features, labels


def load_reference(path: Path, max_events: int | None, max_constituents: int) -> tuple[np.ndarray, np.ndarray, dict]:
    direct = read_direct_arrays(path, max_events)
    if direct is None:
        features, labels = read_table_arrays(path, max_events, max_constituents)
    else:
        features, labels = direct
    features = features[:, :max_constituents, :].astype(np.float32)
    features = np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    labels = labels.astype(np.int64)
    summary = {
        "path": str(path),
        "events": int(len(labels)),
        "feature_shape": list(features.shape),
        "label_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        "schema": infer_h5_schema(path),
    }
    return features, labels, summary


def fourvec_to_constituents(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e = features[:, :, 0]
    px = features[:, :, 1]
    py = features[:, :, 2]
    pz = features[:, :, 3]
    pt = np.sqrt(px * px + py * py)
    p = np.sqrt(px * px + py * py + pz * pz)
    eta = 0.5 * np.log(np.maximum((p + pz) / np.maximum(p - pz, 1e-6), 1e-6))
    phi = np.arctan2(py, px)
    mask = ((pt > 0) & (e > 0)).astype(np.float32)
    sum_pt = np.maximum((pt * mask).sum(axis=1, keepdims=True), 1e-6)
    eta0 = (eta * pt * mask).sum(axis=1, keepdims=True) / sum_pt
    sin_phi0 = (np.sin(phi) * pt * mask).sum(axis=1, keepdims=True) / sum_pt
    cos_phi0 = (np.cos(phi) * pt * mask).sum(axis=1, keepdims=True) / sum_pt
    phi0 = np.arctan2(sin_phi0, cos_phi0)
    constituents = np.stack(
        [
            np.log1p(np.maximum(pt, 0.0)),
            eta - eta0,
            e75c.wrap_delta_phi(phi - phi0),
            np.log1p(np.maximum(e, 0.0)),
        ],
        axis=-1,
    ).astype(np.float32)
    constituents = np.nan_to_num(constituents, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    constituents *= mask[:, :, None]
    return constituents, mask


def standardize_reference(
    train_const: np.ndarray,
    val_const: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = train_mask > 0
    mean = train_const[valid].mean(axis=0, keepdims=True)
    std = train_const[valid].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (
        ((train_const - mean) / std).astype(np.float32) * train_mask[:, :, None],
        ((val_const - mean) / std).astype(np.float32) * val_mask[:, :, None],
    )


class ReferenceTagger(nn.Module):
    def __init__(self, point_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.point_net = nn.Sequential(
            nn.Linear(4, point_dim),
            nn.LayerNorm(point_dim),
            nn.GELU(),
            nn.Linear(point_dim, point_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(point_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, constituents: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        point = self.point_net(constituents) * mask.unsqueeze(-1)
        denom = mask.unsqueeze(-1).sum(dim=1).clamp_min(1.0)
        pooled_mean = point.sum(dim=1) / denom
        pooled_max = point.masked_fill(mask.unsqueeze(-1) == 0, -1e4).max(dim=1).values
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))
        return self.head(torch.cat([pooled_mean, pooled_max], dim=1)).squeeze(-1)


def train_reference(
    train_loader: DataLoader,
    val_const: torch.Tensor,
    val_mask: torch.Tensor,
    val_y: torch.Tensor,
    pos_weight: float,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[ReferenceTagger, list[dict]]:
    model = ReferenceTagger().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    val_const = val_const.to(device)
    val_mask = val_mask.to(device)
    val_y = val_y.to(device)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch_const, batch_mask, batch_y in train_loader:
            batch_const = batch_const.to(device)
            batch_mask = batch_mask.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_const, batch_mask)
            loss = F.binary_cross_entropy_with_logits(logits, batch_y, pos_weight=pos_weight_tensor)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu()) * batch_y.shape[0]
            total_count += batch_y.shape[0]
        model.eval()
        with torch.no_grad():
            val_logits = model(val_const, val_mask)
            val_loss = F.binary_cross_entropy_with_logits(val_logits, val_y, pos_weight=pos_weight_tensor)
        history.append({"epoch": epoch, "train_loss": total_loss / max(total_count, 1), "val_loss": float(val_loss.cpu())})
    return model, history


def score_model(model: ReferenceTagger, const: torch.Tensor, mask: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    scores = []
    loader = DataLoader(TensorDataset(const, mask), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_const, batch_mask in loader:
            logits = model(batch_const.to(device), batch_mask.to(device))
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--train-events", type=int, default=300000)
    parser.add_argument("--val-events", type=int, default=150000)
    parser.add_argument("--max-constituents", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    args = parser.parse_args()

    run_dir = create_run_dir("e78-toptag-reference-calibration")
    REPORTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_features, train_labels, train_summary = load_reference(
        args.data_dir / "train.h5", args.train_events, args.max_constituents
    )
    val_features, val_labels, val_summary = load_reference(args.data_dir / "val.h5", args.val_events, args.max_constituents)
    train_const, train_mask = fourvec_to_constituents(train_features)
    val_const, val_mask = fourvec_to_constituents(val_features)
    train_const, val_const = standardize_reference(train_const, val_const, train_mask, val_mask)
    train_y = train_labels.astype(np.float32)
    val_y = val_labels.astype(np.int64)
    train_const_t = torch.tensor(train_const, dtype=torch.float32)
    train_mask_t = torch.tensor(train_mask, dtype=torch.float32)
    train_y_t = torch.tensor(train_y, dtype=torch.float32)
    val_const_t = torch.tensor(val_const, dtype=torch.float32)
    val_mask_t = torch.tensor(val_mask, dtype=torch.float32)
    val_y_t = torch.tensor(val_y.astype(np.float32), dtype=torch.float32)
    loader = DataLoader(TensorDataset(train_const_t, train_mask_t, train_y_t), batch_size=args.batch_size, shuffle=True)
    pos = max(float((train_y == 1).sum()), 1.0)
    neg = max(float((train_y == 0).sum()), 1.0)
    model, history = train_reference(
        loader,
        val_const_t,
        val_mask_t,
        val_y_t,
        neg / pos,
        args.epochs,
        args.learning_rate,
        device,
    )
    scores = score_model(model, val_const_t, val_mask_t, args.batch_size, device)
    metrics = {
        "physics_auc": weighted_auc(val_y, scores, np.ones_like(val_y, dtype=np.float32)),
        "background_rejection_at_30pct_signal_eff": e75c.background_rejection_at_signal_eff(val_y, scores, 0.30),
        "accuracy_at_0p5": float(((scores >= 0.5).astype(np.int64) == val_y).mean()),
    }
    e75e.write_csv(run_dir / "training_history.csv", history)
    e75e.write_csv(run_dir / "reference_calibration_metrics.csv", [{"candidate": "reference_constituent_baseline", **metrics}])
    config = {
        "data_dir": str(args.data_dir),
        "train_events": args.train_events,
        "val_events": args.val_events,
        "max_constituents": args.max_constituents,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_summary": train_summary,
        "val_summary": val_summary,
        "run_note": "train/val calibration subset for reference score behavior",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report_path = REPORTS / f"e78_toptag_reference_calibration_{dt.datetime.now():%Y%m%d}_seed{args.seed}.md"
    lines = [
        "# E78 TopTag Reference Dataset Calibration",
        "",
        f"- run_dir: `{run_dir}`",
        f"- generated_at: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- data_dir: `{args.data_dir}`",
        "- dataset: Top Quark Tagging Reference Dataset, DOI `10.5281/zenodo.2603256`",
        f"- device: `{device}`",
        f"- train_events_used: {len(train_labels)}",
        f"- val_events_used: {len(val_labels)}",
        f"- max_constituents: {args.max_constituents}",
        f"- epochs: {args.epochs}",
        "- run note: train/val calibration subset for reference score behavior.",
        "",
        "## Metrics",
        "",
        "| model | AUC | bkg rejection @30% sig eff | accuracy @0.5 |",
        "|---|---:|---:|---:|",
        (
            f"| reference_constituent_baseline | {metrics['physics_auc']:.4f} | "
            f"{metrics['background_rejection_at_30pct_signal_eff']:.4f} | {metrics['accuracy_at_0p5']:.4f} |"
        ),
        "",
        "## Interpretation",
        "",
        "- This calibrates whether the small constituent encoder used in E76/E77 behaves like a plausible top tagger on the standard reference dataset.",
        "- This benchmark reports reference score behavior without systematic/domain labels.",
        "- It should be used only to contextualize model strength, while E76/E77 remain the systematic-aware transfer evidence.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "experiment": "E78 TopTag reference calibration",
                "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
                "run_dir": str(run_dir),
                "report": str(report_path),
                "metrics": metrics,
                "config": config,
                "status": "done",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "status.txt").write_text(f"status: done\nreport: {report_path}\n", encoding="utf-8")
    print(f"E78 TopTag reference calibration done: {run_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
