#!/usr/bin/env python3
"""E66 smoke readout on the CMS H4l MC tensor export."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"


class MLPReadout(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_split(labels: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    for label in sorted(np.unique(labels).tolist()):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * val_ratio)))
        val_parts.append(indices[:val_count])
        train_parts.append(indices[val_count:])
    train = np.concatenate(train_parts)
    val = np.concatenate(val_parts)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def weighted_auc(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float | None:
    pos = labels == 1
    neg = labels == 0
    total_pos = float(weights[pos].sum())
    total_neg = float(weights[neg].sum())
    if total_pos <= 0 or total_neg <= 0:
        return None
    order = np.argsort(scores)
    labels = labels[order]
    scores = scores[order]
    weights = weights[order]
    numerator = 0.0
    cum_neg = 0.0
    index = 0
    while index < len(scores):
        score = scores[index]
        tie_pos = 0.0
        tie_neg = 0.0
        while index < len(scores) and scores[index] == score:
            if labels[index] == 1:
                tie_pos += float(weights[index])
            else:
                tie_neg += float(weights[index])
            index += 1
        numerator += tie_pos * (cum_neg + 0.5 * tie_neg)
        cum_neg += tie_neg
    return numerator / (total_pos * total_neg)


def corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def tensor_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x.pow(2).mean() + 1e-8) * (y.pow(2).mean() + 1e-8))
    return (x * y).mean() / denom


def train_model(
    name: str,
    train_loader: DataLoader,
    val_tensor: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    input_dim: int,
    pos_weight: float,
    mass_penalty: float,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[MLPReadout, list[dict]]:
    model = MLPReadout(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = []
    val_x, val_y, val_mass = [item.to(device) for item in val_tensor]
    bce_pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"loss": 0.0, "bce": 0.0, "mass_corr_penalty": 0.0, "count": 0}
        for batch_x, batch_y, batch_mass in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_mass = batch_mass.to(device)
            logits = model(batch_x)
            bce = F.binary_cross_entropy_with_logits(logits, batch_y, pos_weight=bce_pos_weight)
            penalty = tensor_corr(logits, batch_mass).pow(2)
            loss = bce + mass_penalty * penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count = batch_y.shape[0]
            totals["loss"] += float(loss.detach().cpu()) * count
            totals["bce"] += float(bce.detach().cpu()) * count
            totals["mass_corr_penalty"] += float(penalty.detach().cpu()) * count
            totals["count"] += count

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_bce = F.binary_cross_entropy_with_logits(val_logits, val_y, pos_weight=bce_pos_weight)
            val_corr = tensor_corr(val_logits, val_mass)
        count = max(totals.pop("count"), 1)
        history.append(
            {
                "model": name,
                "epoch": epoch,
                "train_loss": totals["loss"] / count,
                "train_bce": totals["bce"] / count,
                "train_mass_corr_penalty": totals["mass_corr_penalty"] / count,
                "val_bce": float(val_bce.detach().cpu()),
                "val_score_mass_corr": float(val_corr.detach().cpu()),
            }
        )
    return model, history


def evaluate_model(model: MLPReadout, x: torch.Tensor, labels: np.ndarray, masses: np.ndarray, weights: np.ndarray, device: torch.device) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device)).detach().cpu().numpy()
    scores = 1.0 / (1.0 + np.exp(-logits))
    pred = scores >= 0.5
    accuracy = float((pred == labels).mean())
    auc = weighted_auc(labels, scores, weights)
    score_mass_corr = corrcoef(scores, masses)
    background = labels == 0
    if background.any():
        background_scores = scores[background]
        background_masses = masses[background]
        threshold = float(np.quantile(background_scores, 0.8))
        selected_background = background_scores >= threshold
        bkg_window_fraction_top20 = float(((115 <= background_masses[selected_background]) & (background_masses[selected_background] <= 135)).mean())
        bkg_window_fraction_all = float(((115 <= background_masses) & (background_masses <= 135)).mean())
        bkg_score_mass_corr = corrcoef(background_scores, background_masses)
    else:
        bkg_window_fraction_top20 = None
        bkg_window_fraction_all = None
        bkg_score_mass_corr = None
    return {
        "accuracy_at_0p5": accuracy,
        "weighted_auc": auc,
        "score_mass_corr": score_mass_corr,
        "background_score_mass_corr": bkg_score_mass_corr,
        "background_higgs_window_fraction_all": bkg_window_fraction_all,
        "background_higgs_window_fraction_top20_score": bkg_window_fraction_top20,
    }


def write_history(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-npz", required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--mass-penalty", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_path = Path(args.tensor_npz)
    run_dir = create_run_dir("e66-cms-h4l-readout-smoke")
    REPORTS.mkdir(parents=True, exist_ok=True)

    with np.load(tensor_path, allow_pickle=False) as payload:
        features = payload["features"].astype(np.float32)
        labels = payload["labels"].astype(np.int64)
        subprocess_ids = payload["subprocess_ids"].astype(np.int64)
        conditions = payload["conditions"].astype(np.float32)

    masses = conditions[:, 0].astype(np.float32)
    weights = conditions[:, 4].astype(np.float32)
    x = features.reshape(features.shape[0], -1)
    train_idx, val_idx = stratified_split(labels, val_ratio=args.val_ratio, seed=args.seed)
    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = x[train_idx].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x = (x - mean) / std
    mass_mean = masses[train_idx].mean()
    mass_std = masses[train_idx].std()
    mass_std = mass_std if mass_std > 1e-6 else 1.0
    mass_z = (masses - mass_mean) / mass_std

    train_x = torch.tensor(x[train_idx], dtype=torch.float32)
    train_y = torch.tensor(labels[train_idx], dtype=torch.float32)
    train_mass = torch.tensor(mass_z[train_idx], dtype=torch.float32)
    val_x = torch.tensor(x[val_idx], dtype=torch.float32)
    val_y = torch.tensor(labels[val_idx], dtype=torch.float32)
    val_mass = torch.tensor(mass_z[val_idx], dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_x, train_y, train_mass), batch_size=args.batch_size, shuffle=True)
    pos_count = max(float((labels[train_idx] == 1).sum()), 1.0)
    neg_count = max(float((labels[train_idx] == 0).sum()), 1.0)
    pos_weight = neg_count / pos_count

    baseline, baseline_history = train_model(
        name="baseline_mlp",
        train_loader=train_loader,
        val_tensor=(val_x, val_y, val_mass),
        input_dim=x.shape[1],
        pos_weight=pos_weight,
        mass_penalty=0.0,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
    )
    decorrelated, decorrelated_history = train_model(
        name="mass_penalty_mlp",
        train_loader=train_loader,
        val_tensor=(val_x, val_y, val_mass),
        input_dim=x.shape[1],
        pos_weight=pos_weight,
        mass_penalty=args.mass_penalty,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
    )

    val_labels = labels[val_idx]
    val_masses = masses[val_idx]
    val_weights = weights[val_idx]
    metrics = {
        "baseline_mlp": evaluate_model(baseline, val_x, val_labels, val_masses, val_weights, device),
        "mass_penalty_mlp": evaluate_model(decorrelated, val_x, val_labels, val_masses, val_weights, device),
    }

    history_rows = baseline_history + decorrelated_history
    history_csv = run_dir / "training_history.csv"
    write_history(history_csv, history_rows)
    metrics_path = run_dir / "metrics.json"
    payload = {
        "experiment": "E66 CMS H4l readout smoke",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "tensor_npz": str(tensor_path),
        "run_dir": str(run_dir),
        "device": str(device),
        "num_examples": int(len(labels)),
        "train_examples": int(len(train_idx)),
        "val_examples": int(len(val_idx)),
        "input_dim": int(x.shape[1]),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "mass_penalty": args.mass_penalty,
        "pos_weight": pos_weight,
        "label_counts": {str(int(v)): int((labels == v).sum()) for v in np.unique(labels)},
        "subprocess_counts": {str(int(v)): int((subprocess_ids == v).sum()) for v in np.unique(subprocess_ids)},
        "metrics": metrics,
        "history_csv": str(history_csv),
        "status": "done",
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = REPORTS / f"e66_cms_h4l_readout_smoke_{dt.datetime.now():%Y%m%d}.md"
    lines = [
        "# E66 CMS H4l Readout Smoke",
        "",
        f"- run_dir: `{run_dir}`",
        f"- generated_at: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- tensor_npz: `{tensor_path}`",
        f"- device: `{device}`",
        f"- train_examples: {len(train_idx)}",
        f"- val_examples: {len(val_idx)}",
        f"- epochs: {args.epochs}",
        f"- mass_penalty: {args.mass_penalty}",
        "",
        "| model | weighted AUC | accuracy@0.5 | score-m4l corr | bkg score-m4l corr | bkg H-window frac all | bkg H-window frac top20 score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in metrics.items():
        lines.append(
            f"| {name} | {format_float(values['weighted_auc'])} | {format_float(values['accuracy_at_0p5'])} | "
            f"{format_float(values['score_mass_corr'])} | {format_float(values['background_score_mass_corr'])} | "
            f"{format_float(values['background_higgs_window_fraction_all'])} | "
            f"{format_float(values['background_higgs_window_fraction_top20_score'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Smoke test of the E65 tensor adapter and readout path.",
            "- The mass-penalty readout is a diagnostic candidate for reducing score-mass coupling.",
            "- A useful next step is to run multi-seed and compare against the conventional `m4l` baseline before manuscript integration.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "status.txt").write_text(f"status: done\nreport: {report_path}\n", encoding="utf-8")
    print(f"E66 readout smoke done: {run_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
