#!/usr/bin/env python3
"""Split-branch H4l MC systematic-disentanglement smoke."""

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

from e66_cms_h4l_readout_smoke import stratified_split, weighted_auc


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"

CORE_DOMAIN_NAMES = [
    "nominal",
    "pt_scale_up_3pct",
    "pt_scale_down_3pct",
    "iso_scale_up_20pct",
    "impact_sig_scale_up_20pct",
]

EXTENDED_DOMAIN_NAMES = CORE_DOMAIN_NAMES + [
    "pt_up_3pct_iso_up_20pct",
    "pt_down_3pct_iso_up_20pct",
    "pt_up_3pct_impact_up_20pct",
    "iso_up_20pct_impact_up_20pct",
]

DOMAIN_NAMES = CORE_DOMAIN_NAMES.copy()


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


def transform_features(features: np.ndarray, domain: str) -> np.ndarray:
    shifted = features.copy()
    if domain == "nominal":
        return shifted
    if domain == "pt_scale_up_3pct":
        shifted[:, :, 0] *= 1.03
    elif domain == "pt_scale_down_3pct":
        shifted[:, :, 0] *= 0.97
    elif domain == "iso_scale_up_20pct":
        shifted[:, :, 5] *= 1.20
    elif domain == "impact_sig_scale_up_20pct":
        shifted[:, :, 6] *= 1.20
        shifted[:, :, 7] *= 1.20
    elif domain == "pt_up_3pct_iso_up_20pct":
        shifted[:, :, 0] *= 1.03
        shifted[:, :, 5] *= 1.20
    elif domain == "pt_down_3pct_iso_up_20pct":
        shifted[:, :, 0] *= 0.97
        shifted[:, :, 5] *= 1.20
    elif domain == "pt_up_3pct_impact_up_20pct":
        shifted[:, :, 0] *= 1.03
        shifted[:, :, 6] *= 1.20
        shifted[:, :, 7] *= 1.20
    elif domain == "iso_up_20pct_impact_up_20pct":
        shifted[:, :, 5] *= 1.20
        shifted[:, :, 6] *= 1.20
        shifted[:, :, 7] *= 1.20
    else:
        raise ValueError(f"unknown domain {domain}")
    return shifted


def make_domain_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    masses: np.ndarray,
    weights: np.ndarray,
    indices: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_parts = []
    y_parts = []
    domain_parts = []
    mass_parts = []
    weight_parts = []
    for domain_id, domain in enumerate(DOMAIN_NAMES):
        shifted = transform_features(features[indices], domain)
        flat = shifted.reshape(shifted.shape[0], -1)
        x_parts.append(((flat - mean) / std).astype(np.float32))
        y_parts.append(labels[indices].astype(np.float32))
        domain_parts.append(np.full(len(indices), domain_id, dtype=np.int64))
        mass_parts.append(masses[indices].astype(np.float32))
        weight_parts.append(weights[indices].astype(np.float32))
    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    domains = np.concatenate(domain_parts)
    mass = np.concatenate(mass_parts)
    raw_weights = np.concatenate(weight_parts)
    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(domains, dtype=torch.long),
        torch.tensor(mass, dtype=torch.float32),
        y.astype(np.int64),
        mass.astype(np.float32),
        raw_weights.astype(np.float32),
        domains.astype(np.int64),
    )


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


class SplitBranchNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_domains: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.physics_branch = nn.Sequential(nn.Linear(hidden_dim, latent_dim), nn.GELU())
        self.nuisance_branch = nn.Sequential(nn.Linear(hidden_dim, latent_dim), nn.GELU())
        self.physics_head = nn.Linear(latent_dim, 1)
        self.nuisance_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_domains),
        )
        self.phys_adv_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_domains),
        )

    def forward(self, x: torch.Tensor, grl_lambda: float = 0.0) -> dict[str, torch.Tensor]:
        shared = self.trunk(x)
        z_phys = self.physics_branch(shared)
        z_nuis = self.nuisance_branch(shared)
        return {
            "z_phys": z_phys,
            "z_nuis": z_nuis,
            "physics_logits": self.physics_head(z_phys).squeeze(-1),
            "nuisance_logits": self.nuisance_head(z_nuis),
            "phys_adv_logits": self.phys_adv_head(grad_reverse(z_phys, grl_lambda)),
        }


def orthogonal_penalty(z_phys: torch.Tensor, z_nuis: torch.Tensor) -> torch.Tensor:
    z_phys = F.normalize(z_phys - z_phys.mean(dim=0, keepdim=True), dim=0)
    z_nuis = F.normalize(z_nuis - z_nuis.mean(dim=0, keepdim=True), dim=0)
    return (z_phys.T @ z_nuis).pow(2).mean()


def corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def domain_score_drift(scores: np.ndarray, domains: np.ndarray) -> float:
    nominal = scores[domains == 0]
    if len(nominal) == 0:
        return 0.0
    nominal_mean = float(nominal.mean())
    shifts = [abs(float(scores[domains == domain_id].mean()) - nominal_mean) for domain_id in range(1, len(DOMAIN_NAMES))]
    return max(shifts) if shifts else 0.0


def train_one(
    mode: str,
    train_loader: DataLoader,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    val_domains: torch.Tensor,
    input_dim: int,
    pos_weight: float,
    epochs: int,
    learning_rate: float,
    adv_lambda: float,
    orth_lambda: float,
    device: torch.device,
) -> tuple[SplitBranchNet, list[dict]]:
    model = SplitBranchNet(input_dim=input_dim, hidden_dim=128, latent_dim=64, num_domains=len(DOMAIN_NAMES)).to(device)
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
            grl = adv_lambda if mode == "split_orth_adv" else 0.0
            out = model(batch_x, grl_lambda=grl)
            physics_loss = F.binary_cross_entropy_with_logits(
                out["physics_logits"], batch_y, pos_weight=pos_weight_tensor
            )
            nuisance_loss = F.cross_entropy(out["nuisance_logits"], batch_domain)
            adv_loss = F.cross_entropy(out["phys_adv_logits"], batch_domain)
            orth_loss = orthogonal_penalty(out["z_phys"], out["z_nuis"])
            if mode == "shared_baseline":
                loss = physics_loss
            elif mode == "split_no_orth":
                loss = physics_loss + nuisance_loss
            elif mode == "split_orth":
                loss = physics_loss + nuisance_loss + orth_lambda * orth_loss
            elif mode == "split_orth_adv":
                loss = physics_loss + nuisance_loss + adv_loss + orth_lambda * orth_loss
            else:
                raise ValueError(f"unknown mode {mode}")
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
            val_orth = float(orthogonal_penalty(val_out["z_phys"], val_out["z_nuis"]).cpu())
        denom = max(totals.pop("count"), 1)
        history.append(
            {
                "mode": mode,
                "epoch": epoch,
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


def embed_and_score(
    model: SplitBranchNet, x: torch.Tensor, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    scores = []
    z_phys = []
    z_nuis = []
    nuisance_logits = []
    loader = DataLoader(TensorDataset(x), batch_size=8192, shuffle=False)
    with torch.no_grad():
        for (batch_x,) in loader:
            out = model(batch_x.to(device), grl_lambda=0.0)
            scores.append(torch.sigmoid(out["physics_logits"]).cpu().numpy())
            z_phys.append(out["z_phys"].cpu().numpy())
            z_nuis.append(out["z_nuis"].cpu().numpy())
            nuisance_logits.append(out["nuisance_logits"].cpu().numpy())
    return np.concatenate(scores), np.concatenate(z_phys), np.concatenate(z_nuis), np.concatenate(nuisance_logits)


def train_domain_probe(
    train_z: np.ndarray,
    train_domains: np.ndarray,
    val_z: np.ndarray,
    val_domains: np.ndarray,
    epochs: int,
    device: torch.device,
) -> float:
    probe = nn.Linear(train_z.shape[1], len(DOMAIN_NAMES)).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=2e-3, weight_decay=1e-4)
    train_ds = TensorDataset(torch.tensor(train_z, dtype=torch.float32), torch.tensor(train_domains, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=8192, shuffle=True)
    for _ in range(epochs):
        probe.train()
        for batch_z, batch_domain in train_loader:
            batch_z = batch_z.to(device)
            batch_domain = batch_domain.to(device)
            loss = F.cross_entropy(probe(batch_z), batch_domain)
            opt.zero_grad()
            loss.backward()
            opt.step()
    probe.eval()
    with torch.no_grad():
        logits = probe(torch.tensor(val_z, dtype=torch.float32, device=device))
        return float((logits.argmax(dim=1).cpu().numpy() == val_domains).mean())


def train_physics_probe_auc(
    train_z: np.ndarray,
    train_labels: np.ndarray,
    val_z: np.ndarray,
    val_labels: np.ndarray,
    val_weights: np.ndarray,
    epochs: int,
    device: torch.device,
) -> float | None:
    probe = nn.Linear(train_z.shape[1], 1).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=2e-3, weight_decay=1e-4)
    pos = max(float((train_labels == 1).sum()), 1.0)
    neg = max(float((train_labels == 0).sum()), 1.0)
    pos_weight = torch.tensor(neg / pos, dtype=torch.float32, device=device)
    train_ds = TensorDataset(torch.tensor(train_z, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=8192, shuffle=True)
    for _ in range(epochs):
        probe.train()
        for batch_z, batch_y in train_loader:
            batch_z = batch_z.to(device)
            batch_y = batch_y.to(device)
            loss = F.binary_cross_entropy_with_logits(probe(batch_z).squeeze(-1), batch_y, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
    probe.eval()
    with torch.no_grad():
        logits = probe(torch.tensor(val_z, dtype=torch.float32, device=device)).squeeze(-1).cpu().numpy()
    scores = 1.0 / (1.0 + np.exp(-logits))
    return weighted_auc(val_labels, scores, val_weights)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-npz", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--probe-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--adv-lambda", type=float, default=0.5)
    parser.add_argument("--orth-lambda", type=float, default=0.25)
    parser.add_argument("--max-events", type=int, default=50000)
    parser.add_argument("--domain-set", choices=["core", "extended"], default="core")
    args = parser.parse_args()

    global DOMAIN_NAMES
    DOMAIN_NAMES = CORE_DOMAIN_NAMES.copy() if args.domain_set == "core" else EXTENDED_DOMAIN_NAMES.copy()
    run_dir = create_run_dir("e68c-cms-h4l-split-branch-disentanglement")
    REPORTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    with np.load(Path(args.tensor_npz), allow_pickle=False) as payload:
        features = payload["features"].astype(np.float32)
        labels = payload["labels"].astype(np.int64)
        conditions = payload["conditions"].astype(np.float32)
    masses = conditions[:, 0].astype(np.float32)
    weights = conditions[:, 4].astype(np.float32)
    if args.max_events > 0 and args.max_events < len(labels):
        rng = np.random.default_rng(12345)
        keep_parts = []
        for label in sorted(np.unique(labels).tolist()):
            label_idx = np.flatnonzero(labels == label)
            take = max(1, int(round(args.max_events * len(label_idx) / len(labels))))
            keep_parts.append(rng.choice(label_idx, size=min(take, len(label_idx)), replace=False))
        keep = np.concatenate(keep_parts)
        rng.shuffle(keep)
        features = features[keep]
        labels = labels[keep]
        masses = masses[keep]
        weights = weights[keep]

    flat = features.reshape(features.shape[0], -1)
    train_idx, val_idx = stratified_split(labels, val_ratio=args.val_ratio, seed=args.seed)
    mean = flat[train_idx].mean(axis=0, keepdims=True)
    std = flat[train_idx].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    train_x, train_y, train_domains, _, train_labels, _, train_weights, train_raw_domains = make_domain_dataset(
        features, labels, masses, weights, train_idx, mean, std
    )
    val_x, val_y, val_domains, _, val_labels, val_masses, val_weights, val_raw_domains = make_domain_dataset(
        features, labels, masses, weights, val_idx, mean, std
    )
    pos = max(float((train_y.numpy() == 1).sum()), 1.0)
    neg = max(float((train_y.numpy() == 0).sum()), 1.0)
    pos_weight = neg / pos
    train_loader = DataLoader(
        TensorDataset(train_x, train_y, train_domains),
        batch_size=args.batch_size,
        shuffle=True,
    )

    rows = []
    history_rows = []
    for mode in ["shared_baseline", "split_no_orth", "split_orth", "split_orth_adv"]:
        model, history = train_one(
            mode=mode,
            train_loader=train_loader,
            val_x=val_x,
            val_y=val_y,
            val_domains=val_domains,
            input_dim=flat.shape[1],
            pos_weight=pos_weight,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            adv_lambda=args.adv_lambda,
            orth_lambda=args.orth_lambda,
            device=device,
        )
        history_rows.extend(history)
        val_scores, val_z_phys, val_z_nuis, val_nuis_logits = embed_and_score(model, val_x, device)
        _, train_z_phys, train_z_nuis, _ = embed_and_score(model, train_x, device)
        rows.append(
            {
                "mode": mode,
                "physics_auc": weighted_auc(val_labels, val_scores, val_weights),
                "score_m4l_corr": corrcoef(val_scores, val_masses),
                "score_domain_drift_max": domain_score_drift(val_scores, val_raw_domains),
                "nuisance_head_acc": float((val_nuis_logits.argmax(axis=1) == val_raw_domains).mean()),
                "z_phys_domain_probe_acc": train_domain_probe(
                    train_z_phys, train_raw_domains, val_z_phys, val_raw_domains, args.probe_epochs, device
                ),
                "z_nuis_domain_probe_acc": train_domain_probe(
                    train_z_nuis, train_raw_domains, val_z_nuis, val_raw_domains, args.probe_epochs, device
                ),
                "z_nuis_physics_probe_auc": train_physics_probe_auc(
                    train_z_nuis, train_labels, val_z_nuis, val_labels, val_weights, args.probe_epochs, device
                ),
            }
        )

    raw_csv = run_dir / "split_branch_disentanglement_metrics.csv"
    history_csv = run_dir / "training_history.csv"
    write_csv(raw_csv, rows)
    write_csv(history_csv, history_rows)

    report_path = REPORTS / f"e68c_cms_h4l_split_branch_disentanglement_{dt.datetime.now():%Y%m%d}.md"
    lines = [
        "# E68c CMS H4l Split-Branch Systematic Disentanglement Smoke",
        "",
        f"- run_dir: `{run_dir}`",
        f"- generated_at: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- tensor_npz: `{args.tensor_npz}`",
        f"- device: `{device}`",
        f"- domains: `{', '.join(DOMAIN_NAMES)}`",
        f"- domain_set: `{args.domain_set}`",
        f"- seed: `{args.seed}`",
        f"- max_events: {args.max_events}",
        f"- epochs: {args.epochs}",
        f"- probe_epochs: {args.probe_epochs}",
        f"- adv_lambda: {args.adv_lambda}",
        f"- orth_lambda: {args.orth_lambda}",
        "- dataset note: lepton-feature domains on CMS H4l MC",
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
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- Useful split-branch behavior means physics AUC stays close to baseline while `z_phys` domain-probe accuracy moves toward the 0.20 random baseline.",
            "- `z_nuis` domain-probe accuracy should be higher than `z_phys` domain-probe accuracy; otherwise nuisance information was not preferentially routed.",
            "- `z_nuis` physics-probe AUC is reported as leakage cost.",
            "- The controlled feature perturbations evaluate method transfer across visible-domain variants.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "experiment": "E68c CMS H4l split-branch systematic disentanglement smoke",
                "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
                "run_dir": str(run_dir),
                "metrics_csv": str(raw_csv),
                "history_csv": str(history_csv),
                "report": str(report_path),
                "metrics": rows,
                "status": "done",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "status.txt").write_text(f"status: done\nreport: {report_path}\n", encoding="utf-8")
    print(f"E68c split-branch disentanglement done: {run_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
