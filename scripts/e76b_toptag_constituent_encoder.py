#!/usr/bin/env python3
"""E76b TopTag small constituent-encoder fine-tune over balanced systematic shards."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import h5py
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
REPORTS = ROOT / "reports"
DEFAULT_RECORD80030_MANIFEST = ROOT / "benchmarks" / "toptag_pyhf" / "manifests" / "record80030_files.csv"
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
E76A_REFERENCE = {
    "best_tradeoff_physics_auc": 0.9415,
    "best_tradeoff_z_nuis_domain_acc": 0.4446,
    "best_tradeoff_z_nuis_physics_auc": 0.8274,
    "lowest_leakage_z_nuis_physics_auc": 0.8191,
}


def load_domain_sequences(
    data_dir: Path,
    cache_dir: Path,
    domain: str,
    filename: str,
    domain_id: int,
    max_events_per_domain: int,
    max_constituents: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    gz_path = data_dir / filename
    h5_path = e75c.ensure_h5_cache(gz_path, cache_dir)
    with h5py.File(h5_path, "r") as handle:
        labels_all = handle["labels"][:].astype(np.int64)
        indices = e75c.select_stratified_indices(labels_all, max_events_per_domain, seed + domain_id * 101)
        indices = np.sort(indices)
        jet_eta = handle["fjet_eta"][indices].astype(np.float32)
        jet_phi = handle["fjet_phi"][indices].astype(np.float32)
        clus_pt = handle["fjet_clus_pt"][indices, :max_constituents].astype(np.float32)
        clus_eta = handle["fjet_clus_eta"][indices, :max_constituents].astype(np.float32)
        clus_phi = handle["fjet_clus_phi"][indices, :max_constituents].astype(np.float32)
        clus_e = handle["fjet_clus_E"][indices, :max_constituents].astype(np.float32)
        mask = (clus_pt > 0).astype(np.float32)
        constituents = np.stack(
            [
                np.log1p(np.maximum(clus_pt, 0.0) / 1000.0),
                clus_eta - jet_eta[:, None],
                e75c.wrap_delta_phi(clus_phi - jet_phi[:, None]),
                np.log1p(np.maximum(clus_e, 0.0) / 1000.0),
            ],
            axis=-1,
        ).astype(np.float32)
        constituents = np.nan_to_num(constituents, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        constituents *= mask[:, :, None]

        high_parts = []
        for key in e75c.HIGH_LEVEL_KEYS:
            values = handle[key][indices].astype(np.float32)
            if key in {"fjet_pt", "fjet_m", "fjet_ECF1", "fjet_ECF2", "fjet_ECF3"}:
                values = np.log1p(np.maximum(values, 0.0) / 1000.0)
            high_parts.append(values[:, None])
        high = np.concatenate(high_parts, axis=1)
        high = np.nan_to_num(high, copy=False, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        labels = labels_all[indices].astype(np.int64)
        domains = np.full(len(indices), domain_id, dtype=np.int64)
        weights = np.ones(len(indices), dtype=np.float32)
        summary = {
            "domain": domain,
            "filename": filename,
            "source_path": str(gz_path),
            "cache_h5": str(h5_path),
            "events": int(len(indices)),
            "label_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        }
    return constituents, mask, high, labels, domains, weights, summary


def standardize_high(train_high: np.ndarray, val_high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_high.mean(axis=0, keepdims=True)
    std = train_high.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((train_high - mean) / std).astype(np.float32), ((val_high - mean) / std).astype(np.float32)


def standardize_constituents(
    train_const: np.ndarray,
    val_const: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = train_mask > 0
    if valid.any():
        mean = train_const[valid].mean(axis=0, keepdims=True)
        std = train_const[valid].std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
    else:
        mean = np.zeros((1, train_const.shape[-1]), dtype=np.float32)
        std = np.ones((1, train_const.shape[-1]), dtype=np.float32)
    train = ((train_const - mean) / std).astype(np.float32) * train_mask[:, :, None]
    val = ((val_const - mean) / std).astype(np.float32) * val_mask[:, :, None]
    return train, val


class ConstituentSplitNet(nn.Module):
    def __init__(
        self,
        constituent_dim: int,
        high_dim: int,
        point_dim: int,
        hidden_dim: int,
        latent_dim: int,
        nuisance_latent_dim: int,
        num_domains: int,
    ):
        super().__init__()
        self.point_net = nn.Sequential(
            nn.Linear(constituent_dim, point_dim),
            nn.LayerNorm(point_dim),
            nn.GELU(),
            nn.Linear(point_dim, point_dim),
            nn.GELU(),
        )
        self.high_net = nn.Sequential(
            nn.Linear(high_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )
        trunk_in = point_dim * 2 + hidden_dim // 2
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.physics_branch = nn.Sequential(nn.Linear(hidden_dim, latent_dim), nn.GELU())
        self.nuisance_branch = nn.Sequential(nn.Linear(hidden_dim, nuisance_latent_dim), nn.GELU())
        self.physics_head = nn.Linear(latent_dim, 1)
        self.nuisance_head = nn.Sequential(
            nn.Linear(nuisance_latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_domains),
        )
        self.phys_adv_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_domains),
        )

    def forward(self, constituents: torch.Tensor, mask: torch.Tensor, high: torch.Tensor, grl_lambda: float = 0.0) -> dict:
        point = self.point_net(constituents)
        mask_expanded = mask.unsqueeze(-1)
        point = point * mask_expanded
        denom = mask_expanded.sum(dim=1).clamp_min(1.0)
        pooled_mean = point.sum(dim=1) / denom
        masked_point = point.masked_fill(mask_expanded == 0, -1e4)
        pooled_max = masked_point.max(dim=1).values
        pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max))
        shared = self.trunk(torch.cat([pooled_mean, pooled_max, self.high_net(high)], dim=1))
        z_phys = self.physics_branch(shared)
        z_nuis = self.nuisance_branch(shared)
        return {
            "z_phys": z_phys,
            "z_nuis": z_nuis,
            "physics_logits": self.physics_head(z_phys).squeeze(-1),
            "nuisance_logits": self.nuisance_head(z_nuis),
            "phys_adv_logits": self.phys_adv_head(e68c.grad_reverse(z_phys, grl_lambda)),
        }


def train_one(
    candidate: dict,
    train_loader: DataLoader,
    val_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    pos_weight: float,
    epochs: int,
    learning_rate: float,
    orth_lambda: float,
    device: torch.device,
) -> tuple[ConstituentSplitNet, list[dict]]:
    val_const, val_mask, val_high, val_y, val_domains = [tensor.to(device) for tensor in val_tensors]
    model = ConstituentSplitNet(
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
        totals = {"loss": 0.0, "physics": 0.0, "nuisance": 0.0, "adv": 0.0, "orth": 0.0, "count": 0}
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
            if candidate["mode"] == "shared_baseline":
                loss = physics_loss
            else:
                loss = (
                    physics_loss
                    + candidate["nuisance_weight"] * nuisance_loss
                    + candidate["phys_adv_weight"] * adv_loss
                    + orth_lambda * orth_loss
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
                "point_dim": candidate["point_dim"],
                "nuisance_weight": candidate["nuisance_weight"],
                "nuisance_latent_dim": candidate["nuisance_latent_dim"],
                "phys_adv_weight": candidate["phys_adv_weight"],
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
    model: ConstituentSplitNet,
    const: torch.Tensor,
    mask: torch.Tensor,
    high: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    scores = []
    z_phys = []
    z_nuis = []
    nuisance_logits = []
    loader = DataLoader(TensorDataset(const, mask, high), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_const, batch_mask, batch_high in loader:
            out = model(batch_const.to(device), batch_mask.to(device), batch_high.to(device), grl_lambda=0.0)
            scores.append(torch.sigmoid(out["physics_logits"]).cpu().numpy())
            z_phys.append(out["z_phys"].cpu().numpy())
            z_nuis.append(out["z_nuis"].cpu().numpy())
            nuisance_logits.append(out["nuisance_logits"].cpu().numpy())
    return np.concatenate(scores), np.concatenate(z_phys), np.concatenate(z_nuis), np.concatenate(nuisance_logits)


def evaluate(
    candidate: dict,
    model: ConstituentSplitNet,
    train_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    val_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    train_y_np: np.ndarray,
    val_y_np: np.ndarray,
    train_domain_np: np.ndarray,
    val_domain_np: np.ndarray,
    val_weights_np: np.ndarray,
    probe_epochs: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    train_const, train_mask, train_high = train_tensors
    val_const, val_mask, val_high = val_tensors
    val_scores, val_z_phys, val_z_nuis, val_nuis_logits = embed_and_score(
        model, val_const, val_mask, val_high, device, batch_size
    )
    _, train_z_phys, train_z_nuis, _ = embed_and_score(model, train_const, train_mask, train_high, device, batch_size)
    return {
        **candidate,
        "physics_auc": weighted_auc(val_y_np, val_scores, val_weights_np),
        "background_rejection_at_30pct_signal_eff": e75c.background_rejection_at_signal_eff(
            val_y_np, val_scores, 0.30
        ),
        "score_domain_drift_max": e68c.domain_score_drift(val_scores, val_domain_np),
        "nuisance_head_acc": float((val_nuis_logits.argmax(axis=1) == val_domain_np).mean()),
        "z_phys_domain_probe_acc": e68c.train_domain_probe(
            train_z_phys, train_domain_np, val_z_phys, val_domain_np, probe_epochs, device
        ),
        "z_nuis_domain_probe_acc": e68c.train_domain_probe(
            train_z_nuis, train_domain_np, val_z_nuis, val_domain_np, probe_epochs, device
        ),
        "z_nuis_physics_probe_auc": e68c.train_physics_probe_auc(
            train_z_nuis, train_y_np, val_z_nuis, val_y_np, val_weights_np, probe_epochs, device
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_RECORD80030_MANIFEST,
    )
    parser.add_argument("--indices", nargs="+", default=DEFAULT_BALANCED_INDICES)
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
    run_dir = e75e.create_run_dir("e76b-toptag-constituent-encoder")
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
        constituents, mask, high, labels, domains, weights, summary = load_domain_sequences(
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
    train_const_np, val_const_np = standardize_constituents(
        constituents[train_idx], constituents[val_idx], masks[train_idx], masks[val_idx]
    )
    train_high_np, val_high_np = standardize_high(high[train_idx], high[val_idx])
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

    candidates = [
        {
            "candidate": "constituent_shared_baseline",
            "mode": "shared_baseline",
            "point_dim": 64,
            "nuisance_weight": 0.0,
            "nuisance_latent_dim": 64,
            "phys_adv_weight": 0.0,
        },
        {
            "candidate": "constituent_domain_push",
            "mode": "split",
            "point_dim": 64,
            "nuisance_weight": 5.0,
            "nuisance_latent_dim": 32,
            "phys_adv_weight": 0.5,
        },
        {
            "candidate": "constituent_balanced",
            "mode": "split",
            "point_dim": 64,
            "nuisance_weight": 2.0,
            "nuisance_latent_dim": 32,
            "phys_adv_weight": 0.5,
        },
        {
            "candidate": "constituent_physics_preserve",
            "mode": "split",
            "point_dim": 96,
            "nuisance_weight": 1.0,
            "nuisance_latent_dim": 64,
            "phys_adv_weight": 0.25,
        },
    ]
    rows = []
    history_rows = []
    train_eval_tensors = (train_const, train_mask, train_high)
    val_eval_tensors = (val_const, val_mask, val_high)
    val_train_tensors = (val_const, val_mask, val_high, val_y, val_domains)
    for candidate in candidates:
        print(f"[fit-start] {candidate['candidate']} {dt.datetime.now().isoformat(timespec='seconds')}", flush=True)
        model, history = train_one(
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
        rows.append(
            evaluate(
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
        )

    metrics_csv = run_dir / "constituent_encoder_metrics.csv"
    history_csv = run_dir / "training_history.csv"
    e75e.write_csv(metrics_csv, rows)
    e75e.write_csv(history_csv, history_rows)
    split_rows = [row for row in rows if row["mode"] != "shared_baseline"]
    best_leakage = min(split_rows, key=lambda row: (row["z_nuis_physics_probe_auc"], -row["physics_auc"]))
    best_tradeoff = max(split_rows, key=lambda row: (row["physics_auc"] - 0.5 * max(row["z_nuis_physics_probe_auc"] - 0.65, 0.0)))
    config = {
        "manifest": str(args.manifest),
        "indices": args.indices,
        "data_dir": str(args.data_dir),
        "cache_dir": str(args.cache_dir),
        "domains": domain_names,
        "max_events_per_domain": args.max_events_per_domain,
        "max_constituents": args.max_constituents,
        "constituent_dim": 4,
        "high_dim": int(high.shape[1]),
        "total_events_loaded": int(len(labels)),
        "seed": args.seed,
        "epochs": args.epochs,
        "probe_epochs": args.probe_epochs,
        "batch_size": args.batch_size,
        "orth_lambda": args.orth_lambda,
        "learning_rate": args.learning_rate,
        "e75e_reference": E75E_REFERENCE,
        "e76a_reference": E76A_REFERENCE,
        "domain_summaries": summaries,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    report_path = REPORTS / f"e76b_toptag_constituent_encoder_d{len(domain_names)}_{dt.datetime.now():%Y%m%d}_seed{args.seed}.md"
    lines = [
        "# E76b TopTag Constituent Encoder Fine-Tune",
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
        f"- max_constituents: {args.max_constituents}",
        f"- high_level_dim: {high.shape[1]}",
        f"- epochs: {args.epochs}",
        f"- probe_epochs: {args.probe_epochs}",
        "- run note: DeepSets-style constituent encoder over first balanced shards.",
        "",
        "## References",
        "",
        f"- E75e best split: physics AUC {E75E_REFERENCE['best_split_physics_auc']:.4f}, "
        f"`z_nuis -> domain` {E75E_REFERENCE['best_split_z_nuis_domain_acc']:.4f}, "
        f"`z_nuis -> physics` {E75E_REFERENCE['best_split_z_nuis_physics_auc']:.4f}",
        f"- E76a best tradeoff: physics AUC {E76A_REFERENCE['best_tradeoff_physics_auc']:.4f}, "
        f"`z_nuis -> domain` {E76A_REFERENCE['best_tradeoff_z_nuis_domain_acc']:.4f}, "
        f"`z_nuis -> physics` {E76A_REFERENCE['best_tradeoff_z_nuis_physics_auc']:.4f}",
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
        "| candidate | point dim | nuisance weight | nuisance dim | phys adv | physics AUC | bkg rejection @30% sig eff | score domain drift | nuisance head acc | z_phys domain probe | z_nuis domain probe | z_nuis physics probe AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {row['point_dim']} | {row['nuisance_weight']:g} | "
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
            "- Upgrade signal: constituent encoder should push `z_nuis -> physics` below E75e 0.8040 and preferably toward 0.70 while preserving physics AUC.",
            "- A larger encoder/run budget is the next escalation path if this remains weak.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "experiment": "E76b TopTag constituent encoder fine-tune",
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
    print(f"E76b TopTag constituent encoder fine-tune done: {run_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
