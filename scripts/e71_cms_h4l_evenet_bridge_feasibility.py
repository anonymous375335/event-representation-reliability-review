#!/usr/bin/env python3
"""Feasibility smoke for feeding CMS H4l tensors through the official EveNet teacher."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from evenet_official_adapter import EveNetOfficialTeacherAdapter, build_evenet_passthrough_normalization


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
DEFAULT_TEACHER_CKPT = Path(
    os.environ.get("EVENET_TEACHER_CKPT", str(ROOT / "data" / "checkpoints" / "teachers" / "evenet_public.ckpt"))
)
DEFAULT_EVENET_REPO = Path(os.environ.get("EVENET_REPO", str(ROOT / "external" / "EveNet_Public")))

EVENET_SOURCE_NAMES = ["energy", "pt", "eta", "phi", "btag", "isLepton", "charge"]
EVENET_CONDITION_NAMES = [
    "met",
    "met_phi",
    "nLepton",
    "nbJet",
    "nJet",
    "HT",
    "HT_lep",
    "M_all",
    "M_leps",
    "M_bjets",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def adapt_h4l_to_evenet_public_schema(features: np.ndarray, conditions: np.ndarray):
    """Map the E65 H4l lepton tensor to EveNet public 7D Source + 10D Conditions."""
    pt = features[:, :, 0]
    eta = features[:, :, 1]
    phi = features[:, :, 2]
    mass = features[:, :, 3]
    charge = features[:, :, 4]
    energy = np.sqrt(np.maximum(mass * mass + (pt * np.cosh(eta)) ** 2, 0.0))

    evenet_features = np.stack(
        [
            energy,
            pt,
            eta,
            phi,
            np.zeros_like(pt),
            np.ones_like(pt),
            charge,
        ],
        axis=-1,
    ).astype(np.float32)

    ht_lep = pt.sum(axis=1)
    m4l = conditions[:, 0]
    evenet_conditions = np.stack(
        [
            np.zeros_like(m4l),
            np.zeros_like(m4l),
            np.full_like(m4l, 4.0),
            np.zeros_like(m4l),
            np.zeros_like(m4l),
            ht_lep,
            ht_lep,
            m4l,
            m4l,
            np.zeros_like(m4l),
        ],
        axis=-1,
    ).astype(np.float32)
    return evenet_features, evenet_conditions


def load_h4l_npz(path: Path, max_events: int, seed: int):
    with np.load(path, allow_pickle=False) as payload:
        raw_features = payload["features"].astype(np.float32)
        valid_masks = payload["valid_masks"].astype(np.float32)
        labels = payload["labels"].astype(np.int64)
        raw_conditions = payload["conditions"].astype(np.float32)
        conditions_mask = payload["conditions_mask"].astype(np.float32)
        raw_feature_names = [str(item) for item in payload.get("feature_names", [])]
        raw_condition_names = [str(item) for item in payload.get("condition_names", [])]

    if max_events > 0 and max_events < raw_features.shape[0]:
        rng = np.random.default_rng(seed)
        indices = rng.choice(raw_features.shape[0], size=max_events, replace=False)
        indices.sort()
        raw_features = raw_features[indices]
        valid_masks = valid_masks[indices]
        labels = labels[indices]
        raw_conditions = raw_conditions[indices]
        conditions_mask = conditions_mask[indices]

    features, conditions = adapt_h4l_to_evenet_public_schema(raw_features, raw_conditions)
    return {
        "features": torch.as_tensor(features),
        "valid_masks": torch.as_tensor(valid_masks),
        "labels": torch.as_tensor(labels),
        "conditions": torch.as_tensor(conditions),
        "conditions_mask": torch.as_tensor(conditions_mask),
        "feature_names": EVENET_SOURCE_NAMES,
        "condition_names": EVENET_CONDITION_NAMES,
        "raw_feature_names": raw_feature_names,
        "raw_condition_names": raw_condition_names,
        "adapter_mapping": {
            "source": "H4l pt, eta, phi, mass, charge -> EveNet energy, pt, eta, phi, btag=0, isLepton=1, charge",
            "conditions": "H4l m4l and summed lepton pt -> EveNet public global condition slots; jet/MET slots set to zero.",
        },
    }


def stratified_split(labels: torch.Tensor, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    labels_np = labels.cpu().numpy()
    train_indices = []
    val_indices = []
    for label in np.unique(labels_np):
        indices = np.where(labels_np == label)[0]
        shuffled = rng.permutation(indices)
        val_count = max(1, int(round(len(indices) * val_ratio)))
        val_count = min(val_count, max(1, len(indices) - 1))
        val_indices.extend(shuffled[:val_count].tolist())
        train_indices.extend(shuffled[val_count:].tolist())
    return np.asarray(sorted(train_indices)), np.asarray(sorted(val_indices))


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_balanced_accuracy(pred: torch.Tensor, labels: torch.Tensor) -> float:
    recalls = []
    for label in torch.unique(labels).tolist():
        mask = labels == label
        recalls.append(float((pred[mask] == labels[mask]).float().mean().item()))
    return float(np.mean(recalls)) if recalls else float("nan")


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


def collect_evenet_embeddings(teacher, payload, batch_size: int, device: torch.device):
    dataset = TensorDataset(
        payload["features"],
        payload["valid_masks"],
        payload["conditions"],
        payload["conditions_mask"],
        payload["labels"],
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embeddings = []
    logits = []
    labels = []
    teacher.eval()
    with torch.no_grad():
        for features, valid_masks, conditions, conditions_mask, batch_labels in loader:
            features = features.to(device)
            valid_masks = valid_masks.to(device)
            conditions = conditions.to(device)
            conditions_mask = conditions_mask.to(device)
            embedding, logit = teacher(features, valid_masks, conditions, conditions_mask)
            embeddings.append(embedding.detach().cpu())
            logits.append(logit.detach().cpu())
            labels.append(batch_labels.cpu())
    return torch.cat(embeddings), torch.cat(logits), torch.cat(labels)


def train_linear_probe(embeddings, labels, train_indices, val_indices, epochs, batch_size, lr, device):
    num_classes = int(labels.max().item() + 1)
    model = LinearProbe(embeddings.shape[-1], num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train_dataset = TensorDataset(embeddings[train_indices], labels[train_indices])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_counts = torch.bincount(labels[train_indices], minlength=num_classes).float()
    class_weights = (train_counts.sum() / train_counts.clamp_min(1.0)).to(device)
    class_weights = class_weights / class_weights.mean().clamp_min(1e-6)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for batch_embeddings, batch_labels in train_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(batch_embeddings), batch_labels, weight=class_weights)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * batch_labels.shape[0]
            count += batch_labels.shape[0]
        history.append({"epoch": epoch, "train_loss": loss_sum / max(count, 1)})

    model.eval()
    with torch.no_grad():
        val_logits = model(embeddings[val_indices].to(device)).cpu()
    val_probs = F.softmax(val_logits, dim=1)
    val_pred = val_logits.argmax(dim=1)
    val_labels = labels[val_indices]
    val_acc = float((val_pred == val_labels).float().mean().item())
    val_balanced_acc = binary_balanced_accuracy(val_pred, val_labels)
    val_auc = binary_auc(val_probs[:, 1].numpy(), val_labels.numpy()) if num_classes == 2 else float("nan")
    return {
        "val_acc": val_acc,
        "val_balanced_acc": val_balanced_acc,
        "val_auc": val_auc,
        "class_weights": class_weights.detach().cpu().tolist(),
        "history": history,
    }


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_history(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-npz", default=str(ROOT / "data_processed" / "cms_h4l_e65" / "cms_h4l_mc_candidates_e65.npz"))
    parser.add_argument("--teacher-ckpt", default=str(DEFAULT_TEACHER_CKPT))
    parser.add_argument("--evenet-repo", default=str(DEFAULT_EVENET_REPO))
    parser.add_argument("--max-events", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--probe-epochs", type=int, default=25)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    RUNS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    run_dir = create_run_dir("e71-cms-h4l-evenet-bridge-feasibility")
    device = select_device(args.device)

    payload = load_h4l_npz(Path(args.tensor_npz), args.max_events, args.seed)
    train_indices, val_indices = stratified_split(payload["labels"], args.val_ratio, args.seed)
    normalization = build_evenet_passthrough_normalization(
        feature_dim=payload["features"].shape[-1],
        condition_dim=payload["conditions"].shape[-1],
        labels=payload["labels"][train_indices],
        device=device,
    )

    metrics = {
        "experiment": "E71 CMS H4l EveNet bridge feasibility",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "tensor_npz": str(Path(args.tensor_npz)),
        "teacher_ckpt": str(Path(args.teacher_ckpt)),
        "evenet_repo": str(Path(args.evenet_repo)),
        "device": str(device),
        "max_events": args.max_events,
        "n_events": int(payload["features"].shape[0]),
        "feature_shape": list(payload["features"].shape),
        "condition_shape": list(payload["conditions"].shape),
        "feature_names": payload["feature_names"],
        "condition_names": payload["condition_names"],
        "raw_feature_names": payload["raw_feature_names"],
        "raw_condition_names": payload["raw_condition_names"],
        "adapter_mapping": payload["adapter_mapping"],
        "class_counts": {
            str(label): int((payload["labels"] == label).sum().item())
            for label in torch.unique(payload["labels"]).tolist()
        },
        "train_events": int(len(train_indices)),
        "val_events": int(len(val_indices)),
    }

    try:
        teacher = EveNetOfficialTeacherAdapter(
            checkpoint_path=Path(args.teacher_ckpt),
            normalization_dict=normalization,
            device=device,
            repo_root=Path(args.evenet_repo),
        ).to(device)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        metrics["teacher_load_status"] = "evenet_official_checkpoint_loaded"
        metrics["teacher_hidden_dim"] = int(teacher.hidden_dim)

        embeddings, teacher_logits, labels = collect_evenet_embeddings(teacher, payload, args.batch_size, device)
        finite = torch.isfinite(embeddings)
        finite_count = int(finite.sum().item())
        total_embedding_values = int(finite.numel())
        dim_std = embeddings.float().std(dim=0, unbiased=False)
        metrics.update(
            {
                "forward_status": "ok",
                "embedding_shape": list(embeddings.shape),
                "teacher_logit_shape": list(teacher_logits.shape),
                "embedding_finite_count": finite_count,
                "embedding_total_values": total_embedding_values,
                "embedding_finite_fraction": finite_count / max(total_embedding_values, 1),
                "embedding_norm_mean": float(embeddings.norm(dim=1).mean().item()),
                "embedding_norm_std": float(embeddings.norm(dim=1).std(unbiased=False).item()),
                "embedding_dim_std_mean": float(dim_std.mean().item()),
                "embedding_dim_std_min": float(dim_std.min().item()),
                "embedding_dim_std_max": float(dim_std.max().item()),
                "teacher_logit_std": float(teacher_logits.float().std(unbiased=False).item()),
            }
        )

        probe = train_linear_probe(
            embeddings=embeddings.float(),
            labels=labels.long(),
            train_indices=train_indices,
            val_indices=val_indices,
            epochs=args.probe_epochs,
            batch_size=args.batch_size,
            lr=args.probe_lr,
            device=device,
        )
        metrics["embedding_probe_val_acc"] = probe["val_acc"]
        metrics["embedding_probe_val_balanced_acc"] = probe["val_balanced_acc"]
        metrics["embedding_probe_val_auc"] = probe["val_auc"]
        metrics["embedding_probe_class_weights"] = probe["class_weights"]
        write_history(run_dir / "embedding_probe_history.csv", probe["history"])

        if metrics["embedding_finite_fraction"] < 0.999 or metrics["embedding_dim_std_mean"] <= 1e-6:
            metrics["bridge_verdict"] = "fail_degenerate_embedding"
        elif metrics["embedding_probe_val_auc"] >= 0.70:
            metrics["bridge_verdict"] = "pass_interface_and_semantic_signal"
        elif metrics["embedding_probe_val_auc"] >= 0.58:
            metrics["bridge_verdict"] = "partial_interface_ok_weak_semantic_signal"
        else:
            metrics["bridge_verdict"] = "partial_interface_ok_no_clear_semantic_signal"
    except Exception as exc:
        metrics["teacher_load_status"] = metrics.get("teacher_load_status", "failed")
        metrics["forward_status"] = "failed"
        metrics["bridge_verdict"] = "fail_interface"
        metrics["error"] = repr(exc)

    write_json(run_dir / "metrics.json", metrics)
    (run_dir / "status.txt").write_text(
        f"status: {metrics['bridge_verdict']}\nmetrics: {run_dir / 'metrics.json'}\n",
        encoding="utf-8",
    )

    report_path = REPORTS / f"e71_cms_h4l_evenet_bridge_feasibility_{dt.datetime.now():%Y%m%d}.md"
    lines = [
        "# E71 CMS H4l EveNet Bridge Feasibility",
        "",
        f"- run_dir: `{run_dir}`",
        f"- generated_at: {metrics['generated_at']}",
        f"- tensor_npz: `{metrics['tensor_npz']}`",
        f"- teacher_ckpt: `{metrics['teacher_ckpt']}`",
        f"- device: `{metrics['device']}`",
        f"- n_events: {metrics['n_events']}",
        f"- adapted_feature_names: `{', '.join(metrics['feature_names'])}`",
        f"- adapted_condition_names: `{', '.join(metrics['condition_names'])}`",
        f"- teacher_load_status: `{metrics['teacher_load_status']}`",
        f"- forward_status: `{metrics['forward_status']}`",
        f"- bridge_verdict: `{metrics['bridge_verdict']}`",
        "",
        "## Readout Metrics",
        "",
        f"- embedding_shape: `{metrics.get('embedding_shape')}`",
        f"- embedding_finite_fraction: {metrics.get('embedding_finite_fraction')}",
        f"- embedding_dim_std_mean: {metrics.get('embedding_dim_std_mean')}",
        f"- embedding_probe_val_acc: {metrics.get('embedding_probe_val_acc')}",
        f"- embedding_probe_val_balanced_acc: {metrics.get('embedding_probe_val_balanced_acc')}",
        f"- embedding_probe_val_auc: {metrics.get('embedding_probe_val_auc')}",
        "",
        "## Bridge Note",
        "",
        "- Bridge feasibility smoke test for the H4l-to-EveNet adapter.",
        "- It uses H4l lepton tensors as EveNet-style point-cloud inputs with passthrough normalization.",
        "- A positive result means the EveNet interface is worth extending to the split-branch systematic-disentanglement experiment.",
        "- A weak result points to redesign of the H4l-to-EveNet semantic adapter before main-paper integration.",
        "",
    ]
    if "error" in metrics:
        lines.extend(["## Error", "", f"`{metrics['error']}`", ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")
    return 0 if not metrics["bridge_verdict"].startswith("fail_interface") else 1


if __name__ == "__main__":
    raise SystemExit(main())
