#!/usr/bin/env python3
"""E102 frozen exact-checkpoint metric-to-likelihood audit on used TopTag shards."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import e68c_cms_h4l_split_branch_disentanglement as e68c
import e75c_toptag_branch_protocol_smoke as e75c
import e75e_toptag_systematic_family_scaleup as e75e
import e76b_toptag_constituent_encoder as e76b
import e88_toptag_signal_tail_protected_split as e88
from e66_cms_h4l_readout_smoke import weighted_auc


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def resolve_from_root(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(rows: list[dict], x_key: str, y_key: str) -> float:
    x = rankdata(np.asarray([float(row[x_key]) for row in rows]))
    y = rankdata(np.asarray([float(row[y_key]) for row in rows]))
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def profile_metrics(run_dir: Path, candidate: str) -> dict:
    rows = read_csv(run_dir / "profile_stress_summary.csv")
    selected = [row for row in rows if row["candidate"] == candidate and row["group"] == "unmodeled_shifted"]
    if len(selected) != 1:
        raise RuntimeError(f"expected one unmodeled row for {candidate} in {run_dir}, found {len(selected)}")
    row = selected[0]
    return {
        "mean_abs_mu_bias": float(row["mean_abs_mu_bias"]),
        "max_abs_mu_bias": float(row["max_abs_mu_bias"]),
        "rms_mu_bias": float(row["rms_mu_bias"]),
        "max_abs_energy_scale_hat": float(row["max_abs_energy_scale_hat"]),
    }


def template_metrics(run_dir: Path, candidate: str) -> dict:
    shape_rows = [row for row in read_csv(run_dir / "shape_metrics.csv") if row["candidate"] == candidate]
    max_tvd = max(float(row["tvd_vs_nominal"]) for row in shape_rows if row["domain"] != "nominal")

    rows = [row for row in read_csv(run_dir / "score_templates.csv") if row["candidate"] == candidate]
    domains = sorted({row["domain"] for row in rows if row["domain"] != "nominal"})
    nominal_bkg = {int(row["bin_index"]): float(row["density"]) for row in rows if row["domain"] == "nominal" and row["label"] == "0"}
    nominal_sig_counts = {int(row["bin_index"]): float(row["count"]) for row in rows if row["domain"] == "nominal" and row["label"] == "1"}
    nominal_bkg_counts = {int(row["bin_index"]): float(row["count"]) for row in rows if row["domain"] == "nominal" and row["label"] == "0"}
    bin_ids = sorted(nominal_bkg)
    signal_fraction = {
        index: nominal_sig_counts[index] / max(nominal_sig_counts[index] + nominal_bkg_counts[index], 1.0)
        for index in bin_ids
    }
    domain_scores = []
    for domain in domains:
        density = {int(row["bin_index"]): float(row["density"]) for row in rows if row["domain"] == domain and row["label"] == "0"}
        domain_scores.append(sum(max(density[index] - nominal_bkg[index], 0.0) * signal_fraction[index] for index in bin_ids))
    return {
        "max_template_tvd": max_tvd,
        "max_signal_weighted_positive_bkg_residual": max(domain_scores),
    }


def selected_rows(e79_config: dict) -> list[dict]:
    manifest = resolve_from_root(e79_config["manifest"])
    return e75e.select_first_files(e75e.read_rows(manifest), e79_config["indices"])


def validate_group(group: dict, config: dict) -> None:
    shard = group["shard"]
    if shard not in config["allowed_shards"]:
        raise RuntimeError(f"forbidden shard in config: {shard}")
    for key in ["e79_run", "e79_profile_run", "e91d_run", "e91d_profile_run"]:
        path = resolve_from_root(group[key])
        if not path.is_dir():
            raise FileNotFoundError(path)
    e79_config = json.loads((resolve_from_root(group["e79_run"]) / "config.json").read_text())
    if int(e79_config["seed"]) != int(group["seed"]):
        raise RuntimeError(f"seed mismatch for {group}")
    data_root = Path(config["data_root"]).resolve()
    data_dir = Path(e79_config["data_dir"]).resolve()
    cache_dir = Path(e79_config["cache_dir"]).resolve()
    if data_root not in data_dir.parents or data_root not in cache_dir.parents:
        raise RuntimeError(f"data/cache path escaped data disk: {data_dir}, {cache_dir}")
    for row in selected_rows(e79_config):
        match = re.search(r"_(\d{3})\.h5\.gz$", row["filename"])
        if not match or match.group(1) != shard:
            raise RuntimeError(f"shard mismatch: expected {shard}, got {row['filename']}")
        if int(match.group(1)) >= int(config["forbidden_confirmation_shard"]):
            raise RuntimeError(f"confirmation shard access forbidden: {row['filename']}")
        if not (data_dir / row["filename"]).is_file():
            raise FileNotFoundError(data_dir / row["filename"])
        cache_name = row["filename"].removesuffix(".gz")
        if not (cache_dir / cache_name).is_file():
            raise FileNotFoundError(cache_dir / cache_name)


def load_group_data(group: dict, config: dict):
    e79_config = json.loads((resolve_from_root(group["e79_run"]) / "config.json").read_text())
    args = SimpleNamespace(
        manifest=resolve_from_root(e79_config["manifest"]),
        indices=e79_config["indices"],
        data_dir=Path(e79_config["data_dir"]),
        cache_dir=Path(e79_config["cache_dir"]),
        max_events_per_domain=int(e79_config["max_events_per_domain"]),
        max_constituents=int(e79_config["max_constituents"]),
        seed=int(group["seed"]),
    )
    domain_names, constituents, masks, high, labels, domains, weights, _ = e88.load_data(args)
    train_idx, val_idx = e75c.joint_stratified_split(labels, domains, config["val_ratio"], args.seed)
    train_const, val_const = e76b.standardize_constituents(
        constituents[train_idx], constituents[val_idx], masks[train_idx], masks[val_idx]
    )
    train_high, val_high = e76b.standardize_high(high[train_idx], high[val_idx])
    return {
        "domain_names": domain_names,
        "train_tensors": (
            torch.tensor(train_const, dtype=torch.float32),
            torch.tensor(masks[train_idx].astype(np.float32), dtype=torch.float32),
            torch.tensor(train_high, dtype=torch.float32),
        ),
        "val_tensors": (
            torch.tensor(val_const, dtype=torch.float32),
            torch.tensor(masks[val_idx].astype(np.float32), dtype=torch.float32),
            torch.tensor(val_high, dtype=torch.float32),
        ),
        "train_y": labels[train_idx],
        "val_y": labels[val_idx],
        "train_domain": domains[train_idx],
        "val_domain": domains[val_idx],
        "val_weights": weights[val_idx],
    }


def evaluate_checkpoint(model_spec: dict, data: dict, config: dict, device: torch.device) -> dict:
    e68c.DOMAIN_NAMES = data["domain_names"].copy()
    model = e76b.ConstituentSplitNet(
        constituent_dim=4,
        high_dim=data["train_tensors"][2].shape[1],
        point_dim=64,
        hidden_dim=128,
        latent_dim=64,
        nuisance_latent_dim=model_spec["nuisance_latent_dim"],
        num_domains=len(data["domain_names"]),
    ).to(device)
    model.load_state_dict(torch.load(model_spec["checkpoint"], map_location=device))
    train_scores, _, train_z_nuis, _ = e76b.embed_and_score(
        model, *data["train_tensors"], device, config["batch_size"]
    )
    val_scores, _, val_z_nuis, val_nuis_logits = e76b.embed_and_score(
        model, *data["val_tensors"], device, config["batch_size"]
    )
    del train_scores
    leakage_values = []
    domain_values = []
    for probe_seed in config["probe_seeds"]:
        e68c.set_seed(int(probe_seed))
        leakage_values.append(
            e68c.train_physics_probe_auc(
                train_z_nuis, data["train_y"], val_z_nuis, data["val_y"], data["val_weights"],
                config["probe_epochs"], device,
            )
        )
        e68c.set_seed(int(probe_seed))
        domain_values.append(
            e68c.train_domain_probe(
                train_z_nuis, data["train_domain"], val_z_nuis, data["val_domain"],
                config["probe_epochs"], device,
            )
        )
    leakage_mean, leakage_sd = mean_sd(leakage_values)
    domain_mean, domain_sd = mean_sd(domain_values)
    return {
        "physics_auc": weighted_auc(data["val_y"], val_scores, data["val_weights"]),
        "score_domain_drift_max": e68c.domain_score_drift(val_scores, data["val_domain"]),
        "nuisance_head_acc": float((val_nuis_logits.argmax(axis=1) == data["val_domain"]).mean()),
        "z_nuis_physics_probe_auc_mean": leakage_mean,
        "z_nuis_physics_probe_auc_sd": leakage_sd,
        "z_nuis_domain_probe_acc_mean": domain_mean,
        "z_nuis_domain_probe_acc_sd": domain_sd,
        "probe_physics_values": "|".join(f"{value:.8f}" for value in leakage_values),
        "probe_domain_values": "|".join(f"{value:.8f}" for value in domain_values),
    }


def model_specs(group: dict) -> list[dict]:
    e79_run = resolve_from_root(group["e79_run"])
    e91d_run = resolve_from_root(group["e91d_run"])
    return [
        {"model":"shared", "checkpoint":e79_run / "constituent_shared_baseline.pt", "nuisance_latent_dim":64,
         "template_run":e79_run, "template_candidate":"constituent_shared_baseline",
         "profile_run":resolve_from_root(group["e79_profile_run"]), "profile_candidate":"constituent_shared_baseline"},
        {"model":"balanced", "checkpoint":e79_run / "constituent_balanced.pt", "nuisance_latent_dim":32,
         "template_run":e79_run, "template_candidate":"constituent_balanced",
         "profile_run":resolve_from_root(group["e79_profile_run"]), "profile_candidate":"constituent_balanced"},
        {"model":"frozen_residual", "checkpoint":e91d_run / "constituent_frozen_residual_target.pt", "nuisance_latent_dim":32,
         "template_run":e91d_run, "template_candidate":"constituent_frozen_residual_target",
         "profile_run":resolve_from_root(group["e91d_profile_run"]), "profile_candidate":"constituent_frozen_residual_target"},
    ]


def paired_contrasts(rows: list[dict], config: dict) -> list[dict]:
    indexed = {(row["shard"], int(row["seed"]), row["model"]): row for row in rows}
    output = []
    for group in config["groups"]:
        key = (group["shard"], int(group["seed"]))
        for reference, candidate in [("shared", "balanced"), ("balanced", "frozen_residual")]:
            ref = indexed[key + (reference,)]
            cand = indexed[key + (candidate,)]
            delta_leakage = cand["z_nuis_physics_probe_auc_mean"] - ref["z_nuis_physics_probe_auc_mean"]
            delta_bias = cand["max_abs_mu_bias"] - ref["max_abs_mu_bias"]
            material = delta_leakage <= config["decision_rules"]["material_leakage_improvement"]
            output.append({
                "shard": key[0], "seed": key[1], "reference": reference, "candidate": candidate,
                "delta_leakage_auc": delta_leakage,
                "delta_max_abs_mu_bias": delta_bias,
                "delta_signal_weighted_residual": cand["max_signal_weighted_positive_bkg_residual"] - ref["max_signal_weighted_positive_bkg_residual"],
                "material_leakage_improvement": material,
                "likelihood_bias_improved": delta_bias < 0.0,
                "discordant_cleaner_but_not_better": material and delta_bias >= 0.0,
            })
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for group in config["groups"]:
        validate_group(group, config)
    if args.preflight_only:
        print(f"E102 preflight passed for {len(config['groups'])} groups; shards={config['allowed_shards']}")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = e75e.create_run_dir("e102-frozen-metric-to-likelihood-audit")
    rows = []
    for group_index, group in enumerate(config["groups"], start=1):
        print(f"[group {group_index}/{len(config['groups'])}] shard={group['shard']} seed={group['seed']}", flush=True)
        data = load_group_data(group, config)
        for spec in model_specs(group):
            print(f"  [model] {spec['model']}", flush=True)
            result = evaluate_checkpoint(spec, data, config, device)
            rows.append({
                "shard": group["shard"], "seed": int(group["seed"]), "model": spec["model"],
                **result,
                **template_metrics(spec["template_run"], spec["template_candidate"]),
                **profile_metrics(spec["profile_run"], spec["profile_candidate"]),
                "checkpoint": str(spec["checkpoint"]),
            })
        del data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    contrasts = paired_contrasts(rows, config)
    rho_leakage = spearman(rows, "z_nuis_physics_probe_auc_mean", "max_abs_mu_bias")
    rho_residual = spearman(rows, "max_signal_weighted_positive_bkg_residual", "max_abs_mu_bias")
    material = [row for row in contrasts if row["material_leakage_improvement"]]
    discordant = [row for row in material if row["discordant_cleaner_but_not_better"]]
    discordant_fraction = len(discordant) / max(len(material), 1)
    rules = config["decision_rules"]
    complete = len(rows) == rules["required_complete_rows"] and len(contrasts) == rules["required_paired_transitions"]
    finite = all(math.isfinite(float(value)) for row in rows for key, value in row.items() if isinstance(value, (int, float)))
    proxy_failure = (
        complete and finite and len(material) > 0
        and discordant_fraction >= rules["proxy_failure_min_discordant_fraction"]
        and abs(rho_leakage) <= rules["weak_leakage_bias_association_abs_spearman_max"]
    )
    aligned_advantage = abs(rho_residual) - abs(rho_leakage)
    summary = {
        "experiment": config["experiment"],
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "rows": len(rows),
        "paired_transitions": len(contrasts),
        "material_leakage_improvement_transitions": len(material),
        "discordant_cleaner_but_not_better": len(discordant),
        "discordant_fraction": discordant_fraction,
        "spearman_leakage_vs_max_bias": rho_leakage,
        "spearman_signal_weighted_residual_vs_max_bias": rho_residual,
        "analysis_aligned_abs_spearman_gain": aligned_advantage,
        "complete": complete,
        "finite": finite,
        "proxy_failure_candidate": proxy_failure,
        "analysis_aligned_proxy_candidate": aligned_advantage >= rules["analysis_aligned_advantage_abs_spearman_min_gain"],
        "decision": "unlock_separate_shard002_confirmation_design" if proxy_failure else "do_not_unlock_confirmation",
        "boundary": config["boundaries"],
        "confirmation_shard_accessed": False,
        "independent_test_accessed": False,
    }
    write_csv(run_dir / "metric_likelihood_rows.csv", rows)
    write_csv(run_dir / "paired_contrasts.csv", contrasts)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (run_dir / "status.txt").write_text("completed\n", encoding="utf-8")
    REPORTS.mkdir(parents=True, exist_ok=True)
    report = REPORTS / f"e102_metric_to_likelihood_audit_{dt.datetime.now():%Y%m%d}.md"
    report.write_text(
        "\n".join([
            "# E102 frozen metric-to-likelihood audit", "",
            f"- run: `{run_dir}`", f"- decision: `{summary['decision']}`",
            f"- leakage-vs-max-bias Spearman: `{rho_leakage:.4f}`",
            f"- signal-weighted-residual-vs-max-bias Spearman: `{rho_residual:.4f}`",
            f"- cleaner-but-not-better: `{len(discordant)}/{len(material)}` material leakage-improving transitions", "",
            "This is a retrospective development-shard audit on shards 000/001. It does not access shard002 and is not independent confirmation.",
        ]) + "\n", encoding="utf-8"
    )
    print(f"E102 done: {run_dir}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
