#!/usr/bin/env python3
"""Run the one-shot E103 shard002 shared-versus-balanced confirmation."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import shutil
import subprocess
import sys
import traceback
import zlib
from pathlib import Path

import torch

import e75e_toptag_systematic_family_scaleup as e75e
import e102_metric_to_likelihood_audit as e102


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adler32(path: Path) -> str:
    value = 1
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value = zlib.adler32(chunk, value)
    return f"adler32:{value & 0xffffffff:08x}"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check_disk(path: Path, minimum: int) -> None:
    free = shutil.disk_usage(path).free
    if free < minimum:
        raise RuntimeError(f"insufficient free space at {path}: {free} < {minimum}")


def selected_manifest_rows(config: dict) -> list[dict]:
    rows = read_csv(resolve(config["full_manifest"]))
    output = []
    for expected in config["selected_files"]:
        matches = [
            row for row in rows
            if row["index_description"] == expected["index_description"]
            and row["filename"] == expected["filename"]
        ]
        if len(matches) != 1:
            raise RuntimeError(f"expected one manifest row for {expected['filename']}, found {len(matches)}")
        row = matches[0]
        for key in ["index_description", "filename", "checksum", "uri", "availability"]:
            if row[key] != str(expected[key]):
                raise RuntimeError(f"manifest mismatch for {expected['filename']} field {key}")
        if int(row["size_bytes"]) != int(expected["size_bytes"]):
            raise RuntimeError(f"manifest size mismatch for {expected['filename']}")
        output.append(row)
    return output


def summarize(rows: list[dict], config: dict) -> tuple[list[dict], dict]:
    indexed = {(int(row["seed"]), row["model"]): row for row in rows}
    contrasts = []
    for seed in config["training"]["seeds"]:
        reference = indexed[(int(seed), "shared")]
        candidate = indexed[(int(seed), "balanced")]
        delta_leakage = candidate["z_nuis_physics_probe_auc_mean"] - reference["z_nuis_physics_probe_auc_mean"]
        delta_bias = candidate["max_abs_mu_bias"] - reference["max_abs_mu_bias"]
        delta_auc = candidate["physics_auc"] - reference["physics_auc"]
        material = delta_leakage <= config["decision_rules"]["material_leakage_improvement"]
        contrasts.append({
            "shard": config["confirmation_shard"], "seed": int(seed),
            "reference": "shared", "candidate": "balanced",
            "delta_leakage_auc": delta_leakage,
            "delta_max_abs_mu_bias": delta_bias,
            "delta_physics_auc": delta_auc,
            "material_leakage_improvement": material,
            "physics_auc_preserved": delta_auc >= config["decision_rules"]["physics_auc_min_delta"],
            "likelihood_bias_improved": delta_bias < 0.0,
            "discordant_cleaner_but_not_better": material and delta_bias >= 0.0,
        })
    material = [row for row in contrasts if row["material_leakage_improvement"]]
    discordant = [row for row in material if row["discordant_cleaner_but_not_better"]]
    fraction = len(discordant) / max(len(material), 1)
    rules = config["decision_rules"]
    complete = len(rows) == rules["required_complete_rows"] and len(contrasts) == rules["required_paired_transitions"]
    finite = all(
        math.isfinite(float(value))
        for row in rows for value in row.values() if isinstance(value, (int, float))
    )
    physics_preserved = all(row["physics_auc_preserved"] for row in contrasts)
    confirmed = (
        complete and finite and physics_preserved
        and len(material) >= rules["minimum_material_transitions"]
        and fraction >= rules["proxy_failure_min_discordant_fraction"]
    )
    summary = {
        "experiment": config["experiment"],
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "confirmation_shard": config["confirmation_shard"],
        "rows": len(rows), "paired_transitions": len(contrasts),
        "material_leakage_improvement_transitions": len(material),
        "discordant_cleaner_but_not_better": len(discordant),
        "discordant_fraction": fraction,
        "physics_auc_preserved_all_pairs": physics_preserved,
        "complete": complete, "finite": finite,
        "independent_proxy_failure_confirmed": confirmed,
        "decision": "proxy_failure_confirmed_on_shard002" if confirmed else "proxy_failure_not_confirmed_on_shard002",
        "confirmation_shard_accessed": True,
        "independent_test_accessed": True,
        "boundary": config["boundaries"],
    }
    return contrasts, summary


def validate_development_decision_logic(config: dict) -> None:
    rows = read_csv(resolve(config["source_e102_summary"]).parent / "metric_likelihood_rows.csv")
    selected = []
    for row in rows:
        if row["shard"] == "000" and row["model"] in {"shared", "balanced"}:
            selected.append({key: (float(value) if key not in {"shard", "seed", "model", "checkpoint", "probe_physics_values", "probe_domain_values"} else value) for key, value in row.items()})
            selected[-1]["seed"] = int(row["seed"])
    contrasts, summary = summarize(selected, {**config, "confirmation_shard": "000"})
    if len(contrasts) != 3 or summary["material_leakage_improvement_transitions"] != 3:
        raise RuntimeError("development decision-logic validation did not reproduce three material transitions")
    if summary["discordant_cleaner_but_not_better"] != 1 or not summary["independent_proxy_failure_confirmed"]:
        raise RuntimeError("development decision-logic validation did not reproduce the frozen shard000 branch")


def validate(config: dict, require_untouched: bool) -> list[dict]:
    if config["confirmation_shard"] != "002":
        raise RuntimeError("E103 is locked to shard002")
    source_summary = resolve(config["source_e102_summary"])
    if sha256(source_summary) != config["source_e102_summary_sha256"]:
        raise RuntimeError("E102 source summary hash mismatch")
    source = json.loads(source_summary.read_text())
    if source["decision"] != "unlock_separate_shard002_confirmation_design" or source["confirmation_shard_accessed"]:
        raise RuntimeError("E102 did not cleanly unlock E103")
    for relative, expected in config["dependency_sha256"].items():
        if sha256(resolve(relative)) != expected:
            raise RuntimeError(f"dependency hash mismatch: {relative}")
    rows = selected_manifest_rows(config)
    data_root = Path(config["data_root"]).resolve()
    data_dir = Path(config["data_dir"]).resolve()
    cache_dir = Path(config["cache_dir"]).resolve()
    if data_root not in data_dir.parents or data_root not in cache_dir.parents:
        raise RuntimeError("raw/cache path escaped the data disk")
    check_disk(data_root, int(config["minimum_data_disk_free_bytes"]))
    check_disk(ROOT, int(config["minimum_system_disk_free_bytes"]))
    if require_untouched:
        existing = []
        for row in rows:
            raw = data_dir / row["filename"]
            cache = cache_dir / row["filename"].removesuffix(".gz")
            if raw.exists() or cache.exists():
                existing.append(str(raw if raw.exists() else cache))
        if existing:
            raise RuntimeError(f"shard002 is not untouched: {existing}")
    validate_development_decision_logic(config)
    return rows


def run_command(command: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"command failed with exit {code}: {' '.join(command)}")


def new_run(before: set[Path], pattern: str) -> Path:
    after = set((ROOT / "runs").glob(pattern))
    created = sorted(after - before)
    if len(created) != 1:
        raise RuntimeError(f"expected one new run matching {pattern}, found {created}")
    return created[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    manifest_rows = validate(config, require_untouched=True)
    if args.preflight_only:
        print("E103 preflight passed: shard002 absent; manifest, disks, dependencies, E102 unlock, and decision logic valid")
        return 0

    run_dir = e75e.create_run_dir("e103-shard002-proxy-failure-confirmation")
    (run_dir / "status.json").write_text(json.dumps({
        "status": "running", "confirmation_shard_accessed": False, "independent_test_accessed": False
    }, indent=2) + "\n")
    selected_manifest = run_dir / "record80030_selected_shard002.csv"
    e75e.write_csv(selected_manifest, manifest_rows)
    groups = []
    try:
        provenance = []
        for row in manifest_rows:
            path = e75e.download_file(row, Path(config["data_dir"]))
            actual = adler32(path)
            if actual != row["checksum"]:
                raise RuntimeError(f"checksum mismatch for {path}: {actual} != {row['checksum']}")
            provenance.append({"filename": row["filename"], "size_bytes": path.stat().st_size, "checksum": actual})
        (run_dir / "download_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        (run_dir / "status.json").write_text(json.dumps({
            "status": "running", "confirmation_shard_accessed": True, "independent_test_accessed": True
        }, indent=2) + "\n")

        train = config["training"]
        index_names = [row["index_description"] for row in manifest_rows]
        for seed in train["seeds"]:
            before = set((ROOT / "runs").glob("*-e79-toptag-score-template-export*"))
            command = [
                sys.executable, "scripts/e79_toptag_score_template_export.py",
                "--manifest", str(selected_manifest), "--indices", *index_names,
                "--data-dir", config["data_dir"], "--cache-dir", config["cache_dir"],
                "--max-events-per-domain", str(train["max_events_per_domain"]),
                "--max-constituents", str(train["max_constituents"]), "--seed", str(seed),
                "--epochs", str(train["epochs"]), "--batch-size", str(train["batch_size"]),
                "--learning-rate", str(train["learning_rate"]), "--val-ratio", str(train["val_ratio"]),
                "--orth-lambda", str(train["orth_lambda"]), "--bins", str(train["bins"]),
            ]
            run_command(command, run_dir / f"e79_seed{seed}.log")
            e79_run = new_run(before, "*-e79-toptag-score-template-export*")
            before = set((ROOT / "runs").glob("*-e81-toptag-profile-stress*"))
            run_command([
                sys.executable, "scripts/e81_toptag_profile_stress.py", "--e79-run-dir", str(e79_run)
            ], run_dir / f"e81_seed{seed}.log")
            e81_run = new_run(before, "*-e81-toptag-profile-stress*")
            groups.append({"seed": int(seed), "e79_run": str(e79_run.relative_to(ROOT)), "e81_run": str(e81_run.relative_to(ROOT))})

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rows = []
        evaluation_config = {
            "data_root": config["data_root"], "val_ratio": train["val_ratio"],
            "batch_size": config["probes"]["batch_size"], "probe_epochs": config["probes"]["epochs"],
            "probe_seeds": config["probes"]["seeds"],
        }
        for group in groups:
            e79_run = resolve(group["e79_run"])
            data = e102.load_group_data({"seed": group["seed"], "e79_run": group["e79_run"]}, evaluation_config)
            specs = [
                {"model":"shared", "checkpoint":e79_run / "constituent_shared_baseline.pt", "nuisance_latent_dim":64,
                 "template_run":e79_run, "template_candidate":"constituent_shared_baseline", "profile_candidate":"constituent_shared_baseline"},
                {"model":"balanced", "checkpoint":e79_run / "constituent_balanced.pt", "nuisance_latent_dim":32,
                 "template_run":e79_run, "template_candidate":"constituent_balanced", "profile_candidate":"constituent_balanced"},
            ]
            for spec in specs:
                result = e102.evaluate_checkpoint(spec, data, evaluation_config, device)
                rows.append({
                    "shard": config["confirmation_shard"], "seed": group["seed"], "model": spec["model"],
                    **result, **e102.template_metrics(e79_run, spec["template_candidate"]),
                    **e102.profile_metrics(resolve(group["e81_run"]), spec["profile_candidate"]),
                    "checkpoint": str(spec["checkpoint"]),
                })
            del data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        contrasts, summary = summarize(rows, config)
        e102.write_csv(run_dir / "metric_likelihood_rows.csv", rows)
        e102.write_csv(run_dir / "paired_contrasts.csv", contrasts)
        (run_dir / "training_runs.json").write_text(json.dumps(groups, indent=2) + "\n")
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (run_dir / "status.json").write_text(json.dumps({
            "status": "completed", "confirmation_shard_accessed": True, "independent_test_accessed": True
        }, indent=2) + "\n")
        REPORTS.mkdir(parents=True, exist_ok=True)
        report = REPORTS / f"e103_shard002_proxy_failure_confirmation_{dt.datetime.now():%Y%m%d}.md"
        report.write_text("\n".join([
            "# E103 untouched-shard002 proxy-failure confirmation", "",
            f"- run: `{run_dir.relative_to(ROOT)}`", f"- decision: `{summary['decision']}`",
            f"- material leakage improvements: `{summary['material_leakage_improvement_transitions']}/3`",
            f"- cleaner but not better: `{summary['discordant_cleaner_but_not_better']}/{summary['material_leakage_improvement_transitions']}`",
            f"- physics AUC preserved in all pairs: `{summary['physics_auc_preserved_all_pairs']}`", "",
            "This is a one-shot confirmation on event shard002 within the same TopTag workflow. It does not identify a superior replacement proxy or establish universal transfer.",
        ]) + "\n")
        print(f"E103 done: {run_dir}")
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        (run_dir / "status.json").write_text(json.dumps({
            "status": "implementation_invalid_or_incomplete", "error": str(exc),
            "confirmation_shard_accessed": any(Path(config["data_dir"]).joinpath(row["filename"]).exists() for row in manifest_rows),
            "independent_test_accessed": any(Path(config["data_dir"]).joinpath(row["filename"]).exists() for row in manifest_rows),
        }, indent=2) + "\n")
        (run_dir / "traceback.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
