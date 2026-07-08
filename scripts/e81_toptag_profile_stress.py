#!/usr/bin/env python3
"""E81 profile TopTag pyhf workspaces on shifted pseudo-data templates."""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np

import e75e_toptag_systematic_family_scaleup as e75e
import e80_toptag_pyhf_workspace_smoke as e80


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PSEUDO_DOMAINS = ["nominal", "esup", "esdown", "cer", "cpos", "bias"]
MODELED_DOMAINS = {"nominal", "esup", "esdown"}


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def pseudo_observation(rows: list[dict], candidate: str, domain: str, count_floor: float) -> list[float]:
    signal = e80.template_counts(rows, candidate, domain, 1, count_floor)
    background = e80.template_counts(rows, candidate, domain, 0, count_floor)
    return [sig + bkg for sig, bkg in zip(signal, background)]


def summarize(rows: list[dict]) -> list[dict]:
    summary_rows = []
    for candidate in sorted({row["candidate"] for row in rows}):
        candidate_rows = [row for row in rows if row["candidate"] == candidate]
        stress_rows = [row for row in candidate_rows if row["pseudo_domain"] != "nominal"]
        modeled_rows = [row for row in stress_rows if row["stress_type"] == "modeled"]
        unmodeled_rows = [row for row in stress_rows if row["stress_type"] == "unmodeled"]
        for group, selected in [
            ("all_shifted", stress_rows),
            ("modeled_shifted", modeled_rows),
            ("unmodeled_shifted", unmodeled_rows),
        ]:
            if not selected:
                continue
            mu_bias = np.array([float(row["mu_bias"]) for row in selected], dtype=float)
            theta = np.array([float(row["energy_scale_hat"]) for row in selected], dtype=float)
            summary_rows.append(
                {
                    "candidate": candidate,
                    "group": group,
                    "n_domains": len(selected),
                    "mean_abs_mu_bias": float(np.mean(np.abs(mu_bias))),
                    "max_abs_mu_bias": float(np.max(np.abs(mu_bias))),
                    "rms_mu_bias": float(math.sqrt(np.mean(mu_bias * mu_bias))),
                    "max_abs_energy_scale_hat": float(np.max(np.abs(theta))),
                }
            )
    return summary_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--e79-run-dir",
        type=Path,
        default=ROOT / "runs" / "20260703-121216-e79-toptag-score-template-export",
    )
    parser.add_argument("--count-floor", type=float, default=1e-6)
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    run_dir = e75e.create_run_dir("e81-toptag-profile-stress")
    template_path = args.e79_run_dir / "score_templates.csv"
    rows = read_rows(template_path)
    candidates = sorted({row["candidate"] for row in rows})
    fit_rows = []

    for candidate in candidates:
        base_spec = e80.build_workspace(rows, candidate, args.count_floor)
        (run_dir / f"workspace_nominal_{candidate}.json").write_text(
            json.dumps(base_spec, indent=2), encoding="utf-8"
        )
        for domain in PSEUDO_DOMAINS:
            spec = copy.deepcopy(base_spec)
            spec["observations"][0]["data"] = pseudo_observation(rows, candidate, domain, args.count_floor)
            fit = e80.fit_workspace(spec)
            fit_rows.append(
                {
                    "candidate": candidate,
                    "pseudo_domain": domain,
                    "stress_type": "modeled" if domain in MODELED_DOMAINS else "unmodeled",
                    "mu_hat": fit["mu_hat"],
                    "mu_bias": fit["mu_hat"] - 1.0,
                    "energy_scale_hat": fit["energy_scale_hat"],
                    "z_mu0_pseudodata": fit["z_nominal_splusb"],
                    "delta_twice_nll_mu1": fit["delta_twice_nll_mu1"],
                    "n_bins": fit["n_bins"],
                    "n_observations": fit["n_observations"],
                }
            )

    summary_rows = summarize(fit_rows)
    e75e.write_csv(run_dir / "profile_stress_by_domain.csv", fit_rows)
    e75e.write_csv(run_dir / "profile_stress_summary.csv", summary_rows)
    config = {
        "e79_run_dir": str(args.e79_run_dir),
        "template_path": str(template_path),
        "count_floor": args.count_floor,
        "modeled_domains": sorted(MODELED_DOMAINS),
        "pseudo_domains": PSEUDO_DOMAINS,
        "stress_policy": "Fit the nominal esup/esdown histosys workspace to shifted validation templates; cer/cpos/bias are evaluated as additional stress domains.",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "status.txt").write_text("completed\n", encoding="utf-8")

    report_path = REPORTS / f"e81_toptag_profile_stress_{dt.datetime.now():%Y%m%d}.md"
    lines = [
        "# E81 TopTag Profile Stress",
        "",
        f"- run_dir: `{run_dir}`",
        f"- generated_at: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- e79_run_dir: `{args.e79_run_dir}`",
        "- modeled shifted domains: `esup/esdown` via the `energy_scale` histosys modifier.",
        "- additional stress domains: `cer/cpos/bias`.",
        "",
        "## Aggregate stress",
        "",
        "| candidate | group | mean |mu bias| | max |mu bias| | RMS mu bias | max |theta_es| |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {candidate} | {group} | {mean_bias} | {max_bias} | {rms} | {theta} |".format(
                candidate=row["candidate"],
                group=row["group"],
                mean_bias=format_float(row["mean_abs_mu_bias"]),
                max_bias=format_float(row["max_abs_mu_bias"]),
                rms=format_float(row["rms_mu_bias"]),
                theta=format_float(row["max_abs_energy_scale_hat"]),
            )
        )
    lines.extend(
        [
            "",
            "## Domain fits",
            "",
            "| candidate | pseudo-domain | stress | mu_hat | mu bias | theta_es | delta 2NLL(mu=1) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in fit_rows:
        lines.append(
            "| {candidate} | {domain} | {stress} | {mu} | {bias} | {theta} | {dnll} |".format(
                candidate=row["candidate"],
                domain=row["pseudo_domain"],
                stress=row["stress_type"],
                mu=format_float(row["mu_hat"]),
                bias=format_float(row["mu_bias"]),
                theta=format_float(row["energy_scale_hat"]),
                dnll=format_float(row["delta_twice_nll_mu1"]),
            )
        )
    lines.extend(
        [
            "",
            "## Run Note",
            "",
            "Single-seed profile-stress benchmark evaluating E79 score templates in a profiled likelihood with shifted validation templates.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"E81 TopTag profile stress done: {run_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
