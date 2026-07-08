#!/usr/bin/env python3
"""E80 build a minimal pyhf workspace from E79 TopTag score templates."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path

import pyhf

import e75e_toptag_systematic_family_scaleup as e75e


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def scalar(value) -> float:
    return float(pyhf.tensorlib.tolist(value)[0])


def template_counts(rows: list[dict], candidate: str, domain: str, label: int, count_floor: float) -> list[float]:
    selected = [
        row
        for row in rows
        if row["candidate"] == candidate and row["domain"] == domain and int(row["label"]) == label
    ]
    selected = sorted(selected, key=lambda row: int(row["bin_index"]))
    if not selected:
        raise ValueError(f"missing template for candidate={candidate} domain={domain} label={label}")
    return [max(float(row["count"]), count_floor) for row in selected]


def build_workspace(rows: list[dict], candidate: str, count_floor: float) -> dict:
    signal_nominal = template_counts(rows, candidate, "nominal", 1, count_floor)
    background_nominal = template_counts(rows, candidate, "nominal", 0, count_floor)
    signal_hi = template_counts(rows, candidate, "esup", 1, count_floor)
    signal_lo = template_counts(rows, candidate, "esdown", 1, count_floor)
    background_hi = template_counts(rows, candidate, "esup", 0, count_floor)
    background_lo = template_counts(rows, candidate, "esdown", 0, count_floor)
    observed = [signal + background for signal, background in zip(signal_nominal, background_nominal)]
    return {
        "channels": [
            {
                "name": "score_region",
                "samples": [
                    {
                        "name": "signal",
                        "data": signal_nominal,
                        "modifiers": [
                            {"name": "mu", "type": "normfactor", "data": None},
                            {
                                "name": "energy_scale",
                                "type": "histosys",
                                "data": {"hi_data": signal_hi, "lo_data": signal_lo},
                            },
                        ],
                    },
                    {
                        "name": "background",
                        "data": background_nominal,
                        "modifiers": [
                            {
                                "name": "energy_scale",
                                "type": "histosys",
                                "data": {"hi_data": background_hi, "lo_data": background_lo},
                            }
                        ],
                    },
                ],
            }
        ],
        "observations": [{"name": "score_region", "data": observed}],
        "measurements": [
            {
                "name": "Measurement",
                "config": {
                    "poi": "mu",
                    "parameters": [
                        {"name": "mu", "bounds": [[0.0, 5.0]], "inits": [1.0]},
                        {"name": "energy_scale", "bounds": [[-5.0, 5.0]], "inits": [0.0]},
                    ],
                },
            }
        ],
        "version": "1.0.0",
    }


def fit_workspace(workspace_spec: dict) -> dict:
    workspace = pyhf.Workspace(workspace_spec)
    model = workspace.model()
    data = workspace.data(model)
    init = model.config.suggested_init()
    bounds = model.config.suggested_bounds()
    fixed = model.config.suggested_fixed()
    bestfit = pyhf.infer.mle.fit(data, model, init_pars=init, par_bounds=bounds, fixed_params=fixed)
    nll_best = scalar(pyhf.infer.mle.twice_nll(bestfit, data, model))
    fixed0 = pyhf.infer.mle.fixed_poi_fit(0.0, data, model, init_pars=init, par_bounds=bounds, fixed_params=fixed)
    fixed1 = pyhf.infer.mle.fixed_poi_fit(1.0, data, model, init_pars=init, par_bounds=bounds, fixed_params=fixed)
    nll0 = scalar(pyhf.infer.mle.twice_nll(fixed0, data, model))
    nll1 = scalar(pyhf.infer.mle.twice_nll(fixed1, data, model))
    q0 = max(nll0 - nll_best, 0.0)
    return {
        "mu_hat": float(bestfit[model.config.poi_index]),
        "energy_scale_hat": float(bestfit[model.config.par_names.index("energy_scale")]),
        "twice_nll_best": nll_best,
        "twice_nll_mu0": nll0,
        "twice_nll_mu1": nll1,
        "q0_nominal_splusb": q0,
        "z_nominal_splusb": math.sqrt(q0),
        "delta_twice_nll_mu1": nll1 - nll_best,
        "n_bins": int(model.config.channel_nbins["score_region"]),
        "n_observations": float(sum(workspace_spec["observations"][0]["data"])),
    }


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
    run_dir = e75e.create_run_dir("e80-toptag-pyhf-workspace-smoke")
    template_path = args.e79_run_dir / "score_templates.csv"
    rows = read_rows(template_path)
    candidates = sorted({row["candidate"] for row in rows})
    fit_rows = []

    for candidate in candidates:
        workspace_spec = build_workspace(rows, candidate, args.count_floor)
        workspace_path = run_dir / f"workspace_{candidate}.json"
        workspace_path.write_text(json.dumps(workspace_spec, indent=2), encoding="utf-8")
        fit = fit_workspace(workspace_spec)
        fit_rows.append(
            {
                "candidate": candidate,
                "workspace": workspace_path.name,
                **fit,
            }
        )

    e75e.write_csv(run_dir / "fit_summary.csv", fit_rows)
    config = {
        "e79_run_dir": str(args.e79_run_dir),
        "template_path": str(template_path),
        "count_floor": args.count_floor,
        "pyhf_version": pyhf.__version__,
        "systematic_policy": "Paired esup/esdown is encoded as histosys; cer/cpos/bias are kept as additional stress domains.",
        "observation_policy": "Observation is the nominal signal+background validation template from E79.",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "status.txt").write_text("completed\n", encoding="utf-8")

    report_path = REPORTS / f"e80_toptag_pyhf_workspace_smoke_{dt.datetime.now():%Y%m%d}.md"
    lines = [
        "# E80 TopTag pyhf Workspace Smoke",
        "",
        f"- run_dir: `{run_dir}`",
        f"- generated_at: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- pyhf_version: `{pyhf.__version__}`",
        f"- e79_run_dir: `{args.e79_run_dir}`",
        "- systematic policy: paired `esup/esdown` via histosys; `cer/cpos/bias` as additional stress domains.",
        "- observation: nominal signal+background validation template.",
        "",
        "## Fit smoke",
        "",
        "| candidate | mu_hat | energy_scale_hat | Z(mu=0, pseudo-data) | delta 2NLL(mu=1) | bins | observations |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fit_rows:
        lines.append(
            "| {candidate} | {mu} | {theta} | {z} | {dnll} | {bins} | {obs} |".format(
                candidate=row["candidate"],
                mu=format_float(row["mu_hat"]),
                theta=format_float(row["energy_scale_hat"]),
                z=format_float(row["z_nominal_splusb"]),
                dnll=format_float(row["delta_twice_nll_mu1"]),
                bins=row["n_bins"],
                obs=format_float(row["n_observations"]),
            )
        )
    lines.extend(
        [
            "",
            "## Run Note",
            "",
            "This run validates that E79 score templates can be represented as a pyhf/HistFactory-style workspace and fitted with a profiled paired nuisance.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"E80 TopTag pyhf workspace smoke done: {run_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
