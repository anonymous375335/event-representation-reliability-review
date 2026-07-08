#!/usr/bin/env python3
"""Export selected CMS H4l MC candidates as EveNet-compatible event tensors."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import awkward as ak
import numpy as np

import e64_cms_h4l_baseline_selection_smoke as smoke


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
DATA_PROCESSED = ROOT / "data_processed" / "cms_h4l_e65"

DEFAULT_MC_COMBOS = [
    "SMHiggsToZZTo4L:FourMuons",
    "SMHiggsToZZTo4L:FourElectrons",
    "SMHiggsToZZTo4L:TwoMuonsTwoElectrons",
    "ZZTo4mu:FourMuons",
    "ZZTo4e:FourElectrons",
    "ZZTo2e2mu:TwoMuonsTwoElectrons",
]

FINAL_STATE_IDS = {
    "FourMuons": 0,
    "FourElectrons": 1,
    "TwoMuonsTwoElectrons": 2,
}

FEATURE_NAMES = [
    "pt",
    "eta",
    "phi",
    "mass",
    "charge",
    "pfreliso",
    "dxy_sig",
    "dz_sig",
    "is_muon",
    "is_electron",
]

CONDITION_NAMES = [
    "Higgs_mass",
    "Z1_mass",
    "Z2_mass",
    "final_state_id",
    "event_weight",
]


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


def parse_combo(combo: str) -> tuple[str, str]:
    if ":" not in combo:
        raise ValueError(f"combo must use Sample:FinalState, got {combo!r}")
    sample, final_state = combo.split(":", 1)
    allowed = [(name, state) for name, states in smoke.SAMPLES.items() for state in states]
    if (sample, final_state) not in allowed:
        raise ValueError(f"unknown combo {combo!r}")
    return sample, final_state


def safe_sig(value: float, err: float) -> float:
    return float(value) / float(err) if float(err) > 0 else 0.0


def make_lepton(pt, eta, phi, mass, charge, iso, dxy, dxy_err, dz, dz_err, is_muon: bool) -> list[float]:
    return [
        float(pt),
        float(eta),
        float(phi),
        float(mass),
        float(charge),
        float(iso),
        safe_sig(dxy, dxy_err),
        safe_sig(dz, dz_err),
        1.0 if is_muon else 0.0,
        0.0 if is_muon else 1.0,
    ]


def sorted_by_pt(leptons: list[list[float]]) -> list[list[float]]:
    return sorted(leptons, key=lambda item: item[0], reverse=True)


def tensor_for_row(row: dict, final_state: str) -> np.ndarray:
    leptons: list[list[float]] = []
    if final_state in {"FourMuons", "TwoMuonsTwoElectrons"}:
        muon_limit = 4 if final_state == "FourMuons" else 2
        for index in range(muon_limit):
            leptons.append(
                make_lepton(
                    row["Muon_pt"][index],
                    row["Muon_eta"][index],
                    row["Muon_phi"][index],
                    row["Muon_mass"][index],
                    row["Muon_charge"][index],
                    row["Muon_pfRelIso04_all"][index],
                    row["Muon_dxy"][index],
                    row["Muon_dxyErr"][index],
                    row["Muon_dz"][index],
                    row["Muon_dzErr"][index],
                    is_muon=True,
                )
            )
    if final_state in {"FourElectrons", "TwoMuonsTwoElectrons"}:
        electron_limit = 4 if final_state == "FourElectrons" else 2
        for index in range(electron_limit):
            leptons.append(
                make_lepton(
                    row["Electron_pt"][index],
                    row["Electron_eta"][index],
                    row["Electron_phi"][index],
                    row["Electron_mass"][index],
                    row["Electron_charge"][index],
                    row["Electron_pfRelIso03_all"][index],
                    row["Electron_dxy"][index],
                    row["Electron_dxyErr"][index],
                    row["Electron_dz"][index],
                    row["Electron_dzErr"][index],
                    is_muon=False,
                )
            )
    leptons = sorted_by_pt(leptons)
    if len(leptons) != 4:
        raise ValueError(f"expected 4 leptons for {final_state}, got {len(leptons)}")
    return np.asarray(leptons, dtype=np.float32)


def process_combo(sample: str, final_state: str, entry_start: int, entry_stop: int) -> tuple[list[np.ndarray], list[dict]]:
    arrays, stop, source_mode, source = smoke.load_sample_arrays(sample, [final_state], entry_start, entry_stop)
    entry_numbers = np.arange(entry_start, stop, dtype=np.int64)
    arrays = ak.with_field(arrays, entry_numbers, "_entry")
    prefiltered = smoke.prefilter_arrays(arrays, final_state)

    tensors = []
    rows = []
    is_signal = sample == "SMHiggsToZZTo4L"
    final_state_id = FINAL_STATE_IDS[final_state]
    weight = smoke.EVENT_WEIGHTS[sample]
    for row in ak.to_list(prefiltered):
        selected = smoke.select_event(type("Row", (), row), final_state)
        if selected is None:
            continue
        tensors.append(tensor_for_row(row, final_state))
        rows.append(
            {
                "sample": sample,
                "final_state": final_state,
                "source_mode": source_mode,
                "source": source,
                "entry": int(row["_entry"]),
                "physics_label": 1 if is_signal else 0,
                "subprocess_id": final_state_id,
                "event_weight": weight,
                "Higgs_mass": selected["Higgs_mass"],
                "Z1_mass": selected["Z1_mass"],
                "Z2_mass": selected["Z2_mass"],
            }
        )
    return tensors, rows


def write_metadata(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "sample",
            "final_state",
            "source_mode",
            "entry",
            "physics_label",
            "subprocess_id",
            "event_weight",
            "Higgs_mass",
            "Z1_mass",
            "Z2_mass",
            "source",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combo", action="append", default=[], help="Sample:FinalState combo; repeatable.")
    parser.add_argument("--entry-start", type=int, default=0)
    parser.add_argument("--entry-stop", type=int, default=0)
    parser.add_argument("--output-name", default="cms_h4l_mc_candidates_e65.npz")
    args = parser.parse_args()

    combos = args.combo or DEFAULT_MC_COMBOS
    run_dir = create_run_dir("e65-cms-h4l-event-tensor-export")
    REPORTS.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    tensors = []
    rows = []
    for combo in combos:
        sample, final_state = parse_combo(combo)
        print(f"[tensor-export] {sample}:{final_state}", flush=True)
        combo_tensors, combo_rows = process_combo(sample, final_state, args.entry_start, args.entry_stop)
        tensors.extend(combo_tensors)
        rows.extend(combo_rows)

    features = np.stack(tensors, axis=0).astype(np.float32)
    valid_masks = np.ones(features.shape[:2], dtype=np.float32)
    labels = np.asarray([row["physics_label"] for row in rows], dtype=np.int64)
    subprocess_ids = np.asarray([row["subprocess_id"] for row in rows], dtype=np.int64)
    conditions = np.asarray(
        [
            [row["Higgs_mass"], row["Z1_mass"], row["Z2_mass"], row["subprocess_id"], row["event_weight"]]
            for row in rows
        ],
        dtype=np.float32,
    )
    conditions_mask = np.ones((features.shape[0],), dtype=np.float32)

    output_path = DATA_PROCESSED / args.output_name
    np.savez_compressed(
        output_path,
        features=features,
        valid_masks=valid_masks,
        labels=labels,
        subprocess_ids=subprocess_ids,
        conditions=conditions,
        conditions_mask=conditions_mask,
        feature_names=np.asarray(FEATURE_NAMES),
        condition_names=np.asarray(CONDITION_NAMES),
    )
    metadata_csv = run_dir / "event_metadata.csv"
    write_metadata(metadata_csv, rows)

    class_counts = {
        "signal": int((labels == 1).sum()),
        "zz_background": int((labels == 0).sum()),
    }
    final_state_counts = {
        name: int((subprocess_ids == final_state_id).sum()) for name, final_state_id in FINAL_STATE_IDS.items()
    }
    manifest = {
        "experiment": "E65 CMS H4l event tensor export",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "output_npz": str(output_path),
        "metadata_csv": str(metadata_csv),
        "combos": combos,
        "features_shape": list(features.shape),
        "valid_masks_shape": list(valid_masks.shape),
        "conditions_shape": list(conditions.shape),
        "feature_names": FEATURE_NAMES,
        "condition_names": CONDITION_NAMES,
        "class_counts": class_counts,
        "final_state_counts": final_state_counts,
        "status": "done",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "status.txt").write_text(f"status: done\noutput_npz: {output_path}\n", encoding="utf-8")

    report_path = REPORTS / f"e65_cms_h4l_event_tensor_export_{dt.datetime.now():%Y%m%d}.md"
    lines = [
        "# E65 CMS H4l Event Tensor Export",
        "",
        f"- run_dir: `{run_dir}`",
        f"- generated_at: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- output_npz: `{output_path}`",
        f"- metadata_csv: `{metadata_csv}`",
        f"- features_shape: `{tuple(features.shape)}`",
        f"- conditions_shape: `{tuple(conditions.shape)}`",
        "",
        "## Feature Schema",
        "",
        "- particle axis: four selected leptons sorted by descending pT",
        f"- feature_names: `{', '.join(FEATURE_NAMES)}`",
        f"- condition_names: `{', '.join(CONDITION_NAMES)}`",
        "",
        "## Counts",
        "",
        f"- signal: {class_counts['signal']}",
        f"- zz_background: {class_counts['zz_background']}",
        f"- FourMuons: {final_state_counts['FourMuons']}",
        f"- FourElectrons: {final_state_counts['FourElectrons']}",
        f"- TwoMuonsTwoElectrons: {final_state_counts['TwoMuonsTwoElectrons']}",
        "",
        "## Interpretation",
        "",
        "- This is an EveNet-compatible tensor adapter feasibility artifact for MC candidates.",
        "- Truth labels are exported only for MC. Real data use the same feature schema without physics labels for supervised training.",
        "- The tensor includes kinematic and quality features needed for representation/readout tests, but it remains a simplified open-data H4l benchmark.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"E65 tensor export done: {run_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
