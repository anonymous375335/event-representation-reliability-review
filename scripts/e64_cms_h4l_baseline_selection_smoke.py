#!/usr/bin/env python3
"""E64 CMS H->4l official-like baseline selection smoke test.

This is a direct, lightweight Python/uproot port of the public `skim.cxx`
selection logic for a bounded number of events per sample. It is intended to
validate the analysis path before full baseline reproduction.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
from uproot.source.xrootd import MultithreadedXRootDSource


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
LOCAL_ROOT_FILES = ROOT / "data_raw" / "cms_h4l_2012_reduced_nanoaod" / "root_files"
BASE = "root://eospublic.cern.ch//eos/opendata/cms/derived-data/AOD2NanoAODOutreachTool/ForHiggsTo4Leptons"

SAMPLES = {
    "SMHiggsToZZTo4L": ["FourMuons", "FourElectrons", "TwoMuonsTwoElectrons"],
    "ZZTo4mu": ["FourMuons"],
    "ZZTo4e": ["FourElectrons"],
    "ZZTo2e2mu": ["TwoMuonsTwoElectrons"],
    "Run2012B_DoubleMuParked": ["FourMuons", "TwoMuonsTwoElectrons"],
    "Run2012C_DoubleMuParked": ["FourMuons", "TwoMuonsTwoElectrons"],
    "Run2012B_DoubleElectron": ["FourElectrons", "TwoMuonsTwoElectrons"],
    "Run2012C_DoubleElectron": ["FourElectrons", "TwoMuonsTwoElectrons"],
}

LOCAL_EXPECTED_BYTES = {
    "SMHiggsToZZTo4L": 42400229,
    "ZZTo4mu": 184624480,
    "ZZTo4e": 161660567,
    "ZZTo2e2mu": 173163793,
    "Run2012B_DoubleMuParked": 3179316378,
    "Run2012C_DoubleMuParked": 4463307037,
    "Run2012B_DoubleElectron": 1817796893,
    "Run2012C_DoubleElectron": 2767819825,
}

INTEGRATED_LUMINOSITY = 11.58 * 1000.0
SCALE_FACTOR_ZZ_TO_4L = 1.386
EVENT_WEIGHTS = {
    "SMHiggsToZZTo4L": 0.0065 / 299973.0 * INTEGRATED_LUMINOSITY,
    "ZZTo4mu": 0.077 / 1499064.0 * SCALE_FACTOR_ZZ_TO_4L * INTEGRATED_LUMINOSITY,
    "ZZTo4e": 0.077 / 1499093.0 * SCALE_FACTOR_ZZ_TO_4L * INTEGRATED_LUMINOSITY,
    "ZZTo2e2mu": 0.18 / 1497445.0 * SCALE_FACTOR_ZZ_TO_4L * INTEGRATED_LUMINOSITY,
    "Run2012B_DoubleMuParked": 1.0,
    "Run2012C_DoubleMuParked": 1.0,
    "Run2012B_DoubleElectron": 1.0,
    "Run2012C_DoubleElectron": 1.0,
}

BASE_BRANCHES = [
    "run",
]

MUON_BRANCHES = [
    "nMuon",
    "Muon_pt",
    "Muon_eta",
    "Muon_phi",
    "Muon_mass",
    "Muon_charge",
    "Muon_pfRelIso04_all",
    "Muon_dxy",
    "Muon_dxyErr",
    "Muon_dz",
    "Muon_dzErr",
]

ELECTRON_BRANCHES = [
    "nElectron",
    "Electron_pt",
    "Electron_eta",
    "Electron_phi",
    "Electron_mass",
    "Electron_charge",
    "Electron_pfRelIso03_all",
    "Electron_dxy",
    "Electron_dxyErr",
    "Electron_dz",
    "Electron_dzErr",
]

HIST_RANGES = {
    "Higgs_mass": (36, 70.0, 180.0),
    "Z1_mass": (36, 40.0, 160.0),
    "Z2_mass": (36, 12.0, 160.0),
}


def delta_phi(phi1: float, phi2: float) -> float:
    value = phi1 - phi2
    while value > math.pi:
        value -= 2 * math.pi
    while value <= -math.pi:
        value += 2 * math.pi
    return value


def delta_r(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    return math.hypot(eta1 - eta2, delta_phi(phi1, phi2))


def four_vector(pt: float, eta: float, phi: float, mass: float) -> tuple[float, float, float, float]:
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    pz = pt * math.sinh(eta)
    energy = math.sqrt(max(px * px + py * py + pz * pz + mass * mass, 0.0))
    return energy, px, py, pz


def add_vectors(vectors: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return tuple(sum(vector[i] for vector in vectors) for i in range(4))


def vector_mass(vector: tuple[float, float, float, float]) -> float:
    energy, px, py, pz = vector
    return math.sqrt(max(energy * energy - px * px - py * py - pz * pz, 0.0))


def pair_vector(pt, eta, phi, mass, i: int, j: int) -> tuple[float, float, float, float]:
    return add_vectors([four_vector(pt[i], eta[i], phi[i], mass[i]), four_vector(pt[j], eta[j], phi[j], mass[j])])


def reconstruct_samekind(pt, eta, phi, mass, charge):
    best = None
    for i in range(4):
        for j in range(i + 1, 4):
            if int(charge[i]) == int(charge[j]):
                continue
            candidate = pair_vector(pt, eta, phi, mass, i, j)
            candidate_mass = vector_mass(candidate)
            distance = abs(91.2 - candidate_mass)
            if best is None or distance < best[0]:
                best = (distance, i, j, candidate)
    if best is None:
        return None
    _, i, j, z1 = best
    rest = [idx for idx in range(4) if idx not in {i, j}]
    if len(rest) != 2:
        return None
    if delta_r(float(eta[i]), float(phi[i]), float(eta[j]), float(phi[j])) < 0.02:
        return None
    if delta_r(float(eta[rest[0]]), float(phi[rest[0]]), float(eta[rest[1]]), float(phi[rest[1]])) < 0.02:
        return None
    z2 = pair_vector(pt, eta, phi, mass, rest[0], rest[1])
    z_vectors = [z1, z2]
    z_vectors.sort(key=lambda vector: abs(vector_mass(vector) - 91.2))
    return z_vectors


def reconstruct_2mu2e(mu_pt, mu_eta, mu_phi, mu_mass, el_pt, el_eta, el_phi, el_mass):
    z_mu = pair_vector(mu_pt, mu_eta, mu_phi, mu_mass, 0, 1)
    z_el = pair_vector(el_pt, el_eta, el_phi, el_mass, 0, 1)
    z_vectors = [z_mu, z_el]
    z_vectors.sort(key=lambda vector: abs(vector_mass(vector) - 91.2))
    return z_vectors


def all_abs_less(values, threshold: float) -> bool:
    return all(abs(float(value)) < threshold for value in values)


def all_less(values, threshold: float) -> bool:
    return all(float(value) < threshold for value in values)


def muon_sip_ok(row) -> bool:
    for dxy, dz, dxy_err, dz_err in zip(row.Muon_dxy, row.Muon_dz, row.Muon_dxyErr, row.Muon_dzErr):
        denom = math.sqrt(float(dxy_err) ** 2 + float(dz_err) ** 2)
        if denom == 0:
            return False
        ip3d = math.sqrt(float(dxy) ** 2 + float(dz) ** 2)
        if ip3d / denom >= 4 or abs(float(dxy)) >= 0.5 or abs(float(dz)) >= 1.0:
            return False
    return True


def electron_sip_ok(row) -> bool:
    for dxy, dz, dxy_err, dz_err in zip(row.Electron_dxy, row.Electron_dz, row.Electron_dxyErr, row.Electron_dzErr):
        denom = math.sqrt(float(dxy_err) ** 2 + float(dz_err) ** 2)
        if denom == 0:
            return False
        ip3d = math.sqrt(float(dxy) ** 2 + float(dz) ** 2)
        if ip3d / denom >= 4 or abs(float(dxy)) >= 0.5 or abs(float(dz)) >= 1.0:
            return False
    return True


def sip_mask(dxy, dz, dxy_err, dz_err):
    denom = np.sqrt(dxy_err * dxy_err + dz_err * dz_err)
    ip3d = np.sqrt(dxy * dxy + dz * dz)
    return ak.all((denom > 0) & ((ip3d / denom) < 4) & (abs(dxy) < 0.5) & (abs(dz) < 1.0), axis=1)


def prefilter_arrays(arrays, final_state: str):
    """Vectorized version of the official minimal selection before reconstruction."""
    if final_state == "FourMuons":
        mask = (
            (arrays.nMuon == 4)
            & ak.all(abs(arrays.Muon_pfRelIso04_all) < 0.40, axis=1)
            & ak.all(arrays.Muon_pt > 5, axis=1)
            & ak.all(abs(arrays.Muon_eta) < 2.4, axis=1)
            & sip_mask(arrays.Muon_dxy, arrays.Muon_dz, arrays.Muon_dxyErr, arrays.Muon_dzErr)
            & (ak.sum(arrays.Muon_charge == 1, axis=1) == 2)
            & (ak.sum(arrays.Muon_charge == -1, axis=1) == 2)
        )
    elif final_state == "FourElectrons":
        mask = (
            (arrays.nElectron == 4)
            & ak.all(abs(arrays.Electron_pfRelIso03_all) < 0.40, axis=1)
            & ak.all(arrays.Electron_pt > 7, axis=1)
            & ak.all(abs(arrays.Electron_eta) < 2.5, axis=1)
            & sip_mask(arrays.Electron_dxy, arrays.Electron_dz, arrays.Electron_dxyErr, arrays.Electron_dzErr)
            & (ak.sum(arrays.Electron_charge == 1, axis=1) == 2)
            & (ak.sum(arrays.Electron_charge == -1, axis=1) == 2)
        )
    elif final_state == "TwoMuonsTwoElectrons":
        mu_pt_sorted = ak.fill_none(ak.pad_none(ak.sort(arrays.Muon_pt, ascending=False), 2, clip=True), -1.0)
        el_pt_sorted = ak.fill_none(ak.pad_none(ak.sort(arrays.Electron_pt, ascending=False), 2, clip=True), -1.0)
        has_two = (arrays.nElectron >= 2) & (arrays.nMuon >= 2)
        mask = (
            has_two
            & ak.all(abs(arrays.Electron_eta) < 2.5, axis=1)
            & ak.all(abs(arrays.Muon_eta) < 2.4, axis=1)
            & (((mu_pt_sorted[:, 0] > 20) & (mu_pt_sorted[:, 1] > 10)) | ((el_pt_sorted[:, 0] > 20) & (el_pt_sorted[:, 1] > 10)))
            & ak.all(abs(arrays.Electron_pfRelIso03_all) < 0.40, axis=1)
            & ak.all(abs(arrays.Muon_pfRelIso04_all) < 0.40, axis=1)
            & sip_mask(arrays.Electron_dxy, arrays.Electron_dz, arrays.Electron_dxyErr, arrays.Electron_dzErr)
            & sip_mask(arrays.Muon_dxy, arrays.Muon_dz, arrays.Muon_dxyErr, arrays.Muon_dzErr)
            & (ak.sum(arrays.Electron_charge, axis=1) == 0)
            & (ak.sum(arrays.Muon_charge, axis=1) == 0)
        )
    else:
        raise ValueError(f"Unknown final state {final_state}")
    return arrays[mask]


def select_event(row, final_state: str):
    if final_state == "FourMuons":
        if int(row.nMuon) != 4:
            return None
        if not all_abs_less(row.Muon_pfRelIso04_all, 0.40):
            return None
        if not all(float(value) > 5 for value in row.Muon_pt):
            return None
        if not all_abs_less(row.Muon_eta, 2.4):
            return None
        if not muon_sip_ok(row):
            return None
        charges = [int(value) for value in row.Muon_charge]
        if charges.count(1) != 2 or charges.count(-1) != 2:
            return None
        z_vectors = reconstruct_samekind(row.Muon_pt, row.Muon_eta, row.Muon_phi, row.Muon_mass, row.Muon_charge)
    elif final_state == "FourElectrons":
        if int(row.nElectron) != 4:
            return None
        if not all_abs_less(row.Electron_pfRelIso03_all, 0.40):
            return None
        if not all(float(value) > 7 for value in row.Electron_pt):
            return None
        if not all_abs_less(row.Electron_eta, 2.5):
            return None
        if not electron_sip_ok(row):
            return None
        charges = [int(value) for value in row.Electron_charge]
        if charges.count(1) != 2 or charges.count(-1) != 2:
            return None
        z_vectors = reconstruct_samekind(row.Electron_pt, row.Electron_eta, row.Electron_phi, row.Electron_mass, row.Electron_charge)
    elif final_state == "TwoMuonsTwoElectrons":
        if int(row.nElectron) < 2 or int(row.nMuon) < 2:
            return None
        if not all_abs_less(row.Electron_eta, 2.5) or not all_abs_less(row.Muon_eta, 2.4):
            return None
        mu_pt_sorted = sorted([float(value) for value in row.Muon_pt], reverse=True)
        el_pt_sorted = sorted([float(value) for value in row.Electron_pt], reverse=True)
        if not ((mu_pt_sorted[0] > 20 and mu_pt_sorted[1] > 10) or (el_pt_sorted[0] > 20 and el_pt_sorted[1] > 10)):
            return None
        if delta_r(float(row.Muon_eta[0]), float(row.Muon_phi[0]), float(row.Muon_eta[1]), float(row.Muon_phi[1])) < 0.02:
            return None
        if delta_r(float(row.Electron_eta[0]), float(row.Electron_phi[0]), float(row.Electron_eta[1]), float(row.Electron_phi[1])) < 0.02:
            return None
        if not all_abs_less(row.Electron_pfRelIso03_all, 0.40) or not all_abs_less(row.Muon_pfRelIso04_all, 0.40):
            return None
        if not electron_sip_ok(row) or not muon_sip_ok(row):
            return None
        if sum(int(value) for value in row.Electron_charge) != 0 or sum(int(value) for value in row.Muon_charge) != 0:
            return None
        z_vectors = reconstruct_2mu2e(
            row.Muon_pt,
            row.Muon_eta,
            row.Muon_phi,
            row.Muon_mass,
            row.Electron_pt,
            row.Electron_eta,
            row.Electron_phi,
            row.Electron_mass,
        )
    else:
        raise ValueError(f"Unknown final state {final_state}")

    if z_vectors is None:
        return None
    z1_mass = vector_mass(z_vectors[0])
    z2_mass = vector_mass(z_vectors[1])
    if not (40 < z1_mass < 120 and 12 < z2_mass < 120):
        return None
    higgs_mass = vector_mass(add_vectors(z_vectors))
    return {
        "run": int(row.run),
        "Higgs_mass": higgs_mass,
        "Z1_mass": z1_mass,
        "Z2_mass": z2_mass,
    }


def source_for_sample(sample: str) -> tuple[str, str]:
    local_path = LOCAL_ROOT_FILES / f"{sample}.root"
    expected_size = LOCAL_EXPECTED_BYTES.get(sample)
    local_ok = local_path.exists() and (expected_size is None or local_path.stat().st_size == expected_size)
    url = str(local_path) if local_ok else f"{BASE}/{sample}.root"
    source_mode = "local" if local_ok else "remote_root"
    return url, source_mode


def branches_for_final_states(final_states: list[str]) -> list[str]:
    branches = list(BASE_BRANCHES)
    if any(final_state in {"FourMuons", "TwoMuonsTwoElectrons"} for final_state in final_states):
        branches.extend(MUON_BRANCHES)
    if any(final_state in {"FourElectrons", "TwoMuonsTwoElectrons"} for final_state in final_states):
        branches.extend(ELECTRON_BRANCHES)
    return list(dict.fromkeys(branches))


def load_sample_arrays(sample: str, final_states: list[str], entry_start: int, entry_stop: int) -> tuple[ak.Array, int, str, str]:
    url, source_mode = source_for_sample(sample)
    branches = branches_for_final_states(final_states)
    open_options = {"timeout": 300}
    if source_mode == "remote_root":
        open_options["handler"] = MultithreadedXRootDSource
    with uproot.open(url, **open_options) as root_file:
        tree = root_file["Events"]
        num_entries = int(tree.num_entries)
        start = min(entry_start, num_entries)
        stop = min(entry_stop, num_entries) if entry_stop else num_entries
        if stop < start:
            raise ValueError(f"entry_stop {entry_stop} is before entry_start {entry_start}")
        arrays = tree.arrays(branches, entry_start=start, entry_stop=stop)
    return arrays, stop, source_mode, url


def process_arrays(sample: str, final_state: str, arrays: ak.Array, entry_start: int, stop: int, source_mode: str, url: str) -> dict:
    prefiltered = prefilter_arrays(arrays, final_state)
    selected = []
    for row in ak.to_list(prefiltered):
        result = select_event(type("Row", (), row), final_state)
        if result is not None:
            selected.append(result)

    weight = EVENT_WEIGHTS[sample]
    histograms = {}
    for variable, (bins, low, high) in HIST_RANGES.items():
        values = np.array([item[variable] for item in selected], dtype=float)
        hist, edges = np.histogram(values, bins=bins, range=(low, high), weights=np.full(len(values), weight))
        histograms[variable] = {
            "bin_edges": edges.tolist(),
            "weighted_counts": hist.tolist(),
            "raw_counts": np.histogram(values, bins=bins, range=(low, high))[0].tolist(),
        }
    return {
        "sample": sample,
        "final_state": final_state,
        "source_mode": source_mode,
        "source": url,
        "entry_start": entry_start,
        "entry_stop": stop,
        "prefiltered_events": len(prefiltered),
        "raw_selected": len(selected),
        "event_weight": weight,
        "weighted_selected": len(selected) * weight,
        "mass_summary": {
            "Higgs_mass_min": min((item["Higgs_mass"] for item in selected), default=None),
            "Higgs_mass_median": float(np.median([item["Higgs_mass"] for item in selected])) if selected else None,
            "Higgs_mass_max": max((item["Higgs_mass"] for item in selected), default=None),
        },
        "histograms": histograms,
    }


def process_sample(sample: str, final_state: str, entry_start: int, entry_stop: int) -> dict:
    arrays, stop, source_mode, url = load_sample_arrays(sample, [final_state], entry_start, entry_stop)
    return process_arrays(sample, final_state, arrays, entry_start, stop, source_mode, url)


def write_csvs(run_dir: Path, results: list[dict]) -> tuple[Path, Path]:
    yields_path = run_dir / "yields.csv"
    with yields_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample",
                "final_state",
                "source_mode",
                "entry_start",
                "entry_stop",
                "prefiltered_events",
                "raw_selected",
                "event_weight",
                "weighted_selected",
                "Higgs_mass_median",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "sample": row["sample"],
                    "final_state": row["final_state"],
                    "source_mode": row["source_mode"],
                    "entry_start": row["entry_start"],
                    "entry_stop": row["entry_stop"],
                    "prefiltered_events": row["prefiltered_events"],
                    "raw_selected": row["raw_selected"],
                    "event_weight": row["event_weight"],
                    "weighted_selected": row["weighted_selected"],
                    "Higgs_mass_median": row["mass_summary"]["Higgs_mass_median"],
                }
            )

    hist_path = run_dir / "histograms.csv"
    with hist_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample", "final_state", "variable", "bin_low", "bin_high", "raw_count", "weighted_count"],
        )
        writer.writeheader()
        for row in results:
            for variable, histogram in row["histograms"].items():
                edges = histogram["bin_edges"]
                for i, count in enumerate(histogram["raw_counts"]):
                    writer.writerow(
                        {
                            "sample": row["sample"],
                            "final_state": row["final_state"],
                            "variable": variable,
                            "bin_low": edges[i],
                            "bin_high": edges[i + 1],
                            "raw_count": count,
                            "weighted_count": histogram["weighted_counts"][i],
                        }
                    )
    return yields_path, hist_path


def write_report(run_dir: Path, results: list[dict], entry_start: int, entry_stop: int, yields_path: Path, hist_path: Path) -> Path:
    path = REPORTS / f"e64_cms_h4l_baseline_selection_smoke_{dt.datetime.now():%Y%m%d}.md"
    sample_scope = "full available sample" if entry_stop == 0 else "bounded event prefix"
    lines = [
        "# E64 CMS H4l Baseline Selection Smoke",
        "",
        f"- run_dir: `{run_dir}`",
        f"- generated_at: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- entry_start: {entry_start}",
        f"- entry_stop_per_sample: {entry_stop}",
        f"- mode: Python/uproot port of public `skim.cxx` selection on {sample_scope}",
        f"- yields_csv: `{yields_path}`",
        f"- histograms_csv: `{hist_path}`",
        "",
        "## Yield Smoke Table",
        "",
        "| sample | final_state | source | prefiltered | raw selected | weighted selected | median m4l |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in results:
        median = row["mass_summary"]["Higgs_mass_median"]
        median_text = "n/a" if median is None else f"{median:.3f}"
        lines.append(
            f"| {row['sample']} | {row['final_state']} | {row['source_mode']} | {row['prefiltered_events']} | {row['raw_selected']} | "
            f"{row['weighted_selected']:.6g} | {median_text} |"
        )
    lines.extend(
        [
            "",
        "## Interpretation",
            "",
            "- Selection and reconstruction smoke test for the E64 workflow.",
            "- The next E64 step is to extend the same workflow to the remaining samples or a documented chunked fraction.",
            "- Main-text integration uses the generated yields, mass distributions, and data/MC comparison.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def append_failures_to_report(report_path: Path, failures: list[dict]) -> None:
    if not failures:
        return
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n## Failures\n\n")
        for failure in failures:
            handle.write(f"- `{failure['sample']}:{failure['final_state']}`: {failure['error']}\n")


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry-start", type=int, default=0)
    parser.add_argument("--entry-stop", type=int, default=50_000)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Restrict to one or more Sample:FinalState combinations.",
    )
    parser.add_argument(
        "--read-mode",
        choices=["per-sample", "per-final-state"],
        default="per-sample",
        help="Read each sample once, or read minimal branches for each final state.",
    )
    args = parser.parse_args()

    run_dir = create_run_dir("e64-cms-h4l-baseline-selection-smoke")
    REPORTS.mkdir(parents=True, exist_ok=True)

    requested = set(args.only)
    results = []
    failures = []
    if args.read_mode == "per-final-state":
        for sample, final_states in SAMPLES.items():
            for final_state in final_states:
                if requested and f"{sample}:{final_state}" not in requested:
                    continue
                print(f"[select] {sample} {final_state}", flush=True)
                try:
                    result = process_sample(sample, final_state, args.entry_start, args.entry_stop)
                    results.append(result)
                    print(
                        "[done] {sample} {final_state}: source={source_mode}, prefiltered={prefiltered_events}, "
                        "selected={raw_selected}, weighted={weighted_selected:.6g}".format(**result),
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    failure = {"sample": sample, "final_state": final_state, "error": repr(exc)}
                    failures.append(failure)
                    print(f"[failed] {sample} {final_state}: {failure['error']}", flush=True)
    else:
        for sample, final_states in SAMPLES.items():
            active_final_states = [
                final_state
                for final_state in final_states
                if not requested or f"{sample}:{final_state}" in requested
            ]
            if not active_final_states:
                continue
            print(f"[load] {sample}: {','.join(active_final_states)}", flush=True)
            try:
                arrays, stop, source_mode, url = load_sample_arrays(
                    sample, active_final_states, args.entry_start, args.entry_stop
                )
                print(f"[loaded] {sample}: source={source_mode}, entries={stop}", flush=True)
            except Exception as exc:  # noqa: BLE001 - preserve load failure for each requested final state.
                for final_state in active_final_states:
                    failure = {"sample": sample, "final_state": final_state, "error": repr(exc)}
                    failures.append(failure)
                    print(f"[failed] {sample} {final_state}: {failure['error']}", flush=True)
                continue
            for final_state in active_final_states:
                print(f"[select] {sample} {final_state}", flush=True)
                try:
                    result = process_arrays(sample, final_state, arrays, args.entry_start, stop, source_mode, url)
                    results.append(result)
                    print(
                        "[done] {sample} {final_state}: source={source_mode}, prefiltered={prefiltered_events}, "
                        "selected={raw_selected}, weighted={weighted_selected:.6g}".format(**result),
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001 - preserve failure and keep auditing other channels.
                    failure = {"sample": sample, "final_state": final_state, "error": repr(exc)}
                    failures.append(failure)
                    print(f"[failed] {sample} {final_state}: {failure['error']}", flush=True)

    yields_path, hist_path = write_csvs(run_dir, results)
    report_path = write_report(run_dir, results, args.entry_start, args.entry_stop, yields_path, hist_path)
    append_failures_to_report(report_path, failures)
    manifest = {
        "experiment": "E64 CMS H4l baseline selection smoke",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "entry_start": args.entry_start,
        "entry_stop_per_sample": args.entry_stop,
        "results": results,
        "failures": failures,
        "yields_csv": str(yields_path),
        "histograms_csv": str(hist_path),
        "report": str(report_path),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    status = "partial" if failures else "done"
    (run_dir / "status.txt").write_text(f"status: {status}\nreport: {report_path}\nmanifest: {run_dir / 'manifest.json'}\n", encoding="utf-8")
    print(f"E64 baseline selection smoke {status}: {run_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
