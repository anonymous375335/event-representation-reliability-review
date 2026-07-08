#!/usr/bin/env python3

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"


COLORS = {
    "shared": "#7884B4",
    "split": "#0F4D92",
    "split_soft": "#B4C0E4",
    "nuis": "#F0C0CC",
    "domain": "#42949E",
    "control": "#767676",
    "warning": "#B64342",
    "positive": "#2E9E44",
    "neutral": "#CFCECE",
    "ink": "#272727",
    "grid": "#E8EAF0",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.4,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "legend.fontsize": 6.2,
        "legend.frameon": False,
        "figure.dpi": 180,
    }
)


def source(name):
    return pd.read_csv(FIGURES / name)


def save(fig, stem, width_mm=183, height_mm=None):
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax, label):
    ax.text(
        -0.15,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def style_ygrid(ax):
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)


def metric_row(df, metric, model):
    row = df[(df["metric"] == metric) & (df["model"] == model)].iloc[0]
    return float(row["mean"]), float(row["sd"])


def fmt_value(value, pattern):
    return pattern.format(value)


def paired_slope(ax, values, errors, ylabel, title, ylim=None, reference=None, value_format="{:.3f}"):
    x = np.array([0.0, 1.0])
    ax.plot(x, values, color=COLORS["control"], linewidth=1.0, zorder=1)
    ax.errorbar(
        x,
        values,
        yerr=errors,
        fmt="o",
        color=COLORS["ink"],
        markerfacecolor="white",
        markeredgecolor=COLORS["ink"],
        markeredgewidth=0.9,
        markersize=4.2,
        elinewidth=0.8,
        capsize=2.0,
        zorder=2,
    )
    ax.scatter([0], [values[0]], s=28, color=COLORS["shared"], zorder=3)
    ax.scatter([1], [values[1]], s=28, color=COLORS["split"], zorder=3)
    for xi, yi, ha, dx in [(0, values[0], "left", 0.055), (1, values[1], "left", 0.055)]:
        ax.text(xi + dx, yi, fmt_value(yi, value_format), ha=ha, va="center", fontsize=6.1, color=COLORS["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels(["shared", "split"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=3)
    if ylim:
        ax.set_ylim(*ylim)
    if reference is not None:
        ax.axhline(reference, color=COLORS["control"], linewidth=0.8, linestyle=(0, (3, 2)))
    style_ygrid(ax)


def figure1_workflow():
    fig, ax = plt.subplots(figsize=(7.2, 4.05))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x, y, w, h, text, fc, ec=COLORS["ink"], fontsize=6.8, weight="normal"):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            linewidth=0.75,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=weight)
        return patch

    def arrow(a, b, yoff=0, color=COLORS["ink"]):
        ax.add_patch(
            FancyArrowPatch(
                a,
                b,
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.8,
                color=color,
                connectionstyle=f"arc3,rad={yoff}",
            )
        )

    ax.text(0.03, 0.93, "Two-channel reliability protocol", fontsize=9.4, fontweight="bold")
    ax.text(
        0.03,
        0.875,
        "Branch routing is evaluated as a representation diagnostic alongside likelihood-facing checks",
        fontsize=6.8,
        color=COLORS["control"],
    )

    # Light group panels make the flow readable without putting cards inside cards.
    ax.text(0.05, 0.795, "inputs", fontsize=6.5, fontweight="bold", color=COLORS["control"])
    ax.text(0.30, 0.795, "representation", fontsize=6.5, fontweight="bold", color=COLORS["control"])
    ax.text(0.56, 0.795, "branch routing", fontsize=6.5, fontweight="bold", color=COLORS["control"])
    ax.text(0.83, 0.795, "readouts", fontsize=6.5, fontweight="bold", color=COLORS["control"])

    box(0.04, 0.62, 0.18, 0.12, "CMS H4l\nopen data", "#F5F6F8")
    box(0.04, 0.43, 0.18, 0.12, "TopTag\nsystematic shards", "#F3F7F7", COLORS["domain"])
    box(0.04, 0.24, 0.18, 0.12, "visible-domain\nchecks", "#F7F7F7", COLORS["control"])

    box(0.29, 0.62, 0.19, 0.12, "four-lepton tensors\n+ frozen EveNet", "#F5F6F8")
    box(0.29, 0.43, 0.19, 0.12, "constituents\n+ DeepSets encoder", "#F3F7F7", COLORS["domain"])
    box(0.29, 0.24, 0.19, 0.12, "matched H4l\nvariants", "#F7F7F7", COLORS["control"])

    box(0.56, 0.63, 0.19, 0.10, "shared baseline", "#EEF0FA", COLORS["shared"], weight="bold")
    box(0.56, 0.45, 0.19, 0.10, "split candidate", "#EDF4F8", COLORS["split"], weight="bold")
    box(0.545, 0.28, 0.10, 0.095, "$z_{phys}$\nphysics task", "#E9F0FA", COLORS["split"], fontsize=6.5)
    box(0.665, 0.28, 0.10, 0.095, "$z_{nuis}$\nresponse", "#F6F1F3", COLORS["nuis"], fontsize=6.5)

    box(0.83, 0.63, 0.14, 0.09, "physics AUC", "#F7F7F7", COLORS["control"], fontsize=6.4)
    box(0.83, 0.50, 0.14, 0.09, "$z_{nuis}$ leakage", "#F7F7F7", COLORS["control"], fontsize=6.4)
    box(0.83, 0.37, 0.14, 0.09, "domain readout", "#F7F7F7", COLORS["control"], fontsize=6.4)
    box(0.83, 0.24, 0.14, 0.09, "likelihood\nboundary", "#F7F7F7", COLORS["control"], fontsize=6.4)

    for y in [0.68, 0.49, 0.30]:
        arrow((0.22, y), (0.29, y))
        arrow((0.48, y), (0.56, 0.68 if y == 0.68 else 0.50 if y == 0.49 else 0.33), yoff=0.02)

    arrow((0.75, 0.68), (0.83, 0.675))
    arrow((0.75, 0.50), (0.83, 0.545))
    arrow((0.75, 0.50), (0.83, 0.415), yoff=-0.08)
    arrow((0.765, 0.33), (0.83, 0.285), yoff=0.05)

    ax.text(
        0.55,
        0.18,
        "Controls: shuffled domains, injected labels, rank checks",
        fontsize=6.5,
        color=COLORS["control"],
    )
    save(fig, "figure1_workflow_schematic")


def figure2_h4l_branch_routing():
    df = source("source_data_figure2_h4l_branch_routing.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.85), gridspec_kw={"wspace": 0.36, "hspace": 0.52})
    specs = [
        ("physics_auc", "Physics task is preserved", "AUC", (0.9878, 0.9902), None, "{:.4f}"),
        ("z_nuis_physics_auc", "$z_{nuis}$ physics leakage decreases", "probe AUC", (0.50, 1.00), 0.5, "{:.3f}"),
        ("z_nuis_domain_acc", "$z_{nuis}$ remains domain-readable", "probe accuracy", (0.18, 0.27), 0.2, "{:.3f}"),
        ("score_domain_drift_max", "Score drift stays small", "max mean-score drift", (0.0, 0.0026), None, "{:.4f}"),
    ]
    for ax, (metric, title, ylabel, ylim, ref, value_format), label in zip(axes.flat, specs, "abcd"):
        rows = [
            metric_row(df, metric, "shared_baseline"),
            metric_row(df, metric, "split_candidate"),
        ]
        vals, errs = [row[0] for row in rows], [row[1] for row in rows]
        paired_slope(ax, vals, errs, ylabel, title, ylim, ref, value_format)
        panel_label(ax, label)
    axes[0, 1].annotate(
        "large reduction",
        xy=(1, metric_row(df, "z_nuis_physics_auc", "split_candidate")[0]),
        xytext=(0.45, 0.72),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "-", "linewidth": 0.65, "color": COLORS["control"]},
        fontsize=6.4,
        color=COLORS["control"],
    )
    handles = [
        plt.Line2D([], [], marker="o", color="none", markerfacecolor=COLORS["shared"], markeredgecolor=COLORS["shared"], markersize=5),
        plt.Line2D([], [], marker="o", color="none", markerfacecolor=COLORS["split"], markeredgecolor=COLORS["split"], markersize=5),
    ]
    fig.legend(handles, ["shared baseline", "split candidate"], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(top=0.88)
    save(fig, "figure2_h4l_branch_routing")


def figure3_h4l_controls():
    df = source("source_data_figure3_h4l_controls.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.85), gridspec_kw={"wspace": 0.42})
    ax = axes[0]
    rows = df[df["panel"] == "probe_control"].copy()
    names = ["$z_{phys}$", "$z_{nuis}$", "shuffled", "injected", "teacher"]
    vals = rows["value"].to_numpy()
    ypos = np.arange(len(vals))[::-1]
    colors = [COLORS["split_soft"], COLORS["split"], COLORS["control"], COLORS["warning"], COLORS["domain"]]
    for y, val, col in zip(ypos, vals, colors):
        ax.plot([0.2, val], [y, y], color=COLORS["grid"], linewidth=1.0, zorder=1)
        ax.scatter(val, y, s=28, color=col, zorder=2)
        ax.text(val + 0.018, y, f"{val:.3f}" if val < 0.999 else "1.0", va="center", fontsize=5.9)
    ax.axvline(0.2, color=COLORS["control"], linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.set_xlim(0.16, 1.08)
    ax.set_xlabel("domain-probe accuracy")
    ax.set_title("Probe controls", loc="left", pad=3)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)
    panel_label(ax, "a")

    ax = axes[1]
    leak = float(df[(df["panel"] == "physics_leakage_control") & (df["metric"] == "z_nuis_physics_probe_auc")]["value"].iloc[0])
    vals = [1.0, leak]
    ypos = np.array([1, 0])
    labels = ["injected control", "$z_{nuis}$ physics"]
    for y, val, col in zip(ypos, vals, [COLORS["warning"], COLORS["split"]]):
        ax.plot([0.5, val], [y, y], color=COLORS["grid"], linewidth=1.0)
        ax.scatter(val, y, s=30, color=col, zorder=2)
        ax.text(val + 0.018, y, f"{val:.3f}" if val < 0.999 else "1.0", va="center", fontsize=5.9)
    ax.axvline(0.5, color=COLORS["control"], linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.45, 1.08)
    ax.set_xlabel("physics-probe AUC")
    ax.set_title("Capacity is sufficient", loc="left", pad=3)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)
    panel_label(ax, "b")

    ax = axes[2]
    rank = df[df["panel"] == "rank"].copy()
    phys_train = float(rank[rank["metric"] == "train_z_phys_effective_rank"]["value"].iloc[0])
    phys_val = float(rank[rank["metric"] == "val_z_phys_effective_rank"]["value"].iloc[0])
    nuis_train = float(rank[rank["metric"] == "train_z_nuis_effective_rank"]["value"].iloc[0])
    nuis_val = float(rank[rank["metric"] == "val_z_nuis_effective_rank"]["value"].iloc[0])
    for y, vals, col, label in [
        (1, [phys_train, phys_val], COLORS["split_soft"], "$z_{phys}$"),
        (0, [nuis_train, nuis_val], COLORS["split"], "$z_{nuis}$"),
    ]:
        ax.plot(vals, [y, y], color=col, linewidth=1.1)
        ax.scatter(vals[0], y, s=28, color="white", edgecolor=col, linewidth=1.0, zorder=2, label="train" if y == 1 else None)
        ax.scatter(vals[1], y, s=28, color=col, zorder=2, label="validation" if y == 1 else None)
        ax.text(vals[1] + 0.16, y, f"{vals[1]:.2f}", va="center", fontsize=6.0, color=COLORS["ink"])
    ax.set_xlim(0, 7.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["$z_{nuis}$", "$z_{phys}$"])
    ax.set_xlabel("effective rank")
    ax.set_title("No latent collapse", loc="left", pad=3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.02), borderaxespad=0)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)
    panel_label(ax, "c")
    save(fig, "figure3_h4l_controls")


def figure4_h4l_boundary_checks():
    df = source("source_data_figure4_h4l_boundary_checks.csv")
    overlay = source("source_data_figure4_control_region_overlay.csv")
    fig = plt.figure(figsize=(7.2, 5.25))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.35], wspace=0.36, hspace=0.62)
    ax = fig.add_subplot(gs[0, 0])
    like = df[(df["panel"] == "likelihood") & (df["metric"] == "asimov_z")]
    order = ["baseline_score", "m4l", "mass_penalty_score"]
    labels = ["score", "$m_{4l}$", "mass-penalty score"]
    vals = [float(like[like["template"] == key]["value"].iloc[0]) for key in order]
    errs = [float(like[like["template"] == key]["sd"].iloc[0]) for key in order]
    ypos = np.arange(3)[::-1]
    for y, val, err, col in zip(ypos, vals, errs, [COLORS["control"], COLORS["positive"], COLORS["nuis"]]):
        ax.plot([0, val], [y, y], color=COLORS["grid"], linewidth=1.1)
        ax.errorbar(val, y, xerr=err, fmt="o", color=col, markersize=4.2, elinewidth=0.8, capsize=2.0)
        ax.text(val + 0.08, y, f"{val:.2f}", va="center", fontsize=6.0)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Asimov separation $Z$")
    ax.set_title("$m_{4l}$ remains stronger", loc="left", pad=3)
    ax.set_xlim(0, 3.25)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    obs = df[(df["panel"] == "control_region") & (df["template"] == "all_not_deduplicated")]
    observed = float(obs[obs["metric"] == "observed_higgs_window"]["value"].iloc[0])
    scaled = float(obs[obs["metric"] == "scaled_mc_higgs_window"]["value"].iloc[0])
    vals = [observed, scaled]
    labels = ["observed H window", "scaled MC H window"]
    ypos = [1, 0]
    for y, val, col in zip(ypos, vals, [COLORS["warning"], COLORS["split"]]):
        ax.plot([0, val], [y, y], color=COLORS["grid"], linewidth=1.1)
        ax.scatter(val, y, s=30, color=col, zorder=2)
        ax.text(val + 0.45, y, f"{val:.1f}", va="center", fontsize=6.0)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("events")
    ax.set_title("Observed-data sanity", loc="left", pad=3)
    ax.set_xlim(0, 20)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, :])
    lows = overlay["bin_low"].to_numpy(float)
    highs = overlay["bin_high"].to_numpy(float)
    centers = 0.5 * (lows + highs)
    widths = highs - lows
    observed_counts = overlay["observed_count"].to_numpy(float)
    observed_err = overlay["observed_err"].to_numpy(float)
    mc_bkg = overlay["mc_bkg_sideband_scaled"].to_numpy(float)
    mc_total = overlay["mc_total_sideband_scaled"].to_numpy(float)
    ax.axvspan(115.0, 135.0, color=COLORS["warning"], alpha=0.09, label="Higgs window, not fitted")
    ax.bar(
        centers,
        mc_bkg,
        width=widths,
        align="center",
        color=COLORS["split"],
        alpha=0.35,
        edgecolor="none",
        label="MC ZZ background, sideband normalized",
    )
    ax.step(
        np.r_[lows[0], highs],
        np.r_[mc_total[0], mc_total],
        where="pre",
        color=COLORS["split"],
        linewidth=1.2,
        label="MC total, same normalization",
    )
    ax.errorbar(
        centers,
        observed_counts,
        yerr=observed_err,
        fmt="o",
        color=COLORS["ink"],
        markersize=2.8,
        elinewidth=0.8,
        capsize=0,
        label="Observed data",
    )
    ax.set_xlim(70, 180)
    ax.set_ylim(-0.8, 16.2)
    ax.set_xlabel("$m_{4l}$ [GeV]")
    ax.set_ylabel("events / bin")
    ax.set_title("Control-region overlay, sideband normalized", loc="left", pad=3)
    ax.legend(loc="upper right", ncol=1, frameon=True, framealpha=0.95, borderpad=0.4, handlelength=1.4)
    style_ygrid(ax)
    panel_label(ax, "c")
    save(fig, "figure4_h4l_boundary_checks")


def figure5_toptag_transfer():
    df = source("source_data_figure5_toptag_transfer.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.85), gridspec_kw={"wspace": 0.42})
    specs = [
        ("physics_auc", "Tagging task is preserved", "AUC", (0.948, 0.954), None, "{:.4f}"),
        ("z_nuis_physics_auc", "$z_{nuis}$ physics leakage decreases", "probe AUC", (0.55, 0.96), 0.5, "{:.3f}"),
        ("z_nuis_domain_acc", "$z_{nuis}$ domain readout increases", "probe accuracy", (0.16, 0.25), None, "{:.3f}"),
    ]
    main = df[df["panel"] == "e76b_multiseed"]
    for ax, (metric, title, ylabel, ylim, ref, value_format), label in zip(axes, specs, "abc"):
        rows = main[main["metric"] == metric]
        shared = rows[rows["model"] == "shared_baseline"].iloc[0]
        split = rows[rows["model"] == "split_candidate"].iloc[0]
        paired_slope(
            ax,
            [float(shared["mean"]), float(split["mean"])],
            [float(shared["sd"]), float(split["sd"])],
            ylabel,
            title,
            ylim,
            ref,
            value_format,
        )
        panel_label(ax, label)
    held = df[(df["panel"] == "e77_holdout") & (df["metric"] == "holdout_z_nuis_physics_auc")]
    shared_h = float(held[held["model"] == "heldout_shared_baseline"]["mean"].iloc[0])
    split_h = float(held[held["model"] == "heldout_split_candidate"]["mean"].iloc[0])
    axes[1].plot(
        [0.13, 1.13],
        [shared_h, split_h],
        color=COLORS["warning"],
        marker="o",
        markerfacecolor="white",
        markeredgewidth=0.9,
        markersize=3.4,
        linewidth=0.9,
        linestyle=(0, (2, 2)),
    )
    axes[1].set_xlim(-0.45, 1.35)
    axes[1].text(
        0.52,
        0.90,
        "held-out\nsystematic",
        transform=axes[1].transAxes,
        fontsize=6.1,
        color=COLORS["warning"],
        ha="left",
        va="top",
    )
    save(fig, "figure5_toptag_transfer")


def main():
    figure1_workflow()
    figure2_h4l_branch_routing()
    figure3_h4l_controls()
    figure4_h4l_boundary_checks()
    figure5_toptag_transfer()


if __name__ == "__main__":
    main()
