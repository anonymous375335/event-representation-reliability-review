#!/usr/bin/env python3

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"


COLORS = {
    "shared": "#FEC47B",
    "split": "#5196C5",
    "split_soft": "#A2D2E6",
    "nuis": "#FEC47B",
    "domain": "#5196C5",
    "domain_soft": "#EAF5FB",
    "analysis": "#A2D2E6",
    "analysis_soft": "#F0F9FC",
    "accent": "#E1422E",
    "accent_soft": "#FCE6E2",
    "panel_bg": "#FFFDF4",
    "panel_header": "#5196C5",
    "control": "#6F7378",
    "warning": "#E1422E",
    "positive": "#5196C5",
    "positive_control": "#7E65A8",
    "positive_control_soft": "#F0ECF7",
    "neutral": "#D8DDE2",
    "ink": "#272727",
    "grid": "#E9EEF3",
}
MODEL_SHARED_COLOR = COLORS["accent"]
MODEL_SPLIT_COLOR = COLORS["split"]
M4L_REFERENCE_COLOR = "#7E65A8"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 9.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.2,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 8.2,
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


def panel_label(ax, label, x=-0.06, y=1.00):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=9.4,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def figure_panel_label(fig, ax, label, dx=0.028, dy=0.010):
    bbox = ax.get_position()
    fig.text(
        bbox.x0 - dx,
        bbox.y1 + dy,
        label,
        fontsize=9.4,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def style_ygrid(ax):
    ax.set_facecolor("#FCFBF8")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)


def style_xgrid(ax):
    ax.set_facecolor("#FCFBF8")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55)
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
        ax.text(xi + dx, yi, fmt_value(yi, value_format), ha=ha, va="center", fontsize=7.3, color=COLORS["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels(["shared\nbaseline", "split\ncandidate"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=3)
    if ylim:
        ax.set_ylim(*ylim)
    if reference is not None:
        ax.axhline(reference, color=COLORS["control"], linewidth=0.8, linestyle=(0, (3, 2)))
    style_ygrid(ax)


def comparison_strip(
    ax,
    shared,
    split,
    errors,
    xlabel,
    xlim,
    value_format="{:.3f}",
    reference=None,
    overlays=None,
    shared_color=None,
    split_color=None,
):
    shared_color = shared_color or COLORS["shared"]
    split_color = split_color or COLORS["split"]
    ax.set_facecolor("#FCFBF8")
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.55, 0.72)
    ax.set_yticks([])
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)
    right_label_transform = ax.get_yaxis_transform()
    if reference is not None:
        ax.axvline(reference, color=COLORS["control"], linewidth=0.75, linestyle=(0, (3, 2)))
    y_shared, y_split = 0.16, -0.14
    ax.errorbar(
        [shared],
        [y_shared],
        xerr=[errors[0]],
        fmt="o",
        color=shared_color,
        ecolor=shared_color,
        markerfacecolor=shared_color,
        markeredgecolor=shared_color,
        markersize=4.8,
        elinewidth=0.8,
        capsize=2.0,
        zorder=3,
    )
    ax.errorbar(
        [split],
        [y_split],
        xerr=[errors[1]],
        fmt="s",
        color=split_color,
        ecolor=split_color,
        markerfacecolor=split_color,
        markeredgecolor=split_color,
        markersize=4.8,
        elinewidth=0.8,
        capsize=2.0,
        zorder=3,
    )
    if overlays:
        for overlay in overlays:
            y = overlay.get("y", -0.30)
            ax.plot(
                [overlay["shared"], overlay["split"]],
                [y, y],
                color=overlay.get("color", COLORS["warning"]),
                linewidth=0.85,
                linestyle=overlay.get("linestyle", (0, (2, 2))),
                zorder=2,
            )
            marker = overlay.get("marker", "o")
            ax.scatter(
                [overlay["shared"], overlay["split"]],
                [y, y],
                s=20,
                marker=marker,
                facecolor=overlay.get("facecolor", "white"),
                edgecolor=overlay.get("color", COLORS["warning"]),
                linewidth=0.9,
                zorder=4,
            )
            ax.text(
                1.01,
                y,
                overlay["label"],
                transform=right_label_transform,
                ha="left",
                va="center",
                fontsize=8.2,
                color=overlay.get("color", COLORS["warning"]),
            )
    ax.text(1.01, y_shared, f"shared {value_format.format(shared)}", transform=right_label_transform, ha="left", va="center", fontsize=8.2, color=shared_color)
    ax.text(1.01, y_split, f"split {value_format.format(split)}", transform=right_label_transform, ha="left", va="center", fontsize=8.2, color=split_color)
    ax.set_xlabel(xlabel)


def ratio_residual_panel(ax, overlay):
    centers = 0.5 * (overlay["bin_low"].to_numpy(float) + overlay["bin_high"].to_numpy(float))
    observed = overlay["observed_count"].to_numpy(float)
    expected = overlay["mc_total_sideband_scaled"].to_numpy(float)
    err = overlay["observed_err"].to_numpy(float)
    residual = (observed - expected) / np.maximum(err, 1.0)
    in_window = overlay["in_higgs_window"].astype(str).str.lower().eq("true").to_numpy()
    ax.axhline(0, color=COLORS["control"], linewidth=0.8)
    ax.axvspan(115.0, 135.0, color=COLORS["warning"], alpha=0.055)
    ax.scatter(centers[~in_window], residual[~in_window], s=14, color=COLORS["split"], alpha=0.85, label="sideband")
    ax.scatter(centers[in_window], residual[in_window], s=16, color=COLORS["warning"], alpha=0.9, label="Higgs window")
    ax.text(171, 2.45, "sideband", fontsize=7.2, color=COLORS["split"], ha="right")
    ax.text(171, 1.95, "Higgs window", fontsize=7.2, color=COLORS["warning"], ha="right")
    ax.set_xlim(70, 180)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xlabel("$m_{4l}$ [GeV]")
    ax.set_ylabel("(data - MC) / stat. err.")
    style_ygrid(ax)


def appendix_likelihood_scan_panel(ax, scan_df, summary_df):
    styles = {
        "m4l": (M4L_REFERENCE_COLOR, "$m_{4l}$ reference", "-"),
        "shared_baseline": (MODEL_SHARED_COLOR, "shared score", "-"),
        "split_orth_adv": (MODEL_SPLIT_COLOR, "split $z_{phys}$ score", "-"),
    }
    for template, (color, label, linestyle) in styles.items():
        rows = scan_df[scan_df["template"] == template].sort_values("mu")
        ax.plot(
            rows["mu"].to_numpy(float),
            rows["minus2_delta_logl"].to_numpy(float),
            color=color,
            linewidth=1.45,
            linestyle=linestyle,
            label=label,
        )
        interval = summary_df[summary_df["template"] == template]
        low = float(interval["mu_lo_approx"].iloc[0])
        high = float(interval["mu_hi_approx"].iloc[0])
        ax.scatter([low, high], [1, 1], s=12, color=color, zorder=3)
    ax.axhline(1.0, color=COLORS["control"], linewidth=0.8, linestyle=(0, (3, 2)))
    ax.axvline(1.0, color=COLORS["control"], linewidth=0.7, linestyle=":")
    ax.text(
        0.98,
        0.18,
        "$m_{4l}$ remains\nthe analysis anchor",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.4,
        color=M4L_REFERENCE_COLOR,
    )
    ax.set_xlim(0.48, 1.62)
    ax.set_ylim(0, 1.35)
    ax.set_xlabel("signal strength $\\mu$")
    ax.set_ylabel("$-2\\Delta\\log L$")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        handlelength=1.35,
        columnspacing=0.95,
    )
    style_ygrid(ax)


def figure1_workflow():
    mass_template = source("source_data_figure4_likelihood_template_bins.csv")
    mass_template = mass_template[mass_template["template"] == "m4l"]
    mass_template = mass_template.groupby(["bin_low", "bin_high"], as_index=False)["signal"].sum()
    mass_x = 0.5 * (mass_template["bin_low"].to_numpy(float) + mass_template["bin_high"].to_numpy(float))
    mass_y = mass_template["signal"].to_numpy(float)
    mass_y = mass_y / mass_y.max()

    fig, ax = plt.subplots(figsize=(7.15, 5.60))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(
        x,
        y,
        w,
        h,
        text,
        fc,
        ec=COLORS["ink"],
        fontsize=7.3,
        weight="normal",
        color=COLORS["ink"],
        lw=1.1,
        radius=0.012,
    ):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.010,rounding_size={radius}",
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=weight,
            color=color,
            linespacing=1.15,
        )
        return patch

    def arrow(a, b, yoff=0, color=COLORS["ink"], lw=1.15, scale=9.0, style="-|>"):
        ax.add_patch(
            FancyArrowPatch(
                a,
                b,
                arrowstyle=style,
                mutation_scale=scale,
                linewidth=lw,
                color=color,
                connectionstyle=f"arc3,rad={yoff}",
            )
        )

    def panel_title(y, label, title):
        ax.text(0.035, y, label, fontsize=10.0, fontweight="bold", va="top", ha="left")
        ax.text(0.075, y, title, fontsize=9.2, fontweight="bold", va="top", ha="left")

    def reduction_card(x, title, start, end):
        y, w, h = 0.015, 0.190, 0.130
        box(x, y, w, h, "", "#FFFFFF", COLORS["neutral"], lw=0.95)
        ax.text(x + 0.018, y + 0.113, title, fontsize=7.1, fontweight="bold", ha="left", va="center")
        ax.text(x + 0.018, y + 0.080, "leakage AUC", fontsize=6.1, color=COLORS["control"], ha="left")
        x0, x1, yy = x + 0.045, x + 0.145, y + 0.032
        ax.scatter([x0], [yy], s=30, color=MODEL_SHARED_COLOR, marker="o", zorder=3)
        arrow((x0 + 0.012, yy), (x1 - 0.012, yy), color=COLORS["control"], lw=1.0, scale=7.5)
        ax.scatter([x1], [yy], s=30, color=MODEL_SPLIT_COLOR, marker="s", zorder=3)
        ax.text(x0, yy + 0.017, f"{start:.3f}", fontsize=6.5, color=MODEL_SHARED_COLOR, ha="center")
        ax.text(x1, yy + 0.017, f"{end:.3f}", fontsize=6.5, color=MODEL_SPLIT_COLOR, ha="center")
        ax.text((x0 + x1) / 2, yy - 0.032, "leakage reduced", fontsize=5.9, color=COLORS["control"], ha="center")

    risk_color = "#B83B73"
    risk_soft = "#FBE8F1"

    # a. Hero: the representation can support two task readouts.
    panel_title(0.965, "a", "High accuracy can hide leakage")
    box(0.045, 0.670, 0.210, 0.100, "", "#FFFFFF", COLORS["neutral"], fontsize=7.0)
    collision = (0.079, 0.720)
    event_tracks = [(0.061, 0.746), (0.063, 0.694), (0.094, 0.752), (0.100, 0.700), (0.104, 0.728)]
    for endpoint in event_tracks:
        ax.plot([collision[0], endpoint[0]], [collision[1], endpoint[1]], color=COLORS["control"], linewidth=1.0)
    ax.scatter([collision[0]], [collision[1]], s=8, color=COLORS["ink"], zorder=3)
    ax.scatter([p[0] for p in event_tracks], [p[1] for p in event_tracks], s=10, color=COLORS["control"], zorder=3)
    ax.plot([0.115, 0.115], [0.685, 0.755], color=COLORS["neutral"], linewidth=0.8)
    ax.text(0.185, 0.720, "collider events", fontsize=7.3, ha="center", va="center", fontweight="bold")

    # Latent cloud: marker shape distinguishes task classes, not the two case studies.
    ax.add_patch(Ellipse((0.470, 0.720), 0.310, 0.205, facecolor="#F8FAFB", edgecolor=COLORS["control"], linewidth=1.15))
    class_one_pts = [(0.385, 0.750), (0.405, 0.735), (0.425, 0.760), (0.445, 0.745), (0.455, 0.735), (0.475, 0.720), (0.400, 0.715), (0.420, 0.700)]
    class_zero_pts = [(0.455, 0.690), (0.475, 0.675), (0.490, 0.705), (0.510, 0.690), (0.520, 0.680), (0.540, 0.665), (0.535, 0.715), (0.555, 0.700)]
    domain_a = COLORS["nuis"]
    domain_b = "#8B98A5"
    for points, marker in ((class_one_pts, "o"), (class_zero_pts, "s")):
        for point_a, point_b in zip(points[::2], points[1::2]):
            ax.plot([point_a[0], point_b[0]], [point_a[1], point_b[1]], color=COLORS["neutral"], linewidth=0.75, zorder=1)
        ax.scatter(*zip(*points[::2]), s=23, facecolors=domain_a, edgecolors=COLORS["control"], linewidths=0.7, marker=marker)
        ax.scatter(*zip(*points[1::2]), s=23, facecolors=domain_b, edgecolors=COLORS["control"], linewidths=0.7, marker=marker)
    ax.scatter([0.690], [0.855], s=13, color=COLORS["control"], marker="o")
    ax.text(0.702, 0.855, "task label 1", fontsize=5.5, color=COLORS["control"], ha="left", va="center")
    ax.scatter([0.790], [0.855], s=13, facecolors="#FFFFFF", edgecolors=COLORS["control"], linewidths=0.9, marker="s")
    ax.text(0.802, 0.855, "task label 0", fontsize=5.5, color=COLORS["control"], ha="left", va="center")
    ax.scatter([0.690], [0.885], s=15, facecolors=domain_a, edgecolors=COLORS["control"], linewidths=0.6, marker="o")
    ax.text(0.702, 0.885, "domain A", fontsize=5.4, color=COLORS["control"], ha="left", va="center")
    ax.scatter([0.790], [0.885], s=15, facecolors=domain_b, edgecolors=COLORS["control"], linewidths=0.6, marker="o")
    ax.text(0.802, 0.885, "domain B", fontsize=5.4, color=COLORS["control"], ha="left", va="center")
    ax.plot([0.3887, 0.5663], [0.6675, 0.7576], color=risk_color, linewidth=1.25, linestyle=(0, (3, 2)))
    ax.text(0.535, 0.755, "task decision\nboundary", fontsize=5.1, color=risk_color, ha="center", va="bottom", linespacing=1.0)
    ax.text(0.470, 0.850, "shared event embedding", fontsize=7.5, fontweight="bold", ha="center", va="center")
    ax.text(0.455, 0.580, "same-label points shift across domains:\ntask and domain structure are jointly readable", fontsize=5.9, color=risk_color, ha="center", va="center", linespacing=1.15)
    arrow((0.255, 0.720), (0.315, 0.720), color=COLORS["control"])

    box(0.700, 0.750, 0.185, 0.075, "main prediction\nhigh accuracy", "#EAF5FB", MODEL_SPLIT_COLOR, fontsize=7.0, weight="bold")
    box(0.700, 0.635, 0.185, 0.075, "task and domain\nare jointly readable", risk_soft, risk_color, fontsize=7.0, weight="bold")
    arrow((0.615, 0.752), (0.700, 0.787), color=MODEL_SPLIT_COLOR)
    arrow((0.615, 0.685), (0.700, 0.672), color=risk_color)

    # b. Audit intervention, visually continuous with the latent representation above.
    panel_title(0.485, "b", "Two-channel audit")
    box(0.090, 0.350, 0.110, 0.070, "audit\nintervention", "#F5F6F7", COLORS["neutral"], fontsize=6.3, color=COLORS["control"], lw=0.9)
    arrow((0.210, 0.385), (0.285, 0.385), color=COLORS["control"])
    box(0.285, 0.350, 0.155, 0.070, "split candidate\nrepresentation", "#FFFFFF", COLORS["control"], fontsize=6.9, weight="bold")
    box(0.515, 0.390, 0.145, 0.060, "$z_{phys}$\nphysics channel", "#EAF5FB", MODEL_SPLIT_COLOR, fontsize=6.7, weight="bold")
    box(0.515, 0.285, 0.145, 0.060, "$z_{nuis}$\nnuisance channel", "#FFF4DE", COLORS["nuis"], fontsize=6.7, weight="bold")
    box(0.750, 0.398, 0.120, 0.048, "task head  $f$", "#EAF5FB", MODEL_SPLIT_COLOR, fontsize=6.6)
    box(0.750, 0.320, 0.120, 0.048, "domain probe  $g_d$", "#FFF4DE", COLORS["nuis"], fontsize=6.4)
    box(0.750, 0.250, 0.120, 0.048, "leakage probe  $g_y$", risk_soft, risk_color, fontsize=6.4)
    arrow((0.440, 0.390), (0.515, 0.420), color=MODEL_SPLIT_COLOR)
    arrow((0.440, 0.380), (0.515, 0.315), color=COLORS["nuis"])
    arrow((0.660, 0.420), (0.750, 0.422), color=MODEL_SPLIT_COLOR)
    arrow((0.660, 0.315), (0.750, 0.344), color=COLORS["nuis"])
    arrow((0.660, 0.300), (0.750, 0.274), color=risk_color)

    # c. Evidence and scope: results first, then a visually separate claim boundary.
    panel_title(0.205, "c", "Evidence and scope")
    reduction_card(0.030, "H4l routing", 0.961, 0.593)

    # H4l-specific likelihood boundary sits in its own scoped card.
    box(0.260, 0.015, 0.140, 0.130, "", "#F7F3FC", M4L_REFERENCE_COLOR, lw=0.95)
    ax.text(0.280, 0.130, "H4l boundary", fontsize=6.6, color=M4L_REFERENCE_COLOR, fontweight="bold", ha="left")
    mass_plot_x = 0.280 + 0.100 * (mass_x - mass_x.min()) / (mass_x.max() - mass_x.min())
    mass_plot_y = 0.045 + 0.045 * mass_y
    ax.plot(mass_plot_x, mass_plot_y, color=M4L_REFERENCE_COLOR, linewidth=1.25)
    ax.fill_between(mass_plot_x, 0.045, mass_plot_y, color="#F1E8FF", alpha=0.9)
    ax.text(0.330, 0.105, "$m_{4\\ell}$ signal template", fontsize=6.0, color=M4L_REFERENCE_COLOR, ha="center")
    ax.text(0.330, 0.022, "before full likelihood", fontsize=5.7, color=COLORS["control"], ha="center")

    box(0.445, 0.015, 0.200, 0.130, "", "#F5F6F7", COLORS["control"], lw=0.95)
    ax.text(0.463, 0.120, "Controls", fontsize=7.1, fontweight="bold", ha="left", va="center")
    for check_y in (0.082, 0.047):
        ax.plot([0.462, 0.467, 0.476], [check_y, check_y - 0.006, check_y + 0.008], color=COLORS["control"], linewidth=1.35, solid_capstyle="round")
    ax.text(0.488, 0.082, "probe sensitivity passed", fontsize=6.1, color=COLORS["ink"], ha="left", va="center")
    ax.text(0.488, 0.047, "non-collapse passed", fontsize=6.1, color=COLORS["ink"], ha="left", va="center")
    reduction_card(0.690, "TopTag cross-workflow check", 0.935, 0.606)
    save(fig, "figure1_workflow_schematic")


def figure2_h4l_branch_routing():
    df = source("source_data_figure2_h4l_branch_routing.csv")
    fig = plt.figure(figsize=(6.3, 3.59))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.55, 1.0], hspace=0.95, wspace=0.72)
    ax = fig.add_subplot(gs[0, :])
    shared, shared_err = metric_row(df, "z_nuis_physics_auc", "shared_baseline")
    split, split_err = metric_row(df, "z_nuis_physics_auc", "split_candidate")
    comparison_strip(
        ax,
        shared,
        split,
        [shared_err, split_err],
        "$z_{nuis}$ physics AUC (lower is less leakage)",
        (0.50, 1.00),
        "{:.3f}",
        0.5,
        shared_color=MODEL_SHARED_COLOR,
        split_color=MODEL_SPLIT_COLOR,
    )
    ax.text(
        0.50,
        0.92,
        f"physics leakage reduced: {shared:.3f} $\\rightarrow$ {split:.3f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.2,
        color=COLORS["ink"],
        fontweight="bold",
    )
    label_axes = [(ax, "a")]

    specs = [
        ("physics_auc", "Physics AUC\n(task preserved)", (0.9878, 0.9902), None, "{:.4f}"),
        ("z_nuis_domain_acc", "$z_{nuis}$ domain accuracy\n(domain remains readable)", (0.18, 0.27), 0.2, "{:.3f}"),
        ("score_domain_drift_max", "maximum score drift\nacross domains", (0.0, 0.0026), None, "{:.4f}"),
    ]
    for ax, (metric, xlabel, xlim, ref, value_format), label in zip([fig.add_subplot(gs[1, i]) for i in range(3)], specs, "bcd"):
        rows = [
            metric_row(df, metric, "shared_baseline"),
            metric_row(df, metric, "split_candidate"),
        ]
        vals, errs = [row[0] for row in rows], [row[1] for row in rows]
        comparison_strip(
            ax,
            vals[0],
            vals[1],
            errs,
            xlabel,
            xlim,
            value_format,
            ref,
            shared_color=MODEL_SHARED_COLOR,
            split_color=MODEL_SPLIT_COLOR,
        )
        if metric == "physics_auc":
            ax.set_xticks([0.988, 0.989, 0.990])
            ax.set_xticklabels(["0.988", "0.989", "0.990"])
        label_axes.append((ax, label))
    handles = [
        plt.Line2D([], [], marker="o", color="none", markerfacecolor=MODEL_SHARED_COLOR, markeredgecolor=MODEL_SHARED_COLOR, markersize=5),
        plt.Line2D([], [], marker="s", color="none", markerfacecolor=MODEL_SPLIT_COLOR, markeredgecolor=MODEL_SPLIT_COLOR, markersize=5),
    ]
    fig.legend(handles, ["shared baseline", "split candidate"], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(top=0.88, right=0.82, left=0.10, bottom=0.12)
    for label_ax, label in label_axes:
        figure_panel_label(fig, label_ax, label)
    save(fig, "figure2_h4l_branch_routing")


def figure3_h4l_controls():
    df = source("source_data_figure3_h4l_controls.csv")
    fig = plt.figure(figsize=(6.3, 3.19))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], wspace=0.45, hspace=0.58)
    ax = fig.add_subplot(gs[:, 0])
    label_axes = [(ax, "a")]
    metrics = [
        "true_z_phys_domain_probe",
        "true_z_nuis_domain_probe",
        "shuffled_z_nuis_domain_probe",
        "teacher_embedding_domain_probe",
    ]
    names = ["$z_{phys}$ domain", "$z_{nuis}$ domain", "shuffled control", "teacher embedding"]
    vals = [
        float(df[(df["panel"] == "probe_control") & (df["metric"] == metric)]["value"].iloc[0])
        for metric in metrics
    ]
    injected_domain = float(df[(df["panel"] == "probe_control") & (df["metric"] == "injected_onehot_domain_probe")]["value"].iloc[0])
    ypos = np.arange(len(vals))[::-1]
    colors = [COLORS["split"], COLORS["nuis"], COLORS["control"], COLORS["domain"]]
    for y, val, col in zip(ypos, vals, colors):
        ax.plot([0.2, val], [y, y], color=COLORS["grid"], linewidth=1.0, zorder=1)
        ax.scatter(val, y, s=28, color=col, zorder=2)
        ax.text(val + 0.006, y, f"{val:.3f}" if val < 0.999 else "1.0", va="center", fontsize=7.2)
    ax.axvline(0.2, color=COLORS["control"], linewidth=0.8, linestyle=(0, (3, 2)))
    ax.text(
        0.205,
        -0.45,
        f"off-axis control = {injected_domain:.1f} (probe capacity)",
        fontsize=7.0,
        color=COLORS["positive_control"],
        ha="left",
    )
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.set_xlim(0.18, 0.30)
    ax.set_ylim(-0.72, len(vals) - 0.35)
    ax.set_xlabel("domain-probe accuracy")
    style_xgrid(ax)

    ax = fig.add_subplot(gs[0, 1])
    label_axes.append((ax, "b"))
    leak = float(df[(df["panel"] == "physics_leakage_control") & (df["metric"] == "z_nuis_physics_probe_auc")]["value"].iloc[0])
    vals = [1.0, leak]
    ypos = np.array([1, 0])
    labels = ["injected physics", "$z_{nuis}$ physics"]
    for y, val, col in zip(ypos, vals, [COLORS["positive_control"], COLORS["nuis"]]):
        ax.plot([0.5, val], [y, y], color=COLORS["grid"], linewidth=1.0)
        ax.scatter(val, y, s=30, color=col, zorder=2)
        ax.text(val + 0.018, y, f"{val:.3f}" if val < 0.999 else "1.0", va="center", fontsize=7.2)
    ax.axvline(0.5, color=COLORS["control"], linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.45, 1.08)
    ax.set_xlabel("physics-probe AUC")
    style_xgrid(ax)

    ax = fig.add_subplot(gs[1, 1])
    label_axes.append((ax, "c"))
    rank = df[df["panel"] == "rank"].copy()
    phys_train = float(rank[rank["metric"] == "train_z_phys_effective_rank"]["value"].iloc[0])
    phys_val = float(rank[rank["metric"] == "val_z_phys_effective_rank"]["value"].iloc[0])
    nuis_train = float(rank[rank["metric"] == "train_z_nuis_effective_rank"]["value"].iloc[0])
    nuis_val = float(rank[rank["metric"] == "val_z_nuis_effective_rank"]["value"].iloc[0])
    train_offset = 0.085
    val_offset = -0.085
    for y, vals, col, label in [
        (1, [phys_train, phys_val], COLORS["split"], "$z_{phys}$"),
        (0, [nuis_train, nuis_val], COLORS["nuis"], "$z_{nuis}$"),
    ]:
        y_train = y + train_offset
        y_val = y + val_offset
        ax.plot(vals, [y_train, y_val], color=col, linewidth=0.9, alpha=0.65, zorder=1)
        ax.scatter(vals[0], y_train, s=30, color="white", edgecolor=col, linewidth=1.0, zorder=3)
        ax.scatter(vals[1], y_val, s=30, color=col, zorder=3)
        ax.text(vals[0] + 0.38, y_train, f"train: {vals[0]:.2f}", va="center", fontsize=7.0, color=COLORS["ink"])
        ax.text(vals[1] + 0.38, y_val, f"val: {vals[1]:.2f}", va="center", fontsize=7.0, color=COLORS["ink"])
    ax.set_xlim(0, 7.45)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["$z_{nuis}$", "$z_{phys}$"])
    ax.set_xlabel("effective rank")
    style_xgrid(ax)
    for label_ax, label in label_axes:
        figure_panel_label(fig, label_ax, label, dx=0.050, dy=0.014)
    save(fig, "figure3_h4l_controls")


def figure4_h4l_boundary_checks():
    overlay = source("source_data_figure4_control_region_overlay.csv")
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(6.3, 3.63),
        sharex=True,
        gridspec_kw={"height_ratios": [2.05, 1.0], "hspace": 0.14},
    )
    ax = axes[0]
    lows = overlay["bin_low"].to_numpy(float)
    highs = overlay["bin_high"].to_numpy(float)
    centers = 0.5 * (lows + highs)
    widths = highs - lows
    observed_counts = overlay["observed_count"].to_numpy(float)
    observed_err = overlay["observed_err"].to_numpy(float)
    mc_bkg = overlay["mc_bkg_sideband_scaled"].to_numpy(float)
    mc_total = overlay["mc_total_sideband_scaled"].to_numpy(float)
    ax.axvspan(115.0, 135.0, color=COLORS["warning"], alpha=0.055, label="Higgs window, not fitted")
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
    ax.set_ylabel("events / bin")
    ax.tick_params(labelbottom=False)
    ax.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.55, 1.16), handlelength=1.2, columnspacing=0.9)
    style_ygrid(ax)

    ratio_residual_panel(axes[1], overlay)
    axes[0].text(
        -0.185,
        0.985,
        "a",
        transform=axes[0].transAxes,
        fontsize=11.2,
        fontweight="bold",
        ha="right",
        va="top",
        color=COLORS["ink"],
        clip_on=False,
        zorder=10,
    )
    axes[1].text(
        -0.185,
        0.985,
        "b",
        transform=axes[1].transAxes,
        fontsize=11.2,
        fontweight="bold",
        ha="right",
        va="top",
        color=COLORS["ink"],
        clip_on=False,
        zorder=10,
    )
    save(fig, "figure4_h4l_boundary_checks")


def figure5_toptag_transfer():
    df = source("source_data_figure5_toptag_transfer.csv")
    fig = plt.figure(figsize=(6.3, 3.59))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.55, 1.0], hspace=0.95, wspace=0.72)
    main = df[df["panel"] == "e76b_multiseed"]

    ax = fig.add_subplot(gs[0, :])
    rows = main[main["metric"] == "z_nuis_physics_auc"]
    shared = rows[rows["model"] == "shared_baseline"].iloc[0]
    split = rows[rows["model"] == "split_candidate"].iloc[0]
    held = df[(df["panel"] == "e77_holdout") & (df["metric"] == "holdout_z_nuis_physics_auc")]
    overlays = [
        {
            "shared": float(held[held["model"] == "heldout_shared_baseline"]["mean"].iloc[0]),
            "split": float(held[held["model"] == "heldout_split_candidate"]["mean"].iloc[0]),
            "label": "held-out domain",
            "color": COLORS["control"],
            "y": -0.32,
        }
    ]
    comparison_strip(
        ax,
        float(shared["mean"]),
        float(split["mean"]),
        [float(shared["sd"]), float(split["sd"])],
        "$z_{nuis}$ physics AUC (lower is less leakage)",
        (0.55, 0.96),
        "{:.3f}",
        0.5,
        overlays,
        shared_color=MODEL_SHARED_COLOR,
        split_color=MODEL_SPLIT_COLOR,
    )
    ax.text(
        0.50,
        0.92,
        f"cross-workflow check repeats the leakage drop: {float(shared['mean']):.3f} $\\rightarrow$ {float(split['mean']):.3f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.2,
        color=COLORS["ink"],
        fontweight="bold",
    )
    label_axes = [(ax, "a")]

    specs = [
        ("physics_auc", "Tagging AUC\n(task preserved)", (0.948, 0.954), None, "{:.4f}"),
        ("z_nuis_domain_acc", "$z_{nuis}$ domain accuracy\n(domain remains readable)", (0.16, 0.25), None, "{:.3f}"),
        ("background_rejection_30pct", "30% efficiency\nbackground rejection", (0, 480), None, "{:.0f}"),
    ]
    for ax, (metric, xlabel, xlim, ref, value_format), label in zip([fig.add_subplot(gs[1, i]) for i in range(3)], specs, "bcd"):
        rows = main[main["metric"] == metric]
        shared = rows[rows["model"] == "shared_baseline"].iloc[0]
        split = rows[rows["model"] == "split_candidate"].iloc[0]
        overlays = []
        if metric == "background_rejection_30pct":
            ref_rows = df[(df["panel"] == "e78_reference") & (df["metric"] == "background_rejection_30pct")]
            ref_val = float(ref_rows[ref_rows["model"] == "reference_constituent_baseline"]["mean"].iloc[0])
            overlays.append(
                {
                    "shared": ref_val,
                    "split": ref_val,
                    "label": f"capacity only {ref_val:.0f}",
                    "color": COLORS["control"],
                    "marker": "^",
                    "linestyle": ":",
                }
            )
        comparison_strip(
            ax,
            float(shared["mean"]),
            float(split["mean"]),
            [float(shared["sd"]), float(split["sd"])],
            xlabel,
            xlim,
            value_format,
            ref,
            overlays,
            shared_color=MODEL_SHARED_COLOR,
            split_color=MODEL_SPLIT_COLOR,
        )
        label_axes.append((ax, label))
    handles = [
        plt.Line2D([], [], marker="o", color="none", markerfacecolor=MODEL_SHARED_COLOR, markeredgecolor=MODEL_SHARED_COLOR, markersize=5),
        plt.Line2D([], [], marker="s", color="none", markerfacecolor=MODEL_SPLIT_COLOR, markeredgecolor=MODEL_SPLIT_COLOR, markersize=5),
        plt.Line2D([], [], marker="o", color=COLORS["control"], markerfacecolor="white", markeredgecolor=COLORS["control"], markersize=4, linestyle=(0, (2, 2)), linewidth=0.85),
        plt.Line2D([], [], marker="^", color="none", markerfacecolor="white", markeredgecolor=COLORS["control"], markersize=5, linewidth=0),
    ]
    fig.legend(handles, ["shared baseline", "split candidate", "held-out domain", "capacity calibration"], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(top=0.88, right=0.82, left=0.10, bottom=0.12)
    for label_ax, label in label_axes:
        figure_panel_label(fig, label_ax, label)
    save(fig, "figure5_toptag_transfer")


def figure_appendix_h4l_likelihood_scans():
    scan_df = source("source_data_figure4_likelihood_scan_points.csv")
    summary_df = source("source_data_figure4_likelihood_scan_summary.csv")
    fig, ax = plt.subplots(figsize=(6.25, 3.05))
    appendix_likelihood_scan_panel(ax, scan_df, summary_df)
    save(fig, "figure_appendix_h4l_likelihood_scans")


def main():
    figure1_workflow()
    figure2_h4l_branch_routing()
    figure3_h4l_controls()
    figure4_h4l_boundary_checks()
    figure5_toptag_transfer()
    figure_appendix_h4l_likelihood_scans()


if __name__ == "__main__":
    main()
