# figures_redraw_v2.py
# ProveTok paper figure redraw — publication-quality style
#
# Style references:
#   - Grouped bar with bold data labels on top (ACM MM style)
#   - Line plots with teal/orange, dashed baselines
#   - Horizontal bars with hatching for ablation variants
#   - Minimal spines, light grid, generous padding
#
# Run:   python Scripts/figures_redraw_v2.py
# Out:   outputs/redrawn_figures/{*.png, *.pdf}

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

OUT_DIR = Path("outputs/redrawn_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXPORT_WATERFALL = False

# ── Palette (soft, print-friendly) ──
C_BLUE   = "#7AAFDD"   # light steel blue
C_GREEN  = "#8FBC8F"   # dark sea green
C_CORAL  = "#E8927C"   # soft coral
C_TEAL   = "#2CA9A0"   # teal (line plots)
C_ORANGE = "#E8923F"   # warm orange (dashed ref)
C_PURPLE = "#B388C9"   # lavender
C_GRAY   = "#B0B0B0"   # neutral gray
C_RED    = "#D64545"    # highlight red

EDGE_BLUE  = "#4A7FAD"
EDGE_GREEN = "#5A8C5A"
EDGE_CORAL = "#C06A56"


# ── Global style ──
def set_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.pad": 4,
        "ytick.major.pad": 4,
        "legend.fontsize": 9,
        "legend.framealpha": 0.0,
        "grid.alpha": 0.35,
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "grid.color": "#CCCCCC",
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.12,
    })


def pct(x, pos):
    return f"{x:.0f}%"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}")


# ── Helper: bar label (bold value on top) ──
def bar_label(ax, bars, fmt="{:.2f}", offset=3, fontsize=8.5, bold=True):
    weight = "bold" if bold else "normal"
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                fmt.format(h), ha="center", va="bottom",
                fontsize=fontsize, fontweight=weight)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1) tau_iou critical-point sweep
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_tau_sweep():
    tau = np.array([0.03, 0.035, 0.04, 0.045, 0.05])
    mediastinum = np.array([0.0, 0.0, 0.0, 100.0, 100.0])
    overall = np.array([8.27, 8.27, 8.27, 12.78, 12.78])

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    ax.plot(tau, mediastinum, marker="o", color=C_TEAL, label="Mediastinum violation",
            markerfacecolor="white", markeredgewidth=1.8, markeredgecolor=C_TEAL)
    ax.plot(tau, overall, marker="s", color=C_ORANGE, label="Overall violation",
            linestyle="--", markerfacecolor="white", markeredgewidth=1.8, markeredgecolor=C_ORANGE)

    ax.axvline(0.04, linestyle=":", linewidth=1.0, color="#888888", zorder=0)
    ax.annotate(
        r"$\tau_{\mathrm{IoU}}=0.04$ (selected)",
        xy=(0.04, 8.27), xytext=(0.043, 35),
        arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#666666"),
        fontsize=8.5, color="#444444",
    )

    ax.set_xlabel(r"$\tau_{\mathrm{IoU}}$")
    ax.set_ylabel("Violation Rate (%)")
    ax.set_ylim(-5, 110)
    ax.set_xticks(tau)
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.grid(True, axis="y")
    ax.legend(frameon=False, loc="center left")

    # labels — stagger to avoid overlap
    for i, (x, y) in enumerate(zip(tau, mediastinum)):
        ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=C_TEAL)
    for i, (x, y) in enumerate(zip(tau, overall)):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, -13), ha="center", fontsize=8, color=C_ORANGE)

    save(fig, "fig_tau_iou_critical_point")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2) 5K ablation trajectory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_ablation_trajectory():
    labels = ["A0", "A1", "E1", "B2'", "B2'v2", "C2'", "D2"]
    values = np.array([7.59, 6.01, 6.18, 6.01, 4.25, 6.18, 5.96])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.4, 4.2))

    ax.plot(x, values, marker="o", color=C_TEAL,
            markerfacecolor="white", markeredgewidth=2.0, markeredgecolor=C_TEAL)

    # best point — filled
    best_idx = 4  # B2'v2
    ax.plot(best_idx, values[best_idx], marker="o", markersize=9,
            color=C_RED, markeredgecolor=C_RED, zorder=5)
    ax.annotate(
        f"Best: {values[best_idx]:.2f}%",
        xy=(best_idx, values[best_idx]),
        xytext=(best_idx + 0.5, values[best_idx] - 0.7),
        arrowprops=dict(arrowstyle="-|>", lw=0.8, color=C_RED),
        fontsize=9, fontweight="bold", color=C_RED,
    )

    # delta labels — above the midpoint of each segment
    for i in range(1, len(values)):
        delta = values[i] - values[i - 1]
        xm = (x[i - 1] + x[i]) / 2
        ym = max(values[i - 1], values[i]) + 0.22
        color = "#2E7D32" if delta < 0 else "#C62828"
        ax.text(xm, ym, f"{delta:+.2f}", fontsize=7.5, ha="center",
                color=color, fontweight="bold")

    # value labels
    for xi, yi in zip(x, values):
        if xi == best_idx:
            continue  # already annotated
        ax.annotate(f"{yi:.2f}%", (xi, yi), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, color="#333333")

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Violation Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.grid(True, axis="y")
    ax.set_ylim(3.2, 8.5)

    # ERM-style dashed reference for baseline
    ax.axhline(values[0], linestyle="--", linewidth=1.0, color=C_ORANGE, alpha=0.6, zorder=0)
    ax.text(len(labels) - 0.5, values[0] + 0.12, "A0 baseline",
            fontsize=7.5, color=C_ORANGE, alpha=0.8)

    save(fig, "fig_ablation_trajectory_5k_official")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3) Optional waterfall
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_ablation_waterfall():
    base = 7.59
    steps = ["A0\u2192A1", "A1\u2192E1", "E1\u2192B2'", "B2'\u2192B2'v2", "B2'v2\u2192C2'", "C2'\u2192D2"]
    deltas = np.array([-1.58, +0.17, -0.17, -1.76, +1.93, -0.22])

    x = np.arange(len(steps) + 2)
    tick_labels = ["A0"] + steps + ["D2"]

    fig, ax = plt.subplots(figsize=(8.4, 4.4))

    # baseline
    ax.bar(0, base, width=0.65, color=C_BLUE, edgecolor=EDGE_BLUE, linewidth=0.8)
    ax.text(0, base + 0.12, f"{base:.2f}%", ha="center", fontsize=8.5, fontweight="bold")

    running = base
    for i, d in enumerate(deltas, start=1):
        color = C_CORAL if d >= 0 else C_GREEN
        edge = EDGE_CORAL if d >= 0 else EDGE_GREEN
        bottom = running if d >= 0 else running + d
        ax.bar(i, abs(d), bottom=bottom, width=0.65,
               color=color, edgecolor=edge, linewidth=0.8)
        ypos = bottom + abs(d) + 0.1 if d >= 0 else bottom - 0.18
        va = "bottom" if d >= 0 else "top"
        ax.text(i, ypos, f"{d:+.2f}", ha="center", va=va, fontsize=8, fontweight="bold")
        running += d

    # final
    ax.bar(len(tick_labels) - 1, running, width=0.65, color=C_BLUE, edgecolor=EDGE_BLUE, linewidth=0.8)
    ax.text(len(tick_labels) - 1, running + 0.12, f"{running:.2f}%",
            ha="center", fontsize=8.5, fontweight="bold")

    ax.set_ylabel("Violation Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=25, ha="right")
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.grid(True, axis="y")
    ax.set_ylim(0, 9.0)

    save(fig, "fig_ablation_waterfall_optional")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4) 5K split stability
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_split_stability():
    splits = ["Train", "Valid", "Test", "All"]
    viol = np.array([3.86, 3.98, 4.05, 3.89])
    sentences = np.array([31919, 3995, 3979, 39893])

    x = np.arange(len(splits))
    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    ax.plot(x, viol, marker="o", color=C_TEAL, markerfacecolor="white",
            markeredgewidth=2.0, markeredgecolor=C_TEAL)

    # dashed mean line
    mean_viol = np.mean(viol)
    ax.axhline(mean_viol, linestyle="--", linewidth=1.0, color=C_ORANGE, alpha=0.6, zorder=0)
    ax.text(3.3, mean_viol + 0.01, f"mean {mean_viol:.2f}%", fontsize=7.5, color=C_ORANGE)

    ax.set_xlabel("Split")
    ax.set_ylabel("Violation Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_ylim(3.7, 4.15)
    ax.grid(True, axis="y")

    for xi, v, n in zip(x, viol, sentences):
        ax.annotate(f"{v:.2f}%", (xi, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8.5, fontweight="bold",
                    color="#333333")
        ax.annotate(f"n={n:,}", (xi, v), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=7, color="#888888")

    save(fig, "fig_split_stability_5k")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5) Split NLG metrics (grouped bar, bold labels)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_split_nlg():
    splits = ["Train", "Valid", "Test", "All"]
    bleu = np.array([0.492, 0.502, 0.487, 0.492])
    rouge = np.array([0.656, 0.661, 0.653, 0.656])
    meteor = np.array([0.636, 0.640, 0.632, 0.636])

    x = np.arange(len(splits))
    width = 0.22

    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    b1 = ax.bar(x - width, bleu, width, label="BLEU-4",
                color=C_BLUE, edgecolor=EDGE_BLUE, linewidth=0.7)
    b2 = ax.bar(x, rouge, width, label="ROUGE-L",
                color=C_GREEN, edgecolor=EDGE_GREEN, linewidth=0.7)
    b3 = ax.bar(x + width, meteor, width, label="METEOR",
                color=C_CORAL, edgecolor=EDGE_CORAL, linewidth=0.7)

    # bold labels on top
    for bars in [b1, b2, b3]:
        bar_label(ax, bars, fmt="{:.3f}", offset=0.004, fontsize=7.5, bold=True)

    ax.set_xlabel("Split")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylim(0.44, 0.72)
    ax.grid(True, axis="y")
    ax.legend(frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.08))

    save(fig, "fig_split_nlg_5k")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6) Residual violation breakdown (horizontal bars)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_residual_breakdown():
    rules = ["R3\nDEPTH", "R6b\nCROSS_PRES", "R1\nLATERALITY"]
    counts = np.array([113, 46, 2])
    shares = np.array([70.2, 28.6, 1.2])
    colors = [C_BLUE, C_CORAL, C_GRAY]
    edges = [EDGE_BLUE, EDGE_CORAL, "#888888"]

    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    y = np.arange(len(rules))

    bars = ax.barh(y, counts, height=0.55,
                   color=colors, edgecolor=edges, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(rules, fontsize=9)
    ax.invert_yaxis()

    ax.set_xlabel("Count")
    ax.grid(True, axis="x")
    ax.set_xlim(0, 140)

    for bar, c, s in zip(bars, counts, shares):
        ax.text(bar.get_width() + 3,
                bar.get_y() + bar.get_height() / 2,
                f"{c}  ({s:.1f}%)",
                va="center", fontsize=9, fontweight="bold")

    save(fig, "fig_residual_violation_breakdown")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7) W_proj trade-off (2-panel grouped bar)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_wproj_tradeoff():
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))

    width = 0.30

    # ── Left panel: Viol% + BLEU-4 ──
    metrics_L = ["Viol.%", "BLEU-4"]
    id_vals = [4.05, 0.487]
    tr_vals = [4.32, 0.477]

    x = np.arange(len(metrics_L))
    b1 = axes[0].bar(x - width/2, id_vals, width, label="Identity + spatial",
                     color=C_BLUE, edgecolor=EDGE_BLUE, linewidth=0.7)
    b2 = axes[0].bar(x + width/2, tr_vals, width, label="Trained $W_{proj}$ + E1",
                     color=C_CORAL, edgecolor=EDGE_CORAL, linewidth=0.7)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            fmt = f"{h:.2f}" if h > 1 else f"{h:.3f}"
            axes[0].text(bar.get_x() + bar.get_width()/2, h + 0.02,
                         fmt, ha="center", fontsize=8, fontweight="bold")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_L)
    axes[0].set_title("Quality trade-off")
    axes[0].grid(True, axis="y")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")

    # ── Right panel: R1 + R3 counts ──
    metrics_R = ["R1", "R3"]
    id_counts = [2, 113]
    tr_counts = [105, 23]

    x2 = np.arange(len(metrics_R))
    b3 = axes[1].bar(x2 - width/2, id_counts, width, label="Identity + spatial",
                     color=C_BLUE, edgecolor=EDGE_BLUE, linewidth=0.7)
    b4 = axes[1].bar(x2 + width/2, tr_counts, width, label="Trained $W_{proj}$ + E1",
                     color=C_CORAL, edgecolor=EDGE_CORAL, linewidth=0.7)

    for bars in [b3, b4]:
        for bar in bars:
            h = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2, h + 2,
                         f"{int(h)}", ha="center", fontsize=8, fontweight="bold")

    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(metrics_R)
    axes[1].set_title("Rule-level trade-off")
    axes[1].grid(True, axis="y")

    fig.subplots_adjust(wspace=0.32)
    save(fig, "fig_wproj_tradeoff_5k")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8) Fig 1 — Ablation waterfall (2-panel: Viol% bars + R1/R3 grouped bars)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_fig1_waterfall():
    configs = ["A0", "A1", "E1", "B2'", "B2'v2", "C2'", "D2"]
    viol = np.array([7.59, 6.01, 6.18, 6.01, 4.25, 6.18, 5.96])
    r1   = np.array([0, 0, 95, 109, 98, 113, 102])
    r3   = np.array([237, 174, 86, 86, 23, 86, 86])

    x = np.arange(len(configs))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # ── Left: Viol% bar chart ──
    colors_left = [C_CORAL] * len(configs)
    colors_left[4] = C_TEAL  # B2'v2 = best
    edge_left = [EDGE_CORAL] * len(configs)
    edge_left[4] = "#1E8C84"

    bars1 = ax1.bar(x, viol, width=0.6, color=colors_left, edgecolor=edge_left, linewidth=0.8)
    # dashed baseline
    ax1.axhline(viol[4], linestyle="--", linewidth=0.8, color=C_TEAL, alpha=0.5, zorder=0)

    for xi, v in zip(x, viol):
        ax1.text(xi, v + 0.15, f"{v:.1f}", ha="center", fontsize=8.5, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=9)
    ax1.set_ylabel("Violation Rate (%)")
    ax1.set_title("(a) Total Violation Rate")
    ax1.set_ylim(0, 9.0)
    ax1.grid(True, axis="y")

    # ── Right: R1 + R3 grouped bars ──
    width = 0.32
    b_r1 = ax2.bar(x - width/2, r1, width, label="R1 (Laterality)",
                   color=C_CORAL, edgecolor=EDGE_CORAL, linewidth=0.7)
    b_r3 = ax2.bar(x + width/2, r3, width, label="R3 (Depth)",
                   color=C_BLUE, edgecolor=EDGE_BLUE, linewidth=0.7)

    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, fontsize=9)
    ax2.set_ylabel("Violation Count")
    ax2.set_title("(b) R1 & R3 Violations")
    ax2.grid(True, axis="y")
    ax2.legend(frameon=False, fontsize=8.5, loc="upper right")

    fig.subplots_adjust(wspace=0.28)
    save(fig, "fig1_waterfall_2panel")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9) Fig 2 — k-sweep (2-row shared x-axis)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_fig2_ksweep():
    k_vals = np.array([1, 2, 4, 6, 8, 12, 14])
    viol   = np.array([10.51, 14.34, 12.60, 12.18, 11.41, 15.31, 15.73])
    r1     = np.array([132, 187, 157, 144, 128, 168, 165])
    r3     = np.array([2, 2, 7, 14, 19, 35, 44])

    # Mimic reference style: teal + orange (kept), add soft blue as 3rd colour
    COL_VIOL = "#2CA9A0"   # teal   – Viol%  (reference colour 1)
    COL_R1   = "#E8923F"   # orange – R1     (reference colour 2)
    COL_R3   = "#6A9FD6"   # soft blue – R3  (same-saturation 3rd colour)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(5.0, 4.6),
                                          sharex=True, height_ratios=[1, 1])
    fig.subplots_adjust(hspace=0.18)

    # ── Top: Violation Rate (teal) ──
    best_idx = 4  # k=8
    ax_top.plot(k_vals, viol, marker="o", color=COL_VIOL, linewidth=1.8,
                markersize=5.5, markerfacecolor=COL_VIOL, markeredgecolor=COL_VIOL)
    # highlight best point
    ax_top.plot(k_vals[best_idx], viol[best_idx], marker="o", markersize=8,
                color=COL_VIOL, markeredgecolor=COL_VIOL, zorder=5)
    # dashed baseline at best
    ax_top.axhline(viol[best_idx], linestyle="--", linewidth=1.0,
                   color=COL_R1, alpha=0.7, zorder=0)

    ax_top.set_ylabel("Violation Rate (%)", fontsize=9)
    ax_top.set_ylim(9.0, 17.0)
    ax_top.legend(["Viol. Rate", f"Best ({viol[best_idx]:.1f}%)"],
                  fontsize=7.5, loc="upper left", frameon=True,
                  edgecolor="#CCCCCC", fancybox=False, framealpha=0.9)
    ax_top.set_title("(a) Violation Rate vs $k$", fontsize=10, pad=6)

    # ── Bottom: R1 (orange) + R3 (soft blue) ──
    ax_bot.plot(k_vals, r1, marker="o", color=COL_R1, linewidth=1.8,
                markersize=5.5, markerfacecolor=COL_R1, markeredgecolor=COL_R1,
                label="R1 (Laterality)")
    ax_bot.plot(k_vals, r3, marker="o", color=COL_R3, linewidth=1.8,
                markersize=5.5, markerfacecolor=COL_R3, markeredgecolor=COL_R3,
                linestyle="--", label="R3 (Depth)")

    ax_bot.set_xlabel("$k$ (tokens per sentence)", fontsize=9)
    ax_bot.set_ylabel("Violation Count", fontsize=9)
    ax_bot.set_xticks(k_vals)
    ax_bot.set_ylim(-5, 210)
    ax_bot.legend(fontsize=7.5, loc="center right", frameon=True,
                  edgecolor="#CCCCCC", fancybox=False, framealpha=0.9)
    ax_bot.set_title("(b) Per-Rule Violation Count vs $k$", fontsize=10, pad=6)

    save(fig, "fig2_ksweep")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10) Fig 3 — Counterfactual sensitivity (2-panel)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_fig3_counterfactual():
    perturbations = ["Original", "Paraphrase\n(control)", "Laterality\nflip", "Presence\nflip"]
    viol = np.array([4.55, 4.55, 15.93, 6.11])
    r1   = np.array([2, 2, 455, 2])
    r3   = np.array([127, 127, 127, 127])
    r6b  = np.array([52, 52, 52, 114])

    x = np.arange(len(perturbations))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # ── Left: Viol% bars ──
    colors_left = [C_GREEN, C_GRAY, C_CORAL, C_ORANGE]
    edges_left = [EDGE_GREEN, "#888888", EDGE_CORAL, "#C07830"]

    bars1 = ax1.bar(x, viol, width=0.58, color=colors_left, edgecolor=edges_left, linewidth=0.8)

    # dashed baseline at original
    ax1.axhline(viol[0], linestyle="--", linewidth=0.8, color=C_GREEN, alpha=0.5, zorder=0)

    for xi, v in zip(x, viol):
        ax1.text(xi, v + 0.35, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(perturbations, fontsize=8.5)
    ax1.set_ylabel("Violation Rate (%)")
    ax1.set_title("(a) Overall Violation Rate")
    ax1.set_ylim(0, 19)
    ax1.grid(True, axis="y")

    # ── Right: R1 / R3 / R6b grouped bars ──
    width = 0.22
    b_r1 = ax2.bar(x - width, r1, width, label="R1 (Laterality)",
                   color=C_CORAL, edgecolor=EDGE_CORAL, linewidth=0.7)
    b_r3 = ax2.bar(x, r3, width, label="R3 (Depth)",
                   color=C_BLUE, edgecolor=EDGE_BLUE, linewidth=0.7)
    b_r6 = ax2.bar(x + width, r6b, width, label="R6b (Cross-presence)",
                   color=C_ORANGE, edgecolor="#C07830", linewidth=0.7)

    ax2.set_xticks(x)
    ax2.set_xticklabels(perturbations, fontsize=8.5)
    ax2.set_ylabel("Violation Count")
    ax2.set_title("(b) Per-Rule Breakdown")
    ax2.grid(True, axis="y")
    ax2.legend(frameon=False, fontsize=8, loc="upper right")

    fig.subplots_adjust(wspace=0.28)
    save(fig, "fig3_counterfactual")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    set_style()
    print("Generating figures...")
    plot_tau_sweep()
    plot_ablation_trajectory()
    if EXPORT_WATERFALL:
        plot_ablation_waterfall()
    plot_split_stability()
    plot_split_nlg()
    plot_residual_breakdown()
    plot_wproj_tradeoff()
    plot_fig1_waterfall()
    plot_fig2_ksweep()
    plot_fig3_counterfactual()
    print(f"\nAll saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
