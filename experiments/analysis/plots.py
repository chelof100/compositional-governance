"""
Paper 4 — Figure generator.

Produces three publication-quality figures from experiment results:
  fig1_ablation_heatmap.pdf   — Thm3 ablation (8 subsets x 4 guarantees)
  fig2_feedback_convergence.pdf — Thm2 D_hat with/without feedback (drift scenario)
  fig3_compatibility_bar.pdf  — Thm1 compatibility rates
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

HERE      = Path(__file__).parent
RESULTS   = HERE / ".." / "results"
FIGS_DIR  = HERE / ".." / "figures"
FIGS_DIR.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       10,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "text.usetex":     False,
})

BLUE   = "#2166ac"
RED    = "#d6604d"
GREEN  = "#4dac26"
GRAY   = "#bababa"

# ── Figure 1: Ablation heatmap ────────────────────────────────────────────────

def fig1_ablation():
    df = pd.read_csv(RESULTS / "ablation_results.csv")
    summary = df.groupby("subset")[["P1","P2","P3","P4"]].mean()

    # Order subsets: full first, then 3-layer, then 2-layer
    order = ["0-1-2-3", "0-1-2", "0-1-3", "0-2-3", "1-2-3", "0-1", "1-2", "1-3"]
    summary = summary.loc[order]

    labels = {
        "0-1-2-3": "L0+L1+L2+L3 (full)",
        "0-1-2":   "L0+L1+L2",
        "0-1-3":   "L0+L1+L3",
        "0-2-3":   "L0+L2+L3",
        "1-2-3":   "L1+L2+L3",
        "0-1":     "L0+L1",
        "1-2":     "L1+L2",
        "1-3":     "L1+L3",
    }

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    data = summary.values
    n_rows, n_cols = data.shape

    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            color = GREEN if v >= 0.99 else (RED if v < 0.5 else "#f4a582")
            ax.add_patch(plt.Rectangle((j, n_rows-1-i), 1, 1,
                                        facecolor=color, edgecolor="white", linewidth=1.5))
            ax.text(j+0.5, n_rows-1-i+0.5, f"{v:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if v >= 0.99 or v < 0.5 else "black",
                    fontweight="bold")

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([x+0.5 for x in range(n_cols)])
    ax.set_xticklabels(["P1\n(Atomicity)", "P2\n(Drift Det.)",
                         "P3\n(Fairness)", "P4\n(Sybil Res.)"])
    ax.set_yticks([n_rows-1-i+0.5 for i in range(n_rows)])
    ax.set_yticklabels([labels[s] for s in order])
    ax.set_title("Ablation: fraction of trials satisfying each guarantee\n"
                 "(4 scenarios x 5 seeds = 20 trials per cell)", pad=8)

    # Legend
    green_p = mpatches.Patch(color=GREEN, label="Pass (1.00)")
    red_p   = mpatches.Patch(color=RED,   label="Fail (<0.50)")
    mid_p   = mpatches.Patch(color="#f4a582", label="Partial")
    ax.legend(handles=[green_p, red_p, mid_p], loc="lower right",
              bbox_to_anchor=(1.0, -0.18), ncol=3, fontsize=8, framealpha=0.9)

    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)

    out = FIGS_DIR / "fig1_ablation_heatmap.pdf"
    fig.savefig(out)
    print(f"Saved: {out}")
    plt.close(fig)


# ── Figure 2: Feedback convergence ───────────────────────────────────────────

def fig2_feedback():
    with open(RESULTS / "feedback_series.json") as f:
        data = json.load(f)

    with_fb  = data["with_feedback"]
    without  = data["without_feedback"]

    # Average series across seeds
    max_len = max(len(r["dhat_series"]) for r in with_fb)
    def pad(series, length):
        return series + [series[-1]] * (length - len(series))

    avg_with    = np.mean([pad(r["dhat_series"], max_len) for r in with_fb],    axis=0)
    avg_without = np.mean([pad(r["dhat_series"], max_len) for r in without],    axis=0)
    std_with    = np.std( [pad(r["dhat_series"], max_len) for r in with_fb],    axis=0)
    std_without = np.std( [pad(r["dhat_series"], max_len) for r in without],    axis=0)

    # X axis: step index (series sampled every 20 steps)
    x = np.arange(len(avg_with)) * 20

    fig, ax = plt.subplots(figsize=(6.0, 3.2))

    ax.plot(x, avg_with, color=BLUE, lw=2.0,
            label=r"With feedback (FC1+FC2, $K{=}0.50$, $\eta{=}0.50$, $\rho{=}0.70$)")
    ax.fill_between(x, avg_with - std_with, avg_with + std_with,
                    alpha=0.15, color=BLUE)

    ax.plot(x, avg_without, color=RED, lw=2.0, linestyle="--", label="Without feedback (open-loop)")
    ax.fill_between(x, avg_without - std_without, avg_without + std_without,
                    alpha=0.12, color=RED)

    # Threshold line
    theta = 0.20
    ax.axhline(theta, color="black", lw=0.9, linestyle=":", label=f"theta = {theta}")

    # Annotate limsup
    limsup_with    = float(np.mean([r["limsup"] for r in with_fb]))
    limsup_without = float(np.mean([r["limsup"] for r in without]))
    ax.annotate(f"lim sup = {limsup_with:.3f}",
                xy=(x[-1]*0.85, limsup_with + 0.01),
                color=BLUE, fontsize=8)
    ax.annotate(f"lim sup = {limsup_without:.3f}",
                xy=(x[-1]*0.5, limsup_without + 0.01),
                color=RED, fontsize=8)

    ax.set_xlabel("Step index t")
    ax.set_ylabel(r"$\hat{D}_t$ (deviation estimate)")
    ax.set_title("Theorem 2 — Feedback convergence (drift scenario, 10 seeds)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, x[-1])
    ax.set_ylim(0, 0.55)
    ax.grid(alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

    out = FIGS_DIR / "fig2_feedback_convergence.pdf"
    fig.savefig(out)
    print(f"Saved: {out}")
    plt.close(fig)


# ── Figure 3: Compatibility bar ───────────────────────────────────────────────

def fig3_compatibility():
    df = pd.read_csv(RESULTS / "compatibility_results.csv")
    scenarios = ["drift", "sybil", "atomic", "mixed"]
    p1_rates = [df[df.scenario == s]["P1_compat"].mean() for s in scenarios]
    p2_rates = [df[df.scenario == s]["P2_compat"].mean() for s in scenarios]
    equiv    = [df[df.scenario == s]["equiv_rate"].mean() for s in scenarios]

    x = np.arange(len(scenarios))
    w = 0.28

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    b1 = ax.bar(x - w,   p1_rates, width=w, label="P1 compat.", color=BLUE,  alpha=0.85)
    b2 = ax.bar(x,       p2_rates, width=w, label="P2 compat.", color=GREEN, alpha=0.85)
    b3 = ax.bar(x + w,   equiv,    width=w, label="Decision equiv.", color=GRAY, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in scenarios])
    ax.set_ylabel("Rate (0–1)")
    ax.set_ylim(0, 1.12)
    ax.set_title("Theorem 1 — Interface compatibility across scenarios\n(5 seeds per scenario)")
    ax.legend(fontsize=8, loc="lower right")
    ax.axhline(1.0, color="black", lw=0.7, linestyle=":")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    out = FIGS_DIR / "fig3_compatibility_bar.pdf"
    fig.savefig(out)
    print(f"Saved: {out}")
    plt.close(fig)


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating Paper 4 figures...")
    fig1_ablation()
    fig2_feedback()
    fig3_compatibility()
    print("Done.")
