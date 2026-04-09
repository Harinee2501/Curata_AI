# plot_results.py
# -------------------------------------------------------------------
# Reads results.json produced by pipeline.py and generates two plots:
#   1. Validation F1 over bandit iterations
#   2. Bar chart comparing all three training regimes on test F1
# -------------------------------------------------------------------

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import config

COLORS = {
    "baseline": "#5F5E5A",
    "naive":    "#D85A30",
    "bandit":   "#1D9E75",
}


def plot(results_path: str = config.RESULTS_PATH):
    with open(results_path) as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("RL-Guided Synthetic Data Selection — Results", fontsize=14, y=1.02)

    # ── Plot 1: Val F1 over iterations ────────────────────────────
    ax1 = axes[0]
    f1_history = results["bandit_guided"]["val_f1_history"]
    ax1.plot(f1_history, color=COLORS["bandit"], linewidth=2, marker="o", markersize=4)
    ax1.axhline(
        results["baseline_real_only"]["val_f1"],
        color=COLORS["baseline"], linestyle="--", linewidth=1.5, label="Real Only"
    )
    ax1.axhline(
        results["naive_augmentation"]["val_f1"],
        color=COLORS["naive"], linestyle="--", linewidth=1.5, label="Naive Augmentation"
    )
    ax1.set_title("Validation F1 — Bandit Iterations", fontsize=12)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Macro F1")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1.0)

    # ── Plot 2: Test F1 bar comparison ────────────────────────────
    ax2 = axes[1]
    labels = ["Real Only", "Naive\nAugmentation", "Bandit-Guided\n(Ours)"]
    values = [
        results["baseline_real_only"]["test_f1"],
        results["naive_augmentation"]["test_f1"],
        results["bandit_guided"]["test_f1"],
    ]
    bar_colors = [COLORS["baseline"], COLORS["naive"], COLORS["bandit"]]

    bars = ax2.bar(labels, values, color=bar_colors, width=0.5, edgecolor="white", linewidth=0.5)

    # Annotate bars with F1 values
    for bar, val in zip(bars, values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax2.set_title("Test F1 — Method Comparison", fontsize=12)
    ax2.set_ylabel("Macro F1")
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("results_plot.png", dpi=150, bbox_inches="tight")
    print("[plot] Saved 'results_plot.png'")
    plt.show()


if __name__ == "__main__":
    plot()
