"""
Compare F1 scores between MNIST and Fashion MNIST datasets.

Usage:
    python compare_datasets.py                    # reads results_seeds.json and results_fashion_seeds.json
    python compare_datasets.py --results-mnist my_results.json --results-fashion my_fashion.json
    python compare_datasets.py --results-mnist results_seeds.json --results-fashion results_fashion_seeds.json --multi-seed

Outputs (in paper/figures/):
    comparison_scatter.pdf        – F1 scatter: MNIST (x-axis) vs Fashion (y-axis)
    comparison_scatter_k5.pdf     – Same but with k=5 overlay (if provided)
    comparison_scatter_k50.pdf    – Same but with k=50 overlay (if provided)
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURES_DIR = Path(__file__).parent / "figures"

# Colors keyed by the exact saliency_metric strings stored in the results file
METRIC_COLORS = {
    "$ShapGap_{L2}$":          "darkorange",
    "$ShapGap_{Cosine}$":      "royalblue",
    "MAE":                     "crimson",
    "MSE":                     "red",
    "Earth Mover's Distance":  "forestgreen",
    "Correlation Distance":    "orange",
    "PSNR":                    "pink",
    "Jaccard Distance":        "gold",
    "Czekanowski Distance":    "gray",
    "Sign Agreement Ratio":    "purple",
    "SSIM":                    "brown",
    "KL Divergence":           "teal",
    "AUC-Judd":                "slateblue",
    "NSS Distance":            "mediumseagreen",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_results(path: Path) -> pd.DataFrame:
    """Load JSONL or JSON-array benchmark results; add config and metric_label columns."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df["metric_label"] = df["saliency_metric"]
    df["config"] = (
        df["clip"].map({True: "C", False: "-"}) +
        df["normalize"].map({True: "N", False: "-"}) +
        df["sobel"].map({True: "S", False: "-"})
    )
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean and std F1 / time grouped by (metric_label, config)."""
    g = df.groupby(["metric_label", "config"], sort=False)
    agg = g.agg(
        f1_mean=("f1_score", "mean"),
        f1_std=("f1_score", "std"),
        time_median=("elapsed_time", "median"),
        time_std=("elapsed_time", "std"),
        n_seeds=("f1_score", "count"),
    ).reset_index()
    return agg


# ── Figure: Dataset Comparison ─────────────────────────────────────────────

def plot_comparison_scatter_ax(
    ax,
    df_mnist: pd.DataFrame,
    df_fashion: pd.DataFrame,
    agg_mnist: pd.DataFrame,
    agg_fashion: pd.DataFrame,
    k_value: str = "20",
    xlim: tuple = None,
    ylim: tuple = None,
):
    """Scatter plot on given axis comparing F1 scores between MNIST and Fashion MNIST."""
    # Get best config per metric for each dataset
    best_config_mnist = (
        agg_mnist.loc[agg_mnist.groupby("metric_label")["f1_mean"].idxmax()]
        .set_index("metric_label")["config"]
    )
    best_config_fashion = (
        agg_fashion.loc[agg_fashion.groupby("metric_label")["f1_mean"].idxmax()]
        .set_index("metric_label")["config"]
    )

    # Filter raw data to best configs per metric
    df_mnist_best = df_mnist[
        df_mnist.apply(
            lambda r: r["config"] == best_config_mnist.get(r["metric_label"], ""),
            axis=1
        )
    ]
    df_fashion_best = df_fashion[
        df_fashion.apply(
            lambda r: r["config"] == best_config_fashion.get(r["metric_label"], ""),
            axis=1
        )
    ]

    # Get unique metrics
    metrics = sorted(set(df_mnist_best["metric_label"]) & set(df_fashion_best["metric_label"]))

    # Plot individual run pairs (faint background, colored by metric)
    for metric in metrics:
        color = METRIC_COLORS.get(metric, "steelblue")
        mnist_scores = df_mnist_best[df_mnist_best["metric_label"] == metric]["f1_score"].values
        fashion_scores = df_fashion_best[df_fashion_best["metric_label"] == metric]["f1_score"].values

        # Pair up scores: zip will use the length of the shorter array
        n_pairs = min(len(mnist_scores), len(fashion_scores))
        for i in range(n_pairs):
            ax.scatter(
                mnist_scores[i],
                fashion_scores[i],
                s=30,
                color=color,
                alpha=0.2,
                zorder=1,
            )

    # Plot mean values (colored)
    best_mnist = agg_mnist.loc[agg_mnist.groupby("metric_label")["f1_mean"].idxmax()]
    best_fashion = agg_fashion.loc[agg_fashion.groupby("metric_label")["f1_mean"].idxmax()]

    # Merge on metric_label to get comparable pairs
    comparison = best_mnist[["metric_label", "f1_mean"]].rename(columns={"f1_mean": "f1_mnist"})
    comparison = comparison.merge(
        best_fashion[["metric_label", "f1_mean"]].rename(columns={"f1_mean": "f1_fashion"}),
        on="metric_label",
        how="inner"
    )
    comparison["color"] = comparison["metric_label"].map(METRIC_COLORS)

    # Scatter plot for means
    for _, row in comparison.iterrows():
        ax.scatter(
            row["f1_mnist"],
            row["f1_fashion"],
            s=200,
            color=row["color"],
            alpha=0.9,
            edgecolors="black",
            linewidth=1,
            label=row["metric_label"],
            zorder=3,
        )

    # Diagonal line (perfect agreement)
    if xlim and ylim:
        lims = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
    else:
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1.5)

    # Labels and formatting
    ax.set_xlabel("MNIST F1 (best config)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Fashion MNIST F1 (best config)", fontsize=12, fontweight="bold")
    ax.set_title(f"k={k_value}", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=10)

    # Set limits if provided
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)


def _compute_limits(
    df_mnist, df_fashion, agg_mnist, agg_fashion,
    df_mnist_k5=None, df_fashion_k5=None, agg_mnist_k5=None, agg_fashion_k5=None,
    df_mnist_k50=None, df_fashion_k50=None, agg_mnist_k50=None, agg_fashion_k50=None,
):
    """Compute shared x and y limits across all datasets."""
    all_x = []
    all_y = []

    for df_m, df_f, agg_m, agg_f in [
        (df_mnist, df_fashion, agg_mnist, agg_fashion),
        (df_mnist_k5, df_fashion_k5, agg_mnist_k5, agg_fashion_k5),
        (df_mnist_k50, df_fashion_k50, agg_mnist_k50, agg_fashion_k50),
    ]:
        if df_m is None or df_f is None:
            continue

        best_config_mnist = (
            agg_m.loc[agg_m.groupby("metric_label")["f1_mean"].idxmax()]
            .set_index("metric_label")["config"]
        )
        best_config_fashion = (
            agg_f.loc[agg_f.groupby("metric_label")["f1_mean"].idxmax()]
            .set_index("metric_label")["config"]
        )

        df_m_best = df_m[
            df_m.apply(
                lambda r: r["config"] == best_config_mnist.get(r["metric_label"], ""),
                axis=1
            )
        ]
        df_f_best = df_f[
            df_f.apply(
                lambda r: r["config"] == best_config_fashion.get(r["metric_label"], ""),
                axis=1
            )
        ]

        all_x.extend(df_m_best["f1_score"].values)
        all_y.extend(df_f_best["f1_score"].values)

    xmin = min(all_x) - 0.02
    xmax = max(all_x) + 0.02
    ymin = min(all_y) - 0.02
    ymax = max(all_y) + 0.02

    return (xmin, xmax), (ymin, ymax)


def plot_legend_only():
    """Create a figure with only the legend."""
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.axis("off")

    # Create legend entries
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=METRIC_COLORS.get(metric, "steelblue"),
                   markersize=10, label=metric, markeredgecolor="black", markeredgewidth=0.8)
        for metric in sorted(METRIC_COLORS.keys())
    ]

    legend = ax.legend(
        handles=legend_elements,
        loc="center",
        fontsize=12,
        title="Metric",
        title_fontsize=13,
        ncol=5,
        framealpha=0.95,
    )

    plt.tight_layout()

    # Save
    out = FIGURES_DIR / "comparison_legend.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close(fig)


def plot_comparison_scatter_all(
    df_mnist: pd.DataFrame,
    df_fashion: pd.DataFrame,
    agg_mnist: pd.DataFrame,
    agg_fashion: pd.DataFrame,
    df_mnist_k5: pd.DataFrame = None,
    df_fashion_k5: pd.DataFrame = None,
    agg_mnist_k5: pd.DataFrame = None,
    agg_fashion_k5: pd.DataFrame = None,
    df_mnist_k50: pd.DataFrame = None,
    df_fashion_k50: pd.DataFrame = None,
    agg_mnist_k50: pd.DataFrame = None,
    agg_fashion_k50: pd.DataFrame = None,
):
    """Create 1x3 subplot figure with k=5, k=20, k=50 comparison scatter plots (ordered left to right)."""
    # Determine how many plots we need
    n_plots = 1
    plots_to_make = [("20", df_mnist, df_fashion, agg_mnist, agg_fashion)]

    if agg_mnist_k5 is not None and agg_fashion_k5 is not None:
        n_plots += 1
        plots_to_make.insert(0, ("5", df_mnist_k5, df_fashion_k5, agg_mnist_k5, agg_fashion_k5))

    if agg_mnist_k50 is not None and agg_fashion_k50 is not None:
        n_plots += 1
        plots_to_make.append(("50", df_mnist_k50, df_fashion_k50, agg_mnist_k50, agg_fashion_k50))

    # Compute shared limits
    xlim, ylim = _compute_limits(
        df_mnist, df_fashion, agg_mnist, agg_fashion,
        df_mnist_k5, df_fashion_k5, agg_mnist_k5, agg_fashion_k5,
        df_mnist_k50, df_fashion_k50, agg_mnist_k50, agg_fashion_k50,
    )

    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    # Plot in order: k=5, k=20, k=50
    for ax_idx, (k_value, df_m, df_f, agg_m, agg_f) in enumerate(plots_to_make):
        plot_comparison_scatter_ax(
            axes[ax_idx], df_m, df_f, agg_m, agg_f,
            k_value=k_value,
            xlim=xlim,
            ylim=ylim,
        )

    fig.suptitle("Metric Performance Comparison across Prototype Sets", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save main figure
    out = FIGURES_DIR / "comparison_scatter_all.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close(fig)

    # Save legend separately
    plot_legend_only()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-mnist", default="results_seeds.json",
                        help="MNIST results file (default: results_seeds.json)")
    parser.add_argument("--results-fashion", default="results_fashion_seeds.json",
                        help="Fashion MNIST results file (default: results_fashion_seeds.json)")
    parser.add_argument("--results-k5-mnist", default=None,
                        help="Optional MNIST k=5 results file")
    parser.add_argument("--results-k5-fashion", default=None,
                        help="Optional Fashion MNIST k=5 results file")
    parser.add_argument("--results-k50-mnist", default=None,
                        help="Optional MNIST k=50 results file")
    parser.add_argument("--results-k50-fashion", default=None,
                        help="Optional Fashion MNIST k=50 results file")
    parser.add_argument("--multi-seed", action="store_true",
                        help="Show ±std annotations in plots")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load primary datasets
    print(f"Loading {args.results_mnist} (MNIST) ...")
    df_mnist = load_results(Path(args.results_mnist))
    agg_mnist = aggregate(df_mnist)
    print(f"  {len(df_mnist)} rows  |  {df_mnist['metric_label'].nunique()} metrics")

    print(f"Loading {args.results_fashion} (Fashion MNIST) ...")
    df_fashion = load_results(Path(args.results_fashion))
    agg_fashion = aggregate(df_fashion)
    print(f"  {len(df_fashion)} rows  |  {df_fashion['metric_label'].nunique()} metrics")

    # Load optional k=5 datasets
    df_mnist_k5, agg_mnist_k5 = None, None
    df_fashion_k5, agg_fashion_k5 = None, None
    if args.results_k5_mnist and args.results_k5_fashion:
        print(f"Loading {args.results_k5_mnist} (MNIST k=5) ...")
        df_mnist_k5 = load_results(Path(args.results_k5_mnist))
        agg_mnist_k5 = aggregate(df_mnist_k5)
        print(f"  {len(df_mnist_k5)} rows")

        print(f"Loading {args.results_k5_fashion} (Fashion MNIST k=5) ...")
        df_fashion_k5 = load_results(Path(args.results_k5_fashion))
        agg_fashion_k5 = aggregate(df_fashion_k5)
        print(f"  {len(df_fashion_k5)} rows")

    # Load optional k=50 datasets
    df_mnist_k50, agg_mnist_k50 = None, None
    df_fashion_k50, agg_fashion_k50 = None, None
    if args.results_k50_mnist and args.results_k50_fashion:
        print(f"Loading {args.results_k50_mnist} (MNIST k=50) ...")
        df_mnist_k50 = load_results(Path(args.results_k50_mnist))
        agg_mnist_k50 = aggregate(df_mnist_k50)
        print(f"  {len(df_mnist_k50)} rows")

        print(f"Loading {args.results_k50_fashion} (Fashion MNIST k=50) ...")
        df_fashion_k50 = load_results(Path(args.results_k50_fashion))
        agg_fashion_k50 = aggregate(df_fashion_k50)
        print(f"  {len(df_fashion_k50)} rows")

    print("Generating comparison figures ...")
    plot_comparison_scatter_all(
        df_mnist, df_fashion, agg_mnist, agg_fashion,
        df_mnist_k5, df_fashion_k5, agg_mnist_k5, agg_fashion_k5,
        df_mnist_k50, df_fashion_k50, agg_mnist_k50, agg_fashion_k50,
    )
    print("Done.")


if __name__ == "__main__":
    main()
