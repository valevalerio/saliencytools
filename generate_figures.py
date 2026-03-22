"""
Generate all paper figures from multi-seed benchmark results.

Usage:
    python paper/generate_figures.py                             # reads results_seeds.json
    python paper/generate_figures.py --results my_results.json
    python paper/generate_figures.py --results results_seeds.json --multi-seed
    python paper/generate_figures.py --results results_seeds.json --results-k5 results_k5.json
    python paper/generate_figures.py --results results_seeds.json --results-k5 results_k5.json --results-k50 results_k50.json

Outputs (in paper/figures/):
    heatmap.pdf          – mean F1 per (metric × config), best config circled
    f1_vs_time.pdf       – mean F1 vs median time per metric, error bars = ±1 std
    stability.pdf        – F1 std of best config per metric (shows sensitivity to seed)
    joyplot.pdf          – ridgeline KDE of F1 at best config per metric (EMD excluded)
                           if --results-k5/--results-k50 are given, dashed/dotted overlays added
    prototypes.pdf       – example prototypical saliency maps with SHAP colormap
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

# SHAP-style colormap: blue → white → pink/magenta  (mirrors tutorial.ipynb)
SHAP_CMAP = LinearSegmentedColormap.from_list(
    "shap", [(0.0, "#008bfb"), (0.5, "#ffffff"), (1.0, "#ff0051")]
)

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


# ── Figure 1: Heatmap ─────────────────────────────────────────────────────────

INVALID_CONFIGS = {
    # Czekanowski and Jaccard require non-negative inputs.
    # Σ(a+b) ≈ 0 on signed maps → denominator collapses → values outside [0,1].
    # Valid only when normalize is the final op (no Sobel after).
    "Czekanowski Distance": {"---", "C--", "--S", "C-S", "-NS", "CNS"},
    "Jaccard Distance":     {"---", "C--", "--S", "C-S", "-NS", "CNS"},
    # SAR measures sign agreement. normalize_mask_0_1 maps to [0,1] → all signs +1
    # → SAR = 1.0 trivially for any pair. Configs where normalize=True are invalid.
    "Sign Agreement Ratio": {"-N-", "CN-", "-NS", "CNS"},
}

def plot_heatmap(agg: pd.DataFrame, multi_seed: bool):
    """Heatmap of mean F1 per (metric × preprocessing config), best config circled.

    Cells where a metric's mathematical assumptions are violated are set to NaN;
    seaborn leaves those cells white automatically via the mask parameter.
    """
    pivot_mean = agg.pivot(index="metric_label", columns="config", values="f1_mean")
    pivot_std  = agg.pivot(index="metric_label", columns="config", values="f1_std")

    for metric, bad_configs in INVALID_CONFIGS.items():
        if metric in pivot_mean.index:
            for cfg in bad_configs:
                if cfg in pivot_mean.columns:
                    pivot_mean.loc[metric, cfg] = np.nan
                    pivot_std.loc[metric, cfg]  = np.nan

    # Sort rows by best valid F1 descending
    order = pivot_mean.max(axis=1).sort_values(ascending=False).index
    pivot_mean = pivot_mean.loc[order]
    pivot_std  = pivot_std.loc[order]

    # Build annotation: "mean\n±std" when multi-seed, else just mean
    if multi_seed and pivot_std.notna().any().any():
        annot = pivot_mean.copy().astype(object)
        for r in pivot_mean.index:
            for c in pivot_mean.columns:
                m = pivot_mean.loc[r, c]
                s = pivot_std.loc[r, c]
                annot.loc[r, c] = f"{m:.3f}\n±{s:.3f}" if not np.isnan(s) else f"{m:.3f}"
        fmt = ""
    else:
        annot = True
        fmt = ".3f"

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.heatmap(
        pivot_mean,
        annot=annot,
        fmt=fmt,
        cmap="Blues",
        mask=pivot_mean.isna(),
        cbar_kws={"label": "Mean F1"},
        ax=ax,
        annot_kws={"size": 7},
    )

    # Circle best config(s) per metric
    for metric in pivot_mean.index:
        best_val = pivot_mean.loc[metric].max()
        for col in pivot_mean.columns:
            if np.isclose(pivot_mean.loc[metric, col], best_val, atol=1e-6):
                y = list(pivot_mean.index).index(metric) + 0.5
                x = list(pivot_mean.columns).index(col) + 0.5
                ax.add_patch(plt.Circle((x, y), 0.4, color="red", fill=False, lw=1.2))

    ax.set_title("Mean F1 by Metric and Preprocessing", fontsize=13)
    ax.set_ylabel("Distance Metric")
    ax.set_xlabel("Preprocessing  (C=Clip  N=Normalize  S=Sobel)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()

    out = FIGURES_DIR / "heatmap.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close(fig)


# ── Figure 2: F1 vs Time ──────────────────────────────────────────────────────

def plot_f1_vs_time(agg: pd.DataFrame, multi_seed: bool):
    """Scatter of best-config mean F1 vs median inference time, with ±std error bars."""
    best = agg.loc[agg.groupby("metric_label")["f1_mean"].idxmax()].copy()
    best["color"] = best["metric_label"].map(METRIC_COLORS)

    fig, ax = plt.subplots(figsize=(7, 5))

    for _, row in best.iterrows():
        ax.errorbar(
            row["time_median"],
            row["f1_mean"],
            yerr=row["f1_std"] if multi_seed else None,
            fmt="o",
            color=row["color"],
            markersize=8,
            capsize=4,
            label=row["metric_label"],
        )

    ax.set_xlabel("Median Inference Time per Config (s)", fontsize=11)
    ax.set_ylabel("Best Mean F1 (macro)", fontsize=11)
    ax.set_title("F1 vs. Computation Time (best config per metric)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=8,
        title="Metric",
    )
    plt.tight_layout()

    out = FIGURES_DIR / "f1_vs_time.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close(fig)


# ── Figure 3: Stability ───────────────────────────────────────────────────────

def plot_stability(agg: pd.DataFrame):
    """F1 std of the best config per metric — how sensitive is each metric to prototype seed?"""
    best = agg.loc[agg.groupby("metric_label")["f1_mean"].idxmax()].copy()
    best = best.sort_values("f1_std", ascending=False)
    best["color"] = best["metric_label"].map(METRIC_COLORS)

    fig, ax = plt.subplots(figsize=(7, 4))
    rects = ax.barh(
        best["metric_label"], best["f1_std"],
        color=best["color"].values, edgecolor="black", linewidth=0.5,
    )
    for rect, val in zip(rects, best["f1_std"]):
        ax.text(
            val + 0.0005, rect.get_y() + rect.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=8,
        )

    ax.set_xlabel("F1 Std Dev across seeds (best config)", fontsize=11)
    ax.set_title("Metric Stability across Prototype Draws", fontsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out = FIGURES_DIR / "stability.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close(fig)


# ── Figure 4: Joyplot ────────────────────────────────────────────────────────

JOYPLOT_EXCLUDE = {"Earth Mover's Distance"}   # outlier — excluded for readability


def _best_config(agg: pd.DataFrame) -> pd.Series:
    """Return Series metric_label → best config string."""
    return (
        agg.loc[agg.groupby("metric_label")["f1_mean"].idxmax()]
        .set_index("metric_label")["config"]
    )


def _filter_best(df: pd.DataFrame, best_cfg: pd.Series) -> pd.DataFrame:
    """Keep only rows matching each metric's best config; drop excluded metrics."""
    return df[
        (~df["metric_label"].isin(JOYPLOT_EXCLUDE)) &
        df.apply(lambda r: r["config"] == best_cfg.get(r["metric_label"], ""), axis=1)
    ][["metric_label", "f1_score"]].copy()


def plot_joyplot(
    df: pd.DataFrame,
    agg: pd.DataFrame,
    df_k5: pd.DataFrame = None,
    agg_k5: pd.DataFrame = None,
    df_k50: pd.DataFrame = None,
    agg_k50: pd.DataFrame = None,
):
    """Ridgeline KDE of F1 across seeds at each metric's best preprocessing config.

    EMD is excluded as an outlier (F1 ~ 0.38) that compresses the x-axis.
    - k=20 (primary): solid fill
    - k=5  (optional): dashed overlay
    - k=50 (optional): dotted overlay
    """
    best_cfg     = _best_config(agg)
    df_best      = _filter_best(df, best_cfg)
    best_cfg_k5  = _best_config(agg_k5)  if agg_k5  is not None else None
    df_best_k5   = _filter_best(df_k5,  best_cfg_k5)  if df_k5  is not None else None
    best_cfg_k50 = _best_config(agg_k50) if agg_k50 is not None else None
    df_best_k50  = _filter_best(df_k50, best_cfg_k50) if df_k50 is not None else None

    # Sort descending so the best lands at i=0 (bottom axis)
    order = (
        df_best.groupby("metric_label")["f1_score"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    x_min = df_best["f1_score"].min() - 0.03
    x_max = df_best["f1_score"].max() + 0.03
    if df_best_k5 is not None:
        x_min = min(x_min, df_best_k5["f1_score"].min() - 0.03)
        x_max = max(x_max, df_best_k5["f1_score"].max() + 0.03)
    if df_best_k50 is not None:
        x_min = min(x_min, df_best_k50["f1_score"].min() - 0.03)
        x_max = max(x_max, df_best_k50["f1_score"].max() + 0.03)

    n          = len(order)
    fig_w      = 12.0    # wider than tall
    row_inch   = 0.14    # tight row spacing
    kde_inch   = 1.2     # KDE height budget (allows overlap)
    label_frac = 0.26    # fraction of fig width for left labels
    fig_h      = row_inch * n + kde_inch   # e.g. 12 metrics → ~6.4 in

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_alpha(0.0)   # transparent figure background

    # Add axes from TOP to BOTTOM so that axes[0] (SAR, bottom) is added
    # last and therefore has the highest z-order — its white background
    # cuts cleanly over the axes above it (classic joyplot behaviour).
    axes = [None] * n
    left  = label_frac + 0.01
    width = 0.97 - left
    for i in range(n - 1, -1, -1):
        bottom_pos = (i * row_inch) / fig_h
        height_pos = kde_inch / fig_h
        axes[i] = fig.add_axes([left, bottom_pos, width, height_pos])

    for i, (metric, ax) in enumerate(zip(order, axes)):
        color = METRIC_COLORS.get(metric, "steelblue")
        cfg   = best_cfg[metric]
        vals  = df_best[df_best["metric_label"] == metric]["f1_score"].values

        if len(vals) >= 2:
            sns.kdeplot(x=vals, ax=ax, fill=True, alpha=0.6,
                        color=color, linewidth=0,
                        bw_adjust=1.2, clip=(x_min, x_max))
            sns.kdeplot(x=vals, ax=ax, fill=False,
                        color=color, linewidth=1.8,
                        bw_adjust=1.2, clip=(x_min, x_max))

        if df_best_k5 is not None and metric in best_cfg_k5.index:
            vals_k5 = df_best_k5[df_best_k5["metric_label"] == metric]["f1_score"].values
            if len(vals_k5) >= 2:
                sns.kdeplot(x=vals_k5, ax=ax, fill=True, alpha=0.4,
                            color=color, linewidth=0,
                            bw_adjust=1.2, clip=(x_min, x_max))
                sns.kdeplot(x=vals_k5, ax=ax, fill=False,
                            color=color, linewidth=1.8, linestyle="--",
                            bw_adjust=1.2, clip=(x_min, x_max))

        if df_best_k50 is not None and metric in best_cfg_k50.index:
            vals_k50 = df_best_k50[df_best_k50["metric_label"] == metric]["f1_score"].values
            if len(vals_k50) >= 2:
                sns.kdeplot(x=vals_k50, ax=ax, fill=True, alpha=0.25,
                            color=color, linewidth=0,
                            bw_adjust=1.2, clip=(x_min, x_max))
                sns.kdeplot(x=vals_k50, ax=ax, fill=False,
                            color=color, linewidth=1.8, linestyle=":",
                            bw_adjust=1.2, clip=(x_min, x_max))

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(bottom=0)
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.patch.set_facecolor("none")
        ax.patch.set_alpha(0.80)

        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        # Colored baseline visible on every row; x-axis ticks only on the bottom row
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["bottom"].set_color(color)
        if i != 0:
            ax.tick_params(bottom=False, labelbottom=False)

        # Label in the left margin via figure coordinates
        label_x = label_frac - 0.01
        label_y = (i * row_inch + row_inch * 0.3) / fig_h
        fig.text(label_x, label_y, f"{metric}  [{cfg}]",
                 ha="right", va="center", fontsize=8.5,
                 transform=fig.transFigure)

    axes[0].set_xlabel("F1 Score (macro)", fontsize=11)
    #axes[0].spines["bottom"].set_color("black")   # x-axis stays black

    if df_best_k5 is not None or df_best_k50 is not None:
        legend_handles = [
            Line2D([0], [0], color="black", linewidth=1.8,
                   label=r"$k=20$ prototypes/class"),
        ]
        if df_best_k5 is not None:
            legend_handles.append(
                Line2D([0], [0], color="black", linewidth=1.8, linestyle="--",
                       label=r"$k=5$ prototypes/class")
            )
            # switch the order so that k=5 is above k=20 in the legend (mirrors plot layering)
            legend_handles = legend_handles[::-1]
        if df_best_k50 is not None:
            legend_handles.append(
                Line2D([0], [0], color="black", linewidth=1.8, linestyle=":",
                       label=r"$k=50$ prototypes/class")
            )
        axes[-1].legend(handles=legend_handles, loc="upper left",
                        fontsize=8.5, framealpha=0.85)

    fig.text(0.5, 1.004,
             "F1 Distribution across Seeds (best config per metric, EMD excluded)",
             ha="center", va="bottom", fontsize=12,
             transform=fig.transFigure)

    out = FIGURES_DIR / "joyplot.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close(fig)


# ── Figure 5: Prototypical samples ───────────────────────────────────────────

def plot_prototypes(n_classes: int = 10, n_proto: int = 2, seed: int = 42):
    """Show n_proto prototypical saliency maps per class with the SHAP colormap."""
    try:
        from scipy.ndimage import sobel as scipy_sobel
        from sklearn.model_selection import train_test_split
        from torchvision.datasets import MNIST
    except ImportError as exc:
        print(f"  Skipping prototypes figure — missing dependency: {exc}", file=sys.stderr)
        return

    dataset = MNIST(root="data", train=True, download=True)
    X = np.array(dataset.data, dtype=np.float32) / 255.0
    y = np.array(dataset.targets)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    mean = X_train.mean()
    X_train = np.array([scipy_sobel(img - mean) for img in X_train])

    rng = np.random.default_rng(seed)
    chosen_per_class = {}
    for cls in range(n_classes):
        idx = np.where(y_train == cls)[0]
        chosen_per_class[cls] = rng.choice(idx, size=n_proto, replace=False)

    vmin, vmax = X_train.min(), X_train.max()
    per_class_dir = FIGURES_DIR / "prototypes_per_class"
    per_class_dir.mkdir(exist_ok=True)

    for cls in range(n_classes):
        fig, axes_pair = plt.subplots(1, n_proto, figsize=(n_proto * 2.0, 2.2))
        if n_proto == 1:
            axes_pair = [axes_pair]
        for col, img_idx in enumerate(chosen_per_class[cls]):
            ax = axes_pair[col]
            im = ax.imshow(X_train[img_idx], cmap=SHAP_CMAP, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
            ax.set_xlabel("Ground Truth" if col == 0 else f"Sample {col + 1}", fontsize=9)
        fig.suptitle(f"Class {cls} — prototype saliency maps", fontsize=9, y=1.02)
        plt.tight_layout(rect=[0, 0, 0.88, 1])
        cbar_ax = fig.add_axes([0.90, 0.15, 0.03, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="Attribution")
        out_cls = per_class_dir / f"class_{cls}.pdf"
        fig.savefig(out_cls, dpi=300, bbox_inches="tight")
        fig.savefig(str(out_cls).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved per-class prototype figures → {per_class_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results_seeds.json",
                        help="Primary JSONL results file (default: results_seeds.json)")
    parser.add_argument("--results-k5", default=None,
                        help="Optional second file (k=5) for joyplot dashed overlay")
    parser.add_argument("--results-k50", default=None,
                        help="Optional third file (k=50) for joyplot dotted overlay")
    parser.add_argument("--multi-seed", action="store_true",
                        help="Show ±std annotations in heatmap and error bars in scatter")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.results} ...")
    df = load_results(Path(args.results))
    n_seeds = df["seed"].nunique() if "seed" in df.columns else 1
    multi_seed = args.multi_seed and n_seeds > 1
    print(f"  {len(df)} rows  |  {n_seeds} seed(s)  |  {df['metric_label'].nunique()} metrics")
    agg = aggregate(df)

    df_k5, agg_k5 = None, None
    if args.results_k5:
        print(f"Loading {args.results_k5} (k=5 overlay) ...")
        df_k5  = load_results(Path(args.results_k5))
        agg_k5 = aggregate(df_k5)
        print(f"  {len(df_k5)} rows  |  {df_k5['metric_label'].nunique()} metrics")

    df_k50, agg_k50 = None, None
    if args.results_k50:
        print(f"Loading {args.results_k50} (k=50 overlay) ...")
        df_k50  = load_results(Path(args.results_k50))
        agg_k50 = aggregate(df_k50)
        print(f"  {len(df_k50)} rows  |  {df_k50['metric_label'].nunique()} metrics")

    print("Generating figures ...")
    plot_heatmap(agg, multi_seed)
    plot_f1_vs_time(agg, multi_seed)
    if multi_seed:
        plot_stability(agg)
    plot_joyplot(df, agg, df_k5=df_k5, agg_k5=agg_k5, df_k50=df_k50, agg_k50=agg_k50)
    plot_prototypes()
    print("Done.")


if __name__ == "__main__":
    main()
