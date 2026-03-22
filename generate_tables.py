"""
Generate LaTeX table files from experiment results.

Supports two formats:
  - Single-run (results_final.json): one row per (metric, config)
  - Multi-seed (results_seeds.json): multiple rows per (metric, config, seed)
    -> aggregated to mean +/- std across seeds

Usage:
    python paper/generate_tables.py                              # single-run default
    python paper/generate_tables.py --results results_seeds.json # auto-detects multi-seed
    python paper/generate_tables.py --results results_seeds.json --seeds 10

Output:
    paper/tables/table_results.tex
"""

import json
import argparse
import math
from pathlib import Path
from collections import defaultdict

REPO_ROOT  = Path(__file__).parent.parent
TABLES_DIR = Path(__file__).parent / "tables"

# Display name overrides (raw JSON key -> LaTeX label)
METRIC_LABELS = {
    "Sign Agreement Ratio":    r"Sign Agreement Ratio (SAR)",
    "SSIM":                    r"SSIM",
    "MAE":                     r"MAE",
    "Czekanowski Distance":    r"Czekanowski Distance",
    "Jaccard Distance":        r"Jaccard Distance",
    "$ShapGap_{Cosine}$":      r"$d_{\cos}$ (ShapGap-Cosine)",
    "Correlation Distance":    r"Correlation Distance",
    "$ShapGap_{L2}$":          r"$d_{L_2}$ (ShapGap-L2)",
    "MSE":                     r"MSE",
    "PSNR":                    r"PSNR",
    "Earth Mover's Distance":  r"Earth Mover's Distance",
    "KL Divergence":           r"KL Divergence",
    "AUC-Judd":                r"AUC-Judd",
}

ROW_ORDER = [
    "Sign Agreement Ratio",
    "SSIM",
    "MAE",
    "Czekanowski Distance",
    "Jaccard Distance",
    "$ShapGap_{Cosine}$",
    "Correlation Distance",
    "$ShapGap_{L2}$",
    "MSE",
    "PSNR",
    "Earth Mover's Distance",
    "KL Divergence",
    "AUC-Judd",
]


# ── Loading ──────────────────────────────────────────────────────────────────

def load_results(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_multi_seed(rows: list[dict]) -> bool:
    return "seed" in rows[0] if rows else False


# ── Aggregation ──────────────────────────────────────────────────────────────

def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def best_per_metric_single(rows: list[dict]) -> dict:
    """Single-run: return {metric: best_row} by highest f1_score."""
    best = {}
    for row in rows:
        m = row["saliency_metric"]
        if m not in best or row["f1_score"] > best[m]["f1_score"]:
            best[m] = row
    return best


def best_per_metric_multi(rows: list[dict]) -> dict:
    """
    Multi-seed: group by (metric, config), compute mean+std F1,
    then return the config with highest mean F1 per metric.
    Returns {metric: {f1_mean, f1_std, elapsed_mean, clip, normalize, sobel, n_seeds}}.
    """
    # Group: metric -> config_key -> list of (f1, elapsed)
    groups: dict[str, dict[tuple, list]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        m = row["saliency_metric"]
        cfg = (row["clip"], row["normalize"], row["sobel"])
        groups[m][cfg].append((row["f1_score"], row["elapsed_time"]))

    def _cfg_op_count(cfg: tuple) -> int:
        """Count number of preprocessing operations (True flags) in config tuple."""
        return sum(1 for v in cfg if v)

    best = {}
    for m, configs in groups.items():
        best_cfg, best_stats = None, None
        for cfg, vals in configs.items():
            f1s    = [v[0] for v in vals]
            times  = [v[1] for v in vals]
            stats  = {
                "f1_mean":      sum(f1s) / len(f1s),
                "f1_std":       std(f1s),
                "elapsed_mean": sum(times) / len(times),
                "clip":         cfg[0],
                "normalize":    cfg[1],
                "sobel":        cfg[2],
                "n_seeds":      len(f1s),
            }
            if best_stats is None or stats["f1_mean"] > best_stats["f1_mean"]:
                best_cfg, best_stats = cfg, stats
            elif stats["f1_mean"] == best_stats["f1_mean"]:
                # Tie-breaking: prefer config with fewer preprocessing operations
                if _cfg_op_count(cfg) < _cfg_op_count(best_cfg):
                    best_cfg, best_stats = cfg, stats
        best[m] = best_stats
    return best


# ── Rendering ────────────────────────────────────────────────────────────────

def config_str(row: dict) -> str:
    c = "C" if row["clip"]      else "-"
    n = "N" if row["normalize"] else "-"
    s = "S" if row["sobel"]     else "-"
    return rf"[{c}{n}{s}]"


def make_table_single(best: dict, source_name: str) -> str:
    header_cols  = r"\textbf{Metric} & \textbf{Best $\text{F}_1$} & \textbf{Config (C/N/S)} & \textbf{Time (s)} & \textbf{Rank} \\"
    col_spec     = "lcccc"
    note         = r"\emph{Note: single-run results; variance estimates pending.}"

    sorted_metrics = _sort_metrics(best, key=lambda m: -best[m]["f1_score"])

    rows = []
    prev_f1, rank, skip = None, 0, 0
    for m in sorted_metrics:
        row   = best[m]
        f1    = row["f1_score"]
        rank, skip, prev_f1 = _rank(f1, prev_f1, rank, skip)
        label = METRIC_LABELS.get(m, m)
        bold  = r"\textbf{" + f"{f1:.3f}" + "}" if rank == 1 else f"{f1:.3f}"
        rows.append(
            rf"{label} & {bold} & {config_str(row)} & {row['elapsed_time']:.1f} & {rank} \\"
        )

    return _wrap_table(col_spec, header_cols, rows, note, source_name)


def make_table_multi(best: dict, source_name: str, n_seeds: int) -> str:
    header_cols = (
        r"\textbf{Metric} & \textbf{Mean $\text{F}_1$} & \textbf{Std} & "
        r"\textbf{Config} & \textbf{Time (s)} & \textbf{Rank} \\"
    )
    col_spec = "lccccc"
    note     = rf"Results aggregated over {n_seeds} random seeds (mean\,$\pm$\,std macro-$\text{{F}}_1$)."

    sorted_metrics = _sort_metrics(best, key=lambda m: -best[m]["f1_mean"])

    rows = []
    prev_f1, rank, skip = None, 0, 0
    for m in sorted_metrics:
        row    = best[m]
        f1     = row["f1_mean"]
        rank, skip, prev_f1 = _rank(f1, prev_f1, rank, skip)
        label  = METRIC_LABELS.get(m, m)
        f1_str = r"\textbf{" + f"{f1:.3f}" + "}" if rank == 1 else f"{f1:.3f}"
        std_str = f"{row['f1_std']:.3f}"
        cfg    = config_str(row)
        t      = f"{row['elapsed_mean']:.1f}"
        rows.append(rf"{label} & {f1_str} & {std_str} & {cfg} & {t} & {rank} \\")

    return _wrap_table(col_spec, header_cols, rows, note, source_name)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sort_metrics(best: dict, key) -> list[str]:
    ordered  = [m for m in ROW_ORDER if m in best]
    ordered += [m for m in best if m not in ordered]
    return sorted(ordered, key=key)


def _rank(f1, prev_f1, rank, skip):
    if f1 != prev_f1:
        rank += 1 + skip
        skip = 0
    else:
        skip += 1
    return rank, skip, f1


def _wrap_table(col_spec, header_cols, data_rows, note, source_name) -> str:
    lines = [
        r"% Auto-generated by paper/generate_tables.py",
        rf"% Source: {source_name}",
        r"% DO NOT EDIT BY HAND — re-run generate_tables.py instead",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Best macro-$\text{F}_1$ per metric across all 8 preprocessing",
        r"         configurations (C\,=\,clip to $[-1,1]$;",
        r"         N\,=\,normalize to $[0,1]$; S\,=\,Sobel filter).",
        r"         Runtime is wall-clock time for the full test set.",
        rf"         {note}}}",
        r"\label{tab:results}",
        r"\small",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header_cols,
        r"\midrule",
        *data_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results_final.json",
                        help="Path to results JSON (relative to repo root, or absolute)")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Override number of seeds shown in caption")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.is_absolute():
        results_path = REPO_ROOT / results_path
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    rows       = load_results(results_path)
    multi_seed = is_multi_seed(rows)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "table_results.tex"

    if multi_seed:
        best    = best_per_metric_multi(rows)
        n_seeds = args.seeds or max(r.get("n_seeds", 1) for r in best.values())
        content = make_table_multi(best, results_path.name, n_seeds)
        print(f"Multi-seed mode: {n_seeds} seeds, {len(best)} metrics")
    else:
        best    = best_per_metric_single(rows)
        content = make_table_single(best, results_path.name)
        print(f"Single-run mode: {len(best)} metrics")

    out.write_text(content)
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
