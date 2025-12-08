"""Plotting functions for analysis visualization."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .correlation import compute_abs_delta


def _resolve_trans_metric(
    metric: str,
    baseline_data: Dict[str, Dict[str, float]],
    finetuned_data: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Resolve a translation metric string to actual data."""
    if metric.startswith("delta_"):
        key = metric.replace("delta_", "")
        base = baseline_data.get(key, {})
        fine = finetuned_data.get(key, {})
        return compute_abs_delta(base, fine)
    elif metric.startswith("finetuned_"):
        key = metric.replace("finetuned_", "")
        return finetuned_data.get(key, {})
    elif metric.startswith("baseline_"):
        key = metric.replace("baseline_", "")
        return baseline_data.get(key, {})
    else:
        return baseline_data.get(metric, {})


def scatter_4d(
    sim_data: Dict[str, Dict[str, float]],
    trans_data: Dict[str, Dict[str, float]],
    x_metric: str,
    y_metric: str,
    size_metric: str,
    color_metric: str,
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    """
    Create a 4D scatter plot where each point is a language pair.
    """
    # Get data for each metric
    x_data = sim_data.get(x_metric, {})
    y_data = trans_data.get(y_metric, {})
    size_data = sim_data.get(size_metric, {})
    color_data = sim_data.get(color_metric, {})

    # Find common pairs across all metrics
    common_pairs = sorted(
        set(x_data.keys())
        & set(y_data.keys())
        & set(size_data.keys())
        & set(color_data.keys())
    )

    if len(common_pairs) < 2:
        print(
            f"Warning: Not enough common pairs for plot ({len(common_pairs)}). Skipping."
        )
        return

    # Extract values
    x_vals = np.array([x_data[p] for p in common_pairs])
    y_vals = np.array([y_data[p] for p in common_pairs])
    size_vals = np.array([size_data[p] for p in common_pairs])
    color_vals = np.array([color_data[p] for p in common_pairs])

    # Normalize size to reasonable point sizes (50-500)
    size_min, size_max = size_vals.min(), size_vals.max()
    if size_max > size_min:
        sizes = 50 + 450 * (size_vals - size_min) / (size_max - size_min)
    else:
        sizes = np.full_like(size_vals, 200)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        x_vals,
        y_vals,
        s=sizes,
        c=color_vals,
        cmap="viridis",
        alpha=0.7,
        edgecolors="white",
        linewidth=0.5,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(color_metric, fontsize=10)

    # Add point labels
    for i, pair in enumerate(common_pairs):
        ax.annotate(
            pair,
            (x_vals[i], y_vals[i]),
            fontsize=8,
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
        )

    # Labels and title
    ax.set_xlabel(f"{x_metric} (similarity)", fontsize=11)
    ax.set_ylabel(f"{y_metric} (translation quality)", fontsize=11)

    if title:
        ax.set_title(title, fontsize=12)
    else:
        ax.set_title(
            f"Translation Quality vs Similarity\n(size={size_metric}, color={color_metric})",
            fontsize=12,
        )

    # Add size legend
    _add_size_legend(ax, size_metric, size_min, size_max)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def _add_size_legend(ax, metric_name: str, val_min: float, val_max: float) -> None:
    """Add a legend showing what point sizes mean."""
    # Create dummy points for legend
    legend_sizes = [50, 200, 400]
    legend_vals = [
        val_min,
        (val_min + val_max) / 2,
        val_max,
    ]

    legend_elements = [
        plt.scatter(
            [],
            [],
            s=s,
            c="gray",
            alpha=0.7,
            edgecolors="white",
            linewidth=0.5,
            label=f"{metric_name}={v:.2f}",
        )
        for s, v in zip(legend_sizes, legend_vals)
    ]

    ax.legend(
        handles=legend_elements,
        title="Point Size",
        loc="upper left",
        fontsize=8,
        title_fontsize=9,
    )


def generate_all_scatter_plots(
    sim_data: Dict[str, Dict[str, float]],
    baseline_data: Dict[str, Dict[str, float]],
    finetuned_data: Dict[str, Dict[str, float]],
    plot_configs: List[Dict],
    output_dir: Path,
) -> None:
    """
    Generate all scatter plots from config.
    """
    for i, cfg in enumerate(plot_configs):
        x_metric = cfg.get("x")
        y_metric = cfg.get("y")
        size_metric = cfg.get("size")
        color_metric = cfg.get("color")

        if not all([x_metric, y_metric, size_metric, color_metric]):
            print(f"Warning: Incomplete plot config at index {i}, skipping")
            continue

        trans_data = {
            y_metric: _resolve_trans_metric(y_metric, baseline_data, finetuned_data)
        }

        # Generate filename
        filename = f"scatter_4d_{x_metric}_{y_metric}_{size_metric}_{color_metric}.png"
        output_path = output_dir / filename

        print(f"\nGenerating plot {i+1}: {filename}")
        scatter_4d(
            sim_data=sim_data,
            trans_data=trans_data,
            x_metric=x_metric,
            y_metric=y_metric,
            size_metric=size_metric,
            color_metric=color_metric,
            output_path=output_path,
        )


def heatmap(
    sim_data: Dict[str, Dict[str, float]],
    baseline_data: Dict[str, Dict[str, float]],
    finetuned_data: Dict[str, Dict[str, float]],
    sim_metrics: List[str],
    trans_metrics: List[str],
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    """
    Create a heatmap with language pairs as rows and metrics as columns.

    Each column is normalized to [0, 1] for comparable coloring.
    """
    # Collect all metrics data
    all_metrics = {}
    for m in sim_metrics:
        all_metrics[m] = sim_data.get(m, {})
    for m in trans_metrics:
        all_metrics[m] = _resolve_trans_metric(m, baseline_data, finetuned_data)

    # Find common pairs across all metrics
    all_pairs_sets = [set(v.keys()) for v in all_metrics.values() if v]
    if not all_pairs_sets:
        print("Warning: No data available for heatmap. Skipping.")
        return
    common_pairs = sorted(set.intersection(*all_pairs_sets))

    if len(common_pairs) < 2:
        print(
            f"Warning: Not enough common pairs for heatmap ({len(common_pairs)}). Skipping."
        )
        return

    # Build matrix
    metric_names = sim_metrics + trans_metrics
    matrix = np.zeros((len(common_pairs), len(metric_names)))

    for j, metric in enumerate(metric_names):
        data = all_metrics.get(metric, {})
        for i, pair in enumerate(common_pairs):
            matrix[i, j] = data.get(pair, np.nan)

    # Normalize each column to [0, 1]
    matrix_normalized = np.zeros_like(matrix)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        col_min, col_max = np.nanmin(col), np.nanmax(col)
        if col_max > col_min:
            matrix_normalized[:, j] = (col - col_min) / (col_max - col_min)
        else:
            matrix_normalized[:, j] = 0.5

    # Create heatmap
    fig, ax = plt.subplots(
        figsize=(max(12, len(metric_names) * 0.8), max(8, len(common_pairs) * 0.4))
    )

    sns.heatmap(
        matrix_normalized,
        ax=ax,
        xticklabels=metric_names,
        yticklabels=common_pairs,
        cmap="YlGnBu",
        annot=matrix,
        fmt=".2f",
        annot_kws={"size": 8},
        cbar_kws={"label": "Normalized (per column)"},
    )

    ax.set_xlabel("Metrics", fontsize=11)
    ax.set_ylabel("Language Pairs", fontsize=11)
    ax.set_title(title or "Metrics Heatmap (normalized per column)", fontsize=12)

    # Rotate x labels
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def generate_all_heatmaps(
    sim_data: Dict[str, Dict[str, float]],
    baseline_data: Dict[str, Dict[str, float]],
    finetuned_data: Dict[str, Dict[str, float]],
    heatmap_configs: List[Dict],
    output_dir: Path,
) -> None:
    """Generate all heatmaps from config."""
    for i, cfg in enumerate(heatmap_configs):
        sim_metrics = cfg.get("sim_metrics", [])
        trans_metrics = cfg.get("trans_metrics", [])
        name = cfg.get("name", f"heatmap_{i+1}")

        if not sim_metrics and not trans_metrics:
            print(f"Warning: Empty heatmap config at index {i}, skipping")
            continue

        output_path = output_dir / f"{name}.png"

        print(f"\nGenerating heatmap {i+1}: {name}")
        heatmap(
            sim_data=sim_data,
            baseline_data=baseline_data,
            finetuned_data=finetuned_data,
            sim_metrics=sim_metrics,
            trans_metrics=trans_metrics,
            output_path=output_path,
            title=cfg.get("title"),
        )
