#!/usr/bin/env python
"""Analyze correlation between language similarity and translation quality."""

import argparse
import math
import sys
from pathlib import Path
from typing import List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis import (
    CorrelationResult,
    analyze_all_correlations,
    apply_fdr_correction,
)
from src.utils import (
    OUTPUT_DIR_ANALYSIS,
    OUTPUT_FILE_CORRELATION,
    Config,
    load_metrics_results,
    save_results,
)


def print_summary(results: List[CorrelationResult]) -> None:
    """Print human-readable summary of correlation results."""
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS (with FDR correction)")
    print("Primary: Spearman ρ (robust to outliers) | Secondary: Pearson r")
    print("=" * 80)

    stages = ["baseline", "finetuned", "delta", "delta_pct"]
    stage_labels = {
        "baseline": "BASELINE (sim vs base quality)",
        "finetuned": "FINETUNED (sim vs fine quality)",
        "delta": "DELTA (sim vs absolute improvement)",
        "delta_pct": "DELTA % (sim vs relative improvement)",
    }

    def sig_marker(p_raw: float, p_adj: float | None) -> str:
        """Return significance marker based on FDR-adjusted p-values."""
        if math.isnan(p_raw):
            return ""
        # Use adjusted p-value if available, otherwise raw
        p = p_adj if p_adj is not None else p_raw
        if p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        return ""

    def effect_label(r: float) -> str:
        """Return effect size label (Cohen's conventions for correlation)."""
        if math.isnan(r):
            return "-"
        abs_r = abs(r)
        if abs_r >= 0.5:
            return "L"  # Large
        elif abs_r >= 0.3:
            return "M"  # Medium
        elif abs_r >= 0.1:
            return "S"  # Small
        return "-"  # Negligible

    for stage in stages:
        stage_results = [r for r in results if r.stage == stage]
        if not stage_results:
            continue

        print(f"\n{stage_labels.get(stage, stage.upper())}:")
        for r in stage_results:
            # Show Spearman first (primary), then Pearson (secondary)
            sig_s = sig_marker(r.spearman_p, r.spearman_p_adj)
            sig_p = sig_marker(r.pearson_p, r.pearson_p_adj)
            eff_s = effect_label(r.spearman_r)
            eff_p = effect_label(r.pearson_r)
            print(
                f"  {r.similarity_metric} vs {r.translation_metric}: "
                f"ρ={r.spearman_r:+.3f}{sig_s} [{eff_s}], r={r.pearson_r:+.3f}{sig_p} [{eff_p}]"
            )

    print("\n" + "-" * 80)
    print("* p_adj<0.05, ** p_adj<0.01 (Benjamini-Hochberg FDR-corrected)")
    print(
        "Effect size: [L]=large |ρ|≥0.5, [M]=medium |ρ|≥0.3, [S]=small |ρ|≥0.1, [-]=negligible"
    )
    print(
        "Note: Large effects may be meaningful even without statistical significance (n=12)."
    )


def main():
    parser = argparse.ArgumentParser(description="Correlation analysis")
    parser.add_argument("--config", required=True, help="Config file path")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    output_path = cfg.output_dir / OUTPUT_DIR_ANALYSIS / OUTPUT_FILE_CORRELATION

    print(f"\nLoading results from: {cfg.output_dir}")

    try:
        sim_results, baseline, finetuned = load_metrics_results(str(cfg.output_dir))
    except Exception as e:
        print(f"Error loading results: {e}")
        print("Run run_similarity.py and run_translation.py first.")
        return 1

    if not sim_results or not baseline:
        print("Missing similarity or baseline translation results.")
        return 1

    results = analyze_all_correlations(sim_results, baseline, finetuned)

    if not results:
        print(
            "No correlations computed. Need similarity + translation results for >=3 pairs."
        )
        return 1

    # Apply FDR correction for multiple comparisons
    results = apply_fdr_correction(results)

    print_summary(results)

    # Count significant results (using adjusted p-values)
    def count_significant(results, get_p_adj):
        return sum(
            1
            for r in results
            if (p := get_p_adj(r)) is not None and not math.isnan(p) and p < 0.05
        )

    sig_pearson_adj = count_significant(results, lambda r: r.pearson_p_adj)
    sig_spearman_adj = count_significant(results, lambda r: r.spearman_p_adj)
    sig_pearson_raw = sum(
        1 for r in results if not math.isnan(r.pearson_p) and r.pearson_p < 0.05
    )
    sig_spearman_raw = sum(
        1 for r in results if not math.isnan(r.spearman_p) and r.spearman_p < 0.05
    )

    # Save results
    metadata = {
        "type": "correlation_analysis",
        "n_correlations": len(results),
        "significant_spearman_raw": sig_spearman_raw,
        "significant_spearman_fdr": sig_spearman_adj,
        "significant_pearson_raw": sig_pearson_raw,
        "significant_pearson_fdr": sig_pearson_adj,
    }
    save_results(output_path, metadata, [r.to_dict() for r in results])
    print(f"\nSaved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
