#!/usr/bin/env python
"""Analyze correlation between language similarity and translation quality."""

import argparse
import math
import sys
from pathlib import Path
from typing import List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis import CorrelationResult, analyze_all_correlations
from src.utils import (
    OUTPUT_DIR_ANALYSIS,
    OUTPUT_FILE_CORRELATION,
    Config,
    load_metrics_results,
    save_results,
)


def print_summary(results: List[CorrelationResult]) -> None:
    """Print human-readable summary of correlation results."""
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    stages = ["baseline", "finetuned", "delta", "delta_pct", "partial"]
    stage_labels = {
        "baseline": "BASELINE (sim vs base quality)",
        "finetuned": "FINETUNED (sim vs fine quality)",
        "delta": "DELTA (sim vs absolute improvement)",
        "delta_pct": "DELTA % (sim vs relative improvement)",
        "partial": "PARTIAL (sim vs fine, controlling for base)",
    }

    for stage in stages:
        stage_results = [r for r in results if r.stage == stage]
        if not stage_results:
            continue

        print(f"\n{stage_labels.get(stage, stage.upper())}:")
        for r in stage_results:
            sig_p = "**" if r.pearson_p < 0.01 else "*" if r.pearson_p < 0.05 else ""
            if stage == "partial":
                print(
                    f"  {r.similarity_metric} vs {r.translation_metric}: r={r.pearson_r:+.3f}{sig_p}"
                )
            else:
                sig_s = (
                    "**" if r.spearman_p < 0.01 else "*" if r.spearman_p < 0.05 else ""
                )
                print(
                    f"  {r.similarity_metric} vs {r.translation_metric}: "
                    f"r={r.pearson_r:+.3f}{sig_p}, ρ={r.spearman_r:+.3f}{sig_s}"
                )

    print("\n* p<0.05, ** p<0.01")
    print(
        "Note: Use delta_pct and partial for fairer analysis (controls for baseline bias)"
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

    print_summary(results)

    # Save results
    metadata = {
        "type": "correlation_analysis",
        "n_correlations": len(results),
        "significant_pearson": sum(
            1 for r in results if not math.isnan(r.pearson_p) and r.pearson_p < 0.05
        ),
        "significant_spearman": sum(
            1 for r in results if not math.isnan(r.spearman_p) and r.spearman_p < 0.05
        ),
        "note": "mBART-50 pretraining used imbalanced data; use delta_pct and partial for fairer analysis",
    }
    save_results(output_path, metadata, [r.to_dict() for r in results])
    print(f"\nSaved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
