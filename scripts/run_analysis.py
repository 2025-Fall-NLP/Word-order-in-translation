#!/usr/bin/env python
"""Analyze correlation between language similarity and translation quality."""

import argparse
import sys
from pathlib import Path
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis import analyze_all_correlations, print_correlation_summary, save_correlation_results


def main():
    parser = argparse.ArgumentParser(description="Correlation analysis")
    parser.add_argument("--config", required=True, help="Config file path")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = config["paths"]["output_dir"]
    analysis_output = Path(output_dir) / "analysis" / "correlation.json"

    print(f"\nLoading results from: {output_dir}")

    try:
        results = analyze_all_correlations(output_dir)
    except Exception as e:
        print(f"Error: {e}\nRun run_similarity.py and run_translation.py first.")
        return 1

    if not results:
        print("No correlations computed. Need similarity + translation results for >=3 pairs.")
        return 1

    print_correlation_summary(results)
    save_correlation_results(results, analysis_output)
    print(f"\nSaved to: {analysis_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
