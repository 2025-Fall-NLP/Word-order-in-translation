#!/usr/bin/env python
"""Compute language similarity for configured language pairs."""

import argparse
import sys
from pathlib import Path

import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import get_dataset
from src.similarity import get_similarity
from src.utils import get_pair_key, save_results


def main():
    parser = argparse.ArgumentParser(description="Compute language similarity")
    parser.add_argument("--config", required=True, help="Config file path")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    language_pairs = config["language_pairs"]
    sim_cfg = config["similarity"]
    data_cfg = config["datasets"]["similarity"]
    output_dir = Path(config["paths"]["output_dir"]) / "similarity"

    # Initialize dataset loader from registry
    data_loader = get_dataset(data_cfg["type"])(data_cfg)

    for method_cfg in sim_cfg["methods"]:
        method_type = method_cfg["type"]
        pooling = method_cfg.get("pooling", "mean")
        output_path = output_dir / f"{method_type}_{pooling}.json"

        print(f"\n{'='*60}\n{method_type} (pooling={pooling})\n{'='*60}")

        metric = get_similarity(method_type)(method_cfg)
        results = {}

        for src, tgt in language_pairs:
            key = get_pair_key(src, tgt)
            print(f"\n{key}...")
            data = data_loader.load(src, tgt)
            score = metric.compute_for_pair(data)
            results[key] = score
            print(f"  {score:.4f}")
            save_results(output_path, metric.get_metadata(), results)

    print("\nDone!")


if __name__ == "__main__":
    main()
