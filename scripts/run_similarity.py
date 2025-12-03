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
from src.utils import get_pair_key, is_computed, load_results, save_results


def main():
    parser = argparse.ArgumentParser(description="Compute language similarity")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    language_pairs = config["language_pairs"]
    sim_config = config["similarity"]
    data_config = config["datasets"]["similarity"]
    output_dir = Path(config["paths"]["output_dir"]) / "similarity"

    # Initialize dataset loader from registry
    data_loader = get_dataset(data_config["type"])(data_config)

    for method in sim_config["methods"]:
        method_type, pooling = method["type"], method.get("pooling", "mean")
        output_path = output_dir / f"{method_type}_{pooling}.json"

        print(f"\n{'='*60}\n{method_type} (pooling={pooling})\n{'='*60}")

        existing = load_results(output_path)
        pairs_to_compute = []

        for src, tgt in language_pairs:
            key = get_pair_key(src, tgt)
            if is_computed(existing, src, tgt):
                print(f"  [SKIP] {key}: {existing['results'][key]:.4f}")
            else:
                pairs_to_compute.append((src, tgt))
                print(f"  [TODO] {key}")

        if args.dry_run:
            print(f"\nDry run: {len(pairs_to_compute)} pairs")
            continue
        if not pairs_to_compute:
            continue

        metric = get_similarity(method_type)(method)
        results = dict(existing.get("results", {}))

        for src, tgt in pairs_to_compute:
            key = get_pair_key(src, tgt)
            print(f"\n{key}...")
            data = data_loader.load(src, tgt, data_config["split"])
            score = metric.compute_for_pair(data)
            results[key] = score
            print(f"  {score:.4f}")
            save_results(output_path, metric.get_metadata(), results)

    print("\nDone!")


if __name__ == "__main__":
    main()
