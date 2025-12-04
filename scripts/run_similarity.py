#!/usr/bin/env python
"""Compute language similarity for configured language pairs."""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import get_dataset
from src.similarity import get_similarity
from src.utils import Config, get_pair_key, save_results


def main():
    parser = argparse.ArgumentParser(description="Compute language similarity")
    parser.add_argument("--config", required=True, help="Config file path")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    output_dir = cfg.output_dir / "similarity"

    data_loader = get_dataset(cfg.similarity_dataset["type"])(cfg.similarity_dataset)

    for method in cfg.similarity_methods:
        metric = get_similarity(method["type"])(method)
        output_path = output_dir / metric.get_output_filename()

        print(f"\n{'='*60}\n{method['type']}\n{'='*60}")

        results = {}
        for src, tgt in cfg.language_pairs:
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
