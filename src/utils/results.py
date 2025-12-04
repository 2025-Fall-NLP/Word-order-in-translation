"""Utilities for saving and loading experiment results in JSON format."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Union

from .consts import (
    OUTPUT_DIR_SIMILARITY,
    OUTPUT_DIR_TRANSLATION,
    OUTPUT_PREFIX_BASELINE,
    OUTPUT_PREFIX_FINETUNED,
)


def get_pair_key(src: str, tgt: str) -> str:
    return f"{src}-{tgt}"


def save_results(
    path: Union[str, Path], metadata: Dict[str, Any], results: Dict[str, Any]
) -> None:
    """Save results with metadata and timestamp."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {**metadata, "computed_at": datetime.now().isoformat()},
                "results": results,
            },
            f,
            indent=2,
        )


def load_metrics_results(
    output_dir: Union[str, Path],
) -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, Dict]]:
    """
    Load similarity, baseline, and finetuned results from output directory.

    Returns:
        Tuple of (similarity_results, baseline_results, finetuned_results)
        Each is a dict mapping metric_name -> {pair_key: score}
    """
    p = Path(output_dir)

    def load_dir(subdir: str, prefix: str = "") -> Dict[str, Dict[str, float]]:
        results = {}
        d = p / subdir
        if d.exists():
            for f in d.glob(f"{prefix}*.json"):
                with open(f) as fp:
                    name = f.stem.replace(prefix, "") if prefix else f.stem
                    results[name] = json.load(fp).get("results", {})
        return results

    return (
        load_dir(OUTPUT_DIR_SIMILARITY),
        load_dir(OUTPUT_DIR_TRANSLATION, OUTPUT_PREFIX_BASELINE),
        load_dir(OUTPUT_DIR_TRANSLATION, OUTPUT_PREFIX_FINETUNED),
    )
