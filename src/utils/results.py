"""Utilities for saving and loading experiment results in JSON format."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union


def get_pair_key(src: str, tgt: str) -> str:
    return f"{src}-{tgt}"


def load_results(path: Union[str, Path]) -> Dict[str, Any]:
    """Load results or return empty structure."""
    path = Path(path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"metadata": {}, "results": {}}


def save_results(path: Union[str, Path], metadata: Dict[str, Any], results: Dict[str, Any]) -> None:
    """Save results with metadata and timestamp."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"metadata": {**metadata, "computed_at": datetime.now().isoformat()}, "results": results}, f, indent=2)


def is_computed(results: Dict[str, Any], src: str, tgt: str) -> bool:
    return get_pair_key(src, tgt) in results.get("results", {})
