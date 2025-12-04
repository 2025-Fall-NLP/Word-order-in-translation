"""Utilities for saving and loading experiment results in JSON format."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union


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
