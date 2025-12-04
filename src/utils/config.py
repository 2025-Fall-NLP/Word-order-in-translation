"""Centralized config access."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


class Config:
    """Wrapper for config.yaml with convenient accessors."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            return cls(yaml.safe_load(f))

    # Core
    @property
    def language_pairs(self) -> List[Tuple[str, str]]:
        return [tuple(p) for p in self._data["language_pairs"]]

    # Paths
    @property
    def output_dir(self) -> Path:
        return Path(self._data["paths"]["output_dir"])

    @property
    def checkpoint_dir(self) -> str:
        return self._data["paths"]["checkpoint_dir"]

    # Datasets
    @property
    def similarity_dataset(self) -> Dict[str, Any]:
        return self._data["datasets"]["similarity"]

    @property
    def evaluation_dataset(self) -> Dict[str, Any]:
        return self._data["datasets"]["evaluation"]

    @property
    def training_dataset(self) -> Dict[str, Any]:
        return self._data["datasets"]["training"]

    # Similarity
    @property
    def similarity_metrics(self) -> List[Dict[str, Any]]:
        return self._data["similarity"]

    # Translation
    @property
    def trans_model(self) -> Dict[str, Any]:
        return self._data["translation"]["model"]

    @property
    def eval_metrics(self) -> List[Dict[str, Any]]:
        return self._data["translation"]["evaluation"]

    @property
    def training(self) -> Dict[str, Any]:
        return self._data["translation"]["training"]

    # Raw access
    def __getitem__(self, key: str) -> Any:
        return self._data[key]
