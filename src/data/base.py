"""Base classes for data loading."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ParallelSentences:
    """Container for parallel sentences in a language pair."""

    src_lang: str
    tgt_lang: str
    src_sentences: List[str]
    tgt_sentences: List[str]

    def __post_init__(self):
        if len(self.src_sentences) != len(self.tgt_sentences):
            raise ValueError(
                f"Mismatched lengths: {len(self.src_sentences)} src, {len(self.tgt_sentences)} tgt"
            )

    def __len__(self) -> int:
        return len(self.src_sentences)

    def __getitem__(self, idx: int) -> tuple:
        return (self.src_sentences[idx], self.tgt_sentences[idx])


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_name = config.get("name")

    @abstractmethod
    def load(
        self,
        src_lang: str,
        tgt_lang: str,
        split: str,
        max_samples: Optional[int] = None,
    ) -> ParallelSentences:
        """Load parallel sentences for a language pair."""
        pass
