"""Base class for similarity metrics using template method pattern."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor

from src.data.base import ParallelSentences

from .functions import POOLING_FNS, SIMILARITY_FNS


class BaseSimilarityMetric(ABC):
    """
    Base class for embedding-based similarity metrics.
    Subclasses implement compute_token_embeddings().
    Pooling and similarity functions are configured via config dict.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        pooling_name = config.get("pooling", None)
        if pooling_name not in POOLING_FNS:
            raise ValueError(f"Unknown pooling: {pooling_name}")
        self.pooling_fn = POOLING_FNS[pooling_name]

        similarity_name = config.get("similarity_fn", "cosine")
        if similarity_name not in SIMILARITY_FNS:
            raise ValueError(f"Unknown similarity: {similarity_name}")
        self.similarity_fn = SIMILARITY_FNS[similarity_name]

        self.batch_size = config.get("batch_size", 32)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def compute_token_embeddings(self, sentences: List[str]) -> Tuple[Tensor, Tensor]:
        """Return (token_embeddings, attention_mask)."""
        pass

    def compute_sentence_embeddings(self, sentences: List[str]) -> Tensor:
        """Compute sentence embeddings with batching."""
        all_emb = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i : i + self.batch_size]
            with torch.no_grad():
                token_emb, mask = self.compute_token_embeddings(batch)
                all_emb.append(self.pooling_fn(token_emb, mask).cpu())
        return torch.cat(all_emb, dim=0)

    def compute_for_pair(
        self, data: ParallelSentences, show_progress: bool = False
    ) -> float:
        """Compute similarity score for a language pair."""
        src_emb = self.compute_sentence_embeddings(data.src_sentences)
        tgt_emb = self.compute_sentence_embeddings(data.tgt_sentences)
        return self.similarity_fn(src_emb, tgt_emb)

    def get_output_filename(self) -> str:
        return f"{self.config.get('type', 'unknown')}_{self.config.get('pooling', 'none')}.json"

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "type": self.config.get("type"),
            "model": self.config.get("model"),
            "pooling": self.config.get("pooling", "none"),
            "similarity_fn": self.config.get("similarity_fn", "cosine"),
        }
