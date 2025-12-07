"""Base class for similarity metrics using template method pattern."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from src.data.base import ParallelSentences

from .pooling import POOLING_FNS


class BaseSimilarityMetric(ABC):
    """
    Base class for embedding-based similarity metrics.
    Subclasses implements either compute_for_pair, compute_sentence_embeddings or compute_token_embeddings.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        pooling_name = config.get("pooling", None)
        if pooling_name not in POOLING_FNS:
            raise ValueError(f"Unknown pooling: {pooling_name}")
        self.pooling_fn = POOLING_FNS[pooling_name]
        self.batch_size = config.get("batch_size", 64)
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

    def compute_for_pair(self, data: ParallelSentences) -> float:
        """Compute similarity score for a language pair."""
        src_emb = self.compute_sentence_embeddings(data.src_sentences)
        tgt_emb = self.compute_sentence_embeddings(data.tgt_sentences)
        return self._cosine_similarity(src_emb, tgt_emb)

    @staticmethod
    def _cosine_similarity(emb1: Tensor, emb2: Tensor) -> float:
        """Mean cosine similarity between two tensors."""
        emb1_norm = F.normalize(emb1, p=2, dim=1)
        emb2_norm = F.normalize(emb2, p=2, dim=1)
        return float(torch.sum(emb1_norm * emb2_norm, dim=1).mean().item())

    def get_output_filename(self) -> str:
        return f"{self.config.get('type', 'unknown')}_{self.config.get('pooling', 'none')}.json"

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "type": self.config.get("type"),
        }
