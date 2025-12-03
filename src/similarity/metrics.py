"""Concrete similarity metric implementations."""

from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from .base import BaseSimilarityMetric
from .registry import register_similarity


@register_similarity("mdeberta")
class MDeBERTaSimilarity(BaseSimilarityMetric):
    """
    mDeBERTa-v3 multilingual embeddings.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        model_name = config.get("model")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = config.get("max_length", 512)

    def compute_token_embeddings(self, sentences: List[str]) -> Tuple[Tensor, Tensor]:
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs.last_hidden_state, inputs["attention_mask"]
