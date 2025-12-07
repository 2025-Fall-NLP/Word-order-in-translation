"""Concrete similarity metric implementations."""

from typing import Any, Dict, List, Tuple

import numpy as np
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from src.data.base import ParallelSentences

from .base import BaseSimilarityMetric
from .registry import register_similarity_metric


@register_similarity_metric("mdeberta")
class MDeBERTaSimilarity(BaseSimilarityMetric):
    """mDeBERTa-v3 multilingual embeddings."""

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


@register_similarity_metric("labse")
class LaBSESimilarity(BaseSimilarityMetric):
    """LaBSE outputs sentence embeddings directly, so we override compute_sentence_embeddings."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        from sentence_transformers import SentenceTransformer

        model_name = config.get("model")
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        self.model.eval()

    def compute_token_embeddings(self, sentences: List[str]) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError("LaBSE uses compute_sentence_embeddings directly")

    def compute_sentence_embeddings(self, sentences: List[str]) -> Tensor:
        """Override to use sentence-transformers' built-in encoding."""
        return self.model.encode(
            sentences, batch_size=self.batch_size, convert_to_tensor=True
        ).cpu()


@register_similarity_metric("script")
class ScriptSimilarity(BaseSimilarityMetric):
    """Script similarity based on character set overlap (Jaccard similarity)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def compute_token_embeddings(self, sentences: List[str]) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError("ScriptSimilarity uses compute_for_pair directly")

    def compute_for_pair(self, data: ParallelSentences) -> float:
        """Compute Jaccard similarity of character sets."""
        src_chars = set("".join(data.src_sentences))
        tgt_chars = set("".join(data.tgt_sentences))
        intersection = len(src_chars & tgt_chars)
        union = len(src_chars | tgt_chars)
        return intersection / union if union > 0 else 0.0


@register_similarity_metric("uriel")
class URIELSimilarity(BaseSimilarityMetric):
    """URIEL typological similarity using lang2vec feature vectors."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.features = config.get("features", "syntax_knn+phonology_knn+inventory_knn")

    def compute_token_embeddings(self, sentences: List[str]) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError("URIELSimilarity uses compute_for_pair directly")

    def compute_for_pair(self, data: ParallelSentences) -> float:
        """Compute cosine similarity of typological feature vectors."""
        import lang2vec.lang2vec as l2v

        features = l2v.get_features([data.src_lang, data.tgt_lang], self.features)
        src_vec = np.array(features[data.src_lang])
        tgt_vec = np.array(features[data.tgt_lang])

        # Handle missing values (NaN) by using only shared valid features
        valid_mask = ~(np.isnan(src_vec) | np.isnan(tgt_vec))
        if not valid_mask.any():
            return 0.0

        src_valid = src_vec[valid_mask]
        tgt_valid = tgt_vec[valid_mask]

        # Cosine similarity
        dot = np.dot(src_valid, tgt_valid)
        norm = np.linalg.norm(src_valid) * np.linalg.norm(tgt_valid)
        return float(dot / norm) if norm > 0 else 0.0
