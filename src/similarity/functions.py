"""Pooling and similarity functions for embeddings."""

import torch
import torch.nn.functional as F
from torch import Tensor


def mean_pooling(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
    """Mean pooling weighted by attention mask."""
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(
        mask.sum(dim=1), min=1e-9
    )


def cls_pooling(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
    """Use first token ([CLS]) embedding."""
    return token_embeddings[:, 0, :]


def max_pooling(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
    """Max pooling with padding masked out."""
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    masked = token_embeddings.masked_fill(mask == 0, -1e9)
    return torch.max(masked, dim=1).values


def no_pooling(embeddings: Tensor, attention_mask: Tensor) -> Tensor:
    """Identity for models with built-in pooling."""
    return embeddings


POOLING_FNS = {
    "mean": mean_pooling,
    "cls": cls_pooling,
    "max": max_pooling,
    None: no_pooling,
}


def cosine_similarity_fn(emb1: Tensor, emb2: Tensor) -> float:
    """Mean cosine similarity between aligned pairs."""
    emb1_norm = F.normalize(emb1, p=2, dim=1)
    emb2_norm = F.normalize(emb2, p=2, dim=1)
    return float(torch.sum(emb1_norm * emb2_norm, dim=1).mean().item())


def euclidean_similarity_fn(emb1: Tensor, emb2: Tensor) -> float:
    """Mean negative euclidean distance (higher = more similar)."""
    return float(-torch.norm(emb1 - emb2, p=2, dim=1).mean().item())


SIMILARITY_FNS = {"cosine": cosine_similarity_fn, "euclidean": euclidean_similarity_fn}
