"""Pooling and similarity functions for embeddings."""

import torch
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
