"""
Registry for similarity metrics.
"""

from src.utils.registry import create_registry

from .base import BaseSimilarityMetric

register_similarity_metric, get_similarity_metric, list_similarity_metrics = (
    create_registry("similarity_metric", BaseSimilarityMetric)
)
