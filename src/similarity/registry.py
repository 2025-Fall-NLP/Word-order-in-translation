"""
Registry for similarity metrics.

Usage:
    from src.similarity.registry import register_similarity, get_similarity

    @register_similarity("mdeberta")
    class MDeBERTaSimilarity(BaseSimilarityMetric):
        ...

    # Later
    MetricClass = get_similarity("mdeberta")
    metric = MetricClass(config)
"""

from src.utils.registry import create_registry

register_similarity, get_similarity, list_similarity_metrics = create_registry(
    "similarity"
)
