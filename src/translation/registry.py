"""
Registry for translation models and evaluation metrics.
"""

from src.utils.registry import create_registry

from .base import BaseEvalMetric, BaseTranslator

# Registry for translation models (mbart, nllb, m2m100, etc.)
register_model, get_model, list_models = create_registry(
    "translation_model", BaseTranslator
)

# Registry for evaluation metrics (bleu, comet, etc.)
register_eval_metric, get_eval_metric, list_eval_metrics = create_registry(
    "eval_metric", BaseEvalMetric
)
