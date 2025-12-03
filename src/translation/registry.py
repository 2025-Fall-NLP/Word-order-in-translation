"""
Registry for translation models and evaluation metrics.

Usage:
    from src.translation.registry import register_model, get_model

    @register_model("mbart")
    class MBARTTranslator(BaseTranslator):
        ...

    # Later
    ModelClass = get_model("mbart")
    model = ModelClass(config, lang_codes)
"""

from src.utils.registry import create_registry

# Registry for translation models (mbart, nllb, m2m100, etc.)
register_model, get_model, list_models = create_registry("translation_model")

# Registry for evaluation metrics (bleu, comet, chrf, etc.)
register_eval, get_eval, list_eval_metrics = create_registry("eval_metric")
