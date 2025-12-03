"""FLORES dataset loader. Requires datasets<3.0.0."""

from typing import Any, Dict, Optional

from datasets import load_dataset

from .base import BaseDatasetLoader, ParallelSentences
from .registry import register_dataset

_cache: Dict[str, object] = {}


@register_dataset("flores")
class FloresLoader(BaseDatasetLoader):
    """FLORES-200 parallel corpus loader."""

    DEFAULT_NAME = "facebook/flores"
    DEFAULT_LANG_CODES = {
        "en": "eng_Latn",
        "ko": "kor_Hang",
        "ja": "jpn_Jpan",
        "zh": "zho_Hans",
        "de": "deu_Latn",
        "fr": "fra_Latn",
        "es": "spa_Latn",
        "ar": "arb_Arab",
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.dataset_name = config.get("name", self.DEFAULT_NAME)
        self.lang_codes = config.get("lang_codes", self.DEFAULT_LANG_CODES)

    def _get_dataset(self, split: str):
        cache_key = f"{self.dataset_name}_{split}"
        if cache_key not in _cache:
            _cache[cache_key] = load_dataset(
                self.dataset_name, "all", split=split, trust_remote_code=True
            )
        return _cache[cache_key]

    def load(
        self,
        src_lang: str,
        tgt_lang: str,
        split: str = "dev",
        max_samples: Optional[int] = None,
    ) -> ParallelSentences:
        if src_lang not in self.lang_codes:
            raise ValueError(f"Unknown language: {src_lang}")
        if tgt_lang not in self.lang_codes:
            raise ValueError(f"Unknown language: {tgt_lang}")

        src_col = f"sentence_{self.lang_codes[src_lang]}"
        tgt_col = f"sentence_{self.lang_codes[tgt_lang]}"

        dataset = self._get_dataset(split)
        src_sentences = [ex[src_col] for ex in dataset]
        tgt_sentences = [ex[tgt_col] for ex in dataset]

        if max_samples and len(src_sentences) > max_samples:
            src_sentences = src_sentences[:max_samples]
            tgt_sentences = tgt_sentences[:max_samples]

        return ParallelSentences(src_lang, tgt_lang, src_sentences, tgt_sentences)
