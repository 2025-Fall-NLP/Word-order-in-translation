"""OPUS dataset loader. Uses English pivot for non-English pairs."""

import random
from typing import Any, Dict, Optional

from datasets import load_dataset

from .base import BaseDatasetLoader, ParallelSentences
from .registry import register_dataset

_cache: Dict[str, object] = {}


@register_dataset("opus")
class OpusLoader(BaseDatasetLoader):
    """OPUS-100 parallel corpus loader. Uses standard 2-letter lang codes."""

    DEFAULT_NAME = "Helsinki-NLP/opus-100"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.dataset_name = config.get("name", self.DEFAULT_NAME)
        self.seed = config.get("seed", 42)

    def _get_dataset(self, pair_name: str, split: str):
        cache_key = f"{self.dataset_name}_{pair_name}_{split}"
        if cache_key not in _cache:
            _cache[cache_key] = load_dataset(self.dataset_name, pair_name, split=split)
        return _cache[cache_key]

    def load(
        self,
        src_lang: str,
        tgt_lang: str,
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> ParallelSentences:
        random.seed(self.seed)
        if src_lang == "en" or tgt_lang == "en":
            return self._load_english_pair(src_lang, tgt_lang, split, max_samples)
        return self._load_non_english_pair(src_lang, tgt_lang, split, max_samples)

    def _load_english_pair(
        self, src_lang: str, tgt_lang: str, split: str, max_samples: Optional[int]
    ) -> ParallelSentences:
        other = tgt_lang if src_lang == "en" else src_lang
        pair_name = f"{other}-en" if other < "en" else f"en-{other}"

        try:
            dataset = self._get_dataset(pair_name, split)
        except Exception as e:
            raise ValueError(f"Failed to load {self.dataset_name} {pair_name}: {e}")

        src_sentences, tgt_sentences = [], []
        for ex in dataset:
            t = ex["translation"]
            if src_lang == "en":
                src_sentences.append(t["en"])
                tgt_sentences.append(t[other])
            else:
                src_sentences.append(t[other])
                tgt_sentences.append(t["en"])

        if max_samples and len(src_sentences) > max_samples:
            indices = random.sample(range(len(src_sentences)), max_samples)
            src_sentences = [src_sentences[i] for i in indices]
            tgt_sentences = [tgt_sentences[i] for i in indices]

        return ParallelSentences(src_lang, tgt_lang, src_sentences, tgt_sentences)

    def _load_non_english_pair(
        self, src_lang: str, tgt_lang: str, split: str, max_samples: Optional[int]
    ) -> ParallelSentences:
        """Load via English pivot matching."""
        en_src = self._load_english_pair("en", src_lang, split, None)
        en_tgt = self._load_english_pair("en", tgt_lang, split, None)

        en_to_src = dict(zip(en_src.src_sentences, en_src.tgt_sentences))

        src_sentences, tgt_sentences = [], []
        for en_sent, tgt_sent in zip(en_tgt.src_sentences, en_tgt.tgt_sentences):
            if en_sent in en_to_src:
                src_sentences.append(en_to_src[en_sent])
                tgt_sentences.append(tgt_sent)

        if not src_sentences:
            raise ValueError(f"No matching sentences for {src_lang}-{tgt_lang}")

        if max_samples and len(src_sentences) > max_samples:
            indices = random.sample(range(len(src_sentences)), max_samples)
            src_sentences = [src_sentences[i] for i in indices]
            tgt_sentences = [tgt_sentences[i] for i in indices]

        return ParallelSentences(src_lang, tgt_lang, src_sentences, tgt_sentences)
