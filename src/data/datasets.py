"""Requires datasets<3.0.0."""

import random
from typing import Any, Dict

from datasets import load_dataset

from .base import BaseDatasetLoader, ParallelSentences
from .registry import register_dataset

_cache: Dict[str, object] = {}


@register_dataset("flores")
class FloresLoader(BaseDatasetLoader):
    """FLORES-200 parallel corpus loader."""

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
        self.lang_codes = config.get("lang_codes", self.DEFAULT_LANG_CODES)

    def _get_dataset(self):
        cache_key = f"{self.dataset_name}_{self.split}"
        if cache_key not in _cache:
            _cache[cache_key] = load_dataset(
                self.dataset_name, "all", split=self.split, trust_remote_code=True
            )
        return _cache[cache_key]

    def load(self, src_lang: str, tgt_lang: str) -> ParallelSentences:
        if src_lang not in self.lang_codes:
            raise ValueError(f"Unknown language: {src_lang}")
        if tgt_lang not in self.lang_codes:
            raise ValueError(f"Unknown language: {tgt_lang}")

        src_col = f"sentence_{self.lang_codes[src_lang]}"
        tgt_col = f"sentence_{self.lang_codes[tgt_lang]}"

        dataset = self._get_dataset()
        src_sentences = [ex[src_col] for ex in dataset]
        tgt_sentences = [ex[tgt_col] for ex in dataset]

        if self.max_samples and len(src_sentences) > self.max_samples:
            src_sentences = src_sentences[: self.max_samples]
            tgt_sentences = tgt_sentences[: self.max_samples]

        return ParallelSentences(src_lang, tgt_lang, src_sentences, tgt_sentences)


@register_dataset("opus")
class OpusLoader(BaseDatasetLoader):
    """OPUS-100 parallel corpus loader. Uses standard 2-letter lang codes."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.seed = config.get("seed", 42)

    def _get_dataset(self, pair_name: str):
        cache_key = f"{self.dataset_name}_{pair_name}_{self.split}"
        if cache_key not in _cache:
            _cache[cache_key] = load_dataset(
                self.dataset_name, pair_name, split=self.split
            )
        return _cache[cache_key]

    def load(self, src_lang: str, tgt_lang: str) -> ParallelSentences:
        random.seed(self.seed)
        if src_lang == "en" or tgt_lang == "en":
            return self._load_english_pair(src_lang, tgt_lang)
        return self._load_non_english_pair(src_lang, tgt_lang)

    def _load_english_pair(self, src_lang: str, tgt_lang: str) -> ParallelSentences:
        other = tgt_lang if src_lang == "en" else src_lang
        pair_name = f"{other}-en" if other < "en" else f"en-{other}"

        try:
            dataset = self._get_dataset(pair_name)
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

        if self.max_samples and len(src_sentences) > self.max_samples:
            indices = random.sample(range(len(src_sentences)), self.max_samples)
            src_sentences = [src_sentences[i] for i in indices]
            tgt_sentences = [tgt_sentences[i] for i in indices]

        return ParallelSentences(src_lang, tgt_lang, src_sentences, tgt_sentences)

    def _load_non_english_pair(self, src_lang: str, tgt_lang: str) -> ParallelSentences:
        """Load via English pivot matching (no sampling until final result)."""
        # Temporarily disable max_samples for pivot loading
        saved_max = self.max_samples
        self.max_samples = None

        en_src = self._load_english_pair("en", src_lang)
        en_tgt = self._load_english_pair("en", tgt_lang)

        self.max_samples = saved_max

        en_to_src = dict(zip(en_src.src_sentences, en_src.tgt_sentences))

        src_sentences, tgt_sentences = [], []
        for en_sent, tgt_sent in zip(en_tgt.src_sentences, en_tgt.tgt_sentences):
            if en_sent in en_to_src:
                src_sentences.append(en_to_src[en_sent])
                tgt_sentences.append(tgt_sent)

        if not src_sentences:
            raise ValueError(f"No matching sentences for {src_lang}-{tgt_lang}")

        if self.max_samples and len(src_sentences) > self.max_samples:
            indices = random.sample(range(len(src_sentences)), self.max_samples)
            src_sentences = [src_sentences[i] for i in indices]
            tgt_sentences = [tgt_sentences[i] for i in indices]

        return ParallelSentences(src_lang, tgt_lang, src_sentences, tgt_sentences)
