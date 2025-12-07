"""Requires datasets<3.0.0."""

import random
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from .base import BaseDatasetLoader, ParallelSentences
from .registry import register_dataset

_cache: Dict[str, object] = {}


@register_dataset("flores")
class FloresLoader(BaseDatasetLoader):
    """FLORES-200 parallel corpus loader."""

    DEFAULT_LANG_CODES = {
        "en": "eng_Latn",
        "de": "deu_Latn",
        "fr": "fra_Latn",
        "es": "spa_Latn",
        "it": "ita_Latn",
        "pt": "por_Latn",
        "ro": "ron_Latn",
        "gl": "glg_Latn",
        "ru": "rus_Cyrl",
        "uk": "ukr_Cyrl",
        "ko": "kor_Hang",
        "ja": "jpn_Jpan",
        "zh": "zho_Hans",
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

    def _get_dataset(self, pair_name: str):
        cache_key = f"{self.dataset_name}_{pair_name}_{self.split}"
        if cache_key not in _cache:
            _cache[cache_key] = load_dataset(
                self.dataset_name, pair_name, split=self.split
            )
        return _cache[cache_key]

    def load(self, src_lang: str, tgt_lang: str) -> ParallelSentences:
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
            src_sentences = src_sentences[: self.max_samples]
            tgt_sentences = tgt_sentences[: self.max_samples]

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
            src_sentences = src_sentences[: self.max_samples]
            tgt_sentences = tgt_sentences[: self.max_samples]

        return ParallelSentences(src_lang, tgt_lang, src_sentences, tgt_sentences)


@register_dataset("nllb")
class NLLBLoader(BaseDatasetLoader):
    """NLLB (No Language Left Behind) parallel corpus loader.

    Supports direct X-Y language pairs (no English pivoting required).
    Includes quality score filtering based on LASER scores.

    Uses allenai/nllb dataset from HuggingFace which has:
    - Direct parallel data for many language pairs
    - LASER-based quality scores for filtering

    Note: Not all language pairs are available. NLLB focuses on specific
    language combinations. Use list_available_pairs() to check availability.
    """

    DEFAULT_LANG_CODES = {
        "en": "eng_Latn",
        "de": "deu_Latn",
        "fr": "fra_Latn",
        "es": "spa_Latn",
        "it": "ita_Latn",
        "pt": "por_Latn",
        "ro": "ron_Latn",
        "gl": "glg_Latn",
        "ru": "rus_Cyrl",
        "uk": "ukr_Cyrl",
        "ko": "kor_Hang",
        "ja": "jpn_Jpan",
        "zh": "zho_Hans",
        "ar": "arb_Arab",
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.lang_codes = config.get("lang_codes", self.DEFAULT_LANG_CODES)

    def _get_pair_names(self, src_lang: str, tgt_lang: str) -> List[str]:
        """Get possible dataset pair names in NLLB format (try both orderings)."""
        src_code = self.lang_codes.get(src_lang, src_lang)
        tgt_code = self.lang_codes.get(tgt_lang, tgt_lang)
        return [f"{src_code}-{tgt_code}", f"{tgt_code}-{src_code}"]

    def _get_dataset(self, pair_names: List[str]) -> Tuple[Any, str]:
        """Load dataset with streaming (allenai/nllb is 450GB+)."""
        for pair_name in pair_names:
            cache_key = f"{self.dataset_name}_{pair_name}_{self.split}"
            if cache_key in _cache:
                return _cache[cache_key], pair_name

            try:
                # Use streaming=True because allenai/nllb is massive
                dataset = load_dataset(
                    self.dataset_name,
                    pair_name,
                    split=self.split,
                    trust_remote_code=True,
                    streaming=True,
                )
                _cache[cache_key] = dataset
                return dataset, pair_name
            except Exception:
                continue

        raise ValueError(
            f"Language pair not available in NLLB. Tried: {pair_names}. "
            f"This pair may not exist in the allenai/nllb dataset."
        )

    def load(self, src_lang: str, tgt_lang: str) -> ParallelSentences:
        """Load parallel sentences from NLLB."""
        if src_lang not in self.lang_codes:
            raise ValueError(f"Unknown source language: {src_lang}")
        if tgt_lang not in self.lang_codes:
            raise ValueError(f"Unknown target language: {tgt_lang}")

        src_code = self.lang_codes[src_lang]
        tgt_code = self.lang_codes[tgt_lang]
        pair_names = self._get_pair_names(src_lang, tgt_lang)

        try:
            dataset, actual_pair_name = self._get_dataset(pair_names)
        except Exception as e:
            raise ValueError(f"Failed to load NLLB data for {src_lang}-{tgt_lang}: {e}")

        # Extract sentences (dataset is sorted by quality, take first max_samples)
        src_sentences, tgt_sentences = self._extract_data(
            dataset, src_code, tgt_code, actual_pair_name
        )

        if not src_sentences:
            raise ValueError(f"No sentences found for {src_lang}-{tgt_lang}")

        return ParallelSentences(src_lang, tgt_lang, src_sentences, tgt_sentences)

    def _extract_data(
        self,
        dataset,
        src_code: str,
        tgt_code: str,
        pair_name: str,
    ) -> Tuple[List[str], List[str]]:
        """Extract sentences from streaming dataset.

        NLLB is sorted by LASER score descending, so first samples are best.
        Simply takes first max_samples (or 50k if not set).
        """
        src_sentences = []
        tgt_sentences = []

        # Determine if we need to swap based on pair_name order
        swap_needed = pair_name.startswith(tgt_code)
        target_count = self.max_samples if self.max_samples else 50000

        for ex in dataset:
            src_sent, tgt_sent = self._get_sentences(
                ex, src_code, tgt_code, swap_needed
            )

            if src_sent and tgt_sent:
                src_sentences.append(src_sent)
                tgt_sentences.append(tgt_sent)

            if len(src_sentences) >= target_count:
                break

        return src_sentences, tgt_sentences

    def _get_sentences(
        self, example: Dict, src_code: str, tgt_code: str, swap_needed: bool
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract source and target sentences from example."""
        # Try translation dict format (like OPUS)
        if "translation" in example:
            t = example["translation"]
            # Try full codes first, then short codes
            src_sent = t.get(src_code) or t.get(src_code.split("_")[0])
            tgt_sent = t.get(tgt_code) or t.get(tgt_code.split("_")[0])
            return (src_sent, tgt_sent)

        # Try direct field format (src/tgt or source/target)
        src_sent = (
            example.get("src") or example.get("source") or example.get("sentence1")
        )
        tgt_sent = (
            example.get("tgt") or example.get("target") or example.get("sentence2")
        )

        if src_sent and tgt_sent:
            if swap_needed:
                return (tgt_sent, src_sent)
            return (src_sent, tgt_sent)

        # Try indexed format (text_0, text_1)
        if "text_0" in example and "text_1" in example:
            if swap_needed:
                return (example["text_1"], example["text_0"])
            return (example["text_0"], example["text_1"])

        return (None, None)
