"""Translation model implementations: mBART-50 and NLLB-200."""

from typing import Any, Dict

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
)

from .base import BaseTranslator, TrainableMixin
from .registry import register_model


@register_model("mbart")
class MBARTTranslator(BaseTranslator, TrainableMixin):
    """mBART-50 translator."""

    DEFAULT_LANG_CODES = {
        "en": "en_XX",
        "de": "de_DE",
        "fr": "fr_XX",
        "es": "es_XX",
        "it": "it_IT",
        "pt": "pt_XX",
        "ro": "ro_RO",
        "gl": "gl_ES",
        "ru": "ru_RU",
        "uk": "uk_UA",
        "ko": "ko_KR",
        "ja": "ja_XX",
        "zh": "zh_CN",
        "ar": "ar_AR",
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._model_name = config.get(
            "name", "facebook/mbart-large-50-many-to-many-mmt"
        )
        self.reload()
        self._lang_codes = config.get("lang_codes", self.DEFAULT_LANG_CODES)

    def load(self, path: str) -> None:
        self._model = MBartForConditionalGeneration.from_pretrained(path)
        self._tokenizer = MBart50TokenizerFast.from_pretrained(path)
        self._model.to(self._device)

    def reload(self) -> None:
        self._model = MBartForConditionalGeneration.from_pretrained(self._model_name)
        self._tokenizer = MBart50TokenizerFast.from_pretrained(self._model_name)
        self._model.to(self._device)


@register_model("nllb")
class NLLBTranslator(BaseTranslator, TrainableMixin):
    """NLLB-200 translator. More balanced performance across low-resource languages."""

    # FLORES-200 language codes used by NLLB
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
        self._model_name = config.get("name", "facebook/nllb-200-distilled-600M")
        self.reload()
        self._lang_codes = config.get("lang_codes", self.DEFAULT_LANG_CODES)

    def load(self, path: str) -> None:
        self._model = AutoModelForSeq2SeqLM.from_pretrained(path)
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._model.to(self._device)

    def reload(self) -> None:
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model.to(self._device)
