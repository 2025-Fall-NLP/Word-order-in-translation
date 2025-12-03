"""mBART translation model implementation."""

from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

from .base import BaseTranslator, TrainableMixin
from .registry import register_model


@register_model("mbart")
class MBARTTranslator(BaseTranslator, TrainableMixin):
    """mBART-50 translator. Set tokenizer.src_lang BEFORE encoding to avoid BLEU=0."""

    DEFAULT_LANG_CODES = {
        "en": "en_XX",
        "ko": "ko_KR",
        "ja": "ja_XX",
        "zh": "zh_CN",
        "de": "de_DE",
        "fr": "fr_XX",
        "es": "es_XX",
        "ar": "ar_AR",
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._model_name = config.get(
            "name", "facebook/mbart-large-50-many-to-many-mmt"
        )
        self._tokenizer = MBart50TokenizerFast.from_pretrained(self._model_name)
        self._model = MBartForConditionalGeneration.from_pretrained(self._model_name)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._max_length = config.get("max_length", 256)
        self._num_beams = config.get("num_beams", 4)
        self.lang_codes = config.get("lang_codes", self.DEFAULT_LANG_CODES)

    # BaseTranslator methods
    def translate(
        self,
        texts: List[str],
        src_lang: str,
        tgt_lang: str,
        batch_size: int = 16,
        show_progress: bool = False,
    ) -> List[str]:
        mbart_src = self._get_mbart_code(src_lang)
        mbart_tgt = self._get_mbart_code(tgt_lang)
        self._tokenizer.src_lang = mbart_src
        self._model.eval()

        all_translations = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Translating {src_lang}->{tgt_lang}")

        for i in iterator:
            batch = texts[i : i + batch_size]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._max_length,
            ).to(self._device)
            with torch.no_grad():
                generated = self._model.generate(
                    **inputs,
                    forced_bos_token_id=self._tokenizer.lang_code_to_id[mbart_tgt],
                    max_length=self._max_length,
                    num_beams=self._num_beams,
                )
            all_translations.extend(
                self._tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
        return all_translations

    def load(self, path: str) -> None:
        self._model = MBartForConditionalGeneration.from_pretrained(path)
        self._tokenizer = MBart50TokenizerFast.from_pretrained(path)
        self._model.to(self._device)

    def reload(self) -> None:
        self._model = MBartForConditionalGeneration.from_pretrained(self._model_name)
        self._tokenizer = MBart50TokenizerFast.from_pretrained(self._model_name)
        self._model.to(self._device)

    # TrainableMixin methods
    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def _get_mbart_code(self, lang: str) -> str:
        if lang not in self.lang_codes:
            raise ValueError(f"Unknown language: {lang}")
        return self.lang_codes[lang]

    def preprocess_batch(
        self,
        examples: Dict[str, List[str]],
        src_lang: str,
        tgt_lang: str,
        src_col: str = "src",
        tgt_col: str = "tgt",
        max_length: int = 128,
    ) -> Dict[str, List]:
        """Tokenize batch for training. Sets tokenizer languages inline."""
        self._tokenizer.src_lang = self._get_mbart_code(src_lang)
        self._tokenizer.tgt_lang = self._get_mbart_code(tgt_lang)
        inputs = self._tokenizer(
            examples[src_col], max_length=max_length, truncation=True, padding=False
        )
        with self._tokenizer.as_target_tokenizer():
            labels = self._tokenizer(
                examples[tgt_col], max_length=max_length, truncation=True, padding=False
            )
        inputs["labels"] = labels["input_ids"]
        return inputs
