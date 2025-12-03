"""mBART translation model implementation."""

from typing import Any, Dict, List, Optional
import torch
from tqdm import tqdm
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from .base import BaseTranslator
from .registry import register_model


@register_model("mbart")
class MBARTTranslator(BaseTranslator):
    """mBART-50 translator. Set tokenizer.src_lang BEFORE encoding to avoid BLEU=0."""

    DEFAULT_LANG_CODES = {
        "en": "en_XX", "ko": "ko_KR", "ja": "ja_XX", "zh": "zh_CN",
        "de": "de_DE", "fr": "fr_XX", "es": "es_XX", "ar": "ar_AR",
    }

    def __init__(self, config: Dict[str, Any], lang_codes: Optional[Dict[str, str]] = None):
        super().__init__(config, lang_codes or self.DEFAULT_LANG_CODES)
        model_name = config.get("name", "facebook/mbart-large-50-many-to-many-mmt")
        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        self.model = MBartForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_length = config.get("max_length", 256)
        self.num_beams = config.get("num_beams", 4)
        self._current_src_lang = None
        self._current_tgt_lang = None

    def _get_mbart_code(self, lang: str) -> str:
        if lang not in self.lang_codes:
            raise ValueError(f"Unknown language: {lang}")
        return self.lang_codes[lang]

    def translate(
        self, texts: List[str], src_lang: str, tgt_lang: str,
        batch_size: int = 16, show_progress: bool = False
    ) -> List[str]:
        mbart_src, mbart_tgt = self._get_mbart_code(src_lang), self._get_mbart_code(tgt_lang)
        self.tokenizer.src_lang = mbart_src
        self.model.eval()

        all_translations = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Translating {src_lang}->{tgt_lang}")

        for i in iterator:
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
            ).to(self.device)
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.lang_code_to_id[mbart_tgt],
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                )
            all_translations.extend(self.tokenizer.batch_decode(generated, skip_special_tokens=True))
        return all_translations

    def prepare_for_training(self, src_lang: str, tgt_lang: str) -> None:
        self.tokenizer.src_lang = self._get_mbart_code(src_lang)
        self.tokenizer.tgt_lang = self._get_mbart_code(tgt_lang)
        self._current_src_lang, self._current_tgt_lang = src_lang, tgt_lang

    def preprocess_for_training(
        self, examples: Dict[str, List[str]], src_col: str = "src", tgt_col: str = "tgt", max_length: int = 128
    ) -> Dict[str, List]:
        if not self._current_src_lang:
            raise RuntimeError("Call prepare_for_training() first")
        self.tokenizer.src_lang = self._get_mbart_code(self._current_src_lang)
        inputs = self.tokenizer(examples[src_col], max_length=max_length, truncation=True, padding=False)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples[tgt_col], max_length=max_length, truncation=True, padding=False)
        inputs["labels"] = labels["input_ids"]
        return inputs

    def get_model_for_training(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_forced_bos_token_id(self) -> int:
        if not self._current_tgt_lang:
            raise RuntimeError("Call prepare_for_training() first")
        return self.tokenizer.lang_code_to_id[self._get_mbart_code(self._current_tgt_lang)]

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str) -> None:
        self.model = MBartForConditionalGeneration.from_pretrained(path)
        self.tokenizer = MBart50TokenizerFast.from_pretrained(path)
        self.model.to(self.device)

    def reload_base_model(self) -> None:
        model_name = self.config.get("name", "facebook/mbart-large-50-many-to-many-mmt")
        self.model = MBartForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
