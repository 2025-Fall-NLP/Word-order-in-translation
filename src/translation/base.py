"""Base classes for translation models and evaluation metrics."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm


class BaseTranslator(ABC):
    """Base class for translation models (inference-only).

    Subclasses must set:
        - self._model
        - self._tokenizer
        - self._lang_codes
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._max_length = config.get("max_length", 256)
        self._num_beams = config.get("num_beams", 4)

    def _get_lang_code(self, lang: str) -> str:
        """Convert short language code to model-specific code."""
        if lang not in self._lang_codes:
            raise ValueError(f"Unknown language: {lang}")
        return self._lang_codes[lang]

    def translate(
        self,
        texts: List[str],
        src_lang: str,
        tgt_lang: str,
        batch_size: int = 16,
        show_progress: bool = False,
    ) -> List[str]:
        """Translate texts from source to target language."""
        src_code = self._get_lang_code(src_lang)
        tgt_code = self._get_lang_code(tgt_lang)
        self._tokenizer.src_lang = src_code
        self._tokenizer.tgt_lang = tgt_code
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
                    forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(tgt_code),
                    max_length=self._max_length,
                    num_beams=self._num_beams,
                )
            all_translations.extend(
                self._tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
        return all_translations

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from a checkpoint."""
        pass

    @abstractmethod
    def reload(self) -> None:
        """Reset to base model weights."""
        pass


class TrainableMixin:
    """Mixin for translators to support fine-tuning. The subclass must implement BaseTranslator."""

    @property
    def model(self):
        """Underlying model for trainer access."""
        return self._model

    @property
    def tokenizer(self):
        """Tokenizer for trainer access."""
        return self._tokenizer

    def preprocess_batch(
        self,
        examples: Dict,
        src_lang: str,
        tgt_lang: str,
        src_col: str = "src",
        tgt_col: str = "tgt",
        max_length: int = 128,
    ) -> Dict:
        """Tokenize batch for training."""
        self._tokenizer.src_lang = self._get_lang_code(src_lang)
        self._tokenizer.tgt_lang = self._get_lang_code(tgt_lang)
        inputs = self._tokenizer(
            examples[src_col], max_length=max_length, truncation=True, padding=False
        )
        labels = self._tokenizer(
            text_target=examples[tgt_col],
            max_length=max_length,
            truncation=True,
            padding=False,
        )
        inputs["labels"] = labels["input_ids"]
        return inputs


class BaseEvalMetric(ABC):
    """Base class for translation evaluation metrics."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def compute(
        self,
        hypotheses: List[str],
        references: List[str],
        sources: Optional[List[str]] = None,
    ) -> float:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def higher_is_better(self) -> bool:
        return True
