"""Base classes for translation models and evaluation metrics."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseTranslator(ABC):
    """Base class for translation models."""

    def __init__(self, config: Dict[str, Any], lang_codes: Optional[Dict[str, str]] = None):
        self.config = config
        self.lang_codes = lang_codes or {}

    @abstractmethod
    def translate(self, texts: List[str], src_lang: str, tgt_lang: str, batch_size: int = 16) -> List[str]:
        pass

    @abstractmethod
    def prepare_for_training(self, src_lang: str, tgt_lang: str) -> None:
        pass

    @abstractmethod
    def get_model_for_training(self):
        pass

    @abstractmethod
    def get_tokenizer(self):
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass


class BaseEvalMetric(ABC):
    """Base class for translation evaluation metrics."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def compute(self, hypotheses: List[str], references: List[str], sources: Optional[List[str]] = None) -> float:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def higher_is_better(self) -> bool:
        return True
