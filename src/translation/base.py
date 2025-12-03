"""Base classes for translation models and evaluation metrics."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseTranslator(ABC):
    """Base class for translation models (inference-only)."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def translate(
        self, texts: List[str], src_lang: str, tgt_lang: str, **kwargs
    ) -> List[str]:
        """Translate text(s) from source to target language."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from a checkpoint."""
        pass

    @abstractmethod
    def reload(self) -> None:
        """Reset to base model weights."""
        pass


class TrainableMixin(ABC):
    """Mixin for translators that support fine-tuning."""

    @property
    @abstractmethod
    def model(self):
        """Underlying model for trainer access."""
        pass

    @property
    @abstractmethod
    def tokenizer(self):
        """Tokenizer for trainer access."""
        pass

    @abstractmethod
    def preprocess_batch(
        self,
        examples: Dict,
        src_lang: str,
        tgt_lang: str,
        src_col: str,
        tgt_col: str,
        max_length: int,
    ) -> Dict:
        """Tokenize batch for training."""
        pass


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
