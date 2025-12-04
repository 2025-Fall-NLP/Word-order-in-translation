"""Translation evaluation metrics: BLEU and COMET."""

from typing import Any, Dict, List, Optional

import torch

from .base import BaseEvalMetric
from .registry import get_eval, register_eval


@register_eval("bleu")
class BLEUMetric(BaseEvalMetric):
    """BLEU score (0-100) using sacrebleu. Fast but doesn't capture semantics well."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        import evaluate

        self._bleu = evaluate.load("sacrebleu")
        self.tokenize = config.get("tokenize", "13a")

    def compute(
        self,
        hypotheses: List[str],
        references: List[str],
        sources: Optional[List[str]] = None,
    ) -> float:
        return self._bleu.compute(
            predictions=hypotheses,
            references=[[r] for r in references],
            tokenize=self.tokenize,
        )["score"]

    @property
    def name(self) -> str:
        return "bleu"


@register_eval("comet")
class COMETMetric(BaseEvalMetric):
    """COMET score (-1 to 1). Neural metric, requires source sentences."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        from comet import download_model, load_from_checkpoint

        model_path = download_model(config.get("model", "Unbabel/wmt22-comet-da"))
        self._model = load_from_checkpoint(model_path)
        self._gpus = 1 if torch.cuda.is_available() else 0
        self.batch_size = config.get("batch_size", 16)

    def compute(
        self,
        hypotheses: List[str],
        references: List[str],
        sources: Optional[List[str]] = None,
    ) -> float:
        if sources is None:
            raise ValueError("COMET requires source sentences")
        data = [
            {"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources, hypotheses, references)
        ]
        return self._model.predict(
            data,
            batch_size=self.batch_size,
            gpus=self._gpus,
            num_workers=0,
            progress_bar=False,
        ).system_score

    @property
    def name(self) -> str:
        return "comet"


@register_eval("bertscore")
class BERTScoreMetric(BaseEvalMetric):
    """BERTScore F1 (0-1)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_type = config.get("model", "bert-base-multilingual-cased")
        self.batch_size = config.get("batch_size", 64)

    def compute(
        self,
        hypotheses: List[str],
        references: List[str],
        sources: Optional[List[str]] = None,
    ) -> float:
        from bert_score import score

        P, R, F1 = score(
            hypotheses,
            references,
            model_type=self.model_type,
            batch_size=self.batch_size,
            verbose=False,
        )
        return F1.mean().item()

    @property
    def name(self) -> str:
        return "bertscore"
