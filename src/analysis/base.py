"""Base data structures for correlation analysis."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class CorrelationResult:
    """Single correlation result between similarity and translation metrics."""

    similarity_metric: str
    translation_metric: str
    stage: str  # baseline, finetuned, delta, delta_pct, partial
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    n_pairs: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "similarity_metric": self.similarity_metric,
            "translation_metric": self.translation_metric,
            "stage": self.stage,
            "pearson": {"r": self.pearson_r, "p": self.pearson_p},
            "spearman": {"rho": self.spearman_r, "p": self.spearman_p},
            "n_pairs": self.n_pairs,
        }
