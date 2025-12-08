"""Base data structures for correlation analysis."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CorrelationResult:
    """Single correlation result between similarity and translation metrics."""

    similarity_metric: str
    translation_metric: str
    stage: str  # baseline, finetuned, delta
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    n_pairs: int
    # FDR-adjusted p-values (set after correction)
    pearson_p_adj: Optional[float] = field(default=None)
    spearman_p_adj: Optional[float] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "similarity_metric": self.similarity_metric,
            "translation_metric": self.translation_metric,
            "stage": self.stage,
            "pearson": {"r": self.pearson_r, "p": self.pearson_p},
            "spearman": {"rho": self.spearman_r, "p": self.spearman_p},
            "n_pairs": self.n_pairs,
        }
        # Include adjusted p-values if computed
        if self.pearson_p_adj is not None:
            result["pearson"]["p_adj"] = self.pearson_p_adj
        if self.spearman_p_adj is not None:
            result["spearman"]["p_adj"] = self.spearman_p_adj
        return result
