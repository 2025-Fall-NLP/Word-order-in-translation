"""Correlation analysis between language similarity and translation quality."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats


@dataclass
class CorrelationResult:
    similarity_metric: str
    translation_metric: str
    stage: str  # baseline, finetuned, improvement
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


def compute_correlation(
    sim_scores: Dict[str, float], trans_scores: Dict[str, float],
    sim_metric: str = "", trans_metric: str = "", stage: str = ""
) -> CorrelationResult:
    """Compute Pearson and Spearman correlations between similarity and translation."""
    common = set(sim_scores.keys()) & set(trans_scores.keys())
    if len(common) < 3:
        raise ValueError(f"Need >=3 pairs, got {len(common)}")

    sim_arr = np.array([sim_scores[p] for p in sorted(common)])
    trans_arr = np.array([trans_scores[p] for p in sorted(common)])

    valid = ~(np.isnan(sim_arr) | np.isnan(trans_arr))
    sim_arr, trans_arr = sim_arr[valid], trans_arr[valid]
    if len(sim_arr) < 3:
        raise ValueError("Not enough valid pairs")

    pr, pp = stats.pearsonr(sim_arr, trans_arr)
    sr, sp = stats.spearmanr(sim_arr, trans_arr)

    return CorrelationResult(sim_metric, trans_metric, stage, float(pr), float(pp), float(sr), float(sp), len(sim_arr))


def load_all_results(output_dir: str) -> Tuple[Dict, Dict, Dict]:
    """Load similarity, baseline, and finetuned results from output directory."""
    p = Path(output_dir)

    def load_dir(subdir, prefix=""):
        results = {}
        d = p / subdir
        if d.exists():
            for f in d.glob(f"{prefix}*.json"):
                with open(f) as fp:
                    name = f.stem.replace(prefix, "") if prefix else f.stem
                    results[name] = json.load(fp).get("results", {})
        return results

    return load_dir("similarity"), load_dir("translation", "baseline_"), load_dir("translation", "finetuned_")


def compute_improvement(baseline: Dict[str, float], finetuned: Dict[str, float]) -> Dict[str, float]:
    """Compute improvement (finetuned - baseline) for each pair."""
    common = set(baseline.keys()) & set(finetuned.keys())
    return {p: finetuned[p] - baseline[p] for p in common if baseline[p] is not None and finetuned[p] is not None}


def analyze_all_correlations(output_dir: str) -> List[CorrelationResult]:
    """Analyze all correlations: similarity vs baseline/finetuned/improvement."""
    sim_results, baseline, finetuned = load_all_results(output_dir)
    results = []

    for sim_name, sim_scores in sim_results.items():
        for trans_name, trans_scores in baseline.items():
            try:
                results.append(compute_correlation(sim_scores, trans_scores, sim_name, trans_name, "baseline"))
            except ValueError as e:
                print(f"Skip {sim_name} vs baseline_{trans_name}: {e}")

        for trans_name, trans_scores in finetuned.items():
            try:
                results.append(compute_correlation(sim_scores, trans_scores, sim_name, trans_name, "finetuned"))
            except ValueError as e:
                print(f"Skip {sim_name} vs finetuned_{trans_name}: {e}")

        for trans_name in baseline:
            if trans_name in finetuned:
                improvement = compute_improvement(baseline[trans_name], finetuned[trans_name])
                try:
                    results.append(compute_correlation(sim_scores, improvement, sim_name, trans_name, "improvement"))
                except ValueError as e:
                    print(f"Skip {sim_name} vs improvement_{trans_name}: {e}")

    return results


def save_correlation_results(results: List[CorrelationResult], output_path: str) -> None:
    """Save correlation results to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "summary": {
            "n_correlations": len(results),
            "significant": sum(1 for r in results if r.pearson_p < 0.05 or r.spearman_p < 0.05),
        },
        "results": [r.to_dict() for r in results],
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)


def print_correlation_summary(results: List[CorrelationResult]) -> None:
    """Print human-readable summary."""
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    for stage in ["baseline", "finetuned", "improvement"]:
        stage_results = [r for r in results if r.stage == stage]
        if not stage_results:
            continue
        print(f"\n{stage.upper()}:")
        for r in stage_results:
            sig_p = "**" if r.pearson_p < 0.01 else "*" if r.pearson_p < 0.05 else ""
            sig_s = "**" if r.spearman_p < 0.01 else "*" if r.spearman_p < 0.05 else ""
            print(f"  {r.similarity_metric} vs {r.translation_metric}: r={r.pearson_r:+.3f}{sig_p}, ρ={r.spearman_r:+.3f}{sig_s}")

    print("\n* p<0.05, ** p<0.01")
