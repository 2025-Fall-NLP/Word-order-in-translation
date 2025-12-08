"""Correlation computation functions."""

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

from .base import CorrelationResult


def apply_fdr_correction(results: List[CorrelationResult]) -> List[CorrelationResult]:
    """
    Apply Benjamini-Hochberg FDR correction to all p-values.
    """
    if not results:
        return results

    # Collect all valid Pearson p-values
    pearson_ps = []
    pearson_indices = []
    for i, r in enumerate(results):
        if not np.isnan(r.pearson_p):
            pearson_ps.append(r.pearson_p)
            pearson_indices.append(i)

    # Collect all valid Spearman p-values
    spearman_ps = []
    spearman_indices = []
    for i, r in enumerate(results):
        if not np.isnan(r.spearman_p):
            spearman_ps.append(r.spearman_p)
            spearman_indices.append(i)

    # Apply BH correction to Pearson p-values
    if pearson_ps:
        pearson_adj = _benjamini_hochberg(np.array(pearson_ps))
        for idx, adj_p in zip(pearson_indices, pearson_adj):
            results[idx].pearson_p_adj = float(adj_p)

    # Apply BH correction to Spearman p-values
    if spearman_ps:
        spearman_adj = _benjamini_hochberg(np.array(spearman_ps))
        for idx, adj_p in zip(spearman_indices, spearman_adj):
            results[idx].spearman_p_adj = float(adj_p)

    return results


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.
    """
    n = len(p_values)
    if n == 0:
        return p_values

    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    ranks = np.arange(1, n + 1)
    adjusted = sorted_p * n / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    result = np.empty(n)
    result[sorted_indices] = adjusted
    return result


def compute_correlation(
    x: np.ndarray,
    y: np.ndarray,
    sim_metric: str = "",
    trans_metric: str = "",
    stage: str = "",
) -> CorrelationResult:
    """Compute Pearson and Spearman correlations between two arrays."""
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    if len(x) < 3:
        raise ValueError("Not enough valid pairs")

    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)

    return CorrelationResult(
        sim_metric,
        trans_metric,
        stage,
        float(pr),
        float(pp),
        float(sr),
        float(sp),
        len(x),
    )


def compute_abs_delta(
    baseline: Dict[str, float], finetuned: Dict[str, float]
) -> Dict[str, float]:
    """Compute absolute delta between baseline and finetuned."""
    common = set(baseline.keys()) & set(finetuned.keys())
    return {p: finetuned[p] - baseline[p] for p in common}


def analyze_all_correlations(
    sim_results: Dict[str, Dict[str, float]],
    baseline: Dict[str, Dict[str, float]],
    finetuned: Dict[str, Dict[str, float]],
) -> List[CorrelationResult]:
    """
    Analyze all correlations between similarity and translation metrics.

    Computes for each (similarity_metric, translation_metric) pair:
    - Similarity vs baseline
    - Similarity vs finetuned
    - Similarity vs delta (absolute improvement)
    """
    results = []

    for sim_name, sim_scores in sim_results.items():
        for trans_name in baseline:
            base_scores = baseline[trans_name]
            fine_scores = finetuned.get(trans_name, {})

            # Get common pairs
            common = sorted(set(sim_scores.keys()) & set(base_scores.keys()))
            if len(common) < 3:
                continue

            sim_arr = np.array([sim_scores[p] for p in common])
            base_arr = np.array([base_scores[p] for p in common])

            # Similarity vs baseline
            try:
                results.append(
                    compute_correlation(
                        sim_arr, base_arr, sim_name, trans_name, "baseline"
                    )
                )
            except ValueError as e:
                print(f"Skip {sim_name} vs baseline_{trans_name}: {e}")

            # If finetuned exists
            if fine_scores:
                common_fine = sorted(set(common) & set(fine_scores.keys()))
                if len(common_fine) >= 3:
                    sim_arr_f = np.array([sim_scores[p] for p in common_fine])
                    fine_arr = np.array([fine_scores[p] for p in common_fine])

                    # Similarity vs finetuned
                    try:
                        results.append(
                            compute_correlation(
                                sim_arr_f, fine_arr, sim_name, trans_name, "finetuned"
                            )
                        )
                    except ValueError as e:
                        print(f"Skip {sim_name} vs finetuned_{trans_name}: {e}")

                    # Compute improvements
                    delta_dict = compute_abs_delta(
                        {p: base_scores[p] for p in common_fine},
                        {p: fine_scores[p] for p in common_fine},
                    )

                    # Similarity vs delta (absolute)
                    delta_arr = np.array([delta_dict[p] for p in common_fine])
                    try:
                        results.append(
                            compute_correlation(
                                sim_arr_f, delta_arr, sim_name, trans_name, "delta"
                            )
                        )
                    except ValueError as e:
                        print(f"Skip {sim_name} vs delta_{trans_name}: {e}")

    return results
