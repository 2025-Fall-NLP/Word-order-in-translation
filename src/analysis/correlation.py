"""Correlation computation functions."""

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

from .base import CorrelationResult


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


def compute_partial_correlation(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> Tuple[float, float]:
    """
    Compute partial correlation between x and y, controlling for z.
    Returns (partial_r, p_value).
    """
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[valid], y[valid], z[valid]
    if len(x) < 4:
        raise ValueError("Need >=4 samples for partial correlation")

    def residualize(a, b):
        slope, intercept, _, _, _ = stats.linregress(b, a)
        return a - (slope * b + intercept)

    x_resid = residualize(x, z)
    y_resid = residualize(y, z)

    r, p = stats.pearsonr(x_resid, y_resid)
    return float(r), float(p)


def compute_improvements(
    baseline: Dict[str, float], finetuned: Dict[str, float]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute absolute (delta) and relative (delta_pct) improvements."""
    common = set(baseline.keys()) & set(finetuned.keys())
    delta = {}
    delta_pct = {}
    for p in common:
        b, f = baseline[p], finetuned[p]
        if b is not None and f is not None:
            delta[p] = f - b
            delta_pct[p] = ((f - b) / b * 100) if b != 0 else 0.0
    return delta, delta_pct


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
    - Similarity vs delta_pct (relative improvement)
    - Partial: Similarity vs finetuned, controlling for baseline
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
                    base_arr_f = np.array([base_scores[p] for p in common_fine])
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
                    delta_dict, delta_pct_dict = compute_improvements(
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

                    # Similarity vs delta_pct (relative)
                    delta_pct_arr = np.array([delta_pct_dict[p] for p in common_fine])
                    try:
                        results.append(
                            compute_correlation(
                                sim_arr_f,
                                delta_pct_arr,
                                sim_name,
                                trans_name,
                                "delta_pct",
                            )
                        )
                    except ValueError as e:
                        print(f"Skip {sim_name} vs delta_pct_{trans_name}: {e}")

                    # Partial correlation: similarity vs finetuned, controlling for baseline
                    try:
                        partial_r, partial_p = compute_partial_correlation(
                            sim_arr_f, fine_arr, base_arr_f
                        )
                        results.append(
                            CorrelationResult(
                                sim_name,
                                trans_name,
                                "partial",
                                partial_r,
                                partial_p,
                                np.nan,  # No Spearman for partial
                                np.nan,
                                len(common_fine),
                            )
                        )
                    except ValueError as e:
                        print(f"Skip {sim_name} vs partial_{trans_name}: {e}")

    return results
