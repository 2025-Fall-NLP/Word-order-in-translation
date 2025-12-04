"""Correlation analysis between language similarity and translation quality."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class CorrelationResult:
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

    # Residualize x and y on z
    def residualize(a, b):
        slope, intercept, _, _, _ = stats.linregress(b, a)
        return a - (slope * b + intercept)

    x_resid = residualize(x, z)
    y_resid = residualize(y, z)

    r, p = stats.pearsonr(x_resid, y_resid)
    return float(r), float(p)


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

    return (
        load_dir("similarity"),
        load_dir("translation", "baseline_"),
        load_dir("translation", "finetuned_"),
    )


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


def analyze_all_correlations(output_dir: str) -> List[CorrelationResult]:
    """
    Analyze all correlations:
    - Similarity vs baseline
    - Similarity vs finetuned
    - Similarity vs delta (absolute improvement)
    - Similarity vs delta_pct (relative improvement)
    - Partial: Similarity vs finetuned, controlling for baseline
    """
    sim_results, baseline, finetuned = load_all_results(output_dir)
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


def save_correlation_results(
    results: List[CorrelationResult], output_path: str
) -> None:
    """Save correlation results to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Group by stage for summary
    by_stage = {}
    for r in results:
        by_stage.setdefault(r.stage, []).append(r)

    summary = {
        "summary": {
            "n_correlations": len(results),
            "significant_pearson": sum(
                1 for r in results if not np.isnan(r.pearson_p) and r.pearson_p < 0.05
            ),
            "significant_spearman": sum(
                1 for r in results if not np.isnan(r.spearman_p) and r.spearman_p < 0.05
            ),
        },
        "note": "mBART-50 pretraining used imbalanced data; use delta_pct and partial correlations for fairer analysis",
        "results": [r.to_dict() for r in results],
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)


def print_correlation_summary(results: List[CorrelationResult]) -> None:
    """Print human-readable summary."""
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    stages = ["baseline", "finetuned", "delta", "delta_pct", "partial"]
    stage_labels = {
        "baseline": "BASELINE (sim vs base quality)",
        "finetuned": "FINETUNED (sim vs fine quality)",
        "delta": "DELTA (sim vs absolute improvement)",
        "delta_pct": "DELTA % (sim vs relative improvement)",
        "partial": "PARTIAL (sim vs fine, controlling for base)",
    }

    for stage in stages:
        stage_results = [r for r in results if r.stage == stage]
        if not stage_results:
            continue

        print(f"\n{stage_labels.get(stage, stage.upper())}:")
        for r in stage_results:
            sig_p = "**" if r.pearson_p < 0.01 else "*" if r.pearson_p < 0.05 else ""
            if stage == "partial":
                print(
                    f"  {r.similarity_metric} vs {r.translation_metric}: r={r.pearson_r:+.3f}{sig_p}"
                )
            else:
                sig_s = (
                    "**" if r.spearman_p < 0.01 else "*" if r.spearman_p < 0.05 else ""
                )
                print(
                    f"  {r.similarity_metric} vs {r.translation_metric}: "
                    f"r={r.pearson_r:+.3f}{sig_p}, ρ={r.spearman_r:+.3f}{sig_s}"
                )

    print("\n* p<0.05, ** p<0.01")
    print(
        "Note: Use delta_pct and partial for fairer analysis (controls for baseline bias)"
    )
