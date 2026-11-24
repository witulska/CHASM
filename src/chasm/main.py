"""Public CHASM significance-testing API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Sequence

import numpy as np
from scipy.stats import bootstrap

from .scoring_corr import (
    bootstrap_ci_np,
    compute_max_spearman_rank_correlation,
)
from .scoring_divergences import (
    bootstrap_ci_divergences,
    compute_entropy_difference,
    compute_max_bhattacharyya_distance,
    compute_max_jeffrey_distance,
    compute_max_kl_divergence,
)
from .scoring_lmoments import (
    compute_l_kurtosis_difference,
    compute_l_mean_difference,
    compute_l_scale_difference,
    compute_l_skewness_difference,
)
from .scoring_robust_moments import (
    compute_exact_kurtosis_difference,
    compute_exact_skewness_difference,
    compute_robust_mean_difference,
    compute_robust_scale_difference,
)
from .utils import (
    generate_corr_matrix_for_sim,
    generate_metadata,
    mdim_pareto_rv,
    t_student_vector,
)

Statistic = Callable[..., Any]

ARTICLE_METRICS: tuple[str, ...] = (
    "sig_l_mean",
    "sig_l_scale",
    "sig_l_skewness",
    "sig_l_kurtosis",
    "sig_robust_scale",
    "sig_spearman_corr",
    "sig_bhattacharyya_dist",
    "sig_jeffrey_dist",
)


def _reference_segments(segment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split the reference segment into the two samples used by the bootstrap."""
    segment = np.asarray(segment)
    if segment.ndim == 1:
        segment = segment[:, None]
    if segment.ndim != 2:
        raise ValueError("segments must be one- or two-dimensional arrays")
    midpoint = len(segment) // 2
    if midpoint < 2 or len(segment) - midpoint < 2:
        raise ValueError("segment1 must contain at least four observations")
    return segment[:midpoint], segment[midpoint:]


def _scalar(value: Any) -> float:
    """Convert the scalar-like statistics used by the scoring modules to float."""
    values = np.asarray(value, dtype=float)
    if values.size == 0:
        return 0.0
    return float(values.reshape(-1)[0])


def _lower_imf(lower_bound: float, stat_value: float, decision: int) -> float:
    if not decision:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return float((lower_bound - stat_value) / lower_bound)


def _upper_imf(upper_bound: float, stat_value: float, decision: int) -> float:
    if not decision:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return float((stat_value - upper_bound) / upper_bound)


def _lower_result(lower_bound: Any, stat_value: Any) -> dict[str, float | int | None]:
    lower = _scalar(lower_bound)
    statistic = _scalar(stat_value)
    decision = int(statistic < lower)
    return {
        "decision": decision,
        "upper_bound": None,
        "lower_bound": lower,
        "stat_value": statistic,
        "imf": _lower_imf(lower, statistic, decision),
    }


def _upper_result(upper_bound: Any, stat_value: Any) -> dict[str, float | int | None]:
    upper = _scalar(upper_bound)
    statistic = _scalar(stat_value)
    decision = int(statistic > upper)
    return {
        "decision": decision,
        "upper_bound": upper,
        "lower_bound": None,
        "stat_value": statistic,
        "imf": _upper_imf(upper, statistic, decision),
    }


def bootstraping_one_dim_stats(
    segment1: np.ndarray,
    segment2: np.ndarray,
    statistic: Statistic,
    method: str = "percentile",
    alternative: str = "greater",
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
) -> dict[str, float | int | None]:
    """Bootstrap a one-dimensional statistic and return the Brainlab result schema."""
    segment1 = np.asarray(segment1)
    segment2 = np.asarray(segment2)
    if segment1.ndim == 1:
        segment1 = segment1[:, None]
    if segment2.ndim == 1:
        segment2 = segment2[:, None]
    reference = _reference_segments(segment1)
    result = bootstrap(
        data=reference,
        statistic=statistic,
        vectorized=False,
        axis=0,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method=method,
        alternative=alternative,
    )
    lower_bound = result.confidence_interval.low
    stat_value = statistic(np.asarray(segment1), np.asarray(segment2), axis=0)
    return _lower_result(lower_bound, stat_value)


def bootstraping_corr(
    segment1: np.ndarray,
    segment2: np.ndarray,
    confidence_level: float = 0.95,
    n_resamples: int = 100,
    statistic: Statistic | None = None,
) -> dict[str, float | int | None]:
    """Bootstrap a correlation-like statistic using an upper confidence bound.

    ``statistic`` must accept two arrays and an ``axis`` keyword and return a
    scalar-like value or a one-element array. The default is the maximum
    pairwise Spearman-correlation change.
    """
    correlation_statistic = statistic or compute_max_spearman_rank_correlation
    if np.asarray(segment1).ndim == 1 or np.asarray(segment1).shape[1] < 2:
        return _upper_result(0.0, 0.0)
    segment1 = np.asarray(segment1)
    segment2 = np.asarray(segment2)
    result = bootstrap_ci_np(
        data=_reference_segments(np.asarray(segment1)),
        statistic=correlation_statistic,
        axis=1,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        alternative="less",
    )
    upper_bound = result["confidence_interval"]["high"]
    stat_value = correlation_statistic(
        np.asarray(segment1), np.asarray(segment2), axis=0
    )
    return _upper_result(upper_bound, stat_value)


def bootstraping_divergences(
    segment1: np.ndarray,
    segment2: np.ndarray,
    statistic: Statistic,
    confidence_level: float = 0.95,
    n_resamples: int = 100,
) -> dict[str, float | int | None]:
    """Bootstrap a divergence statistic and return the Brainlab result schema."""
    segment1 = np.asarray(segment1)
    segment2 = np.asarray(segment2)
    if segment1.ndim == 1:
        segment1 = segment1[:, None]
    if segment2.ndim == 1:
        segment2 = segment2[:, None]
    result = bootstrap_ci_divergences(
        data=_reference_segments(np.asarray(segment1)),
        statistic=statistic,
        axis=1,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        alternative="less",
    )
    upper_bound = result["confidence_interval"]["high"]
    stat_value = statistic(np.asarray(segment1), np.asarray(segment2), axis=0)
    return _upper_result(upper_bound, stat_value)


def chasm_main(
    segment1: np.ndarray,
    segment2: np.ndarray,
    confidence_level: float = 0.95,
    n_resamples: int = 100,
) -> dict[str, dict[str, float | int | None]]:
    """Run all CHASM metrics and return one Brainlab result per metric.

    The bootstrap null is built from the two halves of ``segment1``. The observed statistic
    always compares the complete ``segment1`` and
    ``segment2`` arrays, matching both original branches.
    """
    segment1 = np.asarray(segment1)
    segment2 = np.asarray(segment2)
    if len(segment1) != len(segment2):
        raise ValueError(
            "segment1 and segment2 must have the same number of observations"
        )

    result: dict[str, dict[str, float | int | None]] = {}
    one_dimensional: tuple[tuple[str, Statistic], ...] = (
        ("sig_l_mean", compute_l_mean_difference),
        ("sig_l_scale", compute_l_scale_difference),
        ("sig_l_skewness", compute_l_skewness_difference),
        ("sig_l_kurtosis", compute_l_kurtosis_difference),
        ("sig_robust_mean", compute_robust_mean_difference),
        ("sig_robust_scale", compute_robust_scale_difference),
        ("sig_exact_skewness", compute_exact_skewness_difference),
        ("sig_exact_kurtosis", compute_exact_kurtosis_difference),
    )
    for name, statistic in one_dimensional:
        result[name] = bootstraping_one_dim_stats(
            segment1,
            segment2,
            statistic,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
        )

    result["sig_spearman_corr"] = bootstraping_corr(
        segment1, segment2, confidence_level=confidence_level, n_resamples=n_resamples
    )
    divergence_metrics: tuple[tuple[str, Statistic], ...] = (
        ("sig_entropy_diff", compute_entropy_difference),
        ("sig_bhattacharyya_dist", compute_max_bhattacharyya_distance),
        ("sig_jeffrey_dist", compute_max_jeffrey_distance),
        ("sig_kl_divergence", compute_max_kl_divergence),
    )
    for name, statistic in divergence_metrics:
        result[name] = bootstraping_divergences(
            segment1,
            segment2,
            statistic,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
        )
    return result


def chasm_bounds_main(
    segment1: np.ndarray,
    segment2: np.ndarray,
    confidence_level: float = 0.95,
    n_resamples: int = 100,
    metrics: Sequence[str] | None = None,
) -> dict[str, dict[str, float | int | None]]:
    """Return the metrics used in the article in the unified result format.

    By default, this function returns the subset listed in
    :data:`ARTICLE_METRICS`. Pass ``metrics`` to select a different subset
    from the metrics returned by :func:`chasm_main`.
    """
    results = chasm_main(segment1, segment2, confidence_level, n_resamples)
    selected_metrics = ARTICLE_METRICS if metrics is None else tuple(metrics)
    unknown_metrics = set(selected_metrics) - set(results)
    if unknown_metrics:
        raise ValueError(f"Unknown CHASM metrics: {sorted(unknown_metrics)}")
    return {name: results[name] for name in selected_metrics}


__all__ = [
    "bootstraping_corr",
    "bootstraping_divergences",
    "bootstraping_one_dim_stats",
    "ARTICLE_METRICS",
    "chasm_bounds_main",
    "chasm_main",
    "generate_corr_matrix_for_sim",
    "generate_metadata",
    "mdim_pareto_rv",
    "t_student_vector",
]
