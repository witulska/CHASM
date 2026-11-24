"""Compatible monolithic API for the legacy scoring module.

New code should use the specialized ``scoring_*`` modules.
These functions remain available so older scripts and analyses continue to work.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import numpy as np
import pandas as pd

from .scoring_divergences import (
    Silverman_rule,
    compute_max_bhattacharyya_distance,
    compute_max_jeffrey_distance,
    compute_max_kl_divergence,
    entropy,
    ksdensity,
)
from .scoring_divergences import (
    bhattacharyya_distance as _bhattacharyya_values,
)
from .scoring_divergences import (
    jeffrey_distance as _jeffrey_values,
)
from .scoring_divergences import (
    kl_divergence as _kl_values,
)
from .scoring_lmoments import (
    compute_l_kurtosis_difference,
    compute_l_mean_difference,
    compute_l_scale_difference,
    compute_l_skewness_difference,
    l_kurtosis,
    l_mean,
    l_scale,
    l_skewness,
)


def bhattacharyya_distance(
    y1: np.ndarray, y2: np.ndarray, axis: Optional[int] = None
) -> float | np.ndarray:
    """Compute the Bhattacharyya distance for two samples."""
    del axis
    values = _bhattacharyya_values(y1, y2)
    return float(values[0]) if values.size == 1 else values


def jeffrey_distance(
    y1: np.ndarray, y2: np.ndarray, axis: Optional[int] = None
) -> float | np.ndarray:
    """Compute the Jeffrey distance for two samples."""
    del axis
    values = _jeffrey_values(y1, y2)
    return float(values[0]) if values.size == 1 else values


def kl_divergence(
    y1: np.ndarray, y2: np.ndarray, axis: Optional[int] = None
) -> float | np.ndarray:
    """Compute the KL divergence for two samples."""
    del axis
    values = _kl_values(y1, y2)
    return float(values[0]) if values.size == 1 else values


def entropy_legacy(y: np.ndarray, axis: Optional[int] = None) -> float | np.ndarray:
    """Compute histogram entropy through the legacy interface."""
    return entropy(y, axis=axis)


def function_difference(
    y1: np.ndarray,
    y2: np.ndarray,
    axis: Optional[int] = None,
    function: Callable = l_mean,
) -> float | np.ndarray:
    """Compute the absolute difference between two statistic results."""
    return np.abs(function(y1, axis=axis) - function(y2, axis=axis))


def compute_entropy_difference(
    y1: np.ndarray, y2: np.ndarray, axis: Optional[int] = None
) -> np.ndarray:
    """Compute the normalized entropy difference."""
    difference = np.abs(entropy(y1, axis=axis) - entropy(y2, axis=axis))
    return np.array([np.min(-np.log(difference + np.finfo(float).eps))])


def evaluate_change_points(
    data: pd.DataFrame,
    detected_cps: list[int],
    m: int,
    k: int,
    threshold: float,
    dim_metric_function: str | Callable[[np.ndarray, np.ndarray], Any],
    score_function: str | Callable[[np.ndarray], Any],
) -> dict[int, Optional[dict[str, np.ndarray | float]]]:
    """Evaluate the significance of detected change points in a multivariate time series.

    ``m`` is the sample length before a change point and ``k`` is the sample length
    after it. The result contains per-point metrics and an aggregated score.
    """
    n, dimensions = data.shape
    points = sorted(detected_cps)
    significance: dict[int, Optional[dict[str, np.ndarray | float]]] = {
        point: None for point in points
    }
    metric_functions: dict[str, Callable] = {
        "bhattacharyya": lambda first, second: bhattacharyya_distance(first, second),
        "kl_divergence": lambda first, second: kl_divergence(first, second),
        "l_mean": lambda first, second: abs(l_mean(first) - l_mean(second)),
        "l_scale": lambda first, second: abs(l_scale(first) - l_scale(second)),
        "l_skewness": lambda first, second: abs(l_skewness(first) - l_skewness(second)),
        "l_kurtosis": lambda first, second: abs(l_kurtosis(first) - l_kurtosis(second)),
        "entropy": lambda first, second: abs(entropy(first) - entropy(second)),
    }
    if isinstance(dim_metric_function, str):
        metric = metric_functions.get(dim_metric_function)
        if metric is None:
            raise ValueError(f"Nieznana funkcja metryki: {dim_metric_function}")
    else:
        metric = dim_metric_function

    score_functions: dict[str, Callable] = {
        "sum_log": lambda values: -np.sum(np.log(values + np.finfo(float).eps)),
        "max": np.max,
    }
    if isinstance(score_function, str):
        scorer = score_functions.get(score_function)
        if scorer is None:
            raise ValueError(f"Unknown aggregation function:  {score_function}")
    else:
        scorer = score_function

    first_start = 0
    for index, point in enumerate(points.copy()):
        first = data.iloc[first_start : first_start + m].to_numpy()
        second = data.iloc[point : min(point + k, n)].to_numpy()
        if len(second) < k:
            break
        metrics = [metric(first[:, dimension], second[:, dimension]) for dimension in range(dimensions)]
        max_length = max(len(np.atleast_1d(value)) for value in metrics)
        padded = np.stack(
            [np.pad(np.atleast_1d(value), (0, max_length - len(np.atleast_1d(value)))) for value in metrics]
        )
        score = float(scorer(padded))
        significance[point] = {"single_dim_metrics": padded, "score": score}
        if score > threshold:
            points.remove(point)
        elif index != len(points) - 1:
            first_start = point
    return significance


__all__ = [
    "Silverman_rule",
    "ksdensity",
    "bhattacharyya_distance",
    "jeffrey_distance",
    "kl_divergence",
    "entropy",
    "entropy_legacy",
    "function_difference",
    "compute_l_mean_difference",
    "compute_l_scale_difference",
    "compute_l_kurtosis_difference",
    "compute_l_skewness_difference",
    "compute_entropy_difference",
    "compute_max_bhattacharyya_distance",
    "compute_max_jeffrey_distance",
    "compute_max_kl_divergence",
    "evaluate_change_points",
]
