"""Correlation statistics and bootstrap confidence intervals."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import spearmanr


def fisher_z(r: float | np.ndarray) -> float | np.ndarray:
    """Apply Fisher's z transform after clipping correlation values."""
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))


def inv_fisher_z(z: float | np.ndarray) -> float | np.ndarray:
    """Invert Fisher's z transform."""
    return np.tanh(z)


def _spearman_rank_correlation(data: np.ndarray) -> np.ndarray:
    """Return Fisher-transformed Spearman correlations for all column pairs."""
    data = np.atleast_2d(np.asarray(data))
    if data.shape[1] < 2:
        return np.array([], dtype=float)

    correlations = []
    for i in range(data.shape[1]):
        for j in range(i + 1, data.shape[1]):
            result = spearmanr(data[:, i], data[:, j])
            coefficient = getattr(result, "statistic", result[0])
            correlations.append(float(fisher_z(coefficient)))
    return np.asarray(correlations, dtype=float)


def _difference(y: np.ndarray, axis: int | None = None) -> float | np.ndarray:
    """Compute the maximum absolute difference between correlation arrays."""
    return np.max(np.abs(y), axis=axis)


def difference_axis(y: np.ndarray, axis: int | None = None) -> float | np.ndarray:
    """Compute the maximum absolute difference along the selected axis."""
    return _difference(np.asarray(y), axis=axis)


def compute_max_spearman_rank_correlation(
    *data: np.ndarray, axis: int | None = 1
) -> np.ndarray:
    """Compare the largest pairwise Spearman correlation change."""
    if len(data) != 2:
        raise ValueError("Exactly two data arrays are required")
    correlation_y1 = _spearman_rank_correlation(data[0])
    correlation_y2 = _spearman_rank_correlation(data[1])
    if correlation_y1.size == 0:
        return np.array([0.0])
    return np.array([np.max(np.abs(correlation_y1 - correlation_y2))])


def bootstrap_ci_np(
    data: tuple[np.ndarray, np.ndarray],
    statistic: Callable,
    n_resamples: int = 4,
    confidence_level: float = 0.95,
    alternative: str = "two-sided",
    axis: int = 0,
    random_state: int | None = None,
) -> dict[str, dict[str, float]]:
    """Compute a percentile bootstrap interval by independently resampling rows."""
    del axis  # The public statistic receives the conventional sample axis.
    x1, x2 = (np.asarray(data[0]), np.asarray(data[1]))
    if x1.ndim == 1:
        x1 = x1[:, None]
    if x2.ndim == 1:
        x2 = x2[:, None]
    if x1.shape[1:] != x2.shape[1:]:
        raise ValueError("Input arrays must have matching feature dimensions")

    rng = np.random.default_rng(random_state)
    values = []
    for _ in range(n_resamples):
        indices1 = rng.integers(0, len(x1), len(x1))
        indices2 = rng.integers(0, len(x2), len(x2))
        value = np.asarray(statistic(x1[indices1], x2[indices2], axis=0))
        values.append(float(value.reshape(-1)[0]))
    boot_stats = np.asarray(values)
    alpha = 1.0 - confidence_level
    if alternative == "two-sided":
        low, high = np.quantile(boot_stats, [alpha / 2, 1 - alpha / 2])
    elif alternative == "less":
        low, high = -np.inf, np.quantile(boot_stats, confidence_level)
    elif alternative == "greater":
        low, high = np.quantile(boot_stats, 1 - confidence_level), np.inf
    else:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
    return {"confidence_interval": {"low": float(low), "high": float(high)}}
