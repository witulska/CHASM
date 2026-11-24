"""Kernel-density divergence statistics and bootstrap confidence intervals."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.neighbors import KernelDensity


def Silverman_rule(x: np.ndarray) -> float:
    """Return Silverman's bandwidth estimate with a positive fallback."""
    values = np.asarray(x, dtype=float).ravel()
    bandwidth = (4 * np.std(values) ** 5 / (3 * max(len(values), 1))) ** 0.2
    return float(max(bandwidth, np.finfo(float).eps))


def ksdensity(y1: np.ndarray, y2: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate two one-dimensional densities on a shared grid."""
    values1 = np.asarray(y1, dtype=float).ravel()
    values2 = np.asarray(y2, dtype=float).ravel()
    values = np.concatenate([values1, values2])
    left, right = float(np.nanmin(values)), float(np.nanmax(values))
    if not np.isfinite(left) or not np.isfinite(right):
        return np.array([np.nan]), np.array([np.nan]), np.nan
    if left == right:
        left -= 0.5
        right += 0.5
    points = np.linspace(left, right, 512)
    step = float(points[1] - points[0])
    try:
        kde1 = KernelDensity(kernel="gaussian", bandwidth=Silverman_rule(values1)).fit(values1[:, None])
        density1 = np.exp(kde1.score_samples(points[:, None]))
        kde2 = KernelDensity(kernel="gaussian", bandwidth=Silverman_rule(values2)).fit(values2[:, None])
        density2 = np.exp(kde2.score_samples(points[:, None]))
    except (ValueError, FloatingPointError):
        return np.full(points.shape, np.nan), np.full(points.shape, np.nan), step
    return density1, density2, step


def _entropy(data: np.ndarray, base: float = np.e) -> float:
    values = np.asarray(data, dtype=float).ravel()
    if values.size == 0:
        return 0.0
    counts = np.histogram(values, bins=max(int(np.sqrt(values.size)), 1))[0]
    probabilities = counts[counts > 0] / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)) / np.log(base))


def entropy(y: np.ndarray, axis: int | None = None) -> float | np.ndarray:
    """Compute histogram entropy, optionally column-wise."""
    if axis is None:
        return _entropy(y)
    return np.apply_along_axis(_entropy, axis=axis, arr=np.asarray(y))


def compute_entropy_difference(y1: np.ndarray, y2: np.ndarray, axis: int | None = None) -> np.ndarray:
    difference = np.abs(entropy(y1, axis=axis) - entropy(y2, axis=axis))
    return np.array([np.min(-np.log(difference + np.finfo(float).eps))])


def _bhattacharyya_distance(y1: np.ndarray, y2: np.ndarray, alpha: float = 0.5) -> float:
    density1, density2, step = ksdensity(y1, y2)
    value = np.sum(density1**alpha * density2 ** (1 - alpha) * step)
    return float(-np.log(value + np.finfo(float).eps))


def _column_pairs(y1: np.ndarray, y2: np.ndarray, function: Callable) -> np.ndarray:
    values1, values2 = np.asarray(y1), np.asarray(y2)
    if values1.ndim == 1:
        return np.array([function(values1, values2)])
    if values1.shape != values2.shape:
        raise ValueError("y1 and y2 must have the same shape")
    return np.asarray([function(values1[:, i], values2[:, i]) for i in range(values1.shape[1])])


def bhattacharyya_distance(*data: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return _column_pairs(data[0], data[1], lambda a, b: _bhattacharyya_distance(a, b, alpha))


def compute_max_bhattacharyya_distance(y1: np.ndarray, y2: np.ndarray, axis: int | None = None) -> np.ndarray:
    del axis
    return np.array([np.nanmax(bhattacharyya_distance(y1, y2))])


def _jeffrey_distance(y1: np.ndarray, y2: np.ndarray, alpha: float = 0.5) -> float:
    del alpha
    density1, density2, step = ksdensity(y1, y2)
    return float(np.sum((np.sqrt(density1) - np.sqrt(density2)) ** 2 * step))


def jeffrey_distance(*data: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return _column_pairs(data[0], data[1], lambda a, b: _jeffrey_distance(a, b, alpha))


def compute_max_jeffrey_distance(y1: np.ndarray, y2: np.ndarray, axis: int | None = None) -> np.ndarray:
    del axis
    return np.array([np.nanmax(jeffrey_distance(y1, y2))])


def _kl_divergence(y1: np.ndarray, y2: np.ndarray) -> float:
    values = np.concatenate([np.asarray(y1).ravel(), np.asarray(y2).ravel()])
    bins = np.histogram_bin_edges(values, bins="auto")
    p = np.histogram(y1, bins=bins)[0].astype(float) + 1e-12
    q = np.histogram(y2, bins=bins)[0].astype(float) + 1e-12
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def kl_divergence(*data: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    del alpha
    return _column_pairs(data[0], data[1], _kl_divergence)


def compute_max_kl_divergence(y1: np.ndarray, y2: np.ndarray, axis: int | None = None) -> np.ndarray:
    del axis
    return np.array([np.nanmax(kl_divergence(y1, y2))])


def bootstrap_ci_divergences(
    data: tuple[np.ndarray, np.ndarray],
    statistic: Callable,
    n_resamples: int = 4,
    confidence_level: float = 0.95,
    alternative: str = "two-sided",
    axis: int = 0,
    random_state: int | None = None,
) -> dict[str, dict[str, float]]:
    """Compute a percentile bootstrap interval for a divergence statistic."""
    del axis
    x1, x2 = np.asarray(data[0]), np.asarray(data[1])
    if x1.ndim == 1:
        x1 = x1[:, None]
    if x2.ndim == 1:
        x2 = x2[:, None]
    rng = np.random.default_rng(random_state)
    values = []
    for _ in range(n_resamples):
        sample1 = x1[rng.integers(0, len(x1), len(x1))]
        sample2 = x2[rng.integers(0, len(x2), len(x2))]
        values.append(float(np.asarray(statistic(sample1, sample2, axis=0)).reshape(-1)[0]))
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
