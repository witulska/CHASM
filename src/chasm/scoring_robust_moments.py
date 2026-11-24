"""Robust and classical moment statistics."""

from __future__ import annotations

from typing import Optional

import numpy as np
from astropy.stats import biweight_midvariance
from scipy.stats import kurtosis, skew


def _robust_mean(y: np.ndarray) -> float:
    """Compute the median as a robust location statistic."""
    return float(np.median(y))


def robust_mean(y: np.ndarray, axis: Optional[int] = None) -> float | np.ndarray:
    """Compute the robust mean, i.e. the median, along the selected axis."""
    if axis is None:
        return _robust_mean(np.ravel(y))
    return np.apply_along_axis(_robust_mean, axis=axis, arr=y)


def _robust_scale(y: np.ndarray) -> float:
    """Compute the robust scale using biweight midvariance."""
    return float(biweight_midvariance(y))


def robust_scale(y: np.ndarray, axis: Optional[int] = None) -> float | np.ndarray:
    """Compute the robust scale, optionally along the selected axis."""
    if axis is None:
        return _robust_scale(np.ravel(y))
    return np.apply_along_axis(_robust_scale, axis=axis, arr=y)


def _exact_skewness(y: np.ndarray) -> float:
    """Compute classical skewness with the unbiased correction."""
    return float(skew(y, bias=False))


def exact_skewness(y: np.ndarray, axis: Optional[int] = None) -> float | np.ndarray:
    """Compute classical skewness, optionally along the selected axis."""
    if axis is None:
        return _exact_skewness(np.ravel(y))
    return np.apply_along_axis(_exact_skewness, axis=axis, arr=y)


def _exact_kurtosis(y: np.ndarray) -> float:
    """Compute classical kurtosis."""
    return float(kurtosis(y))


def exact_kurtosis(y: np.ndarray, axis: Optional[int] = None) -> float | np.ndarray:
    """Compute classical kurtosis, optionally along the selected axis."""
    if axis is None:
        return _exact_kurtosis(np.ravel(y))
    return np.apply_along_axis(_exact_kurtosis, axis=axis, arr=y)


def _log_difference(first: np.ndarray | float, second: np.ndarray | float) -> np.ndarray:
    """Convert an absolute difference to the scale used by CHASM."""
    difference = np.abs(np.asarray(first) - np.asarray(second))
    return np.array([np.min(-np.log(difference + np.finfo(float).eps))])


def compute_robust_mean_difference(
    y1: np.ndarray, y2: np.ndarray, axis: Optional[int] = None
) -> np.ndarray:
    """Compute the difference of robust means."""
    return _log_difference(robust_mean(y1, axis=axis), robust_mean(y2, axis=axis))


def compute_robust_scale_difference(
    y1: np.ndarray, y2: np.ndarray, axis: Optional[int] = None
) -> np.ndarray:
    """Compute the difference of robust scales."""
    return _log_difference(robust_scale(y1, axis=axis), robust_scale(y2, axis=axis))


def compute_exact_kurtosis_difference(
    y1: np.ndarray, y2: np.ndarray, axis: Optional[int] = None
) -> np.ndarray:
    """Compute the difference of classical kurtoses."""
    return _log_difference(exact_kurtosis(y1, axis=axis), exact_kurtosis(y2, axis=axis))


def compute_exact_skewness_difference(
    y1: np.ndarray, y2: np.ndarray, axis: Optional[int] = None
) -> np.ndarray:
    """Compute the difference of classical skewness values."""
    return _log_difference(exact_skewness(y1, axis=axis), exact_skewness(y2, axis=axis))
