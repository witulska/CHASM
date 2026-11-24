"""Statystyki oparte na momentach L."""

from __future__ import annotations

from typing import Optional

import lmoments3 as lm
import numpy as np


def _l_mean(y: np.ndarray) -> float:
    """Compute the L-mean for one-dimensional data."""
    values = np.asarray(lm.lmom_ratios(y, nmom=1))
    return float(values.reshape(-1)[0])


def l_mean(y: np.ndarray, axis: Optional[int] = None) -> float | np.ndarray:
    """Compute the L-mean, optionally along the selected axis."""
    if axis is None:
        return _l_mean(np.ravel(y))
    return np.apply_along_axis(_l_mean, axis=axis, arr=y)


def _l_scale(y: np.ndarray) -> float:
    """Compute the L-scale for one-dimensional data."""
    values = np.asarray(lm.lmom_ratios(y, nmom=2))
    return float(values.reshape(-1)[1])


def l_scale(y: np.ndarray, axis: Optional[int] = None) -> float | np.ndarray:
    """Compute the L-scale, optionally along the selected axis."""
    if axis is None:
        return _l_scale(np.ravel(y))
    return np.apply_along_axis(_l_scale, axis=axis, arr=y)


def _l_skewness(y: np.ndarray) -> float:
    """Compute L-skewness for one-dimensional data."""
    values = np.asarray(lm.lmom_ratios(y, nmom=3))
    return float(values.reshape(-1)[2])


def l_skewness(y: np.ndarray, axis: Optional[int] = None) -> float | np.ndarray:
    """Compute L-skewness, optionally along the selected axis."""
    if axis is None:
        return _l_skewness(np.ravel(y))
    return np.apply_along_axis(_l_skewness, axis=axis, arr=y)


def _l_kurtosis(y: np.ndarray) -> float:
    """Compute L-kurtosis for one-dimensional data."""
    values = np.asarray(lm.lmom_ratios(y, nmom=4))
    return float(values.reshape(-1)[3])


def l_kurtosis(y: np.ndarray, axis: Optional[int] = None) -> float | np.ndarray:
    """Compute L-kurtosis, optionally along the selected axis."""
    if axis is None:
        return _l_kurtosis(np.ravel(y))
    return np.apply_along_axis(_l_kurtosis, axis=axis, arr=y)


def _log_difference(first: np.ndarray | float, second: np.ndarray | float) -> np.ndarray:
    """Convert an absolute difference to the scale used by CHASM."""
    difference = np.abs(np.asarray(first) - np.asarray(second))
    return np.array([np.min(-np.log(difference + np.finfo(float).eps))])


def compute_l_mean_difference(
    y1: np.ndarray, y2: np.ndarray, axis: Optional[int] = None
) -> np.ndarray:
    """Compute the normalized difference of L-means."""
    range1 = np.max(y1, axis=axis) - np.min(y1, axis=axis)
    range2 = np.max(y2, axis=axis) - np.min(y2, axis=axis)
    mean1 = l_mean(y1, axis=axis) / range1
    mean2 = l_mean(y2, axis=axis) / range2
    return _log_difference(mean1, mean2)


def compute_l_scale_difference(
    y1: np.ndarray, y2: np.ndarray, axis: Optional[int] = None
) -> np.ndarray:
    """Compute the difference of L-scales."""
    return _log_difference(l_scale(y1, axis=axis), l_scale(y2, axis=axis))


def compute_l_kurtosis_difference(
    y1: np.ndarray, y2: np.ndarray, axis: Optional[int] = None
) -> np.ndarray:
    """Compute the difference of L-kurtoses."""
    return _log_difference(l_kurtosis(y1, axis=axis), l_kurtosis(y2, axis=axis))


def compute_l_skewness_difference(
    y1: np.ndarray, y2: np.ndarray, axis: Optional[int] = None
) -> np.ndarray:
    """Compute the difference of L-skewness values."""
    return _log_difference(l_skewness(y1, axis=axis), l_skewness(y2, axis=axis))
