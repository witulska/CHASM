"""Classical statistical tests used as baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import levene, ttest_ind


def Welch_test(data1: np.ndarray, data2: np.ndarray) -> Any:
    """Run Welch's t-test for two independent samples."""
    return ttest_ind(data1, data2, equal_var=False)


def Levene_test(
    data1: np.ndarray, data2: np.ndarray, center: str = "median"
) -> Any:
    """Run Levene's test for equality of variances."""
    return levene(data1, data2, center=center)
