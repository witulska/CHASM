"""Shared random-data and simulation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def mdim_pareto_rv(beta: float, alpha: float, d: int, n: int) -> np.ndarray:
    """Generate the multivariate Pareto mixture used by the paper simulations."""
    if beta <= 0 or alpha <= 0 or d < 1 or n < 1:
        raise ValueError("beta, alpha, d and n must be positive")
    exponential = np.random.exponential(scale=1 / beta, size=(n, d))
    shape = 1 / alpha
    scale = 1 / alpha
    gamma = np.random.gamma(shape=shape, scale=scale, size=(n, 1))
    return exponential / gamma


def t_student_vector(
    dof: int,
    n: int,
    mean: np.ndarray | None = None,
    corr_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Generate the historical multivariate Student simulation sample.

    The multiplication by the chi-square sample is retained for compatibility
    with the original article experiments.
    """
    if dof <= 0 or n < 1:
        raise ValueError("dof and n must be positive")
    mean_array = np.zeros(2) if mean is None else np.asarray(mean, dtype=float)
    matrix = (
        np.array([[1.0, 0.5], [0.5, 1.0]])
        if corr_matrix is None
        else np.asarray(corr_matrix, dtype=float)
    )
    if matrix.shape != (len(mean_array), len(mean_array)):
        raise ValueError("corr_matrix must match the dimension of mean")
    normal = np.random.multivariate_normal(mean_array, matrix, n)
    chi = np.random.chisquare(df=dof, size=n)
    return normal * chi[:, None]


def generate_metadata(
    mc_trials: int,
    n: int,
    cp_detected: int,
    d: int,
    distribution1: str,
    distribution2: str,
    params1: Any,
    params2: Any,
) -> dict[str, Any]:
    """Build a serializable metadata record for an experiment."""
    return {
        "#MC_trials": mc_trials,
        "N": n,
        "cp_detected": cp_detected,
        "d": d,
        "distribution1": distribution1,
        "distribution2": distribution2,
        "params1": params1,
        "params2": params2,
    }


def generate_corr_matrix_for_sim(
    d: int, rho12: float = 0.9, rho13: float = 0.3, rho23: float = 0.2
) -> np.ndarray:
    """Create the correlation matrix convention used by the simulations."""
    if d < 1:
        raise ValueError("d must be positive")
    if d == 1:
        return np.eye(1)
    if d == 2:
        return np.array([[1.0, rho12], [rho12, 1.0]])
    matrix = np.eye(d)
    matrix[0, 1] = matrix[1, 0] = rho12
    matrix[0, 2] = matrix[2, 0] = rho13
    matrix[1, 2] = matrix[2, 1] = rho23
    return matrix
