"""CHASM: Change-point Hypothesis Assessment for Significance in Multivariate data."""

from .main import (
    ARTICLE_METRICS,
    bootstraping_corr,
    bootstraping_divergences,
    bootstraping_one_dim_stats,
    chasm_bounds_main,
    chasm_main,
)
from .utils import (
    generate_corr_matrix_for_sim,
    generate_metadata,
    mdim_pareto_rv,
    t_student_vector,
)

__all__ = [
    "ARTICLE_METRICS",
    "bootstraping_corr",
    "bootstraping_divergences",
    "bootstraping_one_dim_stats",
    "chasm_bounds_main",
    "chasm_main",
    "generate_corr_matrix_for_sim",
    "generate_metadata",
    "mdim_pareto_rv",
    "t_student_vector",
]
