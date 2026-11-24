import numpy as np

from chasm import (
    generate_corr_matrix_for_sim,
    mdim_pareto_rv,
    t_student_vector,
)


def test_pareto_and_student_shapes():
    assert mdim_pareto_rv(1.0, 0.4, 3, 12).shape == (12, 3)
    assert t_student_vector(5, 12, mean=np.zeros(3), corr_matrix=np.eye(3)).shape == (12, 3)


def test_correlation_matrix_dimensions():
    assert generate_corr_matrix_for_sim(1).shape == (1, 1)
    assert generate_corr_matrix_for_sim(2).shape == (2, 2)
    assert generate_corr_matrix_for_sim(5).shape == (5, 5)
