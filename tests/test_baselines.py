import numpy as np
import pytest

from baselines import first_difference_scores


def test_univariate():
    np.testing.assert_allclose(first_difference_scores(np.array([1.0, 3.0, 2.0, 2.0])), [0, 2, 1, 0])


def test_multivariate_max_over_columns():
    values = np.array([[0.0, 0.0], [1.0, -5.0], [1.5, -5.0]])
    np.testing.assert_allclose(first_difference_scores(values), [0, 5, 0.5])


def test_column_selection():
    values = np.array([[0.0, 0.0], [1.0, -5.0], [1.5, -5.0]])
    np.testing.assert_allclose(first_difference_scores(values, columns=[0]), [0, 1, 0.5])


def test_output_length_matches_input():
    values = np.random.default_rng(0).normal(size=(37, 4))
    assert first_difference_scores(values).shape == (37,)


def test_empty_and_single():
    assert first_difference_scores(np.zeros((0, 3))).shape == (0,)
    np.testing.assert_allclose(first_difference_scores(np.array([[4.0, 2.0]])), [0.0])


def test_rejects_3d():
    with pytest.raises(ValueError):
        first_difference_scores(np.zeros((2, 2, 2)))
