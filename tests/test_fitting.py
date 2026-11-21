"""Tests for appstat.fitting."""

import numpy as np
import pytest
from iminuit import cost

from appstat.fitting import fit_least_squares, polynomial_chi2, pulls


def line(x, a, b):
    return a + b * x


def test_fit_recovers_exact_line():
    # IF the data lie exactly on a line
    x = np.arange(10.0)
    y = 1.0 + 2.0 * x
    sigma = np.full(10, 0.5)

    # THEN
    result = fit_least_squares(line, x, y, sigma, start={"a": 0.0, "b": 1.0})

    # EXPECT
    assert result.values["a"] == pytest.approx(1.0, abs=1e-6)
    assert result.values["b"] == pytest.approx(2.0, abs=1e-6)
    assert result.fval == pytest.approx(0.0, abs=1e-6)


def test_fit_chi2_equals_manual_chi2():
    # IF
    x = np.arange(10.0)
    y = 2 * x + 1 + np.sin(x)
    sigma = np.full(10, 0.5)

    # THEN
    result = fit_least_squares(line, x, y, sigma, start={"a": 0.0, "b": 1.0})

    # EXPECT the chi2 at the fitted values, computed by hand and by iminuit
    manual = np.sum(((y - line(x, *result.values)) / sigma) ** 2)
    assert result.fval == pytest.approx(manual)
    assert cost.LeastSquares(x, y, sigma, line)(*result.values) == pytest.approx(manual)


def test_fit_ndof_is_points_minus_parameters():
    # IF 10 points and a model with 2 parameters
    x = np.arange(10.0)

    # THEN
    result = fit_least_squares(line, x, x, np.ones(10), start={"a": 0.0, "b": 1.0})

    # EXPECT
    assert result.ndof == 8


def test_fit_rejects_nonpositive_sigma():
    # IF
    x = np.arange(5.0)

    # THEN / EXPECT
    with pytest.raises(ValueError):
        fit_least_squares(line, x, x, np.array([1.0, 1.0, 0.0, 1.0, 1.0]), {"a": 0.0, "b": 1.0})


def test_fit_rejects_shape_mismatch():
    # IF y is shorter than x
    x = np.arange(5.0)

    # THEN / EXPECT
    with pytest.raises(ValueError):
        fit_least_squares(line, x, x[:4], np.ones(5), {"a": 0.0, "b": 1.0})


def test_fit_rejects_no_degrees_of_freedom():
    # IF two points and two parameters
    x = np.array([0.0, 1.0])

    # THEN / EXPECT
    with pytest.raises(ValueError):
        fit_least_squares(line, x, x, np.ones(2), {"a": 0.0, "b": 1.0})


def test_pulls_squared_sum_to_chi2():
    # IF
    x = np.arange(10.0)
    y = 2 * x + 1 + np.cos(3 * x)
    sigma = np.full(10, 0.4)
    result = fit_least_squares(line, x, y, sigma, start={"a": 0.0, "b": 1.0})

    # THEN
    p = pulls(line, result, x, y, sigma)

    # EXPECT
    assert np.sum(p**2) == pytest.approx(result.fval)


def test_polynomial_chi2_matches_minuit_row_by_row():
    # IF several data sets and a line fit
    rng = np.random.default_rng(1)
    x = np.arange(8.0)
    y = 1.0 + 0.5 * x + rng.normal(0, 0.3, size=(5, 8))

    # THEN
    direct = polynomial_chi2(y, x, sigma=0.3, degree=1)

    # EXPECT the direct solution to agree with Minuit on every row
    for row, chi2 in zip(y, direct, strict=True):
        fit = fit_least_squares(line, x, row, np.full(8, 0.3), {"a": 0.0, "b": 1.0})
        assert chi2 == pytest.approx(fit.fval, rel=1e-4)


def test_polynomial_chi2_rejects_negative_degree():
    # THEN / EXPECT
    with pytest.raises(ValueError):
        polynomial_chi2(np.zeros((2, 4)), np.arange(4.0), sigma=1.0, degree=-1)
