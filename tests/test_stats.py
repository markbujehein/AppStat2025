"""Tests for appstat.stats. Remove the skip marker once the function exists."""

import numpy as np
import pytest
from scipy import stats

from appstat.stats import chi2_pvalue, propagate_mc, weighted_mean


def test_weighted_mean_matches_hand_calculation():
    # x = [1, 3], sigma = [1, 1]  ->  mean 2, error 1/sqrt(2)
    # IF
    x = np.array([1.0, 3.0])
    sigma = np.array([1.0, 1.0])

    # THEN
    mean, error = weighted_mean(x, sigma)

    # EXPECT
    assert mean == pytest.approx(2.0)
    assert error == pytest.approx(1 / np.sqrt(2))


def test_weighted_mean_matches_scipy_sem():
    # With the sample standard deviation as sigma for every point, the weighted mean is the
    # plain mean and its error is the standard error of the mean.
    # IF
    x = np.array([4.1, 3.8, 4.4, 4.0, 3.7, 4.2])
    sigma = np.full_like(x, np.std(x, ddof=1))

    # THEN
    mean, error = weighted_mean(x, sigma)

    # EXPECT
    assert mean == pytest.approx(np.mean(x))
    assert error == pytest.approx(stats.sem(x))


def test_weighted_mean_rejects_nonpositive_sigma():
    # IF
    x = np.array([1.0, 3.0])
    sigma = np.array([1.0, 0.0])

    # THEN / EXPECT
    with pytest.raises(ValueError):
        weighted_mean(x, sigma)


def test_weighted_mean_rejects_unequal_shapes():
    # IF
    x = np.array([1.0, 3.0, 5.0])
    sigma = np.array([1.0, 1.0])

    # THEN / EXPECT
    with pytest.raises(ValueError):
        weighted_mean(x, sigma)


def test_chi2_pvalue_matches_scipy():
    # IF
    chi2 = 10.0
    ndof = 8

    # THEN
    p_value = chi2_pvalue(chi2, ndof)

    # EXPECT
    assert p_value == pytest.approx(stats.chi2.sf(chi2, ndof))


def test_error_propagation_agrees_with_monte_carlo():
    # f = L * W with L = 3.5 +- 0.4, W = 0.8 +- 0.2 (ErrorPropagation notebook).
    # propagate_mc calls func(L_samples, W_samples) with one sample array per input.
    # IF
    means = np.array([3.5, 0.8])
    sigmas = np.array([0.4, 0.2])

    # THEN
    sigma_mc = propagate_mc(lambda length, width: length * width, means, sigmas)

    # EXPECT
    sigma_analytic = np.sqrt((means[1] * sigmas[0]) ** 2 + (means[0] * sigmas[1]) ** 2)
    assert sigma_mc == pytest.approx(sigma_analytic, rel=0.02)
