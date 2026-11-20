"""Tests for appstat.stats. Remove the skip marker once the function exists."""

import pytest

# import numpy as np; from scipy import stats  (needed once you fill the tests in)
# from appstat.stats import weighted_mean, chi2_pvalue, propagate_mc


@pytest.mark.skip(reason="TODO: implement weighted_mean")
def test_weighted_mean_matches_hand_calculation():
    # x = [1, 3], sigma = [1, 1]  ->  mean 2, error 1/sqrt(2)
    # mean, err = weighted_mean(np.array([1.0, 3.0]), np.array([1.0, 1.0]))
    # assert mean == pytest.approx(2.0)
    # assert err == pytest.approx(1 / np.sqrt(2))
    ...


@pytest.mark.skip(reason="TODO: implement chi2_pvalue")
def test_chi2_pvalue_matches_scipy():
    # chi2 = 10, ndof = 8: compare your function with stats.chi2.sf(10, 8)
    ...


@pytest.mark.skip(reason="TODO: implement propagate_mc")
def test_error_propagation_agrees_with_monte_carlo():
    # f = L * W with L = 3.5 +- 0.4, W = 0.8 +- 0.2 (ErrorPropagation notebook):
    # analytic sigma_f = sqrt((W sL)^2 + (L sW)^2) vs. the width of a large MC sample
    ...
