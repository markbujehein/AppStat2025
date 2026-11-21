"""Tests for the counting helpers in appstat.stats: integer histogram, Pearson chi2, skewness."""

import numpy as np
import pytest
from scipy import stats

from appstat.stats import binomial_pmf, integer_histogram, pearson_chi2, skewness_significance


def test_integer_histogram_counts_each_integer_in_its_own_bin():
    # IF
    values = np.array([0, 1, 1, 3])

    # THEN
    centres, counts = integer_histogram(values, n_max=3)

    # EXPECT one bin per integer 0..n_max, empty bins included
    np.testing.assert_array_equal(centres, [0, 1, 2, 3])
    np.testing.assert_array_equal(counts, [1, 2, 0, 1])


def test_integer_histogram_rejects_values_outside_the_range():
    # THEN / EXPECT
    with pytest.raises(ValueError):
        integer_histogram(np.array([0, 5]), n_max=3)


def test_pearson_chi2_matches_hand_calculation():
    # IF two bins: sum of (obs - exp)^2 / obs = 4/10 + 4/20
    observed = np.array([10.0, 20.0])
    expected = np.array([12.0, 18.0])

    # THEN
    chi2, ndof = pearson_chi2(observed, expected)

    # EXPECT no fitted parameters, so every bin counts
    assert chi2 == pytest.approx(0.6)
    assert ndof == 2


def test_pearson_chi2_skips_empty_bins_and_subtracts_fitted_parameters():
    # IF one bin has no entries (dividing by zero would be a cardinal sin)
    observed = np.array([0.0, 10.0, 20.0, 30.0])
    expected = np.array([0.5, 10.0, 20.0, 30.0])

    # THEN
    chi2, ndof = pearson_chi2(observed, expected, n_params=1)

    # EXPECT the empty bin neither adds to chi2 nor to the degrees of freedom
    assert chi2 == 0.0
    assert ndof == 3 - 1


def test_pearson_chi2_rejects_shape_mismatch():
    # THEN / EXPECT
    with pytest.raises(ValueError):
        pearson_chi2(np.ones(3), np.ones(4))


def test_skewness_significance_is_zero_for_symmetric_data():
    # IF
    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    # THEN
    skewness, _ = skewness_significance(values)

    # EXPECT
    assert skewness == pytest.approx(0.0, abs=1e-12)


def test_skewness_significance_uncertainty_is_about_sqrt_6_over_n():
    # IF many numbers
    values = np.random.default_rng(1).normal(size=10_000)

    # THEN
    _, sigma = skewness_significance(values)

    # EXPECT the known large-sample uncertainty of a skewness
    assert sigma == pytest.approx(np.sqrt(6 / 10_000), rel=0.01)


def test_binomial_pmf_matches_scipy_for_integer_n():
    # IF
    x = np.arange(0, 21)

    # THEN
    ours = binomial_pmf(x, 20, 0.2)

    # EXPECT
    np.testing.assert_allclose(ours, stats.binom.pmf(x, 20, 0.2), rtol=1e-10)


def test_binomial_pmf_accepts_a_non_integer_number_of_trials():
    # IF a fit tries n = 19.7, which scipy's binomial refuses (it returns nan)
    x = np.arange(0, 20)

    # THEN
    values = binomial_pmf(x, 19.7, 0.2)

    # EXPECT finite probabilities that still add up to about one
    assert np.all(np.isfinite(values))
    assert values.sum() == pytest.approx(1.0, abs=0.05)
