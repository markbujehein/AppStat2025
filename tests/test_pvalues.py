"""Tests for p-value helpers in appstat.stats: histogram chi2, dispersion and two-sample tests."""

import numpy as np
import pytest

from appstat.stats import (
    correlation_matrix,
    dispersion_chi2,
    multinomial_chi2,
    two_sample_pvalues,
)


def test_multinomial_chi2_matches_hand_calculation():
    # IF 100 throws, 60 and 40, and a fair coin: expected 50 each, (10^2 + 10^2) / 50
    chi2, ndof = multinomial_chi2(np.array([60, 40]), np.array([0.5, 0.5]))

    # EXPECT one degree of freedom: the total is fixed
    assert chi2 == pytest.approx(4.0)
    assert ndof == 1


def test_multinomial_chi2_skips_bins_that_cannot_occur():
    # IF one bin has probability zero and no entries
    chi2, ndof = multinomial_chi2(np.array([30, 0, 30]), np.array([0.5, 0.0, 0.5]))

    # EXPECT it adds nothing, and no degree of freedom
    assert chi2 == pytest.approx(0.0)
    assert ndof == 1


def test_multinomial_chi2_rejects_probabilities_that_do_not_sum_to_one():
    # THEN / EXPECT
    with pytest.raises(ValueError):
        multinomial_chi2(np.array([5, 5]), np.array([0.5, 0.6]))
    with pytest.raises(ValueError):
        multinomial_chi2(np.array([5, 5, 5]), np.array([0.5, 0.5]))


def test_dispersion_chi2_matches_hand_calculation():
    # IF counts 8 and 12 with mean 10: (4 + 4) / 10
    chi2, ndof = dispersion_chi2(np.array([8, 12]))

    # EXPECT one degree of freedom, as the mean was estimated
    assert chi2 == pytest.approx(0.8)
    assert ndof == 1


def test_dispersion_chi2_is_zero_for_equal_counts_and_rejects_no_counts():
    # THEN / EXPECT
    assert dispersion_chi2(np.array([10, 10, 10]))[0] == 0.0
    with pytest.raises(ValueError):
        dispersion_chi2(np.zeros(5))


def test_two_sample_pvalues_are_one_for_identical_samples():
    # IF the same numbers twice
    x = np.random.default_rng(1).normal(size=200)

    # THEN
    p_mean, p_chi2, p_ks = two_sample_pvalues(x, x, 20, (-5, 5))

    # EXPECT no test sees a difference
    assert (
        p_mean == pytest.approx(1.0) and p_chi2 == pytest.approx(1.0) and p_ks == pytest.approx(1.0)
    )


def test_two_sample_pvalues_see_a_shifted_mean():
    # IF a shift of 0.3 in the mean with 2000 numbers each
    rng = np.random.default_rng(2)

    # THEN
    p = two_sample_pvalues(rng.normal(0.3, 1, 2000), rng.normal(0, 1, 2000), 20, (-5, 5))

    # EXPECT all three tests reject
    assert max(p) < 1e-3


def test_two_sample_pvalues_of_a_wider_distribution_are_missed_by_the_mean_test():
    # IF the same mean but a width of 1.5
    rng = np.random.default_rng(3)

    # THEN
    p_mean, p_chi2, p_ks = two_sample_pvalues(
        rng.normal(0, 1.5, 2000), rng.normal(0, 1, 2000), 20, (-8, 8)
    )

    # EXPECT the mean test is blind, the other two see it
    assert p_mean > 0.01
    assert p_chi2 < 1e-3 and p_ks < 1e-3


def test_two_sample_pvalues_of_identical_distributions_reject_five_percent_of_the_time():
    # IF 300 experiments of two samples from the same Gaussian
    rng = np.random.default_rng(4)
    p = np.array(
        [
            two_sample_pvalues(rng.normal(size=100), rng.normal(size=100), 10, (-5, 5))
            for _ in range(300)
        ]
    )

    # THEN
    rejected = (p < 0.05).mean(axis=0)

    # EXPECT about 5 % for the mean test (within three standard deviations of 300 trials)
    assert 0.02 <= rejected[0] <= 0.09
    assert rejected[2] <= 0.09  # the KS test is conservative for small samples


def test_two_sample_pvalues_reject_empty_samples():
    # THEN / EXPECT
    with pytest.raises(ValueError):
        two_sample_pvalues(np.array([]), np.array([1.0, 2.0]), 10, (0, 5))


def test_two_sample_pvalues_reject_samples_of_different_size():
    # IF the histogram chi2 would compare counts of different totals
    # THEN / EXPECT an error, not a wrong p-value
    with pytest.raises(ValueError):
        two_sample_pvalues(np.arange(10.0), np.arange(20.0), 5, (0, 20))


def test_correlation_matrix_scales_the_covariance_by_the_uncertainties():
    # IF variances 4 and 1 with covariance 1: 1 / (2 * 1)
    corr = correlation_matrix(np.array([[4.0, 1.0], [1.0, 1.0]]))

    # THEN / EXPECT ones on the diagonal and 0.5 off it
    np.testing.assert_allclose(corr, [[1.0, 0.5], [0.5, 1.0]])
