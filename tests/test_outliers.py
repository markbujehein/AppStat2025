"""Tests for outlier rejection and the weighted-mean chi2 in appstat.stats."""

import numpy as np
import pytest
from scipy import stats

from appstat.stats import chauvenet, chauvenet_nsigma, weighted_mean_chi2


def test_chauvenet_nsigma_for_one_number_is_the_two_sided_limit():
    # THEN
    nsigma = chauvenet_nsigma(1, p_global=0.05)

    # EXPECT the familiar 1.96 sigma
    assert nsigma == pytest.approx(1.95996, abs=1e-4)


def test_chauvenet_nsigma_keeps_all_n_numbers_inside_with_probability_one_minus_p():
    # IF 100 Gaussian numbers and a 5 % global probability of any outlier
    n, p_global = 100, 0.05

    # THEN
    nsigma = chauvenet_nsigma(n, p_global)

    # EXPECT all of them inside +- nsigma happens 95 % of the time
    inside_each = 1 - 2 * stats.norm.sf(nsigma)
    assert inside_each**n == pytest.approx(1 - p_global)


def test_chauvenet_rejects_a_far_outlier_and_keeps_the_rest():
    # IF 200 Gaussian numbers and one at 50 sigma
    values = np.append(np.random.default_rng(4).normal(0, 1, 200), 50.0)

    # THEN
    accepted = chauvenet(values, p_global=0.05)

    # EXPECT the outlier is rejected and nearly all the others are kept
    assert not accepted[-1]
    assert accepted.sum() >= 195


def test_chauvenet_keeps_everything_when_nothing_is_an_outlier():
    # THEN
    accepted = chauvenet(np.arange(10.0), p_global=0.05)

    # EXPECT
    assert accepted.all()


def test_chauvenet_does_not_divide_by_a_zero_width():
    # IF all values are equal
    accepted = chauvenet(np.full(5, 3.0))

    # EXPECT nothing is rejected, and nothing crashes
    assert accepted.all()


def test_chauvenet_does_not_change_its_input():
    # IF
    values = np.array([1.0, 1.1, 0.9, 1.0, 40.0])
    before = values.copy()

    # THEN
    chauvenet(values)

    # EXPECT
    np.testing.assert_array_equal(values, before)


def test_weighted_mean_chi2_matches_hand_calculation():
    # IF two values of unit uncertainty: the mean is 2, so chi2 = 1 + 1
    chi2, ndof = weighted_mean_chi2(np.array([1.0, 3.0]), np.array([1.0, 1.0]))

    # EXPECT one degree of freedom is used up by the mean
    assert chi2 == pytest.approx(2.0)
    assert ndof == 1


def test_weighted_mean_chi2_is_zero_for_equal_values():
    # THEN
    chi2, _ = weighted_mean_chi2(np.full(4, 2.5), np.array([0.1, 0.2, 0.3, 0.4]))

    # EXPECT
    assert chi2 == pytest.approx(0.0, abs=1e-12)
