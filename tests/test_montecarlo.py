"""Tests for Monte Carlo estimates in appstat.montecarlo: hyper-spheres and stratification."""

import numpy as np
import pytest
from scipy.special import gamma

from appstat.montecarlo import (
    equal_counts,
    estimate_ball_volume,
    plain_integral,
    proportional_counts,
    stratified_integral,
    stratum_stds,
    unit_ball_volume,
)


def test_unit_ball_volume_matches_the_known_cases():
    # THEN / EXPECT a line, a circle and a sphere
    assert unit_ball_volume(1) == pytest.approx(2.0)
    assert unit_ball_volume(2) == pytest.approx(np.pi)
    assert unit_ball_volume(3) == pytest.approx(4 * np.pi / 3)
    assert unit_ball_volume(4) == pytest.approx(np.pi**2 / 2)
    assert unit_ball_volume(6) == pytest.approx(np.pi**3 / 6)


def test_unit_ball_volume_is_the_gamma_function_formula():
    # THEN / EXPECT
    assert unit_ball_volume(7) == pytest.approx(np.pi**3.5 / gamma(4.5))


def test_estimate_ball_volume_of_a_circle_is_pi_within_its_error():
    # IF 200 000 points in the square around a circle
    volume, error = estimate_ball_volume(2, 200_000, np.random.default_rng(1))

    # THEN / EXPECT pi within four of its own standard deviations, and an error of about 0.004
    assert abs(volume - np.pi) < 4 * error
    assert error == pytest.approx(4 * np.sqrt(0.785 * 0.215 / 200_000), rel=0.05)


def test_estimate_ball_volume_in_one_dimension_is_exact():
    # IF every point of the line [-1, 1] is inside
    volume, error = estimate_ball_volume(1, 1000, np.random.default_rng(2))

    # THEN / EXPECT
    assert volume == 2.0 and error == 0.0


def test_estimate_ball_volume_is_reproducible_with_the_same_seed():
    # THEN / EXPECT
    a = estimate_ball_volume(3, 1000, np.random.default_rng(5))
    b = estimate_ball_volume(3, 1000, np.random.default_rng(5))
    assert a == b


def test_plain_integral_of_a_square_agrees_with_calculus():
    # IF the integral of x^2 from 0 to 3, which is 9
    value = plain_integral(lambda x: x**2, 0, 3, 200_000, np.random.default_rng(3))

    # THEN / EXPECT
    assert value == pytest.approx(9.0, abs=0.1)


def test_stratified_integral_of_a_constant_is_exact_whatever_the_random_numbers():
    # IF a constant function, so no sampling noise inside any stratum
    edges = np.linspace(0, 10, 11)
    value = stratified_integral(
        lambda x: np.full_like(x, 2.0), edges, np.full(10, 5), np.random.default_rng(4)
    )

    # THEN / EXPECT 2 * 10
    assert value == pytest.approx(20.0)


def test_stratified_integral_scatters_less_than_the_plain_one():
    # IF the same 1000 points either way, for a function that varies mostly at small x
    def func(x):
        return x / (x**2 + 1)

    rng = np.random.default_rng(6)
    edges = np.linspace(0, 10, 11)
    plain = [plain_integral(func, 0, 10, 1000, rng) for _ in range(300)]
    stratified = [stratified_integral(func, edges, np.full(10, 100), rng) for _ in range(300)]

    # EXPECT the stratified estimates cluster more tightly, around the exact log(101)/2
    assert np.std(stratified) < 0.7 * np.std(plain)
    assert np.mean(stratified) == pytest.approx(np.log(101) / 2, abs=0.02)


def test_stratum_stds_are_zero_for_a_constant_and_larger_where_the_function_varies():
    # THEN
    edges = np.linspace(0, 10, 6)
    assert np.all(
        stratum_stds(lambda x: np.ones_like(x), edges, 100, np.random.default_rng(1)) == 0
    )

    # EXPECT the spread of x / (x^2 + 1) is largest in the first stratum
    stds = stratum_stds(lambda x: x / (x**2 + 1), edges, 2000, np.random.default_rng(1))
    assert stds.argmax() == 0 and stds[0] > 5 * stds[-1]


def test_proportional_counts_follow_the_spread_and_add_up():
    # THEN / EXPECT twice the spread gets twice the points
    np.testing.assert_array_equal(proportional_counts(np.array([1.0, 3.0]), 8), [2, 6])


def test_proportional_counts_always_add_up_and_give_every_stratum_a_point():
    # IF rounding would lose points, and one stratum has no spread
    counts = proportional_counts(np.array([1.0, 1.0, 1.0, 0.0]), 10)

    # THEN / EXPECT
    assert counts.sum() == 10
    assert counts.min() >= 1


def test_proportional_counts_of_a_constant_function_fall_back_to_equal_shares():
    # IF every stratum has zero spread (a constant function)
    # THEN / EXPECT equal counts adding up to the total, and no endless loop
    counts = proportional_counts(np.zeros(4), 10)
    assert counts.sum() == 10
    assert counts.max() - counts.min() <= 1


def test_proportional_counts_reject_a_nan_spread():
    # IF a spread is nan (one pilot point gives no standard deviation)
    # THEN / EXPECT a clear error instead of an endless loop
    with pytest.raises(ValueError, match="spread"):
        proportional_counts(np.array([1.0, np.nan]), 10)


def test_estimate_ball_volume_with_no_hit_still_reports_an_uncertainty():
    # IF no random point lands inside (12 dimensions, 10 points)
    # THEN / EXPECT zero volume with a non-zero upper-limit error, not "0 +- 0"
    volume, error = estimate_ball_volume(12, 10, np.random.default_rng(1))
    assert volume == 0.0
    assert error > 0


def test_equal_counts_keep_the_remainder():
    # IF 10 points go over 7 strata
    # THEN / EXPECT all 10 are used, none of the strata differing by more than one
    counts = equal_counts(7, 10)
    assert counts.sum() == 10
    assert counts.max() - counts.min() == 1
