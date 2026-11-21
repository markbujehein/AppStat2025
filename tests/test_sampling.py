"""Tests for random sampling and the shapes used in Week 3 of appstat.stats."""

import numpy as np
import pytest

from appstat.stats import (
    accept_reject,
    box_integral,
    power_exponential,
    power_law,
    two_sample_chi2,
    vacation_line,
)


def test_accept_reject_returns_n_points_inside_the_range():
    # IF f(x) = 2x on [0, 1], bounded by 2
    rng = np.random.default_rng(1)

    # THEN
    samples, n_tries = accept_reject(lambda x: 2 * x, 0.0, 1.0, 2.0, 1000, rng)

    # EXPECT exactly n points, in range, and at least as many tries as points
    assert samples.shape == (1000,)
    assert samples.min() >= 0.0 and samples.max() <= 1.0
    assert n_tries >= 1000


def test_accept_reject_follows_the_function():
    # IF f(x) = 2x, whose mean is 2/3
    rng = np.random.default_rng(2)

    # THEN
    samples, _ = accept_reject(lambda x: 2 * x, 0.0, 1.0, 2.0, 20_000, rng)

    # EXPECT the mean within a few standard errors (sigma = sqrt(1/18) / sqrt(n))
    assert samples.mean() == pytest.approx(2 / 3, abs=4 * np.sqrt(1 / 18 / 20_000))


def test_accept_reject_efficiency_is_the_area_under_f_over_the_box():
    # IF f(x) = 2x in the box [0,1]x[0,2]: area under f is 1, box area is 2
    rng = np.random.default_rng(3)

    # THEN
    samples, n_tries = accept_reject(lambda x: 2 * x, 0.0, 1.0, 2.0, 20_000, rng)

    # EXPECT an efficiency of 50 %
    assert len(samples) / n_tries == pytest.approx(0.5, abs=0.02)


def test_accept_reject_is_reproducible_with_the_same_seed():
    # THEN
    a, _ = accept_reject(np.sin, 0.0, np.pi, 1.0, 100, np.random.default_rng(7))
    b, _ = accept_reject(np.sin, 0.0, np.pi, 1.0, 100, np.random.default_rng(7))

    # EXPECT
    np.testing.assert_array_equal(a, b)


def test_accept_reject_rejects_a_ceiling_below_the_function():
    # IF the box is too low: f(x) = 2x reaches 2 but the ceiling is 1
    # THEN / EXPECT the silently wrong distribution is refused
    with pytest.raises(ValueError):
        accept_reject(lambda x: 2 * x, 0.0, 1.0, 1.0, 1000, np.random.default_rng(0))


def test_accept_reject_rejects_a_bad_range_or_count():
    # THEN / EXPECT
    with pytest.raises(ValueError):
        accept_reject(np.sin, 1.0, 0.0, 1.0, 10, np.random.default_rng(0))
    with pytest.raises(ValueError):
        accept_reject(np.sin, 0.0, 1.0, 1.0, 0, np.random.default_rng(0))


def test_box_integral_matches_a_known_area():
    # IF 50 % of the points are accepted in a box of area 2
    integral, error = box_integral(500, 1000, 2.0)

    # EXPECT integral 1, with the binomial error 2 * sqrt(0.5 * 0.5 / 1000)
    assert integral == pytest.approx(1.0)
    assert error == pytest.approx(2 * np.sqrt(0.25 / 1000))


def test_two_sample_chi2_is_zero_for_identical_histograms():
    # THEN
    chi2, ndof = two_sample_chi2(np.array([5, 10, 20]), np.array([5, 10, 20]))

    # EXPECT
    assert chi2 == 0.0
    assert ndof == 3


def test_two_sample_chi2_matches_hand_calculation_and_skips_empty_bins():
    # IF bins (10, 20) vs (14, 16), and a bin empty in both: (10-14)^2/24 + (20-16)^2/36
    chi2, ndof = two_sample_chi2(np.array([10, 20, 0]), np.array([14, 16, 0]))

    # EXPECT the empty bin neither adds to chi2 nor to ndof
    assert chi2 == pytest.approx(16 / 24 + 16 / 36)
    assert ndof == 2


def test_two_sample_chi2_rejects_shape_mismatch():
    # THEN / EXPECT
    with pytest.raises(ValueError):
        two_sample_chi2(np.ones(3), np.ones(4))


def test_vacation_line_is_a_straight_line_before_the_vacation():
    # IF the vacation starts at day 40
    x = np.array([0.0, 10.0, 39.0])

    # THEN
    y = vacation_line(x, 100.0, 2.0, 14.0)

    # EXPECT offset + slope * x
    np.testing.assert_allclose(y, 100 + 2 * x)


def test_vacation_line_is_flat_during_the_vacation():
    # THEN
    y = vacation_line(np.array([40.0, 47.0, 54.0]), 100.0, 2.0, 14.0)

    # EXPECT constant use stops: the value at the start of the vacation
    np.testing.assert_allclose(y, 100 + 2 * 40)


def test_vacation_line_continues_after_the_vacation_shifted_by_its_length():
    # THEN
    y = vacation_line(np.array([60.0]), 100.0, 2.0, 14.0)

    # EXPECT the line, delayed by the 14 days of vacation
    np.testing.assert_allclose(y, 100 + 2 * (60 - 14))


def test_vacation_line_accepts_a_zero_length_vacation():
    # THEN
    x = np.linspace(0, 100, 11)

    # EXPECT no vacation is the plain line
    np.testing.assert_allclose(vacation_line(x, 5.0, 3.0, 0.0), 5 + 3 * x)


def test_power_law_is_N_at_x_equal_one():
    # THEN / EXPECT
    assert power_law(1.0, 500.0, 2.5) == pytest.approx(500.0)
    assert power_law(2.0, 800.0, 2.0) == pytest.approx(200.0)


def test_power_exponential_reduces_to_its_two_parts():
    # IF
    x = np.array([1.0, 5.0, 20.0])

    # THEN / EXPECT only the power law when f = 1, only the exponential when f = 0
    np.testing.assert_allclose(power_exponential(x, 10.0, 1.0, 2.0, 0.1), power_law(x, 10.0, 2.0))
    np.testing.assert_allclose(
        power_exponential(x, 10.0, 0.0, 2.0, 0.1), 10.0 * np.exp(-0.1 * x) / 0.1
    )
