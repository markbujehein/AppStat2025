"""Tests for random sampling and histogram comparison in appstat.stats."""

import numpy as np
import pytest

from appstat.stats import accept_reject, box_integral, two_sample_chi2


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


def test_accept_reject_rejects_a_ceiling_below_a_broad_function():
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
