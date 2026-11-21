"""Tests for calibration helpers: profile_x (appstat.stats) and corrections (appstat.models)."""

import numpy as np
import pytest

from appstat.models import apply_corrections, offset_exponential, parabola
from appstat.stats import profile_x


def test_profile_x_gives_the_mean_and_its_uncertainty_per_bin():
    # IF two bins on (0, 2): x < 1 has y = 1, 3 and x >= 1 has y = 10, 12, 14
    x = np.array([0.2, 0.4, 1.2, 1.4, 1.6])
    y = np.array([1.0, 3.0, 10.0, 12.0, 14.0])

    # THEN
    centres, means, errors = profile_x(x, y, 2, (0, 2))

    # EXPECT the bin centres, the means, and std / sqrt(N)
    np.testing.assert_allclose(centres, [0.5, 1.5])
    np.testing.assert_allclose(means, [2.0, 12.0])
    np.testing.assert_allclose(errors, [np.std([1, 3], ddof=1) / np.sqrt(2), 2 / np.sqrt(3)])


def test_profile_x_drops_bins_with_fewer_than_two_entries():
    # IF the second bin has one entry
    centres, _, _ = profile_x(np.array([0.2, 0.4, 1.5]), np.array([1.0, 2.0, 3.0]), 2, (0, 2))

    # EXPECT only the first bin is left, as no spread can be computed from one value
    np.testing.assert_allclose(centres, [0.5])


def test_profile_x_can_ignore_outliers_in_y():
    # IF one wild value of y
    x = np.array([0.2, 0.4, 0.6])
    y = np.array([1.0, 3.0, 1000.0])

    # THEN
    _, means, _ = profile_x(x, y, 1, (0, 1), y_range=(-10, 10))

    # EXPECT the mean without it
    np.testing.assert_allclose(means, [2.0])


def test_parabola_and_offset_exponential_match_their_formulas():
    # THEN / EXPECT
    assert parabola(2.0, 1.0, 2.0, 3.0) == pytest.approx(1 + 2 * 2 + 3 * 4)
    assert offset_exponential(2.0, 1.0, 2.0, 0.5) == pytest.approx(1 + 2 * np.exp(1.0))


def test_apply_corrections_divides_out_each_relative_bias():
    # IF a value 50 % too high is corrected, then another effect of 100 %: 150 / 1.5 / 2
    corrected = apply_corrections(np.array([150.0]), np.array([0.5]), np.array([1.0]))

    # EXPECT
    np.testing.assert_allclose(corrected, [50.0])


def test_apply_corrections_with_no_correction_changes_nothing():
    # THEN / EXPECT
    np.testing.assert_allclose(apply_corrections(np.array([3.0, 4.0])), [3.0, 4.0])
