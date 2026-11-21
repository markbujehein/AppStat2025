"""Tests for appstat.stats.summary_statistics and appstat.stats.leave_one_out_slope_change."""

import numpy as np
import pytest

from appstat.stats import leave_one_out_slope_change, summary_statistics

# Anscombe's first data set
X1 = np.array([10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0])
Y1 = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])


def test_summary_statistics_of_an_exact_line():
    # IF y = 3 + 0.5 x exactly
    x = np.arange(6.0)

    # THEN
    result = summary_statistics(x, 3 + 0.5 * x)

    # EXPECT
    assert result["slope"] == pytest.approx(0.5)
    assert result["intercept"] == pytest.approx(3.0)
    assert result["correlation"] == pytest.approx(1.0)


def test_summary_statistics_matches_anscombes_published_numbers():
    # IF the first Anscombe set
    # THEN
    result = summary_statistics(X1, Y1)

    # EXPECT the values quoted in the exercise
    assert result["mean_x"] == pytest.approx(9.0)
    assert result["var_x"] == pytest.approx(10.0)
    assert result["mean_y"] == pytest.approx(7.5, abs=0.01)
    assert result["var_y"] == pytest.approx(3.75, abs=0.01)
    assert result["correlation"] == pytest.approx(0.816, abs=0.001)
    assert result["slope"] == pytest.approx(0.5, abs=0.001)
    assert result["intercept"] == pytest.approx(3.0, abs=0.01)


def test_summary_statistics_rejects_shape_mismatch():
    # THEN / EXPECT
    with pytest.raises(ValueError):
        summary_statistics(X1, Y1[:-1])


def test_summary_statistics_rejects_equal_x():
    # IF every x is the same, no line can be fitted
    # THEN / EXPECT
    with pytest.raises(ValueError):
        summary_statistics(np.full(5, 8.0), np.arange(5.0))


def test_leave_one_out_is_zero_for_an_exact_line():
    # IF the points lie exactly on a line, no single point matters
    x = np.arange(8.0)

    # THEN
    change = leave_one_out_slope_change(x, 2.0 * x + 1.0)

    # EXPECT
    assert change == pytest.approx(np.zeros(8), abs=1e-9)


def test_leave_one_out_finds_the_outlier():
    # IF one point sits far above an otherwise exact line
    x = np.arange(10.0)
    y = 0.5 * x + 3.0
    y[7] += 5.0

    # THEN
    change = leave_one_out_slope_change(x, y)

    # EXPECT leaving it out moves the slope the most, and back towards the true 0.5
    assert np.argmax(np.abs(change)) == 7
    assert change[7] < 0


def test_leave_one_out_is_nan_when_the_remaining_x_are_equal():
    # IF all x but one are 8 (Anscombe set IV)
    x = np.array([8.0, 8.0, 8.0, 19.0])
    y = np.array([6.0, 7.0, 8.0, 12.0])

    # THEN
    change = leave_one_out_slope_change(x, y)

    # EXPECT only leaving out x = 19 makes the fit impossible
    assert np.isnan(change[3])
    assert not np.isnan(change[:3]).any()


def test_leave_one_out_rejects_shape_mismatch():
    # THEN / EXPECT
    with pytest.raises(ValueError):
        leave_one_out_slope_change(X1, Y1[:-1])
