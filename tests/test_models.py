"""Tests for the model functions and exact curves in appstat.models."""

import numpy as np
import pytest
from scipy import integrate

from appstat.models import (
    exp_cos2_cdf,
    power_exponential,
    power_law,
    vacation_line,
    x_cos_x_cdf,
)


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


def test_x_cos_x_cdf_matches_numerical_integration():
    # IF the integral of x cos x from 0 to 1.2
    expected, _ = integrate.quad(lambda t: t * np.cos(t), 0, 1.2)

    # THEN / EXPECT the closed form agrees, and the full range gives pi/2 - 1
    assert x_cos_x_cdf(1.2) == pytest.approx(expected)
    assert x_cos_x_cdf(np.pi / 2) == pytest.approx(np.pi / 2 - 1)


def test_exp_cos2_cdf_matches_numerical_integration():
    # IF the integral of exp(-t/3) cos(t)^2 from 0 to 5
    expected, _ = integrate.quad(lambda t: np.exp(-t / 3) * np.cos(t) ** 2, 0, 5)

    # THEN / EXPECT the closed form agrees, and a far point gives the total integral
    assert exp_cos2_cdf(5.0) == pytest.approx(expected)
    total, _ = integrate.quad(lambda t: np.exp(-t / 3) * np.cos(t) ** 2, 0, 200, limit=500)
    assert exp_cos2_cdf(1e3) == pytest.approx(total, rel=1e-6)


def test_exp_cos2_cdf_works_on_an_array_of_bin_edges():
    # THEN
    values = exp_cos2_cdf(np.linspace(0, 20, 5))

    # EXPECT zero at the start, rising
    assert values[0] == pytest.approx(0.0)
    assert np.all(np.diff(values) > 0)
