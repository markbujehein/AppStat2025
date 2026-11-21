"""Tests for appstat.plotting. Skeleton: each test is marked xfail until its function exists.

``strict=True`` turns a passing test into a failure, the reminder to delete the marker.
"""

import matplotlib

matplotlib.use("Agg")  # no window: tests only inspect the figure objects

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from appstat.fitting import FitResult  # noqa: E402
from appstat.plotting import fit_info_text, plot_chi2_distribution, plot_fit, plot_pulls  # noqa: E402

skeleton = pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="skeleton")

RESULT = FitResult(
    values={"a": 3.6, "b": 0.3}, errors={"a": 0.2, "b": 0.02}, chi2=12.3, ndof=15, pvalue=0.66
)


def line(x, a, b):
    return a + b * x


@pytest.fixture
def ax():
    figure, axes = plt.subplots()
    yield axes
    plt.close(figure)


@skeleton
def test_fit_info_text_names_chi2_pvalue_and_every_parameter():
    # THEN
    text = fit_info_text(RESULT)

    # EXPECT
    assert "12.3" in text
    assert "0.66" in text
    assert "a" in text and "b" in text


@skeleton
def test_fit_info_text_gives_the_smaller_error_more_decimals():
    # THEN the line "b = 0.30 ± 0.02" needs more decimals than "a = 3.6 ± 0.2"
    lines = fit_info_text(RESULT).splitlines()
    line_a = next(entry for entry in lines if entry.startswith("a"))
    line_b = next(entry for entry in lines if entry.startswith("b"))

    # EXPECT
    assert len(line_b) > len(line_a)


@skeleton
def test_plot_fit_returns_the_axes_and_draws_data_and_curve(ax):
    # IF
    x = np.arange(10.0)

    # THEN
    returned = plot_fit(ax, x, line(x, 3.6, 0.3), np.full(10, 0.4), line, RESULT)

    # EXPECT
    assert returned is ax
    assert len(ax.containers) >= 1  # the error bars
    assert len(ax.lines) >= 1  # the fitted curve


@skeleton
def test_plot_chi2_distribution_draws_histogram_and_curve(ax):
    # IF chi2 values from many fits with 5 degrees of freedom
    values = np.random.default_rng(0).chisquare(5, size=1000)

    # THEN
    returned = plot_chi2_distribution(ax, values, ndof=5)

    # EXPECT
    assert returned is ax
    assert len(ax.patches) > 0  # the histogram
    assert len(ax.lines) >= 1  # the chi2 curve


@skeleton
def test_plot_chi2_distribution_rejects_nonpositive_ndof(ax):
    # THEN / EXPECT
    with pytest.raises(ValueError):
        plot_chi2_distribution(ax, np.array([1.0, 2.0]), ndof=0)


@skeleton
def test_plot_pulls_colours_only_the_large_pulls_red(ax):
    # IF one pull beyond ±2
    pulls = np.array([0.5, -1.0, 2.5, 1.9])

    # THEN
    plot_pulls(ax, np.arange(4.0), pulls)

    # EXPECT
    red = [bar.get_facecolor() == matplotlib.colors.to_rgba("tab:red") for bar in ax.patches]
    assert red == [False, False, True, False]
