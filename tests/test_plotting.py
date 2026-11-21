"""Tests for appstat.plotting."""

import matplotlib

matplotlib.use("Agg")  # no window: tests only inspect the figure objects

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from iminuit import Minuit  # noqa: E402
from iminuit.cost import LeastSquares  # noqa: E402

from appstat.plotting import fit_info_text, plot_chi2_distribution, plot_fit, plot_pulls  # noqa: E402

X = np.arange(10.0)
Y = 3.6 + 0.3 * X + 0.2 * np.sin(X)
SIGMA = np.full(10, 0.4)


def line(x, a, b):
    return a + b * x


@pytest.fixture
def fit():
    """A real Minuit fit, made here so that these tests do not depend on appstat.fitting."""
    minuit = Minuit(LeastSquares(X, Y, SIGMA, line), a=1.0, b=0.0)
    minuit.migrad()
    return minuit


@pytest.fixture
def ax():
    figure, axes = plt.subplots()
    yield axes
    plt.close(figure)


def test_fit_info_text_names_chi2_and_every_parameter(fit):
    # THEN
    text = fit_info_text(fit)

    # EXPECT
    assert f"{fit.fval:.1f}" in text
    assert "a" in text and "b" in text


def test_fit_info_text_gives_the_smaller_error_more_decimals(fit):
    # IF the error on b is smaller than the error on a
    assert fit.errors["b"] < fit.errors["a"]

    # THEN its line needs more decimals
    lines = fit_info_text(fit).splitlines()
    line_a = next(entry for entry in lines if entry.startswith("a"))
    line_b = next(entry for entry in lines if entry.startswith("b"))

    # EXPECT
    assert len(line_b) > len(line_a)


def test_plot_fit_returns_the_axes_and_draws_data_and_curve(ax, fit):
    # THEN
    returned = plot_fit(ax, X, Y, SIGMA, line, fit)

    # EXPECT
    assert returned is ax
    assert len(ax.containers) >= 1  # the error bars
    assert len(ax.lines) >= 1  # the fitted curve


def test_plot_chi2_distribution_draws_histogram_and_curve(ax):
    # IF chi2 values from many fits with 5 degrees of freedom
    values = np.random.default_rng(0).chisquare(5, size=1000)

    # THEN
    returned = plot_chi2_distribution(ax, values, ndof=5)

    # EXPECT
    assert returned is ax
    assert len(ax.patches) > 0  # the histogram
    assert len(ax.lines) >= 1  # the chi2 curve


def test_plot_chi2_distribution_rejects_nonpositive_ndof(ax):
    # THEN / EXPECT
    with pytest.raises(ValueError):
        plot_chi2_distribution(ax, np.array([1.0, 2.0]), ndof=0)


def test_plot_pulls_colours_only_the_large_pulls_red(ax):
    # IF one pull beyond ±2
    pulls = np.array([0.5, -1.0, 2.5, 1.9])

    # THEN
    plot_pulls(ax, np.arange(4.0), pulls)

    # EXPECT
    red = [bar.get_facecolor() == matplotlib.colors.to_rgba("tab:red") for bar in ax.patches]
    assert red == [False, False, True, False]
