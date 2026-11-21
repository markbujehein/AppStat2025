from collections.abc import Callable

from matplotlib.axes import Axes
lazy import numpy as np

from appstat.fitting import FitResult


def fit_info_text(result: FitResult) -> str:
    """
    Text for the "result of fit" box: chi-square, p-value and each parameter with its error.

    Each parameter is printed with as many decimals as its uncertainty justifies.

    Parameters
    ----------
    result : FitResult
        The fit to describe.

    Returns
    -------
    str
        Multi-line text, ready for ``ax.text`` (math written with ``$...$``).
    """
    raise NotImplementedError


def plot_fit(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    model: Callable[..., np.ndarray],
    result: FitResult,
) -> Axes:
    """
    Draw the data with error bars, the fitted curve and the fit info box.

    Parameters
    ----------
    ax : Axes
        Axes to draw on.
    x, y, sigma : np.ndarray
        The data that was fitted, 1-D.
    model : Callable[..., np.ndarray]
        The function that was fitted. Drawn on a finer grid than ``x``.
    result : FitResult
        The fit to draw.

    Returns
    -------
    Axes
        ``ax``, for chaining.
    """
    raise NotImplementedError


def plot_chi2_distribution(ax: Axes, chi2_values: np.ndarray, ndof: int) -> Axes:
    """
    Histogram of chi-square values from repeated fits, over the chi-square distribution.

    The curve is the distribution for ``ndof``, not a fit. The mean and standard deviation of
    the values are printed in the axes.

    Parameters
    ----------
    ax : Axes
        Axes to draw on.
    chi2_values : np.ndarray
        Chi-square of each repeated fit, 1-D.
    ndof : int
        Degrees of freedom of every fit. Must be positive.

    Returns
    -------
    Axes
        ``ax``, for chaining.
    """
    raise NotImplementedError


def plot_pulls(ax: Axes, x: np.ndarray, pulls: np.ndarray, limit: float = 2.0) -> Axes:
    """
    Bar chart of pulls, with the bars beyond ``±limit`` in red.

    Parameters
    ----------
    ax : Axes
        Axes to draw on.
    x : np.ndarray
        Position of each bar, 1-D.
    pulls : np.ndarray
        Pull of each point, 1-D, dimensionless, same length as ``x``.
    limit : float
        Pulls with ``abs(pull) > limit`` are drawn in red.

    Returns
    -------
    Axes
        ``ax``, for chaining.
    """
    raise NotImplementedError
