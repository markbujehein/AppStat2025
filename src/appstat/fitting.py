from collections.abc import Callable
from dataclasses import dataclass

lazy import numpy as np


@dataclass(frozen=True)
class FitResult:
    """
    Outcome of a least-squares fit.

    Attributes
    ----------
    values : dict[str, float]
        Fitted parameter values, by parameter name.
    errors : dict[str, float]
        Uncertainty of each fitted parameter, by parameter name.
    chi2 : float
        Chi-square at the minimum.
    ndof : int
        Degrees of freedom: number of points minus number of fitted parameters.
    pvalue : float
        Probability of a chi-square at least as large as ``chi2`` for ``ndof``.
    """

    values: dict[str, float]
    errors: dict[str, float]
    chi2: float
    ndof: int
    pvalue: float


def fit_least_squares(
    model: Callable[..., np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    start: dict[str, float],
) -> FitResult:
    """
    Fit ``model`` to ``(x, y)`` by minimising the chi-square, with Minuit.

    Parameters
    ----------
    model : Callable[..., np.ndarray]
        ``model(x, *parameters)``, vectorised over ``x``. Parameter names are taken from the
        signature, and must match the keys of ``start``.
    x : np.ndarray
        Positions, 1-D.
    y : np.ndarray
        Measurements, 1-D, same length as ``x``.
    sigma : np.ndarray
        Uncertainty of each measurement, 1-D, same length and unit as ``y``. Must be positive.
    start : dict[str, float]
        Starting value of each parameter, by name.

    Returns
    -------
    FitResult
        Values, errors, chi-square, degrees of freedom and p-value.

    Raises
    ------
    ValueError
        If the shapes differ, any ``sigma`` is not positive, or there are no degrees of freedom.
    """
    raise NotImplementedError


def pulls(
    model: Callable[..., np.ndarray],
    result: FitResult,
    x: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """
    Distance of each measurement to the fit, in units of its own uncertainty.

    The chi-square is the sum of the squared pulls, so the largest pull marks the point that
    disagrees most with the fit.

    Parameters
    ----------
    model : Callable[..., np.ndarray]
        The function that was fitted.
    result : FitResult
        The fit to compare with.
    x, y, sigma : np.ndarray
        The data that was fitted, 1-D.

    Returns
    -------
    np.ndarray
        ``(y - model(x, **result.values)) / sigma``, 1-D, dimensionless.
    """
    raise NotImplementedError


def polynomial_chi2(y: np.ndarray, x: np.ndarray, sigma: float, degree: int) -> np.ndarray:
    """
    Chi-square of the least-squares polynomial fit of many data sets at once.

    All points share one uncertainty, so each fit has a direct solution and one matrix
    operation fits every data set, with no loop (Barlow 6.2).

    Parameters
    ----------
    y : np.ndarray
        Data sets, shape ``(n_experiments, n_points)``, one data set per row.
    x : np.ndarray
        Positions, shape ``(n_points,)``.
    sigma : float
        Uncertainty on every point, same unit as ``y``. Must be positive.
    degree : int
        Degree of the polynomial: 0 constant, 1 line, 2 parabola. Not negative.

    Returns
    -------
    np.ndarray
        Chi-square of each data set, shape ``(n_experiments,)``.

    Raises
    ------
    ValueError
        If ``y`` does not match ``x``, ``sigma`` is not positive, or ``degree`` is negative.
    """
    raise NotImplementedError
