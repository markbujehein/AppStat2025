from collections.abc import Callable

lazy import numpy as np
lazy from iminuit import Minuit
lazy from iminuit.cost import LeastSquares


def fit_least_squares(
    model: Callable[..., np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    start: dict[str, float],
) -> Minuit:
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
    Minuit
        The fitted Minuit object: ``values``, ``errors``, ``fval`` (the chi-square at the minimum)
        and ``ndof`` (points minus fitted parameters).

    Raises
    ------
    ValueError
        If the shapes differ, any ``sigma`` is not positive, or there are no degrees of freedom.
    """
    x, y, sigma = (np.asarray(array, dtype=float) for array in (x, y, sigma))

    if not x.shape == y.shape == sigma.shape:
        raise ValueError(f"expected equal shapes, got {x.shape}, {y.shape} and {sigma.shape}")
    if np.any(sigma <= 0):
        raise ValueError("all sigma must be positive")
    if len(x) <= len(start):
        raise ValueError(f"need more points than parameters, got {len(x)} and {len(start)}")

    c = LeastSquares(x, y, sigma, model)

    m = Minuit(c, **start)
    m.migrad()
    return m


def pulls(
    model: Callable[..., np.ndarray],
    fit: Minuit,
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
    fit : Minuit
        The fit to compare with.
    x, y, sigma : np.ndarray
        The data that was fitted, 1-D.

    Returns
    -------
    np.ndarray
        ``(y - model(x, *fit.values)) / sigma``, 1-D, dimensionless.
    """
    x, y, sigma = (np.asarray(array, dtype=float) for array in (x, y, sigma))
    return (y - model(x, *fit.values)) / sigma


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
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if degree < 0:
        raise ValueError(f"degree must not be negative, got {degree}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if y.ndim != 2 or y.shape[1] != x.shape[0]:
        raise ValueError(f"expected y of shape (n_experiments, {x.shape[0]}), got {y.shape}")

    design = np.vander(x, degree + 1, increasing=True)  # columns 1, x, x², ...
    coefficients = np.linalg.lstsq(design, y.T, rcond=None)[0]  # (degree + 1, n_experiments)
    residuals = y - (design @ coefficients).T
    return np.sum((residuals / sigma) ** 2, axis=1)
