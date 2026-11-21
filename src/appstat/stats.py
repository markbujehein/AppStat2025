from collections.abc import Callable

from scipy.special import gammaincc
lazy import numpy as np


def weighted_mean(x: np.ndarray, sigma: np.ndarray) -> tuple[float, float]:
    """
    Weighted mean of measurements with individual uncertainties, and its uncertainty.

    Parameters
    ----------
    x : np.ndarray
        Measurements, 1-D, in any unit.
    sigma : np.ndarray
        Uncertainty of each measurement, 1-D, same length and unit as ``x``. Must be positive.

    Returns
    -------
    tuple[float, float]
        The weighted mean and its uncertainty, both in the unit of ``x``.

    Raises
    ------
    ValueError
        If ``x`` and ``sigma`` differ in shape, or if any ``sigma`` is not positive.
    """
    x = np.asarray(x, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if x.shape != sigma.shape:
        raise ValueError(f"expected equal shapes, got {x.shape} and {sigma.shape}")
    if np.any(sigma <= 0):
        raise ValueError("all sigma must be positive")

    weights = 1.0 / sigma**2
    mean = np.average(x, weights=weights)
    error = np.sqrt(1.0 / np.sum(weights))

    return float(mean), float(error)


def chi2_pvalue(chi2: float, ndof: int) -> float:
    """
    Probability of a chi-square value at least as large as ``chi2`` for ``ndof`` degrees of freedom.

    Computed as the regularized upper incomplete gamma function Q(ndof / 2, chi2 / 2), which
    stays accurate for very small p-values (``1 - cdf`` would round to zero).

    Parameters
    ----------
    chi2 : float
        Observed chi-square value. Must not be negative.
    ndof : int
        Number of degrees of freedom. Must be positive.

    Returns
    -------
    float
        The p-value, between 0 and 1.

    Raises
    ------
    ValueError
        If ``chi2`` is negative or ``ndof`` is not positive.
    """
    if chi2 < 0:
        raise ValueError(f"chi2 must not be negative, got {chi2}")
    if ndof <= 0:
        raise ValueError(f"ndof must be positive, got {ndof}")

    return float(gammaincc(ndof / 2.0, chi2 / 2.0))


def propagate_mc(
    func: Callable[..., np.ndarray],
    means: np.ndarray,
    sigmas: np.ndarray,
    n_samples: int = 100_000,
    seed: int | None = None,
) -> float:
    """
    Propagate uncertainties through ``func`` by Monte Carlo sampling.

    Each input is drawn from an independent Gaussian (no correlations), ``func`` is applied to
    the samples, and the spread of the results is the propagated uncertainty.

    Parameters
    ----------
    func : Callable[..., np.ndarray]
        Function of one array per input, e.g. ``lambda length, width: length * width``. It must
        work on whole arrays (use NumPy operations, not ``if`` on an array).
    means : np.ndarray
        Central value of each input, 1-D.
    sigmas : np.ndarray
        Standard deviation of each input, 1-D, same length and unit as ``means``. Not negative.
    n_samples : int, optional
        Number of samples per input. The error on the result shrinks as ``1/sqrt(n_samples)``.
    seed : int, optional
        Seed for the random number generator. Give one to get the same result on every call.

    Returns
    -------
    float
        Standard deviation of ``func`` over the samples, in the unit of its result.

    Raises
    ------
    ValueError
        If ``means`` and ``sigmas`` differ in shape, or if any ``sigmas`` is negative.
    """
    means = np.asarray(means, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)

    if means.shape != sigmas.shape:
        raise ValueError(f"expected equal shapes, got {means.shape} and {sigmas.shape}")
    if np.any(sigmas < 0):
        raise ValueError("all sigmas must be non-negative")

    # One row of Gaussian samples per input, shape (len(means), n_samples)
    rng = np.random.default_rng(seed)
    samples = rng.normal(means[:, None], sigmas[:, None], size=(len(means), n_samples))

    results = func(*samples)

    return float(np.std(results, ddof=1))


def summary_statistics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """
    The numbers that Anscombe's four data sets share: means, variances, correlation and line.

    Parameters
    ----------
    x : np.ndarray
        Positions, 1-D.
    y : np.ndarray
        Measurements, 1-D, same length as ``x``.

    Returns
    -------
    dict[str, float]
        Keys ``mean_x``, ``var_x``, ``mean_y``, ``var_y`` (variances divided by N, as in
        Anscombe's paper), ``correlation`` (Pearson), and ``slope`` and ``intercept`` of the
        least-squares line ``y = intercept + slope * x``.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` differ in shape, or if all ``x`` are equal (no line can be fitted).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError(f"expected equal shapes, got {x.shape} and {y.shape}")
    if np.ptp(x) == 0:
        raise ValueError("all x are equal, no line can be fitted")

    slope, intercept = np.polyfit(x, y, 1)
    return {
        "mean_x": float(x.mean()),
        "var_x": float(x.var()),
        "mean_y": float(y.mean()),
        "var_y": float(y.var()),
        "correlation": float(np.corrcoef(x, y)[0, 1]),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def leave_one_out_slope_change(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    How much the fitted slope changes when each point is left out in turn.

    A point that moves the slope far more than the others is steering the fit.

    Parameters
    ----------
    x : np.ndarray
        Positions, 1-D.
    y : np.ndarray
        Measurements, 1-D, same length as ``x``.

    Returns
    -------
    np.ndarray
        ``slope without point i - slope with all points``, shape ``(len(x),)``. NaN where leaving
        the point out leaves all remaining ``x`` equal, so that no line can be fitted.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` differ in shape.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError(f"expected equal shapes, got {x.shape} and {y.shape}")

    slope = np.polyfit(x, y, 1)[0]
    change = np.full(len(x), np.nan)
    for i in range(len(x)):  # a handful of points, each with its own meaning
        keep = np.arange(len(x)) != i
        if np.ptp(x[keep]) > 0:
            change[i] = np.polyfit(x[keep], y[keep], 1)[0] - slope
    return change
