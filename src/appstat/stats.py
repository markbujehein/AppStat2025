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

    return gammaincc(ndof / 2.0, chi2 / 2.0)


def propagate_mc(
    func: Callable[..., np.ndarray], means: np.ndarray, sigmas: np.ndarray, n_samples: int = 100_000
) -> float:
    means = np.asarray(means, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)

    # One row of Gaussian samples per input, shape (len(means), n_samples)
    rng = np.random.default_rng()
    samples = rng.normal(means[:, None], sigmas[:, None], size=(len(means), n_samples))

    results = func(*samples)

    return np.std(results, ddof=1)
