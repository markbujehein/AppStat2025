"""Model functions and exact curves for the exercises, in plain NumPy."""

lazy import numpy as np


def exponential_bin_counts(
    edges: np.ndarray, n_total: float, tau: float | np.ndarray
) -> np.ndarray:
    """
    Expected counts per bin for ``n_total`` exponential values with lifetime ``tau``.

    Parameters
    ----------
    edges : np.ndarray
        Bin edges, 1-D, one more than the number of bins.
    n_total : float
        Total number of values, including those outside the bins.
    tau : float or np.ndarray
        Lifetime(s), same unit as ``edges``. An array gives one row per lifetime.

    Returns
    -------
    np.ndarray
        Expected counts, shape ``(*np.shape(tau), n_bins)``.
    """
    survival = np.exp(-np.asarray(edges) / np.asarray(tau)[..., np.newaxis])
    return n_total * (survival[..., :-1] - survival[..., 1:])


def vacation_line(
    x: np.ndarray, offset: float, slope: float, length: float, start: float = 40.0
) -> np.ndarray:
    """
    Constant use that pauses for a vacation: a straight line with a flat piece.

    The count is ``offset + slope * x`` until ``start``, stays constant for ``length``, and then
    continues along the same line, delayed by ``length``.

    Parameters
    ----------
    x : np.ndarray
        Time, for example days since the first measurement.
    offset : float
        Value at ``x = 0``.
    slope : float
        Increase per unit time while in use.
    length : float
        Length of the vacation, same unit as ``x``. Zero gives a plain line.
    start : float, default 40.0
        When the vacation starts, same unit as ``x``.

    Returns
    -------
    np.ndarray
        The count at each ``x``.
    """
    x = np.asarray(x, dtype=float)
    time_in_use = x - np.clip(x - start, 0.0, length)
    return offset + slope * time_in_use


def power_law(x: np.ndarray, n: float, alpha: float) -> np.ndarray:
    """``n * x**(-alpha)``: a power law, equal to ``n`` at ``x = 1``."""
    return n * np.asarray(x, dtype=float) ** (-alpha)


def power_exponential(
    x: np.ndarray, n: float, f: float, alpha_pow: float, alpha_exp: float
) -> np.ndarray:
    """
    A power law and an exponential in the fraction ``f`` and ``1 - f``.

    Parameters
    ----------
    x : np.ndarray
        Where to evaluate.
    n : float
        Overall scale.
    f : float
        Fraction in the power law, between 0 and 1.
    alpha_pow : float
        Power of the power law part.
    alpha_exp : float
        Decay constant of the exponential part, positive. That part is divided by ``alpha_exp``.

    Returns
    -------
    np.ndarray
        ``n * (f * x**-alpha_pow + (1 - f) * exp(-alpha_exp * x) / alpha_exp)``.
    """
    x = np.asarray(x, dtype=float)
    return n * (f * x**-alpha_pow + (1 - f) * np.exp(-alpha_exp * x) / alpha_exp)


def x_cos_x_cdf(x: np.ndarray) -> np.ndarray:
    """Integral of ``t * cos(t)`` from 0 to ``x``: ``x sin(x) + cos(x) - 1``."""
    x = np.asarray(x, dtype=float)
    return x * np.sin(x) + np.cos(x) - 1


def exp_cos2_cdf(x: np.ndarray) -> np.ndarray:
    """
    Integral of ``exp(-t/3) * cos(t)**2`` from 0 to ``x``, using ``cos^2 = (1 + cos 2t) / 2``.

    Use a large finite ``x`` (like 1e3) for the total, as ``cos(inf)`` is undefined.

    Parameters
    ----------
    x : np.ndarray
        Upper limit(s), not negative.

    Returns
    -------
    np.ndarray
        The integral at each ``x``.
    """
    x = np.asarray(x, dtype=float)
    a, b = 1 / 3, 2.0
    decay = np.exp(-a * x)
    return 1.5 * (1 - decay) + 0.5 * (decay * (-a * np.cos(b * x) + b * np.sin(b * x)) + a) / (
        a**2 + b**2
    )


def parabola(x: np.ndarray, p0: float, p1: float, p2: float) -> np.ndarray:
    """``p0 + p1 * x + p2 * x**2``."""
    return p0 + p1 * x + p2 * x**2


def offset_exponential(x: np.ndarray, p0: float, p1: float, p2: float) -> np.ndarray:
    """``p0 + p1 * exp(p2 * x)``."""
    return p0 + p1 * np.exp(p2 * x)


def apply_corrections(measured: np.ndarray, *relative_biases: np.ndarray) -> np.ndarray:
    """
    Remove relative biases from a measurement: ``measured / (1 + f1) / (1 + f2) / ...``.

    A measurement 50 % too high has ``f = 0.5`` and is divided by 1.5.

    Parameters
    ----------
    measured : np.ndarray
        The measured values. Not changed.
    *relative_biases : np.ndarray
        One relative bias per effect, each a number or an array like ``measured``.

    Returns
    -------
    np.ndarray
        The corrected values.
    """
    corrected = np.asarray(measured, dtype=float)
    for bias in relative_biases:
        corrected = corrected / (1 + bias)
    return corrected


def two_exponentials(x: np.ndarray, n1: float, tau1: float, n2: float, tau2: float) -> np.ndarray:
    """``n1 * exp(-x / tau1) + n2 * exp(-x / tau2)``: heights that correlate with the lifetimes."""
    return n1 * np.exp(-x / tau1) + n2 * np.exp(-x / tau2)


def exponential_mixture(x: np.ndarray, n: float, f: float, tau1: float, tau2: float) -> np.ndarray:
    """``n`` times a normalised mix of two decays, a fraction ``f`` with lifetime ``tau1``."""
    return n * (f / tau1 * np.exp(-x / tau1) + (1 - f) / tau2 * np.exp(-x / tau2))


def peak_series(
    x: np.ndarray, n: float, f: float, mu: float, dmu: float, sigma: float, cst: float, n_peaks: int
) -> np.ndarray:
    """
    Gaussian peaks at ``mu``, ``mu + dmu``, ... on a constant, with areas ``n f (1 - f)**k``.

    Sharing ``f``, ``dmu`` and ``sigma`` between the peaks is what gives a fit its precision.
    """
    k = np.arange(n_peaks)
    areas = n * f * (1 - f) ** k
    x = np.asarray(x, dtype=float)[..., np.newaxis]
    gauss = np.exp(-0.5 * ((x - (mu + dmu * k)) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    return (areas * gauss).sum(axis=-1) + cst
