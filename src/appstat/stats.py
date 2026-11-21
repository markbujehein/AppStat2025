from collections.abc import Callable

from scipy.special import gammaincc, gammaln, xlogy
from scipy.stats import norm, skew
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


def integer_histogram(values: np.ndarray, n_max: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Count how often each integer 0, 1, ..., ``n_max`` occurs.

    Every integer gets its own bin, centred on the integer, empty bins included.

    Parameters
    ----------
    values : np.ndarray
        Whole numbers between 0 and ``n_max``, 1-D.
    n_max : int
        Largest value that can occur, for example the number of trials.

    Returns
    -------
    centres : np.ndarray
        ``0, 1, ..., n_max``, shape ``(n_max + 1,)``.
    counts : np.ndarray
        Number of ``values`` equal to each centre, shape ``(n_max + 1,)``.

    Raises
    ------
    ValueError
        If a value lies outside ``0..n_max``.
    """
    values = np.asarray(values)
    if values.min() < 0 or values.max() > n_max:
        raise ValueError(f"values must lie in 0..{n_max}, got {values.min()}..{values.max()}")
    counts, edges = np.histogram(values, bins=n_max + 1, range=(-0.5, n_max + 0.5))
    return (edges[1:] + edges[:-1]) / 2, counts


def pearson_chi2(
    observed: np.ndarray, expected: np.ndarray, n_params: int = 0
) -> tuple[float, int]:
    """
    Pearson chi-square, ``sum((observed - expected)**2 / observed)``, over the non-empty bins.

    Empty bins are left out, because the uncertainty ``sqrt(observed)`` is then zero.

    Parameters
    ----------
    observed : np.ndarray
        Counts in each bin, 1-D.
    expected : np.ndarray
        Counts the hypothesis predicts, same shape as ``observed``.
    n_params : int, default 0
        Number of parameters that were fitted to the data.

    Returns
    -------
    chi2 : float
        The chi-square value.
    ndof : int
        Number of non-empty bins minus ``n_params``.

    Raises
    ------
    ValueError
        If ``observed`` and ``expected`` have different shapes.
    """
    observed, expected = np.asarray(observed, dtype=float), np.asarray(expected, dtype=float)
    if observed.shape != expected.shape:
        raise ValueError(f"expected equal shapes, got {observed.shape} and {expected.shape}")
    used = observed > 0
    chi2 = np.sum((observed[used] - expected[used]) ** 2 / observed[used])
    return float(chi2), int(used.sum()) - n_params


def skewness_significance(values: np.ndarray) -> tuple[float, float]:
    """
    Skewness of a sample and its uncertainty, if the sample is Gaussian.

    A Gaussian has skewness zero, so ``skewness / sigma`` measures how many standard deviations
    the sample is away from being Gaussian.

    Parameters
    ----------
    values : np.ndarray
        The sample, 1-D with more than three numbers.

    Returns
    -------
    skewness : float
        The sample skewness.
    sigma : float
        Its uncertainty for ``n`` Gaussian numbers, about ``sqrt(6 / n)``.
    """
    n = len(values)
    sigma = np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))
    return float(skew(values)), float(sigma)


def binomial_pmf(x: np.ndarray, n: float, p: float) -> np.ndarray:
    """
    Binomial probability of ``x`` successes in ``n`` trials, also for a non-integer ``n``.

    ``scipy.stats.binom`` gives nan for a non-integer ``n``, which a fit tries on its way to the
    minimum. The binomial coefficient is therefore written with ``gammaln``.

    Parameters
    ----------
    x : np.ndarray
        Number of successes.
    n : float
        Number of trials.
    p : float
        Probability of success in one trial, between 0 and 1 (not included).

    Returns
    -------
    np.ndarray
        Probability of each ``x``, same shape as ``x``.
    """
    x = np.asarray(x, dtype=float)
    log_coefficient = gammaln(n + 1) - gammaln(x + 1) - gammaln(n - x + 1)
    return np.exp(log_coefficient + x * np.log(p) + (n - x) * np.log1p(-p))


def empirical_pvalue(simulated: np.ndarray, observed: float) -> float:
    """
    Probability of a value at least as large as ``observed``, estimated from simulations.

    This is the goodness of fit of a likelihood fit: simulate the data many times under the fitted
    model, fit each, and see how large the observed value is among the simulated ones. The observed
    value counts as one more simulation, so the result is never exactly zero.

    Parameters
    ----------
    simulated : np.ndarray
        The statistic (for example the negative log likelihood) of each simulated data set, 1-D.
    observed : float
        The same statistic of the real data.

    Returns
    -------
    float
        ``(number of simulated >= observed + 1) / (number of simulated + 1)``.
    """
    simulated = np.asarray(simulated)
    return float((np.sum(simulated >= observed) + 1) / (len(simulated) + 1))


def poisson_nll2(observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """
    Binned likelihood: ``-2 ln L`` of Poisson distributed bin counts, summed over the bins.

    Unlike the chi-square, empty bins are fine.

    Parameters
    ----------
    observed : np.ndarray
        Counts in each bin, last axis over the bins.
    expected : np.ndarray
        Counts the hypothesis predicts, positive, broadcastable to ``observed``.

    Returns
    -------
    np.ndarray
        ``-2 ln L``, with the bin axis summed away.
    """
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    return -2 * np.sum(xlogy(observed, expected) - expected - gammaln(observed + 1), axis=-1)


def exponential_nll2(times: np.ndarray, tau: float | np.ndarray) -> np.ndarray:
    """
    Unbinned likelihood: ``-2 ln L`` of exponentially distributed ``times`` with lifetime ``tau``.

    Parameters
    ----------
    times : np.ndarray
        The measured times, 1-D, positive.
    tau : float or np.ndarray
        Lifetime(s) to test. An array gives one value per lifetime.

    Returns
    -------
    np.ndarray
        ``-2 ln L``, shape ``np.shape(tau)``. Its minimum is at ``tau = mean(times)``.
    """
    tau = np.asarray(tau, dtype=float)
    return 2 * (len(times) * np.log(tau) + np.sum(times) / tau)


def delta_interval(x: np.ndarray, y: np.ndarray, delta: float = 1.0) -> tuple[float, float, float]:
    """
    Where a scan of a chi-square or ``-2 ln L`` has risen by ``delta`` above its minimum.

    ``delta = 1`` gives the one standard deviation interval. The crossings are interpolated
    linearly between the scan points.

    Parameters
    ----------
    x : np.ndarray
        Scanned parameter values, increasing.
    y : np.ndarray
        Chi-square or ``-2 ln L`` at each ``x``, with one minimum.
    delta : float, default 1.0
        The rise above the minimum.

    Returns
    -------
    best, low, high : float
        The scan point with the lowest ``y``, and the crossings on its left and right.

    Raises
    ------
    ValueError
        If the scan does not rise by ``delta`` on both sides of the minimum.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    i = int(np.argmin(y))
    level = y[i] + delta
    if y[0] < level or y[-1] < level:
        raise ValueError("the scan must rise by delta on both sides of the minimum")
    left = np.flatnonzero(y[: i + 1] >= level)[-1]
    right = i + np.flatnonzero(y[i:] >= level)[0]
    low = _cross(x, y, left, left + 1, level)
    high = _cross(x, y, right - 1, right, level)
    return float(x[i]), float(low), float(high)


def _cross(x: np.ndarray, y: np.ndarray, i: int, j: int, level: float) -> float:
    """``x`` where the straight line from ``(x[i], y[i])`` to ``(x[j], y[j])`` reaches ``level``."""
    return x[i] + (level - y[i]) * (x[j] - x[i]) / (y[j] - y[i])


def weighted_mean_chi2(x: np.ndarray, sigma: np.ndarray) -> tuple[float, int]:
    """
    Chi-square of measurements around their weighted mean, and its degrees of freedom.

    A chi-square much larger than ``ndof`` says the measurements disagree more than their
    uncertainties allow. Use ``chi2_pvalue`` to turn the two numbers into a probability.

    Parameters
    ----------
    x : np.ndarray
        Measurements, 1-D.
    sigma : np.ndarray
        Uncertainty of each measurement, same shape and unit as ``x``, positive.

    Returns
    -------
    chi2 : float
        ``sum(((x - mean) / sigma)**2)``, with ``mean`` the weighted mean.
    ndof : int
        ``len(x) - 1``, since the mean was fitted.
    """
    mean, _ = weighted_mean(x, sigma)
    chi2 = np.sum(((np.asarray(x, dtype=float) - mean) / np.asarray(sigma, dtype=float)) ** 2)
    return float(chi2), len(x) - 1


def chauvenet_nsigma(n: int, p_global: float = 0.05) -> float:
    """
    How many standard deviations away a point must be to be an outlier, among ``n`` points.

    Chauvenet's criterion: with ``n`` Gaussian numbers, the chance that **any** of them lies
    farther than the limit is ``p_global``. The limit grows with ``n``, since more numbers
    give more chances of a far one.

    Parameters
    ----------
    n : int
        Number of points.
    p_global : float, default 0.05
        Probability of rejecting at least one good point.

    Returns
    -------
    float
        The limit in standard deviations.
    """
    p_local = 1 - (1 - p_global) ** (1 / n)
    return float(norm.isf(p_local / 2))


def chauvenet(values: np.ndarray, p_global: float = 0.05) -> np.ndarray:
    """
    Reject outliers one by one with Chauvenet's criterion.

    The point farthest from the mean is rejected if it is beyond ``chauvenet_nsigma`` standard
    deviations. The mean and standard deviation are then recalculated without it, and the
    next farthest point is tested, until one is accepted.

    Parameters
    ----------
    values : np.ndarray
        The measurements, 1-D. Not changed.
    p_global : float, default 0.05
        Probability of rejecting at least one good point.

    Returns
    -------
    np.ndarray
        Boolean mask, ``True`` for accepted points, same shape as ``values``.
    """
    values = np.asarray(values, dtype=float)
    accepted = np.ones(len(values), dtype=bool)
    while accepted.sum() > 2:
        kept = values[accepted]
        width = kept.std(ddof=1)
        if width == 0:
            break
        distance = np.where(accepted, np.abs(values - kept.mean()), -np.inf)
        farthest = int(np.argmax(distance))
        if distance[farthest] / width <= chauvenet_nsigma(len(kept), p_global):
            break
        accepted[farthest] = False
    return accepted


def accept_reject(
    pdf: Callable[[np.ndarray], np.ndarray],
    low: float,
    high: float,
    ceiling: float,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """
    Random numbers following ``pdf`` by the accept/reject (von Neumann) method.

    Points are thrown uniformly into the box ``[low, high] x [0, ceiling]`` and kept when they
    fall below the function. Generates in batches, so there is no Python loop over points.

    Parameters
    ----------
    pdf : callable
        The function, vectorised: takes an array of x and returns an array. Need not be normalised.
    low, high : float
        The range of x, with ``low < high``.
    ceiling : float
        At or above the maximum of ``pdf`` in the range. Higher means lower efficiency.
    n : int
        Number of points wanted, at least 1.
    rng : np.random.Generator
        Source of random numbers, for reproducibility.

    Returns
    -------
    samples : np.ndarray
        The ``n`` accepted x values.
    n_tries : int
        How many points were thrown. ``n / n_tries`` is the efficiency.

    Raises
    ------
    ValueError
        If the range or ``n`` is invalid, or if ``pdf`` exceeds ``ceiling`` at a thrown point.
        Only thrown points are checked, so a narrow peak just above the ceiling can be missed with
        few points: take the ceiling from the maximum of the function.
    """
    if not low < high:
        raise ValueError("need low < high")
    if n < 1:
        raise ValueError("need n >= 1")
    accepted: list[np.ndarray] = []
    n_accepted = n_tries = 0
    while n_accepted < n:
        x = rng.uniform(low, high, n)
        y = rng.uniform(0.0, ceiling, n)
        f = pdf(x)
        if np.any(f > ceiling):
            raise ValueError("pdf exceeds the ceiling, so the box does not contain it")
        hits = np.flatnonzero(y < f)
        needed = n - n_accepted
        if len(hits) >= needed:  # last batch: stop at the point that completes n, as a loop would
            hits = hits[:needed]
            n_tries += int(hits[-1]) + 1
        else:
            n_tries += n
        accepted.append(x[hits])
        n_accepted += len(hits)
    return np.concatenate(accepted), n_tries


def box_integral(n_accepted: int, n_tries: int, box_area: float) -> tuple[float, float]:
    """
    Integral estimated from the fraction of accepted points in a box, and its uncertainty.

    Parameters
    ----------
    n_accepted : int
        Points that fell below the function.
    n_tries : int
        Points thrown into the box.
    box_area : float
        Area of the box, ``(high - low) * ceiling``.

    Returns
    -------
    integral, error : float
        ``efficiency * box_area`` and its binomial uncertainty.
    """
    efficiency = n_accepted / n_tries
    error = np.sqrt(efficiency * (1 - efficiency) / n_tries)
    return float(efficiency * box_area), float(error * box_area)


def two_sample_chi2(counts_a: np.ndarray, counts_b: np.ndarray) -> tuple[float, int]:
    """
    Chi-square between two histograms with the same binning, to test if they share a distribution.

    Bins empty in both are skipped. No parameter is fitted, so ``ndof`` is the number of bins used.

    Parameters
    ----------
    counts_a, counts_b : np.ndarray
        Bin counts, same shape, from samples of equal size.

    Returns
    -------
    chi2 : float
        ``sum((a - b)**2 / (a + b))`` over the used bins.
    ndof : int
        Number of bins used.

    Raises
    ------
    ValueError
        If the shapes differ.
    """
    a, b = np.asarray(counts_a, dtype=float), np.asarray(counts_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"expected equal shapes, got {a.shape} and {b.shape}")
    used = a + b > 0
    return float(np.sum((a[used] - b[used]) ** 2 / (a[used] + b[used]))), int(used.sum())
