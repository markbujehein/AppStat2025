"""Monte Carlo estimates: volumes of hyper-spheres and integrals by plain or stratified sampling."""

from collections.abc import Callable

from scipy.special import gammaln
lazy import numpy as np


def unit_ball_volume(n_dim: int) -> float:
    """Volume of the unit ball in ``n_dim`` dimensions: ``pi**(d/2) / Gamma(d/2 + 1)``."""
    return float(np.exp(n_dim / 2 * np.log(np.pi) - gammaln(n_dim / 2 + 1)))


def estimate_ball_volume(
    n_dim: int, n_points: int, rng: np.random.Generator
) -> tuple[float, float]:
    """
    Volume of the unit ball from the fraction of random points in the cube around it inside it.

    Parameters
    ----------
    n_dim : int
        Number of dimensions.
    n_points : int
        Number of random points in the cube ``[-1, 1]**n_dim``.
    rng : np.random.Generator
        Source of random numbers.

    Returns
    -------
    volume, error : float
        ``2**n_dim * f`` and its binomial uncertainty ``2**n_dim * sqrt(f (1 - f) / N)``, with
        ``f`` the fraction of points inside. With no point inside, the error is the upper limit
        ``2**n_dim / N`` instead of 0.
    """
    points = rng.uniform(-1, 1, (n_points, n_dim))
    fraction = np.mean(np.sum(points**2, axis=1) < 1)
    cube = 2.0**n_dim
    error = np.sqrt(fraction * (1 - fraction) / n_points) if fraction > 0 else 1 / n_points
    return float(cube * fraction), float(cube * error)


def plain_integral(
    func: Callable[[np.ndarray], np.ndarray],
    low: float,
    high: float,
    n_points: int,
    rng: np.random.Generator,
) -> float:
    """Integral of ``func`` from ``low`` to ``high``: its mean at uniform points times the width."""
    return float((high - low) * np.mean(func(rng.uniform(low, high, n_points))))


def stratified_integral(
    func: Callable[[np.ndarray], np.ndarray],
    edges: np.ndarray,
    counts: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """
    Integral of ``func`` by stratified sampling: a mean in each stratum, added up with its width.

    Parameters
    ----------
    func : callable
        The function, taking and returning arrays.
    edges : np.ndarray
        Edges of the strata, shape ``(n_strata + 1,)``.
    counts : np.ndarray
        Random points to use in each stratum, shape ``(n_strata,)``, each at least 1.
    rng : np.random.Generator
        Source of random numbers.

    Returns
    -------
    float
        ``sum(width_j * mean_j)``.
    """
    return float(
        sum(
            (high - low) * np.mean(func(rng.uniform(low, high, n)))
            for low, high, n in zip(edges[:-1], edges[1:], counts, strict=True)
        )
    )


def stratum_stds(
    func: Callable[[np.ndarray], np.ndarray],
    edges: np.ndarray,
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Standard deviation of ``func`` in each stratum, from ``n_points`` random points in each."""
    return np.array(
        [
            np.std(func(rng.uniform(low, high, n_points)), ddof=1)
            for low, high in zip(edges[:-1], edges[1:], strict=True)
        ]
    )


def equal_counts(n_strata: int, n_total: int) -> np.ndarray:
    """Split ``n_total`` points evenly over the strata; the first few get the remainder."""
    return n_total // n_strata + (np.arange(n_strata) < n_total % n_strata)


def proportional_counts(stds: np.ndarray, n_total: int) -> np.ndarray:
    """
    Split ``n_total`` points over the strata in proportion to their spread.

    Every stratum gets at least one point, and the counts add up to ``n_total``.

    Parameters
    ----------
    stds : np.ndarray
        Spread in each stratum, not negative.
    n_total : int
        Number of points to share out, at least the number of strata.

    Returns
    -------
    np.ndarray
        Integer counts, shape ``(n_strata,)``.
    """
    stds = np.asarray(stds, dtype=float)
    if not np.isfinite(stds).all():
        raise ValueError("every stratum needs a finite spread (use at least 2 pilot points)")
    if stds.sum() == 0:  # constant function: no stratum matters more than another
        return equal_counts(len(stds), n_total)
    counts = np.maximum(1, np.round(n_total * stds / stds.sum())).astype(int)
    counts[counts.argmax()] += n_total - counts.sum()  # rounding error goes to the largest
    return counts
