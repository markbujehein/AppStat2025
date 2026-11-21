"""Tests for the likelihood fits in appstat.fitting and appstat.stats.empirical_pvalue."""

import numpy as np
import pytest
from scipy import stats

from appstat.fitting import fit_binned_likelihood, fit_unbinned_likelihood
from appstat.stats import (
    delta_interval,
    empirical_pvalue,
    exponential_bin_counts,
    exponential_nll2,
    poisson_nll2,
)


def exponential_density(x, N, tau):
    return N / tau * np.exp(-x / tau)


def exponential_with_total(x, N, tau):
    return N, exponential_density(x, N, tau)


def test_unbinned_likelihood_recovers_the_lifetime():
    # IF many exponential times with lifetime 2
    times = np.random.default_rng(1).exponential(2.0, 5000)

    # THEN
    fit = fit_unbinned_likelihood(exponential_with_total, times, {"N": 5000.0, "tau": 1.5})

    # EXPECT the maximum likelihood lifetime of an exponential is the mean
    assert fit.valid
    assert fit.values["tau"] == pytest.approx(times.mean(), abs=3 * fit.errors["tau"])


def test_binned_likelihood_recovers_the_lifetime():
    # IF a histogram of exponential times with lifetime 2
    times = np.random.default_rng(2).exponential(2.0, 5000)
    counts, edges = np.histogram(times, bins=40, range=(0, 10))

    # THEN
    fit = fit_binned_likelihood(exponential_density, counts, edges, {"N": 5000.0, "tau": 1.5})

    # EXPECT
    assert fit.valid
    assert fit.values["tau"] == pytest.approx(2.0, abs=3 * fit.errors["tau"])


def test_unbinned_likelihood_rejects_no_data():
    # THEN / EXPECT
    with pytest.raises(ValueError):
        fit_unbinned_likelihood(exponential_with_total, np.array([]), {"N": 1.0, "tau": 1.0})


def test_binned_likelihood_rejects_edges_that_do_not_match_the_counts():
    # IF three counts need four edges
    counts = np.array([5, 3, 1])

    # THEN / EXPECT
    with pytest.raises(ValueError):
        fit_binned_likelihood(exponential_density, counts, np.arange(3.0), {"N": 9.0, "tau": 1.0})


def test_empirical_pvalue_is_the_share_of_simulations_at_least_as_extreme():
    # IF four simulated values and an observed 3
    simulated = np.array([1.0, 2.0, 3.0, 4.0])

    # THEN
    p = empirical_pvalue(simulated, observed=3.0)

    # EXPECT (2 + 1) / (4 + 1): the observed value counts as one more simulation
    assert p == pytest.approx(0.6)


def test_empirical_pvalue_is_never_zero():
    # IF the observed value is above every simulation
    p = empirical_pvalue(np.array([1.0, 2.0, 3.0]), observed=10.0)

    # EXPECT 1 / (3 + 1), not 0, since a finite simulation cannot show a probability of zero
    assert p == pytest.approx(0.25)


# --- scans of one parameter (LikelihoodFit) ---------------------------------------------------


def test_exponential_bin_counts_add_up_to_the_total_inside_the_range():
    # IF bins covering almost everything of a lifetime 1 exponential
    edges = np.linspace(0, 40, 41)

    # THEN
    counts = exponential_bin_counts(edges, n_total=100, tau=1.0)

    # EXPECT
    assert counts.shape == (40,)
    assert counts.sum() == pytest.approx(100.0)
    assert counts[0] == pytest.approx(100 * (1 - np.exp(-1.0)))


def test_exponential_bin_counts_give_one_row_per_lifetime():
    # THEN
    counts = exponential_bin_counts(np.linspace(0, 10, 11), 50, np.array([0.5, 1.0, 2.0]))

    # EXPECT
    assert counts.shape == (3, 10)


def test_poisson_nll2_matches_scipy():
    # IF
    observed = np.array([0, 3, 5])
    expected = np.array([0.5, 2.5, 6.0])

    # THEN
    nll2 = poisson_nll2(observed, expected)

    # EXPECT -2 ln of the product of Poisson probabilities
    assert nll2 == pytest.approx(-2 * np.sum(stats.poisson.logpmf(observed, expected)))


def test_poisson_nll2_is_smallest_when_the_expectation_equals_the_count():
    # IF
    observed = np.array([4, 9, 16])

    # THEN
    best = poisson_nll2(observed, observed.astype(float))

    # EXPECT any other expectation is worse
    assert best < poisson_nll2(observed, observed * 1.2)
    assert best < poisson_nll2(observed, observed * 0.8)


def test_exponential_nll2_is_smallest_at_the_mean_of_the_times():
    # IF
    times = np.random.default_rng(3).exponential(1.5, 200)

    # THEN
    taus = np.linspace(0.5, 3.0, 501)
    nll2 = exponential_nll2(times, taus)

    # EXPECT the maximum likelihood lifetime of an exponential is the mean time
    assert nll2.shape == taus.shape
    assert taus[np.argmin(nll2)] == pytest.approx(times.mean(), abs=0.006)


def test_delta_interval_of_a_parabola():
    # IF chi2 = ((x - 2) / 0.5)**2 sampled on a grid
    x = np.linspace(0, 4, 401)
    y = ((x - 2) / 0.5) ** 2 + 3

    # THEN a rise of one above the minimum gives plus and minus 0.5
    best, low, high = delta_interval(x, y, delta=1.0)

    # EXPECT
    assert (best, low, high) == pytest.approx((2.0, 1.5, 2.5), abs=1e-3)


def test_delta_interval_rejects_a_scan_that_does_not_rise_on_both_sides():
    # IF the scan range ends before chi2 has risen by one
    x = np.linspace(1.9, 2.1, 50)

    # THEN / EXPECT
    with pytest.raises(ValueError):
        delta_interval(x, ((x - 2) / 0.5) ** 2)
