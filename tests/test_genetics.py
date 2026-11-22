"""Tests for the genotype probabilities in appstat.genetics."""

import numpy as np
import pytest

from appstat.genetics import genotype_posterior, simulate_reads

MEASUREMENT = np.array([[0.85, 0.15], [0.05, 0.95]])  # row: true base A or B, column: measured
PRIOR = np.array([0.45, 0.15, 0.40])  # AA, AB, BB


def test_without_reads_the_posterior_is_the_prior():
    # THEN / EXPECT
    np.testing.assert_allclose(genotype_posterior([], MEASUREMENT, PRIOR), PRIOR)


def test_one_measured_b_follows_bayes_by_hand():
    # IF one read measured B. P(B read | AA) = 0.15, | AB = (0.15 + 0.95) / 2, | BB = 0.95
    posterior = genotype_posterior([1], MEASUREMENT, PRIOR)

    # THEN / EXPECT prior times this, divided by the sum
    unnormalised = PRIOR * np.array([0.15, 0.55, 0.95])
    np.testing.assert_allclose(posterior, unnormalised / unnormalised.sum())


def test_with_a_perfect_machine_two_a_reads_rule_out_bb():
    # IF no measurement errors, reads [A, A]: AA gives that with probability 1, AB with 1/4
    posterior = genotype_posterior([0, 0], np.eye(2), PRIOR)

    # THEN / EXPECT
    assert posterior[2] == 0.0
    assert posterior[0] == pytest.approx(0.45 / (0.45 + 0.15 / 4))


def test_the_order_of_the_reads_does_not_matter():
    # THEN / EXPECT
    np.testing.assert_allclose(
        genotype_posterior([0, 0, 0, 0, 1], MEASUREMENT, PRIOR),
        genotype_posterior([1, 0, 0, 0, 0], MEASUREMENT, PRIOR),
    )


def test_the_posterior_sums_to_one():
    # THEN / EXPECT
    assert genotype_posterior([0, 1, 1, 0], MEASUREMENT, PRIOR).sum() == pytest.approx(1.0)


def test_a_prior_that_does_not_sum_to_one_is_rejected():
    # THEN / EXPECT
    with pytest.raises(ValueError):
        genotype_posterior([0], MEASUREMENT, np.array([0.5, 0.5, 0.5]))


def test_simulated_reads_of_a_pure_genotype_with_a_perfect_machine_are_all_that_base():
    # THEN
    rng = np.random.default_rng(1)

    # EXPECT genotype 0 (AA) always reads A and genotype 2 (BB) always B
    assert np.all(simulate_reads(0, 50, np.eye(2), rng) == 0)
    assert np.all(simulate_reads(2, 50, np.eye(2), rng) == 1)


def test_simulated_reads_of_ab_through_the_machine_have_the_expected_fraction_of_b():
    # IF many reads of AB: P(B) = (0.15 + 0.95) / 2
    reads = simulate_reads(1, 50_000, MEASUREMENT, np.random.default_rng(2))

    # THEN / EXPECT
    assert reads.mean() == pytest.approx(0.55, abs=0.01)


def test_reads_that_rule_out_every_genotype_are_rejected():
    # IF a perfect machine reads A twice but only BB is possible a priori
    # THEN / EXPECT an error, not a nan posterior
    with pytest.raises(ValueError, match="impossible"):
        genotype_posterior([0, 0], np.eye(2), np.array([0.0, 0.0, 1.0]))
