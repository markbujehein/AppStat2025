"""Tests for Markov chains in appstat.markov, with the Ehrenfest urn as the example."""

import numpy as np
import pytest
from scipy import stats

from appstat.markov import (
    ehrenfest_matrix,
    mean_hitting_time,
    simulate_markov,
    state_distribution,
)


def test_ehrenfest_matrix_has_one_state_more_than_balls_and_rows_that_sum_to_one():
    # IF 10 balls: states are 0 to 10 blue balls
    matrix = ehrenfest_matrix(10)

    # EXPECT
    assert matrix.shape == (11, 11)
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0)


def test_ehrenfest_matrix_moves_a_blue_ball_with_probability_k_over_n():
    # IF 3 blue balls of 10: picking a blue ball (3 of 10) leaves 2, a red ball (7 of 10) gives 4
    matrix = ehrenfest_matrix(10)

    # EXPECT
    assert matrix[3, 2] == pytest.approx(0.3)
    assert matrix[3, 4] == pytest.approx(0.7)
    assert matrix[0, 1] == 1.0 and matrix[10, 9] == 1.0


def test_ehrenfest_matrix_with_p_stay_keeps_rows_summing_to_one():
    # IF half of the turns nothing happens
    matrix = ehrenfest_matrix(10, p_stay=0.5)

    # EXPECT the diagonal is 0.5 and the moves are halved
    np.testing.assert_allclose(np.diag(matrix), 0.5)
    assert matrix[3, 2] == pytest.approx(0.15)
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0)


def test_ehrenfest_matrix_rejects_bad_arguments():
    # THEN / EXPECT
    with pytest.raises(ValueError):
        ehrenfest_matrix(0)
    with pytest.raises(ValueError):
        ehrenfest_matrix(10, p_stay=1.0)


def test_state_distribution_after_no_turns_is_the_start():
    # THEN
    probabilities = state_distribution(3, ehrenfest_matrix(10), 0)

    # EXPECT all probability in the starting state
    assert probabilities[3] == 1.0 and probabilities.sum() == 1.0


def test_state_distribution_cannot_reach_a_state_of_the_wrong_parity():
    # IF each turn changes the blue balls by one: start at 3, after 21 turns the state is even
    matrix = ehrenfest_matrix(10)

    # THEN
    after_21 = state_distribution(3, matrix, 21)
    after_20 = state_distribution(3, matrix, 20)

    # EXPECT state 9 is impossible after 21 turns, and state 7 is possible after 20
    assert after_21[9] == 0.0
    assert after_20[7] > 0.0
    assert after_20.sum() == pytest.approx(1.0)


def test_state_distribution_approaches_the_binomial_for_many_turns():
    # IF turns where nothing happens are allowed (so parity does not matter)
    matrix = ehrenfest_matrix(10, p_stay=0.5)

    # THEN
    probabilities = state_distribution(3, matrix, 2000)

    # EXPECT the equilibrium: every ball is in either basket with probability 1/2
    np.testing.assert_allclose(probabilities, stats.binom.pmf(np.arange(11), 10, 0.5), atol=1e-9)


def test_state_distribution_accepts_a_starting_distribution():
    # IF half the runs start in state 2 and half in state 4
    start = np.zeros(11)
    start[[2, 4]] = 0.5

    # THEN
    probabilities = state_distribution(start, ehrenfest_matrix(10), 1)

    # EXPECT the average of the two single-state results
    expected = 0.5 * state_distribution(2, ehrenfest_matrix(10), 1) + 0.5 * state_distribution(
        4, ehrenfest_matrix(10), 1
    )
    np.testing.assert_allclose(probabilities, expected)


def test_simulate_markov_gives_one_path_per_run_starting_in_the_start_state():
    # THEN
    paths = simulate_markov(ehrenfest_matrix(10), 3, 20, 50, np.random.default_rng(1))

    # EXPECT n_runs rows, n_turns + 1 columns, and each step changes the state by one
    assert paths.shape == (50, 21)
    assert np.all(paths[:, 0] == 3)
    assert np.all(np.abs(np.diff(paths, axis=1)) == 1)


def test_simulate_markov_is_reproducible_with_the_same_seed():
    # THEN
    a = simulate_markov(ehrenfest_matrix(6), 2, 10, 20, np.random.default_rng(9))
    b = simulate_markov(ehrenfest_matrix(6), 2, 10, 20, np.random.default_rng(9))

    # EXPECT
    np.testing.assert_array_equal(a, b)


def test_simulate_markov_follows_the_exact_distribution():
    # IF many runs of 20 turns from state 3
    matrix = ehrenfest_matrix(10)
    paths = simulate_markov(matrix, 3, 20, 40_000, np.random.default_rng(2))

    # THEN
    observed = np.bincount(paths[:, -1], minlength=11) / 40_000

    # EXPECT the fractions match the matrix result within statistical fluctuations
    np.testing.assert_allclose(observed, state_distribution(3, matrix, 20), atol=0.012)


def test_mean_hitting_time_of_two_urn_balls_matches_hand_calculation():
    # IF 2 balls and state 2 absorbs: from 1 it takes 3 turns on average, from 0 it takes 4
    times = mean_hitting_time(ehrenfest_matrix(2), target=2)

    # EXPECT
    np.testing.assert_allclose(times, [4.0, 3.0, 0.0])


def test_mean_hitting_time_is_one_over_p_for_a_single_step():
    # IF a state moves on with probability 0.2 each turn
    matrix = np.array([[0.8, 0.2], [0.0, 1.0]])

    # THEN / EXPECT
    np.testing.assert_allclose(mean_hitting_time(matrix, target=1), [5.0, 0.0])
