"""Tests for the digit sequence helpers in appstat.digits."""

import numpy as np
import pytest

from appstat.digits import (
    difference_probabilities,
    gap_lengths,
    read_digits,
    sequence_counts,
)


def test_read_digits_ignores_line_breaks_and_spaces():
    # THEN
    digits = read_digits("12\n3 4\n")

    # EXPECT
    np.testing.assert_array_equal(digits, [1, 2, 3, 4])


def test_read_digits_rejects_anything_that_is_not_a_digit():
    # THEN / EXPECT
    with pytest.raises(ValueError):
        read_digits("12a4")


def test_gap_lengths_are_the_distances_between_equal_digits():
    # IF the digit 0 sits at positions 0, 2, 3 and 5
    gaps = gap_lengths(np.array([0, 1, 0, 0, 2, 0]), 0)

    # EXPECT the distances 2, 1, 2
    np.testing.assert_array_equal(gaps, [2, 1, 2])


def test_gap_lengths_is_empty_when_the_digit_occurs_at_most_once():
    # THEN / EXPECT
    assert len(gap_lengths(np.array([1, 2, 0, 3]), 0)) == 0


def test_sequence_counts_count_overlapping_runs_of_digits():
    # IF the digits 0 1 2 0 1: the pairs are 01, 12, 20, 01
    counts = sequence_counts(np.array([0, 1, 2, 0, 1]), 2)

    # EXPECT one bin per possible pair, and 01 seen twice
    assert counts.shape == (100,)
    assert counts[1] == 2 and counts[12] == 1 and counts[20] == 1
    assert counts.sum() == 4


def test_sequence_counts_for_length_three_has_a_thousand_bins():
    # THEN / EXPECT
    assert sequence_counts(np.arange(10), 3).shape == (1000,)


def test_difference_probabilities_form_a_triangle():
    # THEN
    differences, probabilities = difference_probabilities()

    # EXPECT -9..9 with probability (10 - |d|) / 100, from counting all 100 pairs of digits
    np.testing.assert_array_equal(differences, np.arange(-9, 10))
    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities[9] == pytest.approx(0.10)  # d = 0
    assert probabilities[0] == pytest.approx(0.01)  # d = -9
