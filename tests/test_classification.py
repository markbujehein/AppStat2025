"""Tests for separating two kinds of events in appstat.classification."""

import numpy as np
import pytest

from appstat.classification import (
    auc,
    fisher_weights,
    roc_curve,
    selection_purity,
    separation,
    tpr_fpr,
)


def test_tpr_fpr_counts_signal_and_background_above_the_cut():
    # IF 2 of 4 signal values and 1 of 4 background values are above 2.5
    tpr, fpr = tpr_fpr(np.array([1, 2, 3, 4]), np.array([0, 1, 2, 3]), 2.5)

    # EXPECT
    assert tpr == 0.5 and fpr == 0.25


def test_tpr_fpr_can_select_below_the_cut():
    # IF the signal is the low values (like pions with a low signal)
    tpr, fpr = tpr_fpr(np.array([1, 2, 3, 4]), np.array([0, 1, 2, 3]), 2.5, signal_above=False)

    # EXPECT
    assert tpr == 0.5 and fpr == 0.75


def test_roc_curve_runs_from_nothing_accepted_to_everything_accepted():
    # THEN
    fpr, tpr, _ = roc_curve(
        np.random.default_rng(1).normal(1, 1, 500), np.random.default_rng(2).normal(0, 1, 500)
    )

    # EXPECT it starts at (0, 0), ends at (1, 1) and never goes down
    assert (fpr[0], tpr[0]) == (0.0, 0.0) and (fpr[-1], tpr[-1]) == (1.0, 1.0)
    assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)


def test_auc_is_one_for_perfect_separation_and_a_half_for_identical_samples():
    # IF the two samples do not overlap
    fpr, tpr, _ = roc_curve(np.arange(10.0, 20.0), np.arange(0.0, 10.0))

    # THEN / EXPECT
    assert auc(fpr, tpr) == pytest.approx(1.0)
    rng = np.random.default_rng(3)
    fpr, tpr, _ = roc_curve(rng.normal(size=5000), rng.normal(size=5000))
    assert auc(fpr, tpr) == pytest.approx(0.5, abs=0.03)


def test_roc_curve_follows_the_signal_direction():
    # IF the signal is the low values
    fpr, tpr, _ = roc_curve(np.arange(0.0, 10.0), np.arange(10.0, 20.0), signal_above=False)

    # THEN / EXPECT it is just as good
    assert auc(fpr, tpr) == pytest.approx(1.0)


def test_selection_purity_follows_bayes():
    # IF 1000 of each and 90 % efficiency at 10 % fake rate: 900 / (900 + 100)
    assert selection_purity(0.9, 0.1, 1000, 1000) == pytest.approx(0.9)

    # THEN / EXPECT a thousand times more background: 900 / (900 + 100000)
    assert selection_purity(0.9, 0.1, 1000, 1_000_000) == pytest.approx(900 / 100_900)


def test_selection_purity_works_on_a_whole_roc_curve():
    # IF three points of a ROC curve, the first accepting nothing
    purity = selection_purity(np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.1, 1.0]), 100, 100)

    # THEN / EXPECT nan where nothing is accepted (the purity is undefined), the rest from Bayes
    assert np.isnan(purity[0])
    np.testing.assert_allclose(purity[1:], [50 / 60, 0.5])


def test_selection_purity_of_a_selection_that_accepts_nothing_is_nan():
    # THEN / EXPECT
    assert np.isnan(selection_purity(0.0, 0.0, 10, 10))


def test_separation_matches_hand_calculation():
    # IF means 1 and 5, each with variance 2: |1 - 5| / sqrt(2 + 2)
    assert separation(np.array([0.0, 2.0]), np.array([4.0, 6.0])) == pytest.approx(2.0)


def test_fisher_weights_point_along_the_mean_difference_for_uncorrelated_variables():
    # IF two unit Gaussians in 2D that differ only in the first variable
    rng = np.random.default_rng(4)
    a = rng.normal(size=(20_000, 2)) + [1.0, 0.0]
    b = rng.normal(size=(20_000, 2))

    # THEN
    w = fisher_weights(a, b)

    # EXPECT the direction (1, 0)
    direction = w / np.linalg.norm(w)
    np.testing.assert_allclose(direction, [1.0, 0.0], atol=0.03)


def test_fisher_discriminant_separates_better_than_any_single_variable():
    # IF strongly correlated variables, shifted by 0.5 in x0 only
    rng = np.random.default_rng(5)
    covariance = [[1.0, 0.9], [0.9, 1.0]]
    a = rng.multivariate_normal([0.5, 0.0], covariance, 20_000)
    b = rng.multivariate_normal([0.0, 0.0], covariance, 20_000)

    # THEN
    w = fisher_weights(a, b)
    best_single = max(separation(a[:, i], b[:, i]) for i in range(2))

    # EXPECT the combination beats both single variables clearly
    assert separation(a @ w, b @ w) > 1.5 * best_single


def test_fisher_weights_reject_mismatched_variables():
    # THEN / EXPECT
    with pytest.raises(ValueError):
        fisher_weights(np.ones((10, 2)), np.ones((10, 3)))


def test_tpr_fpr_agrees_with_the_roc_curve_for_tied_values():
    # IF whole-number values, so many tie at the cut
    signal, background = np.array([3, 3, 4, 5]), np.array([1, 2, 3, 3])

    # THEN the cut at 3 is the ROC point of threshold 3
    tpr, fpr = tpr_fpr(signal, background, 3)
    roc_fpr, roc_tpr, thresholds = roc_curve(signal, background)

    # EXPECT values equal to the cut are accepted by both
    point = list(thresholds).index(3)
    assert (tpr, fpr) == (roc_tpr[point], roc_fpr[point])
    assert tpr == 1.0 and fpr == 0.5
