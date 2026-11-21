"""Helpers to test whether a series of digits is random."""

lazy import numpy as np


def read_digits(text: str) -> np.ndarray:
    """
    The digits in ``text``, in order. Line breaks and spaces are ignored.

    Raises
    ------
    ValueError
        If ``text`` holds anything but digits and white space.
    """
    characters = "".join(text.split())
    if not characters.isdigit():
        raise ValueError("the text must hold only digits")
    return np.frombuffer(characters.encode(), dtype=np.uint8).astype(int) - ord("0")


def gap_lengths(digits: np.ndarray, value: int) -> np.ndarray:
    """
    Distances between successive appearances of ``value`` in ``digits``.

    For random digits, the distance is ``k`` with probability ``0.1 * 0.9**(k - 1)``.
    """
    return np.diff(np.flatnonzero(np.asarray(digits) == value))


def sequence_counts(digits: np.ndarray, length: int) -> np.ndarray:
    """
    How often each run of ``length`` consecutive digits occurs (overlapping runs).

    Parameters
    ----------
    digits : np.ndarray
        Whole numbers 0 to 9, 1-D.
    length : int
        Length of the run, at least 1.

    Returns
    -------
    np.ndarray
        Counts of shape ``(10**length,)``. The run ``d0 d1 d2`` is number ``100*d0 + 10*d1 + d2``.
    """
    digits = np.asarray(digits)
    n_runs = len(digits) - length + 1
    codes = sum(digits[i : i + n_runs] * 10 ** (length - 1 - i) for i in range(length))
    return np.bincount(codes, minlength=10**length)


def difference_probabilities() -> tuple[np.ndarray, np.ndarray]:
    """
    Probability of the difference between neighbouring random digits: a triangle.

    Returns
    -------
    differences, probabilities : np.ndarray
        ``-9, ..., 9`` and ``(10 - abs(d)) / 100`` for each.
    """
    differences = np.arange(-9, 10)
    return differences, (10 - np.abs(differences)) / 100
