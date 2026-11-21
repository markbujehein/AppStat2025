"""Separating two kinds of events: rates, ROC curves, purity and the Fisher discriminant."""

lazy import numpy as np


def tpr_fpr(
    signal: np.ndarray, background: np.ndarray, cut: float, signal_above: bool = True
) -> tuple[float, float]:
    """
    True and false positive rate of a cut: the fractions of signal and of background accepted.

    Parameters
    ----------
    signal, background : np.ndarray
        The variable for each kind of event, 1-D.
    cut : float
        Events at or above (or at or below) the cut are accepted, as at each point of ``roc_curve``.
    signal_above : bool, default True
        Accept values above the cut. ``False`` accepts values below.

    Returns
    -------
    tpr, fpr : float
        Fraction of signal accepted, and fraction of background accepted.
    """
    signal, background = np.asarray(signal), np.asarray(background)
    accept = (lambda x: x >= cut) if signal_above else (lambda x: x <= cut)
    return float(accept(signal).mean()), float(accept(background).mean())


def roc_curve(
    signal: np.ndarray, background: np.ndarray, signal_above: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ROC curve: the signal and background accepted for every possible cut.

    Parameters
    ----------
    signal, background : np.ndarray
        The variable for each kind of event, 1-D.
    signal_above : bool, default True
        Accept values at or above the cut. ``False`` accepts values at or below.

    Returns
    -------
    fpr, tpr, thresholds : np.ndarray
        False and true positive rates, from ``(0, 0)`` to ``(1, 1)``, and the cut for each point.
    """
    sign = 1.0 if signal_above else -1.0
    s, b = (
        np.sort(sign * np.asarray(signal, dtype=float)),
        np.sort(sign * np.asarray(background, dtype=float)),
    )
    thresholds = np.concatenate(([np.inf], np.unique(np.concatenate([s, b]))[::-1]))
    tpr = 1 - np.searchsorted(s, thresholds, side="left") / len(s)
    fpr = 1 - np.searchsorted(b, thresholds, side="left") / len(b)
    return fpr, tpr, sign * thresholds


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Area under a ROC curve: 1 for perfect separation, 0.5 for none."""
    return float(np.trapezoid(tpr, fpr))


def selection_purity(
    tpr: float | np.ndarray, fpr: float | np.ndarray, n_signal: float, n_background: float
) -> float | np.ndarray:
    """
    Fraction of the accepted events that are signal.

    Parameters
    ----------
    tpr, fpr : float or np.ndarray
        Efficiency for signal, and fake rate for background. Arrays (for example a ROC curve)
        give one purity per point.
    n_signal, n_background : float
        How many of each kind there are before the selection.

    Returns
    -------
    float or np.ndarray
        ``tpr * n_signal / (tpr * n_signal + fpr * n_background)``. ``nan`` where the selection
        accepts nothing, as the purity is then undefined.
    """
    accepted_signal = np.asarray(tpr) * n_signal
    accepted = accepted_signal + np.asarray(fpr) * n_background
    with np.errstate(invalid="ignore", divide="ignore"):
        purity = accepted_signal / accepted
    return purity if purity.ndim else float(purity)


def separation(x: np.ndarray, y: np.ndarray) -> float:
    """Distance between the means of two samples, in units of their combined width."""
    return float(abs(np.mean(x) - np.mean(y)) / np.sqrt(np.var(x, ddof=1) + np.var(y, ddof=1)))


def fisher_weights(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Weights of the Fisher linear discriminant between two samples.

    The discriminant ``x @ w`` pulls the two means as far apart as possible compared with the
    spread of each sample: ``w = (cov_a + cov_b)^-1 (mean_a - mean_b)``.

    Parameters
    ----------
    a, b : np.ndarray
        The variables of each kind of event, shape ``(n_events, n_variables)``, same variables.

    Returns
    -------
    np.ndarray
        One weight per variable, shape ``(n_variables,)``.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` do not have the same number of variables.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(f"expected 2-D samples with equal columns, got {a.shape} and {b.shape}")
    summed_covariance = np.cov(a, rowvar=False) + np.cov(b, rowvar=False)
    return np.linalg.solve(summed_covariance, a.mean(axis=0) - b.mean(axis=0))
