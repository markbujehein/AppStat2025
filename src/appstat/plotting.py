"""Plots of a fit, its pulls and the chi-squared distribution, and the fit summary text."""

from collections.abc import Callable

from matplotlib.axes import Axes
lazy import numpy as np
lazy from iminuit import Minuit
lazy from scipy import stats

from appstat.stats import chi2_pvalue


def fit_info_text(fit: Minuit) -> str:
    """
    Text for the "result of fit" box: chi-square, p-value and each parameter with its error.

    Each parameter is printed with as many decimals as its uncertainty justifies.

    Parameters
    ----------
    fit : Minuit
        The fit to describe. The chi-square is ``fit.fval``, the degrees of freedom ``fit.ndof``.

    Returns
    -------
    str
        Multi-line text, ready for ``ax.text`` (math written with ``$...$``).
    """
    lines = [
        f"$\\chi^2$ / $N_\\mathrm{{dof}}$ = {fit.fval:.1f} / {fit.ndof}",
        f"Prob($\\chi^2$, $N_\\mathrm{{dof}}$) = {chi2_pvalue(fit.fval, fit.ndof):.3f}",
    ]
    for name, value, error in zip(fit.parameters, fit.values, fit.errors, strict=True):
        # two significant digits of the error decide the number of decimals
        decimals = max(0, 1 - int(np.floor(np.log10(error)))) if error > 0 else 3
        lines.append(f"{name} = {value:.{decimals}f} ± {error:.{decimals}f}")
    return "\n".join(lines)


def plot_fit(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    model: Callable[..., np.ndarray],
    fit: Minuit,
) -> Axes:
    """
    Draw the data with error bars, the fitted curve and the fit info box.

    Parameters
    ----------
    ax : Axes
        Axes to draw on.
    x, y, sigma : np.ndarray
        The data that was fitted, 1-D.
    model : Callable[..., np.ndarray]
        The function that was fitted. Drawn on a finer grid than ``x``.
    fit : Minuit
        The fit to draw.

    Returns
    -------
    Axes
        ``ax``, for chaining.
    """
    ax.errorbar(x, y, sigma, fmt="o", capsize=3, label="Data")
    grid = np.linspace(np.min(x), np.max(x), 500)
    ax.plot(grid, model(grid, *fit.values), label="Fit")
    ax.text(0.97, 0.05, fit_info_text(fit), transform=ax.transAxes, ha="right", va="bottom")
    ax.legend(loc="upper left")
    return ax


def plot_chi2_distribution(ax: Axes, chi2_values: np.ndarray, ndof: int) -> Axes:
    """
    Histogram of chi-square values from repeated fits, over the chi-square distribution.

    The curve is the distribution for ``ndof``, not a fit. The mean and standard deviation of
    the values are printed in the axes.

    Parameters
    ----------
    ax : Axes
        Axes to draw on.
    chi2_values : np.ndarray
        Chi-square of each repeated fit, 1-D.
    ndof : int
        Degrees of freedom of every fit. Must be positive.

    Returns
    -------
    Axes
        ``ax``, for chaining.
    """
    if ndof <= 0:
        raise ValueError(f"ndof must be positive, got {ndof}")

    values = np.asarray(chi2_values, dtype=float)
    ax.hist(values, bins=50, density=True, histtype="step", label="Simulated")
    grid = np.linspace(0.01, max(values.max(), 2 * ndof), 500)
    ax.plot(
        grid,
        stats.chi2.pdf(grid, df=ndof),
        label=f"$\\chi^2$ distribution, $N_\\mathrm{{dof}}$ = {ndof}",
    )
    ax.text(
        0.97,
        0.5,
        f"Entries {len(values)}\nMean {values.mean():.3f}\nStd {values.std(ddof=1):.3f}",
        transform=ax.transAxes,
        ha="right",
        family="monospace",
    )
    ax.set(xlabel="$\\chi^2$")
    ax.legend()
    return ax


def plot_pulls(ax: Axes, x: np.ndarray, pulls: np.ndarray, limit: float = 2.0) -> Axes:
    """
    Bar chart of pulls, with the bars beyond ``±limit`` in red.

    Parameters
    ----------
    ax : Axes
        Axes to draw on.
    x : np.ndarray
        Position of each bar, 1-D.
    pulls : np.ndarray
        Pull of each point, 1-D, dimensionless, same length as ``x``.
    limit : float
        Pulls with ``abs(pull) > limit`` are drawn in red.

    Returns
    -------
    Axes
        ``ax``, for chaining.
    """
    pulls = np.asarray(pulls, dtype=float)
    colours = np.where(np.abs(pulls) > limit, "tab:red", "tab:blue")
    ax.bar(x, pulls, color=list(colours))
    ax.axhline(0, color="k", lw=1)
    ax.set(xlabel="Point", ylabel="Pull")
    return ax
