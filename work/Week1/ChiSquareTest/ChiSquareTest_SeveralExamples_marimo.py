import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from appstat.fitting import fit_least_squares, polynomial_chi2
    from appstat.plotting import plot_chi2_distribution, plot_fit

    return fit_least_squares, mo, np, plot_chi2_distribution, plot_fit, plt, polynomial_chi2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The $\chi^2$ distribution for three different fits

    The $\chi^2$ is a goodness-of-fit measure, and it follows the $\chi^2$ distribution. Here are three
    examples: a **line**, an **oscillation**, and an **exponential fit to a histogram**. They look very
    different, yet they share one thing, the number of degrees of freedom: 17 points − 2 parameters,
    19 points − 4 parameters, and 17 bins − 2 parameters are all **15**.

    Every interactive block below has its controls directly above its plots. Each block fits one
    data set (left) and many repeated data sets (right).

    References: Barlow chapter 6, Cowan chapters 2.7 and 7, Bevington chapter 6.

    ---

    **Adapted from** the ChiSquare Several Examples exercise by **Troels C. Petersen** (Niels Bohr
    Institute, 10th of November 2025), from the course *Applied Statistics 2025/26*. The original is
    an exercise. This version keeps its three examples and its numbers, and is rewritten as an
    interactive demonstration.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. A line

    17 points $y = \alpha_0 + \alpha_1 x$ with Gaussian noise, fitted with a line. All points share one
    uncertainty, so the $\chi^2$ of every repetition comes from one matrix operation
    (`polynomial_chi2`), and one data set is fitted with Minuit to draw the fit.
    """)
    return


@app.cell
def _(mo):
    lin_points = mo.ui.slider(4, 50, 1, value=17, show_value=True, label="Points")
    lin_sigma = mo.ui.slider(
        0.1, 2.0, 0.1, value=0.4, show_value=True, label="Uncertainty on each point"
    )
    lin_repeats = mo.ui.dropdown(
        {"10": 10, "100": 100, "1 000": 1_000, "10 000": 10_000},
        value="1 000",
        label="Repeat the experiment",
    )
    lin_seed = mo.ui.number(0, 9999, 1, value=42, label="Random seed")
    return lin_points, lin_repeats, lin_seed, lin_sigma


@app.cell
def _(
    fit_least_squares,
    lin_points,
    lin_repeats,
    lin_seed,
    lin_sigma,
    mo,
    np,
    plot_chi2_distribution,
    plot_fit,
    plt,
    polynomial_chi2,
):
    def _line(x, alpha0, alpha1):
        return alpha0 + alpha1 * x

    _x = np.arange(lin_points.value) + 1.0
    _y = (
        3.6
        + 0.3 * _x
        + np.random.default_rng(lin_seed.value).normal(
            0, lin_sigma.value, size=(lin_repeats.value, lin_points.value)
        )
    )
    _chi2 = polynomial_chi2(_y, _x, lin_sigma.value, 1)
    _fit = fit_least_squares(
        _line, _x, _y[0], np.full_like(_x, lin_sigma.value), {"alpha0": 1.0, "alpha1": 1.0}
    )

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    plot_fit(_ax1, _x, _y[0], np.full_like(_x, lin_sigma.value), _line, _fit)
    _ax1.set(xlabel="x", ylabel="y", title="One experiment")
    _ndof = lin_points.value - 2
    if lin_repeats.value >= 10:
        plot_chi2_distribution(_ax2, _chi2, _ndof)
        _ax2.set_title("χ² of all the experiments")
    else:
        _ax2.text(
            0.5,
            0.5,
            "Repeat the experiment 10 times or more",
            ha="center",
            transform=_ax2.transAxes,
        )
    _fig.tight_layout()
    mo.vstack(
        [
            mo.hstack(
                [mo.vstack([lin_points, lin_sigma]), mo.vstack([lin_repeats, lin_seed])],
                widths="equal",
            ),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. An oscillation

    19 points $y = m + A\cos(\omega x + \phi) + \text{noise}$, fitted with the same function by Minuit.
    Unlike the line, this fit is not linear, so Minuit needs **starting values**. With the true values it
    finds the minimum every time. Switch to *Poor*: all four start at 0, and many fits get stuck in a
    wrong minimum, which shows up as a long tail in the $\chi^2$ distribution and as fits that did not converge.
    Each repetition is a real Minuit fit, so 1 000 repetitions take several seconds.
    """)
    return


@app.cell
def _(mo):
    osc_start = mo.ui.dropdown(
        {"Good: the true values": "true", "Poor: all zero": "zero"},
        value="Good: the true values",
        label="Starting values",
    )
    osc_repeats = mo.ui.dropdown(
        {"10": 10, "100": 100, "1 000": 1_000}, value="100", label="Repeat the experiment"
    )
    osc_seed = mo.ui.number(0, 9999, 1, value=42, label="Random seed")
    return osc_repeats, osc_seed, osc_start


@app.cell
def _(
    fit_least_squares,
    mo,
    np,
    osc_repeats,
    osc_seed,
    osc_start,
    plot_chi2_distribution,
    plot_fit,
    plt,
):
    def _oscillation(x, mean, amp, omega, phase):
        return mean + amp * np.cos(omega * x + phase)

    _truth = {"mean": 1.6, "amp": 3.3, "omega": 0.7, "phase": 0.3}
    _start = _truth if osc_start.value == "true" else dict.fromkeys(_truth, 0.0)
    _sigma = 0.5
    _x = np.arange(19) + 1.0
    _sy = np.full_like(_x, _sigma)
    _rng = np.random.default_rng(osc_seed.value)
    _fits = []
    for _ in range(osc_repeats.value):  # one Minuit fit per repetition
        _y = _oscillation(_x, **_truth) + _rng.normal(0, _sigma, size=len(_x))
        _fits.append((_y, fit_least_squares(_oscillation, _x, _y, _sy, _start)))
    _chi2 = np.array([fit.fval for _, fit in _fits])
    _failed = sum(not fit.valid for _, fit in _fits)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    plot_fit(_ax1, _x, _fits[0][0], _sy, _oscillation, _fits[0][1])
    _ax1.set(xlabel="x", ylabel="y", title="One experiment")
    if osc_repeats.value >= 10:
        plot_chi2_distribution(_ax2, _chi2, 15)
        _ax2.set_title("χ² of all the experiments")
    _fig.tight_layout()
    mo.vstack(
        [
            mo.hstack([osc_start, osc_repeats, osc_seed], justify="start", gap=2),
            _fig,
            mo.callout(
                mo.md(
                    f"{_failed} of {osc_repeats.value} fits did not report a valid minimum. "
                    f"Mean χ² = {_chi2.mean():.1f} (expected 15 for a good fit every time)."
                ),
                kind="success" if _failed == 0 else "warn",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. A histogram of exponential data

    Draw numbers from an exponential distribution with lifetime $\tau$, fill 17 bins, and fit
    $N/\tau\,e^{-x/\tau}$. Each bin has the uncertainty $\sqrt{n}$ (where does it come from? the counts are
    Poisson), and empty bins are left out. With many entries per bin that is fine. The red curve is
    **not** a perfect match: make the histogram **emptier**, with fewer entries or a short lifetime, and
    watch the $\chi^2$ distribution move away from the curve. The $\sqrt{n}$ uncertainty is poor when $n$ is small.
    """)
    return


@app.cell
def _(mo):
    exp_entries = mo.ui.slider(
        100, 5000, 100, value=1000, show_value=True, label="Entries per histogram"
    )
    exp_tau = mo.ui.slider(1.5, 6.0, 0.1, value=3.1, show_value=True, label="Lifetime τ")
    exp_repeats = mo.ui.dropdown(
        {"10": 10, "100": 100, "1 000": 1_000}, value="100", label="Repeat the experiment"
    )
    exp_seed = mo.ui.number(0, 9999, 1, value=42, label="Random seed")
    return exp_entries, exp_repeats, exp_seed, exp_tau


@app.cell
def _(
    exp_entries,
    exp_repeats,
    exp_seed,
    exp_tau,
    fit_least_squares,
    mo,
    np,
    plot_chi2_distribution,
    plot_fit,
    plt,
):
    def _exponential(x, tau, N):
        return N / tau * np.exp(-x / tau)

    _n_bins = 17
    _rng = np.random.default_rng(exp_seed.value)
    _chi2 = []
    for _ in range(exp_repeats.value):  # one Minuit fit per histogram
        _data = _rng.exponential(exp_tau.value, exp_entries.value)
        _y, _edges = np.histogram(_data, bins=_n_bins, range=(0, _n_bins))
        _x = 0.5 * (_edges[1:] + _edges[:-1])
        _nonempty = _y > 0
        _fit = fit_least_squares(
            _exponential,
            _x[_nonempty],
            _y[_nonempty],
            np.sqrt(_y[_nonempty]),
            {"tau": exp_tau.value, "N": float(exp_entries.value)},
        )
        _chi2.append(_fit.fval)
        if len(_chi2) == 1:
            _first = (_x[_nonempty], _y[_nonempty], _fit)
    _chi2 = np.array(_chi2)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    plot_fit(_ax1, _first[0], _first[1], np.sqrt(_first[1]), _exponential, _first[2])
    _ax1.set(xlabel="x", ylabel="Entries per bin", title="One histogram")
    if exp_repeats.value >= 10:
        plot_chi2_distribution(_ax2, _chi2, _n_bins - 2)
        _ax2.set_title("χ² of all the histograms")
    _fig.tight_layout()
    mo.vstack(
        [
            mo.hstack(
                [mo.vstack([exp_entries, exp_tau]), mo.vstack([exp_repeats, exp_seed])],
                widths="equal",
            ),
            _fig,
            mo.md(
                f"Mean χ² = **{_chi2.mean():.1f}** and width {_chi2.std(ddof=1):.1f} "
                f"(the χ² distribution with 15 degrees of freedom has mean 15 and width {np.sqrt(30):.1f})."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to take away

    1. How to run a $\chi^2$ fit in three different situations, count the degrees of freedom (points or
       bins minus fitted parameters), and read off the p-value.
    2. With the same number of degrees of freedom, three very different fits give the same $\chi^2$
       distribution, **if** the fit function is right, the uncertainties are right and the minimiser finds the minimum.
    3. Each of those conditions can fail, and the $\chi^2$ distribution shows it: a poor start for the
       oscillation, or poor $\sqrt{n}$ uncertainties for a sparse histogram.
    """)
    return


if __name__ == "__main__":
    app.run()
