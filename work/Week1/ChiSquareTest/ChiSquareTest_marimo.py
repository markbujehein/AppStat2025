import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from iminuit import Minuit
    from iminuit.cost import LeastSquares
    from scipy import stats

    from appstat.stats import chi2_pvalue

    return LeastSquares, Minuit, chi2_pvalue, mo, np, plt, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The $\chi^2$ as a measure of how good a fit is

    We generate data points along a line (or a parabola), add Gaussian noise, and fit them.
    The $\chi^2$ of the fit tells us how well the fit function describes the data, and the
    p-value turns that into a probability. Is it trustworthy?

    The best way to find out is to **repeat the experiment many times** and look at how the
    fit results, the $\chi^2$ and the p-value are distributed. Change the settings below and
    see what follows.

    References: Barlow chapter 6, Cowan chapters 2.7 and 7, Bevington chapter 6.

    ---

    **Adapted from** `ChiSquareTest_original.ipynb` (10th of November 2025) by
    **Troels C. Petersen** (Niels Bohr Institute), from the course *Applied Statistics
    2025/26*. The original is an exercise. This version keeps its data generation and its
    observations, and is rewritten as an interactive demonstration.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Generate data and choose a fit
    """)
    return


@app.cell
def _(mo):
    n_points = mo.ui.slider(4, 50, 1, value=9, label="Points per data set")
    alpha0 = mo.ui.slider(0.0, 10.0, 0.1, value=3.6, label="Constant term α0")
    alpha1 = mo.ui.slider(-1.0, 1.0, 0.05, value=0.3, label="Linear term α1")
    alpha2 = mo.ui.slider(-0.1, 0.1, 0.005, value=0.0, label="Quadratic term α2 (true curve)")
    sigma_y = mo.ui.slider(0.1, 3.0, 0.1, value=1.0, label="Uncertainty on each point")
    fit_model = mo.ui.dropdown(
        {"Line (2 parameters)": "line", "Parabola (3 parameters)": "parabola"},
        value="Line (2 parameters)",
        label="Fit with",
    )
    n_experiments = mo.ui.dropdown(
        {"1": 1, "10": 10, "100": 100, "1 000": 1_000, "10 000": 10_000},
        value="1",
        label="Repeat the experiment",
    )
    seed = mo.ui.number(0, 9999, 1, value=42, label="Random seed")
    mo.hstack(
        [
            mo.vstack([n_points, alpha0, alpha1, alpha2]),
            mo.vstack([sigma_y, fit_model, n_experiments, seed]),
        ],
        widths="equal",
    )
    return (
        alpha0,
        alpha1,
        alpha2,
        fit_model,
        n_experiments,
        n_points,
        seed,
        sigma_y,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The data are the true curve plus a Gaussian random number of width $\sigma_y$ on each point.
    All points have the same uncertainty, so the least-squares fit has a direct solution
    (Barlow 6.2): one matrix operation fits **all experiments at once**, with no loop. That is
    why thousands of repetitions are cheap here.
    """)
    return


@app.cell
def _(
    alpha0,
    alpha1,
    alpha2,
    fit_model,
    n_experiments,
    n_points,
    np,
    seed,
    sigma_y,
    stats,
):
    rng = np.random.default_rng(seed.value)
    x = np.arange(n_points.value) + 1.0
    truth = alpha0.value + alpha1.value * x + alpha2.value * x**2
    y = truth + rng.normal(0, sigma_y.value, size=(n_experiments.value, n_points.value))
    sy = np.full_like(x, sigma_y.value)

    degree = 1 if fit_model.value == "line" else 2
    design = np.vander(x, degree + 1, increasing=True)  # columns 1, x, (x²)
    coefficients = np.linalg.pinv(design) @ y.T  # (degree + 1, n_experiments)
    chi2_all = np.sum(((y - (design @ coefficients).T) / sy) ** 2, axis=1)
    ndof = n_points.value - (degree + 1)  # data points minus fitted parameters
    prob_all = stats.chi2.sf(chi2_all, ndof)
    return chi2_all, coefficients, degree, ndof, prob_all, sy, truth, x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. One fit

    Below is the first experiment, fitted with Minuit (it minimises the $\chi^2$ by trial and
    error, and also gives the errors on the parameters). It finds the same values as the direct
    solution above.
    """)
    return


@app.cell
def _(LeastSquares, Minuit, chi2_pvalue, degree, ndof, sy, x, y):
    def line(x, alpha0, alpha1):
        return alpha0 + alpha1 * x

    def parabola(x, alpha0, alpha1, alpha2):
        return alpha0 + alpha1 * x + alpha2 * x**2

    if degree == 1:
        fit_function, start = line, {"alpha0": 3.0, "alpha1": 0.0}
    else:
        fit_function, start = parabola, {"alpha0": 3.0, "alpha1": 0.0, "alpha2": 0.0}

    minuit = Minuit(LeastSquares(x, y[0], sy, fit_function), **start)
    minuit.migrad()

    chi2_first = minuit.fval
    prob_first = chi2_pvalue(chi2_first, ndof)
    return chi2_first, fit_function, minuit, prob_first


@app.cell
def _(chi2_first, fit_function, minuit, ndof, plt, prob_first, sy, truth, x, y):
    _fig, _ax = plt.subplots(figsize=(9, 5))
    _ax.errorbar(x, y[0], sy, fmt="ko", capsize=3, label="Data")
    _ax.plot(x, truth, "--", color="gray", label="True curve")
    _ax.plot(x, fit_function(x, *minuit.values), "r-", label="Fit")
    _ax.set(xlabel="x", ylabel="y")
    _ax.legend(loc="upper left")
    _ax.text(
        0.97,
        0.05,
        f"χ² = {chi2_first:.2f}\nNdof = {ndof}\np = {prob_first:.3f}",
        family="monospace",
        ha="right",
        transform=_ax.transAxes,
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo, chi2_first, ndof, prob_first):
    mo.md(rf"""
    The fit has $\chi^2$ = {chi2_first:.2f} for {ndof} degrees of freedom (data points minus
    fitted parameters). The p-value, **{prob_first:.3f}**, is the probability of getting a
    $\chi^2$ this large or larger **if the fit function really describes the data**. A very
    small p-value says it probably does not.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Many experiments

    Set "Repeat the experiment" to 1 000 or more. The four panels show how the fit
    parameters, the $\chi^2$ and the p-value distribute over the repetitions.
    """)
    return


@app.cell
def _(
    alpha0,
    alpha1,
    chi2_all,
    coefficients,
    mo,
    n_experiments,
    ndof,
    np,
    plt,
    prob_all,
    stats,
):
    mo.stop(
        n_experiments.value < 10,
        mo.callout(
            mo.md("Increase *Repeat the experiment* to see the distributions."), kind="info"
        ),
    )

    _fig, _axes = plt.subplots(2, 2, figsize=(13, 9))
    _ax0, _ax1, _ax2, _ax3 = _axes.flatten()

    def _stats_text(values):
        return f"mean {values.mean():.3f}\nstd {values.std(ddof=1):.3f}"

    _ax0.hist(coefficients[0], bins=60, histtype="step")
    _ax0.axvline(alpha0.value, color="gray", ls="--", label="true α0")
    _ax0.set(title="Fitted α0")
    _ax0.text(0.03, 0.95, _stats_text(coefficients[0]), va="top", transform=_ax0.transAxes)

    _ax1.hist(coefficients[1], bins=60, histtype="step")
    _ax1.axvline(alpha1.value, color="gray", ls="--", label="true α1")
    _ax1.set(title="Fitted α1")
    _ax1.text(0.03, 0.95, _stats_text(coefficients[1]), va="top", transform=_ax1.transAxes)

    _ax2.hist(chi2_all, bins=60, density=True, histtype="step", label="Simulated")
    _grid = np.linspace(0, max(chi2_all.max(), ndof * 2), 500)
    _ax2.plot(_grid, stats.chi2.pdf(_grid, df=ndof), "r-", label=f"χ² distribution, Ndof = {ndof}")
    _ax2.set(title="χ² of the fits (red curve is not a fit)")
    _ax2.legend()
    _ax2.text(0.97, 0.5, _stats_text(chi2_all), ha="right", transform=_ax2.transAxes)

    _ax3.hist(prob_all, bins=20, range=(0, 1), histtype="step")
    _ax3.axhline(len(prob_all) / 20, color="r", ls="--", label="Flat")
    _ax3.set(title="p-values")
    _ax3.legend()

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If the fit function is correct, the $\chi^2$ follows the $\chi^2$ distribution with
    Ndof degrees of freedom (mean Ndof, width $\sqrt{2\,\mathrm{Ndof}}$), and the p-values are
    **flat** between 0 and 1. Flat is the signature of a correct hypothesis with correct
    uncertainties. A pile-up near 0 means the fit function or the uncertainties are wrong.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. The rule of thumb $\chi^2/\mathrm{Ndof} \approx 1$

    A reduced $\chi^2$ of "about 1" is a good sign, but "about" depends on how many degrees of
    freedom there are. The $\chi^2$ distribution has mean Ndof and width $\sqrt{2\,\mathrm{Ndof}}$,
    so its **relative** width shrinks as Ndof grows. Pick a reduced $\chi^2$ and see how the
    p-value changes with Ndof.
    """)
    return


@app.cell
def _(mo):
    reduced_chi2 = mo.ui.slider(0.5, 3.0, 0.05, value=1.5, label="Reduced χ² (χ²/Ndof)")
    ndof_demo = mo.ui.slider(
        steps=[1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000], value=10, label="Ndof"
    )
    mo.hstack([reduced_chi2, ndof_demo], widths="equal")
    return ndof_demo, reduced_chi2


@app.cell
def _(chi2_pvalue, mo, ndof_demo, np, plt, reduced_chi2, stats):
    _p = chi2_pvalue(reduced_chi2.value * ndof_demo.value, ndof_demo.value)

    _ndofs = np.unique(np.logspace(0, 3, 200).astype(int))
    _fig, _ax = plt.subplots(figsize=(9, 4))
    _ax.semilogx(_ndofs, stats.chi2.sf(reduced_chi2.value * _ndofs, _ndofs), "k")
    _ax.axhline(0.05, color="gray", ls="--", label="p = 0.05")
    _ax.plot(ndof_demo.value, _p, "ro", label=f"Ndof = {ndof_demo.value}, p = {_p:.3f}")
    _ax.set(xlabel="Ndof", ylabel="p-value", title=f"Reduced χ² = {reduced_chi2.value:.2f}")
    _ax.legend()
    mo.vstack([_fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to take away

    1. The $\chi^2$ is the quantity we minimise to fit, and it also measures how good the fit is.
    2. A weighted mean is a $\chi^2$ fit with a constant. With Gaussian uncertainties the $\chi^2$
       is equivalent to the (optimal) likelihood.
    3. Ndof is the number of data points minus the number of fitted parameters.
    4. The p-value is *the probability of a worse $\chi^2$, provided the fit function describes the
       data*. For a correct hypothesis, p-values are flat between 0 and 1.
    5. $\chi^2$/Ndof is **only a rule of thumb**: what counts as "close to 1" depends on Ndof.

    Likelihoods and hypothesis testing come next.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Additional work

    Four more investigations. A uses the first experiment generated above. B, C and D generate their
    own data sets, with their controls placed right under their text.

    ## A. Could the data come from a flat line?

    Take the first experiment and ask whether the slope is needed. Three answers:

    1. fit a **constant** and look at its own p-value: is a flat line acceptable at all?
    2. test the **slope** directly: the fitted slope divided by its uncertainty, which is about
       a standard normal number when the true slope is zero,
    3. the **difference in $\chi^2$** between the constant and the line, which has 1 degree of freedom
       because the line has one extra parameter.

    For a straight line, 2 and 3 are the same test written two ways.
    """)
    return


@app.cell
def _(chi2_of_fit, chi2_pvalue, mo, n_points, np, sigma_y, stats, x, y):
    _chi2_const = chi2_of_fit(y[:1], x, sigma_y.value, 0)[0]
    _chi2_line = chi2_of_fit(y[:1], x, sigma_y.value, 1)[0]
    _slope = np.polyfit(x, y[0], 1)[0]
    _slope_error = sigma_y.value / np.sqrt(np.sum((x - x.mean()) ** 2))
    _delta = max(_chi2_const - _chi2_line, 0.0)
    mo.md(rf"""
    | test | statistic | p-value |
    |---|---|---|
    | flat line, on its own | χ² = {_chi2_const:.2f}, Ndof = {n_points.value - 1} | {chi2_pvalue(_chi2_const, n_points.value - 1):.4f} |
    | slope / its uncertainty | {_slope / _slope_error:.2f} | {2 * stats.norm.sf(abs(_slope / _slope_error)):.4f} |
    | χ²(constant) − χ²(line) | {_delta:.2f}, 1 degree of freedom | {chi2_pvalue(_delta, 1):.4f} |

    The slope test is the sharper tool when the question is "is there a slope?", because it
    spends all its power on that one question. The flat-line $\chi^2$ also reacts to any other mismatch.
    """)
    return


@app.cell
def _(np):
    def chi2_of_fit(y, x, sigma, degree):
        """χ² of the least-squares polynomial fit of every row of y (equal uncertainties)."""
        design = np.vander(x, degree + 1, increasing=True)
        coefficients = np.linalg.pinv(design) @ y.T
        return np.sum(((y - (design @ coefficients).T) / sigma) ** 2, axis=1)

    return (chi2_of_fit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## B. Fix the fit

    Fit every experiment with a constant, a line and a parabola, and look at the p-value
    histograms. The red dashed line is what flat looks like. The two controls directly below belong
    to this test only: the quadratic term of the true curve and the number of repetitions. With the
    quadratic term at 0 the line and the parabola are both flat. At −0.04 the line piles up at small
    p-values, and only the parabola, the function that actually generated the data, is flat again.
    """)
    return


@app.cell
def _(alpha0, alpha1, n_points, np, seed, sigma_y, x):
    def simulate(n_repeats, alpha2, offset):
        """n_repeats noisy data sets (one per row) around alpha0 + alpha1 x + alpha2 x²."""
        _rng = np.random.default_rng(seed.value + offset)
        curve = alpha0.value + alpha1.value * x + alpha2 * x**2
        return curve + _rng.normal(0, sigma_y.value, size=(n_repeats, n_points.value))

    return (simulate,)


@app.cell
def _(mo):
    b_alpha2 = mo.ui.slider(-0.1, 0.1, 0.005, value=-0.04, label="Quadratic term α2 (true curve)")
    b_repeats = mo.ui.dropdown(
        {"100": 100, "1 000": 1_000, "10 000": 10_000}, value="1 000", label="Repeat the experiment"
    )
    mo.hstack([b_alpha2, b_repeats], widths="equal")
    return b_alpha2, b_repeats


@app.cell
def _(b_alpha2, b_repeats, chi2_of_fit, n_points, plt, sigma_y, simulate, stats, x):
    _y = simulate(b_repeats.value, b_alpha2.value, 1)
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for _ax, _degree, _label in zip(
        _axes, (0, 1, 2), ("Constant", "Line", "Parabola"), strict=True
    ):
        _p = stats.chi2.sf(chi2_of_fit(_y, x, sigma_y.value, _degree), n_points.value - _degree - 1)
        _ax.hist(_p, bins=20, range=(0, 1), histtype="step")
        _ax.axhline(len(_p) / 20, color="r", ls="--")
        _ax.set(title=f"{_label} fit", xlabel="p-value")
    _axes[0].set(ylabel="Experiments")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## C. Wrong uncertainties

    Fit a line to data that really come from a straight line (this test generates its own data; choose
    the number of repetitions below), but pretend the uncertainty on each point is 20 % too small, or
    20 % too large. The fitted line does not move (all points are weighted equally), but the $\chi^2$
    scales as $1/\text{factor}^2$. Too small an uncertainty gives too many tiny p-values, too large an
    uncertainty gives too many values near 1. A p-value histogram that is not flat can therefore mean a
    wrong function **or** wrong uncertainties.
    """)
    return


@app.cell
def _(mo):
    c_repeats = mo.ui.dropdown(
        {"100": 100, "1 000": 1_000, "10 000": 10_000}, value="1 000", label="Repeat the experiment"
    )
    c_repeats
    return (c_repeats,)


@app.cell
def _(c_repeats, chi2_of_fit, n_points, plt, sigma_y, simulate, stats, x):
    _chi2_true = chi2_of_fit(simulate(c_repeats.value, 0.0, 3), x, sigma_y.value, 1)
    _fig, _ax = plt.subplots(figsize=(8, 4))
    for _factor in (0.8, 1.0, 1.2):
        _p = stats.chi2.sf(_chi2_true / _factor**2, n_points.value - 2)
        _ax.hist(
            _p, bins=20, range=(0, 1), histtype="step", label=f"assumed uncertainty × {_factor}"
        )
    _ax.set(xlabel="p-value", ylabel="Experiments")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## D. Non-Gaussian noise

    Replace the Gaussian noise with **uniform** noise of the same standard deviation (half-width
    $\sqrt{3}\,\sigma$) and fit 5 000 data sets with a line. The data come from a straight line, so
    the line is the correct function. The $\chi^2$ distribution and the p-value histogram show how
    sensitive the test is to the shape of the noise, and the answer is: not very, because each $\chi^2$ is a sum
    of many small contributions (the central limit theorem).
    """)
    return


@app.cell
def _(alpha0, alpha1, chi2_of_fit, n_points, np, plt, seed, sigma_y, stats, x):
    _rng = np.random.default_rng(seed.value + 2)
    _half_width = np.sqrt(3) * sigma_y.value
    _y = (
        alpha0.value
        + alpha1.value * x
        + _rng.uniform(-_half_width, _half_width, size=(5_000, n_points.value))
    )
    _ndof = n_points.value - 2
    _chi2 = chi2_of_fit(_y, x, sigma_y.value, 1)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4))
    _ax1.hist(_chi2, bins=60, density=True, histtype="step", label="Uniform noise")
    _grid = np.linspace(0.01, _chi2.max(), 400)
    _ax1.plot(_grid, stats.chi2.pdf(_grid, _ndof), "r", label=f"χ² distribution, Ndof = {_ndof}")
    _ax1.set(xlabel="χ²")
    _ax1.legend()
    _ax2.hist(stats.chi2.sf(_chi2, _ndof), bins=20, range=(0, 1), histtype="step")
    _ax2.axhline(len(_chi2) / 20, color="r", ls="--")
    _ax2.set(xlabel="p-value")
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
