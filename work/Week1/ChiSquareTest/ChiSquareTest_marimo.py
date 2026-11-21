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
    fit results, the $\chi^2$ and the p-value are distributed. Every interactive block below has its
    controls directly above its plots. Change them and see what follows.

    References: Barlow chapter 6, Cowan chapters 2.7 and 7, Bevington chapter 6.

    ---

    **Adapted from** `ChiSquareTest_original.ipynb` (10th of November 2025) by
    **Troels C. Petersen** (Niels Bohr Institute), from the course *Applied Statistics
    2025/26*. The original is an exercise. This version keeps its data generation and its
    observations, and is rewritten as an interactive demonstration.
    """)
    return


@app.cell
def _(np):
    DATA = "#0072B2"
    FIT = "#D55E00"
    FLAG = "#CC79A7"

    def simulate(n_repeats, n_points, alpha0, alpha1, alpha2, sigma, seed, noise="Gaussian"):
        """
        ``n_repeats`` noisy data sets (one per row) around ``alpha0 + alpha1 x + alpha2 x²``.

        The noise has standard deviation ``sigma`` whatever its shape. Returns ``x`` (positions
        1..n_points), the true curve and the data.
        """
        rng = np.random.default_rng(seed)
        x = np.arange(n_points) + 1.0
        truth = alpha0 + alpha1 * x + alpha2 * x**2
        shape = (n_repeats, n_points)
        if noise == "Uniform":
            scatter = rng.uniform(-np.sqrt(3) * sigma, np.sqrt(3) * sigma, size=shape)
        elif noise == "Exponential (skewed)":
            scatter = sigma * (rng.exponential(1.0, size=shape) - 1.0)
        else:
            scatter = rng.normal(0, sigma, size=shape)
        return x, truth, truth + scatter

    def chi2_of_fit(y, x, sigma, degree):
        """χ² of the least-squares polynomial fit of every row of y (equal uncertainties)."""
        design = np.vander(x, degree + 1, increasing=True)
        coefficients = np.linalg.pinv(design) @ y.T  # (degree + 1, n_repeats)
        return np.sum(((y - (design @ coefficients).T) / sigma) ** 2, axis=1)

    return DATA, FIT, FLAG, chi2_of_fit, simulate


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Generate data, fit it, and repeat

    The data are the true curve plus a Gaussian random number of width $\sigma_y$ on each point.
    The fit is done twice: by Minuit (it minimises the $\chi^2$ by trial and error, and also gives the
    errors on the parameters) for the first experiment, and by a direct solution for all experiments at
    once. With equal uncertainties the least-squares fit has a direct solution (Barlow 6.2): one
    matrix operation fits **all experiments with no loop**. That is why thousands of repetitions are cheap.

    The p-value is the probability of getting a $\chi^2$ this large or larger **if the fit function
    really describes the data**. A very small p-value says it probably does not.

    Set *Repeat the experiment* to 1 000 or more: the four plots in the bottom row show how the fit
    parameters, the $\chi^2$ and the p-value distribute over the repetitions. If the fit function is
    correct, the $\chi^2$ follows the $\chi^2$ distribution with Ndof degrees of freedom (mean Ndof, width
    $\sqrt{2\,\mathrm{Ndof}}$), and the p-values are **flat** between 0 and 1. A pile-up near 0 means the
    fit function or the uncertainties are wrong.
    """)
    return


@app.cell
def _(mo):
    n_points = mo.ui.slider(4, 50, 1, value=9, show_value=True, label="Points per data set")
    alpha0 = mo.ui.slider(0.0, 10.0, 0.1, value=3.6, show_value=True, label="Constant term α0")
    alpha1 = mo.ui.slider(-1.0, 1.0, 0.05, value=0.3, show_value=True, label="Linear term α1")
    alpha2 = mo.ui.slider(
        -0.1, 0.1, 0.005, value=0.0, show_value=True, label="Quadratic term α2 (true curve)"
    )
    sigma_y = mo.ui.slider(
        0.1, 3.0, 0.1, value=1.0, show_value=True, label="Uncertainty on each point"
    )
    fit_model = mo.ui.dropdown(
        {"Line (2 parameters)": "line", "Parabola (3 parameters)": "parabola"},
        value="Line (2 parameters)",
        label="Fit with",
    )
    n_experiments = mo.ui.dropdown(
        {"1": 1, "10": 10, "100": 100, "1 000": 1_000, "10 000": 10_000},
        value="1 000",
        label="Repeat the experiment",
    )
    seed = mo.ui.number(0, 9999, 1, value=42, label="Random seed")
    return alpha0, alpha1, alpha2, fit_model, n_experiments, n_points, seed, sigma_y


@app.cell
def _(
    DATA,
    FIT,
    LeastSquares,
    Minuit,
    alpha0,
    alpha1,
    alpha2,
    chi2_pvalue,
    fit_model,
    mo,
    n_experiments,
    n_points,
    np,
    plt,
    seed,
    sigma_y,
    simulate,
    stats,
):
    _x, _truth, _y = simulate(
        n_experiments.value,
        n_points.value,
        alpha0.value,
        alpha1.value,
        alpha2.value,
        sigma_y.value,
        seed.value,
    )
    _sy = np.full_like(_x, sigma_y.value)
    _degree = 1 if fit_model.value == "line" else 2
    _design = np.vander(_x, _degree + 1, increasing=True)  # columns 1, x, (x²)
    _coefficients = np.linalg.pinv(_design) @ _y.T  # (degree + 1, n_experiments)
    _chi2_all = np.sum(((_y - (_design @ _coefficients).T) / _sy) ** 2, axis=1)
    _ndof = n_points.value - (_degree + 1)  # data points minus fitted parameters
    _prob_all = stats.chi2.sf(_chi2_all, _ndof)

    def _line(x, alpha0, alpha1):
        return alpha0 + alpha1 * x

    def _parabola(x, alpha0, alpha1, alpha2):
        return alpha0 + alpha1 * x + alpha2 * x**2

    if _degree == 1:
        _fit_function, _start = _line, {"alpha0": 3.0, "alpha1": 0.0}
    else:
        _fit_function, _start = _parabola, {"alpha0": 3.0, "alpha1": 0.0, "alpha2": 0.0}
    _minuit = Minuit(LeastSquares(_x, _y[0], _sy, _fit_function), **_start)
    _minuit.migrad()
    _chi2_first = _minuit.fval
    _prob_first = chi2_pvalue(_chi2_first, _ndof)

    _fig1, _ax = plt.subplots(figsize=(9, 4.2))
    _ax.errorbar(_x, _y[0], _sy, fmt="o", color=DATA, capsize=3, label="Data (first experiment)")
    _ax.plot(_x, _truth, "--", color="gray", label="True curve")
    _ax.plot(_x, _fit_function(_x, *_minuit.values), color=FIT, lw=2, label="Fit")
    _ax.set(xlabel="x", ylabel="y")
    _ax.legend(loc="upper left")
    _ax.text(
        0.97,
        0.05,
        f"χ² = {_chi2_first:.2f}\nNdof = {_ndof}\np = {_prob_first:.3f}",
        family="monospace",
        ha="right",
        transform=_ax.transAxes,
    )
    _fig1.tight_layout()

    if n_experiments.value < 10:
        _many = mo.callout(
            mo.md("Increase *Repeat the experiment* to see the distributions."), kind="info"
        )
    else:

        def _stats_text(values):
            return f"mean {values.mean():.3f}\nstd {values.std(ddof=1):.3f}"

        _fig2, (_a0, _a1, _a2, _a3) = plt.subplots(1, 4, figsize=(16, 3.8))
        _a0.hist(_coefficients[0], bins=60, histtype="step", color=DATA)
        _a0.axvline(alpha0.value, color="gray", ls="--")
        _a0.set(title="Fitted α0 (dashed: true)")
        _a0.text(0.03, 0.95, _stats_text(_coefficients[0]), va="top", transform=_a0.transAxes)
        _a1.hist(_coefficients[1], bins=60, histtype="step", color=DATA)
        _a1.axvline(alpha1.value, color="gray", ls="--")
        _a1.set(title="Fitted α1 (dashed: true)")
        _a1.text(0.03, 0.95, _stats_text(_coefficients[1]), va="top", transform=_a1.transAxes)
        _a2.hist(_chi2_all, bins=60, density=True, histtype="step", color=DATA, label="Simulated")
        _grid = np.linspace(0, max(_chi2_all.max(), _ndof * 2), 500)
        _a2.plot(
            _grid,
            stats.chi2.pdf(_grid, df=_ndof),
            color=FIT,
            label=f"χ², Ndof = {_ndof} (not a fit)",
        )
        _a2.set(title="χ² of the fits")
        _a2.legend()
        _a2.text(0.97, 0.5, _stats_text(_chi2_all), ha="right", transform=_a2.transAxes)
        _a3.hist(_prob_all, bins=20, range=(0, 1), histtype="step", color=DATA)
        _a3.axhline(len(_prob_all) / 20, color=FIT, ls="--", label="Flat")
        _a3.set(title="p-values")
        _a3.legend()
        _fig2.tight_layout()
        _many = _fig2

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack([n_points, alpha0, alpha1, alpha2]),
                    mo.vstack([sigma_y, fit_model, n_experiments, seed]),
                ],
                widths="equal",
            ),
            _fig1,
            mo.md(
                rf"""
                The first experiment has $\chi^2$ = {_chi2_first:.2f} for {_ndof} degrees of freedom (data
                points minus fitted parameters), so its p-value is **{_prob_first:.3f}**.
                """
            ),
            _many,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The rule of thumb $\chi^2/\mathrm{Ndof} \approx 1$

    A reduced $\chi^2$ of "about 1" is a good sign, but "about" depends on how many degrees of
    freedom there are. The $\chi^2$ distribution has mean Ndof and width $\sqrt{2\,\mathrm{Ndof}}$,
    so its **relative** width shrinks as Ndof grows. Pick a reduced $\chi^2$ and see how the
    p-value changes with Ndof.
    """)
    return


@app.cell
def _(mo):
    reduced_chi2 = mo.ui.slider(
        0.5, 3.0, 0.05, value=1.5, show_value=True, label="Reduced χ² (χ²/Ndof)"
    )
    ndof_demo = mo.ui.slider(
        steps=[1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000], value=10, show_value=True, label="Ndof"
    )
    return ndof_demo, reduced_chi2


@app.cell
def _(chi2_pvalue, mo, ndof_demo, np, plt, reduced_chi2, stats):
    _p = chi2_pvalue(reduced_chi2.value * ndof_demo.value, ndof_demo.value)
    _ndofs = np.unique(np.logspace(0, 3, 200).astype(int))
    _fig, _ax = plt.subplots(figsize=(9, 4))
    _ax.semilogx(_ndofs, stats.chi2.sf(reduced_chi2.value * _ndofs, _ndofs), "k")
    _ax.axhline(0.05, color="gray", ls="--", label="p = 0.05")
    _ax.plot(
        ndof_demo.value, _p, "o", color="#D55E00", label=f"Ndof = {ndof_demo.value}, p = {_p:.3f}"
    )
    _ax.set(xlabel="Ndof", ylabel="p-value", title=f"Reduced χ² = {reduced_chi2.value:.2f}")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([mo.hstack([reduced_chi2, ndof_demo], widths="equal"), _fig])
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

    Four more investigations. Each has its own controls directly above its plots, and generates its
    own data.

    ## A. Could the data come from a flat line?

    Draw one data set and ask whether the slope is needed. Three answers:

    1. fit a **constant** and look at its own p-value: is a flat line acceptable at all?
    2. test the **slope** directly: the fitted slope divided by its uncertainty, which is about
       a standard normal number when the true slope is zero,
    3. the **difference in $\chi^2$** between the constant and the line, which has 1 degree of freedom
       because the line has one extra parameter.

    For a straight line, 2 and 3 are the same test written two ways. Make the true slope bigger and
    watch the flat fit lose against the line. Press the button for a new data set.
    """)
    return


@app.cell
def _(mo):
    a_slope = mo.ui.slider(-0.5, 0.5, 0.05, value=0.3, show_value=True, label="True slope α1")
    a_points = mo.ui.slider(4, 50, 1, value=9, show_value=True, label="Points")
    a_sigma = mo.ui.slider(
        0.1, 3.0, 0.1, value=1.0, show_value=True, label="Uncertainty on each point"
    )
    a_new = mo.ui.button(value=0, on_click=lambda count: count + 1, label="Draw a new data set")
    return a_new, a_points, a_sigma, a_slope


@app.cell
def _(
    DATA,
    FIT,
    a_new,
    a_points,
    a_sigma,
    a_slope,
    chi2_of_fit,
    chi2_pvalue,
    mo,
    np,
    plt,
    simulate,
    stats,
):
    _x, _truth, _y = simulate(
        1, a_points.value, 3.6, a_slope.value, 0.0, a_sigma.value, a_new.value
    )
    _n = a_points.value
    _chi2_const = chi2_of_fit(_y, _x, a_sigma.value, 0)[0]
    _chi2_line = chi2_of_fit(_y, _x, a_sigma.value, 1)[0]
    _slope, _intercept = np.polyfit(_x, _y[0], 1)
    _slope_error = a_sigma.value / np.sqrt(np.sum((_x - _x.mean()) ** 2))
    _delta = max(_chi2_const - _chi2_line, 0.0)

    _fig, _ax = plt.subplots(figsize=(9, 4))
    _ax.errorbar(_x, _y[0], a_sigma.value, fmt="o", color=DATA, capsize=3, label="Data")
    _ax.axhline(_y[0].mean(), color="gray", ls="--", lw=2, label="Flat line (best constant)")
    _ax.plot(_x, _intercept + _slope * _x, color=FIT, lw=2, label="Best line")
    _ax.set(xlabel="x", ylabel="y")
    _ax.legend(loc="upper left")
    _fig.tight_layout()

    mo.vstack(
        [
            mo.hstack(
                [mo.vstack([a_slope, a_points]), mo.vstack([a_sigma, a_new])], widths="equal"
            ),
            _fig,
            mo.md(rf"""
            | test | statistic | p-value |
            |---|---|---|
            | flat line, on its own | χ² = {_chi2_const:.2f}, Ndof = {_n - 1} | {chi2_pvalue(_chi2_const, _n - 1):.4f} |
            | slope / its uncertainty | {_slope / _slope_error:.2f} | {2 * stats.norm.sf(abs(_slope / _slope_error)):.4f} |
            | χ²(constant) − χ²(line) | {_delta:.2f}, 1 degree of freedom | {chi2_pvalue(_delta, 1):.4f} |

            The slope test is the sharper tool when the question is "is there a slope?", because it
            spends all its power on that one question. The flat-line $\chi^2$ also reacts to any other mismatch.
            """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## B. Fix the fit

    Fit every experiment with a constant, a line and a parabola, and look at the p-value
    histograms. The red dashed line is what flat looks like. Move the quadratic term of the true
    curve: at 0 the line and the parabola are both flat. At −0.04 the line piles up at small p-values,
    and only the parabola, the function that actually generated the data, is flat again.
    """)
    return


@app.cell
def _(mo):
    b_alpha2 = mo.ui.slider(
        -0.1, 0.1, 0.005, value=-0.04, show_value=True, label="Quadratic term α2 (true curve)"
    )
    b_repeats = mo.ui.dropdown(
        {"100": 100, "1 000": 1_000, "10 000": 10_000}, value="1 000", label="Repeat the experiment"
    )
    return b_alpha2, b_repeats


@app.cell
def _(DATA, FIT, b_alpha2, b_repeats, chi2_of_fit, mo, plt, simulate, stats):
    _x, _, _y = simulate(b_repeats.value, 9, 3.6, 0.3, b_alpha2.value, 1.0, 1)
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for _ax, _degree, _label in zip(
        _axes, (0, 1, 2), ("Constant", "Line", "Parabola"), strict=True
    ):
        _p = stats.chi2.sf(chi2_of_fit(_y, _x, 1.0, _degree), len(_x) - _degree - 1)
        _ax.hist(_p, bins=20, range=(0, 1), histtype="step", color=DATA)
        _ax.axhline(len(_p) / 20, color=FIT, ls="--")
        _ax.set(title=f"{_label} fit", xlabel="p-value")
    _axes[0].set(ylabel="Experiments")
    _fig.tight_layout()
    mo.vstack([mo.hstack([b_alpha2, b_repeats], widths="equal"), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## C. Wrong uncertainties

    Fit a line to data that really come from a straight line, but pretend the uncertainty on each point
    is 20 % too small, or 20 % too large. The fitted line does not move (all points are weighted
    equally), but the $\chi^2$ scales as $1/\text{factor}^2$. Too small an uncertainty gives too many
    tiny p-values, too large an uncertainty gives too many values near 1. A p-value histogram that is
    not flat can therefore mean a wrong function **or** wrong uncertainties. Choose how much the
    uncertainty is off.
    """)
    return


@app.cell
def _(mo):
    c_factor = mo.ui.slider(
        0.5, 1.5, 0.05, value=0.8, show_value=True, label="Assumed uncertainty × factor"
    )
    c_repeats = mo.ui.dropdown(
        {"100": 100, "1 000": 1_000, "10 000": 10_000}, value="1 000", label="Repeat the experiment"
    )
    return c_factor, c_repeats


@app.cell
def _(DATA, FIT, c_factor, c_repeats, chi2_of_fit, mo, plt, simulate, stats):
    _x, _, _y = simulate(c_repeats.value, 9, 3.6, 0.3, 0.0, 1.0, 3)
    _chi2_true = chi2_of_fit(_y, _x, 1.0, 1)
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.hist(
        stats.chi2.sf(_chi2_true, len(_x) - 2),
        bins=20,
        range=(0, 1),
        histtype="step",
        color="gray",
        label="Correct uncertainty (× 1)",
    )
    _ax.hist(
        stats.chi2.sf(_chi2_true / c_factor.value**2, len(_x) - 2),
        bins=20,
        range=(0, 1),
        histtype="step",
        color=FIT,
        lw=2,
        label=f"Assumed uncertainty × {c_factor.value:.2f}",
    )
    _ax.axhline(c_repeats.value / 20, color=DATA, ls=":", label="Flat")
    _ax.set(xlabel="p-value", ylabel="Experiments")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([mo.hstack([c_factor, c_repeats], widths="equal"), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## D. Non-Gaussian noise

    Replace the Gaussian noise with another shape of the **same standard deviation** and fit 5 000
    data sets with a line. The data come from a straight line, so the line is the correct function. The
    $\chi^2$ distribution and the p-value histogram show how sensitive the test is to the shape of the
    noise, and the answer is: not very, because each $\chi^2$ is a sum of many small contributions (the
    central limit theorem). Fewer points make the sum shorter, so try that too.
    """)
    return


@app.cell
def _(mo):
    d_noise = mo.ui.dropdown(
        ["Gaussian", "Uniform", "Exponential (skewed)"], value="Uniform", label="Shape of the noise"
    )
    d_points = mo.ui.slider(4, 50, 1, value=9, show_value=True, label="Points per data set")
    return d_noise, d_points


@app.cell
def _(DATA, FIT, chi2_of_fit, d_noise, d_points, mo, np, plt, simulate, stats):
    _x, _, _y = simulate(5_000, d_points.value, 3.6, 0.3, 0.0, 1.0, 2, noise=d_noise.value)
    _ndof = d_points.value - 2
    _chi2 = chi2_of_fit(_y, _x, 1.0, 1)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4))
    _ax1.hist(
        _chi2, bins=60, density=True, histtype="step", color=DATA, label=f"{d_noise.value} noise"
    )
    _grid = np.linspace(0.01, _chi2.max(), 400)
    _ax1.plot(
        _grid, stats.chi2.pdf(_grid, _ndof), color=FIT, label=f"χ² distribution, Ndof = {_ndof}"
    )
    _ax1.set(xlabel="χ²")
    _ax1.legend()
    _ax2.hist(stats.chi2.sf(_chi2, _ndof), bins=20, range=(0, 1), histtype="step", color=DATA)
    _ax2.axhline(len(_chi2) / 20, color=FIT, ls="--")
    _ax2.set(xlabel="p-value")
    _fig.tight_layout()
    mo.vstack([mo.hstack([d_noise, d_points], widths="equal"), _fig])
    return


if __name__ == "__main__":
    app.run()
