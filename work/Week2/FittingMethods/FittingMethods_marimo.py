import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    from appstat.fitting import (
        fit_binned_likelihood,
        fit_least_squares,
        fit_unbinned_likelihood,
    )
    from appstat.stats import chi2_pvalue, empirical_pvalue

    CHI2, BINNED, UNBINNED = "#D55E00", "#0072B2", "#009E73"
    X_MIN, X_MAX = 0.0, 10.0
    TRUTH = {"Nexp": 5000.0, "tau": 3.0, "Ngauss": 2000.0, "mu": 3.0, "sigma": 0.3}

    def density(x, Nexp, tau, Ngauss, mu, sigma):
        """Expected number per unit x: a falling exponential plus a Gaussian peak."""
        return Nexp * stats.expon.pdf(x, scale=tau) + Ngauss * stats.norm.pdf(x, mu, sigma)

    def with_total(x, Nexp, tau, Ngauss, mu, sigma):
        """The same model in the form the unbinned likelihood wants: (total, density)."""
        return Nexp + Ngauss, density(x, Nexp, tau, Ngauss, mu, sigma)

    def simulate(rng, n_signal, n_background):
        peak = rng.normal(TRUTH["mu"], TRUTH["sigma"], int(n_signal))
        background = rng.exponential(TRUTH["tau"], int(n_background))
        return np.concatenate([peak, background])

    def fit_three_ways(x_all, n_bins, n_signal, n_background):
        """Fit one data set by chi-square, binned and unbinned likelihood. A failed fit is None."""
        counts, edges = np.histogram(x_all, bins=n_bins, range=(X_MIN, X_MAX))
        centres = (edges[1:] + edges[:-1]) / 2
        width = (X_MAX - X_MIN) / n_bins
        start = {"Nexp": n_background, "tau": 2.9, "Ngauss": n_signal, "mu": 3.1, "sigma": 0.3}

        def per_bin(x, Nexp, tau, Ngauss, mu, sigma):
            return width * density(x, Nexp, tau, Ngauss, mu, sigma)

        used = counts > 0
        runs = {
            "Chi-square": lambda: fit_least_squares(
                per_bin, centres[used], counts[used], np.sqrt(counts[used]), start
            ),
            "Binned likelihood": lambda: fit_binned_likelihood(density, counts, edges, start),
            "Unbinned likelihood": lambda: fit_unbinned_likelihood(with_total, x_all, start),
        }
        fits = {}
        for name, run in runs.items():
            try:
                fits[name] = run()
            except ValueError:  # too few non-empty bins for a chi-square fit
                fits[name] = None
        return fits, counts, centres, width

    return (
        BINNED,
        CHI2,
        TRUTH,
        UNBINNED,
        X_MAX,
        X_MIN,
        chi2_pvalue,
        density,
        empirical_pvalue,
        fit_three_ways,
        fit_unbinned_likelihood,
        mo,
        np,
        plt,
        simulate,
        stats,
        with_total,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fitting with $\chi^2$, binned likelihood and unbinned likelihood

    A Gaussian peak on a falling exponential background, fitted in three ways:

    1. **$\chi^2$** of a histogram. The usual start, but it needs non-empty bins.
    2. **Binned likelihood** of a histogram. Poisson probabilities, so empty bins are fine.
    3. **Unbinned likelihood** of the individual values. No binning at all, and no information lost.

    With a lot of data they agree. With little data they do not. Find out when.

    Every interactive block below has its controls directly above its plots.

    References: Barlow 5.3, 5.4 and 5.6, Bevington chapter 10.

    ---

    **Adapted from** the Fitting Methods exercise by **Troels C. Petersen** (Niels Bohr Institute),
    from the course *Applied Statistics 2025/26*. The original is an exercise with an example
    solution. This version keeps its data, its three fits and its questions, and is rewritten as an
    interactive demonstration.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. One data set, three fits

    The truth is 2000 signal values (mean 3, width 0.3) and 5000 background values (lifetime 3).
    Lower the signal and background to **20 and 50**, and change the number of bins. Do the three
    fits still agree, and do they still look good?
    """)
    return


@app.cell
def _(mo):
    n_signal = mo.ui.slider(5, 5000, 5, value=2000, show_value=True, label="Signal values")
    n_background = mo.ui.slider(
        10, 10_000, 10, value=5000, show_value=True, label="Background values"
    )
    n_bins = mo.ui.slider(10, 200, 5, value=100, show_value=True, label="Bins")
    seed = mo.ui.number(0, 9999, 1, value=42, label="Random seed")
    return n_background, n_bins, n_signal, seed


@app.cell
def _(
    BINNED,
    CHI2,
    UNBINNED,
    X_MAX,
    X_MIN,
    chi2_pvalue,
    density,
    fit_three_ways,
    mo,
    n_background,
    n_bins,
    n_signal,
    np,
    plt,
    seed,
    simulate,
):
    _x_all = simulate(np.random.default_rng(seed.value), n_signal.value, n_background.value)
    _fits, _counts, _centres, _width = fit_three_ways(
        _x_all, n_bins.value, n_signal.value, n_background.value
    )
    _colours = {"Chi-square": CHI2, "Binned likelihood": BINNED, "Unbinned likelihood": UNBINNED}

    _fig, _ax = plt.subplots(figsize=(13, 4.2))
    _ax.errorbar(
        _centres, _counts, np.sqrt(_counts), fmt=".k", capsize=1, label="Data, Poisson errors"
    )
    _grid = np.linspace(X_MIN, X_MAX, 800)
    for _name, _fit in _fits.items():
        if _fit is not None:
            _ax.plot(
                _grid,
                _width * density(_grid, *_fit.values),
                color=_colours[_name],
                lw=2,
                ls="-" if _fit.valid else ":",
                label=_name + ("" if _fit.valid else " (did not converge)"),
            )
    _ax.set(xlabel="x", ylabel=f"Entries per {_width:g}", ylim=(0, None))
    _ax.legend()
    _fig.tight_layout()

    _rows = [
        "| fit | " + " | ".join(["Nexp", "tau", "Ngauss", "mu", "sigma"]) + " |",
        "|---|---|---|---|---|---|",
    ]
    for _name, _fit in _fits.items():
        if _fit is None:
            _rows.append(f"| {_name} | not enough non-empty bins | | | | |")
        else:
            _cells = " | ".join(
                f"{v:.3g} ± {e:.2g}" for v, e in zip(_fit.values, _fit.errors, strict=True)
            )
            _rows.append(f"| {_name} | {_cells} |")
    _chi2_fit = _fits["Chi-square"]
    _goodness = (
        f"Chi-square fit: χ² = {_chi2_fit.fval:.1f}, Ndof = {_chi2_fit.ndof}, "
        f"p = {chi2_pvalue(_chi2_fit.fval, _chi2_fit.ndof):.3f}."
        if _chi2_fit is not None and _chi2_fit.ndof > 0
        else "No chi-square fit is possible: too few non-empty bins."
    )
    _good = (
        _chi2_fit is not None
        and _chi2_fit.ndof > 0
        and chi2_pvalue(_chi2_fit.fval, _chi2_fit.ndof) > 0.01
    )
    numbers_md = (
        "\n".join(_rows)
        + "\n\nTruth: Nexp 5000, tau 3, Ngauss 2000, mu 3, sigma 0.3.\n\n"
        + _goodness
    )
    mo.vstack(
        [
            mo.hstack(
                [mo.vstack([n_signal, n_background]), mo.vstack([n_bins, seed])], widths="equal"
            ),
            _fig,
            mo.callout(mo.md(_goodness), kind="success" if _good else "warn"),
        ]
    )
    return (numbers_md,)


@app.cell(hide_code=True)
def _(mo, numbers_md):
    mo.md("### The fitted values\n\nThese follow the controls above.\n\n" + numbers_md)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A. Which method is biased at low statistics?

    Repeat the whole experiment 40 times and look at the fitted **number of signal values**. The
    dashed line is the truth. Lower the signal: the $\chi^2$ fit uses $\sqrt{N}$ as the uncertainty, which is
    poor for a nearly empty bin, and it falls to the wrong side. Each repetition is three real fits.
    """)
    return


@app.cell
def _(mo):
    low_signal = mo.ui.slider(
        10, 500, 10, value=50, show_value=True, label="Signal values (background is 2.5 times that)"
    )
    return (low_signal,)


@app.cell
def _(BINNED, CHI2, UNBINNED, fit_three_ways, low_signal, mo, np, plt, simulate):
    _rng = np.random.default_rng(11)
    _signal, _background = low_signal.value, 2.5 * low_signal.value
    _fitted = {"Chi-square": [], "Binned likelihood": [], "Unbinned likelihood": []}
    for _ in range(40):  # one data set, three fits
        _fits, *_ = fit_three_ways(simulate(_rng, _signal, _background), 50, _signal, _background)
        for _name, _fit in _fits.items():
            if _fit is not None and _fit.valid:
                _fitted[_name].append(_fit.values["Ngauss"])

    _fig, _ax = plt.subplots(figsize=(13, 3.8))
    for (_name, _values), _colour in zip(_fitted.items(), [CHI2, BINNED, UNBINNED], strict=True):
        if _values:
            _ax.hist(
                _values,
                bins=np.linspace(0, 3 * _signal, 40),
                histtype="step",
                lw=2,
                color=_colour,
                label=f"{_name}: mean {np.mean(_values):.1f} ({len(_values)} converged)",
            )
    _ax.axvline(_signal, color="k", ls="--", label=f"Truth {_signal}")
    _ax.set(xlabel="Fitted number of signal values", ylabel="Fits")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([low_signal, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## B. Goodness of fit for the unbinned likelihood

    The unbinned likelihood has no $\chi^2$, so how do you know the fit is good? **Simulate it.** Fit the data,
    draw new data sets from the fitted model, fit each of them, and see where the real
    $-\ln L$ lies among the simulated ones. The p-value is the share of simulations at least as bad as the
    real data. Each simulation is a real fit, so it takes a few seconds.
    """)
    return


@app.cell
def _(mo):
    gof_runs = mo.ui.dropdown(
        {"20": 20, "50": 50, "100": 100}, value="50", label="Simulated data sets"
    )
    gof_signal = mo.ui.slider(
        20,
        2000,
        20,
        value=400,
        show_value=True,
        label="Signal values (background is 2.5 times that)",
    )
    gof_seed = mo.ui.number(0, 9999, 1, value=1, label="Random seed")
    return gof_runs, gof_seed, gof_signal


@app.cell
def _(
    UNBINNED,
    empirical_pvalue,
    fit_unbinned_likelihood,
    gof_runs,
    gof_seed,
    gof_signal,
    mo,
    np,
    plt,
    simulate,
    with_total,
):
    _rng = np.random.default_rng(gof_seed.value)
    _signal, _background = gof_signal.value, 2.5 * gof_signal.value
    _start = {"Nexp": _background, "tau": 2.9, "Ngauss": _signal, "mu": 3.1, "sigma": 0.3}
    _real = fit_unbinned_likelihood(with_total, simulate(_rng, _signal, _background), _start)
    _truth = dict(
        zip(_real.parameters, _real.values, strict=True)
    )  # simulate from the fitted model
    _simulated = []
    for _ in range(gof_runs.value):  # one fit per simulated data set
        _data = np.concatenate(
            [
                _rng.normal(_truth["mu"], _truth["sigma"], int(_truth["Ngauss"])),
                _rng.exponential(_truth["tau"], int(_truth["Nexp"])),
            ]
        )
        _simulated.append(fit_unbinned_likelihood(with_total, _data, _start).fval)
    _p = empirical_pvalue(np.array(_simulated), _real.fval)

    _fig, _ax = plt.subplots(figsize=(13, 3.6))
    _ax.hist(_simulated, bins=20, color=UNBINNED, alpha=0.5, label="Simulated data sets")
    _ax.axvline(_real.fval, color="k", ls="--", lw=2, label=f"Real data: p = {_p:.2f}")
    _ax.set(xlabel="−ln L at the minimum", ylabel="Data sets")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack(
        [
            mo.hstack([gof_signal, gof_runs, gof_seed], justify="start", gap=2),
            _fig,
            mo.callout(
                mo.md(
                    f"p = **{_p:.2f}**: {'a fit this good is common' if _p > 0.05 else 'unusually bad, the model may be wrong'}."
                ),
                kind="success" if _p > 0.05 else "warn",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to take away

    1. **$\chi^2$** is the place to start, but poor for nearly empty bins, where $\sqrt{N}$ is a bad uncertainty.
    2. **Binned likelihood** uses Poisson probabilities, so it copes with empty bins. Use it if the data is already a histogram.
    3. **Unbinned likelihood** is the best when you have the individual values. It loses nothing to binning.
    4. The likelihood has no $\chi^2$ to read off, so its goodness of fit comes from **simulating** the data.
    """)
    return


if __name__ == "__main__":
    app.run()
