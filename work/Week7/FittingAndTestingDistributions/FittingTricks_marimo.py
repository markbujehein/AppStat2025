import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from appstat.fitting import fit_least_squares
    from appstat.models import exponential_mixture, peak_series, two_exponentials
    from appstat.stats import chi2_pvalue, correlation_matrix

    DATA, FIT, FIT2 = "#000000", "#D55E00", "#0072B2"

    def correlation_table(fit):
        """The correlations between the fitted parameters as Markdown, and the largest one."""
        if fit.covariance is None:
            return "The fit gave no covariance matrix, so the correlations are undefined.", np.nan
        names = list(fit.values.to_dict())
        corr = correlation_matrix(np.asarray(fit.covariance))
        if not np.all(np.isfinite(corr)):
            return "The correlations are undefined (an uncertainty is zero).", np.nan
        rows = ["| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        rows += [
            f"| **{name}** | " + " | ".join(f"{c:+.3f}" for c in row) + " |"
            for name, row in zip(names, corr, strict=True)
        ]
        return "\n".join(rows), np.abs(corr - np.eye(len(names))).max()

    return (
        DATA,
        FIT,
        FIT2,
        chi2_pvalue,
        correlation_table,
        exponential_mixture,
        fit_least_squares,
        mo,
        np,
        peak_series,
        plt,
        two_exponentials,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fitting is an art: three tricks

    The same data can be fitted with different **forms of the same function**, and the form decides whether the fit
    works. Three cases:

    1. Data **far from zero**: shift $x$ so the parameters stop being correlated.
    2. **Peaks on a background**: the peaks share parameters, so the fit uses all of them.
    3. A **double exponential**: write it in terms of a fraction and two lifetimes.

    In every case the data is made here with known true values, so you can see which fit finds them.

    Every interactive block below has its controls directly above its plots.

    ---

    **Adapted from** the Fitting Tricks exercises by **Troels C. Petersen** (Niels Bohr Institute), from the course
    *Applied Statistics 2025/26*. The originals are three exercises with no example solutions. This version
    is written from their text and is not an official solution. It makes the same data with the same true values.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Data far from zero

    Thirteen points on $y = 110\,e^{b(x-2020)} - 100$ for $x$ between 2020 and 2021, with noise 0.5. The "bad"
    function $a\,e^{bx}+c$ at $x\approx 2020$ means a small change of $b$ changes $a$ enormously, and starting from
    zero Minuit gets stuck. The "good" one, $a\,e^{b(x-2020)}+c$, does not. Minuit may still say **converged**
    for the bad fit: the $\chi^2$ and the true values tell you it failed.

    The exercise has $b=0.005$. Then the curve is a straight line within the noise, and no function can find $a$, $b$ and
    $c$ separately. Try it, and then raise $b$.
    """)
    return


@app.cell
def _(mo):
    far_model = mo.ui.dropdown(
        ["Bad: a·exp(b·x)+c", "Good: a·exp(b·(x−2020))+c"],
        value="Bad: a·exp(b·x)+c",
        label="Fit function",
    )
    far_start = mo.ui.slider(0.0, 150.0, 5.0, value=0.0, show_value=True, label="Start value of a")
    far_seed = mo.ui.number(0, 9999, 1, value=40, label="Random seed")
    far_b = mo.ui.slider(0.005, 1.0, 0.005, value=0.5, show_value=True, label="True growth rate b")
    return far_b, far_model, far_seed, far_start


@app.cell
def _(
    DATA,
    FIT,
    chi2_pvalue,
    correlation_table,
    far_b,
    far_model,
    far_seed,
    far_start,
    fit_least_squares,
    mo,
    np,
    plt,
):
    _x = np.linspace(2020, 2021, 13)
    _y = (
        110 * np.exp(far_b.value * (_x - 2020))
        - 100
        + np.random.default_rng(far_seed.value).normal(0, 0.5, 13)
    )

    def _bad(x, a, b, c):
        return a * np.exp(b * x) + c

    def _good(x, a, b, c):
        return a * np.exp(b * (x - 2020)) + c

    _good_choice = far_model.value.startswith("Good")
    _model = _good if _good_choice else _bad
    with np.errstate(all="ignore"):
        _fit = fit_least_squares(
            _model, _x, _y, np.full(13, 0.5), {"a": far_start.value, "b": 0.0, "c": 0.0}
        )
    _fig, _ax = plt.subplots(figsize=(13, 3.6))
    _ax.errorbar(_x, _y, 0.5, fmt="o", color=DATA, label="Data")
    _grid = np.linspace(2020, 2021, 200)
    with np.errstate(all="ignore"):
        _ax.plot(_grid, _model(_grid, *_fit.values), color=FIT, lw=2, label="Fit")
    _ax.set(xlabel="x", ylabel="y")
    _ax.legend()
    _fig.tight_layout()
    _table, _worst = correlation_table(_fit)
    mo.vstack(
        [
            mo.hstack([far_model, far_start, far_b, far_seed], justify="start", gap=2),
            _fig,
            mo.callout(
                mo.md(
                    f"{'Minuit says converged.' if _fit.valid else '**The fit did not converge.**'} "
                    f"χ²/Ndof = {_fit.fval:.1f}/{_fit.ndof}, p = {chi2_pvalue(_fit.fval, _fit.ndof):.2g}. "
                    f"True values a = 110, b = {far_b.value:g}, c = −100. "
                    + ", ".join(
                        f"{k} = {v:.4g} ± {_fit.errors[k]:.2g}"
                        for k, v in _fit.values.to_dict().items()
                    )
                    + f". Largest correlation **{_worst:.4f}**.\n\n"
                    + _table
                ),
                kind="success" if chi2_pvalue(_fit.fval, _fit.ndof) > 0.01 else "danger",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Peaks that share parameters

    2000 signal particles fall in Gaussian peaks at 1.43, 2.84, 4.25, ..., each half the size of the one before, on a
    flat background of 5000. A fit to the **first peak alone** uses a fraction of the data and says nothing about the
    others. A fit to **all peaks**, with the spacing $\Delta\mu$, width $\sigma$ and size ratio $f$ shared, measures all of
    these, and the position of the first peak a little better.
    """)
    return


@app.cell
def _(mo):
    peaks_model = mo.ui.dropdown(
        ["First peak alone", "All peaks, shared parameters"],
        value="All peaks, shared parameters",
        label="Fit",
    )
    peaks_seed = mo.ui.number(0, 9999, 1, value=42, label="Random seed")
    return peaks_model, peaks_seed


@app.cell
def _(
    DATA,
    FIT,
    correlation_table,
    fit_least_squares,
    mo,
    np,
    peak_series,
    peaks_model,
    peaks_seed,
    plt,
):
    _rng = np.random.default_rng(peaks_seed.value)
    _k = _rng.geometric(0.5, 2000)
    _data = np.concatenate([_rng.normal(1.43 + 1.41 * (_k - 1), 0.15), _rng.uniform(0, 10, 5000)])
    _counts, _edges = np.histogram(_data, 200, range=(0, 10))
    _x, _w = (_edges[1:] + _edges[:-1]) / 2, 0.05

    def _first(x, n, mu, sigma, cst):
        return _w * peak_series(x, n, 1.0, mu, 0.0, sigma, cst, 1)

    def _all(x, n, f, mu, dmu, sigma, cst):
        return _w * peak_series(x, n, f, mu, dmu, sigma, cst, 8)

    if peaks_model.value.startswith("First"):
        _use = (_x > 0.5) & (_x < 2.3) & (_counts > 0)
        _model, _start = _first, {"n": 1000.0, "mu": 1.4, "sigma": 0.15, "cst": 400.0}
    else:
        _use = _counts > 0
        _model, _start = (
            _all,
            {"n": 2000.0, "f": 0.5, "mu": 1.4, "dmu": 1.4, "sigma": 0.15, "cst": 400.0},
        )
    _fit = fit_least_squares(_model, _x[_use], _counts[_use], np.sqrt(_counts[_use]), _start)
    _fig, _ax = plt.subplots(figsize=(13, 3.8))
    _ax.errorbar(_x, _counts, np.sqrt(_counts), fmt=".", color=DATA, label="Data")
    _grid = np.linspace(0, 10, 1000)
    _ax.plot(_grid, _model(_grid, *_fit.values), color=FIT, lw=2, label="Fit")
    _ax.set(xlabel="x", ylabel="Entries per 0.05")
    _ax.legend()
    _fig.tight_layout()
    _table, _worst = correlation_table(_fit)
    mo.vstack(
        [
            mo.hstack([peaks_model, peaks_seed], justify="start", gap=2),
            _fig,
            mo.callout(
                mo.md(
                    f"χ²/Ndof = {_fit.fval:.0f}/{_fit.ndof}. "
                    f"Position of the first peak **μ = {_fit.values['mu']:.4f} ± {_fit.errors['mu']:.4f}** (true 1.43)"
                    + (
                        f", spacing **Δμ = {_fit.values['dmu']:.4f} ± {_fit.errors['dmu']:.4f}** (true 1.41)"
                        if "dmu" in _fit.values.to_dict()
                        else ""
                    )
                    + ("" if _fit.valid else ". **The fit did not converge.**")
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. A double exponential

    Half of the decays have lifetime 10 and half the second lifetime (2 by default). The "bad" function
    $N_1 e^{-t/\tau_1} + N_2 e^{-t/\tau_2}$ has heights that are strongly correlated with the lifetimes. The "good" one,
    $N\big(\tfrac{f}{\tau_1}e^{-t/\tau_1} + \tfrac{1-f}{\tau_2}e^{-t/\tau_2}\big)$, is a normalised mix, so the total, the
    fraction and the lifetimes are what is fitted. The **fit is equally good** either way, and the correlations only a
    little weaker. What changes is that the parameters mean something. Bring the second lifetime close to 10, or lower the
    number of decays, and **both** fail.
    """)
    return


@app.cell
def _(mo):
    exp_model = mo.ui.dropdown(
        ["Bad: heights and lifetimes", "Good: fraction and lifetimes"],
        value="Bad: heights and lifetimes",
        label="Fit function",
    )
    exp_tau2 = mo.ui.slider(1.0, 8.0, 0.5, value=2.0, show_value=True, label="Second lifetime")
    exp_seed = mo.ui.number(0, 9999, 1, value=42, label="Random seed")
    exp_n = mo.ui.slider(
        500, 20_000, 500, value=10_000, show_value=True, label="Decays of each kind"
    )
    return exp_model, exp_n, exp_seed, exp_tau2


@app.cell
def _(
    DATA,
    FIT,
    correlation_table,
    exp_model,
    exp_seed,
    exp_n,
    exp_tau2,
    exponential_mixture,
    fit_least_squares,
    mo,
    np,
    plt,
    two_exponentials,
):
    _rng = np.random.default_rng(exp_seed.value)
    _data = np.concatenate(
        [_rng.exponential(10.0, exp_n.value), _rng.exponential(exp_tau2.value, exp_n.value)]
    )
    _counts, _edges = np.histogram(_data, 200, range=(0, 20))
    _x, _w = (_edges[1:] + _edges[:-1]) / 2, 0.1
    _use = _counts > 0

    def _bad(x, n1, tau1, n2, tau2):
        return _w * two_exponentials(x, n1, tau1, n2, tau2)

    def _good(x, n, f, tau1, tau2):
        return _w * exponential_mixture(x, n, f, tau1, tau2)

    if exp_model.value.startswith("Bad"):
        _model, _start = (
            _bad,
            {"n1": exp_n.value / 10, "tau1": 8.0, "n2": exp_n.value / 10, "tau2": 3.0},
        )
    else:
        _model, _start = _good, {"n": 2.0 * exp_n.value, "f": 0.5, "tau1": 8.0, "tau2": 3.0}
    _fit = fit_least_squares(_model, _x[_use], _counts[_use], np.sqrt(_counts[_use]), _start)
    _fig, _ax = plt.subplots(figsize=(13, 3.6))
    _ax.errorbar(_x, _counts, np.sqrt(_counts), fmt=".", color=DATA, label="Data")
    _ax.plot(_x, _model(_x, *_fit.values), color=FIT, lw=2, label="Fit")
    _ax.set(xlabel="Time", ylabel="Entries per 0.1", yscale="log")
    _ax.legend()
    _fig.tight_layout()
    _table, _worst = correlation_table(_fit)
    mo.vstack(
        [
            mo.hstack([exp_model, exp_tau2, exp_n, exp_seed], justify="start", gap=2),
            _fig,
            mo.callout(
                mo.md(
                    f"χ²/Ndof = {_fit.fval:.0f}/{_fit.ndof}"
                    + ("" if _fit.valid else ". **The fit did not converge.**")
                    + ". "
                    + ", ".join(
                        f"{k} = {v:.4g} ± {_fit.errors[k]:.2g}"
                        for k, v in _fit.values.to_dict().items()
                    )
                    + f". Largest correlation **{_worst:.3f}**.\n\n"
                    + _table
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to take away

    1. **Centre** your variable. A fit far from zero has two parameters that move together, and Minuit can fail.
    2. Parameters that mean something **physical** (a position, a spacing, a fraction) and are shared between peaks make
       use of **all** the data.
    3. A **normalised** form with a total and a fraction has weaker correlations than separate heights. Both fit equally well.
    4. Look at the **correlation matrix** of a fit. A coefficient near ±1 says the function is badly written.
    """)
    return


if __name__ == "__main__":
    app.run()
