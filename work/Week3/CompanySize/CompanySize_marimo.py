import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import io
    import sys
    from urllib.request import urlopen

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from appstat.fitting import fit_least_squares
    from appstat.stats import chi2_pvalue, power_exponential, power_law

    DATA, FIT = "#000000", "#D55E00"
    NAME = "TableData_Denmark2020.txt"

    location = mo.notebook_location()
    if sys.platform == "emscripten":  # in the browser the data sits in the page's public folder
        text = urlopen(str(location / "public" / NAME)).read().decode()
    else:
        text = (location / NAME).read_text()
    table = np.genfromtxt(io.StringIO(text), skip_header=1, usecols=(0, 1))  # "NA" becomes nan
    wanted = (table[:, 0] >= 1) & (
        table[:, 0] <= 199
    )  # sizes with at least 5 companies, in 1 to 199
    sizes, companies = table[wanted, 0], table[wanted, 1]

    def exponential(x, n, alpha):
        return n * np.exp(-alpha * x)

    MODELS = {
        "Power law": (power_law, {"n": 50_000.0, "alpha": 2.0}),
        "Exponential": (exponential, {"n": 50_000.0, "alpha": 0.06}),
        "Power law + exponential": (
            power_exponential,
            {"n": 50_000.0, "f": 0.9, "alpha_pow": 1.5, "alpha_exp": 0.1},
        ),
    }

    def fit_company(name, low, high):
        """Chi-square fit of a model to the company counts with sizes from ``low`` to ``high``."""
        inside = (sizes >= low) & (sizes <= high)
        model, start = MODELS[name]
        fit = fit_least_squares(
            model, sizes[inside], companies[inside], np.sqrt(companies[inside]), start
        )
        return fit, model, inside

    return (
        DATA,
        FIT,
        MODELS,
        companies,
        chi2_pvalue,
        fit_company,
        mo,
        np,
        plt,
        sizes,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # How big are Danish companies?

    How many companies have 1, 2, 3, ... employees (full time equivalents)? Statistics Denmark counted them in
    2020, for modelling the spread of Covid-19. The distribution falls over four orders of magnitude, and
    **no simple formula describes it**. You try a power law, an exponential, and a mix of the two,
    and judge each by its $\chi^2$.

    Counts below 5 are secret by law, so the table ends there (here at 199 employees).

    Every interactive block below has its controls directly above its plots.

    ---

    **Adapted from** the Company Size exercise by **Troels C. Petersen** (Niels Bohr Institute,
    4th of December 2023), from the course *Applied Statistics 2025/26*, with data from Statistics Denmark. The
    original is an exercise with an example solution. This version keeps its data, its three models and its three
    questions, and is rewritten as an interactive demonstration.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Look at the data

    Switch the axes between linear and logarithmic. On a log-log plot a **power law** is a straight line. On
    a log-linear plot an **exponential** is a straight line. Which one looks straighter?
    """)
    return


@app.cell
def _(mo):
    look_logx = mo.ui.switch(value=False, label="Logarithmic x axis")
    look_logy = mo.ui.switch(value=True, label="Logarithmic y axis")
    return look_logx, look_logy


@app.cell
def _(DATA, companies, look_logx, look_logy, mo, plt, sizes):
    _fig, _ax = plt.subplots(figsize=(13, 3.8))
    _ax.errorbar(sizes, companies, companies**0.5, fmt=".", color=DATA)
    _ax.set(
        xlabel="Size of company (FTE)",
        ylabel="Number of companies",
        xscale="log" if look_logx.value else "linear",
        yscale="log" if look_logy.value else "linear",
    )
    _fig.tight_layout()
    mo.vstack([mo.hstack([look_logx, look_logy], justify="start", gap=2), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Fit a model

    The error on each count is $\sqrt{N}$. The lower panel shows the **pulls**, $(\mathrm{data} - \mathrm{fit})/\sigma$:
    a good fit scatters around zero within about $\pm 2$. Change the model and the fitted range. The first
    few points are the most precise, so they dominate the $\chi^2$.
    """)
    return


@app.cell
def _(MODELS, mo):
    fit_model = mo.ui.dropdown(list(MODELS), value="Power law + exponential", label="Model")
    fit_range = mo.ui.range_slider(1, 199, 1, value=[1, 100], show_value=True, label="Fitted sizes")
    return fit_model, fit_range


@app.cell
def _(DATA, FIT, chi2_pvalue, companies, fit_company, fit_model, fit_range, mo, np, plt, sizes):
    _low, _high = fit_range.value
    _fig, (_ax, _axp) = plt.subplots(
        2, 1, figsize=(13, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}
    )
    _ax.errorbar(sizes, companies, companies**0.5, fmt=".", color=DATA, label="Data (2020)")
    _ax.set(ylabel="Number of companies", yscale="log")
    try:
        _fit, _model, _inside = fit_company(fit_model.value, _low, _high)
        _ax.plot(
            sizes[_inside],
            _model(sizes[_inside], *_fit.values),
            color=FIT,
            lw=2,
            label=fit_model.value,
        )
        _pull = (companies[_inside] - _model(sizes[_inside], *_fit.values)) / np.sqrt(
            companies[_inside]
        )
        _axp.plot(sizes[_inside], _pull, ".", color=FIT)
        _p = chi2_pvalue(_fit.fval, _fit.ndof)
        _note = mo.callout(
            mo.md(
                f"χ²/Ndof = {_fit.fval:.0f}/{_fit.ndof}, p = {_p:.2g}. "
                + ("" if _fit.valid else "**The fit did not converge.** Try another range. ")
                + ", ".join(f"{k} = {v:.4g}" for k, v in _fit.values.to_dict().items())
            ),
            kind="success" if _fit.valid and _p > 0.05 else "warn",
        )
    except ValueError:
        _note = mo.callout(mo.md("Too few sizes in the range for this model."), kind="warn")
    _axp.axhline(0, color="k", lw=1)
    _axp.set(xlabel="Size of company (FTE)", ylabel="Pull")
    _ax.legend()
    mo.vstack([mo.hstack([fit_model, fit_range], justify="start", gap=2), _fig, _note])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Predict the companies with 101 to 150 employees

    A good model should predict what it was **not** fitted to. Fit the sizes you choose, and add up the model for
    101 to 150 employees. The prediction error is from the fitted parameters **and** their correlations, by
    simulation. Compare with the real count. Try leaving out the first sizes, which are irrelevant to this question.
    """)
    return


@app.cell
def _(MODELS, mo):
    predict_model = mo.ui.dropdown(list(MODELS), value="Power law + exponential", label="Model")
    predict_range = mo.ui.range_slider(
        1, 100, 1, value=[10, 100], show_value=True, label="Fitted sizes"
    )
    return predict_model, predict_range


@app.cell
def _(DATA, FIT, companies, fit_company, mo, np, plt, predict_model, predict_range, sizes):
    _low, _high = predict_range.value
    _beyond = (sizes >= 101) & (sizes <= 150)
    _fig, _ax = plt.subplots(figsize=(13, 3.6))
    _ax.errorbar(
        sizes[sizes <= 150],
        companies[sizes <= 150],
        companies[sizes <= 150] ** 0.5,
        fmt=".",
        color=DATA,
        label="Data",
    )
    _ax.axvspan(100.5, 150.5, color="gray", alpha=0.15, label="Not fitted: predicted")
    _ax.set(xlabel="Size of company (FTE)", ylabel="Number of companies", yscale="log")
    try:
        _fit, _model, _ = fit_company(predict_model.value, _low, _high)
        _grid = np.linspace(_low, 150, 300)
        _ax.plot(_grid, _model(_grid, *_fit.values), color=FIT, lw=2, label=predict_model.value)
        _rng = np.random.default_rng(1)
        _draws = _rng.multivariate_normal(
            np.asarray(_fit.values), np.asarray(_fit.covariance), 5000
        )
        _sums = np.array([_model(sizes[_beyond], *row).sum() for row in _draws])
        _predicted, _error = _model(sizes[_beyond], *_fit.values).sum(), _sums.std()
        _actual = companies[_beyond].sum()
        _sigma = (_predicted - _actual) / np.hypot(_error, np.sqrt(_actual))
        _note = mo.callout(
            mo.md(
                f"Predicted **{_predicted:.0f} ± {_error:.0f}**, counted **{_actual:.0f}**: "
                f"{(_predicted - _actual) / _actual:+.0%}, or {_sigma:+.1f} σ with the Poisson error of the count. "
                + (
                    ""
                    if _fit.valid
                    else "**The fit did not converge**, so this is not trustworthy. "
                )
            ),
            kind="success" if abs(_sigma) < 3 and _fit.valid else "warn",
        )
    except ValueError:
        _note = mo.callout(mo.md("Too few sizes in the range for this model."), kind="warn")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([mo.hstack([predict_model, predict_range], justify="start", gap=2), _fig, _note])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to take away

    1. A **power law** has a long tail, an **exponential** has none. The data has both: a mix beats either alone,
       and is still not a good fit, with a $\chi^2$ far above $N_\mathrm{dof}$ and pulls that are not scattered randomly.
    2. With counts this large (up to 52 418), tiny differences from the model are hugely significant. A bad
       $\chi^2$ does not mean the model is useless, only that it is not **exactly** right.
    3. A model should be judged by what it **predicts**, not only by what it fits. Fit the part of the data
       that matters for the question.
    4. Which range you fit is part of the answer. Say which, and show the others.
    """)
    return


if __name__ == "__main__":
    app.run()
