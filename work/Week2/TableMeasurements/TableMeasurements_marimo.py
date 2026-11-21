import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import io

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    from appstat.data import read_text
    from appstat.fitting import fit_least_squares
    from appstat.plotting import plot_fit
    from appstat.stats import (
        chauvenet,
        chauvenet_nsigma,
        chi2_pvalue,
        weighted_mean,
        weighted_mean_chi2,
    )

    GOOD, BAD, FIT = "#0072B2", "#D55E00", "#009E73"
    RULERS = {"30 cm ruler": 0, "2 m folding rule": 2}

    def read_year(year):
        """The four columns of one year: length and uncertainty for each of the two rulers."""
        text = read_text(mo.notebook_location(), f"data_TableMeasurements{year}.txt")
        return np.loadtxt(io.StringIO(text), skiprows=2, unpack=True)

    years = [read_year(year) for year in range(2009, 2026)]
    data = np.concatenate(years, axis=1)  # rows: 30cm length, 30cm unc, 2m length, 2m unc

    def ruler(name):
        """Length and uncertainty of one ruler."""
        return data[RULERS[name]], data[RULERS[name] + 1]

    def unweighted_cleaned(length, p_global):
        """The valid lengths (0 to 6 m), and which of them Chauvenet's criterion accepts."""
        valid = length[(0 < length) & (length < 6)]
        return valid, chauvenet(valid, p_global)

    def weighted_cleaned(length, error, p_global, strictness):
        """Weighted analysis: pulls around the cleaned unweighted mean, and which pass the cut."""
        usable = (0 < length) & (length < 6) & (error > 1e-5)
        length, error = length[usable], error[usable]
        valid, accepted = unweighted_cleaned(length, 0.05)
        pull = (length - valid[accepted].mean()) / error
        limit = strictness * chauvenet_nsigma(len(length), p_global)
        return length, error, pull, np.abs(pull) < limit, limit

    return (
        BAD,
        FIT,
        GOOD,
        RULERS,
        chauvenet_nsigma,
        chi2_pvalue,
        data,
        fit_least_squares,
        mo,
        np,
        plot_fit,
        plt,
        ruler,
        stats,
        unweighted_cleaned,
        weighted_cleaned,
        weighted_mean,
        weighted_mean_chi2,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # How long is the lecture table?

    Every year, students measure the table in Auditorium A at NBI twice: with a **30 cm ruler** and with a
    **2 m folding rule**, and state how uncertain they think they are. The 17 years give 962 pairs of
    measurements. Most are good. Some are wrong, by a ruler length, or by a slip. You must decide which to keep,
    and **argue for each choice with numbers**.

    Every interactive block below has its controls directly above its plots.

    ---

    **Adapted from** the Table Measurements exercise by **Troels C. Petersen** (Niels Bohr Institute,
    3rd of December 2025), from the course *Applied Statistics 2025/26*. The original is an exercise with
    two example solutions. This version keeps its data and its analysis steps, and is rewritten as an
    interactive demonstration. It does not blind the result, as the exercise does, so you can see the
    answer at every step.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Always plot the data first

    Look at the distribution before you compute anything. Zoom in on the peak, and switch the y axis
    between logarithmic and linear. The naive mean and standard deviation of **all** measurements are
    shown below the plot. Is that standard deviation what you would expect from a good ruler?
    """)
    return


@app.cell
def _(RULERS, mo):
    plot_ruler = mo.ui.dropdown(list(RULERS), value="30 cm ruler", label="Ruler")
    plot_zoom = mo.ui.dropdown(["Everything", "Around the peak"], value="Everything", label="Range")
    plot_log = mo.ui.switch(value=True, label="Logarithmic y axis")
    return plot_log, plot_ruler, plot_zoom


@app.cell
def _(GOOD, mo, np, plot_log, plot_ruler, plot_zoom, plt, ruler):
    _length, _ = ruler(plot_ruler.value)
    _valid = _length[(0 < _length) & (_length < 6)]
    _centre = np.median(_valid)
    _low, _high = (0.0, 5.0) if plot_zoom.value == "Everything" else (_centre - 0.3, _centre + 0.3)

    _fig, _ax = plt.subplots(figsize=(13, 3.8))
    _ax.hist(_valid, bins=np.linspace(_low, _high, 200), histtype="step", color=GOOD, lw=1.5)
    if plot_log.value:
        _ax.set_yscale("log")
    _ax.set(xlabel="Table length [m]", ylabel="Measurements per bin")
    _fig.tight_layout()
    mo.vstack(
        [
            mo.hstack([plot_ruler, plot_zoom, plot_log], justify="start", gap=2),
            _fig,
            mo.callout(
                mo.md(
                    f"All {len(_valid)} valid measurements (0 to 6 m): mean {_valid.mean():.3f} m, "
                    f"standard deviation **{_valid.std(ddof=1):.3f} m**, uncertainty on the mean "
                    f"{_valid.std(ddof=1) / np.sqrt(len(_valid)):.3f} m. A good ruler is better than a few millimetres."
                ),
                kind="warn",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Reject outliers: Chauvenet's criterion

    Without the uncertainties, only the distribution can tell. **Chauvenet's criterion** rejects the farthest
    point if a point that far is unlikely among so many measurements, then recalculates the mean and width
    and tries again. $p$ is the probability that it rejects at least one **good** point: a lower $p$ keeps more.
    """)
    return


@app.cell
def _(RULERS, mo):
    cut_ruler = mo.ui.dropdown(list(RULERS), value="30 cm ruler", label="Ruler")
    cut_p = mo.ui.slider(
        0.01,
        0.5,
        0.01,
        value=0.05,
        show_value=True,
        label="Probability p of rejecting a good point",
    )
    return cut_p, cut_ruler


@app.cell
def _(BAD, GOOD, chauvenet_nsigma, cut_p, cut_ruler, mo, np, plt, ruler, unweighted_cleaned):
    _length, _ = ruler(cut_ruler.value)
    _valid, _accepted = unweighted_cleaned(_length, cut_p.value)
    _kept, _rejected = _valid[_accepted], _valid[~_accepted]

    _fig, _ax = plt.subplots(figsize=(13, 3.8))
    _bins = np.linspace(0, 5, 250)
    _ax.hist(
        _kept,
        bins=_bins,
        histtype="stepfilled",
        color=GOOD,
        alpha=0.6,
        label=f"Accepted ({len(_kept)})",
    )
    _ax.hist(
        _rejected,
        bins=_bins,
        histtype="stepfilled",
        color=BAD,
        alpha=0.9,
        label=f"Rejected ({len(_rejected)})",
    )
    _ax.set(xlabel="Table length [m]", ylabel="Measurements per bin", yscale="log")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack(
        [
            mo.hstack([cut_ruler, cut_p], justify="start", gap=2),
            _fig,
            mo.callout(
                mo.md(
                    f"Rejected **{len(_rejected)} of {len(_valid)}**. Standard deviation "
                    f"{_valid.std(ddof=1):.3f} → **{_kept.std(ddof=1):.4f} m**. Mean "
                    f"{_valid.mean():.3f} → **{_kept.mean():.4f} ± {_kept.std(ddof=1) / np.sqrt(len(_kept)):.4f} m**. "
                    f"A point is rejected beyond {chauvenet_nsigma(len(_kept), cut_p.value):.2f} σ."
                ),
                kind="success",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Use the stated uncertainties: pulls and the weighted mean

    Each person also stated an uncertainty $\sigma_i$. The **pull** $z_i = (x_i - \hat\mu)/\sigma_i$ says how many
    of its own $\sigma$ a measurement is from the best estimate. If everyone were right, the pulls would
    be a unit Gaussian (the line). They are not: some people underestimated their uncertainty.
    Cut at a pull limit, and watch the $\chi^2$ of the weighted mean fall.
    """)
    return


@app.cell
def _(RULERS, mo):
    pull_ruler = mo.ui.dropdown(list(RULERS), value="30 cm ruler", label="Ruler")
    pull_p = mo.ui.slider(
        0.01,
        0.5,
        0.01,
        value=0.05,
        show_value=True,
        label="Probability p of rejecting a good point",
    )
    pull_strict = mo.ui.slider(
        0.3, 1.2, 0.05, value=0.9, show_value=True, label="Cut strictness (1 = the Gaussian limit)"
    )
    return pull_p, pull_ruler, pull_strict


@app.cell
def _(
    BAD,
    GOOD,
    chi2_pvalue,
    mo,
    np,
    plt,
    pull_p,
    pull_ruler,
    pull_strict,
    ruler,
    stats,
    weighted_cleaned,
    weighted_mean,
    weighted_mean_chi2,
):
    _length, _error = ruler(pull_ruler.value)
    _length, _error, _pull, _accepted, _limit = weighted_cleaned(
        _length, _error, pull_p.value, pull_strict.value
    )
    _before = weighted_mean_chi2(_length, _error)
    _after = weighted_mean_chi2(_length[_accepted], _error[_accepted])
    _mean, _mean_error = weighted_mean(_length[_accepted], _error[_accepted])

    _fig, _ax = plt.subplots(figsize=(13, 3.8))
    _bins = np.linspace(-10, 10, 100)
    _ax.hist(
        _pull[_accepted],
        bins=_bins,
        histtype="stepfilled",
        color=GOOD,
        alpha=0.6,
        label=f"Accepted ({_accepted.sum()})",
    )
    _ax.hist(
        _pull[~_accepted],
        bins=_bins,
        histtype="stepfilled",
        color=BAD,
        alpha=0.9,
        label=f"Rejected ({(~_accepted).sum()})",
    )
    _grid = np.linspace(-10, 10, 400)
    _ax.plot(_grid, len(_pull) * 0.2 * stats.norm.pdf(_grid), "k--", label="Unit Gaussian")
    _ax.axvline(_limit, color="k", lw=1)
    _ax.axvline(-_limit, color="k", lw=1)
    _ax.set(
        xlabel="Pull (x − mean) / σ", ylabel="Measurements per bin", yscale="log", ylim=(0.5, None)
    )
    _ax.legend()
    _fig.tight_layout()
    _p_after = chi2_pvalue(*_after)
    mo.vstack(
        [
            mo.hstack([mo.vstack([pull_ruler, pull_p]), pull_strict], widths="equal"),
            _fig,
            mo.callout(
                mo.md(
                    f"Weighted mean **{_mean:.4f} ± {_mean_error:.4f} m**. "
                    f"χ²/Ndof: {_before[0]:.0f}/{_before[1]} before the cut, **{_after[0]:.0f}/{_after[1]}** "
                    f"after (p = {_p_after:.3g}). Pull limit ±{_limit:.2f}."
                ),
                kind="success" if _p_after > 0.01 else "warn",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Do the two rulers agree?

    A cross check: the 30 cm ruler and the 2 m folding rule measure the **same** table. Take each ruler's
    best result, with the cuts of blocks 2 and 3, and ask how many $\sigma$ apart they are.
    """)
    return


@app.cell
def _(mo):
    agree_method = mo.ui.dropdown(
        ["Unweighted, Chauvenet", "Weighted, pull cut"],
        value="Unweighted, Chauvenet",
        label="Method",
    )
    return (agree_method,)


@app.cell
def _(
    GOOD,
    agree_method,
    chi2_pvalue,
    mo,
    np,
    plt,
    ruler,
    unweighted_cleaned,
    weighted_cleaned,
    weighted_mean,
):
    _results = {}
    for _name in ("30 cm ruler", "2 m folding rule"):
        _length, _error = ruler(_name)
        if agree_method.value == "Unweighted, Chauvenet":
            _valid, _accepted = unweighted_cleaned(_length, 0.05)
            _kept = _valid[_accepted]
            _results[_name] = (_kept.mean(), _kept.std(ddof=1) / np.sqrt(len(_kept)))
        else:
            _length, _error, _, _accepted, _ = weighted_cleaned(_length, _error, 0.05, 0.9)
            _results[_name] = weighted_mean(_length[_accepted], _error[_accepted])
    (_a, _ea), (_b, _eb) = _results.values()
    _difference, _error_difference = _a - _b, np.hypot(_ea, _eb)
    _nsigma = abs(_difference) / _error_difference

    _fig, _ax = plt.subplots(figsize=(13, 2.8))
    for _y, (_name, (_value, _err)) in enumerate(_results.items()):
        _ax.errorbar(_value, _y, xerr=_err, fmt="o", color=GOOD, capsize=4, ms=7)
    _ax.set(yticks=[0, 1], yticklabels=list(_results), xlabel="Table length [m]", ylim=(-0.5, 1.5))
    _fig.tight_layout()
    mo.vstack(
        [
            agree_method,
            _fig,
            mo.callout(
                mo.md(
                    f"30 cm: **{_a:.4f} ± {_ea:.4f} m**. 2 m: **{_b:.4f} ± {_eb:.4f} m**. "
                    f"Difference {_difference * 1000:+.1f} ± {_error_difference * 1000:.1f} mm, "
                    f"which is **{_nsigma:.1f} σ**, p = {chi2_pvalue(_nsigma**2, 1):.3f}."
                ),
                kind="success" if _nsigma < 3 else "warn",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A. Fit the distribution instead of cutting it

    Cutting data needs a decision for every point. Instead, **fit** the whole peak with a function
    that has room for the bad measurements too. A single Gaussian cannot describe the tails. A **double
    Gaussian** (a narrow and a wide one) or a **Cauchy** can. Widen the window to include more of the tails, and compare the $\chi^2$.
    """)
    return


@app.cell
def _(RULERS, mo):
    fit_ruler = mo.ui.dropdown(list(RULERS), value="2 m folding rule", label="Ruler")
    fit_model = mo.ui.dropdown(
        ["Gaussian", "Double Gaussian", "Cauchy"], value="Gaussian", label="Fit function"
    )
    fit_window = mo.ui.slider(
        0.01, 0.3, 0.01, value=0.05, show_value=True, label="Half width of the fitted window [m]"
    )
    return fit_model, fit_ruler, fit_window


@app.cell
def _(fit_least_squares, fit_model, fit_ruler, fit_window, mo, np, plot_fit, plt, ruler, stats):
    _length, _ = ruler(fit_ruler.value)
    _centre, _half = np.median(_length[(0 < _length) & (_length < 6)]), fit_window.value
    _inside = _length[np.abs(_length - _centre) < _half]
    _counts, _edges = np.histogram(_inside, bins=60, range=(_centre - _half, _centre + _half))
    _x, _width, _used = (_edges[1:] + _edges[:-1]) / 2, _edges[1] - _edges[0], _counts > 0
    _total, _sigma = float(len(_inside)), max(_inside.std(), 1e-3)

    def _gaussian(x, N, mu, sigma):
        return N * _width * stats.norm.pdf(x, mu, sigma)

    def _double(x, N, mu, frac, sigma1, sigma2):
        return (
            N
            * _width
            * (frac * stats.norm.pdf(x, mu, sigma1) + (1 - frac) * stats.norm.pdf(x, mu, sigma2))
        )

    def _cauchy(x, N, mu, gamma):
        return N * _width * stats.cauchy.pdf(x, mu, gamma)

    _models = {
        "Gaussian": (_gaussian, {"N": _total, "mu": _centre, "sigma": _sigma}),
        "Double Gaussian": (
            _double,
            {"N": _total, "mu": _centre, "frac": 0.5, "sigma1": _sigma / 3, "sigma2": _sigma},
        ),
        "Cauchy": (_cauchy, {"N": _total, "mu": _centre, "gamma": _sigma / 2}),
    }
    _model, _start = _models[fit_model.value]
    _fig, _ax = plt.subplots(figsize=(13, 4.0))
    try:
        _fit = fit_least_squares(_model, _x[_used], _counts[_used], np.sqrt(_counts[_used]), _start)
        plot_fit(_ax, _x[_used], _counts[_used], np.sqrt(_counts[_used]), _model, _fit)
        _note = mo.callout(
            mo.md(
                "The fit converged."
                if _fit.valid
                else "The fit did **not** converge. Try another function or window."
            ),
            kind="success" if _fit.valid else "warn",
        )
    except ValueError:
        _ax.hist(_inside, bins=60)
        _note = mo.callout(mo.md("Too few non-empty bins to fit. Widen the window."), kind="warn")
    _ax.set(xlabel="Table length [m]", ylabel="Measurements per bin")
    _fig.tight_layout()
    mo.vstack([mo.hstack([fit_ruler, fit_model, fit_window], justify="start", gap=2), _fig, _note])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to take away

    1. **Plot the data first.** The standard deviation of the raw data was far too large for a ruler, which
       told you at once that some measurements were wrong.
    2. **Remove outliers with a stated criterion** (Chauvenet), never by feel, and keep a list of what you removed.
    3. When the measurements come with uncertainties, judge them by their **pull**. The pulls were
       not a unit Gaussian, so the stated uncertainties were not all honest.
    4. A **weighted mean** is only right if the $\chi^2$ is reasonable. A huge $\chi^2$ means the data disagree.
    5. **Cross check** with an independent measurement, here the other ruler.
    6. Fitting the whole distribution is an alternative to cutting, if the function is flexible enough.
    """)
    return


if __name__ == "__main__":
    app.run()
