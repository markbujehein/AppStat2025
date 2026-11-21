import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import io

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from appstat.data import read_text
    from appstat.fitting import fit_least_squares
    from appstat.models import apply_corrections, offset_exponential, parabola
    from appstat.stats import chi2_pvalue, profile_x

    RAW, CALIBRATED, FIT = "#0072B2", "#D55E00", "#009E73"
    table = np.loadtxt(io.StringIO(read_text(mo.notebook_location(), "data_calib.txt")))
    d_known, d_measured = table[:, 0], table[:, 1]
    variables = dict(zip(["lsig", "lbkg", "temp", "tsky"], table[:, 2:].T, strict=True))
    relative = (d_measured - d_known) / d_known
    # variable: (model, start values, range of the variable, default fitted range)
    TRENDS = {
        "lsig": (parabola, {"p0": 1.0, "p1": 1.0, "p2": 1.0}, (0, 50), (0, 50)),
        "lbkg": (parabola, {"p0": 1.0, "p1": 1.0, "p2": 1.0}, (0, 35), (0, 25)),
        "temp": (parabola, {"p0": 1.0, "p1": 1.0, "p2": 1.0}, (1, 15), (1, 6)),
        "tsky": (offset_exponential, {"p0": -0.2, "p1": 0.02, "p2": 1.0}, (0, 1), (0, 1)),
    }
    CORRECTED = ["lbkg", "temp", "tsky"]  # lsig shows no trend in the exercise
    Y_RANGE = (-1.0, 6.0)  # relative differences outside this are ignored in the profiles

    def fit_trend(name, low, high, rel=relative):
        """Profile of the relative difference ``rel`` against ``name``, and a fit of its trend."""
        model, start, x_range, _ = TRENDS[name]
        centres, means, errors = profile_x(variables[name], rel, 50, x_range, Y_RANGE)
        inside = (low < centres) & (centres < high)
        fit = fit_least_squares(model, centres[inside], means[inside], errors[inside], start)
        return fit, model, centres, means, errors, inside

    def calibrated_distance(names):
        """
        The measured distances corrected for the trends of the variables in ``names``.

        One trend at a time: each is fitted to the data already corrected by the ones before it,
        so the overall offset is removed once and not once per variable.
        """
        corrected = d_measured
        for name in (n for n in CORRECTED if n in names):
            fit, model, *_ = fit_trend(name, *TRENDS[name][3], (corrected - d_known) / d_known)
            corrected = apply_corrections(corrected, model(variables[name], *fit.values))
        return corrected

    return (
        CALIBRATED,
        CORRECTED,
        FIT,
        RAW,
        TRENDS,
        calibrated_distance,
        chi2_pvalue,
        d_known,
        fit_trend,
        mo,
        np,
        plt,
        relative,
        variables,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Calibration: correcting a measurement for what disturbs it

    A new method measures the **distance to stars**. For 10 000 stars the true distance is known, so you can see how
    the method goes wrong. The error depends on four things: the light from the star (`lsig`), the background light
    of the sky (`lbkg`), the temperature of the star (`temp`) and the transparency of the sky (`tsky`).

    The **relative difference** is $R_D = (D_\mathrm{measured} - D_\mathrm{known}) / D_\mathrm{known}$. A good
    method has $R_D$ near 0 and narrow. A measurement that is a fraction $f$ too high is corrected by dividing by
    $1 + f$.

    Every interactive block below has its controls directly above its plots.

    ---

    **Adapted from** the Calibration exercise by **Troels C. Petersen** (Niels Bohr Institute), from the course
    *Applied Statistics 2025/26*. The original is an exercise with an example solution. This version keeps its data
    and its method, and is rewritten as an interactive demonstration.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Does the error depend on this variable?

    The dots are the **mean** of $R_D$ in slices of the variable, with the uncertainty of each mean. The line is a fit
    of the trend: a parabola, or for `tsky` an offset plus an exponential. A trend means the variable disturbs
    the measurement. Which of the four do? Change the fitted range, as a poor fit at the edge can ruin the calibration.
    """)
    return


@app.cell
def _(TRENDS, mo):
    look_variable = mo.ui.dropdown(list(TRENDS), value="temp", label="Variable")
    return (look_variable,)


@app.cell
def _(TRENDS, look_variable, mo):
    _low, _high = TRENDS[look_variable.value][2]
    look_range = mo.ui.range_slider(
        _low,
        _high,
        (_high - _low) / 50,
        value=TRENDS[look_variable.value][3],
        show_value=True,
        label="Fitted range",
    )
    return (look_range,)


@app.cell
def _(FIT, RAW, chi2_pvalue, fit_trend, look_range, look_variable, mo, np, plt):
    _controls = mo.hstack([look_variable, look_range], justify="start", gap=2)
    try:
        _fit, _model, _centres, _means, _errors, _inside = fit_trend(
            look_variable.value, *look_range.value
        )
    except ValueError:
        _output = mo.callout(
            mo.md("Too few slices in the range for this model. Widen it to at least four."),
            kind="warn",
        )
    else:
        _fig, _ax = plt.subplots(figsize=(13, 3.8))
        _ax.errorbar(_centres, _means, _errors, fmt=".", color=RAW, label="Mean of $R_D$")
        _grid = np.linspace(_centres[0], _centres[-1], 300)
        _ax.plot(_grid, _model(_grid, *_fit.values), color=FIT, lw=2, label="Fit")
        _ax.axhline(0, color="k", lw=0.5)
        _ax.set(xlabel=look_variable.value, ylabel="$R_D$")
        _ax.legend()
        _fig.tight_layout()
        _p = chi2_pvalue(_fit.fval, _fit.ndof)
        _output = mo.vstack(
            [
                _fig,
                mo.callout(
                    mo.md(
                        f"{_inside.sum()} slices, χ²/Ndof = {_fit.fval:.1f}/{_fit.ndof}, p = {_p:.2g}. "
                        f"Mean $R_D$ changes by **{np.ptp(_model(_grid, *_fit.values)):.2f}** "
                        "across the variable."
                    ),
                    kind="info",
                ),
            ]
        )
    mo.vstack([_controls, _output])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Apply the corrections

    Divide each measurement by $1 + f(x)$ for each trend you switch on, one after another. See how the offset and the
    width of the distribution change. Which correction reduces the width most?
    """)
    return


@app.cell
def _(CORRECTED, mo):
    use_corrections = mo.ui.multiselect(CORRECTED, value=CORRECTED, label="Corrections")
    return (use_corrections,)


@app.cell
def _(CALIBRATED, RAW, calibrated_distance, d_known, mo, np, plt, relative, use_corrections):
    _after = (calibrated_distance(use_corrections.value) - d_known) / d_known
    _low, _high = -2.0, 4.0
    _edges = np.linspace(_low, _high, 201)
    _fig, _ax = plt.subplots(figsize=(13, 3.8))
    _ax.hist(relative, _edges, histtype="step", lw=2, color=RAW, label="Raw")
    _ax.hist(_after, _edges, histtype="step", lw=2, color=CALIBRATED, label="Calibrated")
    _ax.set(xlabel="$R_D$", ylabel="Stars")
    _ax.legend()
    _fig.tight_layout()

    def _inside(values):
        return values[(_low < values) & (values < _high)]

    mo.vstack(
        [
            use_corrections,
            _fig,
            mo.callout(
                mo.md(
                    f"Within ({_low:g}, {_high:g}): raw mean {_inside(relative).mean():+.3f}, "
                    f"width **{_inside(relative).std(ddof=1):.3f}**. "
                    f"Calibrated mean {_inside(_after).mean():+.3f}, width **{_inside(_after).std(ddof=1):.3f}**."
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

    1. To calibrate, **look at the error against each variable** (a profile) and fit the trend.
    2. A relative bias $f$ is removed by **dividing by $1+f$**, one effect after the other.
    3. Fit the trends **one after another**, each on the data corrected by the one before. Fitted all on the raw
       data, each would include the whole offset, and the mean would end up far from zero.
    4. A trend is fitted in a **range**. A badly fitted edge adds an error of its own.
    """)
    return


if __name__ == "__main__":
    app.run()
