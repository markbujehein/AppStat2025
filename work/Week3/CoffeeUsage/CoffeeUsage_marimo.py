import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    from datetime import date, timedelta

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    from appstat.fitting import fit_least_squares
    from appstat.models import vacation_line
    from appstat.stats import chi2_pvalue

    DATA, FIT, FIT2 = "#000000", "#0072B2", "#D55E00"
    FIRST_DAY = date(2009, 11, 4)  # day 0
    TOTAL_CUPS = 36716  # cups brewed when the machine was decommissioned
    days = np.array([0, 9, 14, 21, 23, 36, 65, 76, 86, 99, 194, 280, 309], dtype=float)
    cups = np.array(
        [28479, 28674, 28777, 28964, 29041, 29374, 29650, 30001, 30221, 30498, 32412, 33676, 34008],
        dtype=float,
    )

    def line(x, offset, slope):
        return offset + slope * x

    def error_slider(label="Uncertainty on each count [cups]"):
        return mo.ui.slider(10, 100, 5, value=30, show_value=True, label=label)

    def draw_data(ax, error):
        ax.errorbar(days, cups, error, fmt="o", color=DATA, label="Data", zorder=3)
        ax.set(xlabel="Days since 4 November 2009", ylabel="Cups brewed in total", xlim=(-20, 330))

    def to_date(day):
        return FIRST_DAY + timedelta(days=int(round(day)))

    return (
        DATA,
        FIT,
        FIT2,
        TOTAL_CUPS,
        cups,
        days,
        draw_data,
        error_slider,
        fit_least_squares,
        chi2_pvalue,
        line,
        mo,
        np,
        plt,
        stats,
        to_date,
        vacation_line,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Coffee usage: how long was the Christmas vacation?

    In 2009-2010 the total number of cups the old coffee machine in the NBI HEP group had ever brewed was written
    down on 13 occasions. If it were used at a **constant rate**, the points would fall on a line. They do not, because
    the machine rests over the Christmas vacation. From a fit, you can find

    - whether the numbers follow constant use, with a given uncertainty,
    - the **length of the vacation**, and
    - when the machine was commissioned, and when it will have brewed its last cup (36716).

    Every interactive block below has its controls directly above its plots.

    ---

    **Adapted from** the Coffee Usage exercise by **Troels C. Petersen** (Niels Bohr Institute,
    4th of December 2023), from the course *Applied Statistics 2025/26*. The original is an exercise with an example
    solution. This version keeps its data and its three questions, and is rewritten as an interactive demonstration.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Constant use?

    Fit a straight line to the points in a range. The uncertainty on each count is a **guess** (30 cups).
    Is the constant-use hypothesis credible? The callout also says **how large the uncertainty has to be** for
    the probability to reach 5 %.
    """)
    return


@app.cell
def _(error_slider, mo):
    line_error = error_slider()
    line_range = mo.ui.range_slider(
        -10, 320, 5, value=[60, 105], show_value=True, label="Fitted days"
    )
    return line_error, line_range


@app.cell
def _(
    FIT,
    cups,
    chi2_pvalue,
    days,
    draw_data,
    fit_least_squares,
    line,
    line_error,
    line_range,
    mo,
    np,
    plt,
    stats,
):
    _low, _high = line_range.value
    _inside = (_low <= days) & (days <= _high)
    _fig, _ax = plt.subplots(figsize=(13, 3.8))
    draw_data(_ax, line_error.value)
    try:
        _fit = fit_least_squares(
            line,
            days[_inside],
            cups[_inside],
            np.full(_inside.sum(), float(line_error.value)),
            {"offset": 28000.0, "slope": 20.0},
        )
        _grid = np.linspace(_low, _high, 100)
        _ax.plot(_grid, line(_grid, *_fit.values), color=FIT, lw=2, label="Straight line")
        _p = chi2_pvalue(_fit.fval, _fit.ndof)
        # chi2 scales as 1/error^2, so the error that gives p = 5 % follows from the critical chi2
        _needed = line_error.value * np.sqrt(_fit.fval / stats.chi2.isf(0.05, _fit.ndof))
        _note = mo.callout(
            mo.md(
                f"{_inside.sum()} points, χ²/Ndof = {_fit.fval:.1f}/{_fit.ndof}, **p = {_p:.2f}**. "
                f"Slope {_fit.values['slope']:.2f} ± {_fit.errors['slope']:.2f} cups per day. "
                f"The probability reaches 5 % for an uncertainty of **{_needed:.0f} cups** or more."
            ),
            kind="success" if _p > 0.05 else "warn",
        )
    except ValueError:
        _note = mo.callout(
            mo.md("Too few points in the range. Widen it to at least three days."), kind="warn"
        )
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([mo.hstack([line_error, line_range], justify="start", gap=2), _fig, _note])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Add the vacation

    Let the use **pause** for a vacation of unknown length, starting 40 days after the first point (14 December).
    The line is the same before and after, but after the vacation it is delayed by its length. Fit that length.
    Does the probability improve? Widen the range to see how far the model holds.
    """)
    return


@app.cell
def _(error_slider, mo):
    vacation_error = error_slider()
    vacation_range = mo.ui.range_slider(
        -10, 320, 5, value=[-10, 110], show_value=True, label="Fitted days"
    )
    vacation_start = mo.ui.slider(
        30, 60, 1, value=40, show_value=True, label="Vacation starts on day"
    )
    return vacation_error, vacation_range, vacation_start


@app.cell
def _(
    FIT2,
    cups,
    chi2_pvalue,
    days,
    draw_data,
    fit_least_squares,
    mo,
    np,
    plt,
    vacation_error,
    vacation_line,
    vacation_range,
    vacation_start,
):
    _low, _high = vacation_range.value
    _inside = (_low <= days) & (days <= _high)

    def _model(x, offset, slope, length):
        return vacation_line(x, offset, slope, length, start=vacation_start.value)

    _fig, _ax = plt.subplots(figsize=(13, 3.8))
    draw_data(_ax, vacation_error.value)
    try:
        _fit = fit_least_squares(
            _model,
            days[_inside],
            cups[_inside],
            np.full(_inside.sum(), float(vacation_error.value)),
            {"offset": 28500.0, "slope": 25.0, "length": 14.0},
        )
        _grid = np.linspace(_low, _high, 400)
        _ax.plot(_grid, _model(_grid, *_fit.values), color=FIT2, lw=2, label="With vacation")
        _p = chi2_pvalue(_fit.fval, _fit.ndof)
        _note = mo.callout(
            mo.md(
                f"χ²/Ndof = {_fit.fval:.1f}/{_fit.ndof}, **p = {_p:.2f}**. "
                f"Vacation length **{_fit.values['length']:.1f} ± {_fit.errors['length']:.1f} days**."
            ),
            kind="success" if _p > 0.05 else "warn",
        )
    except ValueError:
        _note = mo.callout(
            mo.md("Too few points in the range. Widen it to at least four days."), kind="warn"
        )
    _ax.legend()
    _fig.tight_layout()
    mo.vstack(
        [
            mo.hstack([vacation_error, vacation_range, vacation_start], justify="start", gap=2),
            _fig,
            _note,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. When was the machine commissioned?

    The machine brewed 36716 cups in total before it was decommissioned. A straight line through **all** the
    points, running back to 0 cups and forward to 36716 cups, gives a rough date for both. It ignores the vacation
    and the line is not a good fit, so the uncertainties are scaled up by $\sqrt{\chi^2/N_\mathrm{dof}}$. The
    two dates come from the fitted offset and slope **together**, with their correlation, by simulation.
    """)
    return


@app.cell
def _(error_slider):
    commission_error = error_slider()
    return (commission_error,)


@app.cell
def _(
    FIT,
    TOTAL_CUPS,
    cups,
    days,
    draw_data,
    commission_error,
    fit_least_squares,
    line,
    mo,
    np,
    plt,
    to_date,
):
    _fit = fit_least_squares(
        line,
        days,
        cups,
        np.full(len(days), float(commission_error.value)),
        {"offset": 28000.0, "slope": 20.0},
    )
    _scale = max(1.0, np.sqrt(_fit.fval / _fit.ndof))
    _rng = np.random.default_rng(1)
    _offset, _slope = _rng.multivariate_normal(
        np.asarray(_fit.values), np.asarray(_fit.covariance) * _scale**2, 20_000
    ).T
    _start, _end = -_offset / _slope, (TOTAL_CUPS - _offset) / _slope

    _grid = np.linspace(-700, 500, 100)
    _fig, _ax = plt.subplots(figsize=(13, 3.8))
    draw_data(_ax, commission_error.value)
    _ax.plot(_grid, line(_grid, *_fit.values), color=FIT, lw=2, label="Straight line, all points")
    _ax.axhline(0, color="gray", lw=0.5)
    _ax.axhline(TOTAL_CUPS, color="gray", ls=":", label="36716 cups")
    _ax.set(xlim=(-700, 500), ylim=(-1000, 40000))
    _ax.legend(loc="upper left")
    _fig.tight_layout()
    mo.vstack(
        [
            commission_error,
            _fig,
            mo.callout(
                mo.md(
                    f"Commissioned **{to_date(np.median(_start)):%B %Y}** ± {np.std(_start):.0f} days. "
                    f"Decommissioned **{to_date(np.median(_end)):%B %Y}** ± {np.std(_end):.0f} days. "
                    f"χ²/Ndof = {_fit.fval:.0f}/{_fit.ndof}, so the errors were scaled by {_scale:.1f}. "
                    "(The machine was in fact commissioned around July 2005, so the straight line is rough.)"
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

    1. A **p-value** says whether a model is credible given the uncertainties. With guessed uncertainties it can
       be turned around: how large would they have to be?
    2. A model that **knows about the vacation** describes the data far better than a plain line. Its extra parameter
       is a physical quantity, with an uncertainty.
    3. Extrapolating far beyond the data (to commissioning and decommissioning) needs the offset **and** slope
       together, with their correlation, and is only as good as the model.
    4. When the fit is bad, the fitted uncertainties are too small. Scaling by $\sqrt{\chi^2/N_\mathrm{dof}}$ is a first
       guess at an honest value.
    """)
    return


if __name__ == "__main__":
    app.run()
