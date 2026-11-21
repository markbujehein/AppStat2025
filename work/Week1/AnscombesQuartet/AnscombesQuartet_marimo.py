import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from appstat.stats import chi2_pvalue

    return chi2_pvalue, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Anscombe's quartet: never trust the numbers alone

    Four data sets. Each has 11 points, and each gets the same fitted line from the same
    summary numbers. Do they look alike?

    Scroll down, read the numbers first, and **guess** before you look at the data.

    Reference: Anscombe, *Graphs in Statistical Analysis* (1973).

    ---

    **Adapted from** the Anscombe's Quartet exercise by **Troels C. Petersen** (Niels Bohr
    Institute, 10th of November 2025), from the course *Applied Statistics 2025/26*. The
    original is an exercise. This version keeps its four data sets and its straight-line fit, and
    is rewritten as an interactive demonstration.
    """)
    return


@app.cell
def _(np):
    x = np.array(
        [
            [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0],
            [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0],
            [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0],
            [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 19.0, 8.0, 8.0, 8.0],
        ]
    )
    y = np.array(
        [
            [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
            [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74],
            [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73],
            [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89],
        ]
    )
    names = ["I", "II", "III", "IV"]

    # One colour per role, the same in every plot below
    DATA = "#0072B2"
    FIT = "#D55E00"
    FLAG = "#CC79A7"
    return DATA, FIT, FLAG, names, x, y


@app.cell
def _(np):
    def fit_line(x, y):
        """Least-squares line with an uncertainty of 1 on every point (as in the exercise)."""
        (slope, intercept), covariance = np.polyfit(x, y, 1, cov="unscaled")
        return slope, intercept, np.sqrt(np.diag(covariance))  # errors on (slope, intercept)

    def leave_one_out_slope_change(x, y):
        """Change of the slope when each point is left out in turn. NaN if no line is left."""
        slope = fit_line(x, y)[0]
        change = np.full(len(x), np.nan)
        for i in range(len(x)):  # eleven points, each with its own meaning
            keep = np.arange(len(x)) != i
            if np.ptp(x[keep]) > 0:
                change[i] = fit_line(x[keep], y[keep])[0] - slope
        return change

    return fit_line, leave_one_out_slope_change


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The numbers

    Everything you would put in a report, for all four sets:
    """)
    return


@app.cell
def _(chi2_pvalue, fit_line, mo, names, np, x, y):
    _rows = ""
    for _i, _name in enumerate(names):
        _slope, _intercept, _ = fit_line(x[_i], y[_i])
        _chi2 = np.sum((y[_i] - (_intercept + _slope * x[_i])) ** 2)
        _rows += (
            f"| {_name} | {x[_i].mean():.2f} | {x[_i].var():.2f} | {y[_i].mean():.2f} | "
            f"{y[_i].var():.2f} | {np.corrcoef(x[_i], y[_i])[0, 1]:.3f} | "
            f"{_intercept:.2f} + {_slope:.3f} x | {_chi2:.1f} | {chi2_pvalue(_chi2, 9):.2f} |\n"
        )
    mo.vstack(
        [
            mo.md(
                "| set | mean x | variance x | mean y | variance y | correlation | fitted line "
                "| χ² (σ = 1) | p-value |\n|---|---|---|---|---|---|---|---|---|\n" + _rows
            ),
            mo.callout(
                mo.md(
                    "Same line, same correlation, and the **same χ² and p-value**. The goodness of fit "
                    "says every one of these fits is acceptable."
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Now look

    Do you expect four similar clouds of points? Decide, then flip the switch.
    """)
    return


@app.cell
def _(mo):
    reveal = mo.ui.switch(label="Show the four data sets")
    reveal
    return (reveal,)


@app.cell
def _(DATA, FIT, fit_line, mo, names, np, plt, reveal, x, y):
    mo.stop(
        not reveal.value,
        mo.callout(
            mo.md("Waiting for your guess. The plots appear when you flip the switch."),
            kind="neutral",
        ),
    )
    _fig, _axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True, sharey=True)
    _line_x = np.array([3.0, 20.0])
    for _ax, _name, _x, _y in zip(_axes.flatten(), names, x, y, strict=True):
        _slope, _intercept, _errors = fit_line(_x, _y)
        _ax.plot(_line_x, _intercept + _slope * _line_x, color=FIT, lw=2, label="Fitted line")
        _ax.plot(_x, _y, "o", color=DATA, ms=8, label="Data")
        _ax.set_title(f"Set {_name}", loc="left", fontweight="bold")
        _ax.text(
            0.05,
            0.95,
            f"y = {_intercept:.2f} + {_slope:.3f} x",
            transform=_ax.transAxes,
            va="top",
            family="monospace",
        )
    _axes[0, 0].legend(loc="lower right")
    for _ax in _axes[1]:
        _ax.set_xlabel("x")
    for _ax in _axes[:, 0]:
        _ax.set_ylabel("y")
    _axes[0, 0].set(xlim=(3, 20), ylim=(2, 14))
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Ask each data set what is wrong

    Pick a set. Left: the data, the fit, and the point that **steers the fit most** (ringed).
    Right: the **pulls**, the distance of each point to the fit in units of its uncertainty
    (here 1). Randomness gives pulls that scatter around 0 without a pattern. A pattern, or one
    point far from the rest, is a problem the χ² did not report.

    Then switch on *Leave out the ringed point* and watch what the fit does.
    """)
    return


@app.cell
def _(mo, names):
    chosen = mo.ui.dropdown(
        {f"Set {name}": i for i, name in enumerate(names)}, value="Set I", label="Data set"
    )
    leave_out = mo.ui.switch(label="Leave out the ringed point and refit")
    mo.hstack([chosen, leave_out], justify="start", gap=2)
    return chosen, leave_out


@app.cell
def _(
    DATA,
    FIT,
    FLAG,
    chi2_pvalue,
    chosen,
    fit_line,
    leave_one_out_slope_change,
    leave_out,
    mo,
    names,
    np,
    plt,
    x,
    y,
):
    _x, _y = x[chosen.value], y[chosen.value]
    _change = leave_one_out_slope_change(_x, _y)
    _influential = int(np.argmax(np.where(np.isnan(_change), np.inf, np.abs(_change))))
    _keep = np.arange(len(_x)) != _influential if leave_out.value else np.ones(len(_x), dtype=bool)
    _fittable = np.ptp(_x[_keep]) > 0

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [3, 2]})
    _line_x = np.array([3.0, 20.0])
    _slope_all, _intercept_all, _ = fit_line(_x, _y)

    if leave_out.value:
        _ax1.plot(
            _line_x,
            _intercept_all + _slope_all * _line_x,
            color="0.6",
            ls="--",
            label="Fit with every point",
        )
    _ax1.plot(_x[_keep], _y[_keep], "o", color=DATA, ms=8, label="Data")
    if leave_out.value:
        _ax1.plot(
            _x[_influential], _y[_influential], "o", mfc="none", mec=DATA, ms=8, label="Left out"
        )
    _ax1.plot(
        _x[_influential],
        _y[_influential],
        "o",
        mfc="none",
        mec=FLAG,
        mew=2.5,
        ms=22,
        label="Steers the fit most",
    )

    if _fittable:
        _slope, _intercept, _ = fit_line(_x[_keep], _y[_keep])
        _ax1.plot(_line_x, _intercept + _slope * _line_x, color=FIT, lw=2, label="Fitted line")
        _pulls = _y[_keep] - (_intercept + _slope * _x[_keep])
        _colours = np.where(np.abs(_pulls) > 2, FLAG, DATA)
        _ax2.bar(np.flatnonzero(_keep) + 1, _pulls, color=_colours)
        _ax2.axhline(0, color="k", lw=1)
        _ax2.axhspan(-2, 2, color="0.9", zorder=0)
        _chi2 = float(np.sum(_pulls**2))
        _ndof = int(_keep.sum()) - 2
        _summary = (
            f"χ² = **{_chi2:.1f}** for {_ndof} degrees of freedom, p-value **{chi2_pvalue(_chi2, _ndof):.2f}**. "
            f"Fitted slope **{_slope:.3f}** (with every point: {_slope_all:.3f})."
        )
        _kind = "info"
    else:
        _ax2.text(0.5, 0.5, "No fit, no pulls", ha="center", va="center", transform=_ax2.transAxes)
        _summary = (
            f"Without the ringed point every remaining x is {_x[_keep][0]:.0f}, so **no line can be "
            "fitted**. The entire slope in this data set rested on that single point."
        )
        _kind = "danger"

    _ax1.set(xlim=(3, 20), ylim=(2, 14), xlabel="x", ylabel="y", title=f"Set {names[chosen.value]}")
    _ax1.legend(loc="lower right")
    _ax2.set(
        xlabel="Point number",
        ylabel="Pull = (y − fit) / σ",
        ylim=(-4, 4),
        xticks=range(1, len(_x) + 1),
        title="Pulls (grey band: within ±2)",
    )
    _fig.tight_layout()
    mo.vstack([_fig, mo.callout(mo.md(_summary), kind=_kind)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. A number that notices

    The previous plot needs your eyes. Here is a statistic that does not: **leave one point out at a
    time and refit**. If the slope hardly moves, no single point matters. If it jumps, one point is
    steering the fit. A hatched bar means that leaving the point out destroys the fit altogether.
    """)
    return


@app.cell
def _(DATA, FLAG, leave_one_out_slope_change, names, np, plt, x, y):
    _changes = [np.abs(leave_one_out_slope_change(x[_i], y[_i])) for _i in range(len(names))]
    _top = 1.25 * np.nanmax(np.concatenate(_changes))
    _fig, _axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    for _ax, _name, _c in zip(_axes, names, _changes, strict=True):
        _points = np.arange(1, len(_c) + 1)
        _ax.bar(_points, np.nan_to_num(_c), color=DATA)
        _undefined = np.isnan(_c)
        _ax.bar(_points[_undefined], _top, facecolor="none", edgecolor=FLAG, hatch="///", lw=1.5)
        _ax.set(title=f"Set {_name}", xlabel="Point left out", xticks=[1, 4, 7, 11])
    _axes[0].set(ylabel="|change of slope|", ylim=(0, _top))
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to take away

    1. Summary numbers, a fit and even the $\chi^2$ with its p-value can be **identical** for data
       that have nothing in common. Always plot the data.
    2. The **pulls** show what a single number hides: set II is a curve, so a line is the wrong
       function, and the pulls form a pattern instead of scattering.
    3. A single point can decide the result. Leaving one point out at a time finds it, and in set IV
       leaving it out removes the fit altogether.
    4. Looking is not optional, but it can be backed by numbers: pulls for the shape of the model,
       leave-one-out for influence.
    """)
    return


if __name__ == "__main__":
    app.run()
