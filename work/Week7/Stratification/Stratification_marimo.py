import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from appstat.montecarlo import (
        equal_counts,
        plain_integral,
        proportional_counts,
        stratified_integral,
        stratum_stds,
    )

    PLAIN, EQUAL, PROPORTIONAL = "#0072B2", "#D55E00", "#009E73"
    LOW, HIGH = 0.0, 10.0
    EXACT = np.log(HIGH**2 + 1) / 2  # integral of x / (x^2 + 1) from 0 to 10

    def func(x):
        return x / (x**2 + 1)

    def repeat(n_total, n_strata, n_pilot, n_repeats, rng):
        """The three estimates of the integral, repeated ``n_repeats`` times."""
        edges = np.linspace(LOW, HIGH, n_strata + 1)
        equal = equal_counts(n_strata, n_total)
        plain = [plain_integral(func, LOW, HIGH, n_total, rng) for _ in range(n_repeats)]
        by_equal = [stratified_integral(func, edges, equal, rng) for _ in range(n_repeats)]
        by_std = []
        for _ in range(n_repeats):  # the spread in each stratum comes from a pilot run
            counts = proportional_counts(stratum_stds(func, edges, n_pilot, rng), n_total)
            by_std.append(stratified_integral(func, edges, counts, rng))
        return np.array(plain), np.array(by_equal), np.array(by_std)

    return (
        EQUAL,
        EXACT,
        HIGH,
        LOW,
        PLAIN,
        PROPORTIONAL,
        func,
        mo,
        np,
        plt,
        proportional_counts,
        repeat,
        stratum_stds,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Stratification: beating the $1/\sqrt{N}$ law

    To estimate $\int_0^{10} \frac{x}{x^2+1}\,dx = \tfrac12\ln 101 = 2.3$ with random points, the plain way is
    the mean of the function at $N$ uniform points, times the width. The error falls as $1/\sqrt{N}$.

    **Stratify**: cut the range into strata, take the mean in each, and add them with their widths. The function is flat for
    large $x$ and varies a lot for small $x$, so points are better spent where it **varies**:

    1. **Equal** number of points in each stratum.
    2. A number **in proportion to the spread** of the function in each stratum, found with a small pilot run.

    Every interactive block below has its controls directly above its plots.

    ---

    **Adapted from** the Stratification exercise by **Troels C. Petersen** (Niels Bohr Institute), from the course
    *Applied Statistics 2025/26*, after Bevington pages 75-78. The original is an exercise with no example solution.
    This version is written from its text and checked against the exact integral. It is not an official solution.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The function and its strata

    The bars show how many points each method puts in each stratum. The proportional method spends most of its points
    on the first strata, where the function changes.
    """)
    return


@app.cell
def _(mo):
    look_strata = mo.ui.slider(2, 40, 1, value=10, show_value=True, label="Strata")
    look_total = mo.ui.slider(
        200, 20_000, 100, value=10_000, show_value=True, label="Points in all"
    )
    return look_strata, look_total


@app.cell
def _(
    EQUAL,
    HIGH,
    LOW,
    PROPORTIONAL,
    func,
    look_strata,
    look_total,
    mo,
    np,
    plt,
    proportional_counts,
    stratum_stds,
):
    _edges = np.linspace(LOW, HIGH, look_strata.value + 1)
    _counts = proportional_counts(
        stratum_stds(func, _edges, 1000, np.random.default_rng(1)), look_total.value
    )
    _x = np.linspace(LOW, HIGH, 400)
    _fig, (_ax, _axc) = plt.subplots(1, 2, figsize=(13, 3.6))
    _ax.plot(_x, func(_x), "k", lw=2)
    for _edge in _edges:
        _ax.axvline(_edge, color="r", alpha=0.2)
    _ax.set(xlabel="x", ylabel="x / (x² + 1)", ylim=(0, None))
    _centres = (_edges[1:] + _edges[:-1]) / 2
    _width = (HIGH - LOW) / look_strata.value
    _axc.bar(
        _centres - _width / 5,
        np.full(look_strata.value, look_total.value // look_strata.value),
        _width * 0.4,
        color=EQUAL,
        label="Equal",
    )
    _axc.bar(
        _centres + _width / 5,
        _counts,
        _width * 0.4,
        color=PROPORTIONAL,
        label="Proportional to spread",
    )
    _axc.set(xlabel="x", ylabel="Points in the stratum", yscale="log")
    _axc.legend()
    _fig.tight_layout()
    mo.vstack([mo.hstack([look_strata, look_total], justify="start", gap=2), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Repeat the experiment

    Repeat each method many times. All three scatter around the exact value, but the stratified ones **much less**. How many
    times more points would the plain method need to match them?
    """)
    return


@app.cell
def _(mo):
    rep_total = mo.ui.slider(
        200, 20_000, 100, value=10_000, show_value=True, label="Points per estimate"
    )
    rep_strata = mo.ui.slider(2, 40, 1, value=10, show_value=True, label="Strata")
    rep_repeats = mo.ui.slider(100, 1000, 100, value=300, show_value=True, label="Repeats")
    return rep_repeats, rep_strata, rep_total


@app.cell
def _(
    EQUAL,
    EXACT,
    PLAIN,
    PROPORTIONAL,
    mo,
    np,
    plt,
    rep_repeats,
    rep_strata,
    rep_total,
    repeat,
):
    _results = repeat(
        rep_total.value, rep_strata.value, 200, rep_repeats.value, np.random.default_rng(2)
    )
    _names = ["Plain", "Stratified, equal", "Stratified, proportional"]
    _colours = [PLAIN, EQUAL, PROPORTIONAL]
    _fig, _ax = plt.subplots(figsize=(13, 3.6))
    _edges = np.linspace(min(r.min() for r in _results), max(r.max() for r in _results), 60)
    for _values, _name, _colour in zip(_results, _names, _colours, strict=True):
        _ax.hist(_values, _edges, histtype="stepfilled", alpha=0.5, color=_colour, label=_name)
    _ax.axvline(EXACT, color="k", ls="--", label="Exact")
    _ax.set(xlabel="Estimate of the integral", ylabel="Repeats")
    _ax.legend()
    _fig.tight_layout()
    _stds = [r.std(ddof=1) for r in _results]
    _pilot = rep_strata.value * 200  # points of the pilot run, on top of the total
    _gain = (_stds[0] / _stds[2]) ** 2
    _cost = (rep_total.value + _pilot) / rep_total.value
    mo.vstack(
        [
            mo.hstack([rep_total, rep_strata, rep_repeats], justify="start", gap=2),
            _fig,
            mo.callout(
                mo.md(
                    "Width of the estimates: "
                    + ", ".join(f"{n} **{s:.5f}**" for n, s in zip(_names, _stds, strict=True))
                    + f". Plain needs about **{_gain:.0f} times** the points to match the proportional one. "
                    f"The pilot run adds {_pilot:,} points, so the proportional method really used "
                    f"{_cost:.1f} times the total: a net gain of **{_gain / _cost:.1f}**."
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

    1. Plain Monte Carlo spends points evenly, however little they tell you. Its precision is $1/\sqrt{N}$.
    2. **Stratification** gives the same points to regions in proportion to how much the function **varies** there, and
       the same total then gives a smaller error.
    3. The spread in each stratum has to be estimated, here by a small pilot run, which costs a few points.
    4. A variance reduction like this is worth as much as many times more simulated points.
    """)
    return


if __name__ == "__main__":
    app.run()
