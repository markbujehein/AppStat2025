import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from appstat.fitting import fit_binned_likelihood, fit_least_squares, fit_unbinned_likelihood
    from appstat.stats import (
        delta_interval,
        exponential_bin_counts,
        exponential_nll2,
        pearson_chi2,
        poisson_nll2,
    )

    CHI2, BINNED, UNBINNED = "#D55E00", "#0072B2", "#009E73"
    T_MAX = 10.0

    def make_controls(n_times=50, tau=1.0, n_bins=50):
        """Fresh sliders for the data, for a block that needs its own."""
        return mo.ui.dictionary(
            {
                "n_times": mo.ui.slider(
                    5, 2000, 5, value=n_times, show_value=True, label="Measured times"
                ),
                "tau": mo.ui.slider(
                    0.5, 2.0, 0.1, value=tau, show_value=True, label="True lifetime τ"
                ),
                "n_bins": mo.ui.slider(5, 100, 5, value=n_bins, show_value=True, label="Bins"),
                "seed": mo.ui.number(0, 9999, 1, value=42, label="Random seed"),
            }
        )

    def controls_panel(controls):
        """The four controls of ``make_controls`` in two columns."""
        return mo.hstack(
            [
                mo.vstack([controls["n_times"], controls["tau"]]),
                mo.vstack([controls["n_bins"], controls["seed"]]),
            ],
            widths="equal",
        )

    def draw_times(controls):
        """Times from an exponential with the chosen lifetime, and their histogram."""
        times = np.random.default_rng(controls["seed"].value).exponential(
            controls["tau"].value, controls["n_times"].value
        )
        counts, edges = np.histogram(times, bins=controls["n_bins"].value, range=(0, T_MAX))
        return times, counts, edges

    return (
        BINNED,
        CHI2,
        T_MAX,
        UNBINNED,
        controls_panel,
        delta_interval,
        draw_times,
        exponential_bin_counts,
        exponential_nll2,
        fit_binned_likelihood,
        fit_least_squares,
        fit_unbinned_likelihood,
        make_controls,
        mo,
        np,
        pearson_chi2,
        plt,
        poisson_nll2,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The principle of maximum likelihood

    50 exponentially distributed times, and one question: **what is the lifetime $\tau$?** There is
    only one parameter, so you can see what a fit does behind the scenes: try many values of $\tau$,
    compute how badly each one fits, and take the best. Three ways to say "how badly":

    1. **$\chi^2$** of a histogram.
    2. **Binned likelihood**: $-2\ln L$ of Poisson counts in a histogram.
    3. **Unbinned likelihood**: $-2\ln L$ of the individual times.

    The one standard deviation uncertainty is where the curve has risen by **1** above its minimum.
    The exercise ends by letting Minuit do the same job.

    Every interactive block below has its controls directly above its plots.

    References: Barlow chapter 5, Cowan chapter 6.

    ---

    **Adapted from** the Likelihood Fit exercise by **Troels C. Petersen** (Niels Bohr Institute,
    29th of November 2024), from the course *Applied Statistics 2025/26*. The original is an exercise with
    an example solution. This version keeps its three scans and its Minuit comparison, and is
    rewritten as an interactive demonstration.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Scan the lifetime

    The dot is the lowest value, and the dashed lines mark where each curve has risen by 1. For the
    unbinned likelihood the minimum is the **mean of the times**, which you can compute directly. Reduce the
    number of times and see how the three differ.
    """)
    return


@app.cell
def _(make_controls):
    scan_controls = make_controls()
    return (scan_controls,)


@app.cell
def _(
    BINNED,
    CHI2,
    UNBINNED,
    controls_panel,
    delta_interval,
    draw_times,
    exponential_bin_counts,
    exponential_nll2,
    mo,
    np,
    pearson_chi2,
    plt,
    poisson_nll2,
    scan_controls,
):
    _times, _counts, _edges = draw_times(scan_controls)
    _taus = np.geomspace(0.3, 3.0, 400) * scan_controls["tau"].value
    _expected = exponential_bin_counts(_edges, len(_times), _taus)
    _scans = {
        "χ²": ([pearson_chi2(_counts, row)[0] for row in _expected], CHI2),
        "Binned likelihood, −2 ln L": (poisson_nll2(_counts, _expected), BINNED),
        "Unbinned likelihood, −2 ln L": (exponential_nll2(_times, _taus), UNBINNED),
    }

    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.8))
    _lines = []
    for _ax, (_name, (_values, _colour)) in zip(_axes, _scans.items(), strict=True):
        _ax.plot(_taus, _values, color=_colour, lw=2)
        try:
            _best, _low, _high = delta_interval(_taus, _values)
            _ax.plot(_best, min(_values), "o", color="k")
            _ax.axvline(_low, color="k", ls="--", lw=1)
            _ax.axvline(_high, color="k", ls="--", lw=1)
            _ax.set_xlim(_best - 4 * (_best - _low), _best + 4 * (_high - _best))
            _ax.set_ylim(min(_values) - 1, min(_values) + 12)
            _lines.append(
                f"- **{_name}**: τ = {_best:.3f} +{_high - _best:.3f} −{_best - _low:.3f}"
            )
        except ValueError:
            _lines.append(f"- **{_name}**: the curve does not rise by 1 inside the scan range")
        _ax.set(title=_name, xlabel="Lifetime τ")
    _fig.tight_layout()

    _mean = _times.mean()
    _lines.append(
        f"- **Analytic**: τ = mean of the times = {_mean:.3f} ± {_mean / np.sqrt(len(_times)):.3f}"
    )
    mo.vstack(
        [controls_panel(scan_controls), _fig, mo.callout(mo.md("\n".join(_lines)), kind="info")]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Let Minuit do it

    In practice nobody scans. Minuit finds the minimum and the uncertainty for you, here with the
    same three costs. Compare its numbers with the scans above, which should agree.
    """)
    return


@app.cell
def _(make_controls):
    fit_controls = make_controls()
    return (fit_controls,)


@app.cell
def _(
    BINNED,
    CHI2,
    T_MAX,
    UNBINNED,
    controls_panel,
    draw_times,
    fit_binned_likelihood,
    fit_controls,
    fit_least_squares,
    fit_unbinned_likelihood,
    mo,
    np,
    plt,
):
    _times, _counts, _edges = draw_times(fit_controls)
    _centres, _width, _n = (_edges[1:] + _edges[:-1]) / 2, _edges[1] - _edges[0], len(_times)
    _used = _counts > 0

    def _per_bin(x, N, tau):
        return _width * N / tau * np.exp(-x / tau)

    def _density(x, N, tau):
        return N / tau * np.exp(-x / tau)

    def _with_total(x, N, tau):
        return N, _density(x, N, tau)

    _start = {"N": float(_n), "tau": fit_controls["tau"].value}
    _fits = {
        "χ²": (
            fit_least_squares(
                _per_bin, _centres[_used], _counts[_used], np.sqrt(_counts[_used]), _start
            ),
            CHI2,
        ),
        "Binned likelihood": (fit_binned_likelihood(_density, _counts, _edges, _start), BINNED),
        "Unbinned likelihood": (fit_unbinned_likelihood(_with_total, _times, _start), UNBINNED),
    }

    _fig, _ax = plt.subplots(figsize=(13, 3.8))
    _ax.errorbar(
        _centres[_used],
        _counts[_used],
        np.sqrt(_counts[_used]),
        fmt=".k",
        capsize=2,
        label="Data, Poisson errors",
    )
    _grid = np.linspace(0, T_MAX, 500)
    _lines = []
    for _name, (_fit, _colour) in _fits.items():
        _ax.plot(_grid, _per_bin(_grid, *_fit.values), color=_colour, lw=2, label=_name)
        _lines.append(f"- **{_name}**: τ = {_fit.values['tau']:.3f} ± {_fit.errors['tau']:.3f}")
    _ax.set(xlim=(0, T_MAX / 2), ylim=(0, None), xlabel="Time", ylabel=f"Entries per {_width:g}")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack(
        [controls_panel(fit_controls), _fig, mo.callout(mo.md("\n".join(_lines)), kind="info")]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A. Is the likelihood curve a parabola?

    The uncertainty $\pm\Delta\tau$ assumes the curve is a symmetric parabola (the dotted one, with the
    curvature of a Gaussian). The real curve is not. With **few times** it is lopsided, and the interval is
    asymmetric: the lifetime can be much larger than the best value, but not much smaller. Lower the number of times.
    """)
    return


@app.cell
def _(mo):
    parabola_n = mo.ui.slider(3, 300, 1, value=8, show_value=True, label="Measured times")
    parabola_seed = mo.ui.number(0, 9999, 1, value=5, label="Random seed")
    return parabola_n, parabola_seed


@app.cell
def _(UNBINNED, delta_interval, exponential_nll2, mo, np, parabola_n, parabola_seed, plt):
    _times = np.random.default_rng(parabola_seed.value).exponential(1.0, parabola_n.value)
    _mean = _times.mean()
    _taus = np.linspace(0.2, 5.0, 2000) * _mean
    _curve = exponential_nll2(_times, _taus)
    _best, _low, _high = delta_interval(_taus, _curve)
    _symmetric = _mean / np.sqrt(len(_times))
    _parabola = _curve.min() + ((_taus - _best) / _symmetric) ** 2

    _fig, _ax = plt.subplots(figsize=(13, 3.6))
    _ax.plot(_taus, _curve, color=UNBINNED, lw=2, label="Unbinned −2 ln L")
    _ax.plot(_taus, _parabola, "k:", lw=2, label="Symmetric parabola")
    _ax.axvspan(_low, _high, color=UNBINNED, alpha=0.15, label="Rise by 1")
    _ax.set(
        xlim=(0, 3 * _mean),
        ylim=(_curve.min() - 1, _curve.min() + 10),
        xlabel="Lifetime τ",
        ylabel="−2 ln L",
    )
    _ax.legend()
    _fig.tight_layout()
    mo.vstack(
        [
            mo.hstack([parabola_n, parabola_seed], justify="start", gap=2),
            _fig,
            mo.callout(
                mo.md(
                    f"τ = {_best:.2f} **+{_high - _best:.2f} −{_best - _low:.2f}** from the curve, "
                    f"against the symmetric ± {_symmetric:.2f} from the parabola."
                ),
                kind="warn"
                if abs((_high - _best) - (_best - _low)) > 0.2 * _symmetric
                else "success",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to take away

    1. A fit is a **scan** of the parameter that takes the value where the cost is lowest. Minuit just does it faster.
    2. The one standard deviation interval is where the cost has **risen by 1** above its minimum.
    3. The unbinned likelihood uses every value and loses nothing to binning, and for an exponential
       its minimum is simply the mean of the times.
    4. The curve is only a parabola for much data. With little data the uncertainty is **asymmetric**.
    """)
    return


if __name__ == "__main__":
    app.run()
