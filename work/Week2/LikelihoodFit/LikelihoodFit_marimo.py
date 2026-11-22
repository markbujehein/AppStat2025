import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from appstat.fitting import fit_binned_likelihood, fit_least_squares, fit_unbinned_likelihood
    from appstat.models import exponential_bin_counts
    from appstat.stats import (
        delta_interval,
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
def _(
    UNBINNED,
    delta_interval,
    exponential_nll2,
    mo,
    np,
    parabola_n,
    parabola_seed,
    plt,
):
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


@app.cell
def _():
    from iminuit import Minuit, cost
    from scipy.special import xlogy

    from appstat.stats import chi2_pvalue

    return Minuit, chi2_pvalue, cost, xlogy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## B. Which cost function?

    iminuit has one cost function per kind of data. Two questions pick it.

    **1. What kind of data do you have?**

    - **Individual values**: one number per event, nothing thrown away. *E.g.* 50 measured decay times,
      `0.8, 2.3, 0.1, 1.4, …`
    - **Histogrammed data**: the values sorted into bins, only the count per bin is kept. *E.g.* the same
      times as counts in bins of width 1: `21, 12, 8, 4, 3, …`
    - **x, y, error points**: a value $y$ measured at chosen $x$, each with its uncertainty. *E.g.* a
      temperature of $20.3 \pm 0.2$ °C at 1 s and $21.1 \pm 0.2$ °C at 2 s.

    **2. Is the total count itself a measurement?** If yes, use the *extended* cost, which also fits the
    yield. If only the shape matters, use the plain one.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(
        """
        flowchart TD
            D{"What is the data?"}
            D -->|"individual values"| UQ{"Is the total count<br/>a measurement?"}
            UQ -->|"no, shape only"| UN["UnbinnedNLL"]
            UQ -->|"yes"| EU["ExtendedUnbinnedNLL"]
            D -->|"histogram"| HQ{"Is the model a histogram<br/>from simulation?"}
            HQ -->|"yes"| TP["Template"]
            HQ -->|"no"| BQ{"Is the total count<br/>a measurement?"}
            BQ -->|"no, shape only"| BN["BinnedNLL"]
            BQ -->|"yes"| EB["ExtendedBinnedNLL"]
            D -->|"x, y, error points"| LQ{"Outliers in y?"}
            LQ -->|"no"| LS["LeastSquares"]
            LQ -->|"yes"| LS2["LeastSquares with<br/>loss = soft_l1"]
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If values can be kept unbinned, keep them: no information is thrown away (section D). A histogram
    with few entries per bin needs a **Poisson** cost (`BinnedNLL`, `ExtendedBinnedNLL`), not
    `LeastSquares`. Pick the tab for your data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _head = "| Cost | Call | Use it when |\n|---|---|---|\n"
    mo.ui.tabs(
        {
            "Individual values": mo.md(
                _head
                + "| `UnbinnedNLL` | `UnbinnedNLL(data, pdf)` | only the shape is fitted. `pdf` is normalised |\n"
                + "| `ExtendedUnbinnedNLL` | `ExtendedUnbinnedNLL(data, scaled_pdf)` | the yield is fitted too. "
                + "The model returns the expected total and the density |"
            ),
            "Histogram": mo.md(
                _head
                + "| `BinnedNLL` | `BinnedNLL(n, xe, cdf)` | only the shape is fitted. `cdf` is normalised. "
                + "Takes weighted histograms |\n"
                + "| `ExtendedBinnedNLL` | `ExtendedBinnedNLL(n, xe, scaled_cdf)` | the yield is fitted too |\n"
                + "| `Template` | `Template(n, xe, templates)` | the model is a simulated histogram, "
                + "so its own finite statistics matter |"
            ),
            "x, y, error points": mo.md(
                _head
                + "| `LeastSquares` | `LeastSquares(x, y, yerror, model)` | Gaussian errors. "
                + '`loss="soft_l1"` when $y$ has outliers |'
            ),
            "Combine and constrain": mo.md(
                _head
                + "| `CostSum` | `cost_a + cost_b` | several datasets in one fit. "
                + "Parameters with the same name are shared |\n"
                + "| `NormalConstraint` | `NormalConstraint(names, values, errors)` | "
                + "a Gaussian prior from an external measurement |\n"
                + "| `Constant` | `Constant(value)` | adds a number to the cost. Helps numerics only, "
                + "the fit result is unchanged |\n"
                + "| `Cost` | subclass it | a cost iminuit does not have |"
            ),
            "Goodness of fit": mo.md(
                "Functions, not fitted. Give them the data and the fitted expectation.\n\n"
                "| Function | Use it when |\n|---|---|\n"
                "| `chi2(y, ye, ym)` | Gaussian $\\chi^2$ of points. This is what `LeastSquares` sums |\n"
                "| `poisson_chi2(n, mu)` | Poisson likelihood-ratio $\\chi^2$ (Baker and Cousins). "
                "Valid at low counts |\n"
                "| `multinomial_chi2(n, mu)` | the same with the total fixed, needs `sum(mu) == sum(n)`. "
                "`BinnedNLL` is built on it |\n"
                "| `template_chi2_jsc`, `template_chi2_da`, `template_nll_asy` | "
                "the variants for the template uncertainty |"
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Do the costs agree?

    The same simulated times fitted five ways. The **pull** is $(\tau - \tau_\text{true})/\sigma$: how many
    uncertainties a fit is from the truth. A cost that works has a **mean pull of 0** (no bias) and a
    **spread of 1** (the quoted $\sigma$ is the real scatter). The last two columns come from 100 repeated
    experiments, so the mean is known to about $\pm 0.1$ and the spread to about $\pm 0.07$. Lower the number
    of times to 15 and see which costs leave 0 and 1. $\chi^2$ and its p-value exist only for the binned costs.
    """)
    return


@app.cell
def _(make_controls):
    cost_controls = make_controls(n_times=100, n_bins=20)
    return (cost_controls,)


@app.cell(hide_code=True)
def _(
    BINNED,
    CHI2,
    T_MAX,
    UNBINNED,
    Minuit,
    chi2_pvalue,
    controls_panel,
    cost,
    cost_controls,
    draw_times,
    mo,
    np,
    plt,
):
    _tau0, _n_times = cost_controls["tau"].value, cost_controls["n_times"].value
    _edges = np.linspace(0, T_MAX, cost_controls["n_bins"].value + 1)
    _centres, _width = (_edges[1:] + _edges[:-1]) / 2, _edges[1] - _edges[0]

    def _pdf(x, tau):
        return np.exp(-x / tau) / tau / (1 - np.exp(-T_MAX / tau))

    def _cdf(x, tau):
        return (1 - np.exp(-x / tau)) / (1 - np.exp(-T_MAX / tau))

    def _scaled_cdf(x, n, tau):
        return n * _cdf(x, tau)

    def _scaled_pdf(x, n, tau):
        return n, n * _pdf(x, tau)

    def _per_bin(x, n, tau):
        return _width * n * _pdf(x, tau)

    # call shown in the table: (colour, has a chi2 goodness of fit)
    _calls = {
        "LeastSquares(x, y, yerror, model)": (CHI2, True),
        "BinnedNLL(n, xe, cdf)": (BINNED, True),
        "ExtendedBinnedNLL(n, xe, scaled_cdf)": (BINNED, True),
        "UnbinnedNLL(data, pdf)": (UNBINNED, False),
        "ExtendedUnbinnedNLL(data, scaled_pdf)": (UNBINNED, False),
    }

    def _fit_all(times):
        """Fit one set of times with the five costs. A failed fit gives None."""
        times = times[times < T_MAX]  # the models are normalised on [0, T_MAX]
        counts = np.histogram(times, bins=_edges)[0]
        used = counts > 0
        n = len(times)
        makers = [
            (
                lambda: cost.LeastSquares(
                    _centres[used], counts[used], np.sqrt(counts[used]), _per_bin
                ),
                {"n": n, "tau": _tau0},
            ),
            (lambda: cost.BinnedNLL(counts, _edges, _cdf), {"tau": _tau0}),
            (
                lambda: cost.ExtendedBinnedNLL(counts, _edges, _scaled_cdf),
                {"n": n, "tau": _tau0},
            ),
            (lambda: cost.UnbinnedNLL(times, _pdf), {"tau": _tau0}),
            (
                lambda: cost.ExtendedUnbinnedNLL(times, _scaled_pdf),
                {"n": n, "tau": _tau0},
            ),
        ]
        fits = {}
        for call, (make, start) in zip(_calls, makers, strict=True):
            try:
                m = Minuit(make(), **start)
                m.limits["tau"] = (0.05, None)
                m.migrad()
                fits[call] = m if m.valid else None
            except ValueError, RuntimeError:
                fits[call] = None
        return fits

    _times, _counts, _ = draw_times(cost_controls)
    _shown = _fit_all(_times)

    _repeats = np.random.default_rng(12345).exponential(_tau0, (100, _n_times))
    _pulls = {call: [] for call in _calls}
    for _row in _repeats:
        for _call, _m in _fit_all(_row).items():
            if _m is not None:
                _pulls[_call].append((_m.values["tau"] - _tau0) / _m.errors["tau"])

    _rows = [
        "| Cost | τ ± σ, this data set | χ² / ndof | p-value | Mean pull | Spread of pull |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    _used = _counts > 0
    _grid = np.linspace(0, T_MAX, 400)
    _fig, _ax = plt.subplots(figsize=(13, 3.6))
    _ax.errorbar(
        _centres[_used], _counts[_used], np.sqrt(_counts[_used]), fmt=".k", capsize=2, label="Data"
    )
    # one line style per cost, because the two binned and the two unbinned share a colour
    _styles = dict(zip(_calls, ("-", "-", "--", "-", "--"), strict=True))
    for _call, (_colour, _binned) in _calls.items():
        _m = _shown[_call]
        _stats = (
            f"{np.mean(_pulls[_call]):+.2f} | {np.std(_pulls[_call]):.2f}"
            if len(_pulls[_call]) > 1
            else "– | –"
        )
        if _m is None:
            _rows.append(f"| `{_call}` | did not converge | | | {_stats} |")
            continue
        _gof, _p = "–", "–"
        if _binned and _m.ndof > 0:
            _gof, _p = f"{_m.fval:.1f} / {_m.ndof:g}", f"{chi2_pvalue(_m.fval, int(_m.ndof)):.2f}"
        _rows.append(
            f"| `{_call}` | {_m.values['tau']:.3f} ± {_m.errors['tau']:.3f} | {_gof} | {_p} | {_stats} |"
        )
        _tau = _m.values["tau"]
        _ax.plot(
            _grid,
            _width * len(_times) * _pdf(_grid, _tau),
            _styles[_call],
            color=_colour,
            lw=2,
            label=_call.split("(")[0],
        )
    _ax.set(xlim=(0, T_MAX / 2), ylim=(0, None), xlabel="Time", ylabel=f"Entries per {_width:g}")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([controls_panel(cost_controls), _fig, mo.md("\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## C. Asymmetric uncertainties

    ### Detect them

    $\pm\sigma$ from `hesse()` is the curvature at the minimum, a parabola. `minos()` follows the real
    likelihood to where it has risen by 1, on each side separately. **If the two sides differ, the curve is
    not a parabola and the symmetric $\pm$ is wrong.** Warning signs:

    - Minos lower and upper differ **at the precision you quote**. A result is quoted to the digits its
      uncertainty justifies (the PDG rule: two significant digits of the uncertainty if its leading digits
      are 100 to 354, one if 355 to 949). A difference smaller than that last digit is not part of the
      result, a larger one is. So how much asymmetry matters follows from how precise the measurement is.
    - The profile bends away from the dotted parabola before it reaches 1.
    - **Few data**, because the curve only becomes a parabola for many.
    - The parameter is **positive** (lifetime, rate, width) and its error is a large fraction of its value.
    - The parameter is near a **physical boundary**.
    """)
    return


@app.cell
def _(mo):
    asym_n = mo.ui.slider(3, 300, 1, value=8, show_value=True, label="Measured times")
    asym_seed = mo.ui.number(0, 9999, 1, value=5, label="Random seed")
    return asym_n, asym_seed


@app.cell
def _(Minuit, UNBINNED, asym_n, asym_seed, cost, mo, np, plt):
    _times = np.random.default_rng(asym_seed.value).exponential(1.0, asym_n.value)

    def _pdf(x, tau):
        return np.exp(-x / tau) / tau

    _m = Minuit(cost.UnbinnedNLL(_times, _pdf), tau=_times.mean())
    _m.limits["tau"] = (1e-3, None)
    _m.migrad()
    _m.hesse()
    _m.minos("tau")
    _best, _sym = _m.values["tau"], _m.errors["tau"]
    _merror = _m.merrors["tau"]
    mo.stop(
        not _merror.is_valid,
        mo.vstack(
            [
                mo.hstack([asym_n, asym_seed], justify="start", gap=2),
                mo.callout("`minos` did not find the interval. Try more times.", kind="danger"),
            ]
        ),
    )
    _low, _high = -_merror.lower, _merror.upper  # iminuit stores the lower error as negative

    _x, _y, _ = _m.mnprofile("tau", size=80, bound=(0.15 * _best, 4 * _best))
    _rise = (_y - _y.min()) / _m.errordef  # 1 = one standard deviation, whatever the cost's scale

    _fig, _ax = plt.subplots(figsize=(13, 3.6))
    _ax.plot(_x, _rise, color=UNBINNED, lw=2, label="Profile of the likelihood")
    _ax.plot(_x, ((_x - _best) / _sym) ** 2, "k:", lw=2, label="Parabola from hesse")
    _ax.axhline(1, color="k", lw=0.8)
    _ax.axvspan(_best - _low, _best + _high, color=UNBINNED, alpha=0.15, label="minos interval")
    _ax.set(ylim=(-0.3, 6), xlabel="Lifetime τ", ylabel="Rise above minimum")
    _ax.legend()
    _fig.tight_layout()

    # Round to the precision the smaller side is worth (PDG rule), then compare the two sides.
    _exponent = int(np.floor(np.log10(min(_low, _high))))
    _leading = int(min(_low, _high) / 10 ** (_exponent - 2))  # three leading digits, 100 to 999
    if _leading >= 950:  # rounds up to the next power of ten
        _exponent += 1
    _digits = 1 if 355 <= _leading <= 949 else 2
    _places = max(_digits - 1 - _exponent, 0)
    _low_q, _high_q = round(_low, _places), round(_high, _places)
    _asymmetric = _low_q != _high_q
    _tau_q = f"{_best:.{_places}f}"
    _quote = (
        f"τ = {_tau_q} +{_high_q:.{_places}f} −{_low_q:.{_places}f}"
        if _asymmetric
        else f"τ = {_tau_q} ± {_low_q:.{_places}f}"
    )
    mo.vstack(
        [
            mo.hstack([asym_n, asym_seed], justify="start", gap=2),
            _fig,
            mo.md(
                "| | τ | lower | upper |\n|---|---:|---:|---:|\n"
                f"| `hesse` | {_best:.3f} | {_sym:.3f} | {_sym:.3f} |\n"
                f"| `minos` | {_best:.3f} | {_low:.3f} | {_high:.3f} |"
            ),
            mo.callout(
                mo.md(
                    f"Quoted to the precision this uncertainty is worth ({_places} decimals): **{_quote}**. "
                    + (
                        "The sides differ at that precision, so both belong in the result."
                        if _asymmetric
                        else "The sides are equal at that precision, so one ± is the same result."
                    )
                ),
                kind="warn" if _asymmetric else "success",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Remove them, if you can

    1. **More data.** The curve approaches a parabola as $1/\sqrt{N}$. This is the only fix that makes the
       result itself better.
    2. **Fit a better variable.** The likelihood interval does not depend on the parametrisation, but the
       parabola does. Fit $\ln\tau$ or the rate $1/\tau$ if one of them is closer to symmetric (table below).
       Then the $\pm$ is trustworthy **in that variable**. Transform back and the interval becomes asymmetric
       again, so quote the variable you fitted, or the transformed interval ends.
    3. **Do not average** the two sides into one $\pm$. The table shows how often the symmetric
       $\pm$ of `hesse` covers the truth.

    True lifetime $\tau = 1$, each row is 50 000 simulated experiments. A 68.3 % interval should cover the
    truth 68.3 % of the time. The last three columns give (upper side) / (lower side) of the interval in each
    variable: 1 is symmetric. These numbers are for an exponential lifetime. Another model is lopsided in a
    different way.
    """)
    return


@app.cell
def _(delta_interval, mo, np):
    _rng = np.random.default_rng(0)
    _taus = np.geomspace(0.05, 20, 20000)
    _rows = [
        "| Times N | ± hesse covers | likelihood rise < 1 covers | τ | ln τ | 1/τ |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for _n in (3, 5, 10, 30, 100, 300):
        _mean = _rng.gamma(_n, 1 / _n, 50_000)  # the mean of N exponential times with τ = 1
        _hesse = np.mean(np.abs(_mean - 1) < _mean / np.sqrt(_n))
        _like = np.mean(2 * _n * (_mean - 1 - np.log(_mean)) < 1)
        # a typical experiment has mean 1, so its curve depends on N alone
        _, _low, _high = delta_interval(_taus, 2 * _n * (1 / _taus + np.log(_taus)))
        _rows.append(
            f"| {_n} | {_hesse:.1%} | {_like:.1%} | {(_high - 1) / (1 - _low):.2f} "
            f"| {np.log(_high) / -np.log(_low):.2f} | {(1 / _low - 1) / (1 - 1 / _high):.2f} |"
        )
    mo.md("\n".join(_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Report them, if data is limited

    - Quote **both sides**: $\tau = 1.8\,^{+1.2}_{-0.6}$. State that it is the $-2\ln L$ rise by 1 (68 %).
    - **Say how it was made** (`minos`, profile likelihood), not only the numbers.
    - Better, **show the likelihood curve** as in the plot above, or give a 95 % interval or an
      **upper or lower limit** when the curve does not close on one side or the parameter sits at a boundary.
    - Keep the full curve when the result will be combined with others. Two asymmetric $\pm$ cannot be
      combined by averaging.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## D. The risks of binning

    A histogram is a summary, and a summary can lose or distort things. Three ways it goes wrong, each
    shown with simulated experiments. Only the lifetime is fitted, the total count is fixed.

    ### D1. Too few bins lose information, and $\chi^2$ with few entries is biased

    200 experiments of $N$ times, fitted by a Poisson binned likelihood, by a $\chi^2$ that uses
    $\sqrt{n}$ of each bin (empty bins dropped, as in the fit above), and by the unbinned mean. Plotted are the
    **spread** of the 200 results and their **mean minus the true value** ($\tau = 1$).
    """)
    return


@app.cell
def _(mo):
    risk_n = mo.ui.slider(
        20, 500, 10, value=50, show_value=True, label="Measured times per experiment"
    )
    return (risk_n,)


@app.cell
def _(BINNED, CHI2, T_MAX, UNBINNED, mo, np, plt, risk_n, xlogy):
    _rng = np.random.default_rng(1)
    _n, _toys = risk_n.value, 200
    _taus = np.geomspace(0.25, 4, 700)
    _times = _rng.exponential(1.0, (_toys, _n))
    _bin_counts = (3, 5, 10, 20, 50, 100)
    _poisson, _neyman = [], []
    for _bins in _bin_counts:
        _edges = np.linspace(0, T_MAX, _bins + 1)
        _expected = _n * (
            np.exp(-_edges[:-1] / _taus[:, None]) - np.exp(-_edges[1:] / _taus[:, None])
        )  # shape (taus, bins)
        _p, _c = [], []
        for _t in _times:
            _counts = np.histogram(_t, bins=_edges)[0]
            _p.append(_taus[np.argmin((_expected - xlogy(_counts, _expected)).sum(axis=1))])
            _used = _counts > 0
            _terms = (_counts - _expected) ** 2 / np.where(_used, _counts, 1)
            _c.append(_taus[np.argmin(np.where(_used, _terms, 0).sum(axis=1))])
        _poisson.append(_p)
        _neyman.append(_c)
    _unbinned = _times.mean(axis=1)

    _fig, (_ax_sd, _ax_bias) = plt.subplots(1, 2, figsize=(13, 3.8))
    for _label, _results, _colour in (
        ("Poisson binned likelihood", _poisson, BINNED),
        ("χ² with √n errors", _neyman, CHI2),
    ):
        _ax_sd.plot(_bin_counts, np.std(_results, axis=1), "o-", color=_colour, label=_label)
        _ax_bias.errorbar(
            _bin_counts,
            np.mean(_results, axis=1) - 1,
            np.std(_results, axis=1) / np.sqrt(_toys),
            fmt="o-",
            color=_colour,
            capsize=3,
            label=_label,
        )
    _ax_sd.axhline(_unbinned.std(), color=UNBINNED, ls="--", label="Unbinned (the mean)")
    _ax_bias.axhline(_unbinned.mean() - 1, color=UNBINNED, ls="--", label="Unbinned (the mean)")
    _ax_bias.axhline(0, color="k", lw=0.8)
    for _ax, _title in ((_ax_sd, "Spread of the result"), (_ax_bias, "Mean minus true value")):
        _ax.set(xscale="log", xlabel="Number of bins", title=_title)
        _ax.legend()
    _fig.tight_layout()
    mo.vstack(
        [
            risk_n,
            _fig,
            mo.callout(
                mo.md(
                    "Error bars on the right are the uncertainty of the mean over 200 experiments. A point "
                    "further than two of them from zero (about 95 %) is a **bias**, not noise."
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### D2. The bin edges change the answer

    One fixed sample, a coarse histogram, and the edges slid sideways by up to one bin width. The data did not
    change, so a result that moves is an artefact of where the edges fell. The unbinned estimate cannot move.
    """)
    return


@app.cell
def _(make_controls):
    shift_controls = make_controls(n_times=50, n_bins=5)
    return (shift_controls,)


@app.cell
def _(
    BINNED,
    T_MAX,
    UNBINNED,
    controls_panel,
    draw_times,
    mo,
    np,
    plt,
    poisson_nll2,
    shift_controls,
):
    _times, _, _ = draw_times(shift_controls)
    _tau0, _bins = shift_controls["tau"].value, shift_controls["n_bins"].value
    _n, _width = len(_times), T_MAX / _bins
    _taus = np.geomspace(0.25, 4, 600) * _tau0
    _shifts = np.linspace(0, _width, 41)
    _estimates = []
    for _shift in _shifts:
        _edges = _shift + _width * np.arange(_bins + 1)
        _counts = np.histogram(_times, bins=_edges)[0]
        _expected = _n * (
            np.exp(-_edges[:-1] / _taus[:, None]) - np.exp(-_edges[1:] / _taus[:, None])
        )
        _estimates.append(_taus[np.argmin(poisson_nll2(_counts, _expected))])

    _fig, _ax = plt.subplots(figsize=(13, 3.4))
    _ax.plot(_shifts / _width, _estimates, color=BINNED, lw=2, label="Binned likelihood")
    _ax.axhline(_times.mean(), color=UNBINNED, ls="--", label="Unbinned (the mean)")
    _ax.set(xlabel="Edge shift in bin widths", ylabel="Fitted lifetime τ")
    _ax.legend()
    _fig.tight_layout()
    _spread = max(_estimates) - min(_estimates)
    mo.vstack(
        [
            controls_panel(shift_controls),
            _fig,
            mo.callout(
                mo.md(
                    f"Sliding the edges moves τ by up to **{_spread:.3f}**, against a statistical "
                    f"uncertainty of {_times.mean() / np.sqrt(_n):.3f}."
                ),
                kind="warn" if _spread > 0.2 * _times.mean() / np.sqrt(_n) else "info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### D3. Rules that follow

    - **Keep the data unbinned** when the fit allows it. Binning is a choice that only costs.
    - **Fix the binning before looking at the fit.** Tuning edges until the result looks right is a
      way to fool yourself, as the plot above shows.
    - **Bins must be fine enough** for the model to be flat inside one, or the fit uses the model at the
      centre where it should use its integral (`BinnedNLL` integrates the cdf, `LeastSquares` does not).
    - **Few entries per bin: use a Poisson cost**, never $\chi^2$ with $\sqrt{n}$. Do not merge bins
      afterwards just to make a $\chi^2$ behave.
    - **The goodness of fit depends on the binning.** The same data can give a different p-value for different binnings.
    - **In several dimensions the entries per bin fall fast** with the number of bins, so an unbinned fit
      matters even more.
    """)
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
    5. **Pick the cost from the data**: individual values, histogram or points, and whether the total count
       is a measurement (extended). Combine costs with `+`.
    6. **Detect asymmetry with `minos`** against `hesse`. Fix it with more data or a better variable, and
       otherwise report both sides, or the likelihood curve.
    7. **Binning costs information, can bias a $\chi^2$ fit, and makes the result depend on the edges.**
       Fit unbinned if you can.
    """)
    return


if __name__ == "__main__":
    app.run()
