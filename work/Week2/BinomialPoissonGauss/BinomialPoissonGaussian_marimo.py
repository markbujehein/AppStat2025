import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    from appstat.fitting import fit_least_squares
    from appstat.stats import (
        binomial_pmf,
        chi2_pvalue,
        integer_histogram,
        pearson_chi2,
        skewness_significance,
    )

    BINOMIAL, POISSON, GAUSSIAN = "#0072B2", "#D55E00", "#009E73"
    return (
        BINOMIAL,
        GAUSSIAN,
        POISSON,
        binomial_pmf,
        chi2_pvalue,
        fit_least_squares,
        integer_histogram,
        mo,
        np,
        pearson_chi2,
        plt,
        skewness_significance,
        stats,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Binomial, Poisson and Gaussian distributions

    Three distributions that **converge towards each other** in certain limits. Here you see when,
    simulate a binomial process, and fit it with all three.

    - **Binomial**: $n$ trials, each succeeding with probability $p$.
    - **Poisson**: the expected number $\lambda$ of successes, with no trials in sight.
    - **Gaussian**: a mean $\mu$ and a width $\sigma$.

    Every interactive block below has its controls directly above its plots.

    References: Barlow chapter 3, Cowan chapter 2, the Particle Data Group review
    [Probability theorem and distributions](https://pdg.lbl.gov/2020/reviews/rpp2020-rev-probability.pdf).

    ---

    **Adapted from** the Binomial, Poisson and Gaussian exercise by **Troels C. Petersen** (Niels
    Bohr Institute, 17th of November 2025), from the course *Applied Statistics 2025/26*. The original
    is an exercise with an example solution. This version keeps its three distributions, its fits and
    its questions, and is rewritten as an interactive demonstration.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The three distributions with the same mean

    The three share the mean $np$. The Gaussian takes the binomial width $\sqrt{np(1-p)}$. The Poisson
    width is $\sqrt{np}$, slightly larger, and cannot be changed. Which one has the longest tail
    towards high values, and which towards low values? Make $n$ large and $p$ small, then make
    $np$ large, and see when they start to look alike.
    """)
    return


@app.cell
def _(mo):
    pdf_n = mo.ui.slider(1, 400, 1, value=20, show_value=True, label="Number of trials n")
    pdf_p = mo.ui.slider(
        0.005, 0.95, 0.005, value=0.2, show_value=True, label="Success probability p"
    )
    pdf_log = mo.ui.switch(value=True, label="Logarithmic y axis")
    return pdf_log, pdf_n, pdf_p


@app.cell
def _(BINOMIAL, GAUSSIAN, POISSON, mo, np, pdf_log, pdf_n, pdf_p, plt, stats):
    _n, _p = pdf_n.value, pdf_p.value
    _mean = _n * _p
    _sigma = np.sqrt(_n * _p * (1 - _p))
    _x = np.arange(max(0, int(_mean - 6 * _sigma)), int(_mean + 7 * _sigma) + 2)
    _grid = np.linspace(_x[0] - 0.5, _x[-1] + 0.5, 800)

    _fig, _ax = plt.subplots(figsize=(13, 4.2))
    _ax.step(
        _x,
        stats.binom.pmf(_x, _n, _p),
        where="mid",
        color=BINOMIAL,
        lw=2,
        label=f"Binomial, n = {_n}, p = {_p:g}",
    )
    _ax.step(
        _x,
        stats.poisson.pmf(_x, _mean),
        where="mid",
        color=POISSON,
        lw=2,
        label=f"Poisson, λ = {_mean:.3g}",
    )
    _ax.plot(
        _grid,
        stats.norm.pdf(_grid, _mean, _sigma),
        color=GAUSSIAN,
        lw=2,
        label=f"Gaussian, μ = {_mean:.3g}, σ = {_sigma:.3g}",
    )
    if pdf_log.value:
        _ax.set_yscale("log")
        _ax.set_ylim(1e-6, None)
    _ax.set(xlabel="Number of successes x", ylabel="Probability")
    _ax.legend()
    _fig.tight_layout()

    _high = _mean + 3 * _sigma
    _tails = {
        "Binomial": stats.binom.sf(_high, _n, _p),
        "Poisson": stats.poisson.sf(_high, _mean),
        "Gaussian": stats.norm.sf(_high, _mean, _sigma),
    }
    _below_zero = stats.norm.cdf(-0.5, _mean, _sigma)
    mo.vstack(
        [
            mo.hstack([pdf_n, pdf_p, pdf_log], justify="start", gap=2),
            _fig,
            mo.callout(
                mo.md(
                    "Probability above the mean + 3σ: "
                    + ", ".join(f"**{name}** {value:.2e}" for name, value in _tails.items())
                    + f". The Gaussian puts {_below_zero:.2e} below zero, where the other two have none."
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Simulate a binomial process and fit it with all three

    Run many experiments, each with $n$ trials. Histogram the number of successes, with one bin per
    integer and the uncertainty $\sqrt{N}$ on each bin (the counts are Poisson distributed). Then fit
    a $\chi^2$ with the three distributions. Empty bins are left out, since their uncertainty is zero.

    The last line is the $\chi^2$ against the **true** binomial, with no free parameters: the number of
    degrees of freedom is the number of bins, and the probability should be reasonable **by construction**.
    """)
    return


@app.cell
def _(mo):
    sim_experiments = mo.ui.dropdown(
        {"100": 100, "1 000": 1_000, "10 000": 10_000, "100 000": 100_000},
        value="1 000",
        label="Experiments",
    )
    sim_trials = mo.ui.slider(2, 100, 1, value=20, show_value=True, label="Trials per experiment")
    sim_p = mo.ui.slider(0.01, 0.9, 0.01, value=0.2, show_value=True, label="Success probability")
    sim_seed = mo.ui.number(0, 9999, 1, value=42, label="Random seed")
    return sim_experiments, sim_p, sim_seed, sim_trials


@app.cell
def _(
    BINOMIAL,
    GAUSSIAN,
    POISSON,
    binomial_pmf,
    chi2_pvalue,
    fit_least_squares,
    integer_histogram,
    mo,
    np,
    pearson_chi2,
    plt,
    sim_experiments,
    sim_p,
    sim_seed,
    sim_trials,
    stats,
):
    _N, _n, _p = sim_experiments.value, sim_trials.value, sim_p.value
    _outcomes = np.random.default_rng(sim_seed.value).binomial(_n, _p, size=_N)
    _centres, _counts = integer_histogram(_outcomes, _n)
    _used = _counts > 0
    _x, _y, _sy = _centres[_used], _counts[_used], np.sqrt(_counts[_used])
    _mean = _n * _p

    def _binomial(x, N, n, p):
        return N * binomial_pmf(x, n, p)

    def _poisson(x, N, mu):
        return N * stats.poisson.pmf(x, mu)

    def _gaussian(x, N, mu, sigma):
        return N * stats.norm.pdf(x, mu, sigma)

    _fits = {
        "Binomial": (_binomial, {"N": _N, "n": _n, "p": _p}, BINOMIAL),
        "Poisson": (_poisson, {"N": _N, "mu": _mean}, POISSON),
        "Gaussian": (_gaussian, {"N": _N, "mu": _mean, "sigma": np.sqrt(_mean)}, GAUSSIAN),
    }

    _fig, _ax = plt.subplots(figsize=(13, 4.5))
    _ax.errorbar(_x, _y, _sy, xerr=0.5, fmt=".k", capsize=1, label="Simulated experiments")
    for _name, (_model, _start, _colour) in _fits.items():
        _fit = fit_least_squares(_model, _x, _y, _sy, _start)
        _probability = chi2_pvalue(_fit.fval, _fit.ndof) if _fit.ndof > 0 else float("nan")
        _ax.step(
            _centres,
            _model(_centres, *_fit.values),
            where="mid",
            color=_colour,
            lw=2,
            label=f"{_name} fit: χ² = {_fit.fval:.1f}, Ndof = {_fit.ndof}, p = {_probability:.3f}",
        )
    _ax.set(xlabel="Number of successes", ylabel="Frequency", ylim=(0, 1.25 * _y.max()))
    _ax.legend(loc="upper right")
    _fig.tight_layout()

    _expected = _N * binomial_pmf(_centres, _n, _p)
    _chi2_true, _ndof_true = pearson_chi2(_counts, _expected)
    _p_true = chi2_pvalue(_chi2_true, _ndof_true)
    mo.vstack(
        [
            mo.hstack(
                [mo.vstack([sim_experiments, sim_seed]), mo.vstack([sim_trials, sim_p])],
                widths="equal",
            ),
            _fig,
            mo.callout(
                mo.md(
                    f"**χ² against the true binomial** (no free parameters): χ² = {_chi2_true:.1f}, "
                    f"Ndof = {_ndof_true}, p = {_p_true:.3f}."
                ),
                kind="success" if _p_true > 0.01 else "warn",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A. Is the square-root uncertainty on a bin Gaussian?

    All the $\chi^2$ fits **assume** that the count in a bin fluctuates like a Gaussian of width
    $\sqrt{N}$. Repeat the whole histogram 300 times. For every bin, form the **pull**
    $(\text{observed} - \text{expected})/\sqrt{\text{expected}}$. If the assumption holds, the pulls
    follow a unit Gaussian (the line). The assumption is best when there are many entries in the bin. Reduce the
    number of experiments and watch it fail.
    """)
    return


@app.cell
def _(mo):
    pull_experiments = mo.ui.dropdown(
        {"20": 20, "100": 100, "1 000": 1_000, "10 000": 10_000},
        value="1 000",
        label="Experiments per histogram",
    )
    return (pull_experiments,)


@app.cell
def _(
    BINOMIAL, GAUSSIAN, binomial_pmf, mo, np, plt, pull_experiments, skewness_significance, stats
):
    _N, _n, _p = pull_experiments.value, 20, 0.2
    _pmf = binomial_pmf(np.arange(_n + 1), _n, _p)
    _expected = _N * _pmf
    _counts = np.random.default_rng(7).multinomial(_N, _pmf, size=300)
    _keep = _expected > 0.1
    _pulls = ((_counts - _expected) / np.sqrt(_expected))[:, _keep].ravel()
    _skewness, _sigma = skewness_significance(_pulls)

    _fig, _ax = plt.subplots(figsize=(13, 3.8))
    _grid = np.linspace(-5, 5, 400)
    _ax.hist(
        _pulls,
        bins=60,
        range=(-5, 5),
        density=True,
        color=BINOMIAL,
        alpha=0.4,
        label="Pulls of all bins",
    )
    _ax.plot(_grid, stats.norm.pdf(_grid), color=GAUSSIAN, lw=2, label="Unit Gaussian")
    _ax.set(xlabel="(observed − expected) / √expected", ylabel="Density")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack(
        [
            pull_experiments,
            _fig,
            mo.md(
                f"Skewness of the pulls: **{_skewness:.2f}** ± {_sigma:.2f}. A Gaussian has zero, "
                f"and the bins with a low expected count are skewed."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## B. Is the distribution Gaussian yet? The skewness

    A Gaussian has skewness zero. A binomial with $n=1000$ and $p=1/60$ has $np\approx 17$. Is its
    skewness consistent with zero? Change $n$ and $p$: the skewness approaches zero as $np$ grows.
    """)
    return


@app.cell
def _(mo):
    skew_n = mo.ui.slider(10, 5000, 10, value=1000, show_value=True, label="Trials n")
    skew_p = mo.ui.slider(
        0.001, 0.5, 0.001, value=1 / 60, show_value=True, label="Success probability p"
    )
    return skew_n, skew_p


@app.cell
def _(BINOMIAL, GAUSSIAN, mo, np, plt, skew_n, skew_p, skewness_significance, stats):
    _outcomes = np.random.default_rng(3).binomial(skew_n.value, skew_p.value, size=1000)
    _skewness, _sigma = skewness_significance(_outcomes)
    _z = _skewness / _sigma
    _mean, _width = _outcomes.mean(), _outcomes.std(ddof=1)
    _low, _high = int(_mean - 5 * _width), int(_mean + 5 * _width) + 1

    _fig, _ax = plt.subplots(figsize=(13, 3.6))
    _ax.hist(
        _outcomes,
        bins=np.arange(_low, _high) - 0.5,
        density=True,
        color=BINOMIAL,
        alpha=0.4,
        label="1 000 experiments",
    )
    _grid = np.linspace(_low, _high, 400)
    _ax.plot(
        _grid,
        stats.norm.pdf(_grid, _mean, _width),
        color=GAUSSIAN,
        lw=2,
        label="Gaussian with the same mean and width",
    )
    _ax.set(xlabel="Number of successes", ylabel="Density")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack(
        [
            mo.hstack([skew_n, skew_p], justify="start", gap=2),
            _fig,
            mo.callout(
                mo.md(
                    f"np = {skew_n.value * skew_p.value:.1f}. Skewness = {_skewness:.3f} ± {_sigma:.3f}, "
                    f"which is **{_z:.1f} σ** from zero."
                ),
                kind="success" if abs(_z) < 3 else "warn",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to take away

    1. A **binomial** needs a number of trials and a probability. A **Poisson** counts rare events, and
       needs only the expected number $\lambda$.
    2. For large $n$ and small $p$ the binomial becomes the Poisson. For large $\lambda = np$ both become Gaussian.
    3. The Poisson has the longest tail towards high values. The Gaussian is the only one that reaches below zero.
    4. The uncertainty on a count of $N$ is $\sqrt{N}$, but for a low $N$ that is far from Gaussian.
       $\chi^2$ fits suffer when the bins are nearly empty.
    """)
    return


if __name__ == "__main__":
    app.run()
