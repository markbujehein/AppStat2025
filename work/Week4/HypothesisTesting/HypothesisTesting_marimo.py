import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from appstat.stats import two_sample_pvalues

    A_COLOUR, B_COLOUR = "#0072B2", "#D55E00"
    TESTS = ("Means", "χ² of histograms", "Kolmogorov-Smirnov")
    TEST_COLOURS = ("#0072B2", "#D55E00", "#009E73")

    def draw_experiments(n_experiments, n_events, shift, width, n_bins, rng):
        """The p-values of three tests for many pairs of samples, shape ``(n_experiments, 3)``."""
        low, high = -5.0 - abs(shift), 5.0 + abs(shift) + 2 * abs(width - 1)
        return np.array(
            [
                two_sample_pvalues(
                    rng.normal(shift, width, n_events),
                    rng.normal(0.0, 1.0, n_events),
                    n_bins,
                    (low, high),
                )
                for _ in range(n_experiments)
            ]
        )

    return (
        A_COLOUR,
        B_COLOUR,
        TESTS,
        TEST_COLOURS,
        draw_experiments,
        mo,
        np,
        plt,
        two_sample_pvalues,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hypothesis testing: are two distributions the same?

    Two samples, A and B, and one question: **do they come from the same distribution?** Three tests, each
    giving a **p-value**, the probability of a difference at least this large if they really are the same:

    1. **Means**: the difference of the means in units of its uncertainty, which is a unit Gaussian if the samples agree.
    2. **χ² between the histograms.** It needs binning.
    3. **Kolmogorov-Smirnov**: the largest gap between the two cumulative distributions. No binning, and it works for
       any distribution.

    A hypothesis test has four steps: state the **null hypothesis** (the same distribution), state the **criterion**
    (reject if $p < 0.27\,\%$, three σ), compute the **test statistic**, and **decide**. Not rejecting is not a proof
    that they are the same.

    Every interactive block below has its controls directly above its plots.

    References: Barlow pages 155-156, and the Kolmogorov-Smirnov test on Wikipedia.

    ---

    **Adapted from** the Hypothesis Testing exercise by **Troels C. Petersen** (Niels Bohr Institute), from the
    course *Applied Statistics 2025/26*. The original is an exercise with an example solution. This version keeps
    its three tests and its questions, and is rewritten as an interactive demonstration. The histogram χ² compares
    samples of equal size, so unlike the original there is no normalisation change.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. One experiment

    B is a unit Gaussian. Move or widen A, and see **by eye** when you can tell them apart. Then look at the p-values.
    How large a shift of the mean do you need to see?
    """)
    return


@app.cell
def _(mo):
    one_n = mo.ui.slider(20, 2000, 10, value=100, show_value=True, label="Events in each sample")
    one_shift = mo.ui.slider(0.0, 1.0, 0.05, value=0.0, show_value=True, label="Mean of A")
    one_width = mo.ui.slider(0.5, 2.0, 0.05, value=1.0, show_value=True, label="Width of A")
    one_bins = mo.ui.slider(5, 100, 5, value=25, show_value=True, label="Bins")
    one_seed = mo.ui.number(0, 9999, 1, value=42, label="Random seed")
    return one_bins, one_n, one_seed, one_shift, one_width


@app.cell
def _(
    A_COLOUR,
    B_COLOUR,
    mo,
    np,
    one_bins,
    one_n,
    one_seed,
    one_shift,
    one_width,
    plt,
    two_sample_pvalues,
):
    _rng = np.random.default_rng(one_seed.value)
    _a = _rng.normal(one_shift.value, one_width.value, one_n.value)
    _b = _rng.normal(0.0, 1.0, one_n.value)
    _fig, _ax = plt.subplots(figsize=(13, 3.6))
    _edges = np.linspace(-5, 5, one_bins.value + 1)
    _ax.hist(
        _a,
        _edges,
        histtype="step",
        lw=2,
        color=A_COLOUR,
        label=f"A (mean {one_shift.value:.2f}, width {one_width.value:.2f})",
    )
    _ax.hist(_b, _edges, histtype="step", lw=2, color=B_COLOUR, label="B (unit Gaussian)")
    _ax.set(xlabel="x", ylabel="Entries per bin")
    _ax.legend()
    _fig.tight_layout()
    _p_mean, _p_chi2, _p_ks = two_sample_pvalues(_a, _b, one_bins.value, (-5, 5))
    _worst = min(_p_mean, _p_chi2, _p_ks)
    mo.vstack(
        [
            mo.hstack([one_n, one_shift, one_width], justify="start", gap=2),
            mo.hstack([one_bins, one_seed], justify="start", gap=2),
            _fig,
            mo.callout(
                mo.md(
                    f"p-values: means **{_p_mean:.3g}**, χ² **{_p_chi2:.3g}**, KS **{_p_ks:.3g}**. "
                    f"{'At least one test rejects at three σ (p < 0.0027).' if _worst < 0.0027 else 'No test rejects at three σ.'}"
                ),
                kind="warn" if _worst < 0.0027 else "success",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Many experiments: the distribution of p-values

    If A and B really are the same, a good test gives p-values that are **flat between 0 and 1**: 5 % of the time
    $p < 0.05$, by construction. Repeat the experiment and look. Then change the mean or the width and see which
    tests react. The χ² test is only an approximation for few events per bin, and the KS test can only take
    certain values for few events.
    """)
    return


@app.cell
def _(mo):
    many_n = mo.ui.slider(20, 2000, 10, value=100, show_value=True, label="Events in each sample")
    many_experiments = mo.ui.slider(100, 1000, 100, value=500, show_value=True, label="Experiments")
    many_shift = mo.ui.slider(0.0, 0.6, 0.05, value=0.0, show_value=True, label="Mean of A")
    many_width = mo.ui.slider(0.7, 1.5, 0.05, value=1.0, show_value=True, label="Width of A")
    many_bins = mo.ui.slider(5, 100, 5, value=20, show_value=True, label="Bins")
    return many_bins, many_experiments, many_n, many_shift, many_width


@app.cell
def _(
    TESTS,
    TEST_COLOURS,
    draw_experiments,
    many_bins,
    many_experiments,
    many_n,
    many_shift,
    many_width,
    mo,
    np,
    plt,
):
    _p = draw_experiments(
        many_experiments.value,
        many_n.value,
        many_shift.value,
        many_width.value,
        many_bins.value,
        np.random.default_rng(7),
    )
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    for _ax, _name, _colour, _values in zip(_axes, TESTS, TEST_COLOURS, _p.T, strict=True):
        _ax.hist(_values, 20, range=(0, 1), color=_colour, alpha=0.8)
        _ax.axhline(len(_values) / 20, color="k", ls="--", lw=1)
        _ax.set(title=_name, xlabel="p-value")
    _axes[0].set_ylabel("Experiments")
    _fig.tight_layout()
    _rejected = (_p < 0.05).mean(axis=0)
    mo.vstack(
        [
            mo.hstack([many_n, many_experiments, many_shift], justify="start", gap=2),
            mo.hstack([many_width, many_bins], justify="start", gap=2),
            _fig,
            mo.callout(
                mo.md(
                    "Fraction with p < 0.05 (5 % if A and B are the same): "
                    + ", ".join(
                        f"{name} **{r:.0%}**" for name, r in zip(TESTS, _rejected, strict=True)
                    )
                    + ". The dashed line is the flat distribution."
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A. How much of a change does each test see?

    The **power** of a test is how often it rejects when A and B differ. Vary the mean or the width and see how fast each
    test's power rises. The means test is blind to a change of width, and the χ² and KS tests see both.
    """)
    return


@app.cell
def _(mo):
    power_vary = mo.ui.dropdown(["Mean of A", "Width of A"], value="Mean of A", label="Change")
    power_n = mo.ui.slider(20, 1000, 10, value=100, show_value=True, label="Events in each sample")
    power_experiments = mo.ui.slider(
        50, 300, 50, value=100, show_value=True, label="Experiments per point"
    )
    return power_experiments, power_n, power_vary


@app.cell
def _(
    TESTS,
    TEST_COLOURS,
    draw_experiments,
    mo,
    np,
    plt,
    power_experiments,
    power_n,
    power_vary,
):
    _vary_mean = power_vary.value == "Mean of A"
    _values = np.linspace(0, 0.8, 9) if _vary_mean else np.linspace(1.0, 1.6, 9)
    _rng = np.random.default_rng(3)
    _power = np.array(
        [
            (
                draw_experiments(
                    power_experiments.value,
                    power_n.value,
                    v if _vary_mean else 0.0,
                    1.0 if _vary_mean else v,
                    20,
                    _rng,
                )
                < 0.05
            ).mean(axis=0)
            for v in _values
        ]
    )
    _fig, _ax = plt.subplots(figsize=(13, 3.6))
    for _name, _colour, _column in zip(TESTS, TEST_COLOURS, _power.T, strict=True):
        _ax.plot(_values, _column, "o-", color=_colour, label=_name)
    _ax.axhline(0.05, color="k", ls=":", lw=1)
    _ax.axhline(0.95, color="k", ls=":", lw=1)
    _ax.set(xlabel=power_vary.value, ylabel="Fraction rejected (p < 0.05)", ylim=(0, 1.02))
    _ax.legend()
    _fig.tight_layout()
    _sigma = np.sqrt(2 / power_n.value)
    mo.vstack(
        [
            mo.hstack([power_vary, power_n, power_experiments], justify="start", gap=2),
            _fig,
            mo.callout(
                mo.md(
                    f"For the means test the uncertainty on the difference is $\\sqrt{{2/N}}$ = {_sigma:.3f}, "
                    f"so a shift of about **{3.3 * _sigma:.2f}** is needed for 95 % to be rejected at p < 0.05 (3.3 σ)."
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

    1. A p-value says how surprising the data are **if the null hypothesis is true**. If it is, the p-values are flat: 5 % of
       the experiments give $p < 0.05$.
    2. Not rejecting the null hypothesis does not prove it. The means test did not see a change of width.
    3. The tests differ in **power**. The means test is best for a shift of the mean, but blind to anything else.
       Kolmogorov-Smirnov needs no binning and works for any distribution.
    4. The χ² between histograms is only approximately a χ² for few entries per bin. Choose the binning with care.
    5. Decide the **criterion before** looking at the p-value.
    """)
    return


if __name__ == "__main__":
    app.run()
