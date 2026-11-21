import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    from appstat.stats import chi2_pvalue, weighted_mean

    return chi2_pvalue, mo, np, plt, stats, weighted_mean


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The weighted mean, and why it is a $\chi^2$ fit

    Four people measure the same distance (the height of a hook), each with a different
    uncertainty. How do we combine the four numbers into one best estimate, and how do we
    know whether they even agree with each other?

    Every interactive block below has its controls directly above its plots. Move a slider
    and the plots under it respond.

    ---

    **Adapted from** `WeightedMeanSigmaChi2_original.ipynb` (latest update 10-11-2025) by
    **Troels C. Petersen** (Niels Bohr Institute), from the course *Applied Statistics 2025/26*.
    The original is an exercise. This version keeps its idea and its numbers, and is rewritten
    as an interactive demonstration.
    """)
    return


@app.cell
def _(mo):
    DEFAULT_X = [17.8, 18.1, 17.7, 17.7]
    DEFAULT_SIGMA = [0.5, 0.3, 0.5, 0.2]

    def make_measurements():
        """Fresh sliders for the four measurements and their uncertainties."""
        values = mo.ui.array(
            [
                mo.ui.slider(
                    17.0, 19.0, 0.05, value=v, show_value=True, label=f"Measurement {i + 1} [mm]"
                )
                for i, v in enumerate(DEFAULT_X)
            ]
        )
        sigmas = mo.ui.array(
            [
                mo.ui.slider(
                    0.05, 1.0, 0.05, value=s, show_value=True, label=f"Uncertainty {i + 1} [mm]"
                )
                for i, s in enumerate(DEFAULT_SIGMA)
            ]
        )
        return values, sigmas

    return DEFAULT_SIGMA, DEFAULT_X, make_measurements


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The weighted mean

    The weighted mean gives each measurement the weight $w_i = 1/\sigma_i^2$:

    $$\hat\mu = \frac{\sum_i w_i x_i}{\sum_i w_i}, \qquad
    \sigma_{\hat\mu} = \frac{1}{\sqrt{\sum_i w_i}}$$

    A precise measurement pulls the mean towards itself, a sloppy one hardly counts. The
    unweighted mean treats all of them alike.

    It is also a one-parameter $\chi^2$ fit. Suppose the true distance were $\mu$. Then each
    measurement is off by $(x_i - \mu)/\sigma_i$ standard deviations, and the $\chi^2$ adds up the squares:

    $$\chi^2(\mu) = \sum_i \left(\frac{x_i - \mu}{\sigma_i}\right)^2$$

    The curve in the right plot below is that sum for every $\mu$. Its **minimum** is the weighted
    mean. The curve is a parabola, and it rises by exactly 1 at one uncertainty away from the
    minimum. That is where the error bar on the mean comes from. The $\chi^2$ and its p-value
    tell you whether the measurements are consistent.
    """)
    return


@app.cell
def _(make_measurements):
    values, sigmas = make_measurements()
    return sigmas, values


@app.cell
def _(chi2_pvalue, mo, np, plt, sigmas, values, weighted_mean):
    _x = np.array(values.value)
    _sigma = np.array(sigmas.value)
    _mean_w, _error_w = weighted_mean(_x, _sigma)
    _mean_u = _x.mean()
    _error_u = _x.std(ddof=1) / np.sqrt(len(_x))
    _chi2 = float(np.sum(((_x - _mean_w) / _sigma) ** 2))
    _ndof = len(_x) - 1  # four numbers, minus the one parameter (the mean) we fitted
    _p_value = chi2_pvalue(_chi2, _ndof)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 4.3))
    _index = np.arange(1, len(_x) + 1)
    _ax1.errorbar(_index, _x, _sigma, fmt="ko", capsize=3, label="Measurements")
    _ax1.axhline(_mean_w, color="#D55E00", label="Weighted mean")
    _ax1.axhspan(_mean_w - _error_w, _mean_w + _error_w, color="#D55E00", alpha=0.25)
    _ax1.axhline(_mean_u, color="#0072B2", ls="--", label="Unweighted mean")
    _ax1.set(
        xlabel="Measurement",
        ylabel="Distance [mm]",
        xticks=_index,
        title="Weighted against unweighted",
    )
    _ax1.legend()

    _mu = np.linspace(_mean_w - 4 * _error_w, _mean_w + 4 * _error_w, 400)
    _chi2_mu = np.sum(((_x[:, None] - _mu[None, :]) / _sigma[:, None]) ** 2, axis=0)
    _ax2.plot(_mu, _chi2_mu, color="k")
    _ax2.plot(_mean_w, _chi2, "o", color="#D55E00", label="Minimum: the weighted mean")
    _ax2.axhline(_chi2 + 1, color="#D55E00", ls="--", label=r"$\chi^2_{min} + 1$")
    _ax2.axvspan(
        _mean_w - _error_w,
        _mean_w + _error_w,
        color="#D55E00",
        alpha=0.25,
        label="Mean ± uncertainty",
    )
    _ax2.set(
        xlabel=r"Assumed true distance $\mu$ [mm]",
        ylabel=r"$\chi^2(\mu)$",
        title="The weighted mean is a fit",
    )
    _ax2.legend()
    _fig.tight_layout()

    _verdict = (
        mo.callout(
            mo.md("The measurements are **consistent** with one common true value."), kind="success"
        )
        if _p_value > 0.05
        else mo.callout(
            mo.md(
                "The measurements are **inconsistent**: they scatter more than their "
                "uncertainties allow. Do not trust the mean until you know why."
            ),
            kind="danger",
        )
    )
    mo.vstack(
        [
            mo.hstack([values, sigmas], widths="equal"),
            mo.md(rf"""
            | | value [mm] | uncertainty [mm] |
            |---|---|---|
            | unweighted mean | {_mean_u:.3f} | {_error_u:.3f} |
            | **weighted mean** | **{_mean_w:.3f}** | **{_error_w:.3f}** |

            $\chi^2$ = {_chi2:.3f} for {_ndof} degrees of freedom, so the p-value is **{_p_value:.3f}**.
            With the starting values you should see 17.810 ± 0.151 mm, $\chi^2$ = 1.286 and p = 0.733.
            """),
            _verdict,
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to take away

    1. A weighted mean differs from an unweighted one by trusting precise measurements more.
    2. A weighted mean is a $\chi^2$ fit with one parameter, the mean. Its error bar is the
       distance at which $\chi^2$ has grown by 1.
    3. The $\chi^2$ with its number of degrees of freedom (here 4 − 1 = 3) and the p-value tell
       you whether the inputs are consistent before you combine them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Additional work

    Four more investigations. Each one has its own controls directly above its plots.

    ## A and B. Which measurement hurts the $\chi^2$, and what to do about it

    **A. Pulls.** The pull of a measurement is its distance to the weighted mean in units of its own
    uncertainty, $(x_i - \hat\mu)/\sigma_i$. The $\chi^2$ is the sum of the squared pulls, so the
    largest pull belongs to the measurement that disagrees most. For consistent measurements the
    pulls scatter around 0 with a width of about 1 (a little less, because the mean was fitted to the
    same data). Bars beyond ±2 are red. Try moving one measurement far from the others.

    **B. The scale factor.** If $\chi^2$/Ndof is clearly above 1, the measurements scatter more than their
    uncertainties explain. A common remedy (used by the Particle Data Group) is to multiply the
    uncertainty on the mean by $\sqrt{\chi^2/\mathrm{Ndof}}$. That is honest only if you believe **all**
    uncertainties are underestimated by the same factor. A single outlier should be investigated,
    not scaled away. The factor is never below 1.
    """)
    return


@app.cell
def _(make_measurements):
    pull_values, pull_sigmas = make_measurements()
    return pull_sigmas, pull_values


@app.cell
def _(mo, np, plt, pull_sigmas, pull_values, weighted_mean):
    _x = np.array(pull_values.value)
    _sigma = np.array(pull_sigmas.value)
    _mean_w, _error_w = weighted_mean(_x, _sigma)
    _pulls = (_x - _mean_w) / _sigma
    _chi2 = float(np.sum(_pulls**2))
    _ndof = len(_x) - 1
    _scale = max(1.0, (_chi2 / _ndof) ** 0.5)

    _fig, _ax = plt.subplots(figsize=(8, 3.5))
    _index = np.arange(1, len(_x) + 1)
    _ax.bar(_index, _pulls, color=np.where(np.abs(_pulls) > 2, "#CC79A7", "#0072B2"))
    _ax.axhspan(-2, 2, color="0.9", zorder=0)
    _ax.axhline(0, color="k", lw=1)
    _ax.set(xlabel="Measurement", ylabel="Pull", xticks=_index, ylim=(-4, 4))
    _fig.tight_layout()
    mo.vstack(
        [
            mo.hstack([pull_values, pull_sigmas], widths="equal"),
            _fig,
            mo.md(
                rf"""
                The sum of the squared pulls is {np.sum(_pulls**2):.3f}, the $\chi^2$ is {_chi2:.3f}.
                χ²/Ndof = {_chi2 / _ndof:.2f}, so the scale factor is **{_scale:.2f}**: the uncertainty on
                the mean goes from {_error_w:.3f} mm to **{_error_w * _scale:.3f} mm**.
                """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## C. Repeat the whole measurement

    Take the weighted mean of the starting values as the true distance, and simulate new sets of four
    measurements with the starting uncertainties. Change the number of sets or the seed to run the
    simulation again. If the formulas are right, then (left) the pull of the weighted mean,
    $(\hat\mu - \mu_{true})/\sigma_{\hat\mu}$, follows a standard normal distribution, and (right)
    the $\chi^2$ follows the $\chi^2$ distribution with Ndof = 3. Neither red curve is a fit.
    """)
    return


@app.cell
def _(mo):
    n_repeats = mo.ui.dropdown(
        {"1 000": 1_000, "20 000": 20_000, "100 000": 100_000},
        value="20 000",
        label="Repeated sets",
    )
    repeat_seed = mo.ui.number(0, 9999, 1, value=42, label="Random seed")
    return n_repeats, repeat_seed


@app.cell
def _(
    DEFAULT_SIGMA,
    DEFAULT_X,
    mo,
    n_repeats,
    np,
    plt,
    repeat_seed,
    stats,
    weighted_mean,
):
    _sigma = np.array(DEFAULT_SIGMA)
    _mean_w, _error_w = weighted_mean(np.array(DEFAULT_X), _sigma)
    _ndof = len(_sigma) - 1
    _rng = np.random.default_rng(repeat_seed.value)
    _sets = _rng.normal(
        _mean_w, _sigma, size=(n_repeats.value, len(_sigma))
    )  # one row per repetition
    _weights = 1.0 / _sigma**2
    _means = (_sets * _weights).sum(axis=1) / _weights.sum()
    _chi2 = np.sum(((_sets - _means[:, None]) / _sigma) ** 2, axis=1)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4))
    _pull = (_means - _mean_w) / _error_w
    _ax1.hist(_pull, bins=80, density=True, histtype="step", color="#0072B2", label="Simulated")
    _grid = np.linspace(-4, 4, 400)
    _ax1.plot(_grid, stats.norm.pdf(_grid), color="#D55E00", label="Standard normal")
    _ax1.set(xlabel="(weighted mean − true value) / uncertainty", title="Pull of the mean")
    _ax1.legend()

    _ax2.hist(_chi2, bins=80, density=True, histtype="step", color="#0072B2", label="Simulated")
    _grid = np.linspace(0.01, _chi2.max(), 400)
    _ax2.plot(
        _grid,
        stats.chi2.pdf(_grid, _ndof),
        color="#D55E00",
        label=f"χ² distribution, Ndof = {_ndof}",
    )
    _ax2.set(xlabel="χ²", title="χ² of the four measurements")
    _ax2.legend()
    _fig.tight_layout()
    mo.vstack([mo.hstack([n_repeats, repeat_seed], justify="start", gap=2), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## D. Equal uncertainties

    Give all four measurements (the starting values) the same uncertainty with the slider. The weights
    are then equal and the weighted mean **is** the unweighted mean: the two lines on top of each
    other. The uncertainty becomes $\sigma/\sqrt{N}$.
    """)
    return


@app.cell
def _(mo):
    equal_sigma = mo.ui.slider(
        0.05, 1.0, 0.05, value=0.3, show_value=True, label="Common uncertainty [mm]"
    )
    return (equal_sigma,)


@app.cell
def _(DEFAULT_X, equal_sigma, mo, np, plt, weighted_mean):
    _x = np.array(DEFAULT_X)
    _equal = equal_sigma.value
    _mean, _error = weighted_mean(_x, np.full_like(_x, _equal))
    assert np.isclose(_mean, _x.mean())  # equal weights: the weighted mean is the plain mean

    _fig, _ax = plt.subplots(figsize=(8, 3.8))
    _index = np.arange(1, len(_x) + 1)
    _ax.errorbar(_index, _x, _equal, fmt="ko", capsize=3, label="Measurements")
    _ax.axhline(_mean, color="#D55E00", lw=3, label="Weighted mean")
    _ax.axhline(_x.mean(), color="#0072B2", ls="--", lw=2, label="Unweighted mean")
    _ax.axhspan(_mean - _error, _mean + _error, color="#D55E00", alpha=0.25)
    _ax.set(xlabel="Measurement", ylabel="Distance [mm]", xticks=_index)
    _ax.legend()
    _fig.tight_layout()
    mo.vstack(
        [
            equal_sigma,
            _fig,
            mo.md(
                rf"""
                Weighted mean {_mean:.3f} mm, unweighted mean {_x.mean():.3f} mm. Uncertainty:
                σ/√N = {_equal}/√{len(_x)} = {_equal / np.sqrt(len(_x)):.3f} mm, and the function gives
                {_error:.3f} mm.
                """
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
