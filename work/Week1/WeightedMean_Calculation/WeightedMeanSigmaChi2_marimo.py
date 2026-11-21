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

    Move the sliders below and watch three things respond:

    1. the **weighted mean** and its uncertainty,
    2. the **$\chi^2$ value** and its **p-value**, which say whether the measurements are
       consistent,
    3. the **$\chi^2$ curve**, which shows that the weighted mean is just a one-parameter fit.

    ---

    **Adapted from** `WeightedMeanSigmaChi2_original.ipynb` (latest update 10-11-2025) by
    **Troels C. Petersen** (Niels Bohr Institute), from the course *Applied Statistics 2025/26*.
    The original is an exercise. This version keeps its idea and its numbers, and is rewritten
    as an interactive demonstration.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The measurements
    """)
    return


@app.cell
def _(mo):
    values = mo.ui.array(
        [
            mo.ui.slider(17.0, 19.0, 0.05, value=v, label=f"Measurement {i + 1} [mm]")
            for i, v in enumerate([17.8, 18.1, 17.7, 17.7])
        ]
    )
    sigmas = mo.ui.array(
        [
            mo.ui.slider(0.05, 1.0, 0.05, value=s, label=f"Uncertainty {i + 1} [mm]")
            for i, s in enumerate([0.5, 0.3, 0.5, 0.2])
        ]
    )
    mo.hstack([values, sigmas], widths="equal")
    return sigmas, values


@app.cell
def _(chi2_pvalue, np, sigmas, values, weighted_mean):
    x = np.array(values.value)
    sigma = np.array(sigmas.value)

    mean_w, error_w = weighted_mean(x, sigma)
    mean_u = x.mean()
    error_u = x.std(ddof=1) / np.sqrt(len(x))

    chi2 = np.sum(((x - mean_w) / sigma) ** 2)
    ndof = len(x) - 1  # four numbers, minus the one parameter (the mean) we fitted
    p_value = chi2_pvalue(chi2, ndof)
    return chi2, error_u, error_w, mean_u, mean_w, ndof, p_value, sigma, x


@app.cell(hide_code=True)
def _(chi2, error_u, error_w, mean_u, mean_w, mo, ndof, p_value):
    _verdict = (
        mo.callout(
            mo.md("The measurements are **consistent** with one common true value."),
            kind="success",
        )
        if p_value > 0.05
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
            mo.md(rf"""
            | | value [mm] | uncertainty [mm] |
            |---|---|---|
            | unweighted mean | {mean_u:.3f} | {error_u:.3f} |
            | **weighted mean** | **{mean_w:.3f}** | **{error_w:.3f}** |

            $\chi^2$ = {chi2:.3f} for {ndof} degrees of freedom, so the p-value is **{p_value:.3f}**.
            """),
            _verdict,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With the starting values you should see a weighted mean of 17.810 ± 0.151 mm,
    $\chi^2$ = 1.286 and p = 0.733.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Weighted against unweighted

    The weighted mean gives each measurement the weight $w_i = 1/\sigma_i^2$:

    $$\hat\mu = \frac{\sum_i w_i x_i}{\sum_i w_i}, \qquad
    \sigma_{\hat\mu} = \frac{1}{\sqrt{\sum_i w_i}}$$

    A precise measurement pulls the mean towards itself, a sloppy one hardly counts. The
    unweighted mean treats all of them alike.
    """)
    return


@app.cell
def _(error_w, mean_u, mean_w, np, plt, sigma, x):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _index = np.arange(1, len(x) + 1)
    _ax.errorbar(_index, x, sigma, fmt="ko", capsize=3, label="Measurements")
    _ax.axhline(mean_w, color="tab:red", label="Weighted mean")
    _ax.axhspan(mean_w - error_w, mean_w + error_w, color="tab:red", alpha=0.25)
    _ax.axhline(mean_u, color="tab:blue", ls="--", label="Unweighted mean")
    _ax.set(xlabel="Measurement", ylabel="Distance [mm]", xticks=_index)
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. The weighted mean is a one-parameter $\chi^2$ fit

    Suppose the true distance were $\mu$. Then each measurement is off by
    $(x_i - \mu)/\sigma_i$ standard deviations, and the $\chi^2$ adds up the squares:

    $$\chi^2(\mu) = \sum_i \left(\frac{x_i - \mu}{\sigma_i}\right)^2$$

    The curve below is that sum for every $\mu$. Its **minimum** is the weighted mean. The
    curve is a parabola, and it rises by exactly 1 at one uncertainty away from the
    minimum. That is where the error bar on the mean comes from.
    """)
    return


@app.cell
def _(chi2, error_w, mean_w, np, plt, sigma, x):
    _mu = np.linspace(mean_w - 4 * error_w, mean_w + 4 * error_w, 400)
    _chi2_mu = np.sum(((x[:, None] - _mu[None, :]) / sigma[:, None]) ** 2, axis=0)

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(_mu, _chi2_mu, color="k")
    _ax.plot(mean_w, chi2, "ro", label="Minimum: the weighted mean")
    _ax.axhline(chi2 + 1, color="tab:red", ls="--", label=r"$\chi^2_{min} + 1$")
    _ax.axvspan(
        mean_w - error_w, mean_w + error_w, color="tab:red", alpha=0.25, label="Mean ± uncertainty"
    )
    _ax.set(xlabel=r"Assumed true distance $\mu$ [mm]", ylabel=r"$\chi^2(\mu)$")
    _ax.legend()
    _fig
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

    Four more investigations. A and B use the measurements chosen above. C and D have their own
    controls, placed right under their text.

    ## A. Pulls: which measurement hurts the $\chi^2$?

    The pull of a measurement is its distance to the weighted mean in units of its own
    uncertainty, $(x_i - \hat\mu)/\sigma_i$. The $\chi^2$ is the sum of the squared pulls, so the
    largest pull belongs to the measurement that disagrees most. For consistent measurements the
    pulls scatter around 0 with a width of about 1 (a little less, because the mean was fitted to the
    same data). Bars beyond ±2 are red.
    """)
    return


@app.cell
def _(chi2, mean_w, mo, np, plt, sigma, x):
    pulls = (x - mean_w) / sigma
    _index = np.arange(1, len(x) + 1)
    _fig, _ax = plt.subplots(figsize=(8, 3.5))
    _ax.bar(_index, pulls, color=np.where(np.abs(pulls) > 2, "tab:red", "tab:blue"))
    _ax.axhline(0, color="k", lw=1)
    _ax.set(xlabel="Measurement", ylabel="Pull", xticks=_index)
    mo.vstack(
        [
            _fig,
            mo.md(
                f"Check: the sum of the squared pulls is {np.sum(pulls**2):.3f}, the $\\chi^2$ is {chi2:.3f}."
            ),
        ]
    )
    return (pulls,)


@app.cell(hide_code=True)
def _(chi2, error_w, mo, ndof):
    _scale = max(1.0, (chi2 / ndof) ** 0.5)
    mo.md(rf"""
    ## B. When the scatter is too large: the scale factor

    If χ²/Ndof is clearly above 1, the measurements scatter more than their uncertainties explain.
    A common remedy (used by the Particle Data Group) is to multiply the uncertainty on the mean by
    √(χ²/Ndof). That is honest only if you believe **all** uncertainties are underestimated by the
    same factor. A single outlier should be investigated, not scaled away.

    Now: χ²/Ndof = {chi2 / ndof:.2f}, scale factor = {_scale:.2f}, so the uncertainty on the mean
    goes from {error_w:.3f} mm to **{error_w * _scale:.3f} mm**. (The factor is never below 1:
    uncertainties are not shrunk when the data scatter less than expected.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## C. Repeat the whole measurement 20 000 times

    Take the weighted mean as the true distance, and simulate new sets of four measurements with the
    uncertainties above. Choose the number of sets and the random seed below, then change either to run
    the simulation again. If the formulas are right, then (left) the pull of the weighted mean,
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
    mo.hstack([n_repeats, repeat_seed], widths="equal")
    return n_repeats, repeat_seed


@app.cell
def _(error_w, mean_w, n_repeats, ndof, np, plt, repeat_seed, sigma, stats):
    _rng = np.random.default_rng(repeat_seed.value)
    _sets = _rng.normal(
        mean_w, sigma, size=(n_repeats.value, len(sigma))
    )  # one row per repeated measurement
    _weights = 1.0 / sigma**2
    _means = (_sets * _weights).sum(axis=1) / _weights.sum()
    _chi2 = np.sum(((_sets - _means[:, None]) / sigma) ** 2, axis=1)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4))
    _pull = (_means - mean_w) / error_w
    _ax1.hist(_pull, bins=80, density=True, histtype="step", label="Simulated")
    _grid = np.linspace(-4, 4, 400)
    _ax1.plot(_grid, stats.norm.pdf(_grid), "r", label="Standard normal")
    _ax1.set(xlabel="(weighted mean − true value) / uncertainty", title="Pull of the mean")
    _ax1.legend()

    _ax2.hist(_chi2, bins=80, density=True, histtype="step", label="Simulated")
    _grid = np.linspace(0.01, _chi2.max(), 400)
    _ax2.plot(_grid, stats.chi2.pdf(_grid, ndof), "r", label=f"χ² distribution, Ndof = {ndof}")
    _ax2.set(xlabel="χ²", title="χ² of the four measurements")
    _ax2.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## D. Equal uncertainties

    Give all four measurements the same uncertainty with the slider below. The weights are then equal and
    the weighted mean **is** the unweighted mean. The measurements themselves come from the sliders above.
    """)
    return


@app.cell
def _(mo):
    equal_sigma = mo.ui.slider(0.05, 1.0, 0.05, value=0.3, label="Common uncertainty [mm]")
    equal_sigma
    return (equal_sigma,)


@app.cell
def _(equal_sigma, mo, np, weighted_mean, x):
    _equal = equal_sigma.value
    _mean, _error = weighted_mean(x, np.full_like(x, _equal))
    assert np.isclose(_mean, x.mean())  # equal weights: the weighted mean is the plain mean
    mo.md(rf"""
    | | value [mm] | uncertainty [mm] |
    |---|---|---|
    | weighted mean, equal uncertainties | {_mean:.3f} | {_error:.3f} |
    | unweighted mean | {x.mean():.3f} | σ/√N = {_equal}/√{len(x)} = {_equal / np.sqrt(len(x)):.3f} |
    """)
    return


if __name__ == "__main__":
    app.run()
