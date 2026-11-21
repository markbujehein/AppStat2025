import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    return mo, np, plt, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The Central Limit Theorem

    Add random numbers from different distributions and look at the distribution of the sums. The
    **Central Limit Theorem (CLT)** says it ends up Gaussian. It also says how the widths combine: they
    add in quadrature, so a sum of $N$ numbers of unit width has width $\sqrt{N}$. Dividing the sum
    by $\sqrt{N}$ therefore gives a Gaussian of unit width, which is easy to compare with.

    Every interactive block below has its controls directly above its plots. Move them and watch the
    histogram of the sums.

    References: Barlow p. 49 (and p. 45 for the uniform distribution), Cowan p. 33,
    [Wikipedia: Central limit theorem](http://en.wikipedia.org/wiki/Central_limit_theorem).

    ---

    **Adapted from** the Central Limit Theorem exercise by **Troels C. Petersen** (Niels Bohr
    Institute, 10th of November 2025), from the course *Applied Statistics 2025/26*. The original is
    an exercise. This version keeps its three input distributions and its unit-Gaussian comparison,
    and is rewritten as an interactive demonstration.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The three inputs

    Each input is shifted and scaled to **mean 0 and width 1**:

    - **Uniform**: $\sqrt{12}\,(u - 0.5)$ with $u$ uniform between 0 and 1. A uniform distribution of
      full width $w$ has standard deviation $w/\sqrt{12}$, so $w = \sqrt{12}$ gives width 1.
    - **Exponential**: $e - 1$ with $e$ exponential with mean 1. The exponential has mean 1 and width 1,
      so subtracting 1 centres it.
    - **Cauchy**: $1/(1 + x^2)$. It has long tails and **no finite width at all**. Including it makes
      you think about how the theorem is stated.

    Choose how many numbers of each kind go into every sum.
    """)
    return


@app.cell
def _(mo):
    n_experiments = mo.ui.dropdown(
        {"1 000": 1_000, "10 000": 10_000, "100 000": 100_000},
        value="10 000",
        label="Number of sums",
    )
    n_uniform = mo.ui.slider(0, 30, 1, value=10, show_value=True, label="Uniform numbers per sum")
    n_exponential = mo.ui.slider(
        0, 30, 1, value=0, show_value=True, label="Exponential numbers per sum"
    )
    n_cauchy = mo.ui.slider(0, 30, 1, value=0, show_value=True, label="Cauchy numbers per sum")
    seed = mo.ui.number(0, 9999, 1, value=42, label="Random seed")
    return n_cauchy, n_exponential, n_experiments, n_uniform, seed


@app.cell
def _(mo, n_cauchy, n_exponential, n_experiments, n_uniform, np, plt, seed, stats):
    _controls = mo.hstack(
        [
            mo.vstack([n_uniform, n_exponential, n_cauchy]),
            mo.vstack([n_experiments, seed]),
        ],
        widths="equal",
    )
    _counts = {
        "Uniform": n_uniform.value,
        "Exponential": n_exponential.value,
        "Cauchy": n_cauchy.value,
    }
    _n_total = sum(_counts.values())
    mo.stop(
        _n_total == 0,
        mo.vstack(
            [_controls, mo.callout(mo.md("Choose at least one number per sum."), kind="warn")]
        ),
    )

    _rng = np.random.default_rng(seed.value)
    _size = n_experiments.value
    _draws = {
        "Uniform": lambda n: (_rng.uniform(size=(_size, n)) - 0.5) * np.sqrt(12),
        "Exponential": lambda n: _rng.exponential(size=(_size, n)) - 1.0,
        "Cauchy": lambda n: _rng.standard_cauchy(size=(_size, n)),
    }
    _inputs = {name: _draws[name](n) for name, n in _counts.items() if n > 0}
    _sums = sum(values.sum(axis=1) for values in _inputs.values()) / np.sqrt(_n_total)

    _ranges = {"Uniform": (-2.5, 2.5), "Exponential": (-1.5, 8.5), "Cauchy": (-5.0, 5.0)}
    _fig = plt.figure(figsize=(13, 8))
    _grid = _fig.add_gridspec(2, 3, height_ratios=[1, 1.6])
    for _col, _name in enumerate(_counts):
        _ax = _fig.add_subplot(_grid[0, _col])
        if _name in _inputs:
            _values = _inputs[_name].ravel()
            _ax.hist(_values, bins=100, range=_ranges[_name], histtype="step", color="#0072B2")
            _ax.text(
                0.1,
                0.95,
                f"Entries: {len(_values)}\nMean: {_values.mean():5.3f}\nStd: {_values.std(ddof=1):5.3f}",
                family="monospace",
                va="top",
                transform=_ax.transAxes,
            )
        else:
            _ax.text(
                0.5, 0.5, "not used", ha="center", va="center", transform=_ax.transAxes, color="0.5"
            )
        _ax.set(title=_name, xlim=_ranges[_name])

    _ax2 = _fig.add_subplot(_grid[1, :])
    _low, _high, _bins = -5.0, 5.0, 100
    _ax2.hist(
        _sums,
        bins=_bins,
        range=(_low, _high),
        density=True,
        histtype="step",
        color="k",
        lw=2,
        label="Sums",
    )
    _x = np.linspace(_low, _high, 500)
    _ax2.plot(_x, stats.norm.pdf(_x), color="#0072B2", lw=3, label="Unit Gaussian (no fit)")
    _std = _sums.std(ddof=1)
    if np.isfinite(_std) and _std > 0:
        _ax2.plot(
            _x,
            stats.norm.pdf(_x, _sums.mean(), _std),
            color="#D55E00",
            lw=3,
            ls="--",
            label=f"Gaussian with the mean and std of the sums (std = {_std:.2f})",
        )
    _ax2.set(
        xlabel="Sum divided by √(number of values)",
        ylabel="Probability density",
        xlim=(_low, _high),
    )
    _ax2.set_title(f"{_n_total} numbers per sum, {_size} sums")
    _ax2.legend(loc="upper right")
    _fig.tight_layout()

    _rows = ""
    for _k in (3.0, 3.5, 4.0):
        _observed = np.mean(np.abs(_sums) > _k)
        _expected = 2 * stats.norm.sf(_k)
        _rows += f"| beyond ±{_k:g} σ | {_observed:.5f} ({int(_observed * _size)}) | {_expected:.5f} ({_expected * _size:.1f}) |\n"
    _verdict = (
        mo.callout(
            mo.md(
                "The Cauchy distribution has **no finite width**, so the theorem's conditions are broken: "
                "the sums are not Gaussian however many you add, and dividing by √N does not tame them."
            ),
            kind="danger",
        )
        if n_cauchy.value > 0
        else mo.callout(
            mo.md(
                "All inputs have a finite width, so the sums approach the unit Gaussian. How fast depends "
                "on the shape of the inputs: compare the tails below."
            ),
            kind="info",
        )
    )
    mo.vstack(
        [
            _controls,
            _fig,
            _verdict,
            mo.md(
                "| fraction of the sums | observed (count) | unit Gaussian (count) |\n|---|---|---|\n"
                + _rows
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Why divide by $\sqrt{N}$? Widths add in quadrature

    Take $N$ numbers of unit width and add them. Their variances add, $\sigma^2_{sum} = N\cdot 1$, so the
    width of the sum grows as $\sqrt{N}$ (the dashed line), **not** as $N$. Pick an input distribution
    and see that the measured width follows the line for as many numbers as you add.
    """)
    return


@app.cell
def _(mo):
    quadrature_input = mo.ui.dropdown(
        ["Uniform", "Exponential"], value="Uniform", label="Input distribution"
    )
    return (quadrature_input,)


@app.cell
def _(mo, np, plt, quadrature_input):
    _rng = np.random.default_rng(1)
    _size, _n_max = 20_000, 50
    if quadrature_input.value == "Uniform":
        _numbers = (_rng.uniform(size=(_size, _n_max)) - 0.5) * np.sqrt(12)
    else:
        _numbers = _rng.exponential(size=(_size, _n_max)) - 1.0
    _n = np.arange(1, _n_max + 1)
    _width = np.cumsum(_numbers, axis=1).std(
        axis=0, ddof=1
    )  # width of the sum of the first n numbers

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(_n, _width, "o", color="#0072B2", label="Width of the sum (simulated)")
    _ax.plot(_n, np.sqrt(_n), "--", color="#D55E00", lw=2, label=r"$\sqrt{N}$")
    _ax.set(xlabel="Numbers added, N", ylabel="Width of the sum")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([quadrature_input, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to take away

    1. The Gaussian is central in statistics: sums of many different distributions converge towards it.
    2. The convergence is fairly fast for "nice" inputs. Compare the tails in block 1 for uniform and
       for exponential inputs.
    3. Widths add in quadrature, so a sum of $N$ numbers has width $\sqrt{N}$ times that of one number.
       That is why we treat uncertainties as Gaussian unless we know better.
    4. The theorem needs a **finite width**. The Cauchy distribution has none, and it never becomes Gaussian.
    """)
    return


if __name__ == "__main__":
    app.run()
