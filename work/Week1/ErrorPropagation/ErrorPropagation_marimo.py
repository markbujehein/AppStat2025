import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import sympy as sp

    from appstat.stats import propagate_mc

    return mo, np, plt, propagate_mc, sp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Error propagation, two ways

    A class estimates by eye that the table in Auditorium A is $L = 3.5 \pm 0.4$ m long and
    $W = 0.8 \pm 0.2$ m wide. How uncertain are the **perimeter**, the **area** and the
    **diagonal**?

    There are two ways to answer, and this page runs both side by side:

    1. **Analytically**: differentiate the formula and use the error propagation formula.
    2. **By simulation**: draw many (L, W) pairs that fit the uncertainties, compute the
       quantity for each pair, and look at the spread.

    Change the inputs, the correlation and the quantity, and see when the two agree and
    when they do not.

    References: Barlow pp. 48-61, Bevington pp. 36-48.

    ---

    **Adapted from** `ErrorPropagation_original.ipynb` (10th of November 2025) by
    **Troels C. Petersen** (Niels Bohr Institute), from the course *Applied Statistics
    2025/26*. The original is an exercise. This version keeps its table example and its
    analytic-then-simulation approach, and is rewritten as an interactive demonstration.
    """)
    return


@app.cell
def _(sp):
    L, W = sp.symbols("L W")
    expressions = {
        "Perimeter  P = 2L + 2W": 2 * L + 2 * W,
        "Area  A = L W": L * W,
        "Diagonal  D = sqrt(L² + W²)": sp.sqrt(L**2 + W**2),
        "Wiggle  y = 1.1 + sin(20 L)": 1.1 + sp.sin(20 * L),
    }
    # Numerical versions of each formula and of its two partial derivatives
    functions = {name: sp.lambdify((L, W), e, "numpy") for name, e in expressions.items()}
    gradients = {
        name: sp.lambdify((L, W), [sp.diff(e, L), sp.diff(e, W)], "numpy")
        for name, e in expressions.items()
    }
    return L, W, expressions, functions, gradients


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Choose the inputs
    """)
    return


@app.cell
def _(expressions, mo):
    mean_length = mo.ui.slider(0.5, 6.0, 0.05, value=3.5, label="Length L [m]")
    sigma_length = mo.ui.slider(0.01, 1.0, 0.01, value=0.4, label="Uncertainty on L [m]")
    mean_width = mo.ui.slider(0.2, 3.0, 0.05, value=0.8, label="Width W [m]")
    sigma_width = mo.ui.slider(0.01, 1.0, 0.01, value=0.2, label="Uncertainty on W [m]")
    rho = mo.ui.slider(-0.95, 0.95, 0.05, value=0.0, label="Correlation ρ(L, W)")
    quantity = mo.ui.dropdown(list(expressions), value=list(expressions)[0], label="Quantity")
    n_samples = mo.ui.dropdown(
        {"1 000": 1_000, "10 000": 10_000, "100 000": 100_000, "1 000 000": 1_000_000},
        value="100 000",
        label="Simulated pairs",
    )
    seed = mo.ui.number(0, 9999, 1, value=42, label="Random seed")
    mo.hstack(
        [
            mo.vstack([mean_length, sigma_length, mean_width, sigma_width]),
            mo.vstack([rho, quantity, n_samples, seed]),
        ],
        widths="equal",
    )
    return (
        mean_length,
        mean_width,
        n_samples,
        quantity,
        rho,
        seed,
        sigma_length,
        sigma_width,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The analytic answer

    For $y = f(L, W)$ with the partial derivatives collected in a vector
    $\mathbf{g} = (\partial f/\partial L,\ \partial f/\partial W)$, and the covariance matrix
    $C$ of the inputs (variances on the diagonal, $\rho\,\sigma_L\sigma_W$ off it):

    $$\sigma_y^2 = \mathbf{g}^T C\, \mathbf{g}
    = \left(\tfrac{\partial f}{\partial L}\sigma_L\right)^2
    + \left(\tfrac{\partial f}{\partial W}\sigma_W\right)^2
    + 2\,\tfrac{\partial f}{\partial L}\tfrac{\partial f}{\partial W}\,\rho\,\sigma_L\sigma_W$$

    The last term is the one that vanishes when the inputs are uncorrelated.
    """)
    return


@app.cell
def _(
    functions,
    gradients,
    mean_length,
    mean_width,
    np,
    quantity,
    rho,
    sigma_length,
    sigma_width,
):
    means = np.array([mean_length.value, mean_width.value])
    sigmas = np.array([sigma_length.value, sigma_width.value])
    off_diagonal = rho.value * sigmas[0] * sigmas[1]
    covariance = np.array([[sigmas[0] ** 2, off_diagonal], [off_diagonal, sigmas[1] ** 2]])

    function = functions[quantity.value]
    gradient = np.array(gradients[quantity.value](*means), dtype=float)

    y_central = float(function(*means))
    sigma_analytic = float(np.sqrt(gradient @ covariance @ gradient))
    sigma_analytic_no_rho = float(np.sqrt(np.sum((gradient * sigmas) ** 2)))
    return (
        covariance,
        function,
        means,
        sigma_analytic,
        sigma_analytic_no_rho,
        sigmas,
        y_central,
    )


@app.cell(hide_code=True)
def _(L, W, expressions, mo, quantity, sp):
    _expr = expressions[quantity.value]
    mo.md(
        r"The partial derivatives of this quantity are $\partial f/\partial L = "
        + sp.latex(sp.diff(_expr, L))
        + r"$ and $\partial f/\partial W = "
        + sp.latex(sp.diff(_expr, W))
        + r"$."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. The simulated answer

    Draw many $(L, W)$ pairs from a two-dimensional Gaussian with the chosen means,
    uncertainties and correlation. Compute $y$ for every pair. The standard deviation of the
    $y$ values is the uncertainty. Nothing is assumed about the shape of the result, so if
    it is not Gaussian you can see it.
    """)
    return


@app.cell
def _(covariance, function, means, n_samples, np, propagate_mc, seed, sigmas):
    rng = np.random.default_rng(seed.value)
    inputs = rng.multivariate_normal(means, covariance, size=n_samples.value)  # (N, 2)
    results = function(inputs[:, 0], inputs[:, 1])

    mc_mean = results.mean()
    mc_sigma = results.std(ddof=1)

    # The helper in appstat draws L and W independently, so it knows nothing about rho
    sigma_independent_mc = propagate_mc(
        function, means, sigmas, n_samples=n_samples.value, seed=seed.value
    )
    return inputs, mc_mean, mc_sigma, results, sigma_independent_mc


@app.cell(hide_code=True)
def _(
    mc_mean,
    mc_sigma,
    mo,
    rho,
    sigma_analytic,
    sigma_analytic_no_rho,
    sigma_independent_mc,
    y_central,
):
    _agree = abs(mc_sigma - sigma_analytic) < 0.05 * mc_sigma
    _verdict = (
        mo.callout(
            mo.md("The formula and the simulation **agree** within 5 %."),
            kind="success",
        )
        if _agree
        else mo.callout(
            mo.md(
                "The formula and the simulation **disagree**. The formula replaces the function "
                "by a straight line around the central value. That fails when the function "
                "bends noticeably across the spread of the inputs, as in the wiggle."
            ),
            kind="warn",
        )
    )
    mo.vstack(
        [
            mo.md(rf"""
            | | uncertainty on y |
            |---|---|
            | error propagation formula, with ρ = {rho.value:.2f} | {sigma_analytic:.4f} |
            | simulation, with ρ = {rho.value:.2f} | {mc_sigma:.4f} |
            | error propagation formula, ρ ignored | {sigma_analytic_no_rho:.4f} |
            | simulation with `appstat.stats.propagate_mc`, ρ ignored | {sigma_independent_mc:.4f} |

            Central value: formula {y_central:.4f}, simulation mean {mc_mean:.4f}.
            """),
            _verdict,
        ]
    )
    return


@app.cell
def _(inputs, mc_sigma, np, plt, results, sigma_analytic, y_central):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    _ax1.hist2d(inputs[:, 0], inputs[:, 1], bins=80, cmin=1)
    _ax1.set(xlabel="Length L [m]", ylabel="Width W [m]", title="The simulated inputs")

    _low, _high = np.percentile(results, [0.1, 99.9])
    _ax2.hist(
        results,
        bins=100,
        range=(_low, _high),
        density=True,
        histtype="step",
        linewidth=2,
        label=f"Simulation (σ = {mc_sigma:.3f})",
    )
    _x = np.linspace(_low, _high, 500)
    _s = max(sigma_analytic, 1e-12)
    _ax2.plot(
        _x,
        np.exp(-0.5 * ((_x - y_central) / _s) ** 2) / (_s * np.sqrt(2 * np.pi)),
        "r",
        label=f"Error propagation formula (σ = {sigma_analytic:.3f})",
    )
    _ax2.set(xlabel="y", ylabel="Probability density", title="The resulting distribution")
    _ax2.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to take away

    1. Uncertainties propagate in **two ways**: analytically, by differentiating the formula,
       and numerically, by simulating many inputs.
    2. The analytic method is fast and transparent, but it assumes a straight line over the
       range of the inputs. The simulation is robust and shows non-Gaussian shapes, but it is a
       black box and noisy.
    3. Correlations between inputs change the answer. Ignoring them (as
       `appstat.stats.propagate_mc` does) can over- or underestimate the uncertainty.
    4. Using both and comparing is the best way to catch a mistake in either.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Additional work

    Four more investigations. Each answers with a plot that redraws when you move its controls.
    **Orange is always the error propagation formula, blue is always the simulation.** Sections A and D
    have their own sliders, placed directly above their plots.
    """)
    return


@app.cell
def _(np):
    FORMULA = "#D55E00"
    SIMULATION = "#0072B2"

    def gauss(x, mean, sigma):
        return np.exp(-0.5 * ((x - mean) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    return FORMULA, SIMULATION, gauss


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A. A new quantity: the aspect ratio $L/W$

    The formula needs the two partial derivatives, and SymPy supplies them. It replaces $L/W$ by its
    **tangent line** at the central value. Drag the slider: the wider the uncertainty on $W$, the
    further the curve bends away from the tangent (left), and the simulated distribution slides away
    from the formula's Gaussian (right). The inputs are uncorrelated here, and $L$ and its
    uncertainty come from the inputs above.
    """)
    return


@app.cell
def _(mo):
    ratio_sigma_width = mo.ui.slider(
        0.01, 1.0, 0.01, value=0.2, show_value=True, full_width=True, label="Uncertainty on W [m]"
    )
    ratio_sigma_width
    return (ratio_sigma_width,)


@app.cell
def _(
    FORMULA,
    L,
    SIMULATION,
    W,
    gauss,
    mean_length,
    mean_width,
    mo,
    n_samples,
    np,
    plt,
    ratio_sigma_width,
    seed,
    sigma_length,
    sp,
):
    _L0, _W0 = mean_length.value, mean_width.value
    _sigmas = np.array([sigma_length.value, ratio_sigma_width.value])
    _ratio = L / W
    _g = np.array(
        sp.lambdify((L, W), [sp.diff(_ratio, L), sp.diff(_ratio, W)], "numpy")(_L0, _W0),
        dtype=float,
    )
    _central = _L0 / _W0
    _formula = float(np.sqrt(np.sum((_g * _sigmas) ** 2)))
    _pairs = np.random.default_rng(seed.value).normal(
        [_L0, _W0], _sigmas, size=(n_samples.value, 2)
    )
    _values = _pairs[:, 0] / _pairs[:, 1]
    _sim = float(_values.std(ddof=1))

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    _w = np.linspace(max(0.05, _W0 - 3 * _sigmas[1]), _W0 + 3 * _sigmas[1], 300)
    _ax1.axvspan(_W0 - _sigmas[1], _W0 + _sigmas[1], color="0.88", label="W ± its uncertainty")
    _ax1.plot(_w, _L0 / _w, "k", lw=2, label="L / W")
    _ax1.plot(
        _w,
        _central + _g[1] * (_w - _W0),
        color=FORMULA,
        ls="--",
        lw=2,
        label="Tangent line used by the formula",
    )
    _ax1.set(
        xlim=(_w[0], _w[-1]),
        ylim=(0, 3 * _central),
        xlabel="Width W [m]",
        ylabel="L / W",
        title="The formula replaces the curve by its tangent",
    )
    _ax1.legend(loc="upper right")

    _low, _high = np.percentile(_values, [0.5, 99.5])
    _grid = np.linspace(_low, _high, 400)
    _ax2.hist(
        _values,
        bins=100,
        range=(_low, _high),
        density=True,
        color=SIMULATION,
        alpha=0.35,
        label=f"Simulation (σ = {_sim:.3f})",
    )
    _ax2.plot(
        _grid,
        gauss(_grid, _central, _formula),
        color=FORMULA,
        lw=2,
        label=f"Formula (σ = {_formula:.3f})",
    )
    _ax2.axvline(_values.mean(), color=SIMULATION, lw=2)
    _ax2.axvline(_central, color=FORMULA, ls="--", lw=2)
    _ax2.set(xlabel="L / W", ylabel="Probability density", title="The resulting distribution")
    _ax2.legend()
    _fig.tight_layout()

    _agree = abs(_sim - _formula) < 0.05 * _sim
    mo.vstack(
        [
            _fig,
            mo.callout(
                mo.md(
                    f"Central value: formula **{_central:.3f}**, simulation mean **{_values.mean():.3f}**. "
                    + (
                        "The formula and the simulation **agree** within 5 %."
                        if _agree
                        else "They **disagree**: the curve bends across the uncertainty on W."
                    )
                ),
                kind="success" if _agree else "warn",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## B. Zero correlation, but not independent

    Let $z$ be a standard normal number and set $L = \mu_L + \sigma_L z$ and
    $W = \mu_W + \sigma_W (z^2 - 1)/\sqrt{2}$. Then $W$ has the chosen mean and standard deviation, and
    since $E[z^3] = 0$ the **linear correlation is exactly zero**. Yet $W$ is completely determined by
    $L$: the left plot shows the pairs lie on a parabola. The error propagation formula only knows about
    linear correlation, so a straight sum like the perimeter is fine (middle), but the area is not (right).
    These plots use the inputs and the number of pairs chosen above.
    """)
    return


@app.cell
def _(
    FORMULA,
    SIMULATION,
    functions,
    gauss,
    gradients,
    mean_length,
    mean_width,
    means,
    mo,
    n_samples,
    np,
    plt,
    seed,
    sigma_length,
    sigma_width,
    sigmas,
):
    _rng = np.random.default_rng(seed.value + 1)
    _z = _rng.standard_normal(n_samples.value)
    _length = mean_length.value + sigma_length.value * _z
    _width = mean_width.value + sigma_width.value * (_z**2 - 1) / np.sqrt(2)
    _rho = np.corrcoef(_length, _width)[0, 1]

    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.3))
    _axes[0].plot(_length[:3000], _width[:3000], ".", ms=3, alpha=0.4, color=SIMULATION)
    _axes[0].set(
        xlabel="Length L [m]",
        ylabel="Width W [m]",
        title=f"Correlation {_rho:+.3f}, yet W follows L",
    )

    for _ax, _name in zip(_axes[1:], list(functions)[:2], strict=True):  # perimeter, area
        _values = functions[_name](_length, _width)
        _central = float(functions[_name](*means))
        _formula = float(
            np.sqrt(np.sum((np.array(gradients[_name](*means), dtype=float) * sigmas) ** 2))
        )
        _sim = float(_values.std(ddof=1))
        _low, _high = np.percentile(_values, [0.5, 99.5])
        _grid = np.linspace(_low, _high, 400)
        _ax.hist(
            _values,
            bins=100,
            range=(_low, _high),
            density=True,
            color=SIMULATION,
            alpha=0.35,
            label=f"Simulation (σ = {_sim:.3f})",
        )
        _ax.plot(
            _grid,
            gauss(_grid, _central, _formula),
            color=FORMULA,
            lw=2,
            label=f"Formula (σ = {_formula:.3f})",
        )
        _ax.set(xlabel=_name.split()[0], ylabel="Probability density", title=_name.split()[0])
        _ax.legend()
    _fig.tight_layout()
    mo.vstack([_fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## C. How many simulated pairs are enough?

    The simulated uncertainty is itself a random number. For the quantity chosen above, call
    `propagate_mc` with 50 different seeds at each sample size and plot how much the result scatters.
    It should fall as $1/\sqrt{N}$ (the red dashed line).
    """)
    return


@app.cell
def _(function, means, np, plt, propagate_mc, sigmas):
    _sizes = np.array([100, 1_000, 10_000, 100_000])
    _spread = [
        np.std(
            [propagate_mc(function, means, sigmas, n_samples=n, seed=s) for s in range(50)], ddof=1
        )
        for n in _sizes
    ]
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.loglog(_sizes, _spread, "ko-", label="Scatter of the simulated uncertainty")
    _ax.loglog(_sizes, _spread[0] * np.sqrt(_sizes[0] / _sizes), "r--", label=r"$1/\sqrt{N}$")
    _ax.set(xlabel="Simulated pairs N", ylabel="Scatter over 50 seeds")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## D. A harder function

    $$y = \ln\left((L\tan W)^2 + \sqrt{\frac{L - W}{\cos W + 1 + L}}\right)$$

    Nobody wants to differentiate this by hand, so SymPy does it. Which result do you trust more,
    the formula or the simulation? Move the four sliders (for this test only, inputs uncorrelated) and
    watch both plots. Left: the simulated input pairs. Pairs where the function is undefined are marked
    with a red cross and left out of the right plot, and $W = \pi/2$ is where $\tan W$ blows up.
    Right: the distribution of $y$ against the formula's Gaussian.
    """)
    return


@app.cell
def _(mo):
    d_length = mo.ui.slider(0.5, 6.0, 0.05, value=3.5, show_value=True, label="Length L [m]")
    d_sigma_length = mo.ui.slider(
        0.01, 1.0, 0.01, value=0.4, show_value=True, label="Uncertainty on L [m]"
    )
    d_width = mo.ui.slider(0.2, 3.0, 0.05, value=0.8, show_value=True, label="Width W [m]")
    d_sigma_width = mo.ui.slider(
        0.01, 1.0, 0.01, value=0.2, show_value=True, label="Uncertainty on W [m]"
    )
    mo.hstack(
        [mo.vstack([d_length, d_sigma_length]), mo.vstack([d_width, d_sigma_width])],
        widths="equal",
    )
    return d_length, d_sigma_length, d_sigma_width, d_width


@app.cell
def _(
    FORMULA,
    L,
    SIMULATION,
    W,
    d_length,
    d_sigma_length,
    d_sigma_width,
    d_width,
    gauss,
    mo,
    n_samples,
    np,
    plt,
    seed,
    sp,
):
    _means = np.array([d_length.value, d_width.value])
    _sigmas = np.array([d_sigma_length.value, d_sigma_width.value])
    _y = sp.log((L * sp.tan(W)) ** 2 + sp.sqrt((L - W) / (sp.cos(W) + 1 + L)))
    _f = sp.lambdify((L, W), _y, "numpy")
    _g = np.array(
        sp.lambdify((L, W), [sp.diff(_y, L), sp.diff(_y, W)], "numpy")(*_means), dtype=float
    )
    _formula = float(np.sqrt(np.sum((_g * _sigmas) ** 2)))
    _central = float(_f(*_means))
    _pairs = np.random.default_rng(seed.value).normal(_means, _sigmas, size=(n_samples.value, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        _values = _f(_pairs[:, 0], _pairs[:, 1])
    _ok = np.isfinite(_values)

    mo.stop(
        not _ok.any(),
        mo.callout(
            mo.md("The function is undefined for every simulated pair. Change the sliders."),
            kind="danger",
        ),
    )

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    _show = np.arange(min(len(_values), 5000))
    _ax1.plot(
        _pairs[_show][_ok[_show], 0],
        _pairs[_show][_ok[_show], 1],
        ".",
        ms=3,
        alpha=0.4,
        color=SIMULATION,
        label="y defined",
    )
    _ax1.plot(
        _pairs[_show][~_ok[_show], 0],
        _pairs[_show][~_ok[_show], 1],
        "x",
        ms=5,
        color="tab:red",
        label=f"y undefined ({(~_ok).mean():.1%})",
    )
    _ax1.axhline(np.pi / 2, color="0.5", ls=":", label=r"$W = \pi/2$")
    _ax1.set(xlabel="Length L [m]", ylabel="Width W [m]", title="The simulated inputs")
    _ax1.legend(loc="upper right")

    _low, _high = np.percentile(_values[_ok], [0.5, 99.5])
    _grid = np.linspace(_low, _high, 400)
    _sim = float(np.std(_values[_ok], ddof=1))
    _ax2.hist(
        _values[_ok],
        bins=100,
        range=(_low, _high),
        density=True,
        color=SIMULATION,
        alpha=0.35,
        label=f"Simulation (σ = {_sim:.3f})",
    )
    if np.isfinite(_formula) and np.isfinite(_central) and _formula > 0:
        _ax2.plot(
            _grid,
            gauss(_grid, _central, _formula),
            color=FORMULA,
            lw=2,
            label=f"Formula (σ = {_formula:.3f})",
        )
    _ax2.set(xlabel="y", ylabel="Probability density", title="The resulting distribution")
    _ax2.legend()
    _fig.tight_layout()

    _agree = abs(_sim - _formula) < 0.05 * _sim
    mo.vstack(
        [
            _fig,
            mo.callout(
                mo.md(
                    f"Uncertainty on y: formula **{_formula:.4f}**, simulation **{_sim:.4f}**. "
                    + ("They **agree** within 5 %." if _agree else "They **disagree**.")
                ),
                kind="success" if _agree else "warn",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
