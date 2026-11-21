import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from appstat.stats import (
        accept_reject,
        box_integral,
        chi2_pvalue,
        exponential_bin_counts,
        pearson_chi2,
        two_sample_chi2,
    )

    TRANSFORM, ACCEPT, EXACT = "#0072B2", "#D55E00", "#000000"

    def histogram_panel(ax, samples, edges, expected, colour, label):
        """Histogram with Poisson errors and the exact expectation as a line, per bin."""
        counts, _ = np.histogram(samples, bins=edges)
        centres = (edges[1:] + edges[:-1]) / 2
        ax.errorbar(centres, counts, np.sqrt(counts), fmt=".", color=colour, label=label)
        ax.step(
            edges,
            np.append(expected, expected[-1]),
            where="post",
            color=EXACT,
            lw=1.5,
            label="Exact",
        )
        ax.set(xlabel="x", ylabel="Entries per bin")
        ax.legend()
        return counts

    def x_cos_x_cdf(x):
        """Integral of x cos x from 0 to x."""
        return x * np.sin(x) + np.cos(x) - 1

    def exp_cos2_cdf(x):
        """Integral of exp(-t/3) cos(t)^2 from 0 to x, using cos^2 = (1 + cos 2t) / 2."""
        a, b = 1 / 3, 2.0
        decay = np.exp(-a * np.asarray(x, dtype=float))
        return 1.5 * (1 - decay) + 0.5 * (decay * (-a * np.cos(b * x) + b * np.sin(b * x)) + a) / (
            a**2 + b**2
        )

    def fit_quality(counts, expected):
        """Chi-square probability of counts against an exact expectation (no fitted parameters)."""
        chi2, ndof = pearson_chi2(counts, expected)
        return chi2, ndof, chi2_pvalue(chi2, ndof)

    return (
        ACCEPT,
        EXACT,
        TRANSFORM,
        accept_reject,
        box_integral,
        exponential_bin_counts,
        fit_quality,
        histogram_panel,
        mo,
        np,
        plt,
        two_sample_chi2,
        chi2_pvalue,
        exp_cos2_cdf,
        x_cos_x_cdf,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Random numbers from a function: transformation and accept/reject

    A computer gives you **uniform** random numbers $r \in [0,1]$. To get numbers that follow
    another function $f(x)$, there are two methods:

    - **Transformation**: integrate $f$ to the cumulative $F(x)$, and invert it: $x = F^{-1}(r)$.
      It is 100 % efficient, but needs a function you can integrate **and** invert.
    - **Accept/reject** (von Neumann): throw points $(x, y)$ uniformly in a box around $f$, and keep the
      $x$ of those that fall below the curve. It works for any bounded function, but wastes the points that miss.
      The fraction that hit also estimates the **integral** of $f$.

    Every interactive block below has its controls directly above its plots.

    References: Cowan chapter 3, Bevington pages 81-84, the PDG review *Monte Carlo Techniques*.

    ---

    **Adapted from** the Transformation and Accept/Reject exercises by **Troels C. Petersen** (Niels Bohr Institute,
    4th of December 2023 and 29th of November 2024), from the course *Applied Statistics 2025/26*. The original is
    three exercises with example solutions. This version keeps their problems and is rewritten as an interactive
    demonstration.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The same function, two methods

    $f(x) = 2x$ on $[0,1]$. Then $F(x) = x^2$ and $F^{-1}(r) = \sqrt{r}$, so the transformation is one line.
    For accept/reject the box is $[0,1]\times[0,2]$, and half the points hit, so the integral is $1$.
    The black line is **not fitted**: it is the exact expectation. Are the two histograms statistically the same?
    The χ² between them says.
    """)
    return


@app.cell
def _(mo):
    simple_controls = mo.ui.dictionary(
        {
            "n": mo.ui.slider(1000, 100_000, 1000, value=10_000, show_value=True, label="Points"),
            "bins": mo.ui.slider(10, 100, 5, value=50, show_value=True, label="Bins"),
            "seed": mo.ui.number(0, 9999, 1, value=42, label="Random seed"),
        }
    )
    return (simple_controls,)


@app.cell
def _(
    ACCEPT,
    TRANSFORM,
    accept_reject,
    box_integral,
    chi2_pvalue,
    fit_quality,
    histogram_panel,
    mo,
    np,
    plt,
    simple_controls,
    two_sample_chi2,
):
    _n, _bins = simple_controls["n"].value, simple_controls["bins"].value
    _rng = np.random.default_rng(simple_controls["seed"].value)
    _by_transformation = np.sqrt(_rng.uniform(size=_n))
    _by_accept_reject, _tries = accept_reject(lambda x: 2 * x, 0.0, 1.0, 2.0, _n, _rng)

    _edges = np.linspace(0, 1, _bins + 1)
    _expected = _n * (_edges[1:] ** 2 - _edges[:-1] ** 2)
    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
    _a = histogram_panel(_left, _by_transformation, _edges, _expected, TRANSFORM, "Transformation")
    _b = histogram_panel(_right, _by_accept_reject, _edges, _expected, ACCEPT, "Accept/reject")
    _fig.tight_layout()

    _integral, _error = box_integral(_n, _tries, 2.0)
    _chi2_ab, _ndof_ab = two_sample_chi2(_a, _b)
    _p = {
        name: fit_quality(c, _expected)[2]
        for name, c in (("transformation", _a), ("accept/reject", _b))
    }
    mo.vstack(
        [
            mo.hstack(
                [simple_controls["n"], simple_controls["bins"], simple_controls["seed"]],
                justify="start",
                gap=2,
            ),
            _fig,
            mo.callout(
                mo.md(
                    f"Accept/reject efficiency **{_n / _tries:.1%}**, integral **{_integral:.3f} ± {_error:.3f}** (exact: 1). "
                    f"χ² against the exact line: p = {_p['transformation']:.2f} (transformation), "
                    f"{_p['accept/reject']:.2f} (accept/reject). "
                    f"The two histograms against each other: χ²/Ndof = {_chi2_ab:.0f}/{_ndof_ab}, "
                    f"p = {chi2_pvalue(_chi2_ab, _ndof_ab):.2f}."
                ),
                kind="success",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. A cubic, and moving the lower edge

    $f(x) = x^3$ on $[a, C]$, with $C$ chosen so the area is 1: $C = (4 + a^4)^{1/4}$. Then
    $F(x) = (x^4 - a^4)/4$ and $F^{-1}(r) = (4r + a^4)^{1/4}$. Accept/reject needs a ceiling at the top
    of the function, $C^3$. Move the lower edge $a$ and see what happens to the **efficiency**: the box
    gets taller compared with the area under the curve.
    """)
    return


@app.cell
def _(mo):
    cubic_controls = mo.ui.dictionary(
        {
            "a": mo.ui.slider(0.0, 3.0, 0.1, value=0.0, show_value=True, label="Lower edge a"),
            "n": mo.ui.slider(1000, 100_000, 1000, value=10_000, show_value=True, label="Points"),
            "seed": mo.ui.number(0, 9999, 1, value=42, label="Random seed"),
        }
    )
    return (cubic_controls,)


@app.cell
def _(
    ACCEPT,
    EXACT,
    TRANSFORM,
    accept_reject,
    box_integral,
    cubic_controls,
    fit_quality,
    mo,
    np,
    plt,
):
    _a, _n = cubic_controls["a"].value, cubic_controls["n"].value
    _c = (4 + _a**4) ** 0.25
    _rng = np.random.default_rng(cubic_controls["seed"].value)
    _by_transformation = (4 * _rng.uniform(size=_n) + _a**4) ** 0.25
    _by_accept_reject, _tries = accept_reject(lambda x: x**3, _a, _c, _c**3, _n, _rng)

    _edges = np.linspace(_a, _c, 61)
    _expected = _n * (_edges[1:] ** 4 - _edges[:-1] ** 4) / 4
    _fig, _ax = plt.subplots(figsize=(13, 3.8))
    _centres = (_edges[1:] + _edges[:-1]) / 2
    _counts = {}
    for _name, _samples, _colour in (
        ("Transformation", _by_transformation, TRANSFORM),
        ("Accept/reject", _by_accept_reject, ACCEPT),
    ):
        _counts[_name], _ = np.histogram(_samples, bins=_edges)
        _ax.step(
            _edges,
            np.append(_counts[_name], _counts[_name][-1]),
            where="post",
            color=_colour,
            label=_name,
        )
    _ax.step(
        _edges,
        np.append(_expected, _expected[-1]),
        where="post",
        color=EXACT,
        ls="--",
        label="Exact",
    )
    _ax.set(xlabel="x", ylabel="Entries per bin", ylim=(0, None))
    _ax.legend(loc="upper left")
    _fig.tight_layout()

    _integral, _error = box_integral(_n, _tries, (_c - _a) * _c**3)
    _p = {name: fit_quality(c, _expected)[2] for name, c in _counts.items()}
    mo.vstack(
        [
            mo.hstack(list(cubic_controls.values()), justify="start", gap=2),
            _fig,
            mo.callout(
                mo.md(
                    f"Range [{_a:.1f}, {_c:.2f}]. Accept/reject efficiency **{_n / _tries:.1%}** "
                    f"(exact: {1 / ((_c - _a) * _c**3):.1%}), integral {_integral:.3f} ± {_error:.3f} (exact: 1). "
                    f"χ² against the exact curve: p = {_p['Transformation']:.2f} (transformation), "
                    f"{_p['Accept/reject']:.2f} (accept/reject)."
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Three harder problems

    Pick the method the function allows:

    1. $f(x) \propto e^{-x/3}$ on $[0,\infty)$. The range is infinite, so accept/reject has no box.
       **Transformation**: $x = -3\ln r$.
    2. $g(x) \propto x\cos x$ on $[0,\pi/2]$. It integrates, but the integral cannot be inverted. It is bounded, so
       **accept/reject** works. Its maximum is $0.561$: set the ceiling lower and the method **fails** (the
       box does not contain the function), higher and it is just less efficient.
    3. $h(x) \propto e^{-x/3}\cos^2 x$ on $[0,\infty)$. Neither works alone, so **combine** them: draw $x$ from
       the exponential (transformation), then keep it with a random height below $e^{-x/3}$ (accept/reject).
       The accepted fraction times the area of the exponential, 3, is the integral.
    """)
    return


@app.cell
def _(mo):
    hard_problem = mo.ui.dropdown(
        {"1: exponential": 1, "2: x cos x": 2, "3: exponential times cos²": 3},
        value="2: x cos x",
        label="Problem",
    )
    hard_ceiling = mo.ui.slider(
        0.3, 3.0, 0.1, value=1.0, show_value=True, label="Ceiling (problem 2)"
    )
    hard_n = mo.ui.slider(1000, 100_000, 1000, value=10_000, show_value=True, label="Points")
    hard_seed = mo.ui.number(0, 9999, 1, value=42, label="Random seed")
    return hard_ceiling, hard_n, hard_problem, hard_seed


@app.cell
def _(
    ACCEPT,
    EXACT,
    TRANSFORM,
    accept_reject,
    box_integral,
    exp_cos2_cdf,
    exponential_bin_counts,
    fit_quality,
    hard_ceiling,
    hard_n,
    hard_problem,
    hard_seed,
    mo,
    np,
    plt,
    x_cos_x_cdf,
):
    _problem, _n = hard_problem.value, hard_n.value
    _rng = np.random.default_rng(hard_seed.value)
    _note = None

    if _problem == 1:
        _samples, _edges = -3.0 * np.log(_rng.uniform(size=_n)), np.linspace(0, 20, 81)
        _expected = exponential_bin_counts(_edges, _n, 3.0)
        _text, _colour = "Transformation, 100 % efficient. ", TRANSFORM
    elif _problem == 2:
        _edges = np.linspace(0, np.pi / 2, 61)
        _expected = _n * np.diff(x_cos_x_cdf(_edges)) / x_cos_x_cdf(np.pi / 2)
        _colour = ACCEPT
        try:
            _samples, _tries = accept_reject(
                lambda x: x * np.cos(x), 0.0, np.pi / 2, hard_ceiling.value, _n, _rng
            )
            _integral, _error = box_integral(_n, _tries, np.pi / 2 * hard_ceiling.value)
            _text = (
                f"Efficiency **{_n / _tries:.1%}**, integral **{_integral:.4f} ± {_error:.4f}** "
                f"(exact: {np.pi / 2 - 1:.4f}). "
            )
        except ValueError:
            _samples, _note = (
                np.array([]),
                "The ceiling is **below** the maximum 0.561 of x cos x, so the box misses the top.",
            )
    else:
        _tries = 4 * _n
        _x = -3.0 * np.log(_rng.uniform(size=_tries))
        _kept = _x[_rng.uniform(0.0, np.exp(-_x / 3)) < np.exp(-_x / 3) * np.cos(_x) ** 2]
        _samples, _edges, _colour = _kept[:_n], np.linspace(0, 20, 81), ACCEPT
        _total = exp_cos2_cdf(1e3)  # e^(-333) is zero, and cos(inf) is not defined
        _expected = _n * np.diff(exp_cos2_cdf(_edges)) / _total
        _integral, _error = box_integral(len(_kept), _tries, 3.0)
        _text = f"Efficiency **{len(_kept) / _tries:.1%}**, integral **{_integral:.3f} ± {_error:.3f}** (exact: {_total:.3f}). "

    _fig, _ax = plt.subplots(figsize=(13, 3.8))
    if len(_samples):
        _counts, _ = np.histogram(_samples, bins=_edges)
        _centres = (_edges[1:] + _edges[:-1]) / 2
        _ax.errorbar(_centres, _counts, np.sqrt(_counts), fmt=".", color=_colour, label="Generated")
        _ax.step(
            _edges,
            np.append(_expected, _expected[-1]),
            where="post",
            color=EXACT,
            lw=1.5,
            label="Exact",
        )
        _ax.legend()
        _chi2, _ndof, _p = fit_quality(_counts, _expected)
        _note = f"{_text}χ² against the exact curve: {_chi2:.0f}/{_ndof}, p = {_p:.2f}."
    _ax.set(xlabel="x", ylabel="Entries per bin", ylim=(0, None))
    _fig.tight_layout()
    mo.vstack(
        [
            mo.hstack([hard_problem, hard_ceiling], justify="start", gap=2),
            mo.hstack([hard_n, hard_seed], justify="start", gap=2),
            _fig,
            mo.callout(mo.md(_note), kind="success" if len(_samples) else "danger"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to take away

    1. **Transformation** is the best method when it applies: 100 % efficient. It needs $F$ and $F^{-1}$.
    2. **Accept/reject** works for any bounded function and needs no integral. The accepted fraction **is** an
       estimate of the integral, with a binomial uncertainty.
    3. A ceiling below the maximum **silently** gives the wrong distribution, which is why the code refuses it.
       A ceiling far above it is correct, but wastes points.
    4. For infinite ranges, **combine** the two: draw from a function you can invert that lies above the one you want.
    5. Compare generated numbers with the exact expectation with a χ², not by eye.
    """)
    return


if __name__ == "__main__":
    app.run()
