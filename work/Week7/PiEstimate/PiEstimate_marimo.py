import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from appstat.montecarlo import estimate_ball_volume, unit_ball_volume

    INSIDE, OUTSIDE, HIST = "#0072B2", "#D55E00", "#009E73"

    def run_experiments(n_experiments, n_points, rng):
        """Estimates of pi, from ``n_experiments`` experiments of ``n_points`` random points."""
        return np.array([estimate_ball_volume(2, n_points, rng)[0] for _ in range(n_experiments)])

    return (
        HIST,
        INSIDE,
        OUTSIDE,
        estimate_ball_volume,
        mo,
        np,
        plt,
        run_experiments,
        unit_ball_volume,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Estimating π, and volumes, with random points

    Throw random points in the square $[-1,1]\times[-1,1]$. The fraction $f$ that lands inside the unit circle is
    $\pi/4$, so $\pi \approx 4f$. Each point is inside or outside, so $f = n/N$ is a **binomial** fraction with
    uncertainty $\sqrt{f(1-f)/N}$, and

    $$\sigma_\pi = 4\sqrt{\frac{f(1-f)}{N}} \approx \frac{1.64}{\sqrt{N}}$$

    The method shines in many dimensions, where the volume of the unit ball has no simple formula to remember.

    Every interactive block below has its controls directly above its plots.

    ---

    **Adapted from** the Estimating π exercise by **Troels C. Petersen** (Niels Bohr Institute), from the course
    *Applied Statistics 2025/26*, after Bevington pages 75-78. The original is an exercise with no example solution.
    This version is written from its text and questions, and checked against the exact values $\pi$ and
    $\pi^{d/2}/\Gamma(d/2+1)$. It is not an official solution.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. One experiment

    Points inside the circle are blue, outside orange. Add points and watch the uncertainty fall as $1/\sqrt{N}$.
    """)
    return


@app.cell
def _(mo):
    one_points = mo.ui.slider(
        100, 20_000, 100, value=1000, show_value=True, label="Points in the experiment"
    )
    one_seed = mo.ui.number(0, 9999, 1, value=42, label="Random seed")
    return one_points, one_seed


@app.cell
def _(INSIDE, OUTSIDE, mo, np, one_points, one_seed, plt):
    _rng = np.random.default_rng(one_seed.value)
    _xy = _rng.uniform(-1, 1, (one_points.value, 2))
    _inside = (_xy**2).sum(axis=1) < 1
    _f = _inside.mean()
    _fig, _ax = plt.subplots(figsize=(5.5, 5.5))
    _ax.plot(*_xy[_inside].T, ".", color=INSIDE, ms=3)
    _ax.plot(*_xy[~_inside].T, ".", color=OUTSIDE, ms=3)
    _ax.add_artist(plt.Circle((0, 0), 1, color="k", fill=False))
    _ax.set(xlim=(-1, 1), ylim=(-1, 1), aspect="equal")
    _fig.tight_layout()
    _error = 4 * np.sqrt(_f * (1 - _f) / one_points.value)
    mo.vstack(
        [
            mo.hstack([one_points, one_seed], justify="start", gap=2),
            _fig,
            mo.callout(
                mo.md(
                    f"{_inside.sum()} of {one_points.value} inside. π ≈ 4f = **{4 * _f:.4f} ± {_error:.4f}**, "
                    f"which is {(4 * _f - np.pi) / _error:+.1f} σ from the true value."
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Many experiments

    Repeat the experiment. The estimates scatter in a **Gaussian** around π, with the width you predicted. And it makes
    **no difference** whether you do 100 experiments of 1000 points or one of 100 000: only the total counts.
    """)
    return


@app.cell
def _(mo):
    many_experiments = mo.ui.slider(10, 1000, 10, value=100, show_value=True, label="Experiments")
    many_points = mo.ui.slider(
        100, 5000, 100, value=1000, show_value=True, label="Points per experiment"
    )
    return many_experiments, many_points


@app.cell
def _(HIST, many_experiments, many_points, mo, np, plt, run_experiments):
    _pi = run_experiments(many_experiments.value, many_points.value, np.random.default_rng(1))
    _expected = 4 * np.sqrt(np.pi / 4 * (1 - np.pi / 4) / many_points.value)
    _mean, _std = _pi.mean(), _pi.std(ddof=1)
    _fig, _ax = plt.subplots(figsize=(13, 3.6))
    _ax.hist(_pi, 25, color=HIST, alpha=0.8)
    _ax.axvline(np.pi, color="k", ls="--", label="π")
    _ax.set(xlabel="Estimate of π", ylabel="Experiments")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack(
        [
            mo.hstack([many_experiments, many_points], justify="start", gap=2),
            _fig,
            mo.callout(
                mo.md(
                    f"Width of the estimates **{_std:.4f}**, expected {_expected:.4f}. "
                    f"Mean of all **{_mean:.4f} ± {_std / np.sqrt(many_experiments.value):.4f}**, "
                    f"which has the precision of {many_experiments.value * many_points.value:,} points."
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Volumes in higher dimensions

    The same method for a ball in $d$ dimensions, with points in the cube $[-1,1]^d$. The **exact** volume is
    $\pi^{d/2}/\Gamma(d/2+1)$: $4\pi/3$ for $d=3$, $\pi^2/2$ for $d=4$, $\pi^3/6$ for $d=6$. The fraction inside
    **falls fast** with $d$, so the relative uncertainty grows: the cube gets far bigger than the ball.
    """)
    return


@app.cell
def _(mo):
    dim_points = mo.ui.slider(1000, 200_000, 1000, value=50_000, show_value=True, label="Points")
    dim_max = mo.ui.slider(2, 12, 1, value=10, show_value=True, label="Highest dimension")
    return dim_max, dim_points


@app.cell
def _(HIST, dim_max, dim_points, estimate_ball_volume, mo, np, plt, unit_ball_volume):
    _dims = np.arange(1, dim_max.value + 1)
    _rng = np.random.default_rng(3)
    _estimates = [estimate_ball_volume(int(d), dim_points.value, _rng) for d in _dims]
    _volume = np.array([e[0] for e in _estimates])
    _error = np.array([e[1] for e in _estimates])
    _exact = np.array([unit_ball_volume(int(d)) for d in _dims])
    _fig, (_ax, _axr) = plt.subplots(1, 2, figsize=(13, 3.8))
    _ax.errorbar(_dims, _volume, _error, fmt="o", color=HIST, label="Simulation")
    _ax.plot(_dims, _exact, "k-", label="Exact")
    _ax.set(xlabel="Dimension", ylabel="Volume of the unit ball")
    _ax.legend()
    _hit = _volume > 0  # with no point inside the ball there is no relative uncertainty
    _axr.plot(_dims[_hit], (_error / np.where(_hit, _volume, 1))[_hit], "o-", color=HIST)
    _axr.set(xlabel="Dimension", ylabel="Relative uncertainty", yscale="log")
    _fig.tight_layout()
    _pulls = (_volume - _exact)[_error > 0] / _error[_error > 0]  # d = 1 is exact: no error
    _last = _dims[_hit][-1] if _hit.any() else None
    _empty = _dims[~_hit]
    mo.vstack(
        [
            mo.hstack([dim_points, dim_max], justify="start", gap=2),
            _fig,
            mo.callout(
                mo.md(
                    f"Largest deviation from the exact volume **{np.abs(_pulls).max():.1f} σ**. "
                    + (
                        f"Relative uncertainty {_error[_hit][-1] / _volume[_hit][-1]:.1%} at dimension {_last}. "
                        if _hit.any()
                        else ""
                    )
                    + (
                        f"No point fell inside the ball in dimension {', '.join(map(str, _empty))}: only an upper limit."
                        if len(_empty)
                        else ""
                    )
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

    1. A fraction $n/N$ has the binomial uncertainty $\sqrt{f(1-f)/N}$, and a simulation's precision improves only as
       $1/\sqrt{N}$.
    2. Many small experiments and one large one of the same total size are **equally precise**.
    3. Random points give the volume in any dimension, where a formula is hard. Check the method against a
       case with a **known** answer first.
    4. The price of high dimensions is a rapidly falling fraction inside, so the uncertainty grows.
    """)
    return


if __name__ == "__main__":
    app.run()
