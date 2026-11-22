import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import norm

    from appstat.classification import auc, roc_curve

    SIGNAL, BACKGROUND = "#D55E00", "#0072B2"

    def draw(separation, n_entries, seed):
        """Signal and background: unit Gaussians whose means differ by ``separation * sqrt(2)``."""
        rng = np.random.default_rng(seed)
        shift = separation / np.sqrt(2)
        return rng.normal(5 + shift, 1, n_entries), rng.normal(5 - shift, 1, n_entries)

    return BACKGROUND, SIGNAL, auc, draw, mo, norm, np, plt, roc_curve


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ROC curves

    Signal (orange) and background (blue) are Gaussians of unit width. Move a **cut** from left to right: everything
    above it is accepted.

    - **TPR**, the true positive rate: the fraction of signal accepted. Also called efficiency or power.
    - **FPR**, the false positive rate: the fraction of background accepted. Also called the fake rate, or type 1 error.

    The **ROC curve** is TPR against FPR for every cut. The separation $N_\sigma =
    (\mu_s-\mu_b)/\sqrt{\sigma_s^2+\sigma_b^2}$ decides how far the curve bends towards the top left corner, and the **area
    under it** (AUC) is $\Phi(N_\sigma)$ for Gaussians of equal width.

    Every interactive block below has its controls directly above its plots.

    ---

    **Adapted from** the ROC curve illustration by **Troels C. Petersen** (Niels Bohr Institute, 4th of December
    2024), from the course *Applied Statistics 2025/26*. The original is a script that draws the figure. This version
    is rewritten as an interactive demonstration, with the AUC checked against the formula.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Move the cut

    The dot on the ROC curve is the current cut. A cut to the left accepts nearly everything (top right), a cut to the right
    nearly nothing (bottom left). Increase the separation, and the curve pulls into the corner.
    """)
    return


@app.cell
def _(mo):
    sep = mo.ui.slider(0.0, 4.0, 0.25, value=1.5, show_value=True, label="Separation Nσ")
    cut = mo.ui.slider(1.0, 9.0, 0.1, value=5.0, show_value=True, label="Cut")
    entries = mo.ui.slider(1000, 200_000, 1000, value=50_000, show_value=True, label="Entries")
    return cut, entries, sep


@app.cell
def _(BACKGROUND, SIGNAL, auc, cut, draw, entries, mo, norm, np, plt, roc_curve, sep):
    _signal, _background = draw(sep.value, entries.value, 42)
    _fpr, _tpr, _ = roc_curve(_signal, _background)
    _tpr_cut, _fpr_cut = (_signal >= cut.value).mean(), (_background >= cut.value).mean()
    _fig, (_ax, _axr) = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    _edges = np.linspace(0, 10, 100)
    _ax.hist(_signal, _edges, histtype="stepfilled", alpha=0.5, color=SIGNAL, label="Signal")
    _ax.hist(
        _background, _edges, histtype="stepfilled", alpha=0.5, color=BACKGROUND, label="Background"
    )
    _ax.axvline(cut.value, color="k", ls="--")
    _ax.set(xlabel="x", ylabel="Entries")
    _ax.legend()
    _axr.plot(_fpr, _tpr, color="k", lw=2)
    _axr.plot(_fpr_cut, _tpr_cut, "o", color="r", ms=9)
    _axr.plot([0, 1], [0, 1], ":", color="gray")
    _axr.set(xlabel="FPR", ylabel="TPR", aspect="equal")
    _fig.tight_layout()
    mo.vstack(
        [
            mo.hstack([sep, cut, entries], justify="start", gap=2),
            _fig,
            mo.callout(
                mo.md(
                    f"Cut at {cut.value:.1f}: TPR **{_tpr_cut:.3f}**, FPR **{_fpr_cut:.3f}**. "
                    f"AUC **{auc(_fpr, _tpr):.4f}**, the formula Φ(Nσ) gives {norm.cdf(sep.value):.4f}."
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Several separations together

    The curves for separations from 0 to 3. With no separation the curve is the **diagonal**: the cut accepts signal and
    background alike. Pick a **working point**, a fixed fake rate, and see how the efficiency grows with the separation.
    """)
    return


@app.cell
def _(mo):
    fake_rate = mo.ui.slider(
        0.001, 0.5, 0.001, value=0.05, show_value=True, label="Accepted fake rate (FPR)"
    )
    return (fake_rate,)


@app.cell
def _(auc, draw, fake_rate, mo, norm, np, plt, roc_curve):
    _fig, _ax = plt.subplots(figsize=(6, 5.5))
    _rows = ["| Nσ | AUC | Φ(Nσ) | TPR at the chosen FPR |", "|---|---|---|---|"]
    for _n in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        _signal, _background = draw(_n, 100_000, 42)
        _fpr, _tpr, _ = roc_curve(_signal, _background)
        _ax.plot(_fpr, _tpr, label=f"Nσ = {_n:g}")
        _rows.append(
            f"| {_n:g} | {auc(_fpr, _tpr):.4f} | {norm.cdf(_n):.4f} | {np.interp(fake_rate.value, _fpr, _tpr):.3f} |"
        )
    _ax.axvline(fake_rate.value, color="k", ls="--", lw=1)
    _ax.set(xlabel="FPR", ylabel="TPR", aspect="equal")
    _ax.legend(loc="lower right")
    _fig.tight_layout()
    mo.vstack([fake_rate, _fig, mo.md("\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to take away

    1. Every cut gives a TPR and an FPR. The **ROC curve** shows them all, so classifiers can be compared without choosing a cut.
    2. The diagonal is a classifier that guesses. The **AUC** is 0.5 for it and 1 for a perfect one.
    3. For Gaussians of unit width the AUC is $\Phi(N_\sigma)$: the separation in standard deviations decides everything.
    4. The cut is chosen from the **working point** you need, the fake rate you can afford.
    """)
    return


if __name__ == "__main__":
    app.run()
