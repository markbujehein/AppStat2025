import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import io

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from appstat.classification import auc, fisher_weights, roc_curve, separation
    from appstat.data import read_text

    A_COLOUR, B_COLOUR = "#0072B2", "#D55E00"
    VARIABLES = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
    SPECIES = {1: "Setosa", 2: "Versicolor", 3: "Virginica"}
    table = np.loadtxt(
        io.StringIO(read_text(mo.notebook_location(), "DataSet_AndersonFisherIris.txt"))
    )
    measurements, species = table[:, :4], table[:, 4].astype(int)
    return (
        A_COLOUR,
        B_COLOUR,
        SPECIES,
        VARIABLES,
        auc,
        fisher_weights,
        measurements,
        mo,
        np,
        plt,
        roc_curve,
        separation,
        species,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fisher's linear discriminant: combining variables to separate two kinds

    In 1936 Fisher separated three kinds of iris flowers, 50 of each, using four measurements: the length and width of
    the sepal and of the petal. One variable alone often overlaps. A **linear combination**
    $\mathbf{w}\cdot\mathbf{x}$ can pull the two kinds further apart, if the weights are chosen well. Fisher's
    choice maximises the distance between the two means compared with the spread of each:

    $$\mathbf{w} = (\Sigma_A + \Sigma_B)^{-1}(\boldsymbol{\mu}_A - \boldsymbol{\mu}_B)$$

    with $\Sigma$ the covariance matrices and $\boldsymbol\mu$ the means. The **separation** of two samples is
    $|\bar x - \bar y| / \sqrt{\sigma_x^2 + \sigma_y^2}$.

    Every interactive block below has its controls directly above its plots.

    ---

    **Adapted from** the Fisher Discriminant exercises by **Troels C. Petersen** (Niels Bohr Institute), from the
    course *Applied Statistics 2025/26*, with the iris data of Anderson and Fisher. The originals are two exercises with
    example solutions. This version keeps the iris data and the method, and is rewritten as an interactive demonstration.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Two kinds, two variables

    Pick two species and two variables. The line is the direction of the Fisher discriminant, and the lower
    histogram is the data **projected** onto it. Compare the separation with that of each variable alone. Setosa is easy,
    and Versicolor against Virginica is the real test.
    """)
    return


@app.cell
def _(SPECIES, VARIABLES, mo):
    pick_a = mo.ui.dropdown({v: k for k, v in SPECIES.items()}, value="Versicolor", label="Kind A")
    pick_b = mo.ui.dropdown({v: k for k, v in SPECIES.items()}, value="Virginica", label="Kind B")
    pick_x = mo.ui.dropdown(VARIABLES, value="Sepal length", label="First variable")
    pick_y = mo.ui.dropdown(VARIABLES, value="Sepal width", label="Second variable")
    return pick_a, pick_b, pick_x, pick_y


@app.cell
def _(
    A_COLOUR,
    B_COLOUR,
    SPECIES,
    VARIABLES,
    fisher_weights,
    measurements,
    mo,
    np,
    pick_a,
    pick_b,
    pick_x,
    pick_y,
    plt,
    separation,
    species,
):
    _columns = [VARIABLES.index(pick_x.value), VARIABLES.index(pick_y.value)]
    _controls = mo.vstack(
        [
            mo.hstack([pick_a, pick_b], justify="start", gap=2),
            mo.hstack([pick_x, pick_y], justify="start", gap=2),
        ]
    )
    if pick_a.value == pick_b.value or _columns[0] == _columns[1]:
        _output = mo.callout(
            mo.md("Pick two **different** kinds and two **different** variables."), kind="warn"
        )
    else:
        _a = measurements[species == pick_a.value][:, _columns]
        _b = measurements[species == pick_b.value][:, _columns]
        _w = fisher_weights(_a, _b)
        _w = _w / np.linalg.norm(_w)
        _fig, (_ax, _axp) = plt.subplots(1, 2, figsize=(13, 3.8))
        _ax.plot(*_a.T, "o", color=A_COLOUR, alpha=0.7, label=SPECIES[pick_a.value])
        _ax.plot(*_b.T, "o", color=B_COLOUR, alpha=0.7, label=SPECIES[pick_b.value])
        _centre = np.vstack([_a, _b]).mean(axis=0)
        _ax.annotate("", xy=_centre + _w, xytext=_centre, arrowprops={"arrowstyle": "->", "lw": 2})
        _ax.set(xlabel=pick_x.value, ylabel=pick_y.value)
        _ax.legend()
        _edges = np.linspace(
            min((_a @ _w).min(), (_b @ _w).min()), max((_a @ _w).max(), (_b @ _w).max()), 25
        )
        _axp.hist(_a @ _w, _edges, alpha=0.7, color=A_COLOUR)
        _axp.hist(_b @ _w, _edges, alpha=0.7, color=B_COLOUR)
        _axp.set(xlabel="Fisher discriminant", ylabel="Flowers")
        _fig.tight_layout()
        _single = [separation(_a[:, i], _b[:, i]) for i in range(2)]
        _output = mo.vstack(
            [
                _fig,
                mo.callout(
                    mo.md(
                        f"Separation: {pick_x.value} **{_single[0]:.2f}**, {pick_y.value} **{_single[1]:.2f}**, "
                        f"Fisher discriminant **{separation(_a @ _w, _b @ _w):.2f}**. "
                        f"Weights ({_w[0]:+.2f}, {_w[1]:+.2f})."
                    ),
                    kind="info",
                ),
            ]
        )
    mo.vstack([_controls, _output])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. All four variables, and the ROC curve

    With all four measurements the discriminant has four weights. The **ROC curve** shows, for every cut on the
    discriminant, the fraction of kind A kept against the fraction of kind B kept. Compare with the best single
    variable.
    """)
    return


@app.cell
def _(SPECIES, mo):
    roc_a = mo.ui.dropdown({v: k for k, v in SPECIES.items()}, value="Versicolor", label="Kind A")
    roc_b = mo.ui.dropdown({v: k for k, v in SPECIES.items()}, value="Virginica", label="Kind B")
    return roc_a, roc_b


@app.cell
def _(
    A_COLOUR,
    B_COLOUR,
    VARIABLES,
    auc,
    fisher_weights,
    measurements,
    mo,
    plt,
    roc_a,
    roc_b,
    roc_curve,
    species,
):
    if roc_a.value == roc_b.value:
        _output = mo.callout(mo.md("Pick two **different** kinds."), kind="warn")
    else:
        _a, _b = measurements[species == roc_a.value], measurements[species == roc_b.value]
        _w = fisher_weights(_a, _b)
        _fig, _ax = plt.subplots(figsize=(6, 5))
        _lines = []
        for _i, _name in enumerate(VARIABLES):
            _direction = _a[:, _i].mean() > _b[:, _i].mean()
            _fpr, _tpr, _ = roc_curve(_a[:, _i], _b[:, _i], signal_above=_direction)
            _ax.plot(_fpr, _tpr, color="gray", alpha=0.6, lw=1)
            _lines.append(f"- {_name}: AUC {auc(_fpr, _tpr):.3f}")
        _fpr, _tpr, _ = roc_curve(_a @ _w, _b @ _w)
        _ax.plot(_fpr, _tpr, color=A_COLOUR, lw=2.5, label="Fisher discriminant")
        _ax.plot([0, 1], [0, 1], ":", color=B_COLOUR)
        _ax.set(xlabel="Fraction of B kept", ylabel="Fraction of A kept")
        _ax.legend(loc="lower right")
        _fig.tight_layout()
        _lines.append(f"- **Fisher discriminant: AUC {auc(_fpr, _tpr):.3f}**")
        _output = mo.vstack([_fig, mo.callout(mo.md("\n".join(_lines)), kind="info")])
    mo.vstack([mo.hstack([roc_a, roc_b], justify="start", gap=2), _output])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to take away

    1. Variables that overlap alone can separate well **in combination**, because they are correlated.
    2. Fisher's weights come from just the **means and covariances**, in one matrix equation. No fit or iteration.
    3. The discriminant is a single number per flower, so one cut and one ROC curve are enough.
    4. The weights are found on the data they are tested on, so the separation is a little optimistic. Use separate
       training and test data when it matters.
    """)
    return


if __name__ == "__main__":
    app.run()
