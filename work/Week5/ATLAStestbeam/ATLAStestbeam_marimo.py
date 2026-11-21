import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import io

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from appstat.classification import auc, roc_curve, selection_purity, tpr_fpr
    from appstat.data import read_text

    ELECTRON, PION = "#D55E00", "#0072B2"
    COLUMNS = ["Cher", "nLT", "nHT", "EM0", "EM1", "EM2", "EM3", "Had0", "Had1", "Had2", "Muon"]
    # variable: (electrons above the cut?, plot range)
    VARIABLES = {
        "Cher": (True, (400, 1400)),
        "nHT": (True, (0, 30)),
        "EM0": (True, (-0.2, 1.0)),
        "EM1": (True, (-0.2, 1.5)),
        "EM2": (True, (-0.2, 2.0)),
        "EM3": (False, (-0.2, 0.5)),
        "Had0": (False, (0.0, 1.5)),
    }

    table = np.loadtxt(
        io.StringIO(read_text(mo.notebook_location(), "DataSet_AtlasPid_ElectronPion_2GeV.txt")),
        skiprows=1,
    )
    table = table[~np.isnan(table).any(axis=1)]  # a few rows have missing values
    data = dict(zip(COLUMNS, table.T, strict=True))
    # Three independent detectors each vote electron (True) or pion (False)
    VOTES = {
        "Cher": data["Cher"] >= 640,
        "nHT": data["nHT"] >= 3,
        "EM1": data["EM1"] >= 0.35,
    }
    DETECTOR = {"Cher": "Cher", "nHT": "nHT"}  # every other variable is a calorimeter one

    def labels(variable):
        """Clean electron and pion samples for testing ``variable``, from the other two detectors."""
        voters = [v for v in VOTES if v != DETECTOR.get(variable, "EM1")]
        electrons = np.logical_and.reduce([VOTES[v] for v in voters])
        pions = ~np.logical_or.reduce([VOTES[v] for v in voters])
        return electrons, pions

    return (
        COLUMNS,
        ELECTRON,
        PION,
        VARIABLES,
        auc,
        data,
        labels,
        mo,
        np,
        plt,
        roc_curve,
        selection_purity,
        tpr_fpr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Electrons or pions? Particle identification at the ATLAS test beam

    33 000 particles of 2 GeV, each measured by **three independent detectors**: a **Cherenkov** counter, the
    **TRT** (transition radiation, `nHT` high-threshold hits) and the **calorimeters** (`EM0` to `EM3`, `Had0`
    and `Had1`). Electrons and pions are not labelled in the data. You must **create labels** from what you know, then
    measure how well each variable separates the two.

    Labels come from the detectors themselves. Each detector gives a **vote**: Cherenkov above 640, `nHT` at least 3
    and `EM1` above 0.35 all mean *electron*. To test one variable, use only the **two other detectors**, and keep
    the particles where both agree. The sample is then clean, and the variable tested did not make its own labels.
    Particles where they disagree are left out. The calorimeter variables are all tested against `Cher` and `nHT`.

    For a cut on a variable, the **true positive rate** (TPR) is the fraction of electrons kept, and the
    **false positive rate** (FPR) the fraction of pions kept.

    Every interactive block below has its controls directly above its plots.

    ---

    **Adapted from** the ATLAS Test Beam exercise by **Troels C. Petersen** (Niels Bohr Institute), from the course
    *Applied Statistics 2025/26*, with test beam data from ATLAS. The original is an exercise with an example
    solution. This version keeps the labelling, the cuts and the ROC curves, and is rewritten as an interactive
    demonstration. The TRT hit probability questions are left out.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. One variable, one cut

    Move the cut. Electrons are those above it (or below, for the variables where electrons are low). Which cut
    keeps most electrons for few pions?
    """)
    return


@app.cell
def _(VARIABLES, mo):
    cut_variable = mo.ui.dropdown(list(VARIABLES), value="EM1", label="Variable")
    return (cut_variable,)


@app.cell
def _(VARIABLES, cut_variable, mo):
    _low, _high = VARIABLES[cut_variable.value][1]
    # the cuts of the votes for the detectors, the middle of the range for the others
    _start = {"Cher": 640.0, "nHT": 3.0, "EM1": 0.35}.get(cut_variable.value, (_low + _high) / 2)
    cut_value = mo.ui.slider(
        _low, _high, (_high - _low) / 200, value=_start, show_value=True, label="Cut"
    )
    return (cut_value,)


@app.cell
def _(
    ELECTRON,
    PION,
    VARIABLES,
    cut_value,
    cut_variable,
    data,
    labels,
    mo,
    np,
    plt,
    tpr_fpr,
):
    _above, _range = VARIABLES[cut_variable.value]
    _x = data[cut_variable.value]
    _cut = cut_value.value
    _electrons, _pions = labels(cut_variable.value)
    _tpr, _fpr = tpr_fpr(_x[_electrons], _x[_pions], _cut, signal_above=_above)
    _fig, _ax = plt.subplots(figsize=(13, 3.6))
    _edges = np.linspace(*_range, 100)
    _ax.hist(_x[_electrons], _edges, histtype="step", lw=2, color=ELECTRON, label="Electrons")
    _ax.hist(_x[_pions], _edges, histtype="step", lw=2, color=PION, label="Pions")
    _ax.axvline(_cut, color="k", ls="--")
    _ax.set(xlabel=cut_variable.value, ylabel="Particles", yscale="log")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack(
        [
            mo.hstack([cut_variable, cut_value], justify="start", gap=2),
            _fig,
            mo.callout(
                mo.md(
                    f"Cut at **{_cut:.3g}**, electrons {'above' if _above else 'below'}: "
                    f"TPR (electrons kept) **{_tpr:.3f}**, FPR (pions kept) **{_fpr:.4f}**."
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. ROC curves

    A **ROC curve** shows the TPR against the FPR for every possible cut. The closer to the top left corner, the better
    the variable, and the **area under the curve** (AUC) is 1 for perfect separation and 0.5 for none. Add the
    variables you want to compare.
    """)
    return


@app.cell
def _(VARIABLES, mo):
    roc_variables = mo.ui.multiselect(
        list(VARIABLES), value=["Cher", "nHT", "EM1", "Had0"], label="Variables"
    )
    return (roc_variables,)


@app.cell
def _(VARIABLES, auc, data, labels, mo, plt, roc_curve, roc_variables):
    _fig, _ax = plt.subplots(figsize=(7, 5))
    _lines = []
    for _name in roc_variables.value:
        _x = data[_name]
        _electrons, _pions = labels(_name)
        _fpr, _tpr, _ = roc_curve(_x[_electrons], _x[_pions], signal_above=VARIABLES[_name][0])
        _area = auc(_fpr, _tpr)
        _ax.plot(_fpr, _tpr, label=f"{_name} (AUC {_area:.3f})")
        _lines.append(f"- **{_name}**: AUC {_area:.3f}")
    _ax.plot([0, 1], [0, 1], "k:", lw=1)
    _ax.set(xlabel="False positive rate (pions kept)", ylabel="True positive rate (electrons kept)")
    _ax.legend(loc="lower right")
    _fig.tight_layout()
    mo.vstack(
        [
            roc_variables,
            _fig,
            mo.callout(
                mo.md(
                    "\n".join(_lines)
                    + "\n\nEach variable is tested on samples labelled by the two other detectors."
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A. What if pions are a thousand times more common?

    The efficiency and the fake rate belong to the detector, but the **purity** of the selection also depends on how many
    of each there are. Bayes' rule: of everything accepted, the fraction of electrons is
    $\varepsilon N_e / (\varepsilon N_e + f N_\pi)$. Raise the number of pions per electron, and see whether any cut
    still gives 90 % purity.
    """)
    return


@app.cell
def _(VARIABLES, mo):
    purity_variable = mo.ui.dropdown(list(VARIABLES), value="EM1", label="Variable")
    purity_ratio = mo.ui.slider(
        0, 4, 0.25, value=0, show_value=True, label="log10 of pions per electron"
    )
    return purity_ratio, purity_variable


@app.cell
def _(
    ELECTRON,
    VARIABLES,
    data,
    labels,
    mo,
    np,
    plt,
    purity_ratio,
    purity_variable,
    roc_curve,
    selection_purity,
):
    _x = data[purity_variable.value]
    _electrons, _pions = labels(purity_variable.value)
    _fpr, _tpr, _ = roc_curve(
        _x[_electrons], _x[_pions], signal_above=VARIABLES[purity_variable.value][0]
    )
    _ratio = 10.0**purity_ratio.value
    _purity = selection_purity(_tpr, _fpr, 1.0, _ratio)
    _fig, _ax = plt.subplots(figsize=(13, 3.6))
    _ax.plot(_tpr, _purity, color=ELECTRON, lw=2)
    _ax.axhline(0.9, color="k", ls="--", lw=1)
    _ax.set(xlabel="Electron efficiency (TPR)", ylabel="Purity of the electrons", ylim=(0, 1.02))
    _fig.tight_layout()
    _good = _purity >= 0.9
    mo.vstack(
        [
            mo.hstack([purity_variable, purity_ratio], justify="start", gap=2),
            _fig,
            mo.callout(
                mo.md(
                    f"With {_ratio:,.0f} pions per electron: "
                    + (
                        f"90 % purity is reached with efficiency up to **{np.max(_tpr[_good]):.3f}**."
                        if _good.any()
                        else "**no cut** reaches 90 % purity."
                    )
                ),
                kind="success" if _good.any() else "danger",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to take away

    1. Real data has no labels. You make them from **independent** information, and say how.
    2. A cut gives a **true** and a **false positive rate**. The ROC curve shows all of them, and the AUC is one number.
    3. A variable used to label cannot be tested honestly on those labels.
    4. **Purity** depends on how common each kind is, not only on the detector. A small fake rate is not enough when
       the background is huge.
    """)
    return


if __name__ == "__main__":
    app.run()
