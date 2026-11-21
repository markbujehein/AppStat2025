import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import functools

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    from appstat.data import read_text
    from appstat.digits import difference_probabilities, gap_lengths, read_digits, sequence_counts
    from appstat.stats import chi2_pvalue, dispersion_chi2, multinomial_chi2

    OBSERVED, EXPECTED = "#0072B2", "#D55E00"
    N_DIGITS = 4532  # the length of each of the five data sets; π and NumPy are cut to match
    SETS = ["A", "B", "C", "D", "E", "π (first digits)", "Random numbers from NumPy"]
    TESTS = [
        "Each digit equally likely",
        "Even or odd",
        "Below 5 or 5 and above",
        "Difference of neighbours",
        "Gaps between zeros",
        "Three-digit sequences",
    ]
    GAP_MAX = 30  # gaps longer than this share one bin

    @functools.cache  # the table asks for every series once per test
    def load_set(name, seed=1):
        """The digits of one data set, ``N_DIGITS`` of them."""
        if name == "Random numbers from NumPy":
            return np.random.default_rng(seed).integers(0, 10, N_DIGITS)
        file = "pi1000000.txt" if name.startswith("π") else f"data_RandomDigits2025_{name}.txt"
        return read_digits(read_text(mo.notebook_location(), file))[:N_DIGITS]

    def run_test(digits, test):
        """Observed counts, expected counts, x positions, chi-square, ndof and p-value of one test."""
        n = len(digits)
        if test == "Three-digit sequences":
            counts = sequence_counts(digits, 3)
            chi2, ndof = dispersion_chi2(counts)
            return {"counts": counts, "chi2": chi2, "ndof": ndof, "p": chi2_pvalue(chi2, ndof)}
        if test == "Each digit equally likely":
            x, counts, probabilities = (
                np.arange(10),
                np.bincount(digits, minlength=10),
                np.full(10, 0.1),
            )
        elif test == "Even or odd":
            x, counts, probabilities = (
                np.arange(2),
                np.bincount(digits % 2, minlength=2),
                np.full(2, 0.5),
            )
        elif test == "Below 5 or 5 and above":
            x, counts, probabilities = (
                np.arange(2),
                np.bincount(digits // 5, minlength=2),
                np.full(2, 0.5),
            )
        elif test == "Difference of neighbours":
            x, probabilities = difference_probabilities()
            counts = np.bincount(np.diff(digits) + 9, minlength=19)
        else:  # gaps between zeros: geometric, with a bin for everything longer than GAP_MAX
            gaps = gap_lengths(digits, 0)
            x = np.arange(1, GAP_MAX + 2)
            counts = np.append(
                np.bincount(gaps, minlength=GAP_MAX + 1)[1 : GAP_MAX + 1], (gaps > GAP_MAX).sum()
            )
            probabilities = np.append(0.1 * 0.9 ** np.arange(GAP_MAX), 0.9**GAP_MAX)
        chi2, ndof = multinomial_chi2(counts, probabilities)
        return {
            "x": x,
            "counts": counts,
            "expected": counts.sum() * probabilities,
            "chi2": chi2,
            "ndof": ndof,
            "p": chi2_pvalue(chi2, ndof),
            "n": n,
        }

    return (
        EXPECTED,
        OBSERVED,
        SETS,
        TESTS,
        load_set,
        mo,
        np,
        plt,
        run_test,
        stats,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Are these digits random?

    Seven series of digits 0 to 9: five of unknown origin (A to E), the first digits of **π**, and numbers from the
    computer. Which are random, and which might have come from a person? There is no single test. You must
    **think of what random digits would do**, turn that into a probability, and see if the data agree. A few ideas:

    - each digit is equally likely, and so is even or odd,
    - the difference of two neighbours has a **triangle** distribution,
    - the gap between two zeros is **geometric**,
    - each of the 1000 three-digit sequences occurs about equally often (Poisson counts).

    Every interactive block below has its controls directly above its plots.

    ---

    **Adapted from** the Random Digits exercise by **Troels C. Petersen** (Niels Bohr Institute), from the course
    *Applied Statistics 2025/26*. The original is an exercise with an example solution, where you design the tests.
    This version offers the tests above on all seven series, and is rewritten as an interactive demonstration.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. One test on one series

    The bars are the counts in the data, and the dots what random digits would give. The p-value is that of the
    $\chi^2$. A very small p-value means the series is **not** random in that respect, and a large one only means
    this test found nothing: a clever person can fool one test, not all of them.
    """)
    return


@app.cell
def _(SETS, TESTS, mo):
    one_set = mo.ui.dropdown(SETS, value=SETS[0], label="Series")
    one_test = mo.ui.dropdown(TESTS, value=TESTS[0], label="Test")
    one_seed = mo.ui.number(0, 9999, 1, value=1, label="Seed (NumPy series)")
    return one_seed, one_set, one_test


@app.cell
def _(EXPECTED, OBSERVED, load_set, mo, np, one_seed, one_set, one_test, plt, run_test, stats):
    _digits = load_set(one_set.value, one_seed.value)
    _result = run_test(_digits, one_test.value)
    _fig, _ax = plt.subplots(figsize=(13, 3.6))
    if one_test.value == "Three-digit sequences":
        _counts = _result["counts"]
        _values = np.arange(0, _counts.max() + 2)
        _ax.hist(
            _counts,
            bins=np.arange(-0.5, _counts.max() + 1.5),
            color=OBSERVED,
            alpha=0.7,
            label="Data",
        )
        _ax.plot(
            _values,
            1000 * stats.poisson.pmf(_values, _counts.mean()),
            "o",
            color=EXPECTED,
            label="Poisson",
        )
        _ax.set(xlabel="Times a three-digit sequence occurs", ylabel="Number of sequences")
    else:
        _ax.bar(_result["x"], _result["counts"], color=OBSERVED, alpha=0.7, label="Data")
        _ax.plot(_result["x"], _result["expected"], "o", color=EXPECTED, label="Random digits")
        _ax.set(xlabel=one_test.value, ylabel="Counts")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack(
        [
            mo.hstack([one_set, one_test, one_seed], justify="start", gap=2),
            _fig,
            mo.callout(
                mo.md(
                    f"χ²/Ndof = {_result['chi2']:.1f}/{_result['ndof']}, **p = {_result['p']:.3g}**."
                ),
                kind="success" if _result["p"] > 0.0027 else "danger",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Every test on every series

    The p-values of all the tests. A value below 0.0027 (three σ) is marked in **bold**. The computer's numbers should
    pass them all, apart from the one in about 370 times that fails by chance.
    """)
    return


@app.cell
def _(SETS, TESTS, load_set, run_test):
    _rows = ["| Series | " + " | ".join(TESTS) + " |", "|---|" + "---|" * len(TESTS)]
    for _name in SETS:
        _cells = []
        for _test in TESTS:
            _p = run_test(load_set(_name), _test)["p"]
            _cells.append(f"**{_p:.2g}**" if _p < 0.0027 else f"{_p:.2g}")
        _rows.append(f"| {_name} | " + " | ".join(_cells) + " |")
    p_value_table = "\n".join(_rows)
    return (p_value_table,)


@app.cell
def _(mo, p_value_table):
    mo.md(p_value_table)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to take away

    1. Randomness has no single test. **Think of properties** random numbers must have, and test each one.
    2. Turn every result into a **probability**, and decide the criterion before looking.
    3. A series can pass the test of one property and fail another. Digits made up by a person often have the right
       frequencies but the wrong neighbours, which the difference, gap and sequence tests catch.
    4. Run many tests, and expect one in 370 to fail at three σ **by chance** even for truly random numbers. That is
       why the computer's numbers are the reference.
    """)
    return


if __name__ == "__main__":
    app.run()
