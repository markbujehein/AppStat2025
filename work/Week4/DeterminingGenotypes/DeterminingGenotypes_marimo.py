import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from appstat.genetics import genotype_posterior, simulate_reads

    GENOTYPES = ["AA", "AB", "BB"]
    POSTERIOR, SIMULATED = "#0072B2", "#D55E00"

    def parse_reads(text):
        """The reads typed as letters, for example ``AAAAB``, as 0 for A and 1 for B."""
        letters = [c for c in text.upper() if c in "AB"]
        return np.array([0 if c == "A" else 1 for c in letters], dtype=int)

    def measurement_matrix(a_read_as_b, b_read_as_a):
        """Rows are the true base (A, B), columns the measured base (A, B)."""
        return np.array([[1 - a_read_as_b, a_read_as_b], [b_read_as_a, 1 - b_read_as_a]])

    MAX_SIM_READS = 10  # more reads than this match almost nobody, and the arrays grow with it

    def simulate_matches(reads, measurement, prior, n_individuals, rng):
        """The genotypes of the simulated individuals whose measured reads equal ``reads``."""
        n_reads = len(reads)
        found = []
        for genotype, n in enumerate(rng.multinomial(n_individuals, prior)):
            sequences = simulate_reads(genotype, n * n_reads, measurement, rng).reshape(n, n_reads)
            found += [genotype] * int(np.all(sequences == reads, axis=1).sum())
        return np.bincount(found, minlength=3)

    return (
        GENOTYPES,
        MAX_SIM_READS,
        POSTERIOR,
        SIMULATED,
        genotype_posterior,
        measurement_matrix,
        mo,
        np,
        parse_reads,
        plt,
        simulate_matches,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Determining genotypes with Bayes' theorem

    A gene has two versions, **A** and **B**, and everyone has two copies: the genotype is **AA**, **AB** or **BB**. A
    machine reads copies of the gene, and sometimes reads them wrongly. Given the reads, say **A A A A B**, what is the
    probability of each genotype?

    The model used here, stated openly:

    - Each read looks at **one of the two copies, picked at random**. AA always gives A, BB always B, and AB gives A or B
      with probability ½.
    - The machine reads a true A as B with some probability, and a true B as A with another.
    - The reads are independent. The probability of a genotype in the population, the **prior**, is known.

    Then Bayes: $P(\text{genotype}\mid\text{reads}) \propto P(\text{reads}\mid\text{genotype})\,P(\text{genotype})$.
    Every interactive block below has its controls directly above its plots.

    ---

    **Adapted from** the Determining Genotypes exercise by **Mathias L. Heltberg** (Niels Bohr Institute), from the
    course *Applied Statistics 2025/26*. The original is an exercise with no example solution, and describes the
    calculation with matrices. This version states the model above directly and is written from the exercise text. It
    is checked against a simulation, and is not an official solution.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. From reads to genotype probabilities

    The exercise's numbers are preset: A is read as B 15 % of the time, B as A 5 % of the time, and the prior is
    AA 45 %, AB 15 %, BB 40 %. Type other reads, and change the numbers. Note how **one B among many A** is
    still compatible with AA, because the machine makes errors.
    """)
    return


@app.cell
def _(mo):
    reads_text = mo.ui.text(value="AAAAB", label="Reads (A and B)")
    a_as_b = mo.ui.slider(0.0, 0.5, 0.01, value=0.15, show_value=True, label="A read as B")
    b_as_a = mo.ui.slider(0.0, 0.5, 0.01, value=0.05, show_value=True, label="B read as A")
    prior_aa = mo.ui.slider(0.0, 1.0, 0.05, value=0.45, show_value=True, label="Prior AA")
    prior_ab = mo.ui.slider(0.0, 1.0, 0.05, value=0.15, show_value=True, label="Prior AB")
    return a_as_b, b_as_a, prior_aa, prior_ab, reads_text


@app.cell
def _(
    GENOTYPES,
    POSTERIOR,
    a_as_b,
    b_as_a,
    genotype_posterior,
    measurement_matrix,
    mo,
    np,
    parse_reads,
    plt,
    prior_aa,
    prior_ab,
    reads_text,
):
    _controls = mo.vstack(
        [
            mo.hstack([reads_text, a_as_b, b_as_a], justify="start", gap=2),
            mo.hstack([prior_aa, prior_ab], justify="start", gap=2),
        ]
    )
    _prior = np.array([prior_aa.value, prior_ab.value, 1 - prior_aa.value - prior_ab.value])
    _reads = parse_reads(reads_text.value)
    _problem = None
    if _prior[2] < -1e-9:
        _problem = "The priors of AA and AB add up to more than 1."
    else:
        _prior = np.clip(_prior, 0, None)
        try:
            _posterior = genotype_posterior(
                _reads, measurement_matrix(a_as_b.value, b_as_a.value), _prior
            )
        except ValueError:
            _problem = "These reads are impossible for this prior and machine: change them."
    if _problem:
        _output = mo.callout(mo.md(_problem), kind="warn")
    else:
        _fig, _ax = plt.subplots(figsize=(7, 3.4))
        _x = np.arange(3)
        _ax.bar(_x - 0.2, _prior, 0.4, color="gray", label="Prior")
        _ax.bar(_x + 0.2, _posterior, 0.4, color=POSTERIOR, label="After the reads")
        _ax.set(xticks=_x, xticklabels=GENOTYPES, ylabel="Probability", ylim=(0, 1))
        _ax.legend()
        _fig.tight_layout()
        _output = mo.vstack(
            [
                _fig,
                mo.callout(
                    mo.md(
                        f"Reads **{''.join('AB'[r] for r in _reads) or '(none)'}**: "
                        + ", ".join(
                            f"{g} **{p:.3f}**" for g, p in zip(GENOTYPES, _posterior, strict=True)
                        )
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
    ## 2. Check with a simulation

    Simulate many individuals with genotypes drawn from the prior, read each the same number of times, and keep only
    those whose reads **equal the ones above**. Their genotypes must follow the Bayes probabilities. This is
    the "compare with a simulation" of the exercise.
    """)
    return


@app.cell
def _(mo):
    sim_n = mo.ui.slider(
        10_000, 1_000_000, 10_000, value=300_000, show_value=True, label="Simulated individuals"
    )
    return (sim_n,)


@app.cell
def _(
    GENOTYPES,
    MAX_SIM_READS,
    POSTERIOR,
    SIMULATED,
    a_as_b,
    b_as_a,
    genotype_posterior,
    measurement_matrix,
    mo,
    np,
    parse_reads,
    plt,
    prior_aa,
    prior_ab,
    reads_text,
    sim_n,
    simulate_matches,
):
    _prior = np.array([prior_aa.value, prior_ab.value, 1 - prior_aa.value - prior_ab.value])
    _reads = parse_reads(reads_text.value)
    if _prior[2] < -1e-9 or len(_reads) == 0:
        _output = mo.callout(
            mo.md("Give some reads, and priors that add up to at most 1."), kind="warn"
        )
    elif len(_reads) > MAX_SIM_READS:
        _output = mo.callout(
            mo.md(
                f"Simulate at most {MAX_SIM_READS} reads: a longer sequence matches almost nobody."
            ),
            kind="warn",
        )
    else:
        _matrix = measurement_matrix(a_as_b.value, b_as_a.value)
        _counts = simulate_matches(
            _reads, _matrix, np.clip(_prior, 0, None), sim_n.value, np.random.default_rng(1)
        )
        try:
            _exact = genotype_posterior(_reads, _matrix, np.clip(_prior, 0, None))
        except ValueError:
            _exact = None
        if _exact is None:
            _output = mo.callout(mo.md("These reads are impossible for this prior."), kind="warn")
        elif _counts.sum() == 0:
            _output = mo.callout(
                mo.md(
                    "No simulated individual had these reads. Add individuals or shorten the reads."
                ),
                kind="warn",
            )
        else:
            _fraction = _counts / _counts.sum()
            _fig, _ax = plt.subplots(figsize=(7, 3.4))
            _x = np.arange(3)
            _ax.bar(_x - 0.2, _exact, 0.4, color=POSTERIOR, label="Bayes")
            _ax.bar(_x + 0.2, _fraction, 0.4, color=SIMULATED, label="Simulation")
            _ax.set(xticks=_x, xticklabels=GENOTYPES, ylabel="Probability", ylim=(0, 1))
            _ax.legend()
            _fig.tight_layout()
            _output = mo.vstack(
                [
                    _fig,
                    mo.callout(
                        mo.md(
                            f"{_counts.sum()} of {sim_n.value:,} individuals had the reads. Simulation: "
                            + ", ".join(
                                f"{g} {f:.3f}" for g, f in zip(GENOTYPES, _fraction, strict=True)
                            )
                            + "."
                        ),
                        kind="success",
                    ),
                ]
            )
    mo.vstack([sim_n, _output])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What to take away

    1. **Bayes' theorem** turns "how likely are these reads for each genotype" into "how likely is each genotype
       for these reads". The prior matters most when there is little data.
    2. Independent reads **multiply**: the probability of the whole sequence is the product of each read's.
    3. A measurement error means no single read settles the genotype: one B among many A is still probably AA.
    4. A **simulation** that keeps only what matches the data reproduces the same probabilities, a check that needs no
       algebra.
    """)
    return


if __name__ == "__main__":
    app.run()
