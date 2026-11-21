import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    from appstat.markov import (
        ehrenfest_matrix,
        mean_hitting_time,
        simulate_markov,
        state_distribution,
    )

    SIMULATED, EXACT = "#0072B2", "#D55E00"

    def make_controls(turns=20, start=3):
        """Sliders for the urn, for a block that needs its own."""
        return mo.ui.dictionary(
            {
                "balls": mo.ui.slider(2, 30, 1, value=10, show_value=True, label="Balls"),
                "start": mo.ui.slider(
                    0, 30, 1, value=start, show_value=True, label="Blue balls at the start"
                ),
                "turns": mo.ui.slider(0, 100, 1, value=turns, show_value=True, label="Turns"),
                "p_stay": mo.ui.slider(
                    0.0, 0.9, 0.1, value=0.0, show_value=True, label="Chance that nothing moves"
                ),
                "runs": mo.ui.slider(
                    100, 20_000, 100, value=5000, show_value=True, label="Simulated runs"
                ),
                "seed": mo.ui.number(0, 9999, 1, value=42, label="Random seed"),
            }
        )

    def controls_panel(controls):
        """The six controls in two rows."""
        return mo.vstack(
            [
                mo.hstack(
                    [controls["balls"], controls["start"], controls["turns"]],
                    justify="start",
                    gap=2,
                ),
                mo.hstack(
                    [controls["p_stay"], controls["runs"], controls["seed"]], justify="start", gap=2
                ),
            ]
        )

    return (
        EXACT,
        SIMULATED,
        controls_panel,
        ehrenfest_matrix,
        make_controls,
        mean_hitting_time,
        mo,
        np,
        plt,
        simulate_markov,
        state_distribution,
        stats,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The Ehrenfest urn: Markov chains

    Two baskets, a blue and a red, and 10 balls. Each turn a **random ball** is picked and moved to the other
    basket. The question: **if 3 balls start in the blue basket, what is the probability of 7 after 20 turns?**

    Each turn depends on the last one, so the probabilities cannot be multiplied as for independent events. Two ways
    to answer:

    - **Simulate** many runs and count.
    - Write the chance to go from each state to each other as a **matrix** $M$. The distribution after $n$ turns is
      the starting distribution times $M^n$, and needs no simulation at all.

    Every interactive block below has its controls directly above its plots.

    ---

    **Adapted from** the Ehrenfest Ball Experiment exercise by **Mathias L. Heltberg** (Niels Bohr Institute,
    6th of December 2024), from the course *Applied Statistics 2025/26*. The original is an exercise with an
    example solution, partly left to fill in. This version keeps its problem and its questions and is rewritten as an
    interactive demonstration.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Simulation against the matrix

    The state is the number of blue balls, **0 to 10**, so there is one more state than balls. From $k$ blue balls the
    state goes to $k-1$ with probability $k/N$ (a blue ball was picked) and to $k+1$ with probability $(N-k)/N$.
    The bars are the simulation, the dots are the exact matrix result. Try 21 turns instead of 20 and ask for 9
    balls: every turn changes the state by one, so only states of the right **parity** can be reached. Then let nothing
    move in half of the turns.
    """)
    return


@app.cell
def _(make_controls):
    urn_controls = make_controls()
    return (urn_controls,)


@app.cell
def _(
    EXACT,
    SIMULATED,
    controls_panel,
    ehrenfest_matrix,
    mo,
    np,
    plt,
    simulate_markov,
    state_distribution,
    urn_controls,
):
    _balls, _turns = urn_controls["balls"].value, urn_controls["turns"].value
    _start = min(urn_controls["start"].value, _balls)
    _matrix = ehrenfest_matrix(_balls, urn_controls["p_stay"].value)
    _rng = np.random.default_rng(urn_controls["seed"].value)
    _paths = simulate_markov(_matrix, _start, _turns, urn_controls["runs"].value, _rng)
    _simulated = np.bincount(_paths[:, -1], minlength=_balls + 1) / len(_paths)
    _exact = state_distribution(_start, _matrix, _turns)

    _fig, (_ax_path, _ax_dist) = plt.subplots(
        1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1, 1]}
    )
    for _run in _paths[:5]:
        _ax_path.plot(_run, "-", lw=1, alpha=0.8)
    _ax_path.set(
        xlabel="Turn", ylabel="Blue balls", ylim=(-0.5, _balls + 0.5), title="Five of the runs"
    )
    _states = np.arange(_balls + 1)
    _ax_dist.bar(_states, _simulated, color=SIMULATED, alpha=0.6, label="Simulation")
    _ax_dist.plot(_states, _exact, "o", color=EXACT, label="Matrix")
    _ax_dist.set(xlabel=f"Blue balls after {_turns} turns", ylabel="Probability")
    _ax_dist.legend()
    _fig.tight_layout()

    _target = min(7, _balls)
    mo.vstack(
        [
            controls_panel(urn_controls),
            _fig,
            mo.callout(
                mo.md(
                    f"Probability of **{_target}** blue balls after {_turns} turns: matrix "
                    f"**{_exact[_target]:.4f}**, simulation {_simulated[_target]:.4f} "
                    f"(± {np.sqrt(_exact[_target] * (1 - _exact[_target]) / len(_paths)):.4f})."
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Equilibrium

    After very many turns the memory of the start is lost, and the distribution no longer changes. In equilibrium
    **every ball is in either basket with probability ½, independently**, so the number of blue balls is
    **binomial**, with mean $N/2$. Let nothing move in half of the turns, so that the parity does not alternate,
    and add turns.
    """)
    return


@app.cell
def _(mo):
    eq_balls = mo.ui.slider(2, 30, 1, value=10, show_value=True, label="Balls")
    eq_turns = mo.ui.slider(0, 1000, 10, value=20, show_value=True, label="Turns")
    eq_stay = mo.ui.slider(
        0.0, 0.9, 0.1, value=0.5, show_value=True, label="Chance that nothing moves"
    )
    return eq_balls, eq_stay, eq_turns


@app.cell
def _(
    EXACT,
    SIMULATED,
    ehrenfest_matrix,
    eq_balls,
    eq_stay,
    eq_turns,
    mo,
    np,
    plt,
    state_distribution,
    stats,
):
    _n = eq_balls.value
    _distribution = state_distribution(0, ehrenfest_matrix(_n, eq_stay.value), eq_turns.value)
    _binomial = stats.binom.pmf(np.arange(_n + 1), _n, 0.5)
    _fig, _ax = plt.subplots(figsize=(13, 3.6))
    _ax.bar(
        np.arange(_n + 1),
        _distribution,
        color=SIMULATED,
        alpha=0.6,
        label=f"After {eq_turns.value} turns, from 0",
    )
    _ax.plot(np.arange(_n + 1), _binomial, "o", color=EXACT, label="Binomial(N, ½)")
    _ax.set(xlabel="Blue balls", ylabel="Probability")
    _ax.legend()
    _fig.tight_layout()
    _distance = 0.5 * np.abs(_distribution - _binomial).sum()
    mo.vstack(
        [
            mo.hstack([eq_balls, eq_turns, eq_stay], justify="start", gap=2),
            _fig,
            mo.callout(
                mo.md(f"Distance from the binomial (total variation): **{_distance:.4f}**."),
                kind="success" if _distance < 0.01 else "warn",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A. How long until all balls are blue?

    Make the state with every ball in the blue basket **absorbing**: once there, nothing changes. How many turns does
    it take on average? Solving the matrix equation $t = 1 + Qt$ gives it exactly, with $Q$ the matrix without the
    absorbing state. Run it from other starting states and compare with a simulation. The time grows like
    $2^N$: the chance of every ball being in one basket falls fast with more balls.
    """)
    return


@app.cell
def _(mo):
    abs_balls = mo.ui.slider(2, 9, 1, value=8, show_value=True, label="Balls")
    abs_start = mo.ui.slider(0, 9, 1, value=3, show_value=True, label="Blue balls at the start")
    abs_runs = mo.ui.slider(100, 1000, 100, value=500, show_value=True, label="Simulated runs")
    return abs_balls, abs_runs, abs_start


@app.cell
def _(
    SIMULATED,
    abs_balls,
    abs_runs,
    abs_start,
    ehrenfest_matrix,
    mean_hitting_time,
    mo,
    np,
    plt,
    simulate_markov,
):
    _n = abs_balls.value
    _start = min(abs_start.value, _n)
    _matrix = ehrenfest_matrix(_n)
    _exact = mean_hitting_time(_matrix, _n)
    _absorbing = _matrix.copy()
    _absorbing[_n] = 0.0
    _absorbing[_n, _n] = 1.0
    _cap = int(8 * _exact[_start]) + 20  # long enough for nearly every run to arrive
    _paths = simulate_markov(_absorbing, _start, _cap, abs_runs.value, np.random.default_rng(1))
    _arrived = (_paths == _n).any(axis=1)
    _times = np.argmax(_paths == _n, axis=1)[_arrived]

    _fig, _ax = plt.subplots(figsize=(13, 3.4))
    _ax.hist(_times, bins=50, color=SIMULATED, alpha=0.7)
    _ax.axvline(_exact[_start], color="k", ls="--", label=f"Exact mean {_exact[_start]:.0f}")
    _ax.set(xlabel=f"Turns until all {_n} balls are blue", ylabel="Runs")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack(
        [
            mo.hstack([abs_balls, abs_start, abs_runs], justify="start", gap=2),
            _fig,
            mo.callout(
                mo.md(
                    f"Exact mean **{_exact[_start]:.1f}** turns, simulated mean {_times.mean():.1f} "
                    f"({_arrived.sum()} of {abs_runs.value} runs arrived within {_cap} turns)."
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

    1. When each step depends on the last, write the chance to go between states as a **matrix**. The distribution
       after $n$ steps is $M^n$ applied to the start.
    2. Check a calculation with a **simulation**, and a simulation with a calculation.
    3. Every step changed the state by one, so some states are **impossible** after a given number of steps.
       Allowing "nothing happens" removes the parity.
    4. The chain forgets its start and settles into an **equilibrium**, here the binomial.
    5. An **absorbing** state is reached after a random time. The matrix gives its mean without any simulation.
    """)
    return


if __name__ == "__main__":
    app.run()
