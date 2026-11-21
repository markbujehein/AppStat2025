"""Markov chains: transition matrices, exact distributions, simulation and hitting times."""

lazy import numpy as np


def ehrenfest_matrix(n_balls: int, p_stay: float = 0.0) -> np.ndarray:
    """
    Transition matrix of the Ehrenfest urn: ``n_balls`` balls in two baskets, one moved per turn.

    The state is the number of balls in the first basket, ``0, 1, ..., n_balls``. A random ball
    is picked each turn, so from state ``k`` the state goes to ``k - 1`` with probability
    ``k / n_balls`` and to ``k + 1`` with probability ``(n_balls - k) / n_balls``.

    Parameters
    ----------
    n_balls : int
        Number of balls, at least 1.
    p_stay : float, default 0.0
        Probability that nothing is moved in a turn, from 0 up to (not including) 1.

    Returns
    -------
    np.ndarray
        Matrix of shape ``(n_balls + 1, n_balls + 1)``. Element ``[i, j]`` is the probability to go
        from state ``i`` to state ``j``, so each row sums to one.

    Raises
    ------
    ValueError
        If ``n_balls`` is less than 1 or ``p_stay`` is outside ``[0, 1)``.
    """
    if n_balls < 1:
        raise ValueError("need at least one ball")
    if not 0 <= p_stay < 1:
        raise ValueError("p_stay must be in [0, 1)")
    k = np.arange(n_balls + 1)
    matrix = np.zeros((n_balls + 1, n_balls + 1))
    matrix[k[1:], k[1:] - 1] = k[1:] / n_balls
    matrix[k[:-1], k[:-1] + 1] = (n_balls - k[:-1]) / n_balls
    matrix *= 1 - p_stay
    matrix[k, k] += p_stay
    return matrix


def state_distribution(initial: int | np.ndarray, matrix: np.ndarray, n_turns: int) -> np.ndarray:
    """
    Probability of each state after ``n_turns`` turns of a Markov chain.

    Parameters
    ----------
    initial : int or np.ndarray
        The starting state, or a probability for each starting state.
    matrix : np.ndarray
        Transition matrix, rows summing to one.
    n_turns : int
        Number of turns, not negative.

    Returns
    -------
    np.ndarray
        Probability of each state, shape ``(n_states,)``.
    """
    start = np.zeros(len(matrix))
    if np.ndim(initial) == 0:
        start[initial] = 1.0
    else:
        start = np.asarray(initial, dtype=float)
    return start @ np.linalg.matrix_power(matrix, n_turns)


def simulate_markov(
    matrix: np.ndarray,
    initial_state: int,
    n_turns: int,
    n_runs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate many runs of a Markov chain at once.

    Parameters
    ----------
    matrix : np.ndarray
        Transition matrix, rows summing to one.
    initial_state : int
        State every run starts in.
    n_turns : int
        Number of turns.
    n_runs : int
        Number of independent runs.
    rng : np.random.Generator
        Source of random numbers, for reproducibility.

    Returns
    -------
    np.ndarray
        The states, shape ``(n_runs, n_turns + 1)``. Column 0 is ``initial_state``.
    """
    cumulative = np.cumsum(matrix, axis=1)
    paths = np.empty((n_runs, n_turns + 1), dtype=int)
    paths[:, 0] = initial_state
    for turn in range(n_turns):  # each turn depends on the one before
        draw = rng.random(n_runs)[:, np.newaxis]
        next_state = (draw > cumulative[paths[:, turn]]).sum(axis=1)
        paths[:, turn + 1] = np.minimum(next_state, len(matrix) - 1)  # guards rounding in the sums
    return paths


def mean_hitting_time(matrix: np.ndarray, target: int) -> np.ndarray:
    """
    Average number of turns to first reach ``target``, from each starting state.

    With ``t`` the times, ``t = 1 + Q t`` where ``Q`` is the matrix without the target state.

    Parameters
    ----------
    matrix : np.ndarray
        Transition matrix, rows summing to one.
    target : int
        The state to reach.

    Returns
    -------
    np.ndarray
        Average number of turns from each state, shape ``(n_states,)``. Zero for the target.
    """
    others = np.arange(len(matrix)) != target
    transient = matrix[np.ix_(others, others)]
    times = np.zeros(len(matrix))
    times[others] = np.linalg.solve(np.eye(others.sum()) - transient, np.ones(others.sum()))
    return times
