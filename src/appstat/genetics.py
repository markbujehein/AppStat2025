"""Probabilities of genotypes from measured gene copies, by Bayes' theorem."""

lazy import numpy as np

# Genotype order everywhere: AA, AB, BB. Base 0 is A, base 1 is B.
_COPY_PROBABILITY = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])  # P(true base | genotype)


def genotype_posterior(reads, measurement: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """
    Probability of each genotype given a series of measured gene copies.

    Each read looks at one of the two copies, picked at random (so AB gives A or B with
    probability 1/2), and the machine may measure it wrongly.

    Parameters
    ----------
    reads : array_like of int
        Measured bases, 0 for A and 1 for B. May be empty.
    measurement : np.ndarray
        Shape ``(2, 2)``. Element ``[i, j]`` is the probability to measure base ``j`` when the
        true base is ``i``. Each row sums to one.
    prior : np.ndarray
        Probability of AA, AB and BB in the population, summing to one.

    Returns
    -------
    np.ndarray
        Probability of AA, AB and BB, summing to one.

    Raises
    ------
    ValueError
        If ``prior`` does not sum to one, or the reads are impossible for it.
    """
    prior = np.asarray(prior, dtype=float)
    if not np.isclose(prior.sum(), 1.0):
        raise ValueError("the prior must sum to one")
    measured_given_genotype = _COPY_PROBABILITY @ measurement  # shape (3, 2)
    reads = np.asarray(reads, dtype=int)
    likelihood = np.prod(measured_given_genotype[:, reads], axis=1)  # independent reads
    posterior = prior * likelihood
    if posterior.sum() == 0:
        raise ValueError("these reads are impossible for this prior and measurement")
    return posterior / posterior.sum()


def simulate_reads(
    genotype: int, n_reads: int, measurement: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """
    Measured bases of ``n_reads`` reads of one genotype (0 is AA, 1 is AB, 2 is BB).

    Returns
    -------
    np.ndarray
        ``n_reads`` values, 0 for A and 1 for B.
    """
    true_bases = (rng.random(n_reads) < _COPY_PROBABILITY[genotype, 1]).astype(int)
    return (rng.random(n_reads) < measurement[true_bases, 1]).astype(int)
