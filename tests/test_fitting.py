"""Tests for appstat.fitting. Remove the skip marker once the function exists."""

import numpy as np
import pytest
from iminuit import cost

# from appstat.fitting import ...


@pytest.mark.skip(reason="TODO: decide what appstat.fitting provides")
def test_least_squares_equals_manual_chi2():
    x = np.arange(10.0)
    y = 2 * x + 1 + np.sin(x)
    sy = np.full(10, 0.5)
    model = lambda x, a, b: a + b * x  # noqa: E731
    manual = np.sum(((y - model(x, 1.0, 2.0)) / sy) ** 2)
    assert cost.LeastSquares(x, y, sy, model)(1.0, 2.0) == pytest.approx(manual)
