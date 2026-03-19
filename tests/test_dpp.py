"""Tests for DPP compliance and benchmarking."""

import logging
import time

import cvxpy as cp
import numpy as np

from dccp.linearize import LinearizationData, linearize

_logger = logging.getLogger(__name__)


def test_linearization_data_update() -> None:
    """Test DPP cache creation and parameter updates reflect in expression value."""
    x = cp.Variable()
    x.value = 2.0
    expr = x**2

    cache = {}
    lin_expr = linearize(expr, linearization_map=cache)

    assert len(cache) == 1
    assert len(lin_expr.parameters()) > 0
    data = cache[id(expr)]
    assert isinstance(data, LinearizationData)
    assert np.isclose(data.offset.value, -4.0)

    # Current linearization at x=2:
    # f(x) ~ f(x0) + f'(x0)(x - x0) = 4 + 4(x - 2)
    assert np.isclose(lin_expr.value, 4.0)  # with x.value=2

    # Move x to 3, but parameters are still at x0=2
    x.value = 3.0
    # The affine approximation at x0=2 evaluated at x=3: 4 + 4(3-2) = 8
    assert np.isclose(lin_expr.value, 8.0)

    # Now update parameters to x0=3
    data.update()

    # New linearization at x=3:
    # f(x) ~ 9 + 6(x - 3)
    # Evaluated at x=3, should be 9
    assert np.isclose(lin_expr.value, 9.0)


def test_benchmark_dpp_vs_rebuild() -> None:
    """Benchmark DPP parameter update vs rebuilding expression."""
    n = 100
    x = cp.Variable(n)
    rng = np.random.default_rng()
    x.value = rng.standard_normal(n)
    expr = cp.sum_squares(x)

    iterations = 20

    # --- Non-DPP: Rebuild expression every time ---
    start_time = time.time()
    for _ in range(iterations):
        x.value += 0.01
        lin = linearize(expr)
        prob = cp.Problem(cp.Minimize(lin))
        prob.get_problem_data(cp.CLARABEL)
    end_time = time.time()
    rebuild_time = end_time - start_time

    # --- DPP: Update parameters ---
    cache = {}
    lin_dpp = linearize(expr, linearization_map=cache)  # Initial build
    prob_dpp = cp.Problem(cp.Minimize(lin_dpp))
    assert prob_dpp.is_dcp(dpp=True)  # Confirm it is DPP compliant

    # Pre-compile
    prob_dpp.get_problem_data(cp.CLARABEL)

    data = cache[id(expr)]

    start_time = time.time()
    for _ in range(iterations):
        x.value += 0.01
        data.update()  # Just update parameters
        prob_dpp.get_problem_data(cp.CLARABEL)
    end_time = time.time()
    update_time = end_time - start_time

    print(f"\nRebuild time (incl. canonicalization): {rebuild_time:.4f}s")  # noqa: T201
    print(f"Update time (incl. canonicalization):  {update_time:.4f}s")  # noqa: T201
    print(f"Speedup:      {rebuild_time / update_time:.2f}x")  # noqa: T201
