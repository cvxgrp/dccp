"""Tests for DPP compliance and benchmarking."""

import contextlib
import time

import cvxpy as cp
import numpy as np

from dccp.linearize import LinearizationData, linearize


def test_linearize_dpp_basic() -> None:
    """Test that linearize with cache returns parameterized expression."""
    x = cp.Variable(2)
    x.value = np.array([1.0, 2.0])
    expr = x[0] ** 2 + x[1] ** 2

    cache = {}
    lin_expr = linearize(expr, linearization_map=cache)

    assert len(cache) == 1
    assert isinstance(lin_expr, cp.Expression)
    # Check that it contains parameters
    assert len(lin_expr.parameters()) > 0

    data = cache[id(expr)]
    assert isinstance(data, LinearizationData)
    # Check that it contains offset
    assert data.offset is not None
    assert isinstance(data.offset, cp.Parameter)

    # We simplified LinearizationData to not store base_value,
    # as offset = base_value - dot(grad, x0)
    # At x0=[1,2], f(x0)=5, grad=[2,4], offset = 5 - (2*1 + 4*2) = 5 - 10 = -5
    assert np.allclose(data.offset.value, -5.0)


def test_linearization_data_update() -> None:
    """Test that updating LinearizationData reflects in the expression."""
    x = cp.Variable()
    x.value = 2.0
    expr = x**2

    cache = {}
    lin_expr = linearize(expr, linearization_map=cache)

    # Current linearization at x=2:
    # f(x) ~ f(x0) + f'(x0)(x - x0) = 4 + 4(x - 2)
    assert np.isclose(lin_expr.value, 4.0)  # with x.value=2

    # Move x to 3, but parameters are still at x0=2
    x.value = 3.0
    # The affine approximation at x0=2 evaluated at x=3: 4 + 4(3-2) = 8
    assert np.isclose(lin_expr.value, 8.0)

    # Now update parameters to x0=3
    cache[id(expr)].update()

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
        with contextlib.suppress(Exception):
            prob.get_problem_data(cp.SCS)
    end_time = time.time()
    rebuild_time = end_time - start_time

    # --- DPP: Update parameters ---
    cache = {}
    lin_dpp = linearize(expr, linearization_map=cache)  # Initial build
    prob_dpp = cp.Problem(cp.Minimize(lin_dpp))
    assert prob_dpp.is_dcp(dpp=True)  # Confirm it is DPP compliant

    # Pre-compile
    with contextlib.suppress(Exception):
        prob_dpp.get_problem_data(cp.SCS)

    data = cache[id(expr)]

    start_time = time.time()
    for _ in range(iterations):
        x.value += 0.01
        data.update()  # Just update parameters
        with contextlib.suppress(Exception):
            prob_dpp.get_problem_data(cp.SCS)
    end_time = time.time()
    update_time = end_time - start_time

    print(f"\nRebuild time (incl. canonicalization): {rebuild_time:.4f}s")  # noqa: T201
    print(f"Update time (incl. canonicalization):  {update_time:.4f}s")  # noqa: T201
    print(f"Speedup:      {rebuild_time / update_time:.2f}x")  # noqa: T201
