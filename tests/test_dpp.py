"""Tests for DPP compliance and benchmarking."""

import contextlib
import time

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from dccp.linearize import LinearizationData, linearize


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


def test_sparse_gradient_parameter_preserved_in_dpp_cache() -> None:
    """Test sparse gradients are cached in sparse DPP parameters when possible."""
    x = cp.Variable(5)
    x.value = np.arange(1, 6, dtype=float)
    expr = cp.sum_squares(x)

    cache = {}
    linearize(expr, linearization_map=cache)
    data = cache[id(expr)]

    grad_param = data.grads[x]
    assert grad_param.sparse_idx is not None
    assert sp.issparse(grad_param.value_sparse)


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


def test_benchmark_sparse_update_vs_dense_assignment() -> None:
    """Benchmark sparse-preserving parameter updates against dense assignment."""
    n = 20000
    nnz = 200
    iterations = 200

    rows = np.linspace(0, n - 1, nnz, dtype=int)
    cols = np.zeros(nnz, dtype=int)

    sparse_param = cp.Parameter((n, 1), sparsity=(rows, cols))
    dense_param = cp.Parameter((n, 1))

    # Warm-up both paths once.
    warm_values = np.linspace(1.0, 2.0, nnz)
    warm_grad = sp.coo_array((warm_values, (rows, cols)), shape=(n, 1))
    LinearizationData._assign_sparse_param_value(sparse_param, warm_grad)
    dense_param.value = warm_grad.toarray()

    # Sparse-preserving update path.
    start_time = time.time()
    for i in range(iterations):
        values = np.sin(np.linspace(0.01 * i, 0.01 * i + 1.0, nnz))
        grad = sp.coo_array((values, (rows, cols)), shape=(n, 1))
        LinearizationData._assign_sparse_param_value(sparse_param, grad)
    sparse_update_time = time.time() - start_time

    # Dense baseline path.
    start_time = time.time()
    for i in range(iterations):
        values = np.sin(np.linspace(0.01 * i, 0.01 * i + 1.0, nnz))
        grad = sp.coo_array((values, (rows, cols)), shape=(n, 1))
        dense_param.value = grad.toarray()
    dense_update_time = time.time() - start_time

    print(  # noqa: T201
        f"\nSparse update time (value_sparse path): {sparse_update_time:.4f}s"
    )
    print(  # noqa: T201
        f"Dense update time (toarray path):       {dense_update_time:.4f}s"
    )
    print(  # noqa: T201
        f"Speedup (dense/sparse):                  "
        f"{dense_update_time / sparse_update_time:.2f}x"
    )
