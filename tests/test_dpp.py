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


def test_benchmark_sparse_gradient_update() -> None:  # noqa: PLR0915
    """Benchmark sparse SPMV vs dense BLAS for the update() dot-product.

    update() calls toarray() once (CVXPY canonicalization requires a dense
    parameter value) and then reuses the original sparse gradient for the
    offset dot-product via SPMV — O(nnz) instead of O(n).

    The CVXPY gradient-computation overhead hides the arithmetic difference
    at small n.  Part 2 isolates the arithmetic at large n (50 000, 0.5%
    fill) where sparse SPMV is significantly faster than dense BLAS.
    """
    # ------------------------------------------------------------------ #
    # Part 1: full DPP update() at small n — sparse vs dense dot-product  #
    # ------------------------------------------------------------------ #
    n_small = 500
    step_small = 50  # 2% fill => ~10 nonzeros
    active_small = np.arange(0, n_small, step_small)

    rng = np.random.default_rng(42)
    x = cp.Variable(n_small)
    x.value = rng.standard_normal(n_small)
    expr = cp.sum_squares(x[active_small])

    cache: dict = {}
    lin_dpp = linearize(expr, linearization_map=cache)
    prob_dpp = cp.Problem(cp.Minimize(lin_dpp))
    with contextlib.suppress(Exception):
        prob_dpp.get_problem_data(cp.SCS)
    data = cache[id(expr)]

    g_sample = expr.grad[x]
    assert sp.issparse(g_sample), "Expected sparse gradient for indexed expression"

    iterations = 30

    # Sparse path (current): toarray for param, SPMV for dot — O(nnz).
    start = time.perf_counter()
    for _ in range(iterations):
        x.value = rng.standard_normal(n_small)
        data.update()
        with contextlib.suppress(Exception):
            prob_dpp.get_problem_data(cp.SCS)
    sparse_time = time.perf_counter() - start

    # Dense path (baseline): toarray for param AND dot — O(n).
    # Simulate what a dense-only implementation would do.
    param_grad_d = cp.Parameter((n_small, 1))
    offset_param_d = cp.Parameter(())
    lin_dense = cp.transpose(param_grad_d) @ x + offset_param_d
    prob_dense = cp.Problem(cp.Minimize(lin_dense))
    with contextlib.suppress(Exception):
        prob_dense.get_problem_data(cp.SCS)

    start = time.perf_counter()
    for _ in range(iterations):
        x.value = rng.standard_normal(n_small)
        g = expr.grad[x]
        g_d = g.toarray()
        param_grad_d.value = g_d
        offset_param_d.value = float(expr.value) - float(
            (np.transpose(g_d) @ x.value).item()
        )
        with contextlib.suppress(Exception):
            prob_dense.get_problem_data(cp.SCS)
    dense_time = time.perf_counter() - start

    # ------------------------------------------------------------------ #
    # Part 2: isolated arithmetic at large n — sparse wins clearly         #
    # ------------------------------------------------------------------ #
    n_large = 50_000
    nnz_large = 250  # 0.5% fill
    rows = rng.choice(n_large, size=nnz_large, replace=False)
    g_sp = sp.csc_array(
        (rng.standard_normal(nnz_large), (rows, np.zeros(nnz_large, dtype=int))),
        shape=(n_large, 1),
    )
    x_val = rng.standard_normal(n_large)
    arith_iters = 500

    g_d = g_sp.toarray()
    struct_rows = np.sort(rows)
    struct_cols = np.zeros(nnz_large, dtype=int)

    # Sparse path (current): value_sparse lookup + SPMV — no toarray().
    start = time.perf_counter()
    for _ in range(arith_iters):
        np.asarray(g_sp[struct_rows, struct_cols]).flatten()  # O(nnz) lookup
        float((g_sp.T @ x_val).item())  # O(nnz) dot
    sparse_arith = time.perf_counter() - start

    # Dense path (baseline): toarray + BLAS — O(n) for both steps.
    start = time.perf_counter()
    for _ in range(arith_iters):
        gd = g_sp.toarray()  # O(n) allocation
        float((np.transpose(gd) @ x_val).item())  # O(n) dot
    dense_arith = time.perf_counter() - start

    fill_small = 100.0 * g_sample.nnz / n_small
    print(  # noqa: T201
        f"\nSparse-gradient update benchmark"
        f"\n--- Full DPP update() n={n_small}, fill={fill_small:.0f}%"
        f" ({iterations} iters, incl. canonicalization) ---"
        f"\n  Sparse SPMV O(nnz): {sparse_time:.3f}s"
        f"\n  Dense  BLAS O(n)  : {dense_time:.3f}s"
        f"  [{dense_time / sparse_time:.2f}x sparse speedup]"
        f"\n--- Isolated arithmetic n={n_large}, nnz={nnz_large}"
        f" ({arith_iters} iters) ---"
        f"\n  Sparse SPMV O(nnz): {sparse_arith:.3f}s"
        f"\n  Dense  BLAS O(n)  : {dense_arith:.3f}s"
        f"  [{dense_arith / sparse_arith:.2f}x sparse speedup]"
    )
