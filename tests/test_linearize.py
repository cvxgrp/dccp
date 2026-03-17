"""Unit tests for DCCP example problems."""

from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import pytest
import scipy.sparse as sp

from dccp import linearize
from dccp.linearize import LinearizationData, _linearize_param
from tests.utils import assert_almost_equal


def _expr_stub(
    *,
    value: float | np.ndarray | None,
    grad: dict[cp.Variable, object],
    shape: tuple[int, ...] = (),
    name: str = "expr_stub",
) -> object:
    """Return a minimal expression-like object for branch tests."""
    return SimpleNamespace(
        value=value,
        grad=grad,
        shape=shape,
        name=lambda: name,
        is_complex=lambda: False,
        variables=lambda: list(grad.keys()),
        is_affine=lambda: False,
        parameters=list,
    )


class TestLinearize:
    """Unit test linearization routine."""

    def test_linearize(self) -> None:
        """Test the linearize function."""
        z = cp.Variable((1, 5))
        expr = cp.square(z)
        z.value = np.reshape(np.array([1, 2, 3, 4, 5]), (1, 5))
        lin = linearize(expr)
        assert lin is not None
        assert lin.value is not None
        assert lin.shape == (1, 5)
        assert_almost_equal(lin.value[0], np.array([1, 4, 9, 16, 25]))

    def test_linearize_affine(self) -> None:
        """Test the linearize function on an affine expression."""
        z = cp.Variable((1, 5))
        expr = 3 * z + 2
        z.value = np.reshape(np.array([1, 2, 3, 4, 5]), (1, 5))
        lin = linearize(expr)
        assert lin is not None
        assert lin.value is not None
        assert lin.shape == (1, 5)
        assert_almost_equal(lin.value[0], np.array([5, 8, 11, 14, 17]))
        assert lin.is_affine()

    def test_linearize_complex(self) -> None:
        """Test the linearize function on a complex expression."""
        z = cp.Variable((1, 5), complex=True)
        expr = z + 1
        with pytest.raises(ValueError, match="Linearization does not support complex"):
            linearize(expr)

    def test_user_param(self) -> None:
        """Test the linearize function on an expression with user-defined parameters."""
        z = cp.Variable((1, 5))
        p = cp.Parameter((1, 5), value=np.array([[1, 2, 3, 4, 5]]))
        expr = z**2 + p
        z.value = np.reshape(np.array([1, 2, 3, 4, 5]), (1, 5))
        with pytest.raises(
            ValueError,
            match="Linearization does not support user-defined parameters",
        ):
            linearize(expr)

    def test_user_param_allowed(self) -> None:
        """User defined parameters are allowed in affine parts of the expression."""
        z = cp.Variable((1, 5))
        p = cp.Parameter((1, 5), value=np.array([[1, 2, 3, 4, 5]]))
        expr = 3 * z + p
        z.value = np.reshape(np.array([1, 2, 3, 4, 5]), (1, 5))
        lin = linearize(expr)
        assert lin is not None
        assert lin.value is not None
        assert lin.shape == (1, 5)
        assert_almost_equal(lin.value[0], np.array([4, 8, 12, 16, 20]))
        assert lin.is_affine()

    def test_no_value(self) -> None:
        """Test the linearize function when variable values are not set."""
        z = cp.Variable((1, 5))
        expr = cp.square(z)
        lin = linearize(expr)
        assert lin is None

    def test_linearization_data_update_none_value_lines(self) -> None:
        """Test LinearizationData.update raises ValueError when expr value is None."""
        x = cp.Variable()
        # No value assigned
        expr = x**2

        # Manually create data
        grads = {x: cp.Parameter(shape=x.shape)}
        fx0 = cp.Parameter(shape=expr.shape)
        grad_dot_x0 = cp.Parameter(shape=expr.shape)
        data = LinearizationData(grads, fx0, grad_dot_x0, expr)

        with pytest.raises(ValueError, match="Expression value is None"):
            data.update()

    def test_linearize_cache_hit(self) -> None:
        """Test linearize hits the cache."""
        x = cp.Variable()
        x.value = 1.0
        expr = x**2
        lin_map = {}

        # First call populates cache
        lin1 = linearize(expr, lin_map)
        assert lin1 is not None
        assert id(expr) in lin_map

        # Second call should hit line 88
        lin2 = linearize(expr, lin_map)
        assert lin2 is lin1

    def test_linearization_data_update_sparse_gradient(self) -> None:
        """Test LinearizationData.update handles sparse gradients."""
        x = cp.Variable(3, name="x_vec")
        x.value = np.array([1.0, 2.0, 3.0])
        expr = cp.sum(cp.square(x))
        grad = expr.grad[x]
        assert grad is not None
        assert sp.issparse(grad)

        param_grad = cp.Parameter(shape=grad.shape)
        grads = {x: param_grad}
        fx0 = cp.Parameter(shape=())
        grad_dot_x0 = cp.Parameter(shape=())

        data = LinearizationData(grads, fx0, grad_dot_x0, expr)
        data.update()

        # Check if param_grad.value becomes dense
        assert isinstance(param_grad.value, np.ndarray)
        assert not sp.issparse(param_grad.value)

    def test_linearization_data_update_skips_none_var_value_and_continues(self) -> None:
        """Test update loop continues when one variable has no value."""
        x = cp.Variable(name="x")
        y = cp.Variable(name="y")
        y.value = 3.0

        expr = _expr_stub(
            value=5.0,
            grad={x: np.array(1.0), y: np.array(2.0)},
            shape=(),
            name="partial_value_expr",
        )

        grads = {x: cp.Parameter(shape=()), y: cp.Parameter(shape=())}
        fx0 = cp.Parameter(shape=())
        grad_dot_x0 = cp.Parameter(shape=())
        data = LinearizationData(grads, fx0, grad_dot_x0, expr)  # type: ignore[arg-type]

        data.update()

        # Only y contributes to dot product: 2 * 3 = 6, so offset = 5 - 6 = -1
        assert np.isclose((fx0.value - grad_dot_x0.value), -1.0)

    def test_assign_sparse_param_value_raises_for_dense_parameter(self) -> None:
        """Sparse assignment should fail clearly for dense parameters."""
        param_grad = cp.Parameter((2, 1))
        grad = sp.coo_array(([1.0], ([0], [0])), shape=(2, 1))

        with pytest.raises(
            ValueError, match="Sparse assignment requested for a dense parameter"
        ):
            LinearizationData._assign_sparse_param_value(param_grad, grad)

    def test_assign_sparse_param_value_raises_on_pattern_mismatch(self) -> None:
        """Sparse assignment should fail when gradient pattern leaves cached support."""
        rows = np.array([0])
        cols = np.array([0])
        param_grad = cp.Parameter((2, 1), sparsity=(rows, cols))
        # Nonzero is at (1, 0), outside parameter sparsity pattern {(0, 0)}
        grad = sp.coo_array(([2.0], ([1], [0])), shape=(2, 1))

        with pytest.raises(
            ValueError,
            match="Gradient sparsity pattern changed outside cached DPP structure",
        ):
            LinearizationData._assign_sparse_param_value(param_grad, grad)

    def test_linearize_param_dense_gradient_uses_dense_parameter(self) -> None:
        """Dense gradient path should create dense cvxpy parameters."""
        x = cp.Variable(name="x_dense_grad")
        x.value = 2.0
        expr = _expr_stub(
            value=4.0,
            grad={x: np.array(4.0)},
            shape=(),
            name="dense_grad_expr",
        )

        linearization_map: dict[int, LinearizationData] = {}
        tangent = _linearize_param(expr, linearization_map)  # type: ignore[arg-type]

        assert tangent is not None
        data = linearization_map[id(expr)]
        assert data.grads[x].sparse_idx is None

    def test_update_converts_sparse_term_before_accumulation(self) -> None:
        """Sparse intermediate term is converted before adding into dot product."""
        x = cp.Variable(name="x_scalar_sparse_term")
        x.value = 3.0

        grad = sp.coo_array(([2.0], ([0], [0])), shape=(1, 1))
        expr = _expr_stub(
            value=np.array([[6.0]]),
            grad={x: grad},
            shape=(1, 1),
            name="sparse_term_expr",
        )

        rows = np.array([0])
        cols = np.array([0])
        grads = {x: cp.Parameter((1, 1), sparsity=(rows, cols))}
        fx0 = cp.Parameter((1, 1))
        grad_dot_x0 = cp.Parameter((1, 1))
        data = LinearizationData(grads, fx0, grad_dot_x0, expr)  # type: ignore[arg-type]

        data.update()

        assert np.allclose(fx0.value, np.array([[6.0]]))
        assert np.allclose(grad_dot_x0.value, np.array([[6.0]]))

    def test_add_term_scalar_with_sparse_term(self) -> None:
        """Scalar dot-product accumulation handles sparse terms."""
        term = sp.coo_array(([1.5, 2.5], ([0, 0], [0, 1])), shape=(1, 2))
        updated = LinearizationData._add_term(3.0, term)
        assert np.isclose(updated, 7.0)
