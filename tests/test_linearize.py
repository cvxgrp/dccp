"""Unit tests for DCCP example problems."""

from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import pytest
import scipy.sparse as sp

from dccp import linearize
from dccp.linearize import LinearizationData
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
        offset = cp.Parameter(shape=expr.shape)
        data = LinearizationData(grads, offset, expr)

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
        """Sparse gradient is densified before being assigned to the parameter."""
        x = cp.Variable(3, name="x_vec")
        x.value = np.array([1.0, 2.0, 3.0])
        expr = cp.sum(cp.square(x))
        grad = expr.grad[x]
        assert grad is not None
        assert sp.issparse(grad)

        param_grad = cp.Parameter(shape=grad.shape)
        grads = {x: param_grad}
        offset = cp.Parameter(shape=())

        data = LinearizationData(grads, offset, expr)
        data.update()

        # Parameter always receives a dense value regardless of gradient sparsity.
        assert isinstance(param_grad.value, np.ndarray)
        assert not sp.issparse(param_grad.value)

    def test_linearization_data_update_sparse_gradient_correct_offset(self) -> None:
        """Sparse gradient dot-product uses sparse arithmetic and yields correct offset."""
        x = cp.Variable(4, name="x_vec")
        # all nonzero so the initial gradient has the full structural pattern
        x.value = np.array([1.0, 2.0, 3.0, 4.0])
        expr = cp.sum(cp.square(x))

        cache: dict = {}
        lin = linearize(expr, cache)
        assert lin is not None

        data = cache[id(expr)]
        # At x0=[1,2,3,4]: f=30, grad=[2,4,6,8], <grad,x0>=2+8+18+32=60
        assert np.isclose(data.offset.value, 30.0 - 60.0)

        # Simulate a later iteration where one variable is zero.
        x.value = np.array([5.0, 0.0, 3.0, -1.0])
        data.update()
        # f=35, grad=[10,0,6,-2], <grad,x0>=50+0+18+2=70  →  offset=35-70=-35
        assert np.isclose(data.offset.value, 35.0 - 70.0)

    def test_linearize_dense_params_for_sparse_gradients(self) -> None:
        """linearize() creates dense gradient parameters even for sparse gradients."""
        x = cp.Variable(3, name="x_vec")
        x.value = np.array([1.0, 2.0, 3.0])
        expr = cp.sum(cp.square(x))

        cache: dict = {}
        lin = linearize(expr, cache)
        assert lin is not None

        data = cache[id(expr)]
        for param in data.grads.values():
            # Parameters are always dense for fast DPP canonicalization.
            assert isinstance(param.value, np.ndarray)
            assert not sp.issparse(param.value)

    def test_linearize_dpp_compliance(self) -> None:
        """Linearization produces a DPP-compliant problem that stays compliant after update."""
        x = cp.Variable(10)
        x.value = np.ones(10)
        expr = cp.sum(cp.square(x))

        cache: dict = {}
        lin = linearize(expr, cache)
        assert lin is not None

        prob = cp.Problem(cp.Minimize(lin))
        assert prob.is_dcp(dpp=True)

        x.value = np.ones(10) * 2.0
        cache[id(expr)].update()
        assert prob.is_dcp(dpp=True)

    def test_linearize_matrix_variable(self) -> None:
        """Linearization handles matrix variables (var.ndim > 1 branch in update)."""
        X = cp.Variable((2, 3))
        X.value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        expr = cp.sum(cp.square(X))

        cache: dict = {}
        lin = linearize(expr, cache)
        assert lin is not None

        # At the linearization point the tangent equals the function value.
        assert_almost_equal(float(lin.value), float(expr.value))

        # After moving X, update should refresh the linearization correctly.
        X.value = np.ones((2, 3))
        cache[id(expr)].update()
        assert_almost_equal(float(lin.value), float(expr.value))

    def test_update_dense_gradient_matrix_variable(self) -> None:
        """Dense gradient with matrix variable exercises np.transpose branch in update."""
        X = cp.Variable((2, 2))
        X.value = np.eye(2)

        # Dense (4,1) gradient for a (2,2) variable — non-sparse path.
        g_dense = np.array([[1.0], [0.0], [0.0], [1.0]])
        expr = _expr_stub(
            value=2.0,
            grad={X: g_dense},
            shape=(),
            name="dense_matrix_grad_expr",
        )

        grads = {X: cp.Parameter(shape=(4, 1))}
        offset = cp.Parameter(shape=())
        data = LinearizationData(grads, offset, expr)  # type: ignore[arg-type]
        data.update()

        # g.T @ vec(X) = [1,0,0,1] · [1,0,0,1] = 2.0  →  offset = 2.0 - 2.0 = 0.0
        assert np.isclose(offset.value, 0.0)

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
        offset = cp.Parameter(shape=())
        data = LinearizationData(grads, offset, expr)  # type: ignore[arg-type]

        data.update()

        # Only y contributes to dot product: 2 * 3 = 6, so offset = 5 - 6 = -1
        assert offset.value == -1.0