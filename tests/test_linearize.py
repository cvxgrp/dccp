"""Unit tests for DCCP example problems."""

import cvxpy as cp
import numpy as np
import pytest
import scipy.sparse as sp

from dccp import linearize
from dccp.linearize import LinearizationData
from tests.utils import FakeExpression, assert_almost_equal


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

    def test_linearize_return_none_if_grad_none(self) -> None:
        """Test linearize returns None if gradient is None (linearize.py:88)."""
        x = cp.Variable()
        x.value = 0.0

        # Mock expression behavior to return None for gradient
        mock_expr = FakeExpression(shape=(), value=1.0)
        mock_expr._variables = [x]
        mock_expr.grad = {x: None}
        mock_expr._is_complex = False
        mock_expr._is_affine = False
        mock_expr._parameters_list = []
        mock_expr._name = "mock_expr"

        res = linearize(mock_expr, {})
        assert res is None

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

    def test_linearization_data_update_gradient_none(self) -> None:
        """Test LinearizationData.update raises ValueError when gradient is None."""
        x = cp.Variable(name="x_var")
        x.value = 1.0

        # Mock expression to return None in gradient map
        mock_expr = FakeExpression(shape=(), value=1.0)
        mock_expr.grad = {x: None}

        grads = {x: cp.Parameter(shape=x.shape)}
        offset = cp.Parameter(shape=mock_expr.shape)
        data = LinearizationData(grads, offset, mock_expr)

        with pytest.raises(ValueError, match=r"Gradient for .* is None"):
            data.update()

    def test_linearization_data_update_scalar_array_value(self) -> None:
        """Test LinearizationData.update handles scalar array values correctly."""
        x = cp.Variable()
        x.value = 2.0

        mock_expr = FakeExpression(shape=(), value=np.array([4.0]))
        # Gradient must be scalar/0-d to match param_grad shape (x is scalar)
        mock_expr.grad = {x: np.array(4.0)}

        grads = {x: cp.Parameter(shape=x.shape)}
        offset = cp.Parameter(shape=())

        data = LinearizationData(grads, offset, mock_expr)

        data.update()

        # Offset should be converted to scalar by the logic in linearize.py
        assert np.ndim(offset.value) == 0
        assert offset.value == -4.0

    def test_linearize_cache_hit(self) -> None:
        """Test linearize hits the cache (linearize.py:88 coverage)."""
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
        # x**2 returns vector expression.

        # Mocking a sparse gradient return
        # expr.grad is a dict {var: value}
        # We need to manually construct LinearizationData or mock expr.grad

        # Let's mock expr
        mock_expr = FakeExpression(shape=(), value=14.0)

        # Create a sparse matrix for gradient
        # Safer: use a variable x and set grad to strictly a sparse matrix
        # regardless of shape consistency for this unit test of *branch coverage*.
        mock_expr.grad = {x: sp.coo_matrix([[1], [0], [3]])}

        # Construct LinearizationData
        param_grad = cp.Parameter(shape=(3, 1))  # match sparse shape
        grads = {x: param_grad}
        offset = cp.Parameter(shape=())

        data = LinearizationData(grads, offset, mock_expr)
        data.update()

        # Check if param_grad.value becomes dense
        assert isinstance(param_grad.value, np.ndarray)
        assert not sp.issparse(param_grad.value)

    def test_linearize_vector_variable_branch(self) -> None:
        """Test linearization of expression with vector variables."""
        x = cp.Variable(3)
        x.value = np.array([1.0, 2.0, 3.0])
        # x**2 returns vector expression.
        expr_vec = x**2
        lin_map = {}
        lin_expr = linearize(expr_vec, lin_map)
        assert lin_expr is not None
        assert x in lin_map[id(expr_vec)].grads

        expected_constant = -(x.value**2)
        assert np.allclose(lin_map[id(expr_vec)].offset.value, expected_constant)
