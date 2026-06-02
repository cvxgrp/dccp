"""Unit tests for DCCP example problems."""

from types import SimpleNamespace
from typing import cast

import cvxpy as cp
import numpy as np
import pytest
import scipy.sparse as sp

from dccp import linearize
from dccp.linearize import GradientSparsityPatternError, LinearizationData
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
        """Test LinearizationData.update preserves sparse gradients."""
        x = cp.Variable(3, name="x_vec")
        x.value = np.array([1.0, 2.0, 3.0])
        expr = cast("cp.Expression", cp.sum(cp.square(x)))
        grad = expr.grad[x]
        assert isinstance(grad, sp.csc_array)

        rows, cols = grad.nonzero()
        param_grad = cp.Parameter(shape=grad.shape, sparsity=(rows, cols))
        grads = {x: param_grad}
        offset = cp.Parameter(shape=())

        data = LinearizationData(grads, offset, expr)
        data.update()

        sparse_value = param_grad.value_sparse
        assert sparse_value is not None
        assert isinstance(sparse_value, sp.coo_array)
        assert np.array_equal(sparse_value.row, rows)
        assert np.array_equal(sparse_value.col, cols)
        assert_almost_equal(sparse_value.toarray(), grad.toarray())

    def test_linearize_creates_sparse_gradient_parameters(self) -> None:
        """Test cached linearization parameters preserve sparse gradients."""
        x = cp.Variable(3, name="x_vec")
        x.value = np.array([1.0, 2.0, 3.0])
        expr = cast("cp.Expression", cp.sum(cp.square(x)))
        cache = {}

        lin = linearize(expr, cache)

        assert lin is not None
        param_grad = cache[id(expr)].grads[x]
        assert getattr(param_grad, "sparse_idx", None) is not None
        assert sp.issparse(param_grad.value_sparse)

    def test_sparse_vector_gradient_offset_accumulation(self) -> None:
        """Test vector sparse gradients update offsets without densifying params."""
        x = cp.Variable(3, name="x_vec")
        x.value = np.array([1.0, 2.0, 3.0])
        expr = cast("cp.Expression", cp.sum(cp.square(x)))
        cache = {}

        lin = linearize(expr, cache)

        assert lin is not None
        assert lin.value is not None
        assert expr.value is not None
        assert_almost_equal(lin.value, expr.value)
        param_grad = cache[id(expr)].grads[x]
        assert sp.issparse(param_grad.value_sparse)
        assert_almost_equal(cache[id(expr)].offset.value, -14.0)

    def test_sparse_matrix_gradient_offset_accumulation(self) -> None:
        """Test matrix sparse gradients update offsets without densifying params."""
        x = cp.Variable((2, 2), name="x_mat")
        x.value = np.array([[1.0, 2.0], [3.0, 4.0]])
        expr = cp.square(x)
        cache = {}

        lin = linearize(expr, cache)

        assert lin is not None
        assert lin.value is not None
        assert x.value is not None
        assert expr.value is not None
        assert_almost_equal(lin.value, expr.value)
        param_grad = cache[id(expr)].grads[x]
        assert sp.issparse(param_grad.value_sparse)
        assert_almost_equal(cache[id(expr)].offset.value, -(x.value**2))

    def test_sparse_gradient_pattern_change_raises(self) -> None:
        """Test sparse updates detect new nonzeros outside the cached pattern."""
        x = cp.Variable(3, name="x_vec")
        x.value = np.array([0.0, 2.0, 0.0])
        expr = cast("cp.Expression", cp.sum(cp.square(x)))
        cache = {}

        assert linearize(expr, cache) is not None

        x.value = np.array([1.0, 2.0, 3.0])
        with pytest.raises(GradientSparsityPatternError):
            cache[id(expr)].update()

    def test_vector_expr_scalar_variable_offset_shape(self) -> None:
        """Regression test for #127/#129: offset value must match the param shape.

        A vector-shaped expression depending on a scalar variable produces a
        gradient term of shape ``(1, n)``. Subtracting it from the expression
        value (shape ``(n,)``) broadcasts ``val`` to ``(1, n)``, which does not
        match the ``(n,)`` offset parameter. Without reshaping the offset value,
        cvxpy rejects it with "Invalid dimensions (1, n) for Parameter value".
        """
        x = cp.Variable(name="scalar")
        x.value = 2.0
        a = np.array([1.0, 2.0, 3.0])
        # shape (3,) expression that is nonlinear in the scalar variable x
        expr = cp.square(cp.multiply(a, x))
        assert expr.shape == (3,)

        # Without the offset reshape this raises:
        # "ValueError: Invalid dimensions (1, 3) for Parameter value".
        lin = linearize(expr)

        assert lin is not None
        assert lin.value is not None
        assert expr.value is not None
        # Tangent at the linearization point equals the function value there.
        assert_almost_equal(
            np.asarray(lin.value).reshape(-1), np.asarray(expr.value).reshape(-1)
        )

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
