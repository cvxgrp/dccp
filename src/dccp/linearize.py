"""Linearization of cvxpy expressions."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from dccp.utils import ORDER


@dataclass
class LinearizationData:
    """Cache for linearization parameters of an expression.

    Attributes
    ----------
    grads : dict[cp.Variable, cp.Parameter]
        A mapping from variables to their gradients at the linearization point.
    fx0 : cp.Parameter
        The function value term f(x0) for the linearization.
    grad_dot_x0 : cp.Parameter
        The inner-product term <grad, x0> evaluated at the linearization point.
    expr : cp.Expression
        The original expression being linearized.
    tangent_expr : cp.Expression | None
        The cached tangent expression (constructed from parameters).

    """

    grads: dict[cp.Variable, cp.Parameter]
    fx0: cp.Parameter
    grad_dot_x0: cp.Parameter
    expr: cp.Expression
    tangent_expr: cp.Expression | None = None

    @property
    def offset(self) -> cp.Expression:
        """Compatibility alias exposing the legacy offset term f(x0) - <grad, x0>."""
        return self.fx0 - self.grad_dot_x0

    @staticmethod
    def _coerce_param_value(shape: tuple[int, ...], value: object) -> object:
        """Convert value to a form accepted by cvxpy.Parameter for given shape."""
        if sp.issparse(value):
            arr = np.asarray(value.todense())
        else:
            arr = np.asarray(value)
        if shape == () and np.ndim(arr) > 0 and arr.size == 1:
            return arr.item()
        if shape == ():
            return arr
        return np.reshape(arr, shape, order=ORDER)

    @staticmethod
    def _has_sparse_pattern_mismatch(
        g_sparse: sp.spmatrix | sp.sparray, sparse_idx: tuple[np.ndarray, np.ndarray]
    ) -> bool:
        """Return True if gradient has nonzeros outside parameter sparsity pattern."""
        g_coo = g_sparse.tocoo()
        grad_nnz = set(zip(g_coo.row.tolist(), g_coo.col.tolist(), strict=False))
        pattern_nnz = set(
            zip(sparse_idx[0].tolist(), sparse_idx[1].tolist(), strict=False)
        )
        return not grad_nnz.issubset(pattern_nnz)

    @staticmethod
    def _assign_sparse_param_value(
        param_grad: cp.Parameter,
        g: sp.spmatrix | sp.sparray,
    ) -> None:
        """Assign sparse gradient to a sparse CVXPY parameter."""
        sparse_idx = param_grad.sparse_idx
        if sparse_idx is None:
            msg = "Sparse assignment requested for a dense parameter."
            raise ValueError(msg)
        if LinearizationData._has_sparse_pattern_mismatch(g, sparse_idx):
            msg = "Gradient sparsity pattern changed outside cached DPP structure."
            raise ValueError(msg)

        g_coo = g.tocoo()
        value_map = {
            (row, col): val
            for row, col, val in zip(
                g_coo.row.tolist(),
                g_coo.col.tolist(),
                g_coo.data.tolist(),
                strict=False,
            )
        }
        values = np.fromiter(
            (
                value_map.get((row, col), 0.0)
                for row, col in zip(sparse_idx[0], sparse_idx[1], strict=False)
            ),
            dtype=float,
            count=len(sparse_idx[0]),
        )
        sparse_val = sp.coo_array((values, sparse_idx), shape=param_grad.shape)
        param_grad.value_sparse = sparse_val

    @staticmethod
    def _transpose_numeric(value: object) -> object:
        """Transpose numeric/sparse arrays while preserving sparse type."""
        return value.transpose() if sp.issparse(value) else np.transpose(value)

    @staticmethod
    def _add_term(dot_product: float | np.ndarray, term: object) -> float | np.ndarray:
        """Accumulate one term into dot_product, avoiding full sparse densification."""
        if not sp.issparse(term):
            return dot_product + term

        term_coo = term.tocoo()
        if np.isscalar(dot_product):
            return float(dot_product) + float(np.sum(term_coo.data))

        updated = dot_product.copy()
        updated[term_coo.row, term_coo.col] += term_coo.data
        return updated

    @staticmethod
    def _assign_gradient_parameter(param_grad: cp.Parameter, grad_value: object) -> object:
        """Assign gradient to parameter and return value for numeric use."""
        if sp.issparse(grad_value) and param_grad.sparse_idx is not None:
            LinearizationData._assign_sparse_param_value(param_grad, grad_value)
            return grad_value

        dense_or_scalar = LinearizationData._coerce_param_value(
            param_grad.shape, grad_value
        )
        param_grad.value = dense_or_scalar
        return dense_or_scalar

    def _dot_term(self, var: cp.Variable, grad_value: object) -> object:
        """Return <grad, x0> contribution for one variable at current value."""
        if var.ndim > 1:
            temp = var.value.reshape(-1, 1, order=ORDER)
            flattened = self._transpose_numeric(grad_value) @ temp
            return flattened.reshape(self.expr.shape, order=ORDER)
        if var.size > 1:
            return self._transpose_numeric(grad_value) @ var.value
        return grad_value * var.value

    def update(self) -> None:
        """Update the parameters with current variable values."""
        if self.expr.value is None:
            msg = "Expression value is None"
            raise ValueError(msg)

        # Fetch gradient map
        grad_map = self.expr.grad

        # Calculate term <grad, x0>
        if self.expr.shape == ():
            dot_product: float | np.ndarray = 0.0
        else:
            dot_product = np.zeros(self.expr.shape)

        for var, param_grad in self.grads.items():
            g = grad_map[var]
            if g is None:
                msg = f"Gradient for {var.name()} is None"
                raise ValueError(msg)

            grad_value = self._assign_gradient_parameter(param_grad, g)

            # Accumulate <grad, var_val>
            if var.value is not None:
                dot_product = self._add_term(dot_product, self._dot_term(var, grad_value))

        self.fx0.value = self._coerce_param_value(self.fx0.shape, self.expr.value)
        self.grad_dot_x0.value = self._coerce_param_value(
            self.grad_dot_x0.shape, dot_product
        )


def _linearize_param(
    expr: cp.Expression, linearization_map: dict[int, LinearizationData]
) -> cp.Expression | None:
    """DPP Path: Linearize using cached parameters."""
    if id(expr) in linearization_map:
        return linearization_map[id(expr)].tangent_expr

    grad_map = expr.grad
    param_grads = {}

    # Create function-value and inner-product parameters matching expression shape
    param_fx0 = cp.Parameter(expr.shape)
    param_grad_dot_x0 = cp.Parameter(expr.shape)
    tangent = param_fx0 - param_grad_dot_x0

    for var in expr.variables():
        g = grad_map[var]
        if g is None:
            return None

        # Create parameter matching gradient shape
        if sp.issparse(g):
            g_coo = g.tocoo()
            param = cp.Parameter(g.shape, sparsity=(g_coo.row, g_coo.col))
        else:
            param = cp.Parameter(g.shape)
        param_grads[var] = param

        # Build term (inlined logic)
        if var.ndim > 1:
            temp = cp.reshape(
                cp.vec(var, order=ORDER),
                (var.shape[0] * var.shape[1], 1),
                order=ORDER,
            )
            flattened = cp.transpose(param) @ temp
            term = cp.reshape(flattened, expr.shape, order=ORDER)
        elif var.size > 1:
            term = cp.transpose(param) @ var
        else:
            term = param * var

        tangent = tangent + term

    # Store in cache
    data = LinearizationData(param_grads, param_fx0, param_grad_dot_x0, expr, tangent)
    linearization_map[id(expr)] = data

    # Populate initial values
    data.update()
    return tangent


def linearize(
    expr: cp.Expression, linearization_map: dict[int, LinearizationData] | None = None
) -> cp.Expression | None:
    """Return the tangent approximation to the expression.

    Linearize non-convex CVXPY expressions using first-order Taylor expansion around
    given points. The linearization approximates a function by:

    .. math::
        f(x) ≈ f(x_0) + ∇f(x_0)^T(x - x_0)

    Where :math:`x_0` is the point of linearization, :math:`f(x_0)` is the function
    value at that point, and :math:`∇f(x_0)` is the gradient at that point.

    Parameters
    ----------
    expr : cvxpy.Expression
        An expression to linearize.
    linearization_map : dict, optional
        A dictionary to cache linearization parameters. If provided, repeated calls
        for the same expression reuse parameters for in-place updates. If omitted,
        a temporary cache is used for this call.

    Returns
    -------
    cvxpy.Expression
        An affine expression representing the tangent approximation.

    Raises
    ------
    ValueError
        If the expression is non-affine and has missing variable values.

    """
    expr_str = f"Affected expression [{expr.name()}]: {expr}."
    if expr.is_complex() or any(v.is_complex() for v in expr.variables()):
        msg = (
            "Linearization does not support complex variables or expressions. "
            f"Please use real-valued expressions and variables. {expr_str}"
        )
        raise ValueError(msg)
    if expr.is_affine():
        return expr
    if expr.parameters():
        msg = (
            "Linearization does not support user-defined parameters in non-convex "
            f"expressions. Remove any such parameters before linearizing. {expr_str}"
        )
        raise ValueError(msg)

    if expr.value is None:
        return None

    cache = linearization_map if linearization_map is not None else {}
    return _linearize_param(expr, cache)
