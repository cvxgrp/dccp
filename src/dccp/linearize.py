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
    offset : cp.Parameter
        The bias term (f(x0) - <grad, x0>) for the linearization.
    expr : cp.Expression
        The original expression being linearized.
    tangent_expr : cp.Expression | None
        The cached tangent expression (constructed from parameters).

    """

    grads: dict[cp.Variable, cp.Parameter]
    offset: cp.Parameter
    expr: cp.Expression
    tangent_expr: cp.Expression | None = None

    def update(self) -> None:
        """Update the parameters with current variable values."""
        if self.expr.value is None:
            msg = "Expression value is None"
            raise ValueError(msg)

        # Fetch gradient map
        grad_map = self.expr.grad

        # Calculate term <grad, x0>
        dot_product = 0.0

        for var, param_grad in self.grads.items():
            g = grad_map[var]
            if g is None:
                msg = f"Gradient for {var.name()} is None"
                raise ValueError(msg)

            # The CVXPY parameter always receives a dense value.  Assigning a
            # sparse scipy array to a plain cp.Parameter causes the DPP
            # canonicalization path to call toarray() internally, which is
            # significantly slower than keeping the parameter dense.
            param_grad.value = g.toarray() if sp.issparse(g) else g

            # Accumulate <grad, var_val> using the original (possibly sparse) g
            # so that the dot-product is O(nnz) rather than O(n) for sparse
            # gradients.  scipy sparse @ dense always returns a dense result,
            # so no explicit densification is needed for the vector/matrix cases.
            if var.value is not None:
                if var.ndim > 1:
                    # Matrix variable: flatten to column vector first.
                    temp = var.value.reshape(-1, 1, order=ORDER)
                    g_t = g.T if sp.issparse(g) else np.transpose(g)
                    flattened = g_t @ temp
                    term = np.reshape(flattened, self.expr.shape, order=ORDER)
                elif var.size > 1:
                    # Vector variable.
                    g_t = g.T if sp.issparse(g) else np.transpose(g)
                    term = g_t @ var.value
                else:
                    # Scalar variable: CVXPY always returns float grad here.
                    term = g * var.value

                dot_product += term

        # Update offset: f(x0) - <grad, x0>
        val = self.expr.value - dot_product
        if self.expr.shape == () and np.ndim(val) > 0 and val.size == 1:
            val = val.item()
        self.offset.value = val


def _linearize_param(
    expr: cp.Expression, linearization_map: dict[int, LinearizationData]
) -> cp.Expression | None:
    """DPP Path: Linearize using cached parameters."""
    if id(expr) in linearization_map:
        return linearization_map[id(expr)].tangent_expr

    grad_map = expr.grad
    param_grads = {}

    # Create one offset parameter matching expression shape
    param_offset = cp.Parameter(expr.shape)
    tangent = param_offset

    for var in expr.variables():
        g = grad_map[var]
        if g is None:
            return None

        # Create a dense parameter for the gradient.  CVXPY's DPP
        # canonicalization path is faster with dense parameter values;
        # sparsity of the gradient is only exploited in update() arithmetic.
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
    data = LinearizationData(param_grads, param_offset, expr, tangent)
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
