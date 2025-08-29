"""Linearization of cvxpy expressions."""

import cvxpy as cp
import numpy as np

from dccp.utils import ORDER


def linearize(expr: cp.Expression) -> cp.Expression | None:
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

    # no numeric point to linearize at yet; let caller try damping
    if expr.value is None:
        return None

    tangent = expr.value
    grad_map = expr.grad

    # compute contribution from each variable to the gradients
    for var in expr.variables():
        if grad_map[var] is None:
            return None
        if var.ndim > 1:
            temp = cp.reshape(
                cp.vec(var - var.value, order=ORDER),
                (var.shape[0] * var.shape[1], 1),
                order=ORDER,
            )
            flattened = np.transpose(grad_map[var]) @ temp
            tangent = tangent + cp.reshape(flattened, expr.shape, order=ORDER)
        elif var.size > 1:
            tangent = tangent + np.transpose(grad_map[var]) @ (var - var.value)
        else:
            tangent = tangent + grad_map[var] * (var - var.value)

    return tangent  # type: ignore[reportReturnType]
