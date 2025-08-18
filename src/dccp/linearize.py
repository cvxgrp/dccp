"""Linearization of cvxpy expressions."""

import cvxpy as cp
import numpy as np

from dccp.utils import ORDER


def linearize(expr: cp.Expression) -> cp.Expression | None:
    """Return the tangent approximation to the expression.

    Gives an elementwise lower (upper) bound for convex (concave)
    expressions. No guarantees for non-DCP expressions.

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

    Notes
    -----
    This function computes the first-order Taylor approximation of the given
    expression around the current variable values. The linearization is exact
    for affine expressions.

    """
    if expr.is_affine():
        return expr
    if expr.value is None:
        msg = "Cannot linearize non-affine expression with missing variable values."
        raise ValueError(msg)
    tangent = expr.value
    grad_map = expr.grad
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
    if not isinstance(tangent, cp.Expression):
        tangent = cp.Constant(tangent)
    return tangent
