"""Linearize.py"""

import cvxpy as cp
import numpy as np


def linearize(expr):
    """Returns the tangent approximation to the expression.

    Gives an elementwise lower (upper) bound for convex (concave)
    expressions. No guarantees for non-DCP expressions.

    Args:
        expr: An expression.

    Returns:
        An affine expression.
    """
    if expr.is_affine():
        return expr
    else:
        if expr.value is None:
            raise ValueError(
                "Cannot linearize non-affine expression with missing variable values."
            )
        tangent = expr.value
        grad_map = expr.grad
        for var in expr.variables():
            if grad_map[var] is None:
                return None
            if var.ndim > 1:
                temp = cp.reshape(
                    cp.vec(var - var.value), (var.shape[0] * var.shape[1], 1)
                )
                flattened = np.transpose(grad_map[var]) @ temp
                tangent = tangent + cp.reshape(flattened, expr.shape)
            elif var.size > 1:
                tangent = tangent + np.transpose(grad_map[var]) @ (var - var.value)
            else:
                tangent = tangent + grad_map[var] * (var - var.value)
        return tangent
