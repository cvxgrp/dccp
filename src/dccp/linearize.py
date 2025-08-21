"""Linearization of cvxpy expressions."""

import cvxpy as cp
import numpy as np

from dccp.utils import ORDER

PARAMS_X0: dict[cp.Variable, cp.Parameter] = {}


def linearize(expr: cp.Expression) -> cp.Expression | None:
    """Return the tangent approximation to the expression.

    linearize non-convex CVXPY expressions using first-order Taylor expansion around
    given points. The linearization approximates a function:
    f(x) ≈ f(x₀) + ∇f(x₀)ᵀ(x - x₀).
         = (f(x₀) - ∇f(x₀)ᵀx₀) + ∇f(x₀)ᵀx
         = (  cp.Parameter   ) + (cp.Parameter)ᵀ @ x

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
    expr_str = f"Affected expression [{expr.name()}]: {expr}."
    if expr.is_affine():
        return expr
    if expr.is_complex() or any(v.is_complex() for v in expr.variables()):
        msg = (
            "Linearization does not support complex variables or expressions. "
            f"Please use real-valued expressions and variables. {expr_str}"
        )
        raise ValueError(msg)
    if expr.parameters():
        msg = (
            "Linearization does not support user-defined parameters in non-convex "
            f"expressions. Remove any such parameters before linearizing. {expr_str}"
        )
        raise ValueError(msg)
    if expr.value is None:
        msg = (
            "Cannot linearize non-affine expression with missing variable values. "
            f"{expr_str}"
        )
        raise ValueError(msg)

    # create linearization parameters for variables if not already present
    for var in expr.variables():
        if var in PARAMS_X0:
            continue
        PARAMS_X0[var] = cp.CallbackParam(
            lambda var=var: var.value, var.shape, name=f"x0_{var.id}"
        )

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
