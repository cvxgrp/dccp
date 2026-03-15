"""Linearization of cvxpy expressions."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from dccp.utils import ORDER


@dataclass
class LinearizationData:
    """Cache for linearization parameters of an expression.

    Attributes
    ----------
    grads : dict[cp.Variable, cp.Parameter]
        A mapping from variables to their gradients at the linearization point.
    biases : dict[cp.Variable, cp.Parameter]
        A mapping from variables to their bias contributions (f(x0) component).
    base_value : cp.Parameter
        The value of the expression at the linearization point.
    expr : cp.Expression
        The original expression being linearized.
    tangent_expr : cp.Expression | None
        The cached tangent expression (constructed from parameters).

    """

    grads: dict[cp.Variable, cp.Parameter]
    biases: dict[cp.Variable, cp.Parameter]
    base_value: cp.Parameter
    expr: cp.Expression
    tangent_expr: cp.Expression | None = None

    def _compute_bias(
        self, grad: np.ndarray, x0: np.ndarray, var: cp.Variable
    ) -> float | np.ndarray:
        """Compute the bias term for a variable."""
        # Logic mirroring _linearize_term construction
        if var.ndim > 1:
            # Matrix variable
            temp = x0.reshape(-1, 1, order=ORDER)
            g_t = np.transpose(grad)
            flattened = g_t @ temp
            bias_val = -flattened.reshape(self.expr.shape, order=ORDER)
        elif var.size > 1:
            # Vector variable
            bias_val = -np.transpose(grad) @ x0
        else:
            # Scalar variable
            bias_val = -grad * x0

        if np.ndim(bias_val) > 0 and self.expr.shape == ():
            return float(bias_val.item())
        return bias_val

    def update(self) -> None:
        """Update the parameters with current variable values."""
        if self.expr.value is None:
            return

        self.base_value.value = self.expr.value
        grad_map = self.expr.grad

        for var, param_grad in self.grads.items():
            if grad_map[var] is not None:
                g = grad_map[var]
                param_grad.value = g

                if var.value is not None:
                    self.biases[var].value = self._compute_bias(g, var.value, var)


def _linearize_term(
    expr_shape: tuple[int, ...],
    var: cp.Variable,
    grad: cp.Parameter,
    bias: cp.Parameter,
) -> cp.Expression:
    """Compute the linearized term for a single variable."""
    if var.ndim > 1:
        temp = cp.reshape(
            cp.vec(var, order=ORDER),
            (var.shape[0] * var.shape[1], 1),
            order=ORDER,
        )
        flattened = cp.transpose(grad) @ temp
        linear_term = cp.reshape(flattened, expr_shape, order=ORDER)
        return linear_term + bias

    if var.size > 1:
        return cp.transpose(grad) @ var + bias
    return grad * var + bias


def _linearize_param(
    expr: cp.Expression, linearization_map: dict[int, LinearizationData]
) -> cp.Expression | None:
    """DPP Path: Linearize using cached parameters."""
    if id(expr) in linearization_map:
        return linearization_map[id(expr)].tangent_expr

    grad_map = expr.grad
    param_grads = {}
    param_biases = {}
    param_base_value = cp.Parameter(expr.shape)  # Value populated via update()

    # Create Parameters
    for var in expr.variables():
        if grad_map[var] is None:
            return None

        g = grad_map[var]
        param_grads[var] = cp.Parameter(g.shape)
        param_biases[var] = cp.Parameter(expr.shape)

    # Build Expression Graph (Symbolic only)
    tangent = param_base_value
    for var in expr.variables():
        tangent = tangent + _linearize_term(
            expr.shape, var, param_grads[var], param_biases[var]
        )

    # Store in cache
    data = LinearizationData(
        param_grads, param_biases, param_base_value, expr, tangent
    )
    linearization_map[id(expr)] = data

    # Populate initial values
    data.update()
    return tangent


def _linearize_legacy(expr: cp.Expression) -> cp.Expression | None:
    """Legacy Path: Linearize by rebuilding expression with constants."""
    tangent = expr.value
    try:
        grad_map = expr.grad
    except AttributeError as e:
        msg = (
            f"Cannot compute gradient for expression {expr}. "
            f"This may indicate an incompatible cvxpy version. {expr_str}"
        )
        raise ValueError(msg) from e

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

    return tangent


def linearize(
    expr: cp.Expression, linearization_map: dict[int, LinearizationData] | None = None
) -> cp.Expression | None:
    """Return the tangent approximation to the expression."""
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

    # DPP Path
    if linearization_map is not None:
        return _linearize_param(expr, linearization_map)

    # Legacy Path (Non-DPP, rebuilds expression with constants)
    return _linearize_legacy(expr)

