__author__ = "Xinyue"
import cvxpy as cvx
import numpy as np


def linearize_para(expr):
    """
    input:
        expr: an expression
    return:
        linear_expr: linearized expression
        zero_order: zero order parameter
        linear_dictionary: {variable: [value parameter, [gradient parameter]]}
        dom: domain
    """
    zero_order = cvx.Parameter(expr.shape)  # zero order
    linear_expr = zero_order
    linear_dictionary = {}
    for var in expr.variables():
        value_para = cvx.Parameter(var.shape)
        if var.ndim > 1:  # matrix to vector
            gr = []
            for d in range(var.shape[1]):
                g = cvx.Parameter((var.shape[0], expr.shape[0]))
                # g = g.T
                linear_expr += g.T @ (var[:, d] - value_para[:, d])  # first order
                gr.append(g)
            linear_dictionary[var] = [value_para, gr]
        else:  # vector to vector
            g = cvx.Parameter(var.shape[0], expr.shape[0])
            linear_expr += g.T @ (var[:, d] - value_para[:, d])  # first order
            gr.append(g)
        linear_dictionary[var] = [value_para, gr]
    dom = expr.domain
    return linear_expr, zero_order, linear_dictionary, dom


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
        tangent = np.real(expr.value)  # + np.imag(expr.value)
        grad_map = expr.grad
        for var in expr.variables():
            if grad_map[var] is None:
                return None
            complex_flag = False
            if var.is_complex() or np.any(np.iscomplex(grad_map[var])):
                complex_flag = True
            if var.ndim > 1:
                temp = cvx.reshape(
                    cvx.vec(var - var.value), (var.shape[0] * var.shape[1], 1)
                )
                if complex_flag:
                    flattened = np.transpose(np.real(grad_map[var])) @ cvx.real(
                        temp
                    ) + np.transpose(np.imag(grad_map[var])) @ cvx.imag(temp)
                else:
                    flattened = np.transpose(np.real(grad_map[var])) @ temp
                tangent = tangent + cvx.reshape(flattened, expr.shape)
            elif var.size > 1:
                if complex_flag:
                    tangent = (
                        tangent
                        + np.transpose(np.real(grad_map[var]))
                        @ (cvx.real(var) - np.real(var.value))
                        + np.transpose(np.imag(grad_map[var]))
                        @ (cvx.imag(var) - np.imag(var.value))
                    )
                else:
                    tangent = tangent + np.transpose(np.real(grad_map[var])) @ (
                        var - var.value
                    )
            else:
                if complex_flag:
                    tangent = (
                        tangent
                        + np.real(grad_map[var]) * (cvx.real(var) - np.real(var.value))
                        + np.imag(grad_map[var]) * (cvx.imag(var) - np.imag(var.value))
                    )
                else:
                    tangent = tangent + np.real(grad_map[var]) * (var - var.value)
        return tangent
