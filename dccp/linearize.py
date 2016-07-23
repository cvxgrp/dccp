__author__ = 'Xinyue'
import numpy as np
from cvxpy import *

def linearize_para(expr):
    '''
    input:
        expr: an expression
    return:
        linear_expr: linearized expression
        zero_order: zero order parameter
        linear_dictionary: {variable: [value parameter, [gradient parameter]]}
        dom: domain
    '''
    zero_order = Parameter(expr.size[0],expr.size[1]) # zero order
    linear_expr = zero_order
    linear_dictionary = {}
    for var in expr.variables():
        value_para = Parameter(var.size[0],var.size[1])
        if var.size[1]>1: # matrix to vector
            gr = []
            for d in range(var.size[1]):
                g = Parameter(var.size[0],expr.size[0])
                # g = g.T
                linear_expr += g.T * (var[:,d] - value_para[:,d]) # first order
                gr.append(g)
            linear_dictionary[var] = [value_para, gr]
        else: # vector to vector
            g = Parameter(var.size[0],expr.size[0])
            linear_expr += g.T * (var[:,d] - value_para[:,d]) # first order
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
        tangent = expr.value
        if tangent is None:
            raise ValueError(
        "Cannot linearize non-affine expression with missing variable values."
            )
        grad_map = expr.grad
        for var in expr.variables():
            if var.size[1] > 1:
                flattened = np.transpose(grad_map[var])*vec(var - var.value)
                tangent = tangent + reshape(flattened, *expr.size)
            else:
                tangent = tangent + np.transpose(grad_map[var])*(var - var.value)
        return tangent

#def linearize(expr):
#        """linearize an expression at a given point
#        """
#        flag = 0
#        flag_var = []
#        line_expr = expr.value
#        for key in expr.gradient:
#            D = expr.gradient[key]
#            rows, cols, imag_rows, imag_cols = D.shape
#            if cols>1: # matrix to vector
#                for d in range(cols):
#                    g = D[:,d,:,0]
#                    g = g.T
#                    line_expr = line_expr + g * (key[:,d] - key.value[:,d])
#            else: # vector to vector
#                g = D[:,0,:,0]
#                g = g.T
#                line_expr = line_expr + g * (key - key.value)
#            if np.any(np.isnan(g)) or np.any(np.isinf(g)):
#                flag = 1
#                if key not in flag_var:
#                    flag_var.append(key)
#        dom = expr.domain
#        return line_expr, dom, flag, flag_var
