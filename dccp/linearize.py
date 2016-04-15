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
    for var in expr.variables(): # for each variable in the expression
        value_para = Parameter(var.size[0],var.size[1]) # parameterize its value
        gr = [] # a list for grad w.r.t. this variable, the length of gr equals the number of columns of var
        for d in range(var.size[1]): # for each slice
            g = Parameter(var.size[0],expr.size[0])
            linear_expr += g.T * (var[:,d] - value_para[:,d]) # first order
            gr.append(g)
        linear_dictionary[var] = [value_para, gr]
    dom = expr.domain
    return linear_expr, zero_order, linear_dictionary, dom


# the following function is not used anymore in the parameterized version
def linearize(expr):
        """linearize an expression at a given point
        """
        flag = 0
        flag_var = []
        line_expr = expr.value
        for key in expr.gradient:
            D = expr.gradient[key]
            rows, cols, imag_rows, imag_cols = D.shape
            if cols>1: # matrix to vector
                for d in range(cols):
                    g = D[:,d,:,0]
                    g = g.T
                    line_expr = line_expr + g * (key[:,d] - key.value[:,d])
            else: # vector to vector
                g = D[:,0,:,0]
                g = g.T
                line_expr = line_expr + g * (key - key.value)
            if np.any(np.isnan(g)) or np.any(np.isinf(g)):
                flag = 1
                if key not in flag_var:
                    flag_var.append(key)
        dom = expr.domain
        return line_expr, dom, flag, flag_var
