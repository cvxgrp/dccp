__author__ = 'Xinyue'

from linearize import linearize
from linearize import linearize_para

def convexify_para_constr(self):
    '''
    input:
        self: a constraint of a problem
    return:
        if the constraint is dcp, return itself;
        otherwise, return
            a convexified constraint
            para: [left side, right side]
                if the left/right-hand side of the the constraint is linearized,
                left/right side = [zero order parameter, {variable: [value parameter, [gradient parameter]]}]
                else,
                left/right side = []
            dom: domain
    '''
    if not self.is_dcp():
        dom = [] # domain
        para = [] # a list for parameters
        if self.args[0].curvature == 'CONCAVE': # left-hand concave
            lin = linearize_para(self.args[0]) # linearize the expression
            left = lin[0]
            para.append([lin[1],lin[2]]) # [zero order parameter, {variable: [value parameter, [gradient parameter]]}]
            for con in lin[3]:
                dom.append(con)
        else:
            left = self.args[0]
            para.append([]) # appending an empty list indicates the expression has the right curvature
        if self.args[1].curvature == 'CONVEX': # right-hand convex
            lin = linearize_para(self.args[1]) # linearize the expression
            right = lin[0]
            para.append([lin[1],lin[2]])
            for con in lin[3]:
                dom.append(con)
        else:
            right = self.args[1]
            para.append([])
        return left<=right, para, dom
    else:
        return self

# the following function is not used anymore in the parameterized version
def convexify_constr(self):
    if not self.is_dcp():
        dom = []
        flag = 0
        flag_var = []
        # left hand concave
        if self.args[0].curvature == 'CONCAVE':
            lin = linearize(self.args[0]) # expression, domain, flag
            left = lin[0]
            for con in lin[1]:
                dom.append(con)
            flag = lin[2]
            flag_var.append(lin[3])
        else:
            left = self.args[0]
        #right hand convex
        if self.args[1].curvature == 'CONVEX':
            lin = linearize(self.args[1])
            right = lin[0]
            for con in lin[1]:
                dom.append(con)
            flag = lin[2]
            flag_var.append(lin[3])
        else:
            right = self.args[1]
        return left<=right, dom, flag, flag_var
    else:
        return self
