__author__ = 'Xinyue'

from linearize import linearize
from linearize import linearize_para

def convexify_para_constr(self):
        dom = []
        para = []
        # left hand concave
        if self.args[0].curvature == 'CONCAVE':
            lin = linearize_para(self.args[0]) # expression, vars, grads, domain
            left = lin[0]
            para.append([lin[1],lin[2]])
            for con in lin[3]:
                dom.append(con)
        else:
            left = self.args[0]
            para.append([])
        # right hand convex
        if self.args[1].curvature == 'CONVEX':
            lin = linearize_para(self.args[1])
            right = lin[0]
            para.append([lin[1],lin[2]])
            for con in lin[3]:
                dom.append(con)
        else:
            right = self.args[1]
            para.append([])
        return left<=right, para, dom

def convexify_constr(self):
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