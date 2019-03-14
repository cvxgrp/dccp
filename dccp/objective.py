__author__ = 'Xinyue'
from dccp.linearize import linearize, linearize_para
import cvxpy as cvx

#from linearize import linearize_para
def convexify_para_obj(obj):
    """
    input:
        obj: an objective of a problem
    return:
        if the objective is dcp,
        return the cost function (an expression);
        if the objective has a wrong curvature,
        return the linearized expression of the cost function,
        the zeros order parameter,
        the dictionary of parameters indexed by variables,
        the domain
    """
    if obj.is_dcp() == False:
        return linearize_para(obj.expr)
    else:
        return obj.expr

def is_dccp(objective):
    """
    input:
        objective: an objective of a problem
    return:
        if the objective is dccp
        the objective must be convex, concave, affine, or constant
    """
    if objective.expr.curvature == 'UNKNOWN':
        return False
    else:
        return True

def convexify_obj(obj):
    """
    :param obj: objective of a problem
    :return: convexified onjective or None
    """
    # not dcp
    if obj.is_dcp() == False:
        lin = linearize(obj.expr)
        # non-sub/super-diff
        if lin is None:
            return None
        else:
            if obj.NAME == 'minimize':
                result = cvx.Minimize(lin)
            else:
                result = cvx.Maximize(lin)
    else:
        result = obj
    return result