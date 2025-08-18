"""Objective.py"""

import cvxpy as cp

from dccp.linearize import linearize


def is_dccp(objective):
    """
    input:
        objective: an objective of a problem
    return:
        if the objective is dccp
        the objective must be convex, concave, affine, or constant
    """
    if objective.expr.curvature == "UNKNOWN":
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
            if obj.NAME == "minimize":
                result = cp.Minimize(lin)
            else:
                result = cp.Maximize(lin)
    else:
        result = obj
    return result
