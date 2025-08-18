"""DCCP package."""

from dccp.linearize import linearize


def convexify_constr(constr):
    """
    :param constr: a constraint of a problem
    :return:
    for a dcp constraint, return itself;
    for a non-dcp constraint, return a convexified constraint and domain constraints;
    return None if non-sub/super-diff
    """
    if not constr.is_dcp():
        dom = []
        # left hand concave
        if constr.args[0].curvature == "CONCAVE":
            left = linearize(constr.args[0])
            if left is None:
                return None
            else:
                for con in constr.args[0].domain:
                    dom.append(con)
        else:
            left = constr.args[0]
        # right hand convex
        if constr.args[1].curvature == "CONVEX":
            right = linearize(constr.args[1])
            if right is None:
                return None
            else:
                for con in constr.args[1].domain:
                    dom.append(con)
        else:
            right = constr.args[1]
        return left - right <= 0, dom
    else:
        return constr
