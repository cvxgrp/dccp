"""DCCP package."""

from dataclasses import dataclass

import cvxpy as cp

from dccp.linearize import linearize


@dataclass
class ConvexConstraint:
    """A class to represent a convex constraint."""

    constr: cp.Constraint
    domain: list[cp.Constraint]


def convexify_constr(constr: cp.Constraint) -> ConvexConstraint | None:
    """Convexify a constraint.

    for a dcp constraint, return itself;
    for a non-dcp constraint, return a convexified constraint and domain constraints;
    return None if non-sub/super-diff
    """
    if constr.is_dcp():
        return ConvexConstraint(constr, [])

    dom = []

    # left hand concave
    if constr.args[0].curvature == "CONCAVE":
        left = linearize(constr.args[0])
        if left is None:
            return None
        dom.extend(list(constr.args[0].domain))
    else:
        left = constr.args[0]

    # right hand convex
    if constr.args[1].curvature == "CONVEX":
        right = linearize(constr.args[1])
        if right is None:
            return None
        dom.extend(list(constr.args[1].domain))
    else:
        right = constr.args[1]
    return ConvexConstraint(left - right <= 0, dom)
