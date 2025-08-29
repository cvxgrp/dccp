"""Constraint convexification for DCCP problems."""

from dataclasses import dataclass

import cvxpy as cp

from dccp.linearize import linearize


@dataclass
class ConvexConstraint:
    """A class to represent a convex constraint with domain constraints.

    Attributes
    ----------
    constr : cp.Constraint
        The convexified constraint.
    domain : list[cp.Constraint]
        Additional domain constraints from linearization.

    """

    constr: cp.Constraint
    domain: list[cp.Constraint]


def convexify_constr(constr: cp.Constraint) -> ConvexConstraint | None:
    """Convexify a constraint for DCCP problems.

    For DCP constraints, returns the constraint unchanged.
    For non-DCP constraints, linearizes the appropriate sides and returns
    a convexified constraint along with any domain constraints.

    Parameters
    ----------
    constr : cp.Constraint
        The constraint to convexify.

    Returns
    -------
    ConvexConstraint or None
        A ConvexConstraint object containing the convexified constraint and
        domain constraints, or None if linearization fails.

    Notes
    -----
    The function handles constraints by:
    - If the constraint is already DCP, return it unchanged
    - If the left side is concave, linearize it
    - If the right side is convex, linearize it
    - Collect domain constraints from linearization

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
