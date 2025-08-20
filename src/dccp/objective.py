"""Convexify an objective function for DCCP problems."""

import logging

import cvxpy as cp

from dccp.linearize import linearize

logger = logging.getLogger("dccp")


def is_dccp(objective: cp.Minimize | cp.Maximize) -> bool:
    """Check if the objective is DCCP compliant."""
    return objective.expr.curvature != "UNKNOWN"


def convexify_obj(obj: cp.Minimize | cp.Maximize) -> cp.Minimize | None:
    """Convexify an objective function for DCCP problems.

    Linearize non-DCP objectives. If the objective is already DCP, returns it unchanged.
    If the objective is a maximization problem, it is negated to convert it into a
    minimization problem.

    Parameters
    ----------
    obj : cp.Minimize | cp.Maximize
        Objective of a problem to be convexified.

    Returns
    -------
    cp.Minimize or None
        Convexified objective if linearization is possible, None otherwise.

    """
    # negate the objective if it is a maximization problem
    expr = obj.expr
    if isinstance(obj, cp.Maximize):
        expr = -expr
    if obj.is_dcp():
        return cp.Minimize(expr)

    if not is_dccp(obj):
        msg = (
            "The objective is not DCCP compliant. Please ensure the objective is "
            "constructed from DCP atoms so its curvature can be inferred."
        )
        raise ValueError(msg)
    lin = linearize(expr)
    if lin is not None:
        return cp.Minimize(lin)
    return None
