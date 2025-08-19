"""DCCP utilities module."""

import logging

import cvxpy as cp

ORDER = "F"


LOGGER = logging.getLogger("dccp")
LOGGER.setLevel(logging.DEBUG)


class NonDCCPError(Exception):
    """Exception raised when a problem is not DCCP compliant."""

    def __init__(self, message: str = "Problem is not DCCP compliant.") -> None:
        """Initialize NonDCCPError exception."""
        super().__init__(message)


def is_obj_dccp(objective: cp.Minimize | cp.Maximize) -> bool:
    """Check if the objective is DCCP."""
    return objective.expr.curvature != "UNKNOWN"


def is_sparse(x: cp.Variable | cp.Parameter) -> bool:
    """Check if a CVXPY variable or parameter is sparse."""
    return x.sparse_idx is not None


def is_dccp(problem: cp.Problem) -> bool:
    """Check if a CVXPY problem is DCCP compliant.

    This function verifies whether a convex optimization problem satisfies
    the Disciplined Convex-Concave Programming (DCCP) rules, which allow
    for the solution of certain non-convex problems.

    Parameters
    ----------
    problem : cp.Problem
        A CVXPY Problem object to check for DCCP compliance.

    Returns
    -------
    bool
        True if the problem is DCCP compliant, False otherwise.

    Notes
    -----
    A problem is DCCP compliant if:
    1. The objective function is DCCP compliant
    2. All constraint arguments have known curvature (not "UNKNOWN")

    """
    if not is_obj_dccp(problem.objective):
        return False

    # check DCCP compliance for each argument of each constraint individually
    return not any(
        any(arg.curvature == "UNKNOWN" for arg in constr.args)
        for constr in problem.constraints
    )
