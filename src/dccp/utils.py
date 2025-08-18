"""DCCP utilities module."""

import cvxpy as cp

ORDER = "F"


class NonDCCPError(Exception):
    """Exception raised when a problem is not DCCP compliant."""

    def __init__(self, message: str = "Problem is not DCCP compliant.") -> None:
        """Initialize NonDCCPError exception."""
        super().__init__(message)


def is_dccp(problem: cp.Problem) -> bool:
    """Check if a problem is DCCP compliant."""
    if problem.objective.expr.curvature == "UNKNOWN":
        return False
    for constr in problem.constraints:
        for arg in constr.args:
            if arg.curvature == "UNKNOWN":
                return False
    return True
