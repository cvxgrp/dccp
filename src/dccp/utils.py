"""DCCP utilities module."""

import logging
from dataclasses import dataclass

import cvxpy as cp
import numpy as np

ORDER = "F"


LOGGER = logging.getLogger("dccp")
LOGGER.setLevel(logging.DEBUG)


@dataclass(kw_only=True)
class DCCPSettings:
    """Settings for the DCCP algorithm."""

    max_iter: int = 100
    tau_ini: float = 0.005
    mu: float = 1.2
    tau_max: float = 1e8
    solver: str | None = None
    k_ini: int = 1
    k_ccp: int = 1
    max_slack: float = 1e-3
    ep: float = 1e-5
    std: float = 10.0


class NonDCCPError(Exception):
    """Exception raised when a problem is not DCCP compliant."""

    def __init__(self, message: str = "Problem is not DCCP compliant.") -> None:
        """Initialize NonDCCPError exception."""
        super().__init__(message)


@dataclass
class DCCPIter:
    """Dataclass to hold DCCP iteration results."""

    prob: cp.Problem
    cost: float = np.inf
    slack: float = np.inf

    def __post_init__(self) -> None:
        """Post-initialization."""

    @property
    def status(self) -> str:
        """Return the status of the DCCP sub-problem."""
        return self.prob.status


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
