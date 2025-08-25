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
    """Settings for the DCCP algorithm.

    This dataclass contains all configurable parameters for the DCCP algorithm,
    controlling convergence criteria, algorithm behavior, and initialization.

    Attributes
    ----------
    max_iter : int, default=100
        Maximum number of iterations in the CCP algorithm.
    max_iter_damp : int, default=10
        Maximum number of damping iterations when convergence fails.
    tau_ini : float, default=0.005
        Initial value for tau parameter (trades off constraints vs objective).
    mu : float, default=1.2
        Rate at which tau increases during the algorithm.
    tau_max : float, default=1e8
        Upper bound for tau parameter.
    k_ini : int, default=1
        Number of random projections for variable initialization.
    k_ccp : int, default=1
        Number of random restarts for the CCP algorithm.
    max_slack : float, default=1e-3
        Maximum slack variable value for convergence.
    ep : float, default=1e-5
        Convergence tolerance for objective value changes.
    std : float, default=10.0
        Standard deviation for random variable initialization.
    seed : int, optional
        Random seed for reproducible results.
    verify_dccp : bool, default=True
        Whether to verify DCCP compliance before solving.

    """

    max_iter: int = 100
    max_iter_damp: int = 10
    tau_ini: float = 0.005
    mu: float = 1.2
    tau_max: float = 1e8
    k_ini: int = 1
    k_ccp: int = 1
    max_slack: float = 1e-3
    ep: float = 1e-5
    std: float = 10.0
    seed: int | None = None
    verify_dccp: bool = True


class NonDCCPError(Exception):
    """Exception raised when a problem is not DCCP compliant.

    This exception is raised when:
    - A problem is already DCP compliant (should use standard solvers)
    - A problem doesn't satisfy DCCP rules (has unknown curvature)
    - Linearization fails during the algorithm

    """

    def __init__(self, message: str = "Problem is not DCCP compliant.") -> None:
        """Initialize NonDCCPError exception.

        Parameters
        ----------
        message : str
            Error message describing the DCCP compliance issue.

        """
        super().__init__(message)


@dataclass
class DCCPIter:
    """Dataclass to hold DCCP iteration results.

    This class is used internally by the DCCP algorithm to track
    the state and results of each iteration.

    Attributes
    ----------
    prob : cp.Problem
        The current convexified subproblem.
    cost : float, default=np.inf
        The objective value of the current iteration.
    slack : float, default=np.inf
        The maximum slack variable value.

    """

    prob: cp.Problem
    cost: float = np.inf
    slack: float = np.inf

    def __post_init__(self) -> None:
        """Post-initialization hook for dataclass."""


def is_obj_dccp(objective: cp.Minimize | cp.Maximize) -> bool:
    """Check if an objective function satisfies DCCP rules.

    Parameters
    ----------
    objective : cp.Minimize or cp.Maximize
        The objective function to check.

    Returns
    -------
    bool
        True if the objective has known curvature, False otherwise.

    """
    return objective.expr.curvature != "UNKNOWN"


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
