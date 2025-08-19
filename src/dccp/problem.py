"""DCCP package."""

import logging
from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp
import numpy as np

from .initialization import initialize
from .utils import NonDCCPError, is_dccp

logger = logging.getLogger("dccp")
logger.setLevel(logging.INFO)


@dataclass
class DCCPIter:
    """Store results of a DCCP iteration."""

    prob: cp.Problem
    vars_slack: list[cp.Variable] = field(default_factory=list)
    k: int = 0

    @property
    def status(self) -> str | None:
        """Get the status of the DCCP iteration."""
        return self.prob.status

    @property
    def slack(self) -> float:
        """Get the maximum slack variable value."""
        slack_values = []
        for s in self.vars_slack:
            val = self.prob.var_dict[s.name()].value
            if val is not None:
                slack_values.append(np.max(val))
        return max(slack_values, default=-np.inf)


class DCCP:
    """Implementation of the DCCP algorithm."""

    def __init__(  # noqa: PLR0913
        self,
        prob: cp.Problem,
        *,
        max_iter: int,
        tau: float,
        mu: float,
        tau_max: float,
        solver: str | None,
        ccp_times: int,
        max_slack: float,
        ep: float,
        seed: int | None,
        **kwargs: Any,
    ) -> None:
        """Initialize the DCCP solver."""
        if prob.is_dcp():
            msg = "The problem is DCP compliant, solve it with a DCP solver."
            raise NonDCCPError(msg)
        if not is_dccp(prob):
            msg = "Problem is not DCCP."
            raise NonDCCPError(msg)

        # construction of DCCP sub-problem
        iter = self._construct_subproblem(prob)

    def _construct_subproblem(self, prob: cp.Problem) -> DCCPIter:
        """Construct the DCCP sub-problem."""
        # initialization of variables
        initialize(prob, random=(ccp_times > 1), solver=solver, seed=seed, **kwargs)

    def _solve(self) -> float:
        """Solve the DCCP problem."""
        return np.inf

    def __call__(self) -> float:
        """Solve a problem using the Disciplined Convex-Concave Procedure."""
        return self._solve()


def dccp(  # noqa: PLR0913
    prob: cp.Problem,
    *,
    max_iter: int = 100,
    tau: float = 0.005,
    mu: float = 1.2,
    tau_max: float = 1e8,
    solver: str | None = None,
    ccp_times: int = 1,
    max_slack: float = 1e-3,
    ep: float = 1e-5,
    seed: int | None = None,
    **kwargs: Any,
) -> float:
    """Run the DCCP algorithm on the given problem."""
    dccp_solver = DCCP(
        prob,
        max_iter=max_iter,
        tau=tau,
        mu=mu,
        tau_max=tau_max,
        solver=solver,
        ccp_times=ccp_times,
        max_slack=max_slack,
        ep=ep,
        seed=seed,
        **kwargs,
    )
    return dccp_solver()
