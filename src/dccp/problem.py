"""DCCP package."""

import logging
from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp
import numpy as np
from cvxpy.constraints.zero import Equality

from .constraint import convexify_constr
from .initialization import initialize
from .objective import convexify_obj
from .utils import DCCPSettings, NonDCCPError, is_dccp

logger = logging.getLogger("dccp")
logger.setLevel(logging.INFO)


@dataclass
class DCCPIter:
    """Store results of a DCCP iteration."""

    prob: cp.Problem
    tau: cp.Parameter = field(default_factory=lambda: cp.Parameter(value=0.005))
    vars_slack: list[cp.Variable] = field(default_factory=list)
    k: int = 0
    cost: float = np.inf

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

    @property
    def cost_ns(self) -> float:
        """Get the cost without slack."""
        if self.prob.objective.value is not None:
            return self.prob.objective.value - self.tau.value * self.slack  # type: ignore[reportOptionalOperand]
        return np.inf

    def solve(self, *args: Any, **kwargs: Any) -> float | None:
        """Solve the DCCP sub-problem."""
        self.k += 1
        result = self.prob.solve(*args, **kwargs)
        if isinstance(result, (int, float, np.floating)):
            self.cost = float(result)
            return self.cost
        return None


class DCCP:
    """Implementation of the DCCP algorithm."""

    def __init__(
        self,
        prob: cp.Problem,
        *,
        settings: DCCPSettings | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the DCCP solver."""
        if settings is None:
            settings = DCCPSettings()
        if prob.is_dcp():
            msg = "The problem is DCP compliant, solve it with a DCP solver."
            raise NonDCCPError(msg)
        if not is_dccp(prob):
            msg = "Problem is not DCCP."
            raise NonDCCPError(msg)

        # store the problem
        self.prob_in = prob

        # DCCP settings
        self.solve_args = kwargs
        self.conf = settings

        # slack loss weight tau
        self.tau = cp.Parameter(nonneg=True, value=settings.tau_ini, name="tau")

        # construction of DCCP sub-problem
        initialize(
            prob,
            random=(self.conf.k_ccp > 1),
            solver=self.conf.solver,
            seed=self.conf.seed,
            std=self.conf.std,
            k_ini=self.conf.k_ini,
        )
        self.iter = DCCPIter(prob)
        self._construct_subproblem()

    def _construct_subproblem(self) -> None:
        """Construct the DCCP sub-problem."""
        prob = self.iter.prob

        # split non-affine equality constraints
        constr: list[cp.Constraint] = []
        for constraint in prob.constraints:
            if isinstance(constraint, Equality) and not constraint.is_dcp():
                constr.append(constraint.args[0] <= constraint.args[1])
                constr.append(constraint.args[0] >= constraint.args[1])

        # each non-dcp constraint needs a slack variable
        var_slack: list[cp.Variable] = []

        # add objective domain constraints into problem constraints
        obj = convexify_obj(prob.objective)
        constr.extend(list(prob.objective.expr.domain))

        # build new linearized, convexified constraints
        new_constr: list[cp.Constraint] = []
        for c in constr:
            if c.is_dcp():
                new_constr.append(c)
                continue

            # add slack variable for non-convex constraint
            var_slack.append(cp.Variable(c.shape, name=f"slack_{c.id}", nonneg=True))

            # convexify the constraint
            c_conv = convexify_constr(c)
            new_constr.extend(list(c_conv.domain))
            new_constr.append(c_conv.constr.expr <= var_slack)

        # build new problem
        cost = cp.sum(obj.expr) + self.tau * cp.sum(var_slack)
        self.iter = DCCPIter(
            cp.Problem(cp.Minimize(cost), new_constr),
            vars_slack=var_slack,
            tau=self.tau,
        )

    def _solve(self) -> float:
        """Solve the DCCP problem."""
        converged = False
        prev_cost = np.inf
        prev_cost_ns = np.inf
        while not (converged or self.iter.k > self.conf.max_iter):
            print(f"Iteration {self.iter.k}, tau={self.iter.tau.value}")
            self._construct_subproblem()
            print("Solving DCCP sub-problem...")
            new_cost = self.iter.solve(self.solve_args)
            new_cost_ns = self.iter.cost_ns
            slack = self.iter.slack

            # check all convergence criteria
            if (
                new_cost is not None
                and np.abs(prev_cost - new_cost) <= self.conf.ep
                and np.abs(prev_cost_ns - new_cost_ns) <= self.conf.ep
                and (slack is None or slack <= self.conf.max_slack)
            ):
                converged = True

            # update previous values
            prev_cost = new_cost if new_cost is not None else prev_cost
            prev_cost_ns = new_cost_ns if new_cost_ns is not None else prev_cost_ns

            # update tau for the next iteration
            self.iter.tau.value = min(
                self.conf.tau_max,
                self.iter.tau.value * self.conf.mu,  # type: ignore[reportOptionalOperand]
            )

            logger.debug(
                "Iteration %d: cost=%s, cost_ns=%s, slack=%s, tau=%s",
                self.iter.k,
                self.iter.cost,
                self.iter.cost_ns,
                self.iter.slack,
                self.iter.tau.value,
            )

        self.prob_in._status = cp.INFEASIBLE  # noqa: SLF001

        # write the solution back to the original problem
        if converged:
            self.prob_in._status = cp.OPTIMAL  # noqa: SLF001
            for var in self.prob_in.variables():
                var.value = self.iter.prob.var_dict[var.name()].value

        return self.iter.cost if converged else np.inf

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
    k_ccp: int = 1,
    max_slack: float = 1e-3,
    ep: float = 1e-5,
    seed: int | None = None,
    **kwargs: Any,
) -> float:
    """Run the DCCP algorithm on the given problem."""
    dccp_solver = DCCP(
        prob,
        settings=DCCPSettings(
            max_iter=max_iter,
            tau_ini=tau,
            mu=mu,
            tau_max=tau_max,
            solver=solver,
            k_ccp=k_ccp,
            max_slack=max_slack,
            ep=ep,
            seed=seed,
        ),
        **kwargs,
    )
    return dccp_solver()
