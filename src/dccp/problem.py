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
        if not self.vars_slack:
            return 0.0
        slack_values = []
        for s in self.vars_slack:
            val = s.value
            if val is not None:
                slack_values.append(np.max(val))
        return max(slack_values, default=0.0)

    @property
    def cost_ns(self) -> float:
        """Get the cost without slack."""
        if self.prob.objective.value is not None:
            return self.prob.objective.value - self.tau.value * self.slack  # type: ignore[reportOptionalOperand]
        return np.inf

    def solve(self, **kwargs: Any) -> float | None:
        """Solve the DCCP sub-problem."""
        logger.debug("Solving iteration %d with kwargs=%s", self.k, kwargs)
        self.k += 1
        result = self.prob.solve(**kwargs)
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
        init_kwargs = {}
        if self.conf.k_ccp is not None and self.conf.k_ccp > 1:
            init_kwargs["random"] = True
        if self.conf.solver is not None:
            init_kwargs["solver"] = self.conf.solver
        if self.conf.seed is not None:
            init_kwargs["seed"] = self.conf.seed

        initialize(prob, **init_kwargs)
        self.iter = DCCPIter(
            prob=prob,  # Use the original problem initially
            tau=self.tau,
        )

    def _apply_damping(self) -> None:
        """Apply damping to variable values using previous iteration values."""
        logger.debug(
            "APPLYING DAMPING: iteration %d, tau=%s", self.iter.k, self.tau.value
        )
        for var in self.prob_in.variables():
            if var.value is not None and var in self._prev_var_values:
                prev_val = self._prev_var_values[var]
                var.value = 0.8 * var.value + 0.2 * prev_val

    def _store_previous_values(self) -> None:
        """Store current variable values for damping."""
        if not hasattr(self, "_prev_var_values"):
            self._prev_var_values = {}

        for var in self.prob_in.variables():
            if var.value is not None:
                val = var.value
                self._prev_var_values[var] = val.copy() if hasattr(val, "copy") else val

    def _construct_subproblem(self) -> None:
        """Construct the DCCP sub-problem."""
        prob = self.prob_in

        # Store previous variable values for damping
        self._store_previous_values()

        # split non-affine equality constraints
        constr: list[cp.Constraint] = []
        for constraint in prob.constraints:
            if isinstance(constraint, Equality) and not constraint.is_dcp():
                constr.append(constraint.args[0] <= constraint.args[1])
                constr.append(constraint.args[0] >= constraint.args[1])
            else:
                constr.append(constraint)

        # each non-dcp constraint needs a slack variable
        var_slack: list[cp.Variable] = []

        # convexify objective with damping if needed
        obj = convexify_obj(prob.objective)
        if not prob.objective.is_dcp():
            while obj is None:
                self._apply_damping()
                obj = convexify_obj(prob.objective)

        # add objective domain constraints into problem constraints
        constr.extend(list(prob.objective.expr.domain))

        # build new linearized, convexified constraints
        new_constr: list[cp.Constraint] = []
        for c in constr:
            if c.is_dcp():
                new_constr.append(c)
                continue

            # add slack variable for non-convex constraint
            v_slack = cp.Variable(c.shape, name=f"slack_{c.id}", nonneg=True)
            var_slack.append(v_slack)

            # convexify the constraint with damping if needed
            c_conv = convexify_constr(c)
            while c_conv is None:
                logger.debug("Applying damping: iteration %d", self.iter.k)
                self._apply_damping()
                c_conv = convexify_constr(c)

            new_constr.extend(list(c_conv.domain))
            new_constr.append(c_conv.constr.expr <= v_slack)

        # build new problem
        cost = obj.expr + self.tau * cp.sum(var_slack) if var_slack else obj.expr
        new_prob = cp.Problem(cp.Minimize(cost), new_constr)

        # Update the existing iter object instead of creating a new one
        self.iter.prob = new_prob
        self.iter.vars_slack = var_slack

    def _solve(self) -> float:
        """Solve the DCCP problem."""
        converged = False
        prev_cost = np.inf
        prev_cost_ns = np.inf

        # run DCCP iterations until convergence or max iterations
        while not (converged or self.iter.k > self.conf.max_iter):
            logger.debug("Iteration %d, tau=%s", self.iter.k, self.iter.tau.value)
            self._construct_subproblem()
            new_cost = self.iter.solve(**self.solve_args)
            new_cost_ns = self.iter.cost_ns
            slack = self.iter.slack

            # check all convergence criteria
            if (
                new_cost is not None
                and np.abs(prev_cost - new_cost) <= self.conf.ep
                and np.abs(prev_cost_ns - new_cost_ns) <= self.conf.ep
                and slack <= self.conf.max_slack
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

        # Return the original objective value for the original problem
        if converged:
            obj_value = self.prob_in.objective.value
            if obj_value is not None and isinstance(
                obj_value, (int, float, np.floating)
            ):
                return float(obj_value)
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
    logger.debug("Running DCCP with solver=%s, kwargs=%s", solver, kwargs)
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
