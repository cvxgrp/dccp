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
    def slack(self) -> float:
        """Get the maximum slack variable value."""
        if not self.vars_slack:
            return 0.0
        slack_values = [
            np.max(s.value) if s.value is not None else 0.0 for s in self.vars_slack
        ]
        return max(slack_values, default=0.0)

    @property
    def slack_sum(self) -> float:
        """Sum of all slack elements."""
        return sum(
            float(np.sum(s.value)) if s.value is not None else 0.0
            for s in self.vars_slack
        )

    @property
    def cost_ns(self) -> float:
        """Objective value minus τ * sum(slack)."""
        if self.prob.objective.value is None:
            return np.inf
        obj_val = float(np.asarray(self.prob.objective.value).item())
        tau_val = float(np.asarray(self.tau.value).item())
        return obj_val - tau_val * self.slack_sum

    def solve(self, **kwargs: Any) -> float | None:
        """Solve the DCCP sub-problem."""
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
        if settings.verify_dccp and not is_dccp(prob):
            msg = "Problem is not DCCP."
            raise NonDCCPError(msg)

        # store the problem and settings
        self.is_maximization = isinstance(prob.objective, cp.Maximize)
        self.prob_in = prob
        self.solve_args = kwargs.copy()
        self.conf = settings

        # slack loss weight tau
        self.tau = cp.Parameter(nonneg=True, value=settings.tau_ini, name="tau")

        # construction of DCCP sub-problem
        init_kwargs = {}
        if self.conf.k_ccp is not None and self.conf.k_ccp > 1:
            init_kwargs["random"] = True
        if self.conf.seed is not None:
            init_kwargs["seed"] = self.conf.seed
        init_kwargs["solver"] = kwargs.get("solver")

        initialize(prob, **init_kwargs)
        self.iter = DCCPIter(
            prob=prob,  # Use the original problem initially
            tau=self.tau,
        )

        self._prev_var_values = {}
        self._store_previous_values()

    def _apply_damping(self) -> None:
        """Apply damping to variable values using previous iteration values."""
        for var in self.prob_in.variables():
            if var.value is not None and var in self._prev_var_values:
                prev_val = self._prev_var_values[var]
                cur_val = var.value.copy()
                damped_step = 0.8 * cur_val + 0.2 * prev_val
                var.value = damped_step
                logger.debug(
                    "Suggested value for %s: %s, previous value: %s, damped value: %s",
                    var.name(),
                    cur_val,
                    prev_val,
                    damped_step,
                )

    def _store_previous_values(self) -> None:
        """Store current variable values for damping."""
        for var in self.prob_in.variables():
            if var.value is not None:
                val = var.value
                self._prev_var_values[var] = val.copy() if hasattr(val, "copy") else val

    def _construct_subproblem(self) -> None:
        """Construct the DCCP sub-problem."""
        prob = self.prob_in

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
            k_damp = 0
            while obj is None and k_damp < self.conf.max_iter_damp:
                self._apply_damping()
                obj = convexify_obj(prob.objective)
                k_damp += 1
            if obj is None:
                msg = (
                    "Damping did not yield a convexified objective after "
                    f"{self.conf.max_iter_damp} iterations."
                )
                raise NonDCCPError(msg)

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
                self._apply_damping()
                c_conv = convexify_constr(c)

            new_constr.extend(list(c_conv.domain))
            new_constr.append(c_conv.constr.expr <= v_slack)

        # build new problem
        if var_slack:
            slack_sum = cp.sum([cp.sum(s) for s in var_slack])
            cost = obj.expr + self.tau * slack_sum  # type: ignore[reportOptionalMemberAccess]
        else:
            cost = obj.expr  # type: ignore[reportOptionalMemberAccess]
        new_prob = cp.Problem(cp.Minimize(cost), new_constr)

        # Update the existing iter object instead of creating a new one
        self.iter.prob = new_prob
        self.iter.vars_slack = var_slack

        # store previous variable values for damping
        self._store_previous_values()

    def _solve(self) -> float:
        """Solve the DCCP problem."""
        converged = False
        prev_cost = np.inf
        prev_cost_ns = np.inf

        # run DCCP iterations until convergence or max iterations
        while not (converged or self.iter.k > self.conf.max_iter):
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
            if self.iter.tau.value is not None:
                tau_mul = float(self.iter.tau.value) * self.conf.mu
                self.iter.tau.value = min(self.conf.tau_max, tau_mul)

            logger.debug(
                "Iteration %d: cost=%s, cost_ns=%s, slack=%s, tau=%s",
                self.iter.k,
                self.iter.cost,
                self.iter.cost_ns,
                self.iter.slack,
                self.iter.tau.value,
            )

        # terminate with infeasibility if not converged after max iterations
        self.prob_in._status = cp.INFEASIBLE  # noqa: SLF001

        # write the solution back to the original problem
        if converged:
            self.prob_in._status = cp.OPTIMAL  # noqa: SLF001
            for var in self.prob_in.variables():
                var.value = self.iter.prob.var_dict[var.name()].value
            return self.iter.cost

        # return the objective value
        return self.iter.cost if converged else np.inf

    def __call__(self) -> float:
        """Solve a problem using the Disciplined Convex-Concave Procedure."""
        return self._solve() * (-1 if self.is_maximization else 1)

    def solve_multi_init(self, num_inits: int) -> float:
        """Solve with multiple random initializations and return the best result."""
        if num_inits <= 1:
            return self()

        best_cost = np.inf
        best_var_values = {}
        best_status = cp.INFEASIBLE

        for _ in range(num_inits):
            # store original variable values
            orig_values = {}
            for var in self.prob_in.variables():
                orig_values[var] = var.value.copy() if var.value is not None else None

            # reset and solve with new random initialization
            initialize(self.prob_in, random=True)
            self.iter.k = 0
            self.iter.cost = np.inf
            self._prev_var_values = {}
            self._store_previous_values()

            try:
                cost = self._solve()
                if self.prob_in.status == cp.OPTIMAL and cost < best_cost:
                    best_cost = cost
                    best_status = self.prob_in.status
                    best_var_values = {
                        var: var.value.copy() if var.value is not None else None
                        for var in self.prob_in.variables()
                    }
            except (NonDCCPError, RuntimeError):
                continue

            # restore original values for next iteration
            for var in self.prob_in.variables():
                var.value = orig_values[var]

        # set the best solution
        self.prob_in._status = best_status  # noqa: SLF001
        for var in self.prob_in.variables():
            if var in best_var_values:
                var.value = best_var_values[var]

        return best_cost * (-1 if self.is_maximization else 1)


def dccp(  # noqa: PLR0913
    prob: cp.Problem,
    *,
    max_iter: int = 100,
    tau: float = 0.005,
    mu: float = 1.2,
    tau_max: float = 1e8,
    k_ccp: int = 1,
    max_slack: float = 1e-3,
    ep: float = 1e-5,
    seed: int | None = None,
    verify_dccp: bool = True,
    **kwargs: Any,
) -> float:
    """Run the DCCP algorithm on the given problem."""
    logger.debug("Running DCCP with solver=%s, kwargs=%s", kwargs.get("solver"), kwargs)
    dccp_solver = DCCP(
        prob,
        settings=DCCPSettings(
            max_iter=max_iter,
            tau_ini=tau,
            mu=mu,
            tau_max=tau_max,
            k_ccp=k_ccp,
            max_slack=max_slack,
            ep=ep,
            seed=seed,
            verify_dccp=verify_dccp,
        ),
        **kwargs,
    )
    if k_ccp > 1:
        return dccp_solver.solve_multi_init(k_ccp)
    return dccp_solver()
