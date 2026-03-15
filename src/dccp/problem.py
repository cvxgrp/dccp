"""DCCP solver implementation for Disciplined Convex-Concave Programming."""

from __future__ import annotations

import logging
from concurrent.futures import (  # pylint: disable=no-name-in-module
    ProcessPoolExecutor,
    as_completed,
)
from dataclasses import dataclass, field
from numbers import Number
from typing import TYPE_CHECKING, Any, TypeAlias

import cvxpy as cp
import numpy as np
from cvxpy.constraints.zero import Equality

from .constraint import convexify_constr
from .initialization import initialize
from .objective import convexify_obj
from .utils import DCCPSettings, NonDCCPError, is_dccp

if TYPE_CHECKING:
    from multiprocessing.context import BaseContext

logger = logging.getLogger("dccp")
logger.setLevel(logging.INFO)

ProblemValue: TypeAlias = (
    Number | np.generic | complex | str | bytes | memoryview | None
)


def _set_problem_status(prob: cp.Problem, status: str) -> None:
    """Set problem status via internal _status attribute.

    Workaround since cvxpy's status property is read-only.
    Directly sets the internal _status attribute which the status property reads.
    """
    prob._status = status  # noqa: SLF001  # type: ignore[reportPrivateUsage]


def _set_problem_value(prob: cp.Problem, value: ProblemValue) -> None:
    """Set problem value via internal _value attribute.

    Workaround since cvxpy's value property is read-only.
    Directly sets the internal _value attribute which the value property reads.
    """
    prob._value = value  # noqa: SLF001  # type: ignore[reportPrivateUsage]


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
    def cost_no_slack(self) -> float:
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
    """Implementation of the DCCP (Disciplined Convex-Concave Programming) algorithm.

    The DCCP class provides a systematic approach to solving nonconvex optimization
    problems that satisfy the DCCP rules. It uses an iterative convex-concave
    procedure (CCP) to find approximate solutions.

    Parameters
    ----------
    prob : cp.Problem
        A CVXPY Problem that satisfies DCCP rules but not DCP rules.
    settings : DCCPSettings, optional
        Configuration settings for the DCCP algorithm. If None, default
        settings are used.
    **kwargs
        Additional keyword arguments passed to the underlying solver.

    Attributes
    ----------
    is_maximization : bool
        True if the original problem is a maximization problem.
    prob_in : cp.Problem
        The original input problem.
    solve_args : dict
        Solver arguments passed to subproblems.
    conf : DCCPSettings
        Algorithm configuration settings.
    tau : cp.Parameter
        Penalty parameter for constraint violations.
    iter : DCCPIter
        Current iteration state and results.

    Raises
    ------
    NonDCCPError
        If the problem is DCP compliant (should use standard solvers) or
        if the problem doesn't satisfy DCCP rules.

    """

    def __init__(
        self,
        prob: cp.Problem,
        *,
        settings: DCCPSettings | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the DCCP solver.

        Parameters
        ----------
        prob : cp.Problem
            A CVXPY Problem that satisfies DCCP rules.
        settings : DCCPSettings, optional
            Configuration settings for the algorithm.
        **kwargs
            Additional solver arguments.

        """
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
        if self.conf.k_ccp > 1:
            init_kwargs["random"] = True
        if self.conf.seed is not None:
            init_kwargs["seed"] = self.conf.seed
        init_kwargs["solver"] = kwargs.get("solver")

        initialize(prob, **init_kwargs)
        self.iter = DCCPIter(prob=prob, tau=self.tau)
        self.linearization_map = {}  # type: ignore[var-annotated]

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

    def _update_linearizations(self) -> None:
        """Update parameters of the subproblem."""
        params_updated = False
        k = 0
        while not params_updated and k < self.conf.max_iter_damp:
            try:
                for data in self.linearization_map.values():
                    # Ensure expression has a value
                    if data.expr.value is None:
                        try:
                            val = data.expr.value
                            if val is None:
                                # Trigger evaluation
                                val = data.expr.save_value(data.expr.value)
                        except Exception:  # noqa: BLE001, S110
                            pass

                        if data.expr.value is None:
                            msg = f"Expression {data.expr} value is None"
                            raise ValueError(msg)  # noqa: TRY301

                    data.update()

                params_updated = True
            except (ValueError, Exception) as e:  # noqa: BLE001
                logger.debug("Parameter update failed: %s. Applying damping.", e)
                self._apply_damping()
                k += 1

        if not params_updated:
            msg = (
                "Damping did not yield valid parameters after "
                f"{self.conf.max_iter_damp} iterations."
            )
            raise NonDCCPError(msg)

    def _construct_subproblem(self) -> None:
        """Construct the DCCP sub-problem."""
        if self.linearization_map:
            self._update_linearizations()
            self._store_previous_values()
            return

        # First time construction
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
        obj = convexify_obj(prob.objective, self.linearization_map)
        if not prob.objective.is_dcp():
            k_damp = 0
            while obj is None and k_damp < self.conf.max_iter_damp:
                self._apply_damping()
                obj = convexify_obj(prob.objective, self.linearization_map)
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
            c_conv = convexify_constr(c, self.linearization_map)
            while c_conv is None:
                self._apply_damping()
                c_conv = convexify_constr(c, self.linearization_map)

            new_constr.extend(list(c_conv.domain))
            new_constr.append(c_conv.constr.expr <= v_slack)

        # build new problem
        if var_slack:
            slack_sum = cp.sum([cp.sum(s) for s in var_slack])
            cost = obj.expr + self.tau * slack_sum  # type: ignore  # noqa: PGH003
        else:
            cost = obj.expr  # type: ignore  # noqa: PGH003
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
        prev_cost_no_slack = np.inf

        # run DCCP iterations until convergence or max iterations
        while not (converged or self.iter.k > self.conf.max_iter):
            self._construct_subproblem()
            new_cost = self.iter.solve(**self.solve_args)
            new_cost_no_slack = self.iter.cost_no_slack
            slack = self.iter.slack

            # check all convergence criteria
            if (
                new_cost is not None
                and np.abs(prev_cost - new_cost) <= self.conf.ep
                and np.abs(prev_cost_no_slack - new_cost_no_slack) <= self.conf.ep
                and slack <= self.conf.max_slack
            ):
                converged = True

            # update previous values
            prev_cost = new_cost if new_cost is not None else prev_cost
            prev_cost_no_slack = new_cost_no_slack

            # update tau for the next iteration
            if self.iter.tau.value is not None:
                tau_mul = float(self.iter.tau.value) * self.conf.mu
                self.iter.tau.value = min(self.conf.tau_max, tau_mul)

            logger.debug(
                "Iteration %d: cost=%s, cost_no_slack=%s, slack=%s, tau=%s",
                self.iter.k,
                self.iter.cost,
                self.iter.cost_no_slack,
                self.iter.slack,
                self.iter.tau.value,
            )

        # terminate with infeasibility if not converged after max iterations
        _set_problem_status(self.prob_in, cp.INFEASIBLE)

        # write the solution back to the original problem
        if converged:
            _set_problem_status(self.prob_in, cp.OPTIMAL)
            _set_problem_value(self.prob_in, self.iter.prob.value)
            for var in self.prob_in.variables():
                var.value = self.iter.prob.var_dict[var.name()].value
            return self.iter.cost

        return self.iter.cost if converged else np.inf

    def __call__(self) -> float:
        """Solve a problem using the Disciplined Convex-Concave Procedure."""
        return self._solve() * (-1 if self.is_maximization else 1)

    def _solve_one_init(self) -> tuple[float | None, dict[int, Any] | None]:
        """Solve with current initialization and return cost and variable values.

        Returns
        -------
        tuple
            (cost, variable_values_by_id) if optimal, (None, None) otherwise.
            Variable values are keyed by variable id for pickling compatibility.

        """
        self._prev_var_values = {}
        self._store_previous_values()
        self.iter.k = 0
        self.iter.cost = np.inf
        cost = self._solve()
        if self.prob_in.status == cp.OPTIMAL:
            var_values = {
                var.id: var.value.copy() if var.value is not None else None
                for var in self.prob_in.variables()
            }
            return cost, var_values
        return None, None

    def solve_multi_init(
        self,
        num_inits: int,
        *,
        parallel: bool = True,
        max_workers: int | None = None,
        mp_context: BaseContext | None = None,
    ) -> float:
        """Solve with multiple random initializations and return the best result.

        Parameters
        ----------
        num_inits : int
            Number of random initializations to try.
        parallel : bool, default=True
            If True, run initializations in parallel using multiprocessing.
        max_workers : int, optional
            Number of parallel worker processes. Only used if parallel=True.
            Defaults to the number of available CPUs.
        mp_context : BaseContext, optional
            Multiprocessing context for creating worker processes.
            Only used if parallel=True.

        Returns
        -------
        float
            The best objective value found across all initializations.

        """
        if num_inits <= 1:
            return self()

        best_cost = np.inf
        best_var_values: dict[int, Any] | None = None
        best_status = cp.INFEASIBLE

        orig_values = {
            var.id: var.value.copy() if var.value is not None else None
            for var in self.prob_in.variables()
        }

        if parallel:
            best_cost, best_var_values, best_status = self._solve_multi_parallel(
                num_inits, max_workers, mp_context
            )
        else:
            best_cost, best_var_values, best_status = self._solve_multi_sequential(
                num_inits
            )

        # Set the best solution
        _set_problem_status(self.prob_in, best_status)
        _set_problem_value(self.prob_in, best_cost)
        for var in self.prob_in.variables():
            if best_var_values and var.id in best_var_values:
                var.value = best_var_values[var.id]
            else:
                var.value = orig_values[var.id]

        return best_cost * (-1 if self.is_maximization else 1)

    def _solve_multi_sequential(
        self, num_inits: int
    ) -> tuple[float, dict[int, Any] | None, str]:
        """Run multiple initializations sequentially."""
        best_cost = np.inf
        best_var_values: dict[int, Any] | None = None
        best_status = cp.INFEASIBLE

        for _ in range(num_inits):
            initialize(self.prob_in, random=True)
            try:
                cost, var_values = self._solve_one_init()
                if cost is not None and cost < best_cost:
                    best_cost = cost
                    best_status = cp.OPTIMAL
                    best_var_values = var_values
            except (NonDCCPError, RuntimeError):
                continue

        return best_cost, best_var_values, best_status

    def _solve_multi_parallel(
        self,
        num_inits: int,
        max_workers: int | None,
        mp_context: BaseContext | None,
    ) -> tuple[float, dict[int, Any] | None, str]:
        """Run multiple initializations in parallel using multiprocessing."""
        best_cost = np.inf
        best_var_values: dict[int, Any] | None = None
        best_status = cp.INFEASIBLE

        with ProcessPoolExecutor(max_workers, mp_context) as executor:
            futures = []
            for _ in range(num_inits):
                initialize(self.prob_in, random=True)
                futures.append(executor.submit(self._solve_one_init))

            for future in as_completed(futures):
                try:
                    cost, var_values = future.result()
                    if cost is not None and cost < best_cost:
                        best_cost = cost
                        best_status = cp.OPTIMAL
                        best_var_values = var_values
                except (NonDCCPError, RuntimeError):
                    continue

        return best_cost, best_var_values, best_status


def dccp(  # noqa: PLR0913
    prob: cp.Problem,
    *,
    max_iter: int = 100,
    tau_ini: float = 0.005,
    mu: float = 1.2,
    tau_max: float = 1e8,
    k_ccp: int = 1,
    max_slack: float = 1e-3,
    ep: float = 1e-5,
    seed: int | None = None,
    verify_dccp: bool = True,
    parallel: bool = True,
    max_workers: int | None = None,
    mp_context: BaseContext | None = None,
    **kwargs: Any,
) -> float:
    """Run the DCCP algorithm on the given problem.

    This is the main entry point for solving DCCP problems. It creates a DCCP
    solver instance with the specified settings and solves the problem.

    Parameters
    ----------
    prob : cp.Problem
        A CVXPY Problem that satisfies DCCP rules.
    max_iter : int, default=100
        Maximum number of iterations in the CCP algorithm.
    tau_ini : float, default=0.005
        Initial value for tau parameter (trades off constraints vs objective).
    mu : float, default=1.2
        Rate at which tau increases during the algorithm.
    tau_max : float, default=1e8
        Upper bound for tau parameter.
    k_ccp : int, default=1
        Number of random restarts for the CCP algorithm.
    max_slack : float, default=1e-3
        Maximum slack variable value for convergence.
    ep : float, default=1e-5
        Convergence tolerance for objective value changes.
    seed : int, optional
        Random seed for reproducible results.
    verify_dccp : bool, default=True
        Whether to verify DCCP compliance before solving.
    parallel : bool, default=True
        If True and k_ccp > 1, run restarts in parallel using multiprocessing.
    max_workers : int, optional
        Number of parallel worker processes if k_ccp > 1 and parallel=True.
        Defaults to the number of available CPUs.
    mp_context : BaseContext, optional
        Multiprocessing context for creating worker processes.
        Only used if parallel=True.
    **kwargs
        Additional keyword arguments passed to the underlying solver.

    Returns
    -------
    float
        The optimal objective value (or best found value if not converged).

    Raises
    ------
    NonDCCPError
        If the problem doesn't satisfy DCCP rules or is already DCP compliant.

    See Also
    --------
    DCCP : The main DCCP solver class for more advanced usage.
    DCCPSettings : Configuration object for algorithm parameters.

    Examples
    --------
    >>> import cvxpy as cp
    >>> import dccp
    >>> x = cp.Variable(2)
    >>> problem = cp.Problem(cp.Maximize(cp.norm(x, 2)), [cp.norm(x, 1) <= 1])
    >>> result = dccp.dccp(problem, max_iter=50, k_ccp=3)

    """
    logger.debug("Running DCCP with solver=%s, kwargs=%s", kwargs.get("solver"), kwargs)
    dccp_solver = DCCP(
        prob,
        settings=DCCPSettings(
            max_iter=max_iter,
            tau_ini=tau_ini,
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
        return dccp_solver.solve_multi_init(
            k_ccp, parallel=parallel, max_workers=max_workers, mp_context=mp_context
        )
    return dccp_solver()
