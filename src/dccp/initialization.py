"""Initialization module for DCCP problems."""

from typing import Any

import cvxpy as cp
import numpy as np


def initialize(  # noqa: PLR0913
    prob: cp.Problem,
    k_ini: int = 1,
    seed: int | None = None,
    solver: str | None = None,
    std: float = 10.0,
    mean: float = 0.0,
    *,
    random: bool = False,
    **kwargs: Any,
) -> None:
    """Set initial values for DCCP problem variables.

    This function initializes variables in a DCCP problem by solving auxiliary
    optimization problems. It can perform multiple random projections and
    average the results to obtain better initial values.

    Parameters
    ----------
    prob : Problem
        The CVXPY Problem instance to initialize.
    k_ini : int, default=1
        Number of random projections for each variable. Higher values may
        lead to better initialization but increase computation time.
    random : int, default=0
        If non-zero, forces random initial values for all variables,
        overriding any user-provided initial values.
    seed : int or None, default=None
        Random seed for reproducible initialization. If None, uses
        system entropy for random number generation.
    solver : str or None, default=None
        Solver to use for the initialization subproblems. If None,
        uses CVXPY's default solver selection.
    std : float, default=10.0
        Standard deviation for the random initialization. This scales the
        random values generated for the variables.
    mean : float, default=0.0
        Mean for the random initialization. This shifts the random values
        generated for the variables.
    **kwargs
        Additional keyword arguments passed to the solver.

    Returns
    -------
    None
        This function modifies the problem variables in-place by setting
        their `.value` attributes.

    Notes
    -----
    The initialization process works by:
    1. Collecting domain constraints from the objective and constraints
    2. For each variable without user-provided values (or if random=True),
       creating a least-squares problem to find values close to random points
    3. Solving multiple initialization subproblems and averaging the results
    4. Setting the averaged values as initial points for the variables

    The random initialization uses standard normal distribution scaled by 10.
    Variables with user-provided initial values are preserved unless random=True.

    """
    rng = np.random.default_rng(seed)
    dom_constr = prob.objective.args[0].domain  # domain of the objective function

    # add domain constraints from the problem
    for c in prob.constraints:
        for arg in c.args:
            dom_constr.extend(arg.domain)

    # placeholder for variables that still need a value
    z_j: dict[cp.Variable, cp.Parameter] = {}

    # if random initialization is mandatory, set all variables to zero
    ini_cost = cp.sum(0)
    for x in prob.variables():
        if random or x.value is None:
            shape = x.shape if len(x.shape) > 1 else x.size
            z_j[x] = cp.Parameter(shape)
            ini_cost += cp.norm(x - z_j[x] * std, "fro")

    # no variables to initialize
    if len(z_j) == 0:
        return

    # store results for each initialization k in k_ini
    result_record: list[dict[cp.Variable, Any]] = []
    ini_prob = cp.Problem(cp.Minimize(ini_cost), dom_constr)

    # find a point x which minimizes ||x - x_k||_2 for each random projection x_k
    for _ in range(k_ini):
        for z in z_j.values():
            z.value = rng.standard_normal(z.shape) * std + mean

        # solve the initialization problem
        ini_prob.solve(solver=solver, **kwargs)
        result_record.append({var: var.value for var in prob.variables()})

    # set the variables' values to the average of the results
    for z in z_j:
        z.value = np.mean([res[z] for res in result_record], axis=0)
