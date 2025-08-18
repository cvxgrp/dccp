"""Initialization module for DCCP problems."""

import cvxpy as cp
import numpy as np


def initialize(self, times=1, random=0, seed=None, solver=None, **kwargs) -> None:
    """Set initial values.

    :param times: number of random projections for each variable
    :param random: mandatory random initial values
    """
    rng = np.random.default_rng(seed)
    dom_constr = self.objective.args[0].domain  # domain of the objective function
    for arg in self.constraints:
        for l in range(len(arg.args)):
            for dom in arg.args[l].domain:
                dom_constr.append(dom)  # domain on each side of constraints
    var_store = {}  # store initial values for each variable
    init_flag = {}  # indicate if any variable is initialized by the user
    var_user_ini = {}
    for var in self.variables():
        var_store[var] = np.zeros(var.shape)  # to be averaged
        init_flag[var] = var.value is None
        if var.value is None:
            var_user_ini[var] = np.zeros(var.shape)
        else:
            var_user_ini[var] = var.value
    for t in range(times):  # for each time of random projection
        # setup the problem
        ini_cost = 0
        for var in self.variables():
            if (
                init_flag[var] or random
            ):  # if the variable is not initialized by the user, or random initialization is mandatory
                if len(var.shape) > 1:
                    ini_cost += cp.norm(
                        var - rng.standard_normal((var.shape[0], var.shape[1])) * 10,
                        "fro",
                    )
                else:
                    ini_cost += cp.norm(var - rng.standard_normal(var.size) * 10)
        ini_obj = cp.Minimize(ini_cost)
        ini_prob = cp.Problem(ini_obj, dom_constr)
        # print("ini problem", ini_prob, "ini obj", ini_obj, "dom constr", dom_constr)
        if solver is None:
            ini_prob.solve(**kwargs)
        else:
            ini_prob.solve(solver=solver, **kwargs)
        # print("end solving ini problem")
        for var in self.variables():
            var_store[var] = var_store[var] + var.value / float(times)  # average
    # set initial values
    for var in self.variables():
        if init_flag[var] or random:
            var.value = var_store[var]
        else:
            var.value = var_user_ini[var]
