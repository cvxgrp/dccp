__author__ = 'Xinyue'

import numpy as np
import cvxpy as cvx
import logging

from dccp.objective import convexify_obj
from dccp.objective import convexify_para_obj
from dccp.constraint import convexify_para_constr
from dccp.constraint import convexify_constr

logger = logging.getLogger('dccp')
logger.addHandler(logging.FileHandler(filename='dccp.log', mode='w'))
logger.setLevel(logging.INFO)
logger.propagate = False


def dccp(self, max_iter = 100, tau = 0.005, mu = 1.2, tau_max = 1e8,
         solver = None, ccp_times = 1, max_slack = 1e-3, ep = 1e-5, **kwargs):
    """
    main algorithm ccp
    :param max_iter: maximum number of iterations in ccp
    :param tau: initial weight on slack variables
    :param mu:  increment of weight on slack variables
    :param tau_max: maximum weight on slack variables
    :param solver: specify the solver for the transformed problem
    :param ccp_times: times of running ccp to solve a problem with random initial values on variables
    :return
        if the transformed problem is infeasible, return None;
    """
    if not is_dccp(self):
        raise Exception("Problem is not DCCP.")

    result = None
    if self.objective.NAME == 'minimize':
        cost_value = float("inf") # record on the best cost value
    else:
        cost_value = -float("inf")
    for t in range(ccp_times): # for each time of running ccp
        dccp_ini(self, random=(ccp_times>1), solver = solver, **kwargs) # initialization; random initial value is mandatory if ccp_times>1
        # iterations
        result_temp = iter_dccp(self, max_iter, tau, mu, tau_max, solver, ep, max_slack, **kwargs)
        if result_temp[0] is not None:
            if (self.objective.NAME == 'minimize' and result_temp[0]<cost_value) \
            or (self.objective.NAME == 'maximize' and result_temp[0]>cost_value): # find a better cost value
                # first ccp; no slack; slack small enough
                if t==0 or len(result_temp)<3 or result[1] < max_slack:
                    result = result_temp # update the result
                    cost_value = result_temp[0] # update the record on the best cost value
    return result

def dccp_ini(self, times = 1, random = 0, solver = None, **kwargs):
    """
    set initial values
    :param times: number of random projections for each variable
    :param random: mandatory random initial values
    """
    dom_constr = self.objective.args[0].domain # domain of the objective function
    for arg in self.constraints:
        for l in range(2):
            for dom in arg.args[l].domain:
                dom_constr.append(dom) # domain on each side of constraints
    var_store = [] # store initial values for each variable
    init_flag = [] # indicate if any variable is initialized by the user
    var_user_ini = []
    for var in self.variables():
        var_store.append(np.zeros(var.shape)) # to be averaged
        init_flag.append(var.value is None)
        if var.value is None:
            var_user_ini.append(np.zeros(var.shape))
        else:
            var_user_ini.append(var.value)
    # setup the problem
    ini_cost = 0
    var_ind = 0
    value_para = []
    for var in self.variables():
        if init_flag[var_ind] or random: # if the variable is not initialized by the user, or random initialization is mandatory
            value_para.append(cvx.Parameter(var.shape))
            ini_cost += cvx.pnorm(var-value_para[-1], 2)
        var_ind += 1
    ini_obj = cvx.Minimize(ini_cost)
    ini_prob = cvx.Problem(ini_obj, dom_constr)
    # solve it several times with random points
    for t in range(times): # for each time of random projection
        count_para = 0
        var_ind = 0
        for var in self.variables():
            # if the variable is not initialized by the user, or random
            # initialization is mandatory
            if init_flag[var_ind] or random:
                # set a random point
                if len(var.shape) > 1:
                    value_para[count_para].value = np.random.randn(var.shape[0], var.shape[1])*10
                else:
                    value_para[count_para].value = np.random.randn(var.size)*10
                count_para += 1
            var_ind += 1
        if solver is None:
            ini_prob.solve(**kwargs)
        else:
            ini_prob.solve(solver = solver, **kwargs)
        var_ind = 0
        for var in self.variables():
            var_store[var_ind] = var_store[var_ind] + var.value/float(times) # average
            var_ind += 1
    # set initial values
    var_ind = 0
    for var in self.variables():
        if init_flag[var_ind] or random:
            var.value = var_store[var_ind]
        else:
            var.value = var_user_ini[var_ind]
        var_ind += 1

def is_dccp(problem):
    """
    :param
        a problem
    :return
        a boolean indicating if the problem is dccp
    """
    if problem.objective.expr.curvature == 'UNKNOWN':
        return False
    for constr in problem.constraints:
        for arg in constr.args:
            if arg.curvature == 'UNKNOWN':
                return False
    return True


def iter_dccp(self, max_iter, tau, mu, tau_max, solver, ep, max_slack_tol, **kwargs):
    """
    ccp iterations
    :param max_iter: maximum number of iterations in ccp
    :param tau: initial weight on slack variables
    :param mu:  increment of weight on slack variables
    :param tau_max: maximum weight on slack variables
    :param solver: specify the solver for the transformed problem
    :return
        value of the objective function, maximum value of slack variables, value of variables
    """
    # split non-affine equality constraints
    constr = []
    for constraint in self.constraints:
        if str(type(constraint)) == "<class 'cvxpy.constraints.zero.Equality'>" and not constraint.is_dcp():
            constr.append(constraint.args[0] <= constraint.args[1])
            constr.append(constraint.args[0] >= constraint.args[1])
        else:
            constr.append(constraint)
    obj = self.objective
    self = cvx.Problem(obj, constr)
    it = 1
    converge = False
    # keep the values from the previous iteration or initialization
    previous_cost = float("inf")
    previous_org_cost = self.objective.value
    variable_pres_value = []
    for var in self.variables():
        variable_pres_value.append(var.value)
    # each non-dcp constraint needs a slack variable
    var_slack = []
    for constr in self.constraints:
        if not constr.is_dcp():
            var_slack.append(cvx.Variable(constr.shape))

    while it <= max_iter and all(var.value is not None for var in self.variables()):
        constr_new = []
        # objective
        convexified_obj = convexify_obj(self.objective)
        if not self.objective.is_dcp():
            # non-sub/super-diff
            while convexified_obj is None:
                # damping
                var_index = 0
                for var in self.variables():
                    #var_index = self.variables().index(var)
                    var.value = 0.8*var.value + 0.2* variable_pres_value[var_index]
                    var_index += 1
                convexified_obj = convexify_obj(self.objective)
            # domain constraints
            for dom in self.objective.expr.domain:
                constr_new.append(dom)
        # new cost function
        cost_new =  convexified_obj.expr

        # constraints
        count_slack = 0
        for arg in self.constraints:
            temp = convexify_constr(arg)
            if not arg.is_dcp():
                while temp is None:
                    # damping
                    for var in self.variables:
                        var_index = self.variables().index(var)
                        var.value = 0.8 * var.value + 0.2 * variable_pres_value[var_index]
                    temp = convexify_constr(arg)
                newcon = temp[0]  # new constraint without slack variable
                for dom in temp[1]:# domain
                    constr_new.append(dom)
                constr_new.append(newcon.expr <= var_slack[count_slack])
                constr_new.append(var_slack[count_slack] >= 0)
                count_slack = count_slack + 1
            else:
                constr_new.append(arg)

        # objective
        if self.objective.NAME == 'minimize':
            for var in var_slack:
                cost_new += tau*cvx.sum(var)
            obj_new = cvx.Minimize(cost_new)
        else:
            for var in var_slack:
                cost_new -= tau*cvx.sum(var)
            obj_new = cvx.Maximize(cost_new)

        # new problem
        prob_new = cvx.Problem(obj_new, constr_new)
        # keep previous value of variables
        variable_pres_value = []
        for var in self.variables():
            variable_pres_value.append(var.value)
        # solve
        if solver is None:
            logger.info("iteration=%d, cost value=%.5f, tau=%.5f", it, prob_new.solve(**kwargs), tau)
        else:
            logger.info("iteration=%d, cost value=%.5f, tau=%.5f", it, prob_new.solve(solver=solver, **kwargs), tau)
        max_slack = None
        # print slack
        if (prob_new._status == "optimal" or prob_new._status == "optimal_inaccurate") and not var_slack == []:
            slack_values = [v.value for v in var_slack if v.value is not None]
            max_slack = max([np.max(v) for v in slack_values] + [-np.inf])
            logger.info("max slack = %.5f", max_slack)
        #terminate
        if prob_new.value is not None and np.abs(previous_cost - prob_new.value) <= ep and np.abs(self.objective.value - previous_org_cost) <= ep \
                and (max_slack is None or max_slack <= max_slack_tol ):
            it = max_iter+1
            converge = True
        else:
            previous_cost = prob_new.value
            previous_org_cost = self.objective.value
            tau = min([tau*mu,tau_max])
            it += 1
    # return
    if converge:
        self._status = "Converged"
    else:
        self._status = "Not_converged"
    var_value = []
    for var in self.variables():
        var_value.append(var.value)
    if not var_slack == []:
        return(self.objective.value, max_slack, var_value)
    else:
        return(self.objective.value, var_value)

cvx.Problem.register_solve("dccp", dccp)
