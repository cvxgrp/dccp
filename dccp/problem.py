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
         solver = None, ccp_times = 1, max_slack = 1e-4, **kwargs):
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

    #convex_prob = dccp_transform(self) # convexify problem
    result = None
    if self.objective.NAME == 'minimize':
        cost_value = float("inf") # record on the best cost value
    else:
        cost_value = -float("inf")
    for t in range(ccp_times): # for each time of running ccp
        dccp_ini(self, random=(ccp_times>1), solver = solver, **kwargs) # initialization; random initial value is mandatory if ccp_times>1
        # iterations
        result_temp = iter_dccp(self, max_iter, tau, mu, tau_max, solver, **kwargs)
        if result_temp[0] is not None:
            if (self.objective.NAME == 'minimize' and result_temp[0]<cost_value) \
            or (self.objective.NAME == 'maximize' and result_temp[0]>cost_value): # find a better cost value
                # first ccp; no slack; slack small enough
                if t==0 or len(result_temp)<3 or result[1] < max_slack:
                    result = result_temp # update the result
                    cost_value = result_temp[0] # update the record on the best cost value
    return result

def dccp_ini(self, times = 3, random = 0, solver = None, **kwargs):
    """
    set initial values
    :param times: number of random projections for each variable
    :param random: mandatory random initial values
    """
    dom_constr = self.objective.args[0].domain # domain of the objective function
    for arg in self.constraints:
        for l in range(2):
            for dom in arg.expr.args[l].domain:
                dom_constr.append(dom) # domain on each side of constraints
    var_store = [] # store initial values for each variable
    init_flag = [] # indicate if any variable is initialized by the user
    for var in self.variables():
        var_store.append(np.zeros(var.size)) # to be averaged
        init_flag.append(var.value is None)
    # setup the problem
    ini_cost = 0
    var_ind = 0
    value_para = []
    for var in self.variables():
        if init_flag[var_ind] or random: # if the variable is not initialized by the user, or random initialization is mandatory
            value_para.append(cvx.Parameter(var.size))
            ini_cost += cvx.pnorm(var-value_para[-1], 2)
        var_ind += 1
    ini_obj = cvx.Minimize(ini_cost)
    ini_prob = cvx.Problem(ini_obj,dom_constr)
    # solve it several times with random points
    for t in range(times): # for each time of random projection
        count_para = 0
        var_ind = 0
        for var in self.variables():
            # if the variable is not initialized by the user, or random
            # initialization is mandatory
            if init_flag[var_ind] or random:
                # set a random point
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
            var_ind += 1

def is_dccp(problem):
    """
    :param
        a problem
    :return
        a boolean indicating if the problem is dccp
    """
    flag = True
    for constr in problem.constraints + problem.objective.args:
        for arg in constr.expr.args:
            if arg.curvature == 'UNKNOWN':
                flag = False
                return flag
    return flag

def iter_dccp(self, max_iter, tau, mu, tau_max, solver, **kwargs):
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
    for arg in self.constraints:
        if str(type(arg)) == "<class 'cvxpy.constraints.zero.Zero'>" and not arg.is_dcp():
            constr.append(arg.expr.args[0] + arg.expr.args[1] <= 0)
            constr.append(-arg.expr.args[0] - arg.expr.args[1] <= 0)
        else:
            constr.append(arg)
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
            var_slack.append(cvx.Variable(constr.size))

    while it<=max_iter and all(var.value is not None for var in self.variables()):
        constr_new = []
        # objective
        temp = convexify_obj(self.objective)
        if not self.objective.is_dcp():
            # non-sub/super-diff
            while temp is None:
                # damping
                var_index = 0
                for var in self.variables():
                    #var_index = self.variables().index(var)
                    var.value = 0.8*var.value + 0.2* variable_pres_value[var_index]
                    var_index += 1
                temp = convexify_obj(self.objective)
            # domain constraints
            for dom in self.objective.args[0].domain:
                constr_new.append(dom)
        # new cost function
        cost_new =  temp.args[0]

        # constraints
        count_slack = 0
        for arg in self.constraints:
            temp = convexify_constr(arg)
            if not arg.is_dcp():
                while temp is None:
                    # damping
                    for var in self.variables:
                        var_index = self.variables().index(var)
                        var.value = 0.8*var.value + 0.2* variable_pres_value[var_index]
                    temp = convexify_constr(arg)
                newcon = temp[0]  # new constraint without slack variable
                for dom in temp[1]:# domain
                    constr_new.append(dom)
                constr_new.append(newcon.expr <= var_slack[count_slack])
                constr_new.append(var_slack[count_slack]>=0)
                count_slack = count_slack+1
            else:
                constr_new.append(temp)

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
        if np.abs(previous_cost - prob_new.value) <= 1e-5 and np.abs(self.objective.value - previous_org_cost) <= 1e-5:
            it_real = it
            it = max_iter+1
            converge = True
        else:
            previous_cost = prob_new.value
            previous_org_cost = self.objective.value
            it_real = it
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
