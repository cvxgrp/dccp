__author__ = 'Xinyue'

from cvxpy import *
import numpy as np
import cvxpy as cvx
from objective import convexify_obj
from objective import convexify_para_obj
from constraint import convexify_para_constr
from constraint import convexify_constr

def dccp(self, max_iter = 100, tau = 0.005, mu = 1.2, tau_max = 1e8, solver = None, ccp_times = 1):
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
    if is_dccp(self)==True:
        #convex_prob = dccp_transform(self) # convexify problem
        result = None
        if self.objective.NAME == 'minimize':
            cost_value = float("inf") # record on the best cost value
        else:
            cost_value = -float("inf")
        for t in range(ccp_times): # for each time of running ccp
            dccp_ini(self, random=(ccp_times>1)) # initialization; random initial value is mandatory if ccp_times>1
            #result_temp = iter_dccp_para(self, convex_prob, max_iter, tau, mu ,tau_max, solver)
            # iterations
            result_temp = iter_dccp(self, max_iter, tau, mu, tau_max, solver)
            if (self.objective.NAME == 'minimize' and result_temp[0]<cost_value) \
            or (self.objective.NAME == 'maximize' and result_temp[0]>cost_value): # find a better cost value
                if t==0 or len(result_temp)<3 or result[1]<1e-4: # first ccp; no slack; slack small enough
                    result = result_temp # update the result
                    cost_value = result_temp[0] # update the record on the best cost value
        return result
    else:
        print "not a dccp problem"

def dccp_ini(self, times = 3, random = 0):
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
    for var in self.variables():
        var_store.append(np.zeros((var._rows,var._cols))) # to be averaged
        init_flag.append(var.value is None)
    # setup the problem
    ini_cost = 0
    var_ind = 0
    value_para = []
    for var in self.variables():
        if init_flag[var_ind] or random: # if the variable is not initialized by the user, or random initialization is mandatory
            value_para.append(Parameter(var._rows,var._cols))
            ini_cost += pnorm(var-value_para[-1],2)
        var_ind += 1
    ini_obj = Minimize(ini_cost)
    ini_prob = Problem(ini_obj,dom_constr)
    # solve it several times with random points
    for t in range(times): # for each time of random projection
        count_para = 0
        var_ind = 0
        for var in self.variables():
            if init_flag[var_ind] or random: # if the variable is not initialized by the user, or random initialization is mandatory
                value_para[count_para].value = np.random.randn(var._rows,var._cols)*10 # set a random point
                count_para += 1
            var_ind += 1
        ini_prob.solve()
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
    for constr in problem.constraints + [problem.objective]:
        for arg in constr.args:
            if arg.curvature == 'UNKNOWN':
                flag = False
                return flag
    return flag

def dccp_transform(self):
    """
    problem transformation
    return:
        prob_new: a new dcp problem
        parameters: parameters in the constraints
        flag: indicate if each constraint is transformed
        parameters_cost: parameters in the cost function
        flag_cost: indicate if the cost function is transformed
        var_slack: a list of slack variables
    """
    # split non-affine equality constraints
    constr = []
    for arg in self.constraints:
        if arg.OP_NAME == "==" and not arg.is_dcp():
            constr.append(arg[0]<=arg[1])
            constr.append(arg[1]<=arg[0])
        else:
            constr.append(arg)
    self.constraints = constr

    constr_new = [] # new constraints
    parameters = []
    flag = []
    parameters_cost = []
    flag_cost = []
    # constraints
    var_slack = [] # slack
    for constr in self.constraints:
        if not constr.is_dcp():
            flag.append(1)
            var_slack.append(Variable(constr.size[0], constr.size[1]))
            temp = convexify_para_constr(constr)
            newcon = temp[0]   # new constraint without slack variable
            right = newcon.args[1] + var_slack[-1] # add slack variable on the right side
            constr_new.append(newcon.args[0]<=right) # new constraint with slack variable
            constr_new.append(var_slack[-1]>=0) # add constraint on the slack variable
            parameters.append(temp[1])
            for dom in temp[2]: # domain
                constr_new.append(dom)
        else:
            flag.append(0)
            constr_new.append(constr)
    # cost functions
    if not self.objective.is_dcp():
        flag_cost.append(1)
        temp = convexify_para_obj(self.objective)
        cost_new =  temp[0] # new cost function
        parameters_cost.append(temp[1])
        parameters_cost.append(temp[2])
        for dom in temp[3]: # domain constraints
            constr_new.append(dom)
    else:
        flag_cost.append(0)
        cost_new = self.objective.args[0]
    # objective
    tau = Parameter()
    parameters.append(tau)
    if self.objective.NAME == 'minimize':
        for var in var_slack:
            cost_new += np.ones((var._cols,var._rows))*var*tau
        obj_new = Minimize(cost_new)
    else:
        for var in var_slack:
            cost_new -= np.ones((var._cols,var._rows))*var*tau
        obj_new = Maximize(cost_new)
    # new problem
    prob_new = Problem(obj_new, constr_new)
    return prob_new, parameters, flag, parameters_cost, flag_cost, var_slack

def iter_dccp_para(self, convex_prob, max_iter, tau, mu, tau_max, solver):
    """
    ccp iterations
    :param convex_prob: result from dccp_transform
    :param max_iter: maximum number of iterations in ccp
    :param tau: initial weight on slack variables
    :param mu:  increment of weight on slack variables
    :param tau_max: maximum weight on slack variables
    :param solver: specify the solver for the transformed problem
    :return
        value of the objective function, maximum value of slack variables, value of variables
    """
    # keep the values from the initialization
    # split non-affine equality constraints
    constr = []
    for arg in self.constraints:
        if arg.OP_NAME == "==" and not arg.is_dcp():
            constr.append(arg[0]<=arg[1])
            constr.append(arg[0]>=arg[1])
        else:
            constr.append(arg)
    self.constraints = constr

    previous_cost = float("inf")
    variable_pres_value = []
    for var in self.variables():
        variable_pres_value.append(var.value)
    it = 1
    while it<=max_iter and all(var.value is not None for var in self.variables()):
        # cost functions
        if convex_prob[4][0] == 1:
            convex_prob[3][0].value = self.objective.args[0].value
            G = self.objective.args[0].gradient
            for key in G:
                # damping
                flag_G = np.any(np.isnan(G[key])) or np.any(np.isinf(G[key]))
                while flag_G:
                    var_index = self.variables().index(key)
                    key.value = 0*key.value + 1* variable_pres_value[var_index]
                    G = self.objective.args[0].gradient
                    flag_G = np.any(np.isnan(G[key])) or np.any(np.isinf(G[key]))
                # gradient parameter
                for d in range(key.size[1]):
                    convex_prob[3][1][key][1][d].value = G[key][:,d,:,0]
                # var value parameter
                convex_prob[3][1][key][0].value = key.value
        #constraints
        count_constr = 0
        count_con_constr = 0
        for arg in self.constraints:
            if convex_prob[2][count_constr] == 1:
                for l in range(2):
                    if not len(convex_prob[1][count_con_constr][l]) == 0:
                        convex_prob[1][count_con_constr][l][0].value = arg.args[l].value
                        G = arg.args[l].gradient
                        for key in G:
                            # damping
                            flag_G = np.any(np.isnan(G[key])) or np.any(np.isinf(G[key]))
                            while flag_G:
                                var_index = self.variables().index(key)
                                key.value = 0.8*key.value + 0.2*variable_pres_value[var_index]
                                G = arg.args[l].gradient
                                flag_G = np.any(np.isnan(G[key])) or np.any(np.isinf(G[key]))
                            # gradient parameter
                            for d in range(key.size[1]):
                                convex_prob[1][count_con_constr][l][1][key][1][d].value = G[key][:,d,:,0]
                            # var value parameter
                            convex_prob[1][count_con_constr][l][1][key][0].value = key.value
                count_con_constr += 1
            count_constr += 1
        # keep the values from the previous iteration
        variable_pres_value = []
        for var in self.variables():
            variable_pres_value.append(var.value)
        # parameter tau
        convex_prob[1][-1].value = tau
        # solve the transformed problem
        if solver==None:
            print "iteration=",it, "cost value = ", convex_prob[0].solve(), "tau = ", tau
        else:
            print "iteration=",it, "cost value = ", convex_prob[0].solve(solver = solver), "tau = ", tau
        # print slack variables
        if not len(convex_prob[5])==0:
            max_slack = []
            for i in range(len(convex_prob[5])):
                max_slack.append(np.max(convex_prob[5][i].value))
            max_slack = np.max(max_slack)
            print "max slack = ", max_slack
        if np.abs(previous_cost - convex_prob[0].value) <= 1e-4: # terminate
            it = max_iter+1
        else:
            previous_cost = convex_prob[0].value
            tau = min([tau*mu,tau_max])
            it += 1
    var_value = []
    for var in self.variables():
        var_value.append(var.value)
    if not len(convex_prob[5])==0:
        return(self.objective.value, max_slack, var_value)
    else:
        return(self.objective.value, var_value)


def iter_dccp(self, max_iter, tau, mu, tau_max, solver):
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
        if arg.OP_NAME == "==" and not arg.is_dcp():
            constr.append(arg.args[0]<=arg.args[1])
            constr.append(arg.args[1]<=arg.args[0])
        else:
            constr.append(arg)
    self.constraints = constr

    it = 1
    # keep the values from the previous iteration or initialization
    previous_cost = float("inf")
    variable_pres_value = []
    for var in self.variables():
        variable_pres_value.append(var.value)
    # each non-dcp constraint needs a slack variable
    var_slack = []
    for constr in self.constraints:
        if not constr.is_dcp():
            rows, cols = constr.size
            var_slack.append(Variable(rows, cols))

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
                right = newcon.args[1] + var_slack[count_slack]
                constr_new.append(newcon.args[0]<=right)
                constr_new.append(var_slack[count_slack]>=0)
                count_slack = count_slack+1
            else:
                constr_new.append(temp)

        # objective
        if self.objective.NAME == 'minimize':
            for var in var_slack:
                cost_new += np.ones((var._cols,var._rows))*var*tau
            obj_new = Minimize(cost_new)
        else:
            for var in var_slack:
                cost_new -= var*tau
            obj_new = Maximize(cost_new)

        # new problem
        prob_new = Problem(obj_new, constr_new)
        # keep previous value of variables
        variable_pres_value = []
        for var in self.variables():
            variable_pres_value.append(var.value)
        # solve
        if solver is None:
            print "iteration=",it, "cost value = ", prob_new.solve(), "tau = ", tau
        else:
            print "iteration=",it, "cost value = ", prob_new.solve(solver = solver), "tau = ", tau
        # print slack
        if not var_slack == []:
            max_slack = []
            for i in range(len(var_slack)):
                max_slack.append(np.max(var_slack[i].value))
            max_slack = np.max(max_slack)
            print "max slack = ", max_slack
        #terminate
        if np.abs(previous_cost - prob_new.value) <= 1e-4:
            it_real = it
            it = max_iter+1
        else:
            previous_cost = prob_new.value
            it_real = it
            tau = min([tau*mu,tau_max])
            it += 1
    # return
    var_value = []
    for var in self.variables():
        var_value.append(var.value)
    if not var_slack == []:
        return(self.objective.value, max_slack, var_value)
    else:
        return(self.objective.value, var_value)

cvx.Problem.register_solve("dccp", dccp)
