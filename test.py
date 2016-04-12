__author__ = 'Xinyue'
from cvxpy import *
from linearize import linearize_para
from convexify_constr import convexify_para_constr
from dccp_problem import dccp_transform
from  dccp_problem import dccp_ini
import numpy as np
import matplotlib.pyplot as plt

def iter_dccp(self, max_iter=100, tau=0.005, mu=1.2 ,tau_max=1e8, solver = None):
    it = 1
    # keep the values from the previous iteration or initialization
    previous_cost = float("inf")
    variable_pres_value = []
    for var in self.variables():
        variable_pres_value.append(var.value)
    convex_prob = dccp_transform(self)
    dccp_ini(self)
    while it<=max_iter and all(var.value is not None for var in self.variables()):
        #cost functions
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
                                key.value = 0.8*key.value + 0.2* variable_pres_value[var_index]
                                G = arg.args[l].gradient
                                flag_G = np.any(np.isnan(G[key])) or np.any(np.isinf(G[key]))
                            # gradient parameter
                            for d in range(key.size[1]):
                                convex_prob[1][count_con_constr][l][1][key][1][d].value = G[key][:,d,:,0]
                            # var value parameter
                            convex_prob[1][count_con_constr][l][1][key][0].value = key.value
                count_con_constr += 1
            count_constr += 1
        variable_pres_value = []
        for var in self.variables():
            variable_pres_value.append(var.value)
        convex_prob[1][-1].value = tau
        if solver==None:
            print "iteration=",it, "cost value = ", convex_prob[0].solve(), "tau = ", tau
        else:
            print "iteration=",it, "cost value = ", convex_prob[0].solve(solver = solver), "tau = ", tau
        if not len(convex_prob[5])==0:
            max_slack = []
            for i in range(len(convex_prob[5])):
                max_slack.append(np.max(convex_prob[5][i].value))
            max_slack = np.max(max_slack)
            print "max slack = ", max_slack
        if np.abs(previous_cost - convex_prob[0].value) <= 1e-4: #terminate
            it = max_iter+1
        else:
            previous_cost = convex_prob[0].value
            tau = min([tau*mu,tau_max])
            it += 1


x = Variable(1)
x.value = 1
y = Variable(1)
myprob = Problem(Minimize(sqrt(x)))
iter_dccp(myprob, solver = 'MOSEK')
print "x = ", x.value

