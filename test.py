__author__ = 'Xinyue'
from cvxpy import *
import examples.extensions.dccp
from examples.extensions.dccp.linearize import linearize_para
from examples.extensions.dccp.convexify_constr import convexify_para_constr
from examples.extensions.dccp.dccp_problem import dccp_transform
from  examples.extensions.dccp.dccp_problem import dccp_ini
import numpy as np

x = Variable(1)
y = Variable(1)
x.value = 2
y.value = 3
expr = sqrt(x)
constr = [square(x)>=x]
prob = Problem(Minimize(expr), constr)

def iter_dccp(self, max_iter=1, tau=0.001, mu=1.2 ,tau_max=1e8):
    it = 1
    # keep the values from the previous iteration or initialization
    previous_cost = float("inf")
    variable_pres_value = []
    for var in self.variables():
        variable_pres_value.append(var.value)
    convex_prob = dccp_transform(self)
    ### print out
    print convex_prob[0]
    print convex_prob[1][0]
    print convex_prob[2]
    print convex_prob[3]
    print convex_prob[4]
    ###
    dccp_ini(prob)
    while it<=max_iter and all(var.value is not None for var in self.variables()):
        #cost functions
        if convex_prob[4][0] == 1:
            convex_prob[3][0].value = self.objective.args[0].value
            G = self.objective.args[0].gradient
            for key in G:
                convex_prob[3][1][key][0].value = key.value
                convex_prob[3][1][key][1][0].value = G[key]
        #constraints
        count_constr = 0
        for arg in self.constraints:
            if convex_prob[2][count_constr] == 1:
                for l in range(2):
                    if not len(convex_prob[1][count_constr][l]) == 0:
                        convex_prob[1][count_constr][l][0].value = arg.args[l].value
                        G = arg.args[l].gradient
                        for key in G:
                            convex_prob[1][count_constr][l][1][key][0].value = key.value
                            convex_prob[1][count_constr][l][1][key][1][0].value = G[key]
            count_constr += 1
        variable_pres_value = []
        for var in self.variables():
            variable_pres_value.append(var.value)
        convex_prob[1][-1] = tau
        print "iteration=",it, "cost value = ", convex_prob[0].solve()
        '''
        if not var_slack == []:
            if solver is None:
                print "iteration=",it, "cost value = ", prob_new.solve(), "tau = ", tau
            else:
                print "iteration=",it, "cost value = ", prob_new.solve(solver = solver), "tau = ", tau
            max_slack = []
            for i in range(len(var_slack)):
                max_slack.append(np.max(var_slack[i].value))
            max_slack = np.max(max_slack)
            print "max slack = ", max_slack
        else:
            if solver is None:
                co = prob_new.solve()
            else:
                co = prob_new.solve(solver = solver)
            print "iteration=",it, "cost value = ", co , "tau = ", tau
        '''
        if np.abs(previous_cost - convex_prob[0].value) <= 1e-4: #terminate
            it_real = it
            it = max_iter+1
        else:
            previous_cost = convex_prob[0].value
            it_real = it
            tau = min([tau*mu,tau_max])
            it += 1

iter_dccp(prob)
